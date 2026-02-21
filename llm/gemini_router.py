"""
Gemini LLM middleware for trade-signal adjudication.

Supports:
- .env loading (without external dotenv dependency)
- API key rotation across multiple free-tier keys
- model failover (e.g., gemini-3-flash-preview -> gemini-3.1-pro)
- strict JSON output parsing with fail-safe fallback
- simulated mode for backtests/offline validation
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


POWER_SYSTEM_PROMPT = """You are the LLM risk adjudicator for a production quant strategy.

Mission:
- Convert raw model/context features into a conservative trade adjudication.
- Detect narrative traps, contradictory signals, overfit behavior, and noisy distractions.
- Approve only high-quality actions; otherwise force HOLD.

Decision standard:
- BUY only when upward evidence is strong, coherent, and robust to noise.
- SELL only when downward evidence is strong, coherent, and robust to noise.
- HOLD whenever confidence is insufficient OR features conflict OR noise/distraction risk is elevated.

Rules:
- You must reason ONLY from provided inputs.
- Never invent missing market/news facts.
- Penalize actions when volatility/noise/distraction scores are high.
- Return compact, parseable JSON only.

Output JSON schema (exact keys):
{
  "decision": "BUY|SELL|HOLD",
  "buy_score": 0.0,
  "sell_score": 0.0,
  "hold_score": 0.0,
  "alignment_score": 0.0,
  "distraction_risk": 0.0,
  "rationale": "<=160 chars"
}

Scoring notes:
- All numeric scores must be in [0, 1].
- alignment_score means confidence that your decision aligns with robust signal quality.
- If uncertain, choose HOLD and raise hold_score/distraction_risk.
"""


@dataclass
class GeminiLLMConfig:
    enabled: bool = False
    mode: str = "off"  # off|simulated|api
    api_keys: list[str] = field(default_factory=list)
    models: list[str] = field(
        default_factory=lambda: ["gemini-3-flash-preview", "gemini-3.1-pro"]
    )
    min_alignment_score: float = 0.80
    timeout_seconds: float = 20.0
    max_attempts: int = 8
    temperature: float = 0.05
    max_output_tokens: int = 350
    fail_mode: str = "hold"  # hold|pass
    env_path: Optional[str] = None


@dataclass
class LLMDecision:
    decision: str = "HOLD"
    buy_score: float = 0.0
    sell_score: float = 0.0
    hold_score: float = 1.0
    alignment_score: float = 0.0
    distraction_risk: float = 1.0
    rationale: str = "fallback"
    mode: str = "off"
    model: str = ""
    key_slot: int = -1
    raw_text: str = ""
    error: str = ""

    def score_for_action(self, action: int) -> float:
        if action > 0:
            return float(self.buy_score)
        if action < 0:
            return float(self.sell_score)
        return float(self.hold_score)

    def aligns_with_action(self, action: int) -> bool:
        d = self.decision.upper().strip()
        if action > 0:
            return d == "BUY"
        if action < 0:
            return d == "SELL"
        return d == "HOLD"


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(np.clip(v, 0.0, 1.0))


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _load_env_file(path: Path) -> None:
    """Tiny .env loader (does not overwrite already-set variables)."""
    if not path.exists() or not path.is_file():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _collect_api_keys(explicit_keys: Optional[Iterable[str]] = None) -> list[str]:
    keys: list[str] = []

    if explicit_keys:
        keys.extend([k.strip() for k in explicit_keys if str(k).strip()])

    keys.extend(_split_csv(os.getenv("GEMINI_API_KEYS")))

    single = os.getenv("GEMINI_API_KEY")
    if single:
        keys.append(single.strip())

    for i in range(1, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def _default_user_prompt(context: Dict[str, Any]) -> str:
    return (
        "Evaluate this trade context and return only JSON with the exact schema.\n\n"
        "trade_context_json:\n"
        f"{json.dumps(context, ensure_ascii=True, sort_keys=True)}"
    )


class GeminiRouter:
    """Rotating Gemini client with strict fail-safe behavior."""

    def __init__(self, config: GeminiLLMConfig):
        self.config = config
        self._session = requests.Session()
        self._key_cursor = 0
        self._model_cursor = 0

    @classmethod
    def from_env(
        cls,
        enabled: bool,
        mode: str | None = None,
        models: Optional[Iterable[str]] = None,
        min_alignment_score: float = 0.80,
        fail_mode: str = "hold",
        timeout_seconds: float = 20.0,
        max_attempts: int = 8,
        env_path: str | None = None,
    ) -> "GeminiRouter":
        env_file = Path(env_path or ".env")
        _load_env_file(env_file)

        env_mode = os.getenv("GEMINI_MODE", "").strip().lower()
        if not mode:
            mode = env_mode or ("simulated" if enabled else "off")
        mode = mode.strip().lower()
        if mode not in {"off", "simulated", "api"}:
            mode = "off"

        env_models = _split_csv(os.getenv("GEMINI_MODELS"))
        model_list = list(models or env_models or ["gemini-3-flash-preview", "gemini-3.1-pro"])

        keys = _collect_api_keys()
        if mode == "api" and not keys:
            logger.warning("Gemini mode=api but no keys found; switching to off")
            mode = "off"

        cfg = GeminiLLMConfig(
            enabled=bool(enabled),
            mode=mode,
            api_keys=keys,
            models=model_list,
            min_alignment_score=_clamp01(min_alignment_score, default=0.80),
            timeout_seconds=float(max(1.0, timeout_seconds)),
            max_attempts=max(1, int(max_attempts)),
            fail_mode="pass" if str(fail_mode).lower() == "pass" else "hold",
            env_path=str(env_file),
        )
        return cls(cfg)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.mode != "off")

    @property
    def min_alignment_score(self) -> float:
        return float(self.config.min_alignment_score)

    def evaluate(self, context: Dict[str, Any], proposed_action: int) -> LLMDecision:
        """
        Evaluate trade context.

        Returns a safe decision object in all failure modes.
        """
        if not self.enabled:
            return self._pass_decision(proposed_action, mode="off")

        if self.config.mode == "simulated":
            return self._simulate(context, proposed_action)

        if self.config.mode != "api":
            return self._fallback(proposed_action, error="unsupported_mode")

        attempts = max(1, min(self.config.max_attempts, len(self.config.api_keys) * max(1, len(self.config.models))))
        last_error = ""

        for _ in range(attempts):
            key_slot = self._key_cursor % len(self.config.api_keys)
            model_slot = self._model_cursor % len(self.config.models)
            api_key = self.config.api_keys[key_slot]
            model_name = self.config.models[model_slot]

            self._key_cursor += 1
            self._model_cursor += 1

            try:
                raw = self._call_gemini(model_name=model_name, api_key=api_key, context=context)
                decision = self._parse_decision(raw)
                decision.mode = "api"
                decision.model = model_name
                decision.key_slot = key_slot + 1
                decision.raw_text = raw[:1000]
                return decision
            except Exception as exc:
                last_error = str(exc)
                continue

        return self._fallback(proposed_action, error=last_error or "all_attempts_failed")

    def _call_gemini(self, model_name: str, api_key: str, context: Dict[str, Any]) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            f"?key={api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": POWER_SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": _default_user_prompt(context)}],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }

        response = self._session.post(url, json=payload, timeout=self.config.timeout_seconds)
        if response.status_code != 200:
            raise RuntimeError(f"gemini_http_{response.status_code}")

        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError("gemini_no_candidates")

        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        if not parts:
            raise RuntimeError("gemini_no_parts")

        text = str((parts[0] or {}).get("text", "")).strip()
        if not text:
            raise RuntimeError("gemini_empty_text")
        return text

    def _parse_decision(self, raw_text: str) -> LLMDecision:
        data = self._extract_json(raw_text)

        decision = str(data.get("decision", "HOLD")).upper().strip()
        if decision not in {"BUY", "SELL", "HOLD"}:
            decision = "HOLD"

        buy_score = _clamp01(data.get("buy_score"), default=0.0)
        sell_score = _clamp01(data.get("sell_score"), default=0.0)
        hold_score = _clamp01(data.get("hold_score"), default=1.0)
        alignment = _clamp01(data.get("alignment_score"), default=max(buy_score, sell_score, hold_score))
        distraction_risk = _clamp01(data.get("distraction_risk"), default=1.0 - alignment)
        rationale = str(data.get("rationale", "")).strip()[:160]

        return LLMDecision(
            decision=decision,
            buy_score=buy_score,
            sell_score=sell_score,
            hold_score=hold_score,
            alignment_score=alignment,
            distraction_risk=distraction_risk,
            rationale=rationale or "parsed",
            mode="api",
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Fast path
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Find first JSON object
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError("no_json_object")
        return json.loads(m.group(0))

    def _simulate(self, context: Dict[str, Any], proposed_action: int) -> LLMDecision:
        class_probs = context.get("class_probs") or {}
        buy_prob = _clamp01(class_probs.get("buy", class_probs.get("up", 0.0)))
        sell_prob = _clamp01(class_probs.get("sell", class_probs.get("down", 0.0)))
        hold_prob = _clamp01(class_probs.get("hold", max(0.0, 1.0 - max(buy_prob, sell_prob))))

        model_conf = _clamp01(context.get("model_confidence", 0.0))
        signal_strength = float(np.clip(abs(float(context.get("signal_value", 0.0))), 0.0, 1.0))
        trend_score = float(np.clip(context.get("trend_score", 0.0), -1.0, 1.0))
        volatility = _clamp01(context.get("volatility_score", 0.0))
        noise = _clamp01(context.get("noise_score", 0.0))

        buy_score = np.clip(
            0.58 * buy_prob
            + 0.18 * max(trend_score, 0.0)
            + 0.14 * signal_strength
            + 0.10 * model_conf
            - 0.22 * noise
            - 0.14 * volatility,
            0.0,
            1.0,
        )
        sell_score = np.clip(
            0.58 * sell_prob
            + 0.18 * max(-trend_score, 0.0)
            + 0.14 * signal_strength
            + 0.10 * model_conf
            - 0.22 * noise
            - 0.14 * volatility,
            0.0,
            1.0,
        )
        hold_score = np.clip(
            0.52 * hold_prob
            + 0.24 * noise
            + 0.14 * volatility
            + 0.10 * (1.0 - model_conf),
            0.0,
            1.0,
        )

        if hold_score >= max(buy_score, sell_score):
            decision = "HOLD"
        elif buy_score >= sell_score:
            decision = "BUY"
        else:
            decision = "SELL"

        alignment = float(np.clip(max(buy_score, sell_score, hold_score) - 0.25 * noise, 0.0, 1.0))

        return LLMDecision(
            decision=decision,
            buy_score=float(buy_score),
            sell_score=float(sell_score),
            hold_score=float(hold_score),
            alignment_score=alignment,
            distraction_risk=float(np.clip(0.6 * noise + 0.4 * volatility, 0.0, 1.0)),
            rationale="simulated_llm",
            mode="simulated",
            model="simulated_gemini_guard",
            key_slot=0,
        )

    def _pass_decision(self, proposed_action: int, mode: str) -> LLMDecision:
        if proposed_action > 0:
            return LLMDecision(
                decision="BUY",
                buy_score=1.0,
                sell_score=0.0,
                hold_score=0.0,
                alignment_score=1.0,
                distraction_risk=0.0,
                rationale="llm_disabled_pass",
                mode=mode,
            )
        if proposed_action < 0:
            return LLMDecision(
                decision="SELL",
                buy_score=0.0,
                sell_score=1.0,
                hold_score=0.0,
                alignment_score=1.0,
                distraction_risk=0.0,
                rationale="llm_disabled_pass",
                mode=mode,
            )
        return LLMDecision(
            decision="HOLD",
            buy_score=0.0,
            sell_score=0.0,
            hold_score=1.0,
            alignment_score=1.0,
            distraction_risk=0.0,
            rationale="llm_disabled_pass",
            mode=mode,
        )

    def _fallback(self, proposed_action: int, error: str) -> LLMDecision:
        if self.config.fail_mode == "pass":
            decision = self._pass_decision(proposed_action, mode="fallback_pass")
            decision.error = error
            decision.rationale = "llm_fail_open"
            return decision

        decision = LLMDecision(
            decision="HOLD",
            buy_score=0.0,
            sell_score=0.0,
            hold_score=1.0,
            alignment_score=0.0,
            distraction_risk=1.0,
            rationale="llm_fail_closed",
            mode="fallback_hold",
            error=error,
        )
        return decision
