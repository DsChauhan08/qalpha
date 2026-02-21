"""
LLM teacher-distillation utilities for model training.

This module converts feature windows into compact LLM contexts, queries the
Gemini router (simulated or API mode), and returns:
- distilled labels/confidence targets
- per-sample training weights
- teacher action/alignment diagnostics

Design goals:
- Keep inference feature dimensions unchanged
- Fail-safe defaults on all errors
- Rate-limit friendly via max-calls subsampling
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .gemini_router import GeminiRouter


_TS_RE = re.compile(r"^t(\d+)_(.+)$")


@dataclass
class DistillConfig:
    enabled: bool = False
    mode: str | None = None
    env_path: str | None = None
    min_alignment: float = 0.75
    blend: float = 0.35
    max_calls: int = 0
    seed: int = 42
    fail_mode: str = "hold"


def _clip01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(np.clip(v, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return float(default)
    return float(np.mean(vals))


def _safe_get(vec: np.ndarray, index: Dict[str, int], names: list[str]) -> float:
    for name in names:
        i = index.get(name)
        if i is None:
            continue
        try:
            v = float(vec[i])
            if np.isfinite(v):
                return v
        except Exception:
            continue
    return 0.0


def _normalize_feature_view(
    X: np.ndarray,
    feature_names: list[str] | None,
) -> Tuple[np.ndarray, list[str]]:
    """
    Return (latest_feature_matrix, normalized_feature_names).

    - For sequence tensors [N, T, F], use the latest timestep.
    - For tabular [N, F], collapse flattened names like t0_x/t1_x to latest tK_x.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 3:
        n_feat = int(X_arr.shape[2])
        names = list(feature_names or [])
        if len(names) != n_feat:
            names = [f"f{i}" for i in range(n_feat)]
        return X_arr[:, -1, :], names

    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D or 3D, got shape {X_arr.shape}")

    n_feat = int(X_arr.shape[1])
    names = list(feature_names or [])
    if len(names) != n_feat:
        return X_arr, [f"f{i}" for i in range(n_feat)]

    latest: Dict[str, Tuple[int, int]] = {}
    for idx, name in enumerate(names):
        m = _TS_RE.match(name)
        if not m:
            # Keep plain names as-is with synthetic time index 0.
            base = name
            t_idx = 0
        else:
            t_idx = int(m.group(1))
            base = m.group(2)

        old = latest.get(base)
        if old is None or t_idx > old[0]:
            latest[base] = (t_idx, idx)

    ordered = sorted(latest.items(), key=lambda kv: kv[0])
    keep_idx = [idx for _, (_, idx) in ordered]
    keep_names = [base for base, _ in ordered]
    return X_arr[:, keep_idx], keep_names


def _label_to_action(label: int, n_classes: int) -> int:
    y = int(label)
    if n_classes == 2:
        return 1 if y == 1 else -1
    if y == 2:
        return 1
    if y == 0:
        return -1
    return 0


def _decision_to_class(decision: str, n_classes: int) -> int | None:
    d = str(decision).upper().strip()
    if n_classes == 2:
        if d == "BUY":
            return 1
        if d == "SELL":
            return 0
        return None

    if d == "BUY":
        return 2
    if d == "SELL":
        return 0
    if d == "HOLD":
        return 1
    return None


def _decision_to_action(decision: str) -> int:
    d = str(decision).upper().strip()
    if d == "BUY":
        return 1
    if d == "SELL":
        return -1
    return 0


def _build_context(vec: np.ndarray, names: list[str], n_classes: int) -> tuple[dict, int]:
    idx = {n: i for i, n in enumerate(names)}

    trend = np.tanh(
        _safe_mean(
            [
                _safe_get(vec, idx, ["trend_strength"]),
                _safe_get(vec, idx, ["returns_accel"]),
                _safe_get(vec, idx, ["sentiment_momentum", "sentiment_momentum_3d", "sentiment_momentum_7d"]),
                _safe_get(vec, idx, ["mom_3m"]),
                _safe_get(vec, idx, ["mom_12m"]),
                _safe_get(vec, idx, ["macd_hist"]),
            ]
        )
    )
    sentiment = np.tanh(
        _safe_mean(
            [
                _safe_get(vec, idx, ["weighted_sentiment"]),
                _safe_get(vec, idx, ["mean_sentiment"]),
                _safe_get(vec, idx, ["sentiment_proxy"]),
                _safe_get(vec, idx, ["news_sentiment"]),
            ]
        )
    )
    volatility = np.clip(
        _safe_mean(
            [
                abs(_safe_get(vec, idx, ["vol_regime"])),
                abs(_safe_get(vec, idx, ["return_zscore"])),
                abs(_safe_get(vec, idx, ["atr_pct"])),
                abs(_safe_get(vec, idx, ["range_surprise"])),
            ]
        )
        / 1.8,
        0.0,
        1.0,
    )
    noise = np.clip(
        _safe_mean(
            [
                abs(_safe_get(vec, idx, ["vol_price_div"])),
                abs(_safe_get(vec, idx, ["gap_fill_rate"])),
                abs(_safe_get(vec, idx, ["overnight_gap_zscore"])),
                abs(_safe_get(vec, idx, ["volume_surprise"])),
            ]
        )
        / 1.8,
        0.0,
        1.0,
    )
    signal_value = float(np.clip(0.62 * trend + 0.38 * sentiment, -1.0, 1.0))
    sentiment_strength = float(np.clip(abs(sentiment), 0.0, 1.0))

    buy_raw = 2.1 * signal_value - 0.95 * noise - 0.85 * volatility
    sell_raw = -2.1 * signal_value - 0.95 * noise - 0.85 * volatility
    buy_prob = _sigmoid(buy_raw)
    sell_prob = _sigmoid(sell_raw)
    hold_prob = float(
        np.clip(1.0 - max(buy_prob, sell_prob) + 0.32 * (noise + volatility), 0.0, 1.0)
    )
    denom = buy_prob + sell_prob + hold_prob + 1e-8
    buy_prob /= denom
    sell_prob /= denom
    hold_prob /= denom

    if signal_value > 0.08:
        proposed_action = 1
    elif signal_value < -0.08:
        proposed_action = -1
    else:
        proposed_action = 0

    if n_classes == 2:
        class_probs = {"down": sell_prob, "up": buy_prob}
    else:
        class_probs = {"sell": sell_prob, "hold": hold_prob, "buy": buy_prob}

    context = {
        "class_probs": class_probs,
        "model_confidence": float(max(buy_prob, sell_prob, hold_prob)),
        "signal_value": signal_value,
        "trend_score": float(np.clip(trend, -1.0, 1.0)),
        "volatility_score": float(volatility),
        "noise_score": float(noise),
        "sentiment_strength": sentiment_strength,
    }
    return context, proposed_action


def _resolve_eval_indices(n_samples: int, max_calls: int, mode: str, seed: int) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros(0, dtype=np.int64)

    if max_calls <= 0:
        # Evaluate all rows in simulated mode; cap API calls for safety.
        max_calls = n_samples if mode != "api" else min(600, n_samples)

    n_eval = int(min(n_samples, max_calls))
    if n_eval >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n_samples, size=n_eval, replace=False))
    return idx.astype(np.int64)


def distill_supervision(
    X: np.ndarray,
    y_signal: np.ndarray,
    feature_names: list[str] | None,
    n_classes: int,
    y_conf: np.ndarray | None = None,
    *,
    config: DistillConfig | None = None,
) -> Dict[str, Any]:
    """
    Distill teacher knowledge from Gemini into training targets and weights.

    Returns:
      {
        "y_signal": np.ndarray[int64],
        "y_conf": np.ndarray[float32] | None,
        "sample_weight": np.ndarray[float32],
        "teacher_action": np.ndarray[int8],   # -1 sell, 0 hold, +1 buy
        "alignment": np.ndarray[float32],
        "report": dict
      }
    """
    cfg = config or DistillConfig()
    y_in = np.asarray(y_signal, dtype=np.int64)
    n_samples = int(len(y_in))

    y_out = y_in.copy()
    y_conf_out = None if y_conf is None else np.asarray(y_conf, dtype=np.float32).copy()
    sample_weight = np.ones(n_samples, dtype=np.float32)
    teacher_action = np.zeros(n_samples, dtype=np.int8)
    alignment = np.full(n_samples, 0.5, dtype=np.float32)

    if not cfg.enabled or n_samples == 0:
        return {
            "y_signal": y_out,
            "y_conf": y_conf_out,
            "sample_weight": sample_weight,
            "teacher_action": teacher_action,
            "alignment": alignment,
            "report": {
                "enabled": 0.0,
                "mode": "off",
                "n_samples": float(n_samples),
                "n_evaluated": 0.0,
                "mean_alignment": 0.5,
                "mean_weight": float(sample_weight.mean()) if n_samples else 1.0,
                "relabeled_to_hold": 0.0,
                "relabeled_from_hold": 0.0,
            },
        }

    mode = str(cfg.mode or "").strip().lower() or "simulated"
    if mode not in {"simulated", "api", "off"}:
        mode = "simulated"
    if mode == "off":
        cfg.enabled = False
        return distill_supervision(
            X,
            y_signal=y_in,
            feature_names=feature_names,
            n_classes=n_classes,
            y_conf=y_conf_out,
            config=cfg,
        )

    X_latest, names = _normalize_feature_view(np.asarray(X), feature_names)
    if len(X_latest) != n_samples:
        raise ValueError(
            f"X/y length mismatch: X={len(X_latest)} y={n_samples}"
        )

    router = GeminiRouter.from_env(
        enabled=True,
        mode=mode,
        min_alignment_score=float(cfg.min_alignment),
        fail_mode=str(cfg.fail_mode),
        env_path=cfg.env_path,
    )

    eval_idx = _resolve_eval_indices(
        n_samples=n_samples,
        max_calls=int(cfg.max_calls),
        mode=router.config.mode,
        seed=int(cfg.seed),
    )
    eval_set = set(int(i) for i in eval_idx.tolist())

    relabel_hold = 0
    relabel_direction = 0
    high_alignment = 0

    for i in range(n_samples):
        if i not in eval_set:
            continue

        row = X_latest[i]
        context, proposed_action = _build_context(row, names, n_classes=n_classes)
        decision = router.evaluate(context, proposed_action=proposed_action)

        a = _clip01(decision.alignment_score, default=0.0)
        d = _clip01(decision.distraction_risk, default=1.0)
        alignment[i] = a
        teacher_action[i] = np.int8(_decision_to_action(decision.decision))
        if a >= float(cfg.min_alignment):
            high_alignment += 1

        # Distilled confidence target (if provided)
        if y_conf_out is not None:
            true_action = _label_to_action(int(y_in[i]), n_classes=n_classes)
            teacher_score = _clip01(decision.score_for_action(true_action), default=0.0)
            target_conf = float(np.clip(0.62 * a + 0.38 * teacher_score, 0.0, 1.0))
            blend = _clip01(cfg.blend, default=0.35)
            y_conf_out[i] = np.float32(
                np.clip((1.0 - blend) * float(y_conf_out[i]) + blend * target_conf, 0.0, 1.0)
            )

        # Conservative relabeling for ternary tasks
        teacher_cls = _decision_to_class(decision.decision, n_classes=n_classes)
        if n_classes == 3 and teacher_cls is not None and a >= float(cfg.min_alignment):
            if teacher_cls == 1 and y_out[i] != 1:
                y_out[i] = 1
                relabel_hold += 1
            elif y_out[i] == 1 and teacher_cls in {0, 2}:
                y_out[i] = np.int64(teacher_cls)
                relabel_direction += 1

        # Per-sample weighting
        w = 0.80 + 0.75 * a - 0.35 * d
        if teacher_cls is not None and a >= float(cfg.min_alignment):
            if teacher_cls == int(y_in[i]):
                w += 0.20
            else:
                w -= 0.20
        if str(decision.decision).upper().strip() == "HOLD":
            w -= 0.05
        sample_weight[i] = np.float32(np.clip(w, 0.20, 2.50))

    return {
        "y_signal": y_out.astype(np.int64),
        "y_conf": y_conf_out.astype(np.float32) if y_conf_out is not None else None,
        "sample_weight": sample_weight.astype(np.float32),
        "teacher_action": teacher_action,
        "alignment": alignment.astype(np.float32),
        "report": {
            "enabled": 1.0,
            "mode": router.config.mode,
            "n_samples": float(n_samples),
            "n_evaluated": float(len(eval_idx)),
            "mean_alignment": float(alignment[eval_idx].mean()) if len(eval_idx) else 0.0,
            "mean_weight": float(sample_weight.mean()) if n_samples else 1.0,
            "high_alignment_count": float(high_alignment),
            "relabeled_to_hold": float(relabel_hold),
            "relabeled_from_hold": float(relabel_direction),
        },
    }

