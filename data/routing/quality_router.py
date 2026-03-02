"""Deterministic quality-scored router for multi-provider selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from quantum_alpha.data.providers.base import ProviderResult


@dataclass
class RouterDecision:
    domain: str
    selected_provider: str
    selected_score: float
    degraded: bool
    tried: List[Dict[str, object]] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None


class DeterministicQualityRouter:
    """Quality router with deterministic scoring and static tie-breaks."""

    DEFAULT_WEIGHTS = {
        "freshness": 0.20,
        "completeness": 0.40,
        "latency": 0.20,
        "reliability": 0.20,
    }

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}

    @staticmethod
    def _freshness_score(result: ProviderResult) -> float:
        age_s = float(result.metadata.get("age_seconds", 0.0) or 0.0)
        if age_s <= 0:
            return 1.0
        return float(1.0 / (1.0 + age_s / 300.0))

    @staticmethod
    def _latency_score(result: ProviderResult) -> float:
        latency_ms = max(float(result.latency_ms or 0.0), 0.0)
        return float(1.0 / (1.0 + latency_ms / 1000.0))

    def _weights_for_domain(self, domain: str) -> Dict[str, float]:
        domain_cfg = (self.config.get(domain) or {}) if isinstance(self.config, dict) else {}
        weights = dict(self.DEFAULT_WEIGHTS)
        weights.update(domain_cfg.get("weights", {}))
        total = sum(max(float(v), 0.0) for v in weights.values())
        if total <= 0:
            return dict(self.DEFAULT_WEIGHTS)
        return {k: float(max(float(v), 0.0) / total) for k, v in weights.items()}

    def _provider_order(self, domain: str) -> List[str]:
        domain_cfg = (self.config.get(domain) or {}) if isinstance(self.config, dict) else {}
        order = domain_cfg.get("provider_order", [])
        if not isinstance(order, list):
            return []
        return [str(x) for x in order]

    def _min_score(self, domain: str) -> float:
        domain_cfg = (self.config.get(domain) or {}) if isinstance(self.config, dict) else {}
        return float(domain_cfg.get("min_score", 0.0))

    def _score(self, result: ProviderResult, weights: Dict[str, float]) -> float:
        freshness = self._freshness_score(result)
        completeness = float(min(max(result.completeness, 0.0), 1.0))
        latency = self._latency_score(result)
        reliability = float(min(max(result.reliability, 0.0), 1.0))
        return float(
            weights.get("freshness", 0.0) * freshness
            + weights.get("completeness", 0.0) * completeness
            + weights.get("latency", 0.0) * latency
            + weights.get("reliability", 0.0) * reliability
        )

    def select(self, domain: str, candidates: List[ProviderResult]) -> RouterDecision:
        if not candidates:
            return RouterDecision(
                domain=domain,
                selected_provider="none",
                selected_score=0.0,
                degraded=True,
                error="no_candidates",
            )

        weights = self._weights_for_domain(domain)
        order = self._provider_order(domain)
        order_idx = {name: idx for idx, name in enumerate(order)}

        scored: List[tuple[float, int, ProviderResult]] = []
        tried = []
        for result in candidates:
            score = self._score(result, weights)
            tie_rank = order_idx.get(result.provider, len(order_idx) + 999)
            scored.append((score, -tie_rank, result))
            tried.append(
                {
                    "provider": result.provider,
                    "score": score,
                    "latency_ms": float(result.latency_ms),
                    "completeness": float(result.completeness),
                    "reliability": float(result.reliability),
                    "error": result.error,
                }
            )

        # Higher score first; then better tie-break from provider_order.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected_score, _, selected = scored[0]
        degraded = bool(selected.error) or selected_score < self._min_score(domain)

        return RouterDecision(
            domain=domain,
            selected_provider=selected.provider,
            selected_score=float(selected_score),
            degraded=degraded,
            tried=tried,
            error=selected.error,
        )


__all__ = ["RouterDecision", "DeterministicQualityRouter"]
