from datetime import datetime, timezone

from quantum_alpha.data.providers.base import ProviderResult
from quantum_alpha.data.routing.quality_router import DeterministicQualityRouter


def _result(provider: str, completeness: float, latency_ms: float, reliability: float = 1.0):
    return ProviderResult(
        data={"ok": True},
        provider=provider,
        domain="market_data",
        fetched_at=datetime.now(timezone.utc),
        latency_ms=latency_ms,
        completeness=completeness,
        reliability=reliability,
        metadata={},
    )


def test_quality_router_selects_best_score():
    router = DeterministicQualityRouter(
        config={
            "market_data": {
                "weights": {
                    "freshness": 0.2,
                    "completeness": 0.5,
                    "latency": 0.2,
                    "reliability": 0.1,
                },
                "provider_order": ["openbb_sdk", "yfinance"],
                "min_score": 0.1,
            }
        }
    )
    candidates = [
        _result("yfinance", completeness=0.8, latency_ms=20),
        _result("openbb_sdk", completeness=0.95, latency_ms=35),
    ]

    decision = router.select("market_data", candidates)
    assert decision.selected_provider == "openbb_sdk"
    assert decision.degraded is False
    assert decision.selected_score > 0.1


def test_quality_router_tie_break_uses_provider_order():
    router = DeterministicQualityRouter(
        config={
            "market_data": {
                "weights": {
                    "freshness": 0.0,
                    "completeness": 1.0,
                    "latency": 0.0,
                    "reliability": 0.0,
                },
                "provider_order": ["preferred", "backup"],
                "min_score": 0.0,
            }
        }
    )
    candidates = [
        _result("backup", completeness=1.0, latency_ms=1),
        _result("preferred", completeness=1.0, latency_ms=999),
    ]

    decision = router.select("market_data", candidates)
    assert decision.selected_provider == "preferred"


def test_quality_router_marks_degraded_when_below_min_score():
    router = DeterministicQualityRouter(
        config={
            "market_data": {
                "weights": {
                    "freshness": 0.0,
                    "completeness": 1.0,
                    "latency": 0.0,
                    "reliability": 0.0,
                },
                "provider_order": ["a"],
                "min_score": 0.9,
            }
        }
    )
    decision = router.select("market_data", [_result("a", completeness=0.2, latency_ms=10)])
    assert decision.selected_provider == "a"
    assert decision.degraded is True
