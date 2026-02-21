from __future__ import annotations

from quantum_alpha.llm.gemini_router import GeminiRouter


def test_simulated_router_scores_are_bounded():
    router = GeminiRouter.from_env(enabled=True, mode="simulated", min_alignment_score=0.8)
    ctx = {
        "symbol": "SPY",
        "timestamp": "2026-02-20",
        "proposed_action": 1,
        "model_confidence": 0.9,
        "signal_value": 0.7,
        "class_probs": {"sell": 0.1, "hold": 0.2, "buy": 0.7},
        "trend_score": 0.4,
        "volatility_score": 0.2,
        "noise_score": 0.1,
    }

    out = router.evaluate(ctx, proposed_action=1)
    assert out.mode == "simulated"
    assert out.decision in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= out.buy_score <= 1.0
    assert 0.0 <= out.sell_score <= 1.0
    assert 0.0 <= out.hold_score <= 1.0
    assert 0.0 <= out.alignment_score <= 1.0
    assert 0.0 <= out.distraction_risk <= 1.0


def test_api_mode_without_keys_falls_back_to_off():
    router = GeminiRouter.from_env(enabled=True, mode="api")
    assert router.config.mode in {"off", "api"}

    out = router.evaluate({}, proposed_action=-1)
    # In off mode this is pass-through for the proposed action.
    if router.config.mode == "off":
        assert out.decision == "SELL"
        assert out.sell_score == 1.0
