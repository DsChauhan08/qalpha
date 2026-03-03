from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantum_alpha.execution.realtime_paper import (
    SessionConfig,
    _compute_target_weights_meta_blend,
    _resolve_ab_group,
)


class _FakeGateStrategy:
    def generate_signals_live(self, df, symbol="SPY", headlines=None):
        if symbol == "AAA":
            return {"llm_gate_pass": False, "llm_decision": "HOLD"}
        return {"llm_gate_pass": True, "llm_decision": "BUY"}


def _cfg() -> SessionConfig:
    return SessionConfig(
        symbols=["AAA", "BBB", "CCC"],
        full_universe_requested=True,
        interval="5m",
        duration_minutes=30,
        poll_seconds=60,
        lookback_days=10,
        capital=10000.0,
        max_position_size=0.4,
        max_portfolio_leverage=1.0,
        signal_threshold=0.1,
        min_long_signal=0.0,
        long_only=True,
        strategy_type="meta_blend_hybrid",
        checkpoint_name=None,
        llm_enabled=True,
        llm_mode="simulated",
        llm_models=None,
        llm_min_alignment=0.8,
        llm_fail_mode="hold",
        llm_scope="latest",
        llm_max_calls=100,
        llm_decision_interval_cycles=3,
        llm_env_path=None,
        news_poll_seconds=300,
        news_max_articles=6,
        news_symbol_limit=10,
        liquid_subset_size=0,
        liquid_adv_window=20,
        liquid_min_history=10,
        liquid_for_full_only=True,
        universe_refresh_cycles=30,
        commission_rate=0.0,
        slippage_bps=0.0,
        min_commission=0.0,
        min_trade_notional=25.0,
        output_dir=Path("."),
    )


def _frames():
    idx = pd.date_range("2026-02-20", periods=40, freq="5min")
    out = {}
    for s in ["AAA", "BBB", "CCC"]:
        close = np.linspace(100, 101, len(idx))
        out[s] = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.full(len(idx), 1_000_000.0),
                "returns": pd.Series(close).pct_change().fillna(0.0).values,
            },
            index=idx,
        )
    return out


def test_meta_blend_topn_gate_blocks_only_top_candidate():
    cfg = _cfg()
    featured = _frames()
    anchors = {
        "AAA": {
            "up_probability_blend": 0.90,
            "confidence": 0.80,
            "up_probability_base": 0.85,
            "up_probability_mc": 0.92,
            "missing_feature_ratio_base": 0.01,
            "missing_feature_ratio_mc": 0.02,
            "model_used": "blended",
        },
        "BBB": {
            "up_probability_blend": 0.70,
            "confidence": 0.40,
            "up_probability_base": 0.68,
            "up_probability_mc": 0.72,
            "missing_feature_ratio_base": 0.01,
            "missing_feature_ratio_mc": 0.02,
            "model_used": "blended",
        },
        "CCC": {
            "up_probability_blend": 0.52,
            "confidence": 0.04,
            "up_probability_base": 0.52,
            "up_probability_mc": 0.52,
            "missing_feature_ratio_base": 0.01,
            "missing_feature_ratio_mc": 0.02,
            "model_used": "blended",
        },
    }

    weights, decisions, alloc_diag, health = _compute_target_weights_meta_blend(
        featured=featured,
        cfg=cfg,
        anchor_predictions=anchors,
        llm_active=True,
        news_cache={},
        gate_strategy=_FakeGateStrategy(),
        allocator=None,
        timestamp=None,
        top_n_for_gate=1,
    )

    assert isinstance(alloc_diag, dict)
    assert weights["AAA"] == 0.0  # gated out
    assert weights["BBB"] > 0.0   # not in top-1 gate scope
    assert float(health["feature_missing_ratio_base"]) < 0.1

    d_by_sym = {d["symbol"]: d for d in decisions}
    assert d_by_sym["AAA"]["llm_gate_pass"] is False
    assert d_by_sym["BBB"]["llm_gate_pass"] is True


def test_resolve_ab_group_auto_parity():
    cfg = _cfg()
    cfg.ab_group = "auto"
    # 2026-03-02 -> even day token => A
    grp = _resolve_ab_group(cfg, datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc))
    assert grp == "A"
