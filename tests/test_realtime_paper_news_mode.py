from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantum_alpha.execution.realtime_paper import (
    SessionConfig,
    _interval_max_lookback_days,
    _llm_cycle_active,
    _compute_target_weights_news_lstm,
    _minutes_until_close,
    _normalize_target_weights,
)


class _FakeNewsStrategy:
    def __init__(self, payload_by_symbol):
        self.payload_by_symbol = payload_by_symbol

    def generate_signals_live(self, df, symbol="SPY", headlines=None):
        return dict(self.payload_by_symbol.get(symbol, {"action": 0, "signal": 0.0, "confidence": 0.0}))


def _cfg(long_only=True, llm_enabled=True):
    return SessionConfig(
        symbols=["AAA", "BBB", "CCC"],
        full_universe_requested=True,
        interval="5m",
        duration_minutes=60,
        poll_seconds=60,
        lookback_days=10,
        capital=10000.0,
        max_position_size=0.4,
        max_portfolio_leverage=1.0,
        signal_threshold=0.1,
        min_long_signal=0.0,
        long_only=long_only,
        strategy_type="news_lstm",
        checkpoint_name=None,
        llm_enabled=llm_enabled,
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


def test_minutes_until_close_at_1040_et():
    dt_utc = datetime(2026, 2, 27, 15, 40, tzinfo=timezone.utc)  # 10:40 ET
    assert _minutes_until_close(dt_utc) == 320


def test_interval_lookback_caps():
    assert _interval_max_lookback_days("1m") == 8
    assert _interval_max_lookback_days("5m") == 59
    assert _interval_max_lookback_days("1h") == 730


def test_llm_cycle_schedule():
    cfg = _cfg(long_only=True, llm_enabled=True)
    assert _llm_cycle_active(1, cfg) is True
    assert _llm_cycle_active(2, cfg) is False
    assert _llm_cycle_active(3, cfg) is False
    assert _llm_cycle_active(4, cfg) is True


def test_normalize_target_weights_respects_caps():
    cfg = _cfg(long_only=True, llm_enabled=False)
    raw_scores = {"AAA": 10.0, "BBB": 1.0, "CCC": 1.0}
    weights = _normalize_target_weights(raw_scores, ["AAA", "BBB", "CCC"], cfg)

    assert all(abs(v) <= cfg.max_position_size + 1e-9 for v in weights.values())
    assert sum(abs(v) for v in weights.values()) <= cfg.max_portfolio_leverage + 1e-9


def test_news_weights_respect_llm_gate_and_long_only():
    cfg = _cfg(long_only=True, llm_enabled=True)
    strategy = _FakeNewsStrategy(
        {
            "AAA": {
                "action": 1,
                "signal": 0.7,
                "confidence": 0.6,
                "llm_alignment_score": 0.9,
                "llm_gate_pass": True,
                "llm_decision": "BUY",
            },
            "BBB": {
                "action": 1,
                "signal": 0.8,
                "confidence": 0.9,
                "llm_alignment_score": 0.9,
                "llm_gate_pass": False,
                "llm_decision": "HOLD",
            },
            "CCC": {
                "action": -1,
                "signal": -0.9,
                "confidence": 0.9,
                "llm_alignment_score": 0.9,
                "llm_gate_pass": True,
                "llm_decision": "SELL",
            },
        }
    )

    featured = _frames()
    weights, decisions, alloc_diag = _compute_target_weights_news_lstm(
        strategy=strategy,
        featured=featured,
        cfg=cfg,
        news_cache={"AAA": ["Headline"]},
        llm_active=True,
    )

    assert weights["AAA"] > 0.0
    assert weights["BBB"] == 0.0  # blocked by llm gate
    assert weights["CCC"] == 0.0  # suppressed by long-only
    assert any(d["symbol"] == "AAA" for d in decisions)
    assert isinstance(alloc_diag, dict)
