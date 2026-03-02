from datetime import datetime

import numpy as np
import pandas as pd

from quantum_alpha.portfolio import PortfolioAllocatorEngine
from quantum_alpha.portfolio.risk_metrics import historical_cvar, historical_var, ulcer_index
from quantum_alpha.portfolio.volatility import (
    garman_klass_vol,
    parkinson_vol,
    rogers_satchell_vol,
    yang_zhang_vol,
)


def _ohlcv(seed: int, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0.0, 0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.003, 0.002, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.003, 0.002, size=n)))
    volume = rng.integers(900_000, 1_500_000, size=n)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    df["returns"] = df["close"].pct_change().fillna(0.0)
    return df


def test_volatility_estimators_return_finite_positive_values():
    df = _ohlcv(42)
    vals = [
        parkinson_vol(df["high"], df["low"]),
        garman_klass_vol(df["open"], df["high"], df["low"], df["close"]),
        rogers_satchell_vol(df["open"], df["high"], df["low"], df["close"]),
        yang_zhang_vol(df["open"], df["high"], df["low"], df["close"]),
    ]
    assert all(np.isfinite(v) for v in vals)
    assert all(v >= 0.0 for v in vals)


def test_risk_metrics_are_ordered_for_left_tail():
    r = pd.Series([0.01, -0.02, 0.005, -0.04, 0.02, -0.01, 0.015])
    var95 = historical_var(r, alpha=0.95)
    cvar95 = historical_cvar(r, alpha=0.95)
    ui = ulcer_index(r)
    assert cvar95 <= var95
    assert ui >= 0.0


def test_allocator_enforces_long_short_net_gross_and_position_caps():
    featured = {
        "AAA": _ohlcv(1),
        "BBB": _ohlcv(2),
        "CCC": _ohlcv(3),
        "DDD": _ohlcv(4),
    }
    raw_scores = {"AAA": 1.5, "BBB": -1.2, "CCC": 0.8, "DDD": -0.9}

    engine = PortfolioAllocatorEngine.from_config(
        {
            "constraints": {
                "long_short_enabled": True,
                "net_min": -0.15,
                "net_max": 0.15,
                "gross_max": 0.85,
                "max_position_abs": 0.25,
            },
            "selector": {"window_cycles": 10, "switch_margin": 0.01, "min_hold_cycles": 1},
            "vol_estimator": {"primary": "yang_zhang"},
        }
    )

    out = engine.allocate(
        signal_scores=raw_scores,
        featured=featured,
        timestamp=datetime(2026, 2, 28),
    )

    weights = pd.Series(out.weights)
    assert (weights.abs() <= 0.25 + 1e-9).all()
    assert float(weights.abs().sum()) <= 0.85 + 1e-6
    assert -0.15 - 1e-6 <= float(weights.sum()) <= 0.15 + 1e-6
    assert out.chosen_stack in {"static", "regime"}
