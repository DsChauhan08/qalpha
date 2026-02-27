import pandas as pd

from quantum_alpha.backtesting.robustness_suite import (
    _benchmark_relative_metrics,
    _score_candidate,
)
from quantum_alpha.main import _rolling_oos_vs_benchmark, _select_liquid_subset


def _price_df(close_vals, vol_vals):
    idx = pd.date_range("2025-01-01", periods=len(close_vals), freq="D")
    return pd.DataFrame(
        {
            "open": close_vals,
            "high": close_vals,
            "low": close_vals,
            "close": close_vals,
            "volume": vol_vals,
        },
        index=idx,
    )


def test_select_liquid_subset_by_dollar_volume():
    frames = {
        "LOW": _price_df([10, 10, 10, 10, 10, 10], [100, 100, 100, 100, 100, 100]),
        "MID": _price_df(
            [20, 20, 20, 20, 20, 20],
            [500, 500, 500, 500, 500, 500],
        ),
        "HI": _price_df(
            [30, 30, 30, 30, 30, 30],
            [1000, 1000, 1000, 1000, 1000, 1000],
        ),
    }
    selected = _select_liquid_subset(
        frames,
        subset_size=2,
        adv_window=3,
        min_history=3,
    )
    assert selected == ["HI", "MID"]


def test_rolling_oos_vs_benchmark_passes_with_three_beats():
    idx = pd.date_range("2025-01-01", periods=12, freq="D")
    strat = pd.Series(
        [0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00],
        index=idx,
    )
    bench = pd.Series(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        index=idx,
    )
    res = _rolling_oos_vs_benchmark(
        strategy_returns=strat,
        benchmark_returns=bench,
        window_days=3,
        min_windows=3,
    )
    assert res["available"] is True
    assert res["n_windows"] == 4
    assert res["beats"] == 4
    assert res["passed"] is True


def test_benchmark_relative_metrics_information_ratio_positive():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    strat = pd.Series([0.012, 0.003, 0.011, 0.004, 0.013, 0.002, 0.010, 0.005, 0.011, 0.004], index=idx)
    bench = pd.Series([0.007, 0.004, 0.008, 0.003, 0.009, 0.003, 0.007, 0.004, 0.008, 0.003], index=idx)
    rel = _benchmark_relative_metrics(strat, bench)
    assert rel["information_ratio"] > 0


def test_score_candidate_rewards_excess_vs_quant():
    seg = {
        "full": {
            "model": {"annual_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.10},
            "equal_weight": {
                "annual_return": 0.08,
                "sharpe": 0.5,
                "max_drawdown": -0.12,
            },
            "spy": {"annual_return": 0.10, "sharpe": 0.6, "max_drawdown": -0.11},
            "quant_composite": {
                "annual_return": 0.15,
                "sharpe": 0.8,
                "max_drawdown": -0.13,
            },
        },
        "old": {
            "model": {"annual_return": 0.18, "sharpe": 0.9, "max_drawdown": -0.12},
            "equal_weight": {
                "annual_return": 0.07,
                "sharpe": 0.4,
                "max_drawdown": -0.11,
            },
            "spy": {"annual_return": 0.09, "sharpe": 0.5, "max_drawdown": -0.10},
            "quant_composite": {
                "annual_return": 0.13,
                "sharpe": 0.7,
                "max_drawdown": -0.12,
            },
        },
        "recent": {
            "model": {"annual_return": 0.22, "sharpe": 1.1, "max_drawdown": -0.09},
            "equal_weight": {
                "annual_return": 0.10,
                "sharpe": 0.6,
                "max_drawdown": -0.11,
            },
            "spy": {"annual_return": 0.12, "sharpe": 0.7, "max_drawdown": -0.10},
            "quant_composite": {
                "annual_return": 0.16,
                "sharpe": 0.9,
                "max_drawdown": -0.12,
            },
        },
        "very_recent": {
            "model": {"annual_return": 0.21, "sharpe": 1.0, "max_drawdown": -0.08},
            "equal_weight": {
                "annual_return": 0.09,
                "sharpe": 0.5,
                "max_drawdown": -0.10,
            },
            "spy": {"annual_return": 0.11, "sharpe": 0.6, "max_drawdown": -0.09},
            "quant_composite": {
                "annual_return": 0.14,
                "sharpe": 0.8,
                "max_drawdown": -0.11,
            },
        },
    }
    score = _score_candidate(seg, rel_full={"information_ratio": 0.2})
    assert score > 0
