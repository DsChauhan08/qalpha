from datetime import datetime

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.benchmark_profiles import (
    BenchmarkProfile,
    benchmark_rows,
    build_profile_returns,
    evaluate_quant_firm_benchmarks,
    resolve_profiles,
)


class _CollectorStub:
    def __init__(self, price_map):
        self.price_map = price_map

    def fetch_ohlcv(self, symbol, start, end, interval="1d", use_cache=True):
        if symbol not in self.price_map:
            raise ValueError(f"missing {symbol}")
        return self.price_map[symbol]


def _price_df(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="D")
    arr = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": arr,
            "high": arr,
            "low": arr,
            "close": arr,
            "volume": np.full(len(arr), 1_000_000),
        },
        index=idx,
    )


def test_resolve_profiles_rejects_unknown():
    try:
        resolve_profiles(["not_real"])
    except ValueError as exc:
        assert "Unknown quant benchmark profile" in str(exc)
    else:
        assert False, "Expected ValueError for unknown profile"


def test_build_profile_returns_renormalises_partial_weights():
    profile = BenchmarkProfile(
        key="demo",
        label="Demo",
        description="demo",
        tickers=("AAA", "BBB", "CCC"),
        weights=(0.2, 0.3, 0.5),
    )
    collector = _CollectorStub(
        {
            "AAA": _price_df([100, 101, 102, 103]),
            "BBB": _price_df([100, 102, 104, 106]),
        }
    )

    ret, used, failed = build_profile_returns(
        collector=collector,
        profile=profile,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 4),
    )

    assert used == ["AAA", "BBB"]
    assert failed == ["CCC"]
    assert len(ret) == 4
    # Available weights should be renormalized from (0.2, 0.3) -> (0.4, 0.6)
    expected_last = 0.4 * (103 / 102 - 1) + 0.6 * (106 / 104 - 1)
    assert abs(ret.iloc[-1] - expected_last) < 1e-9


def test_evaluate_quant_firm_benchmarks_and_rows():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    strategy_returns = pd.Series(
        [0.0, 0.01, -0.005, 0.007, 0.002, 0.004], index=idx
    )
    collector = _CollectorStub(
        {
            "MTUM": _price_df([100, 101, 100.5, 101.2, 101.7, 102.2]),
            "QUAL": _price_df([100, 100.8, 100.7, 101.0, 101.2, 101.5]),
            "VLUE": _price_df([100, 99.8, 99.9, 100.0, 100.2, 100.3]),
        }
    )

    results = evaluate_quant_firm_benchmarks(
        strategy_returns=strategy_returns,
        collector=collector,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 6),
        profile_names=["aqr_style_factors"],
        min_assets=2,
    )

    assert "aqr_style_factors" in results
    blob = results["aqr_style_factors"]
    assert blob["status"] == "ok"
    assert blob["n_days"] > 0
    assert "strategy_vs_profile_metrics" in blob

    rows = benchmark_rows(results)
    assert len(rows) == 1
    assert rows[0]["profile_key"] == "aqr_style_factors"


def test_evaluate_quant_firm_benchmarks_insufficient_assets():
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    strategy_returns = pd.Series([0.0, 0.01, 0.0, -0.01], index=idx)
    collector = _CollectorStub({"MTUM": _price_df([100, 101, 100, 100.5])})

    results = evaluate_quant_firm_benchmarks(
        strategy_returns=strategy_returns,
        collector=collector,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 4),
        profile_names=["aqr_style_factors"],
        min_assets=2,
    )

    assert results["aqr_style_factors"]["status"] == "insufficient_assets"

