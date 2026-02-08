import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch

from quantum_alpha import main
from quantum_alpha.data.collectors.market_data import DataCollector


def _fake_fetch_ohlcv(self, symbol, start, end, interval="1d", use_cache=True):
    """Return a tiny, deterministic OHLCV frame for smoke tests."""
    idx = pd.date_range(end - timedelta(days=30), end, freq="D")
    close = np.linspace(100, 102, len(idx))
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1e6, 1.2e6, len(idx)),
        },
        index=idx,
    )
    df["returns"] = df["close"].pct_change()
    return df


def test_run_paper_smoke():
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()

    with patch.object(DataCollector, "fetch_ohlcv", _fake_fetch_ohlcv):
        result = main.run_paper(
            symbols=["SPY"],
            start_date=start,
            end_date=end,
            initial_capital=100_000,
            strategy_type="momentum",
            paper_bars=10,
            verbose=False,
        )

    assert "metrics" in result
    metrics = result["metrics"]
    # Basic sanity: metrics numeric and trades processed
    assert metrics["n_trades"] >= 0
    assert not np.isnan(metrics["total_return"])

