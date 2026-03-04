import numpy as np
import pandas as pd
import yaml
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


def test_run_paper_ab_comparison_smoke(tmp_path):
    start = datetime.now() - timedelta(days=60)
    end = datetime.now()

    cfg_a = tmp_path / "cfg_a"
    cfg_b = tmp_path / "cfg_b"
    cfg_a.mkdir(parents=True, exist_ok=True)
    cfg_b.mkdir(parents=True, exist_ok=True)
    with open(cfg_a / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"strategy": {"signal_scale": 1.0}}, f)
    with open(cfg_b / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"strategy": {"signal_scale": 1.2}}, f)

    with patch.object(DataCollector, "fetch_ohlcv", _fake_fetch_ohlcv):
        summary = main.run_paper_ab_comparison(
            symbols=["SPY", "QQQ", "IWM"],
            start_date=start,
            end_date=end,
            config_a=str(cfg_a),
            config_b=str(cfg_b),
            initial_capital=50_000,
            strategy_type="momentum",
            paper_bars=30,
            output_path=str(tmp_path / "paper_ab_summary.json"),
            verbose=False,
        )

    assert "group_a" in summary and "group_b" in summary
    assert "comparisons" in summary
    assert "promotion_decision" in summary
    assert (tmp_path / "paper_ab_summary.json").exists()
