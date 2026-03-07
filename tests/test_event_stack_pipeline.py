import numpy as np
import pandas as pd

from quantum_alpha.backtesting.event_sleeve_tools import validate_viewer_bundle
from quantum_alpha.train_event_stack import train_event_stack


def test_event_stack_pipeline_writes_viewer_artifacts(tmp_path):
    idx = pd.date_range("2025-01-01", periods=90, freq="B")
    cross = pd.DataFrame(
        {
            "date": idx.astype(str),
            "strategy_return": 0.0008 + 0.0002 * np.sin(np.arange(len(idx)) / 5.0),
            "benchmark_return": 0.0003 + 0.0001 * np.sin(np.arange(len(idx)) / 8.0),
            "equal_weight_return": 0.00025 + 0.0001 * np.cos(np.arange(len(idx)) / 9.0),
        }
    )
    rv = pd.DataFrame(
        {
            "date": idx.astype(str),
            "strategy_return": 0.0004 + 0.00015 * np.cos(np.arange(len(idx)) / 6.0),
        }
    )
    meta = pd.DataFrame(
        {
            "date": idx.astype(str),
            "model": 0.0005 + 0.0001 * np.sin(np.arange(len(idx)) / 7.0),
            "spy": 0.0003 + 0.0001 * np.cos(np.arange(len(idx)) / 10.0),
            "equal_weight": 0.0002 + 0.00005 * np.sin(np.arange(len(idx)) / 11.0),
        }
    )

    cross_path = tmp_path / "cross.csv"
    rv_path = tmp_path / "rv.csv"
    meta_path = tmp_path / "meta.csv"
    cross.to_csv(cross_path, index=False)
    rv.to_csv(rv_path, index=False)
    meta.to_csv(meta_path, index=False)

    out_dir = tmp_path / "event_stack"
    summary = train_event_stack(
        event_cross_daily_returns=cross_path,
        event_rv_daily_returns=rv_path,
        meta_daily_returns=meta_path,
        output_dir=out_dir,
    )

    assert summary["strategy"] == "event_stack"
    assert "annual_return" in summary["metrics"]
    assert validate_viewer_bundle(out_dir / "viewer")["valid"] is True
