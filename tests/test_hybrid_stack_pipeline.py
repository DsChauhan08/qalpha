import numpy as np
import pandas as pd

from quantum_alpha.train_hybrid_stack import train_hybrid_stack
from quantum_alpha.visualization.meta_ensemble_video import validate_output_dir


def test_hybrid_stack_pipeline_writes_viewer_artifacts(tmp_path):
    idx = pd.date_range("2025-01-01", periods=90, freq="B")
    intraday = pd.DataFrame({"date": idx.astype(str), "strategy_return": 0.0008 + 0.0002 * np.sin(np.arange(len(idx)) / 5)})
    rv = pd.DataFrame({"date": idx.astype(str), "strategy_return": 0.0006 + 0.00015 * np.cos(np.arange(len(idx)) / 7)})
    meta = pd.DataFrame(
        {
            "date": idx.astype(str),
            "model": 0.0009 + 0.00025 * np.sin(np.arange(len(idx)) / 9),
            "spy": 0.0003 + 0.0001 * np.sin(np.arange(len(idx)) / 11),
            "equal_weight": 0.0002 + 0.00005 * np.cos(np.arange(len(idx)) / 13),
        }
    )

    intraday_path = tmp_path / "intraday.csv"
    rv_path = tmp_path / "rv.csv"
    meta_path = tmp_path / "meta.csv"
    intraday.to_csv(intraday_path, index=False)
    rv.to_csv(rv_path, index=False)
    meta.to_csv(meta_path, index=False)

    out_dir = tmp_path / "hybrid"
    summary = train_hybrid_stack(
        intraday_daily_returns=intraday_path,
        rv_daily_returns=rv_path,
        meta_daily_returns=meta_path,
        benchmark_daily_returns=meta_path,
        output_dir=out_dir,
    )

    validation = validate_output_dir(out_dir)
    assert validation["valid"] is True
    assert summary["strategy"] == "hybrid_stack"
    assert "annual_excess_vs_spy" in summary["metrics"]
