from pathlib import Path

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.enhanced_factor_diagnostics import (
    run_enhanced_factor_diagnostics,
)


def _mock_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=160, freq="D")
    close = 100 + np.cumsum(rng.normal(0.0, 1.0, size=len(idx)))
    df = pd.DataFrame({"close": close}, index=idx)
    # Deterministic component columns expected by diagnostics.
    df["component_momentum"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_mean_rev"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_trend"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_breakout"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_ts_mom"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_xs_momentum"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_stat_arb"] = rng.normal(0.0, 0.5, size=len(idx))
    df["component_regime_mom"] = rng.normal(0.0, 0.5, size=len(idx))
    return df


def test_run_enhanced_factor_diagnostics_writes_artifacts(tmp_path: Path):
    frames = {
        "AAA": _mock_frame(1),
        "BBB": _mock_frame(2),
        "CCC": _mock_frame(3),
    }
    out = run_enhanced_factor_diagnostics(
        frames=frames,
        output_dir=tmp_path / "diag",
        forward_periods=5,
    )

    assert out["n_rows_panel"] > 0
    full_csv = Path(out["factor_quality_full_csv"])
    seg_csv = Path(out["factor_quality_by_segment_csv"])
    ablation_json = Path(out["factor_ablation_summary_json"])
    assert full_csv.exists()
    assert seg_csv.exists()
    assert ablation_json.exists()

    full_df = pd.read_csv(full_csv)
    assert {"factor", "rolling_ic_mean", "hit_rate"}.issubset(full_df.columns)
    assert len(full_df) >= 8
