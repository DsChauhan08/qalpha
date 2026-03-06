import json
from pathlib import Path

import pandas as pd

from quantum_alpha.visualization.meta_ensemble_video import validate_output_dir


def test_video_validation_requires_summary_json(tmp_path: Path):
    (tmp_path / "normalized_curves.csv").write_text(
        "timestamp,model,spy,equal_weight\n2024-01-01,1,1,1\n",
        encoding="utf-8",
    )
    (tmp_path / "playback_snapshot.png").write_bytes(b"png")
    (tmp_path / "backtest_playback.mp4").write_bytes(b"mp4")

    validation = validate_output_dir(tmp_path)

    assert validation["valid"] is False
    assert "summary_json" in validation["missing"]


def test_video_validation_accepts_complete_artifact_set(tmp_path: Path):
    pd.DataFrame(
        {"timestamp": ["2024-01-01"], "model": [1.0], "spy": [1.0], "equal_weight": [1.0]}
    ).to_csv(tmp_path / "normalized_curves.csv", index=False)
    (tmp_path / "playback_snapshot.png").write_bytes(b"png")
    (tmp_path / "backtest_playback.mp4").write_bytes(b"mp4")
    (tmp_path / "summary.json").write_text(
        json.dumps({"backtest_metrics": {"total_return": 0.1}}),
        encoding="utf-8",
    )

    validation = validate_output_dir(tmp_path)

    assert validation["valid"] is True
    assert validation["curve_rows"] == 1
