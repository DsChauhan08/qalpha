import json

from quantum_alpha.train_event_cross_sectional import train_event_cross_sectional
from quantum_alpha.backtesting.event_sleeve_tools import validate_viewer_bundle


def test_event_cross_sectional_pipeline_writes_outputs(tmp_path):
    output_dir = tmp_path / "event_cross"
    checkpoint_dir = tmp_path / "models"
    summary = train_event_cross_sectional(
        symbols=["SPY", "AAPL", "MSFT", "NVDA", "AMZN"],
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        use_fixture=True,
        fixture_days=120,
        quick=True,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "daily_returns.csv").exists()
    assert (output_dir / "robustness.json").exists()
    assert (output_dir / "synthetic_stress.json").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["strategy"] == "event_cross_sectional"
    assert "annual_return" in payload["metrics"]
    assert summary["artifacts"]["checkpoint"].endswith(".pkl")
    assert validate_viewer_bundle(output_dir / "viewer")["valid"] is True
