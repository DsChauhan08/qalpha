import json

from quantum_alpha.backtesting.event_sleeve_tools import validate_viewer_bundle
from quantum_alpha.train_event_rv import train_event_rv


def test_event_rv_pipeline_writes_outputs(tmp_path):
    output_dir = tmp_path / "event_rv"
    checkpoint_dir = tmp_path / "models"
    summary = train_event_rv(
        symbols=["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "XOM"],
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        use_fixture=True,
        fixture_days=160,
        quick=True,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "daily_returns.csv").exists()
    assert (output_dir / "pairs.json").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["strategy"] == "event_rv"
    assert payload["pairs"]
    assert "annual_return" in summary["metrics"]
    assert validate_viewer_bundle(output_dir / "viewer")["valid"] is True
