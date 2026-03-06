import json

from quantum_alpha.train_rv_stat_arb import train_rv_stat_arb


def test_rv_stat_arb_pipeline_selects_pairs_and_writes_outputs(tmp_path):
    output_dir = tmp_path / "rv"
    checkpoint_dir = tmp_path / "models"
    summary = train_rv_stat_arb(
        replay_dir=None,
        symbols=["SPY", "XLK", "AAPL", "MSFT"],
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        fixture_days=3,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "daily_returns.csv").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["strategy"] == "rv_stat_arb"
    assert payload["pairs"]
    assert "annual_return" in summary["metrics"]
