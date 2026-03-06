import json

from quantum_alpha.train_intraday_microstructure import train_intraday_microstructure


def test_intraday_microstructure_pipeline_writes_summary_and_daily_returns(tmp_path):
    output_dir = tmp_path / "intraday_micro"
    checkpoint_dir = tmp_path / "models"
    summary = train_intraday_microstructure(
        replay_dir=None,
        symbols=["SPY", "XLK", "AAPL", "MSFT"],
        market_symbol="SPY",
        sector_symbol="XLK",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        fixture_days=1,
        top_k=2,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "daily_returns.csv").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["strategy"] == "intraday_microstructure"
    assert "annual_return" in payload["metrics"]
    assert summary["artifacts"]["checkpoint"].endswith(".pkl")
