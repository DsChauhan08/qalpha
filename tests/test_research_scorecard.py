import argparse

from quantum_alpha.run_research_scorecard import run_research_scorecard


def test_research_scorecard_writes_comparison_manifest(tmp_path):
    args = argparse.Namespace(
        strategies="event_cross_sectional,event_rv,event_stack",
        mode="baseline,state_graph",
        event_symbols="SPY,AAPL,MSFT,NVDA,AMZN,XOM",
        event_universe_size=50,
        event_use_fixture=True,
        fixture_days=100,
        seed=9,
        quick=True,
        output_dir=str(tmp_path),
    )

    manifest = run_research_scorecard(args)

    assert (tmp_path / "comparison_manifest.json").exists()
    assert manifest["research_spine"]["dataset_hash"]
    assert manifest["ledger_records"]
    combos = {(row["strategy"], row["mode"]) for row in manifest["records"]}
    assert ("event_cross_sectional", "baseline") in combos
    assert ("event_cross_sectional", "state_graph") in combos
    assert ("event_rv", "baseline") in combos
    assert ("event_rv", "state_graph") in combos
    assert ("event_stack", "baseline") in combos
    assert ("event_stack", "state_graph") in combos
