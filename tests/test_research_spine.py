import pandas as pd
from pandas.testing import assert_frame_equal

from quantum_alpha.research_spine import (
    append_research_ledger,
    build_or_load_research_spine,
    load_research_ledger,
)


def test_research_spine_build_is_deterministic_and_cached(tmp_path):
    spine_dir = tmp_path / "spine"
    first = build_or_load_research_spine(
        spine_dir=spine_dir,
        symbols=["SPY", "AAPL", "MSFT"],
        use_fixture=True,
        fixture_days=90,
        seed=11,
    )
    second = build_or_load_research_spine(
        spine_dir=spine_dir,
        symbols=["SPY", "AAPL", "MSFT"],
        use_fixture=True,
        fixture_days=90,
        seed=11,
    )

    cols = [
        "date",
        "symbol",
        "research_peer_group",
        "research_market_return",
        "research_residual_return_1d",
        "research_residual_return_5d",
        "research_residual_return_20d",
        "research_market_beta_63d",
        "research_peer_beta_63d",
        "research_event_quality_score",
        "research_event_quality_flag",
    ]
    assert first.panel_path.exists()
    assert first.metadata_path.exists()
    assert first.dataset_hash == second.dataset_hash
    assert_frame_equal(first.panel[cols], second.panel[cols])
    assert pd.to_numeric(first.panel["research_residual_return_5d"], errors="coerce").isna().sum() == 0


def test_research_spine_ledger_appends_records(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    append_research_ledger(ledger_path, {"strategy": "event_cross_sectional", "model_family": "baseline"})
    append_research_ledger(ledger_path, {"strategy": "event_rv", "model_family": "state_graph"})

    rows = load_research_ledger(ledger_path)

    assert len(rows) == 2
    assert rows[0]["strategy"] == "event_cross_sectional"
    assert rows[1]["model_family"] == "state_graph"
