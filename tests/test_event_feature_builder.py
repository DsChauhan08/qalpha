import pandas as pd
from pandas.testing import assert_frame_equal

from quantum_alpha.data.collectors.event_panel import build_synthetic_event_panel
from quantum_alpha.features.event_feature_builder import UnifiedEventFeatureBuilder


def _build_features(days: int = 80):
    bundle = build_synthetic_event_panel(["SPY", "AAPL", "MSFT"], days=days, seed=7)
    builder = UnifiedEventFeatureBuilder()
    return bundle, builder.build(bundle.panel)


def test_event_feature_builder_emits_prefixed_features_and_is_deterministic():
    bundle, first = _build_features()
    builder = UnifiedEventFeatureBuilder()
    second = builder.build(bundle.panel)

    feature_cols = [c for c in first.features.columns if c.startswith(("ev_", "rv_", "dp_", "ex_"))]
    assert feature_cols
    assert first.features[feature_cols].isna().sum().sum() == 0
    assert_frame_equal(first.features[feature_cols], second.features[feature_cols])


def test_event_feature_builder_is_past_only_before_cutoff():
    bundle, built = _build_features(days=90)
    original = built.features.copy()
    mutated = bundle.panel.copy()
    cutoff = pd.Timestamp("2024-03-15")

    mask = pd.to_datetime(mutated["date"]) >= cutoff
    mutated.loc[mask, "close"] = mutated.loc[mask, "close"] * 1.4
    mutated.loc[mask, "tone"] = mutated.loc[mask, "tone"] * -1.0
    mutated.loc[mask, "options_total_volume_raw"] = mutated.loc[mask, "options_total_volume_raw"] * 2.5
    mutated.loc[mask, "congress_sentiment_raw"] = 1.0

    rebuilt = UnifiedEventFeatureBuilder().build(mutated).features
    feature_cols = [c for c in original.columns if c.startswith(("ev_", "rv_", "dp_", "ex_"))]
    before_cutoff = pd.to_datetime(original["date"]) < cutoff
    assert_frame_equal(
        original.loc[before_cutoff, feature_cols].reset_index(drop=True),
        rebuilt.loc[before_cutoff, feature_cols].reset_index(drop=True),
    )
