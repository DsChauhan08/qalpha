import pandas as pd
from pandas.testing import assert_frame_equal

from quantum_alpha.data.collectors.intraday_replay import (
    IntradayReplayStore,
    build_synthetic_intraday_replay,
)
from quantum_alpha.features.intraday_feature_builder import UnifiedIntradayFeatureBuilder


def _load_bundle(tmp_path):
    replay_root = tmp_path / "replay"
    build_synthetic_intraday_replay(replay_root, date="2025-01-06", symbols=["AAPL"])
    store = IntradayReplayStore(replay_root)
    bundle = store.load_symbol_bundle("2025-01-06", "AAPL", domains=("trades", "quotes", "depth", "bars_1m"))
    return bundle


def test_intraday_feature_builder_emits_prefixed_features_and_is_deterministic(tmp_path):
    bundle = _load_bundle(tmp_path)
    builder = UnifiedIntradayFeatureBuilder()
    first = builder.build(symbol="AAPL", trades=bundle["trades"], quotes=bundle["quotes"], depth=bundle["depth"], bars_1m=bundle["bars_1m"])
    second = builder.build(symbol="AAPL", trades=bundle["trades"], quotes=bundle["quotes"], depth=bundle["depth"], bars_1m=bundle["bars_1m"])

    feature_cols = [c for c in first.features.columns if c.startswith(("ms_", "tp_", "rp_", "ps_"))]
    assert feature_cols
    assert first.features[feature_cols].isna().sum().sum() == 0
    assert_frame_equal(first.features[feature_cols], second.features[feature_cols])


def test_intraday_feature_builder_is_past_only_before_cutoff(tmp_path):
    bundle = _load_bundle(tmp_path)
    builder = UnifiedIntradayFeatureBuilder()
    original = builder.build(symbol="AAPL", trades=bundle["trades"], quotes=bundle["quotes"], depth=bundle["depth"], bars_1m=bundle["bars_1m"]).features

    cutoff = pd.Timestamp("2025-01-06 16:30:00+00:00")
    mutated = {k: v.copy() for k, v in bundle.items()}
    for name in ("bars_1m", "quotes", "trades", "depth"):
        df = mutated[name]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        mask = df["timestamp"] >= cutoff
        if name == "bars_1m":
            df.loc[mask, "close"] = df.loc[mask, "close"] * 1.5
        elif name == "quotes":
            df.loc[mask, "bid"] = df.loc[mask, "bid"] * 0.8
            df.loc[mask, "ask"] = df.loc[mask, "ask"] * 1.2
        elif name == "trades":
            df.loc[mask, "price"] = df.loc[mask, "price"] * 1.4
        elif name == "depth":
            df.loc[mask, "bid_size"] = df.loc[mask, "bid_size"] * 3
            df.loc[mask, "ask_size"] = df.loc[mask, "ask_size"] * 3
        mutated[name] = df

    rebuilt = builder.build(symbol="AAPL", trades=mutated["trades"], quotes=mutated["quotes"], depth=mutated["depth"], bars_1m=mutated["bars_1m"]).features
    feature_cols = [c for c in original.columns if c.startswith(("ms_", "tp_", "rp_", "ps_"))]
    before_cutoff = original.index < cutoff
    assert_frame_equal(original.loc[before_cutoff, feature_cols], rebuilt.loc[before_cutoff, feature_cols])
