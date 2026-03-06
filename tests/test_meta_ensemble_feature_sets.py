from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import quantum_alpha.meta_ensemble as me


def test_get_feature_columns_respects_feature_set(monkeypatch):
    base_cols = ["rsi", "macd"]
    mc_col = "mc_1d_prob_up"
    ps_col = "ps_hurst_21"
    monkeypatch.setattr(me, "BASE_FEATURE_COLS", list(base_cols))
    monkeypatch.setattr(me, "PATH_SHAPE_FEATURES", [ps_col])
    monkeypatch.setattr(me, "ALL_FEATURE_COLS", list(base_cols) + [mc_col, ps_col])
    monkeypatch.setattr(me, "_get_mc_pade_generator", lambda: object())
    monkeypatch.setattr(me, "_get_path_shape_generator", lambda: object())

    df = pd.DataFrame({"rsi": [50.0], "macd": [0.1], mc_col: [0.7], ps_col: [0.2]})
    cols_base = me.get_feature_columns(df, feature_set="base")
    cols_mc = me.get_feature_columns(df, feature_set="mc_pade")
    cols_path = me.get_feature_columns(df, feature_set="path_shape")
    cols_hybrid = me.get_feature_columns(df, feature_set="hybrid_math")

    assert mc_col not in cols_base
    assert mc_col in cols_mc
    assert ps_col not in cols_base
    assert ps_col in cols_path
    assert mc_col in cols_hybrid
    assert ps_col in cols_hybrid


def test_compute_features_drops_unknown_target_tail(monkeypatch):
    # Keep this test fast by stubbing expensive external feature inputs.
    monkeypatch.setattr(me, "_get_mc_pade_generator", lambda: None)
    monkeypatch.setattr(me, "_get_path_shape_generator", lambda: None)
    monkeypatch.setattr(me, "_get_gdelt_data", lambda: pd.DataFrame())
    monkeypatch.setattr(me, "_get_ai_regime_data", lambda: pd.DataFrame())
    monkeypatch.setattr(me, "_get_symbol_fundamentals", lambda symbol: {})

    n = 420
    start = datetime(2020, 1, 1)
    idx = pd.date_range(start, start + timedelta(days=n - 1), freq="D")
    close = np.linspace(100.0, 140.0, n) + 0.5 * np.sin(np.linspace(0, 20, n))
    df = pd.DataFrame(
        {
            "open": close + 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1_000_000, 1_500_000, n),
            "returns": pd.Series(close).pct_change().fillna(0.0).values,
        },
        index=idx,
    )

    out = me.compute_features_single_symbol("TEST", df)
    assert len(out) > 0
    assert out["target"].isna().sum() == 0
    # The final FORWARD_PERIOD bars should not appear because their target is unknown.
    assert pd.Timestamp(out.index.max()) <= pd.Timestamp(df.index.max() - pd.Timedelta(days=me.FORWARD_PERIOD))
