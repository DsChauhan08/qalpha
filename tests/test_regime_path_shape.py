from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantum_alpha.features.regime_path_shape import RegimePathShapeFeatureGenerator


def _price_frame(n: int = 320) -> pd.DataFrame:
    start = datetime(2020, 1, 1)
    idx = pd.date_range(start, start + timedelta(days=n - 1), freq="D")
    base = np.linspace(100.0, 135.0, n)
    noise = 1.2 * np.sin(np.linspace(0, 24, n))
    close = base + noise
    return pd.DataFrame(
        {
            "open": close + 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, n),
        },
        index=idx,
    )


def test_path_shape_generator_emits_finite_features():
    gen = RegimePathShapeFeatureGenerator()
    df = _price_frame()

    out = gen.generate(df)
    cols = gen.get_feature_names()

    assert set(cols).issubset(out.columns)
    assert np.isfinite(out[cols].to_numpy(dtype=float)).all()


def test_path_shape_generator_is_past_only():
    gen = RegimePathShapeFeatureGenerator()
    df = _price_frame()
    base_out = gen.generate(df)

    future = _price_frame(20)
    future.index = pd.date_range(df.index.max() + timedelta(days=1), periods=20, freq="D")
    extended = pd.concat([df, future], axis=0)
    extended_out = gen.generate(extended)

    cols = gen.get_feature_names()
    assert np.allclose(
        base_out[cols].to_numpy(dtype=float),
        extended_out.loc[base_out.index, cols].to_numpy(dtype=float),
        atol=1e-10,
    )
