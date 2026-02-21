from __future__ import annotations

import numpy as np

from quantum_alpha.llm.distillation import DistillConfig, distill_supervision


def test_distillation_returns_expected_shapes_for_tabular():
    rng = np.random.default_rng(17)
    X = rng.normal(0, 1, size=(80, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=80, dtype=np.int64)
    names = [
        "trend_strength",
        "mean_sentiment",
        "vol_regime",
        "return_zscore",
        "vol_price_div",
        "mom_3m",
    ]

    out = distill_supervision(
        X,
        y_signal=y,
        y_conf=None,
        feature_names=names,
        n_classes=2,
        config=DistillConfig(enabled=True, mode="simulated", max_calls=40, seed=11),
    )

    assert out["y_signal"].shape == y.shape
    assert out["sample_weight"].shape == y.shape
    assert out["teacher_action"].shape == y.shape
    assert out["alignment"].shape == y.shape
    assert out["report"]["enabled"] == 1.0
    assert out["report"]["n_evaluated"] == 40.0
    assert np.isfinite(out["sample_weight"]).all()
    assert (out["sample_weight"] > 0).all()


def test_distillation_adjusts_confidence_for_sequence_inputs():
    rng = np.random.default_rng(23)
    X = rng.normal(0, 1, size=(64, 10, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=64, dtype=np.int64)
    yc = rng.random(64, dtype=np.float32)
    names = [
        "trend_strength",
        "sentiment_momentum",
        "vol_regime",
        "vol_price_div",
        "returns_accel",
    ]

    out = distill_supervision(
        X,
        y_signal=y,
        y_conf=yc,
        feature_names=names,
        n_classes=3,
        config=DistillConfig(enabled=True, mode="simulated", max_calls=64, seed=29),
    )

    assert out["y_conf"] is not None
    assert out["y_conf"].shape == yc.shape
    assert np.isfinite(out["y_conf"]).all()
    assert (out["y_conf"] >= 0.0).all()
    assert (out["y_conf"] <= 1.0).all()
    assert out["report"]["n_evaluated"] == 64.0
