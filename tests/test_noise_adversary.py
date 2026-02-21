from __future__ import annotations

import numpy as np

from quantum_alpha.models.lstm_v4.noise_adversary import (
    augment_with_noise_traps,
    evaluate_noise_probe,
)


def test_noise_augmentation_adds_hold_samples_for_ternary():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(120, 30, 21)).astype(np.float32)
    y = rng.integers(0, 3, size=120, dtype=np.int64)
    yc = rng.random(120, dtype=np.float32)

    Xa, ya, yca, report = augment_with_noise_traps(
        X_train=X,
        y_signal_train=y,
        y_conf_train=yc,
        n_sentiment_features=16,
        n_classes=3,
        ratio=0.25,
        seed=11,
    )

    assert Xa.shape[0] == 150
    assert ya.shape[0] == 150
    assert yca.shape[0] == 150
    assert report["enabled"] == 1.0
    assert report["added_samples"] == 30.0
    assert (ya == 1).sum() >= (y == 1).sum()


def test_noise_augmentation_skips_for_binary():
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, size=(80, 30, 21)).astype(np.float32)
    y = rng.integers(0, 2, size=80, dtype=np.int64)
    yc = rng.random(80, dtype=np.float32)

    Xa, ya, yca, report = augment_with_noise_traps(
        X_train=X,
        y_signal_train=y,
        y_conf_train=yc,
        n_sentiment_features=16,
        n_classes=2,
        ratio=0.5,
        seed=13,
    )

    assert Xa.shape == X.shape
    assert np.array_equal(ya, y)
    assert np.array_equal(yca, yc)
    assert report["enabled"] == 0.0
    assert report["skipped"] == 1.0


class _DummyModel:
    def predict(self, X):
        return {"trade_action": np.zeros(len(X), dtype=int)}


def test_noise_probe_with_dummy_model_is_hold_only():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(90, 30, 21)).astype(np.float32)

    metrics = evaluate_noise_probe(
        model=_DummyModel(),
        X_reference=X,
        n_sentiment_features=16,
        n_samples=64,
        seed=19,
    )

    assert metrics["probe_samples"] == 64.0
    assert metrics["distraction_rate"] == 0.0
    assert metrics["hold_rate"] == 1.0
