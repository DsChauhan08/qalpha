import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from quantum_alpha.strategy.meta_dual_blend_strategy import MetaDualBlendStrategy


class _ConstModel:
    def __init__(self, up_prob: float):
        self.up_prob = float(up_prob)

    def predict_proba(self, X):
        n = len(X)
        up = np.full(n, self.up_prob, dtype=float)
        down = 1.0 - up
        return np.vstack([down, up]).T


def _write_ckpt(
    ckpt_dir: Path,
    name: str,
    up_prob: float,
    feature_cols: list[str],
    feature_set: str = "base",
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _ConstModel(up_prob=up_prob),
        "scaler": None,
        "feature_cols": feature_cols,
        "feature_set": feature_set,
    }
    with open(ckpt_dir / name, "wb") as f:
        pickle.dump(payload, f)


def test_dual_blend_uses_weighted_probability(tmp_path):
    base_dir = tmp_path / "base"
    mc_dir = tmp_path / "mc"
    _write_ckpt(base_dir, "meta_ensemble_model.pkl", 0.60, ["f1", "f2"])
    _write_ckpt(
        mc_dir,
        "meta_ensemble_mc_pade_model.pkl",
        0.80,
        ["f1", "f2", "mc_5d_prob_up", "pade_prob_up_2pct"],
        feature_set="mc_pade",
    )

    strategy = MetaDualBlendStrategy(
        base_checkpoint_dir=base_dir,
        mc_checkpoint_dir=mc_dir,
        blend_weights=(0.35, 0.65),
    )
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0], "f3": [3.0]})
    pred = strategy.predict_from_features(df)

    assert pred is not None
    expected = 0.35 * 0.60 + 0.65 * 0.80
    assert abs(pred["up_probability_blend"] - expected) < 1e-9
    assert pred["model_used"] == "blended"
    health = strategy.model_health()
    assert health["mc_ok"] is True
    assert int(health["mc_mc_feature_count"]) == 1
    assert int(health["mc_pade_feature_count"]) == 1


def test_dual_blend_falls_back_to_base_only(tmp_path):
    base_dir = tmp_path / "base"
    mc_dir = tmp_path / "mc"
    _write_ckpt(base_dir, "meta_ensemble_model.pkl", 0.55, ["f1"])
    mc_dir.mkdir(parents=True, exist_ok=True)

    strategy = MetaDualBlendStrategy(
        base_checkpoint_dir=base_dir,
        mc_checkpoint_dir=mc_dir,
    )
    df = pd.DataFrame({"f1": [1.0]})
    pred = strategy.predict_from_features(df)

    assert pred is not None
    assert pred["model_used"] == "base_only"
    assert abs(pred["up_probability_blend"] - 0.55) < 1e-9


def test_dual_blend_rejects_fake_mc_checkpoint_without_mc_features(tmp_path):
    base_dir = tmp_path / "base"
    mc_dir = tmp_path / "mc"
    _write_ckpt(base_dir, "meta_ensemble_model.pkl", 0.60, ["f1", "f2"])
    _write_ckpt(
        mc_dir,
        "meta_ensemble_mc_pade_model.pkl",
        0.80,
        ["f1", "f2", "f3"],
        feature_set="base",
    )

    strategy = MetaDualBlendStrategy(
        base_checkpoint_dir=base_dir,
        mc_checkpoint_dir=mc_dir,
    )
    pred = strategy.predict_from_features(pd.DataFrame({"f1": [1.0], "f2": [2.0]}))

    assert pred is not None
    assert pred["model_used"] == "base_only"
    health = strategy.model_health()
    assert health["base_ok"] is True
    assert health["mc_ok"] is False
    assert "feature_set='base'" in str(health["mc_error"])
