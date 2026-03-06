"""
Dual-model meta-ensemble blend strategy.

Blends probabilities from:
1) base meta-ensemble checkpoint
2) MC/Padé-enhanced meta-ensemble checkpoint
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "meta_ensemble"
DEFAULT_MC_CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "meta_ensemble"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _feature_family_counts(feature_cols: list[str]) -> dict[str, int]:
    mc_count = sum(1 for c in feature_cols if str(c).startswith(("mc_", "jd_", "rs_")))
    pade_count = sum(1 for c in feature_cols if str(c).startswith("pade_"))
    path_shape_count = sum(1 for c in feature_cols if str(c).startswith("ps_"))
    return {
        "feature_count": int(len(feature_cols)),
        "mc_feature_count": int(mc_count),
        "pade_feature_count": int(pade_count),
        "path_shape_feature_count": int(path_shape_count),
    }


def _infer_feature_set(declared_feature_set: object, feature_cols: list[str]) -> str:
    value = str(declared_feature_set or "").strip().lower()
    if value in {"base", "mc_pade", "path_shape", "hybrid_math"}:
        return value
    stats = _feature_family_counts(feature_cols)
    if stats["path_shape_feature_count"] > 0 and (
        stats["mc_feature_count"] > 0 or stats["pade_feature_count"] > 0
    ):
        return "hybrid_math"
    if stats["path_shape_feature_count"] > 0:
        return "path_shape"
    if stats["mc_feature_count"] > 0 or stats["pade_feature_count"] > 0:
        return "mc_pade"
    return "base"


class MetaDualBlendStrategy:
    """
    Load base + MC/Padé meta checkpoints and emit blended probabilities.
    """

    def __init__(
        self,
        base_checkpoint_dir: str | Path | None = None,
        mc_checkpoint_dir: str | Path | None = None,
        blend_weights: tuple[float, float] = (0.35, 0.65),
    ) -> None:
        self.base_checkpoint_dir = (
            Path(base_checkpoint_dir) if base_checkpoint_dir else DEFAULT_BASE_CKPT_DIR
        )
        self.mc_checkpoint_dir = (
            Path(mc_checkpoint_dir) if mc_checkpoint_dir else DEFAULT_MC_CKPT_DIR
        )

        b0, b1 = _safe_float(blend_weights[0], 0.35), _safe_float(blend_weights[1], 0.65)
        wsum = b0 + b1
        if wsum <= 0:
            b0, b1, wsum = 0.35, 0.65, 1.0
        self.blend_weights = (b0 / wsum, b1 / wsum)

        self._base_bundle: Optional[Dict[str, object]] = None
        self._mc_bundle: Optional[Dict[str, object]] = None
        self._base_error: Optional[str] = None
        self._mc_error: Optional[str] = None

        self._load_models()

    def _find_bundle_path(
        self,
        checkpoint_dir: Path,
        prefer: list[str],
    ) -> Path | None:
        for name in prefer:
            p = checkpoint_dir / name
            if p.exists():
                return p
        return None

    def _load_bundle(self, model_path: Path) -> Dict[str, object]:
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Invalid checkpoint payload in {model_path}")
        model = checkpoint.get("model")
        if model is None:
            raise RuntimeError(f"Missing model in {model_path}")
        feature_cols = checkpoint.get("feature_cols")
        if not isinstance(feature_cols, list) or len(feature_cols) == 0:
            raise RuntimeError(f"Missing feature_cols in {model_path}")
        declared_feature_set = checkpoint.get("feature_set")
        feature_stats = _feature_family_counts(list(feature_cols))
        return {
            "model_path": str(model_path),
            "model": model,
            "scaler": checkpoint.get("scaler"),
            "feature_cols": list(feature_cols),
            "feature_set": _infer_feature_set(declared_feature_set, list(feature_cols)),
            "declared_feature_set": (
                str(declared_feature_set).strip().lower() if declared_feature_set else None
            ),
            "feature_stats": feature_stats,
            "metadata": checkpoint,
        }

    def _validate_bundle(self, bundle: Dict[str, object], role: str) -> Dict[str, object]:
        if role != "mc":
            return bundle

        stats = bundle.get("feature_stats", {})
        mc_count = int(stats.get("mc_feature_count", 0))
        pade_count = int(stats.get("pade_feature_count", 0))
        declared_feature_set = bundle.get("declared_feature_set")

        if declared_feature_set and declared_feature_set != "mc_pade":
            raise RuntimeError(
                f"MC checkpoint declares feature_set={declared_feature_set!r}: "
                f"{bundle.get('model_path')}"
            )
        if mc_count <= 0 and pade_count <= 0:
            raise RuntimeError(
                "MC checkpoint contains no MC/Padé features: "
                f"{bundle.get('model_path')}"
            )
        return bundle

    def _load_models(self) -> None:
        # Base model
        base_path = self._find_bundle_path(
            self.base_checkpoint_dir,
            [
                "meta_ensemble_model.pkl",
                "meta_ensemble_best_wf.pkl",
            ],
        )
        if base_path is None:
            self._base_error = f"No base checkpoint found in {self.base_checkpoint_dir}"
        else:
            try:
                self._base_bundle = self._load_bundle(base_path)
            except Exception as exc:
                self._base_error = str(exc)

        # MC/Padé model
        mc_path = self._find_bundle_path(
            self.mc_checkpoint_dir,
            [
                "meta_ensemble_mc_pade_model.pkl",
                "meta_ensemble_mc_pade_best_wf.pkl",
            ],
        )
        if mc_path is None:
            self._mc_error = f"No MC checkpoint found in {self.mc_checkpoint_dir}"
        else:
            try:
                self._mc_bundle = self._validate_bundle(self._load_bundle(mc_path), role="mc")
            except Exception as exc:
                self._mc_error = str(exc)

    def model_health(self) -> Dict[str, object]:
        base_stats = (
            dict(self._base_bundle.get("feature_stats", {}))
            if isinstance(self._base_bundle, dict)
            else {}
        )
        mc_stats = (
            dict(self._mc_bundle.get("feature_stats", {}))
            if isinstance(self._mc_bundle, dict)
            else {}
        )
        return {
            "base_ok": self._base_bundle is not None,
            "mc_ok": self._mc_bundle is not None,
            "base_error": self._base_error,
            "mc_error": self._mc_error,
            "base_model_path": self._base_bundle.get("model_path") if self._base_bundle else None,
            "mc_model_path": self._mc_bundle.get("model_path") if self._mc_bundle else None,
            "base_feature_set": self._base_bundle.get("feature_set") if self._base_bundle else None,
            "mc_feature_set": self._mc_bundle.get("feature_set") if self._mc_bundle else None,
            "base_declared_feature_set": (
                self._base_bundle.get("declared_feature_set") if self._base_bundle else None
            ),
            "mc_declared_feature_set": (
                self._mc_bundle.get("declared_feature_set") if self._mc_bundle else None
            ),
            "base_feature_count": int(base_stats.get("feature_count", 0)),
            "base_mc_feature_count": int(base_stats.get("mc_feature_count", 0)),
            "base_pade_feature_count": int(base_stats.get("pade_feature_count", 0)),
            "mc_feature_count": int(mc_stats.get("feature_count", 0)),
            "mc_mc_feature_count": int(mc_stats.get("mc_feature_count", 0)),
            "mc_pade_feature_count": int(mc_stats.get("pade_feature_count", 0)),
            "checkpoints_distinct": (
                None
                if not self._base_bundle or not self._mc_bundle
                else bool(
                    str(self._base_bundle.get("model_path"))
                    != str(self._mc_bundle.get("model_path"))
                )
            ),
            "blend_weights": [float(self.blend_weights[0]), float(self.blend_weights[1])],
        }

    def _predict_with_bundle(
        self,
        last_row: pd.DataFrame,
        bundle: Dict[str, object],
    ) -> tuple[Optional[float], float]:
        feature_cols = list(bundle.get("feature_cols", []))
        if len(feature_cols) == 0:
            return None, 1.0

        x = pd.DataFrame(index=last_row.index)
        missing = 0
        for col in feature_cols:
            if col in last_row.columns:
                x[col] = last_row[col].values
            else:
                x[col] = 0.0
                missing += 1

        x_vals = x.values.astype(np.float64)
        x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=3.0, neginf=-3.0)
        scaler = bundle.get("scaler")
        if scaler is not None:
            try:
                x_vals = scaler.transform(x_vals)
                x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=3.0, neginf=-3.0)
            except Exception:
                pass

        model = bundle.get("model")
        if model is None:
            return None, 1.0
        try:
            proba = model.predict_proba(x_vals)
            up_prob = float(proba[0, 1])
        except Exception:
            return None, float(missing / max(len(feature_cols), 1))

        miss_ratio = float(missing / max(len(feature_cols), 1))
        return up_prob, miss_ratio

    def predict_from_features(self, features_df: pd.DataFrame) -> Optional[Dict[str, object]]:
        if features_df is None or len(features_df) == 0:
            return None
        last_row = features_df.iloc[[-1]]

        p_base = None
        p_mc = None
        miss_base = 1.0
        miss_mc = 1.0

        if self._base_bundle is not None:
            p_base, miss_base = self._predict_with_bundle(last_row, self._base_bundle)
        if self._mc_bundle is not None:
            p_mc, miss_mc = self._predict_with_bundle(last_row, self._mc_bundle)

        if p_base is None and p_mc is None:
            return None

        if p_base is not None and p_mc is not None:
            p_blend = float(self.blend_weights[0] * p_base + self.blend_weights[1] * p_mc)
            model_used = "blended"
        elif p_base is not None:
            p_blend = float(p_base)
            model_used = "base_only"
        else:
            p_blend = float(p_mc)  # type: ignore[arg-type]
            model_used = "mc_only"

        confidence = abs(p_blend - 0.5) * 2.0
        signal = (p_blend - 0.5) * 2.0
        return {
            "up_probability_base": None if p_base is None else float(p_base),
            "up_probability_mc": None if p_mc is None else float(p_mc),
            "up_probability_blend": float(p_blend),
            "up_probability": float(p_blend),
            "confidence": float(confidence),
            "signal_score": float(signal),
            "signal": "LONG" if p_blend >= 0.5 else "SHORT",
            "missing_feature_ratio_base": float(miss_base),
            "missing_feature_ratio_mc": float(miss_mc),
            "model_used": model_used,
        }

    def predict_symbol(self, df: pd.DataFrame, symbol: str) -> Optional[Dict[str, object]]:
        try:
            from quantum_alpha.meta_ensemble import compute_features_single_symbol

            featured = compute_features_single_symbol(symbol, df)
            return self.predict_from_features(featured)
        except Exception as exc:
            logger.debug("Dual blend predict failed for %s: %s", symbol, exc)
            return None

    def build_anchor_predictions(self, featured: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for symbol, df in sorted(featured.items()):
            pred = self.predict_symbol(df, symbol)
            if pred is None:
                continue
            out[str(symbol)] = pred
        return out
