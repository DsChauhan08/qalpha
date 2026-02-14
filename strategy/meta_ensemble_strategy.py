"""
Meta-Ensemble Trading Strategy
================================
Wraps the trained HistGradientBoosting meta-ensemble model for use
in the backtest framework.

Interface matches NewsLSTMStrategy:
  generate_signals(df, symbol) -> df with signal/signal_confidence/position_signal
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).parent.parent / "models" / "checkpoints" / "meta_ensemble"
)


class MetaEnsembleStrategy:
    """
    Trading strategy powered by the Meta-Ensemble (HistGradientBoosting).

    Loads a trained model from checkpoint, computes features on the fly,
    and generates trading signals.

    generate_signals(df, symbol) contract:
      Input:  DataFrame with OHLCV columns (open, high, low, close, volume)
      Output: Same DataFrame with added columns:
              - signal (float [-1,1])
              - signal_confidence (float [0,1])
              - position_signal (float [-1,1], can be fractional if proportional sizing)

    Confidence-based position sizing:
      The walk-forward analysis shows high-confidence predictions are significantly
      more accurate. This strategy supports:
      - confidence_threshold: minimum confidence to trade (default 0.10)
      - proportional_sizing: if True, position_signal scales with confidence
      - signal_threshold: minimum up/down probability to trigger signal (default 0.52)
    """

    def __init__(
        self,
        checkpoint_dir: str = None,
        signal_threshold: float = 0.52,
        confidence_threshold: float = 0.10,
        proportional_sizing: bool = True,
    ):
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
        )
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        self.proportional_sizing = proportional_sizing

        self._model = None
        self._scaler = None
        self._feature_cols = None
        self._loaded = False
        self._mc_pade_gen = None

    def _load_model(self):
        """Load the trained meta-ensemble model from checkpoint."""
        if self._loaded:
            return

        model_path = self.checkpoint_dir / "meta_ensemble_model.pkl"
        if not model_path.exists():
            # Try best walk-forward model
            model_path = self.checkpoint_dir / "meta_ensemble_best_wf.pkl"

        if not model_path.exists():
            logger.warning(f"No meta-ensemble model found in {self.checkpoint_dir}")
            self._loaded = True
            return

        try:
            with open(model_path, "rb") as f:
                checkpoint = pickle.load(f)

            if isinstance(checkpoint, dict):
                self._model = checkpoint.get("model")
                self._scaler = checkpoint.get("scaler")
                self._feature_cols = checkpoint.get("feature_cols")
            else:
                # Assume it's just the model
                self._model = checkpoint

            if self._model is not None:
                logger.info(
                    f"Loaded meta-ensemble model from {model_path.name} "
                    f"({len(self._feature_cols) if self._feature_cols else '?'} features)"
                )
            else:
                logger.warning("Model checkpoint loaded but model is None")

        except Exception as e:
            logger.error(f"Failed to load meta-ensemble model: {e}")

        self._loaded = True

    def _get_mc_pade_gen(self):
        """Lazy-load MC/Pade feature generator."""
        if self._mc_pade_gen is None:
            try:
                from quantum_alpha.features.mc_pade_features import (
                    MCPadeFeatureGenerator,
                )

                self._mc_pade_gen = MCPadeFeatureGenerator()
            except Exception as e:
                logger.warning(f"MC/Padé features unavailable: {e}")
                self._mc_pade_gen = False
        return self._mc_pade_gen if self._mc_pade_gen is not False else None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features needed by the meta-ensemble model.
        This mirrors compute_features_single_symbol() from meta_ensemble.py
        but without the target column.
        """
        from quantum_alpha.meta_ensemble import (
            CROSS_SECTIONAL_FEATURES,
            compute_technical_features,
            compute_strategy_signals,
            compute_regime_features,
            compute_price_derived_features,
        )

        featured = compute_technical_features(df)
        featured = compute_strategy_signals(featured)
        featured = compute_regime_features(featured)
        featured = compute_price_derived_features(featured)

        # Cross-sectional features: these require multi-symbol data that isn't
        # available at single-symbol inference time. Fill with neutral defaults:
        # ranks default to 0.5 (median), relative strength defaults to 0.0.
        for col in CROSS_SECTIONAL_FEATURES:
            if col not in featured.columns:
                if "rank" in col:
                    featured[col] = 0.5  # Neutral rank
                elif "relative_strength" in col:
                    featured[col] = 0.0  # Average relative strength
                else:
                    featured[col] = 0.5

        # MC/Padé features
        mc_gen = self._get_mc_pade_gen()
        if mc_gen is not None:
            try:
                featured = mc_gen.generate_features_fast(featured)
            except Exception as e:
                logger.debug(f"MC/Padé features failed: {e}")

        # Clean
        featured = featured.replace([np.inf, -np.inf], np.nan)

        return featured

    def generate_signals(self, df: pd.DataFrame, symbol: str = "SPY") -> pd.DataFrame:
        """
        Generate trading signals from the meta-ensemble model.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being traded

        Returns:
            DataFrame with signal, signal_confidence, position_signal columns added
        """
        df = df.copy()
        df["signal"] = 0.0
        df["signal_confidence"] = 0.0
        df["position_signal"] = 0.0

        self._load_model()

        if self._model is None:
            logger.warning("No meta-ensemble model loaded — returning hold signals")
            return df

        try:
            return self._generate_signals_impl(df, symbol)
        except Exception as e:
            logger.error(f"Meta-ensemble signal generation failed for {symbol}: {e}")
            import traceback

            traceback.print_exc()
            return df

    def _generate_signals_impl(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Core signal generation logic."""
        if len(df) < 300:
            logger.warning(
                f"Insufficient data for {symbol}: {len(df)} rows (need 300+)"
            )
            return df

        # Compute features
        featured = self._compute_features(df)

        # Determine which feature columns to use.
        # CRITICAL: The model expects EXACTLY the same features in the same order
        # as training. We must provide ALL features, filling missing ones with
        # sensible defaults rather than silently dropping them.
        if self._feature_cols is not None:
            expected_cols = self._feature_cols
        else:
            from quantum_alpha.meta_ensemble import ALL_FEATURE_COLS

            expected_cols = list(ALL_FEATURE_COLS)

        if len(expected_cols) == 0:
            logger.warning(f"No feature columns available for {symbol}")
            return df

        # Ensure all expected features exist; fill missing with 0.0
        missing_count = 0
        for col in expected_cols:
            if col not in featured.columns:
                featured[col] = 0.0
                missing_count += 1
                logger.debug(f"Filled missing feature '{col}' with 0.0")

        feature_cols = expected_cols
        if missing_count > 0:
            logger.info(
                f"[{symbol}] {missing_count}/{len(feature_cols)} features filled "
                f"with defaults (cross-sectional or MC/Padé)"
            )

        # Prepare feature matrix
        X = featured[feature_cols].fillna(0.0)

        # Scale if scaler available
        if self._scaler is not None:
            try:
                X_scaled = pd.DataFrame(
                    self._scaler.transform(X),
                    index=X.index,
                    columns=feature_cols,
                )
            except Exception:
                X_scaled = X
        else:
            X_scaled = X

        # Skip warmup period (first 252 rows typically have poor features)
        valid_mask = featured.index.isin(df.index)
        warmup = 252
        if len(X_scaled) > warmup:
            X_pred = X_scaled.iloc[warmup:]
        else:
            X_pred = X_scaled

        # Predict probabilities
        try:
            proba = self._model.predict_proba(X_pred)
            # proba[:, 1] = probability of class 1 (up)
            up_prob = proba[:, 1]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return df

        # Convert probabilities to signals
        # signal = (up_prob - 0.5) * 2  =>  maps [0,1] -> [-1,1]
        signals = (up_prob - 0.5) * 2.0
        # confidence = distance from 0.5 => how certain the model is
        confidence = np.abs(up_prob - 0.5) * 2.0

        # Map back to original DataFrame index
        pred_index = X_pred.index
        common_idx = df.index.intersection(pred_index)

        if len(common_idx) == 0:
            logger.warning(f"No overlapping indices for {symbol}")
            return df

        df.loc[common_idx, "signal"] = signals[: len(common_idx)]
        df.loc[common_idx, "signal_confidence"] = confidence[: len(common_idx)]

        # Position signal — vectorized, with confidence-based sizing
        df_common = df.loc[common_idx]
        sig_vals = df_common["signal"].values
        conf_vals = df_common["signal_confidence"].values
        prob_vals = (sig_vals + 1.0) / 2.0  # back to probability

        pos_signals = np.zeros(len(common_idx))
        conf_mask = conf_vals >= self.confidence_threshold
        buy_mask = conf_mask & (prob_vals >= self.signal_threshold)
        sell_mask = conf_mask & (prob_vals <= (1.0 - self.signal_threshold))

        if self.proportional_sizing:
            # Scale position size by confidence: higher confidence = larger position
            # confidence ranges from 0 to 1; we normalize so that max observed
            # confidence maps to full position (1.0), and threshold maps to a
            # minimum position size (0.25).
            min_size = 0.25
            max_conf = max(conf_vals.max(), self.confidence_threshold + 0.01)
            # Linear scale from [threshold, max_conf] -> [min_size, 1.0]
            scale = np.clip(
                min_size
                + (1.0 - min_size)
                * (conf_vals - self.confidence_threshold)
                / (max_conf - self.confidence_threshold),
                min_size,
                1.0,
            )
            pos_signals[buy_mask] = scale[buy_mask]
            pos_signals[sell_mask] = -scale[sell_mask]
        else:
            # Binary: full position or nothing
            pos_signals[buy_mask] = 1.0
            pos_signals[sell_mask] = -1.0

        df.loc[common_idx, "position_signal"] = pos_signals

        # Log stats
        n_buy = (df["position_signal"] > 0).sum()
        n_sell = (df["position_signal"] < 0).sum()
        n_hold = (df["position_signal"] == 0).sum()
        active_pct = (n_buy + n_sell) / len(df) * 100
        avg_conf = conf_vals[conf_mask].mean() if conf_mask.any() else 0.0

        logger.info(
            f"[{symbol}] Meta-ensemble signals: "
            f"BUY={n_buy}, SELL={n_sell}, HOLD={n_hold} "
            f"(active={active_pct:.1f}%, avg_conf={avg_conf:.3f}, "
            f"sizing={'proportional' if self.proportional_sizing else 'binary'})"
        )

        return df
