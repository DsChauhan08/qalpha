"""
Machine-learning driven trading strategies.

Provides a light-weight MLTradingStrategy that can use XGBoost,
RandomForest, or GradientBoosting to forecast next-period returns and
map them to trading signals.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _default_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select a safe subset of features that commonly exist in the pipeline."""
    cols = [
        "rsi",
        "macd_hist",
        "bb_position",
        "atr_pct",
        "stoch_k",
        "stoch_d",
        "mom_3m",
        "mom_12m",
        "volume",
        "returns",
    ]
    present = [c for c in cols if c in df.columns]
    return df[present].copy()


class MLTradingStrategy:
    """
    Train a supervised model on historical features to forecast next-bar
    returns. Signals are the sign of the forecast; confidence is the
    squashed absolute forecast.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        horizon: int = 1,
        feature_cols: Optional[List[str]] = None,
        threshold: float = 0.0,
        random_state: int = 7,
    ) -> None:
        self.model_type = model_type
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.random_state = random_state
        self.model = None

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        """Instantiate a model with graceful fallbacks."""
        if self.model_type.lower() == "xgboost":
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                )
            except Exception as exc:
                logger.warning("XGBoost unavailable (%s); falling back to GradientBoostingRegressor", exc)
                self.model_type = "gbrt"

        if self.model_type.lower() in {"rf", "random_forest"}:
            try:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=200, max_depth=6, random_state=self.random_state
                )
            except Exception as exc:
                logger.warning("RandomForest unavailable (%s); falling back to GradientBoostingRegressor", exc)

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=self.random_state)
        except Exception as exc:
            logger.warning("sklearn unavailable (%s); using naive baseline", exc)
            return None

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame):
        feats = _default_features(df) if self.feature_cols is None else df[self.feature_cols]
        target = df["returns"].shift(-self.horizon).dropna()
        X = feats.iloc[: len(target)].fillna(0.0)

        self.model = self._build_model()
        if self.model is None:
            # Baseline: mean of historical returns
            self.baseline_mu = float(target.mean())
            return

        try:
            self.model.fit(X, target)
        except Exception as exc:
            logger.warning("Model fit failed (%s); reverting to baseline mean", exc)
            self.model = None
            self.baseline_mu = float(target.mean())

    def _predict(self, feats: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(feats), getattr(self, "baseline_mu", 0.0))
        try:
            return self.model.predict(feats)
        except Exception as exc:
            logger.warning("Model predict failed (%s); returning zeros", exc)
            return np.zeros(len(feats))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit (if needed) and emit signals aligned to the input frame."""
        if self.model is None:
            self.fit(df)

        feats = _default_features(df) if self.feature_cols is None else df[self.feature_cols]
        feats = feats.fillna(0.0)
        preds = self._predict(feats)

        out = df.copy()
        out["ml_forecast"] = preds
        out["signal_confidence"] = np.tanh(np.abs(preds))
        out["signal"] = 0.0
        out.loc[preds > self.threshold, "signal"] = 1.0
        out.loc[preds < -self.threshold, "signal"] = -1.0

        return out


__all__ = ["MLTradingStrategy"]

