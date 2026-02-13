"""
News-Driven LSTM Trading Strategy.

Integrates the NewsDrivenLSTM model into the Quantum Alpha strategy
framework.  Produces `signal`, `signal_confidence`, and `position_signal`
columns compatible with the backtesting engine.

Usage:
    strategy = NewsLSTMStrategy(checkpoint_name="news_lstm_20260212")
    df = strategy.generate_signals(df)
    # df now has signal/signal_confidence/position_signal columns
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quantum_alpha.data.collectors.news_collector import (
    NewsCollector,
    SENTIMENT_FEATURE_COLS,
)

logger = logging.getLogger(__name__)

# Lazy import — only used when use_real_sentiment=True
_SentimentPipeline = None
_REAL_FEATURE_COLS = None
_ALL_FEATURE_COLS = None


def _get_real_sentiment_imports():
    """Lazy-load the real sentiment pipeline to avoid heavy imports at module level."""
    global _SentimentPipeline, _REAL_FEATURE_COLS, _ALL_FEATURE_COLS
    if _SentimentPipeline is None:
        from quantum_alpha.data.collectors.sentiment_pipeline import (
            SentimentPipeline,
            REAL_SENTIMENT_FEATURE_COLS,
            ALL_FEATURE_COLS,
        )

        _SentimentPipeline = SentimentPipeline
        _REAL_FEATURE_COLS = REAL_SENTIMENT_FEATURE_COLS
        _ALL_FEATURE_COLS = ALL_FEATURE_COLS
    return _SentimentPipeline, _REAL_FEATURE_COLS, _ALL_FEATURE_COLS


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NewsLSTMStrategy:
    """
    Trading strategy powered by the News-Driven LSTM.

    generate_signals(df) contract:
        Input:  DataFrame with OHLCV columns (open, high, low, close, volume)
        Output: Same DataFrame with added columns:
                - signal:            float in [-1, 1]
                - signal_confidence: float in [0, 1]
                - position_signal:   float in {-1, 0, 1}

    The strategy:
    1. Builds sentiment proxy features from price data (via NewsCollector)
    2. Z-score normalizes using the SAME scaler from training
    3. Creates rolling windows of length seq_len
    4. Runs the trained LSTM to get trade_action + confidence
    5. Maps output to signal/signal_confidence/position_signal columns
    """

    def __init__(
        self,
        checkpoint_name: str = None,
        checkpoint_dir: str = None,
        signal_threshold: float = 0.35,
        confidence_threshold: float = 0.10,
        use_mc_dropout: bool = False,
        mc_iterations: int = 30,
        use_real_sentiment: bool = None,
    ):
        """
        Args:
            checkpoint_name: Name of saved checkpoint (without extension)
            checkpoint_dir:  Directory containing checkpoints
            signal_threshold: Min probability to emit a trade signal
            confidence_threshold: Min confidence to emit a trade signal
            use_mc_dropout: Use Monte Carlo dropout for uncertainty
            mc_iterations: Number of MC forward passes
            use_real_sentiment: If True, use FinBERT sentiment from DB.
                If None (default), auto-detect from checkpoint name
                (checkpoints with '_real_' use real sentiment).
        """
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = checkpoint_dir or str(
            Path(__file__).parent.parent / "models" / "checkpoints" / "news_lstm"
        )
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        self.use_mc_dropout = use_mc_dropout
        self.mc_iterations = mc_iterations
        self._use_real_sentiment = use_real_sentiment

        self._model = None
        self._scaler_params = None
        self._config = None
        self._loaded = False
        self._collector = NewsCollector()
        self._pipeline = None  # Lazy-loaded SentimentPipeline

    def _load_model(self):
        """Lazy-load the trained model and scaler."""
        if self._loaded:
            return

        if not HAS_TORCH:
            warnings.warn(
                "PyTorch not available. NewsLSTMStrategy will output hold signals."
            )
            self._loaded = True
            return

        if self.checkpoint_name is None:
            # Find the latest checkpoint by MODIFICATION TIME (not alphabetical)
            ckpt_dir = Path(self.checkpoint_dir)
            if ckpt_dir.exists():
                pt_files = list(ckpt_dir.glob("*.pt"))
                # Filter out config files
                pt_files = [f for f in pt_files if "_config" not in f.stem]
                if pt_files:
                    # Sort by modification time, newest last
                    pt_files.sort(key=lambda f: f.stat().st_mtime)
                    self.checkpoint_name = pt_files[-1].stem
                    logger.info("Auto-detected checkpoint: %s", self.checkpoint_name)

        if self.checkpoint_name is None:
            warnings.warn("No checkpoint found. Strategy will output hold signals.")
            self._loaded = True
            return

        # Auto-detect use_real_sentiment from checkpoint name
        if self._use_real_sentiment is None:
            self._use_real_sentiment = "_real_" in self.checkpoint_name
            if self._use_real_sentiment:
                logger.info(
                    "Auto-detected real sentiment model from checkpoint name: %s",
                    self.checkpoint_name,
                )

        try:
            from quantum_alpha.models.lstm_v4.news_lstm import (
                NewsDrivenLSTM,
                NewsLSTMConfig,
            )

            # Load metadata
            meta_path = os.path.join(
                self.checkpoint_dir, f"{self.checkpoint_name}_meta.json"
            )
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self._config = NewsLSTMConfig(**meta["config"])
            else:
                self._config = NewsLSTMConfig()

            # Override thresholds if set
            self._config.signal_threshold = self.signal_threshold

            # Load model
            model_path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_name}.pt")
            self._model = NewsDrivenLSTM(self._config)
            self._model.load(model_path)

            # Load scaler
            scaler_path = os.path.join(
                self.checkpoint_dir, f"{self.checkpoint_name}_scaler.json"
            )
            if os.path.exists(scaler_path):
                with open(scaler_path) as f:
                    data = json.load(f)
                self._scaler_params = {k: np.array(v) for k, v in data.items()}
            else:
                self._scaler_params = None

            self._loaded = True
            logger.info(
                "Loaded NewsLSTM checkpoint: %s (seq_len=%d, features=%d)",
                self.checkpoint_name,
                self._config.sequence_length,
                self._config.total_features,
            )

        except Exception as e:
            logger.error("Failed to load NewsLSTM checkpoint: %s", e)
            self._model = None
            self._loaded = True

    def generate_signals(self, df: pd.DataFrame, symbol: str = "SPY") -> pd.DataFrame:
        """
        Generate trading signals from price data.

        This is the main interface consumed by the backtesting engine.

        Args:
            df: DataFrame with OHLCV columns and DatetimeIndex
            symbol: Ticker symbol (for news collector context)

        Returns:
            DataFrame with signal, signal_confidence, position_signal columns added
        """
        df = df.copy()

        # Initialize output columns with defaults (hold / no signal)
        df["signal"] = 0.0
        df["signal_confidence"] = 0.0
        df["position_signal"] = 0.0

        self._load_model()

        if self._model is None:
            logger.warning("No model loaded. Returning hold signals.")
            return df

        try:
            return self._generate_signals_impl(df, symbol)
        except Exception as e:
            logger.error("Signal generation failed: %s", e)
            return df

    def _generate_signals_impl(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Internal signal generation with the loaded model."""
        cfg = self._config
        seq_len = cfg.sequence_length

        if len(df) < seq_len + 20:
            logger.warning(
                "Insufficient data (%d bars) for seq_len=%d. Need at least %d.",
                len(df),
                seq_len,
                seq_len + 20,
            )
            return df

        # Step 1: Build features — real sentiment or price-proxy
        if self._use_real_sentiment:
            features_df, feature_cols = self._build_real_sentiment_features(df, symbol)
        else:
            features_df, feature_cols = self._build_proxy_features(df, symbol)

        available = [c for c in feature_cols if c in features_df.columns]
        if len(available) < 10:
            logger.warning("Only %d features available, need 10+", len(available))
            return df

        X_raw = features_df[available].values

        # Step 3: Normalize using training scaler
        if self._scaler_params is not None:
            mean = self._scaler_params["mean"]
            std = self._scaler_params["std"]
            # Handle dimension mismatch (if feature set changed)
            n_feats = min(X_raw.shape[1], len(mean))
            X_scaled = (X_raw[:, :n_feats] - mean[:n_feats]) / std[:n_feats]
        else:
            # Fallback: z-score on the available data
            mean = np.nanmean(X_raw, axis=0)
            std = np.nanstd(X_raw, axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            X_scaled = (X_raw - mean) / std

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Step 4: Create rolling windows
        n_windows = len(X_scaled) - seq_len
        if n_windows <= 0:
            logger.warning("Not enough data for windowing")
            return df

        X_windows = np.zeros((n_windows, seq_len, X_scaled.shape[1]))
        for i in range(n_windows):
            X_windows[i] = X_scaled[i : i + seq_len]

        # Step 5: Run model prediction
        if self.use_mc_dropout:
            preds = self._model.predict_with_uncertainty(
                X_windows, n_iterations=self.mc_iterations
            )
        else:
            preds = self._model.predict(X_windows)

        trade_actions = preds["trade_action"]  # {-1, 0, 1}
        confidences = preds["confidence"]
        signal_probs = preds["signal_probs"]

        # Step 6: Map predictions back to DataFrame index
        # The prediction at index i corresponds to features_df.index[i + seq_len]
        pred_indices = features_df.index[seq_len : seq_len + n_windows]

        # Create signal series aligned to prediction indices
        pred_signal = pd.Series(0.0, index=pred_indices)
        pred_confidence = pd.Series(0.0, index=pred_indices)
        pred_position = pd.Series(0.0, index=pred_indices)

        for i, idx in enumerate(pred_indices):
            action = trade_actions[i]
            conf = float(confidences[i])

            # Signal: continuous in [-1, 1] based on class probabilities
            n_classes = signal_probs.shape[1]
            if n_classes == 2:
                # Binary: 0=down, 1=up
                up_prob = float(signal_probs[i, 1])
                down_prob = float(signal_probs[i, 0])
                continuous_signal = up_prob - down_prob  # [-1, 1]
            else:
                # Ternary: 0=sell, 1=hold, 2=buy
                buy_prob = float(signal_probs[i, 2])
                sell_prob = float(signal_probs[i, 0])
                continuous_signal = buy_prob - sell_prob  # [-1, 1]

            pred_signal[idx] = continuous_signal
            pred_confidence[idx] = conf

            # Position signal: discrete {-1, 0, 1} from gated action
            if abs(action) > 0 and conf >= self.confidence_threshold:
                pred_position[idx] = float(action)

        # Step 7: Map back to original df index
        # Only fill indices that exist in both
        common_idx = df.index.intersection(pred_signal.index)
        df.loc[common_idx, "signal"] = pred_signal.reindex(common_idx).fillna(0.0)
        df.loc[common_idx, "signal_confidence"] = pred_confidence.reindex(
            common_idx
        ).fillna(0.0)
        df.loc[common_idx, "position_signal"] = pred_position.reindex(
            common_idx
        ).fillna(0.0)

        # Log signal stats
        n_buys = (df["position_signal"] > 0).sum()
        n_sells = (df["position_signal"] < 0).sum()
        n_holds = (df["position_signal"] == 0).sum()
        logger.info(
            "NewsLSTM signals: %d buys, %d sells, %d holds (%.1f%% active)",
            n_buys,
            n_sells,
            n_holds,
            100.0 * (n_buys + n_sells) / max(len(df), 1),
        )

        return df

    def _build_proxy_features(
        self, df: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list]:
        """Build price-proxy sentiment features (legacy path)."""
        features_df = self._collector.build_training_features(df, symbol)
        return features_df, SENTIMENT_FEATURE_COLS

    def _build_real_sentiment_features(
        self, df: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list]:
        """
        Build real FinBERT sentiment features from the SQLite DB.

        Uses SentimentPipeline.build_training_features() which:
        - Fetches scored articles from the DB
        - Aggregates into 16 daily sentiment features
        - Adds 5 price context features
        - Lags sentiment by 1 day (no lookahead)

        Returns:
            (features_df, feature_cols) tuple
        """
        SentimentPipeline, _, ALL_FEAT_COLS = _get_real_sentiment_imports()

        if self._pipeline is None:
            self._pipeline = SentimentPipeline()

        features_df = self._pipeline.build_training_features(
            price_df=df,
            symbol=symbol,
            forward_period=1,  # targets not used at inference, but needed for dropna
            use_real_sentiment=True,
        )

        return features_df, ALL_FEAT_COLS

    def generate_signals_live(
        self,
        price_df: pd.DataFrame,
        symbol: str = "SPY",
        headlines: list = None,
    ) -> Dict:
        """
        Generate a single live trading signal.

        For paper/live trading where we have real headlines.

        Args:
            price_df: Recent OHLCV data (at least seq_len bars)
            symbol: Ticker
            headlines: Optional list of recent headline strings

        Returns:
            Dict with 'action' (-1/0/1), 'confidence', 'signal_probs'
        """
        self._load_model()

        if self._model is None:
            return {"action": 0, "confidence": 0.0, "reason": "no_model"}

        # Generate signals on the full DataFrame
        df_with_signals = self.generate_signals(price_df, symbol)

        if df_with_signals.empty:
            return {"action": 0, "confidence": 0.0, "reason": "no_data"}

        # Return the last signal
        last = df_with_signals.iloc[-1]
        return {
            "action": int(last.get("position_signal", 0)),
            "signal": float(last.get("signal", 0.0)),
            "confidence": float(last.get("signal_confidence", 0.0)),
            "timestamp": str(df_with_signals.index[-1]),
        }
