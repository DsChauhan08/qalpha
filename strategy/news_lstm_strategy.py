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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantum_alpha.data.collectors.news_collector import (
    NewsCollector,
    SENTIMENT_FEATURE_COLS,
)
from quantum_alpha.llm.gemini_router import GeminiRouter, LLMDecision

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
        llm_enabled: bool = False,
        llm_mode: str | None = None,
        llm_models: list[str] | None = None,
        llm_min_alignment: float = 0.80,
        llm_fail_mode: str = "hold",
        llm_scope: str | None = None,
        llm_max_calls: int = 100000,
        llm_env_path: str | None = None,
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
            llm_enabled: Enable Gemini adjudication layer
            llm_mode: Gemini mode: off|simulated|api
            llm_models: Optional ordered model list for failover
            llm_min_alignment: Min LLM score required for action approval
            llm_fail_mode: On LLM failure, hold|pass
            llm_scope: all|latest (defaults: api->latest, otherwise->all)
            llm_max_calls: Max LLM calls per generate_signals() pass
            llm_env_path: Optional .env path for Gemini keys/config
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

        # LLM middleware: fail-safe by construction (default off).
        self._llm_router = GeminiRouter.from_env(
            enabled=llm_enabled,
            mode=llm_mode,
            models=llm_models,
            min_alignment_score=llm_min_alignment,
            fail_mode=llm_fail_mode,
            env_path=llm_env_path,
        )
        if llm_scope:
            self._llm_scope = llm_scope.strip().lower()
        elif self._llm_router.config.mode == "api":
            self._llm_scope = "latest"
        else:
            self._llm_scope = "all"
        if self._llm_scope not in {"all", "latest"}:
            self._llm_scope = "latest"
        self._llm_max_calls = max(1, int(llm_max_calls))

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
        df["llm_buy_score"] = 0.0
        df["llm_sell_score"] = 0.0
        df["llm_hold_score"] = 1.0
        df["llm_alignment_score"] = 0.0
        df["llm_distraction_risk"] = 1.0
        df["llm_gate_pass"] = 0.0
        df["llm_decision"] = "HOLD"
        df["llm_mode"] = "off"
        df["llm_rationale"] = ""

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
        pred_llm_buy = pd.Series(0.0, index=pred_indices)
        pred_llm_sell = pd.Series(0.0, index=pred_indices)
        pred_llm_hold = pd.Series(1.0, index=pred_indices)
        pred_llm_alignment = pd.Series(0.0, index=pred_indices)
        pred_llm_distraction = pd.Series(1.0, index=pred_indices)
        pred_llm_gate = pd.Series(0.0, index=pred_indices)
        pred_llm_decision = pd.Series("HOLD", index=pred_indices, dtype=object)
        pred_llm_mode = pd.Series("off", index=pred_indices, dtype=object)
        pred_llm_rationale = pd.Series("", index=pred_indices, dtype=object)

        llm_calls = 0
        latest_idx = pred_indices[-1] if len(pred_indices) > 0 else None

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
            gated_action = float(action) if abs(action) > 0 and conf >= self.confidence_threshold else 0.0

            run_llm = self._llm_router.enabled and gated_action != 0.0
            llm_skip_reason = ""
            if run_llm and self._llm_scope == "latest" and idx != latest_idx:
                run_llm = False
                llm_skip_reason = "scope"
            if run_llm and llm_calls >= self._llm_max_calls:
                run_llm = False
                llm_skip_reason = "budget"

            if run_llm:
                context = self._build_llm_context(
                    symbol=symbol,
                    timestamp=idx,
                    action=int(gated_action),
                    confidence=conf,
                    signal_value=float(continuous_signal),
                    signal_probs=signal_probs[i],
                    feature_row=features_df.loc[idx] if idx in features_df.index else None,
                    window=X_windows[i],
                    n_classes=signal_probs.shape[1],
                )
                decision, gate_pass = self._run_llm_gate(
                    context=context,
                    action=int(gated_action),
                )
                llm_calls += 1
                pred_llm_buy[idx] = decision.buy_score
                pred_llm_sell[idx] = decision.sell_score
                pred_llm_hold[idx] = decision.hold_score
                pred_llm_alignment[idx] = decision.alignment_score
                pred_llm_distraction[idx] = decision.distraction_risk
                pred_llm_gate[idx] = 1.0 if gate_pass else 0.0
                pred_llm_decision[idx] = decision.decision
                pred_llm_mode[idx] = decision.mode
                pred_llm_rationale[idx] = decision.rationale
                pred_position[idx] = gated_action if gate_pass else 0.0
            else:
                if not self._llm_router.enabled:
                    # LLM disabled => pass-through.
                    pred_llm_buy[idx] = 1.0 if gated_action > 0 else 0.0
                    pred_llm_sell[idx] = 1.0 if gated_action < 0 else 0.0
                    pred_llm_hold[idx] = 1.0 if gated_action == 0 else 0.0
                    pred_llm_alignment[idx] = 1.0
                    pred_llm_distraction[idx] = 0.0
                    pred_llm_gate[idx] = 1.0 if gated_action != 0 else 0.0
                    pred_llm_decision[idx] = (
                        "BUY" if gated_action > 0 else "SELL" if gated_action < 0 else "HOLD"
                    )
                    pred_llm_mode[idx] = "off"
                    pred_llm_rationale[idx] = "llm_disabled"
                    pred_position[idx] = gated_action
                else:
                    # Scope skips pass-through; budget skips obey fail-mode.
                    fail_open = self._llm_router.config.fail_mode == "pass"
                    if llm_skip_reason == "scope" or fail_open:
                        pred_position[idx] = gated_action
                        pred_llm_gate[idx] = 1.0 if gated_action != 0 else 0.0
                        pred_llm_decision[idx] = (
                            "BUY" if gated_action > 0 else "SELL" if gated_action < 0 else "HOLD"
                        )
                        pred_llm_mode[idx] = f"skip_{llm_skip_reason or 'pass'}"
                        pred_llm_rationale[idx] = "llm_skipped_pass"
                    else:
                        pred_position[idx] = 0.0
                        pred_llm_decision[idx] = "HOLD"
                        pred_llm_mode[idx] = f"skip_{llm_skip_reason or 'hold'}"
                        pred_llm_rationale[idx] = "llm_skipped_hold"

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
        df.loc[common_idx, "llm_buy_score"] = pred_llm_buy.reindex(common_idx).fillna(0.0)
        df.loc[common_idx, "llm_sell_score"] = pred_llm_sell.reindex(common_idx).fillna(0.0)
        df.loc[common_idx, "llm_hold_score"] = pred_llm_hold.reindex(common_idx).fillna(1.0)
        df.loc[common_idx, "llm_alignment_score"] = pred_llm_alignment.reindex(
            common_idx
        ).fillna(0.0)
        df.loc[common_idx, "llm_distraction_risk"] = pred_llm_distraction.reindex(
            common_idx
        ).fillna(1.0)
        df.loc[common_idx, "llm_gate_pass"] = pred_llm_gate.reindex(common_idx).fillna(0.0)
        df.loc[common_idx, "llm_decision"] = pred_llm_decision.reindex(common_idx).fillna(
            "HOLD"
        )
        df.loc[common_idx, "llm_mode"] = pred_llm_mode.reindex(common_idx).fillna("off")
        df.loc[common_idx, "llm_rationale"] = pred_llm_rationale.reindex(common_idx).fillna(
            ""
        )

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
        if self._llm_router.enabled:
            logger.info(
                "LLM gate: calls=%d scope=%s min_score=%.2f",
                llm_calls,
                self._llm_scope,
                self._llm_router.min_alignment_score,
            )

        return df

    def _build_llm_context(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        action: int,
        confidence: float,
        signal_value: float,
        signal_probs: np.ndarray,
        feature_row: pd.Series | None,
        window: np.ndarray,
        n_classes: int,
    ) -> Dict[str, Any]:
        class_probs: Dict[str, float]
        if n_classes == 2:
            class_probs = {
                "down": float(signal_probs[0]),
                "up": float(signal_probs[1]),
            }
        else:
            class_probs = {
                "sell": float(signal_probs[0]),
                "hold": float(signal_probs[1]),
                "buy": float(signal_probs[2]),
            }

        # Simple low-cost noise diagnostics from the active sequence window.
        win = np.asarray(window, dtype=float)
        if win.ndim != 2 or win.shape[0] < 3:
            noise_score = 0.0
            trend_score = 0.0
            vol_score = 0.0
        else:
            step_changes = np.abs(np.diff(win, axis=0))
            noise_score = float(np.clip(np.nanmean(step_changes) / 2.0, 0.0, 1.0))
            trend_score = float(np.tanh(np.nanmean(win[-3:, :]) / 1.8))
            vol_score = float(
                np.clip(np.nanstd(win[:, : min(4, win.shape[1])]) / 2.0, 0.0, 1.0)
            )

        feature_slice: Dict[str, float] = {}
        if feature_row is not None:
            for k in ("returns", "return_zscore", "vol_regime", "trend_strength", "sentiment_proxy"):
                if k in feature_row:
                    try:
                        feature_slice[k] = float(feature_row[k])
                    except Exception:
                        continue

        return {
            "symbol": str(symbol),
            "timestamp": str(timestamp),
            "proposed_action": int(action),
            "model_confidence": float(np.clip(confidence, 0.0, 1.0)),
            "signal_value": float(np.clip(signal_value, -1.0, 1.0)),
            "class_probs": class_probs,
            "trend_score": trend_score,
            "volatility_score": vol_score,
            "noise_score": noise_score,
            "feature_slice": feature_slice,
        }

    def _run_llm_gate(self, context: Dict[str, Any], action: int) -> tuple[LLMDecision, bool]:
        decision = self._llm_router.evaluate(context=context, proposed_action=action)
        action_score = decision.score_for_action(action)
        gate_pass = (
            decision.aligns_with_action(action)
            and action_score >= self._llm_router.min_alignment_score
            and decision.alignment_score >= self._llm_router.min_alignment_score
        )
        return decision, bool(gate_pass)

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

        Then optionally adds MC/Padé features if the loaded model was
        trained with them (detected via scaler dimension > 21).

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

        feature_cols = list(ALL_FEAT_COLS)

        # Add MC/Padé features if the model was trained with them.
        # Detection: if the scaler has more dimensions than the base 21 features,
        # the model expects MC/Padé features.
        n_base = len(ALL_FEAT_COLS)
        scaler_dim = (
            len(self._scaler_params["mean"]) if self._scaler_params is not None else 0
        )
        needs_mc_pade = scaler_dim > n_base

        if needs_mc_pade:
            features_df, mc_cols = self._add_mc_pade_features(features_df)
            for c in mc_cols:
                if c not in feature_cols:
                    feature_cols.append(c)
            logger.info(
                "MC/Padé features added for inference: %d cols (scaler expects %d, base %d)",
                len(mc_cols),
                scaler_dim,
                n_base,
            )

        return features_df, feature_cols

    def _add_mc_pade_features(
        self, features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list]:
        """
        Compute the same curated MC/Padé features used during LSTM training.

        Returns:
            (features_df with MC/Padé columns, list of MC/Padé column names added)
        """
        MC_PADE_LSTM_COLS = [
            "mc_5d_prob_up",
            "mc_5d_mean_return",
            "mc_5d_var_5",
            "mc_5d_cvar_5",
            "mc_5d_gain_loss_ratio",
            "mc_5d_prob_touch_up_3pct",
            "mc_5d_prob_touch_down_3pct",
            "jd_5d_prob_up",
            "jd_5d_skew",
            "jd_5d_kurt",
            "mc_est_mu",
            "mc_est_sigma",
            "pade_prob_down_2pct",
            "pade_prob_up_2pct",
            "pade_tail_ratio",
            "pade_es_5pct",
        ]
        try:
            from quantum_alpha.features.mc_pade_features import MCPadeFeatureGenerator

            if not hasattr(self, "_mc_gen"):
                self._mc_gen = MCPadeFeatureGenerator()
            mc_result = self._mc_gen.generate_features_fast(features_df)
            mc_available = [c for c in MC_PADE_LSTM_COLS if c in mc_result.columns]
            for c in mc_available:
                features_df[c] = mc_result[c]
            logger.info("MC/Padé inference features: %d added", len(mc_available))
            return features_df, mc_available
        except Exception as e:
            logger.warning("MC/Padé features skipped at inference: %s", e)
            return features_df, []

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
            "llm_decision": str(last.get("llm_decision", "HOLD")),
            "llm_alignment_score": float(last.get("llm_alignment_score", 0.0)),
            "llm_gate_pass": bool(float(last.get("llm_gate_pass", 0.0)) > 0.5),
            "timestamp": str(df_with_signals.index[-1]),
        }
