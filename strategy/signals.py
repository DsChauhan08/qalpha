"""
Strategy Module - V1
Signal aggregation and trading logic.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal with metadata."""

    symbol: str
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    timestamp: pd.Timestamp


class SignalAggregator:
    """
    Aggregates signals from multiple sources with confidence weighting.
    """

    def __init__(self, signal_threshold: float = 0.3):
        self.threshold = signal_threshold
        self.weights = {
            "rsi": 0.20,
            "macd": 0.25,
            "bollinger": 0.20,
            "trend": 0.25,
            "volume": 0.10,
        }

    def aggregate(
        self, signals: Dict[str, float], confidences: Dict[str, float] = None
    ) -> Tuple[float, float]:
        """
        Aggregate multiple signals into single direction.

        Args:
            signals: Dict of source -> signal value (-1 to 1)
            confidences: Dict of source -> confidence (0 to 1)

        Returns:
            Tuple of (aggregated_signal, combined_confidence)
        """
        if not signals:
            return 0.0, 0.0

        if confidences is None:
            confidences = {k: 1.0 for k in signals}

        weighted_sum = 0.0
        total_weight = 0.0

        for source, signal in signals.items():
            weight = self.weights.get(source, 0.1)
            conf = confidences.get(source, 1.0)

            weighted_sum += signal * weight * conf
            total_weight += weight * conf

        if total_weight == 0:
            return 0.0, 0.0

        agg_signal = weighted_sum / total_weight

        # Combined confidence
        avg_conf = np.mean(list(confidences.values()))

        # Agreement factor - signals agreeing increases confidence
        signal_values = list(signals.values())
        if len(signal_values) > 1:
            agreement = 1 - np.std(signal_values) / 2
        else:
            agreement = 0.5

        combined_conf = avg_conf * agreement

        return float(agg_signal), float(combined_conf)


class MomentumStrategy:
    """
    Multi-indicator momentum strategy.

    Generates signals based on:
    - RSI oversold/overbought
    - MACD crossovers
    - Bollinger Band position
    - Trend strength (ADX)
    - Volume confirmation
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        adx_threshold: float = 25,
        signal_threshold: float = 0.3,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_threshold = adx_threshold
        self.threshold = signal_threshold
        self.aggregator = SignalAggregator(signal_threshold)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for each bar.

        Args:
            df: DataFrame with technical indicators

        Returns:
            DataFrame with signal columns added
        """
        result = df.copy()

        # RSI signal
        result["rsi_signal"] = 0.0
        result.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1.0
        result.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1.0
        # Gradual signal in between
        mid_range = (df["rsi"] >= self.rsi_oversold) & (
            df["rsi"] <= self.rsi_overbought
        )
        result.loc[mid_range, "rsi_signal"] = (50 - df.loc[mid_range, "rsi"]) / 50

        # MACD signal - histogram direction
        result["macd_signal_dir"] = np.tanh(
            df["macd_hist"] / df["atr"].clip(lower=0.01)
        )

        # Bollinger signal - mean reversion
        result["bb_signal"] = 0.0
        result.loc[df["bb_position"] < 0.2, "bb_signal"] = 1.0
        result.loc[df["bb_position"] > 0.8, "bb_signal"] = -1.0

        # Trend signal using ADX
        is_trending = df["adx"] > self.adx_threshold
        result["trend_signal"] = 0.0
        # In trending market, go with MACD
        result.loc[is_trending, "trend_signal"] = result.loc[
            is_trending, "macd_signal_dir"
        ]

        # Volume signal - OBV trend
        obv_trend = np.sign(df["obv"] - df["obv_sma"])
        result["volume_signal"] = obv_trend * 0.5  # Lower weight

        # Aggregate signals
        signals = []
        confidences = []

        for i in range(len(result)):
            row_signals = {
                "rsi": result.iloc[i]["rsi_signal"],
                "macd": result.iloc[i]["macd_signal_dir"],
                "bollinger": result.iloc[i]["bb_signal"],
                "trend": result.iloc[i]["trend_signal"],
                "volume": result.iloc[i]["volume_signal"],
            }

            # Confidence based on indicator clarity
            row_conf = {
                "rsi": min(abs(result.iloc[i]["rsi"] - 50) / 30, 1.0),
                "macd": min(
                    abs(result.iloc[i]["macd_hist"]) / result.iloc[i]["atr"], 1.0
                )
                if result.iloc[i]["atr"] > 0
                else 0,
                "bollinger": 1.0
                if result.iloc[i]["bb_position"] < 0.2
                or result.iloc[i]["bb_position"] > 0.8
                else 0.5,
                "trend": min(result.iloc[i]["adx"] / 40, 1.0)
                if result.iloc[i]["adx"] > 0
                else 0,
                "volume": 0.5,
            }

            agg, conf = self.aggregator.aggregate(row_signals, row_conf)
            signals.append(agg)
            confidences.append(conf)

        result["signal"] = signals
        result["signal_confidence"] = confidences

        # Apply threshold
        result["position_signal"] = np.where(
            abs(result["signal"]) >= self.threshold, np.sign(result["signal"]), 0
        )

        return result


class MeanReversionStrategy:
    """
    Mean reversion strategy for range-bound markets.
    """

    def __init__(
        self, zscore_entry: float = 2.0, zscore_exit: float = 0.5, lookback: int = 20
    ):
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        result = df.copy()

        # Calculate z-score of price
        rolling_mean = df["close"].rolling(self.lookback).mean()
        rolling_std = df["close"].rolling(self.lookback).std()
        result["zscore"] = (df["close"] - rolling_mean) / rolling_std.clip(lower=0.01)

        # Generate signals
        result["signal"] = 0.0
        result.loc[result["zscore"] > self.zscore_entry, "signal"] = -1.0
        result.loc[result["zscore"] < -self.zscore_entry, "signal"] = 1.0

        # Exit signals
        result["exit_signal"] = (abs(result["zscore"]) < self.zscore_exit).astype(float)

        # Confidence based on z-score magnitude
        result["signal_confidence"] = np.clip(abs(result["zscore"]) / 3, 0, 1)

        return result


class TrendFollowingStrategy:
    """
    Simple trend following based on moving average crossover.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast = fast_period
        self.slow = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following signals."""
        result = df.copy()

        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()

        result["fast_ma"] = fast_ma
        result["slow_ma"] = slow_ma

        # Signal is difference normalized by volatility
        diff = fast_ma - slow_ma
        norm_diff = diff / df["atr"].clip(lower=0.01)

        result["signal"] = np.tanh(norm_diff)
        result["signal_confidence"] = np.clip(abs(norm_diff) / 2, 0, 1)

        return result


class BreakoutTrendStrategy:
    """
    Breakout trend strategy with MA filter.
    """

    def __init__(
        self, short_ma: int = 50, long_ma: int = 200, breakout_window: int = 55
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.breakout_window = breakout_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        close = df["close"]
        short_ma = close.rolling(self.short_ma).mean()
        long_ma = close.rolling(self.long_ma).mean()
        upper = close.rolling(self.breakout_window).max()
        lower = close.rolling(self.breakout_window).min()

        signal = np.zeros(len(result), dtype=float)
        long_break = (close > upper) & (short_ma > long_ma)
        short_break = (close < lower) & (short_ma < long_ma)
        signal[long_break] = 1.0
        signal[short_break] = -1.0

        result["signal"] = signal
        if "atr" in result.columns:
            dist = np.where(signal > 0, close - upper, lower - close)
            confidence = np.clip((dist / result["atr"]).abs(), 0, 1)
        else:
            confidence = np.clip(abs(signal), 0, 1)
        result["signal_confidence"] = pd.Series(confidence).fillna(0.0).values

        return result


class TimeSeriesMomentumStrategy:
    """
    Time-series momentum using multi-horizon returns.
    """

    def __init__(self, long_window: int = 252, short_window: int = 63):
        self.long_window = long_window
        self.short_window = short_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        close = df["close"]
        long_mom = close.pct_change(self.long_window)
        short_mom = close.pct_change(self.short_window)

        result["ts_mom_long"] = long_mom
        result["ts_mom_short"] = short_mom

        signal = np.zeros(len(result), dtype=float)
        both_pos = (long_mom > 0) & (short_mom > 0)
        both_neg = (long_mom < 0) & (short_mom < 0)
        signal[both_pos] = 1.0
        signal[both_neg] = -1.0

        result["signal"] = signal
        result["signal_confidence"] = np.clip(long_mom.abs() * 4, 0, 1).fillna(0.0)

        return result


class CompositeStrategy:
    """
    Combines multiple strategies with regime-based weighting.
    """

    def __init__(self):
        self.momentum = MomentumStrategy()
        self.mean_rev = MeanReversionStrategy()
        self.trend = TrendFollowingStrategy()
        self.aggregator = SignalAggregator()

    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime.

        Returns:
            'trending', 'mean_reverting', or 'mixed'
        """
        if len(df) < 30:
            return "mixed"

        recent = df.tail(30)

        # ADX for trend strength
        avg_adx = recent["adx"].mean()

        # Hurst exponent approximation using variance ratio
        returns = recent["returns"].dropna()
        if len(returns) < 20:
            return "mixed"

        var_1 = returns.var()
        var_5 = returns.rolling(5).sum().var() / 5

        if var_5 > 0:
            vr = var_1 / var_5
        else:
            vr = 1

        # VR > 1 suggests mean reversion, < 1 suggests trend
        if avg_adx > 30 and vr < 0.8:
            return "trending"
        elif avg_adx < 20 and vr > 1.2:
            return "mean_reverting"
        else:
            return "mixed"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite signals based on regime.
        """
        regime = self.detect_regime(df)

        # Generate from all strategies
        mom_df = self.momentum.generate_signals(df)
        mr_df = self.mean_rev.generate_signals(df)
        trend_df = self.trend.generate_signals(df)

        result = df.copy()

        # Weight based on regime
        if regime == "trending":
            weights = {"momentum": 0.4, "mean_rev": 0.1, "trend": 0.5}
        elif regime == "mean_reverting":
            weights = {"momentum": 0.2, "mean_rev": 0.6, "trend": 0.2}
        else:
            weights = {"momentum": 0.4, "mean_rev": 0.3, "trend": 0.3}

        # Combine signals
        result["signal"] = (
            mom_df["signal"] * weights["momentum"]
            + mr_df["signal"] * weights["mean_rev"]
            + trend_df["signal"] * weights["trend"]
        )

        result["signal_confidence"] = (
            mom_df["signal_confidence"] * weights["momentum"]
            + mr_df["signal_confidence"] * weights["mean_rev"]
            + trend_df["signal_confidence"] * weights["trend"]
        )

        result["regime"] = regime

        return result


class AdaptiveCompositeStrategy:
    """
    Composite strategy with long-term trend filter and time-series momentum.
    """

    def __init__(
        self,
        long_trend_window: int = 200,
        adx_trend_threshold: float = 25,
        adx_mean_threshold: float = 20,
        trend_filter_strength: float = 0.75,
    ):
        self.momentum = MomentumStrategy()
        self.mean_rev = MeanReversionStrategy()
        self.trend = TrendFollowingStrategy()
        self.breakout = BreakoutTrendStrategy()
        self.ts_mom = TimeSeriesMomentumStrategy()
        self.long_trend_window = long_trend_window
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_mean_threshold = adx_mean_threshold
        self.trend_filter_strength = trend_filter_strength

    def _regime(self, df: pd.DataFrame) -> str:
        if len(df) < max(60, self.long_trend_window):
            return "mixed"

        recent = df.tail(30)
        avg_adx = recent["adx"].mean()

        close = df["close"]
        long_ma = close.rolling(self.long_trend_window).mean()
        long_trend = close.iloc[-1] - long_ma.iloc[-1]

        if avg_adx >= self.adx_trend_threshold and long_trend >= 0:
            return "trending_up"
        if avg_adx >= self.adx_trend_threshold and long_trend < 0:
            return "trending_down"
        if avg_adx <= self.adx_mean_threshold:
            return "mean_reverting"
        return "mixed"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        regime = self._regime(df)

        mom_df = self.momentum.generate_signals(df)
        mr_df = self.mean_rev.generate_signals(df)
        trend_df = self.trend.generate_signals(df)
        brk_df = self.breakout.generate_signals(df)
        tsm_df = self.ts_mom.generate_signals(df)

        result = df.copy()

        if regime == "trending_up":
            weights = {
                "momentum": 0.15,
                "mean_rev": 0.1,
                "trend": 0.2,
                "breakout": 0.25,
                "ts_mom": 0.3,
            }
        elif regime == "trending_down":
            weights = {
                "momentum": 0.1,
                "mean_rev": 0.2,
                "trend": 0.2,
                "breakout": 0.25,
                "ts_mom": 0.25,
            }
        elif regime == "mean_reverting":
            weights = {
                "momentum": 0.15,
                "mean_rev": 0.55,
                "trend": 0.05,
                "breakout": 0.05,
                "ts_mom": 0.2,
            }
        else:
            weights = {
                "momentum": 0.25,
                "mean_rev": 0.2,
                "trend": 0.15,
                "breakout": 0.1,
                "ts_mom": 0.3,
            }

        result["signal"] = (
            mom_df["signal"] * weights["momentum"]
            + mr_df["signal"] * weights["mean_rev"]
            + trend_df["signal"] * weights["trend"]
            + brk_df["signal"] * weights["breakout"]
            + tsm_df["signal"] * weights["ts_mom"]
        )

        result["signal_confidence"] = (
            mom_df["signal_confidence"] * weights["momentum"]
            + mr_df["signal_confidence"] * weights["mean_rev"]
            + trend_df["signal_confidence"] * weights["trend"]
            + brk_df["signal_confidence"] * weights["breakout"]
            + tsm_df["signal_confidence"] * weights["ts_mom"]
        )

        close = df["close"]
        long_ma = close.rolling(self.long_trend_window).mean()
        trend_dir = np.sign(close - long_ma).fillna(0.0)

        # Reduce counter-trend exposure instead of hard blocking
        signal = result["signal"].copy()
        counter_trend = signal * trend_dir < 0
        signal[counter_trend] = signal[counter_trend] * (1 - self.trend_filter_strength)
        result["signal"] = signal

        result["regime"] = regime
        result["long_trend"] = trend_dir

        return result


class EnhancedCompositeStrategy:
    """
    Enhanced composite strategy that layers cross-asset signals on top of
    the single-asset AdaptiveCompositeStrategy.

    Cross-asset signal sources:
    - Cross-sectional momentum (rank-based long/short)
    - PCA-based statistical arbitrage residuals
    - Regime-adaptive momentum with variable lookbacks

    These are computed once across all symbols, then merged per-symbol
    into the single-asset signal from AdaptiveCompositeStrategy.

    The final signal is a weighted blend:
      signal = w_base * adaptive_signal + w_xs * cross_sectional_signal
    where w_xs increases when cross-asset signals have high agreement.
    """

    def __init__(
        self,
        base_weight: float = 0.60,
        cross_sectional_weight: float = 0.40,
        momentum_lookback: int = 126,
        momentum_n_top: int = 5,
        stat_arb_n_factors: int = 3,
        residual_threshold: float = 1.5,
        long_trend_window: int = 200,
    ):
        self.base_weight = base_weight
        self.cross_sectional_weight = cross_sectional_weight
        self.momentum_lookback = momentum_lookback
        self.momentum_n_top = momentum_n_top
        self.stat_arb_n_factors = stat_arb_n_factors
        self.residual_threshold = residual_threshold

        # Base strategy
        self.adaptive = AdaptiveCompositeStrategy(
            long_trend_window=long_trend_window,
        )

        # Cross-asset signals (computed lazily)
        self._xs_momentum_signals: Optional[pd.DataFrame] = None
        self._stat_arb_signals: Optional[pd.DataFrame] = None
        self._regime_signals: Optional[pd.DataFrame] = None

    def fit_cross_asset(self, price_data: Dict[str, pd.DataFrame]) -> None:
        """
        Pre-compute cross-asset signals from multi-symbol price data.

        This must be called BEFORE generate_signals() to provide
        cross-asset context. The backtest pipeline calls this once
        after all data is loaded.

        Args:
            price_data: Dict of symbol -> DataFrame with 'close' column
        """
        # Build aligned close price matrix
        closes = {}
        for sym, df in price_data.items():
            if "close" in df.columns:
                closes[sym] = df["close"]

        if len(closes) < 3:
            logger.warning(
                "Need >= 3 symbols for cross-asset signals, got %d", len(closes)
            )
            return

        price_matrix = pd.DataFrame(closes).dropna()
        if len(price_matrix) < self.momentum_lookback + 50:
            logger.warning(
                "Insufficient data for cross-asset signals: %d rows", len(price_matrix)
            )
            return

        # 1. Cross-sectional momentum
        self._compute_xs_momentum(price_matrix)

        # 2. PCA stat arb residuals
        self._compute_stat_arb(price_matrix)

        # 3. Regime-adaptive momentum
        self._compute_regime_momentum(price_matrix)

        logger.info(
            "Cross-asset signals computed for %d symbols over %d periods",
            len(closes),
            len(price_matrix),
        )

    def _compute_xs_momentum(self, prices: pd.DataFrame) -> None:
        """Compute cross-sectional momentum rankings."""
        returns = prices.pct_change(self.momentum_lookback)
        ranks = returns.rank(axis=1, ascending=False, na_option="bottom")
        n_assets = len(prices.columns)
        n_select = min(self.momentum_n_top, max(1, n_assets // 3))

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[ranks <= n_select] = 1.0
        signals[ranks > n_assets - n_select] = -1.0

        # Normalize to [-1, 1] with smooth transition
        mid = (n_assets + 1) / 2
        smooth = -(ranks - mid) / mid
        # Blend: 70% hard signal, 30% smooth
        self._xs_momentum_signals = signals * 0.7 + smooth.clip(-1, 1) * 0.3

    def _compute_stat_arb(self, prices: pd.DataFrame) -> None:
        """Compute PCA stat arb residual z-scores."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("scikit-learn not available, skipping PCA stat arb")
            return

        returns = prices.pct_change().dropna()
        if len(returns) < 60:
            return

        n_factors = min(
            self.stat_arb_n_factors, len(prices.columns) - 1, len(returns) - 1
        )
        if n_factors < 1:
            return

        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(returns.values)
        loadings = pca.components_  # (n_factors, n_assets)

        # Reconstruct systematic returns
        systematic = factor_returns @ loadings
        residuals = returns.values - systematic

        # Rolling z-score of residuals
        residual_df = pd.DataFrame(
            residuals, index=returns.index, columns=returns.columns
        )
        rolling_mean = residual_df.rolling(20).mean()
        rolling_std = residual_df.rolling(20).std().clip(lower=1e-8)
        z_scores = (residual_df - rolling_mean) / rolling_std

        # Generate mean-reversion signals from residuals
        signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        signals[z_scores < -self.residual_threshold] = 1.0  # Underperformed -> buy
        signals[z_scores > self.residual_threshold] = -1.0  # Overperformed -> sell

        # Smooth with tanh
        signals = np.tanh(signals * 0.8 + (-z_scores / 3.0) * 0.2)

        self._stat_arb_signals = signals

    def _compute_regime_momentum(self, prices: pd.DataFrame) -> None:
        """Compute regime-adaptive momentum signals."""
        returns = prices.pct_change().dropna()
        if len(returns) < 252:
            return

        # Detect regime from market-wide volatility
        market_return = returns.mean(axis=1)
        rolling_vol = market_return.rolling(63).std() * np.sqrt(252)

        signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

        for i in range(252, len(returns)):
            vol = rolling_vol.iloc[i]
            if pd.isna(vol) or vol <= 0:
                continue

            # Adaptive lookback: shorter in high-vol, longer in low-vol
            if vol > 0.25:
                lookback = 21  # High vol: fast momentum
            elif vol > 0.15:
                lookback = 63  # Normal: medium momentum
            else:
                lookback = 126  # Low vol: slow momentum

            if i >= lookback:
                mom = prices.iloc[i] / prices.iloc[i - lookback] - 1
                ranks = mom.rank(ascending=False)
                n_assets = len(prices.columns)
                n_select = min(self.momentum_n_top, max(1, n_assets // 3))
                row_signal = pd.Series(0.0, index=prices.columns)
                row_signal[ranks <= n_select] = 1.0
                row_signal[ranks > n_assets - n_select] = -1.0
                signals.iloc[i] = row_signal

        self._regime_signals = signals

    def generate_signals(
        self, df: pd.DataFrame, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate enhanced signals for a single asset.

        Merges single-asset adaptive signals with cross-asset signals
        for the given symbol.

        Args:
            df: Single-asset DataFrame with technical indicators
            symbol: Symbol name (needed to look up cross-asset signals)

        Returns:
            DataFrame with signal, signal_confidence, regime columns
        """
        # Base adaptive signal
        result = self.adaptive.generate_signals(df)

        if symbol is None:
            return result

        # Merge cross-asset signals
        xs_signals = []
        xs_weights = []

        if (
            self._xs_momentum_signals is not None
            and symbol in self._xs_momentum_signals.columns
        ):
            xs_mom = self._xs_momentum_signals[symbol]
            xs_signals.append(("xs_momentum", xs_mom))
            xs_weights.append(0.40)

        if (
            self._stat_arb_signals is not None
            and symbol in self._stat_arb_signals.columns
        ):
            sa_sig = self._stat_arb_signals[symbol]
            xs_signals.append(("stat_arb", sa_sig))
            xs_weights.append(0.35)

        if self._regime_signals is not None and symbol in self._regime_signals.columns:
            regime_sig = self._regime_signals[symbol]
            xs_signals.append(("regime_mom", regime_sig))
            xs_weights.append(0.25)

        if not xs_signals:
            return result

        # Normalize weights
        total_w = sum(xs_weights)
        xs_weights = [w / total_w for w in xs_weights]

        # Compute blended cross-asset signal aligned to df's index
        xs_combined = pd.Series(0.0, index=df.index, dtype=float)
        for (name, sig), w in zip(xs_signals, xs_weights):
            aligned = sig.reindex(df.index, method=None).fillna(0.0)
            xs_combined += aligned * w

        # Blend with base signal
        base_signal = result["signal"].copy()
        base_conf = result["signal_confidence"].copy()

        # Cross-asset agreement boosts confidence
        agreement = (np.sign(base_signal) == np.sign(xs_combined)).astype(float)
        agreement_boost = agreement * 0.15  # Up to 15% confidence boost

        blended = (
            base_signal * self.base_weight + xs_combined * self.cross_sectional_weight
        )

        result["signal"] = blended
        result["signal_confidence"] = (base_conf + agreement_boost).clip(0, 1)
        result["xs_signal"] = xs_combined

        return result
