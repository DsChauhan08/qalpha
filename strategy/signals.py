"""
Strategy Module - V1
Signal aggregation and trading logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


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
