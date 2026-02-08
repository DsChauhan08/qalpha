"""
Sentiment & alternative-data strategy wrappers.

These lightweight wrappers sit on top of the feature-level analyzers in
`quantum_alpha.features.alternative` and emit standard `signal` and
`signal_confidence` columns so they can plug into the backtest/portfolio
pipeline without additional glue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from quantum_alpha.features.alternative.social_buzz import SocialBuzzAnalyzer
from quantum_alpha.features.alternative.options_sentiment import (
    OptionsSentimentAnalyzer,
)
from quantum_alpha.features.alternative.insider_momentum import (
    InsiderMomentumGenerator,
)
from quantum_alpha.features.alternative.congress_signal import CongressSignalGenerator
from quantum_alpha.features.alternative.earnings_surprise import EarningsSurpriseDetector
from quantum_alpha.features.alternative.short_interest import ShortInterestAnalyzer


def _with_confidence(df: pd.DataFrame, default: float = 0.5) -> pd.DataFrame:
    """Attach a signal_confidence column if missing."""
    out = df.copy()
    if "signal_confidence" in out.columns:
        return out
    if "confidence" in out.columns:
        out["signal_confidence"] = np.clip(out["confidence"], 0.0, 1.0)
    elif "sentiment" in out.columns:
        out["signal_confidence"] = np.clip(out["sentiment"].abs(), 0.0, 1.0)
    else:
        out["signal_confidence"] = np.clip(out.get("signal", 0).abs(), 0.0, 1.0)
        out.loc[out["signal_confidence"] == 0, "signal_confidence"] = default
    return out


class SocialSentimentStrategy:
    """Wrapper around SocialBuzzAnalyzer."""

    def __init__(
        self,
        sentiment_window: int = 7,
        sentiment_threshold: float = 0.3,
        volume_threshold: int = 100,
    ) -> None:
        self.analyzer = SocialBuzzAnalyzer(
            sentiment_window=sentiment_window,
            sentiment_threshold=sentiment_threshold,
            volume_threshold=volume_threshold,
        )

    def generate_signals(self, mentions: pd.DataFrame) -> pd.DataFrame:
        sigs = self.analyzer.generate_signals(mentions)
        # Confidence rises with sentiment magnitude and volume quality
        vol_norm = np.clip(
            sigs.get("volume", pd.Series(0)).astype(float) / self.analyzer.volume_threshold,
            0.0,
            2.0,
        )
        conf = np.clip(np.abs(sigs["sentiment_ma"]) / max(self.analyzer.sentiment_threshold, 1e-4), 0, 2)
        sigs["signal_confidence"] = np.clip(0.3 * vol_norm + 0.7 * conf, 0.0, 1.0)
        sigs["date"] = pd.to_datetime(sigs["date"])
        return sigs.set_index("date")


class OptionsSentimentStrategy:
    """Wrapper for options flow / sentiment signals (contrarian)."""

    def __init__(
        self,
        lookback: int = 20,
        pc_ratio_upper: float = 1.5,
        pc_ratio_lower: float = 0.5,
        volume_z_threshold: float = 2.0,
    ) -> None:
        self.analyzer = OptionsSentimentAnalyzer(
            lookback=lookback,
            pc_ratio_upper=pc_ratio_upper,
            pc_ratio_lower=pc_ratio_lower,
            volume_z_threshold=volume_z_threshold,
        )

    def generate_signals(self, options_data: pd.DataFrame) -> pd.DataFrame:
        sigs = self.analyzer.generate_signals(options_data)
        # Confidence: fraction of sub-signals agreeing
        sub_cols = [c for c in ["pc_signal", "iv_signal", "vol_signal"] if c in sigs]
        if sub_cols:
            agreement = sigs[sub_cols].abs().mean(axis=1)
            sigs["signal_confidence"] = np.clip(agreement, 0.0, 1.0)
        else:
            sigs["signal_confidence"] = np.clip(sigs["signal"].abs(), 0.0, 1.0)
        return sigs


class InsiderTradingStrategy:
    """Follow-the-insiders strategy."""

    def __init__(
        self,
        lookback_days: int = 90,
        min_transaction_value: float = 10_000.0,
        buy_threshold: float = 0.5,
        sell_threshold: float = -0.5,
    ) -> None:
        self.generator = InsiderMomentumGenerator(
            lookback_days=lookback_days,
            min_transaction_value=min_transaction_value,
            buy_sentiment_threshold=buy_threshold,
            sell_sentiment_threshold=sell_threshold,
        )

    def generate_signals(self, filings: pd.DataFrame) -> pd.DataFrame:
        sigs = self.generator.generate_signals(filings)
        sigs = _with_confidence(sigs, default=0.6)
        if "transaction_date" in sigs:
            sigs["timestamp"] = pd.to_datetime(sigs["transaction_date"])
            sigs = sigs.set_index("timestamp")
        return sigs


class CongressTradingStrategy:
    """Trade alongside (or against) congressional disclosures."""

    def __init__(self, min_amount: float = 10_000.0) -> None:
        self.generator = CongressSignalGenerator(min_amount=min_amount)

    def generate_signals(self, trades: pd.DataFrame) -> pd.DataFrame:
        sigs = self.generator.generate_signals(trades)
        sigs = _with_confidence(sigs, default=0.55)
        if "transaction_date" in sigs:
            sigs["timestamp"] = pd.to_datetime(sigs["transaction_date"])
            sigs = sigs.set_index("timestamp")
        return sigs


class EarningsSurpriseStrategy:
    """Post-earnings announcement drift (PEAD) wrapper."""

    def __init__(self, surprise_threshold: float = 0.05, sue_threshold: float = 1.5):
        self.detector = EarningsSurpriseDetector(
            surprise_threshold=surprise_threshold, sue_threshold=sue_threshold
        )

    def generate_signals(self, earnings: pd.DataFrame, use_sue: bool = False) -> pd.DataFrame:
        sigs = (
            self.detector.generate_sue_signals(earnings)
            if use_sue
            else self.detector.generate_signals(earnings)
        )
        sigs = _with_confidence(sigs, default=0.6)
        if "announcement_date" in sigs:
            sigs["timestamp"] = pd.to_datetime(sigs["announcement_date"])
            sigs = sigs.set_index("timestamp")
        return sigs


class ShortInterestStrategy:
    """Short-squeeze / bearish-follow strategy."""

    def __init__(
        self,
        squeeze_threshold: float = 1.5,
        bearish_threshold: float = 0.1,
        mode: str = "squeeze",
    ) -> None:
        self.analyzer = ShortInterestAnalyzer(
            squeeze_threshold=squeeze_threshold, bearish_threshold=bearish_threshold
        )
        self.mode = mode

    def generate_signals(self, short_data: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "bearish":
            sigs = self.analyzer.generate_bearish_signals(short_data)
        else:
            sigs = self.analyzer.generate_signals(short_data)
        sigs = _with_confidence(sigs, default=0.5)
        return sigs


__all__ = [
    "SocialSentimentStrategy",
    "OptionsSentimentStrategy",
    "InsiderTradingStrategy",
    "CongressTradingStrategy",
    "EarningsSurpriseStrategy",
    "ShortInterestStrategy",
]

