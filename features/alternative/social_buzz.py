"""
Social Media Buzz Analyzer.

Aggregates and analyses retail sentiment from social media platforms
to detect momentum shifts before they appear in price.

Data sources:
- Reddit (r/wallstreetbets, r/investing, r/stocks)
- Twitter / X
- StockTwits

Key signals:
- Rolling sentiment mean/median
- Mention volume anomalies (z-score spikes)
- Sentiment regime changes
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SocialBuzzAnalyzer:
    """
    Aggregate social media mentions and produce sentiment signals.

    Args:
        sentiment_window: Rolling window (days) for sentiment smoothing.
        sentiment_threshold: Minimum absolute smoothed sentiment to
            trigger a directional signal.
        volume_threshold: Minimum daily mention count to consider a
            symbol's sentiment reliable.
        anomaly_z_threshold: Z-score threshold for detecting sentiment
            anomalies (spike detection).
    """

    def __init__(
        self,
        sentiment_window: int = 7,
        sentiment_threshold: float = 0.3,
        volume_threshold: int = 100,
        anomaly_z_threshold: float = 2.0,
    ) -> None:
        self.sentiment_window = sentiment_window
        self.sentiment_threshold = sentiment_threshold
        self.volume_threshold = volume_threshold
        self.anomaly_z = anomaly_z_threshold

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_sentiment(self, mentions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw social media mentions to daily sentiment per symbol.

        Expected columns:
            timestamp, symbol, sentiment_score, volume

        ``sentiment_score`` is in [-1, 1]; ``volume`` is the number of
        mentions in that record (may be pre-aggregated or per-post).

        Returns:
            DataFrame with [date, symbol, sentiment_score, volume,
            sentiment_ma].
        """
        df = mentions.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        aggregated = (
            df.groupby(["date", "symbol"])
            .agg({"sentiment_score": "mean", "volume": "sum"})
            .reset_index()
        )

        # Rolling mean per symbol
        aggregated["sentiment_ma"] = aggregated.groupby("symbol")[
            "sentiment_score"
        ].transform(lambda x: x.rolling(self.sentiment_window, min_periods=1).mean())

        return aggregated

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_sentiment_anomalies(
        self,
        sentiment_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect unusual sentiment spikes via z-score of sentiment changes.

        Large positive z-score => sudden bullish shift.
        Large negative z-score => sudden bearish shift.

        Returns:
            Input DataFrame augmented with [sentiment_change,
            sentiment_zscore, is_anomaly].
        """
        out_frames: List[pd.DataFrame] = []

        for symbol in sentiment_data["symbol"].unique():
            sdf = sentiment_data[sentiment_data["symbol"] == symbol].copy()
            sdf["sentiment_change"] = sdf["sentiment_score"].diff()
            mean_chg = sdf["sentiment_change"].mean()
            std_chg = sdf["sentiment_change"].std()

            if std_chg > 0:
                sdf["sentiment_zscore"] = (sdf["sentiment_change"] - mean_chg) / std_chg
            else:
                sdf["sentiment_zscore"] = 0.0

            sdf["is_anomaly"] = sdf["sentiment_zscore"].abs() > self.anomaly_z
            out_frames.append(sdf)

        if not out_frames:
            return sentiment_data.assign(
                sentiment_change=np.nan, sentiment_zscore=np.nan, is_anomaly=False
            )

        return pd.concat(out_frames, ignore_index=True)

    def detect_volume_anomalies(
        self,
        sentiment_data: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """
        Detect unusual mention-volume spikes.

        A sudden increase in chatter often precedes (or accompanies)
        large price moves.

        Returns:
            DataFrame augmented with [volume_ma, volume_zscore,
            volume_anomaly].
        """
        out_frames: List[pd.DataFrame] = []

        for symbol in sentiment_data["symbol"].unique():
            sdf = sentiment_data[sentiment_data["symbol"] == symbol].copy()
            sdf["volume_ma"] = sdf["volume"].rolling(lookback, min_periods=1).mean()
            sdf["volume_std"] = sdf["volume"].rolling(lookback, min_periods=1).std()

            sdf["volume_zscore"] = np.where(
                sdf["volume_std"] > 0,
                (sdf["volume"] - sdf["volume_ma"]) / sdf["volume_std"],
                0.0,
            )
            sdf["volume_anomaly"] = sdf["volume_zscore"] > self.anomaly_z
            out_frames.append(sdf)

        if not out_frames:
            return sentiment_data.assign(
                volume_ma=np.nan, volume_zscore=np.nan, volume_anomaly=False
            )

        return pd.concat(out_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, mentions: pd.DataFrame) -> pd.DataFrame:
        """
        End-to-end pipeline: aggregate -> smooth -> threshold -> signal.

        Signal logic (per symbol, per day):
            sentiment_ma >  threshold  AND volume >= volume_threshold => +1
            sentiment_ma < -threshold  AND volume >= volume_threshold => -1
            else                                                       =>  0

        Returns:
            DataFrame with [date, symbol, sentiment_score, volume,
            sentiment_ma, signal].
        """
        aggregated = self.aggregate_sentiment(mentions)

        # Volume filter
        vol_mask = aggregated["volume"] >= self.volume_threshold
        aggregated["signal"] = 0
        aggregated.loc[
            vol_mask & (aggregated["sentiment_ma"] > self.sentiment_threshold),
            "signal",
        ] = 1
        aggregated.loc[
            vol_mask & (aggregated["sentiment_ma"] < -self.sentiment_threshold),
            "signal",
        ] = -1

        return aggregated

    def compute_contrarian_signal(
        self,
        sentiment_data: pd.DataFrame,
        extreme_z: float = 3.0,
    ) -> pd.DataFrame:
        """
        Generate contrarian signals at sentiment extremes.

        When retail is *overwhelmingly* bullish (z > extreme_z),
        a mean-reversion short may be warranted, and vice-versa.

        Returns:
            DataFrame with [date, symbol, contrarian_signal].
        """
        df = self.detect_sentiment_anomalies(sentiment_data)
        df["contrarian_signal"] = 0
        df.loc[df["sentiment_zscore"] > extreme_z, "contrarian_signal"] = -1
        df.loc[df["sentiment_zscore"] < -extreme_z, "contrarian_signal"] = 1
        return df[["date", "symbol", "contrarian_signal"]].copy()
