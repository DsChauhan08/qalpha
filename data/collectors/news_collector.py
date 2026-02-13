"""
News data collector for Quantum Alpha.

Fetches financial news from free sources and builds a historical
sentiment time-series for LSTM training.

Sources (all free, no API key required for basic use):
- Yahoo Finance RSS feeds (via yfinance)
- GDELT Project (public news database)
- Synthetic historical sentiment from price-implied events

For backtesting, we generate sentiment proxies from historical
price action and volatility events, since free historical news
APIs are limited.  When live, we fetch real headlines.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NewsCollector:
    """
    Collect financial news headlines and convert to sentiment features.

    For historical backtesting, uses a price-implied sentiment proxy
    (since free historical headline APIs are limited).
    For live/paper trading, fetches real headlines via yfinance news.
    """

    def __init__(self, use_cache: bool = True, cache_dir: str = None):
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Live news fetching (for paper/live trading)
    # ------------------------------------------------------------------

    def fetch_live_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Fetch recent news headlines for a symbol via yfinance.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            max_articles: Maximum articles to fetch

        Returns:
            List of dicts with 'title', 'summary', 'source', 'published_at'
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news or []

            articles = []
            for item in news[:max_articles]:
                articles.append(
                    {
                        "title": item.get("title", ""),
                        "summary": item.get("summary", item.get("title", "")),
                        "source": item.get("publisher", "unknown"),
                        "published_at": datetime.fromtimestamp(
                            item.get("providerPublishTime", time.time())
                        ).isoformat(),
                        "url": item.get("link", ""),
                    }
                )

            return articles

        except Exception as e:
            logger.warning("Failed to fetch live news for %s: %s", symbol, e)
            return []

    # ------------------------------------------------------------------
    # Historical sentiment proxy (for backtesting)
    # ------------------------------------------------------------------

    def build_historical_sentiment(
        self,
        price_df: pd.DataFrame,
        symbol: str = "SPY",
    ) -> pd.DataFrame:
        """
        Build a historical sentiment time-series from price data.

        This creates a *proxy* for news sentiment using observable
        market signals that correlate with news flow:

        1. Overnight gap (pre-market news reaction)
        2. Intraday volatility spike (breaking news)
        3. Volume surprise (institutional activity after news)
        4. Return momentum shift (narrative change)
        5. VIX-implied fear (for indices)

        These features are LAGGED by 1 bar to prevent lookahead bias -
        the model sees yesterday's sentiment proxy to predict today's move.

        Args:
            price_df: DataFrame with OHLCV columns (DatetimeIndex)
            symbol: Ticker symbol

        Returns:
            DataFrame with sentiment proxy features
        """
        df = price_df.copy()

        # Ensure we have required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # 1. Overnight gap (proxy for overnight news)
        #    Large gaps suggest significant news broke
        df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["overnight_gap_abs"] = df["overnight_gap"].abs()
        df["overnight_gap_zscore"] = (
            df["overnight_gap"] - df["overnight_gap"].rolling(20).mean()
        ) / df["overnight_gap"].rolling(20).std().clip(lower=1e-8)

        # 2. Intraday range surprise (breaking news causes volatility)
        df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
        df["range_surprise"] = (
            df["intraday_range"]
            / df["intraday_range"].rolling(20).mean().clip(lower=1e-8)
        ) - 1.0

        # 3. Volume surprise (news drives volume)
        df["volume_sma20"] = df["volume"].rolling(20).mean()
        df["volume_surprise"] = (df["volume"] / df["volume_sma20"].clip(lower=1)) - 1.0

        # 4. Return acceleration (narrative shift)
        df["returns"] = df["close"].pct_change()
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_accel"] = df["returns"] - df["returns"].shift(1)

        # 5. Composite sentiment proxy
        #    Positive gap + high volume = positive news
        #    Negative gap + high volume = negative news
        #    High range + low directional move = uncertainty
        df["sentiment_proxy"] = np.where(
            df["volume_surprise"] > 0.5,  # High volume
            np.sign(df["overnight_gap"]) * df["overnight_gap_abs"] * 10,
            df["overnight_gap"] * 5,
        )
        df["sentiment_proxy"] = df["sentiment_proxy"].clip(-3, 3)

        # 6. Sentiment momentum (rolling sentiment trend)
        df["sentiment_ma3"] = df["sentiment_proxy"].rolling(3).mean()
        df["sentiment_ma7"] = df["sentiment_proxy"].rolling(7).mean()
        df["sentiment_momentum"] = df["sentiment_ma3"] - df["sentiment_ma7"]

        # 7. News intensity proxy (high vol + high range = major news day)
        df["news_intensity"] = df["volume_surprise"].clip(0) * df[
            "range_surprise"
        ].clip(0)
        df["news_intensity"] = df["news_intensity"].clip(0, 10)

        # 8. Fear/greed proxy
        df["fear_greed"] = np.where(
            df["returns"] < 0,
            -df["range_surprise"].clip(0) * df["volume_surprise"].clip(0),
            df["range_surprise"].clip(0) * df["volume_surprise"].clip(0),
        )

        # 9. Mean-reversion signal: RSI-like zscore of returns
        ret_mean = df["returns"].rolling(14).mean()
        ret_std = df["returns"].rolling(14).std().clip(lower=1e-8)
        df["return_zscore"] = (df["returns"] - ret_mean) / ret_std

        # 10. Regime detection: rolling std of returns (vol clustering)
        df["vol_regime"] = df["returns"].rolling(10).std() / df["returns"].rolling(
            40
        ).std().clip(lower=1e-8)

        # 11. Trend strength: directional consistency
        pos_days = (df["returns"] > 0).rolling(10).mean()
        df["trend_strength"] = (pos_days - 0.5) * 2  # [-1, 1]

        # 12. Volume-price divergence (smart money)
        df["vol_price_div"] = df["volume_surprise"] * np.sign(-df["returns"])

        # 13. Gap fill tendency
        df["gap_fill"] = np.where(
            df["overnight_gap"] > 0,
            (df["close"] < df["open"]).astype(float),
            (df["close"] > df["open"]).astype(float),
        )
        df["gap_fill_rate"] = df["gap_fill"].rolling(20).mean()

        # Select sentiment features
        sentiment_cols = [
            "overnight_gap",
            "overnight_gap_zscore",
            "range_surprise",
            "volume_surprise",
            "returns_accel",
            "sentiment_proxy",
            "sentiment_ma3",
            "sentiment_ma7",
            "sentiment_momentum",
            "news_intensity",
            "fear_greed",
            "return_zscore",
            "vol_regime",
            "trend_strength",
            "vol_price_div",
            "gap_fill_rate",
        ]

        return df[sentiment_cols].copy()

    def build_training_features(
        self,
        price_df: pd.DataFrame,
        symbol: str = "SPY",
    ) -> pd.DataFrame:
        """
        Build complete feature set for News-Driven LSTM training.

        Combines:
        - Sentiment proxy features (16 features)
        - Minimal price context (5 features)
        - Target variable (forward return)

        All features are LAGGED appropriately to prevent lookahead.

        Args:
            price_df: DataFrame with OHLCV columns
            symbol: Ticker symbol

        Returns:
            DataFrame ready for LSTM windowing
        """
        df = price_df.copy()

        # Build sentiment features
        sentiment_df = self.build_historical_sentiment(df, symbol)

        # Minimal price context (not the primary signal, just context)
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = df["returns"].rolling(20).std()
        df["price_vs_sma20"] = df["close"] / df["close"].rolling(20).mean() - 1
        df["price_vs_sma50"] = df["close"] / df["close"].rolling(50).mean() - 1

        price_context = df[
            [
                "returns",
                "log_returns",
                "volatility_20d",
                "price_vs_sma20",
                "price_vs_sma50",
            ]
        ].copy()

        # Combine features
        features = pd.concat([sentiment_df, price_context], axis=1)

        # Add target: next-day return
        features["target_1d"] = df["close"].pct_change().shift(-1)
        features["target_5d"] = df["close"].pct_change(5).shift(-5)

        # Add close price for return calculations in windowing
        features["close"] = df["close"]

        # LAG all sentiment features by 1 to prevent lookahead
        lag_cols = list(sentiment_df.columns)
        for col in lag_cols:
            features[col] = features[col].shift(1)

        # Drop NaN rows
        features = features.dropna()

        return features


SENTIMENT_FEATURE_COLS = [
    # Sentiment proxy features (16)
    "overnight_gap",
    "overnight_gap_zscore",
    "range_surprise",
    "volume_surprise",
    "returns_accel",
    "sentiment_proxy",
    "sentiment_ma3",
    "sentiment_ma7",
    "sentiment_momentum",
    "news_intensity",
    "fear_greed",
    "return_zscore",
    "vol_regime",
    "trend_strength",
    "vol_price_div",
    "gap_fill_rate",
    # Price context features (5)
    "returns",
    "log_returns",
    "volatility_20d",
    "price_vs_sma20",
    "price_vs_sma50",
]
