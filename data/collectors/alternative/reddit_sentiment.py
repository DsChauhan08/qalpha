"""
Reddit Sentiment Collector

Scrapes Reddit for market sentiment from financial subreddits.
No API key required - uses public Reddit JSON endpoints.

Key Subreddits:
- r/wallstreetbets - Retail sentiment, meme stocks
- r/investing - General investing discussion
- r/stocks - Stock-specific discussion
- r/options - Options trading sentiment
"""

import os
import re
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Reddit post data."""

    id: str
    title: str
    selftext: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    author: str
    url: str
    tickers: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1


@dataclass
class SubredditConfig:
    """Configuration for a subreddit."""

    name: str
    weight: float  # Importance weight for sentiment
    post_limit: int = 100
    min_score: int = 10


# Subreddits for financial sentiment
FINANCIAL_SUBREDDITS: Dict[str, SubredditConfig] = {
    "wallstreetbets": SubredditConfig("wallstreetbets", weight=0.3, min_score=50),
    "investing": SubredditConfig("investing", weight=0.25, min_score=20),
    "stocks": SubredditConfig("stocks", weight=0.25, min_score=20),
    "options": SubredditConfig("options", weight=0.1, min_score=10),
    "stockmarket": SubredditConfig("stockmarket", weight=0.1, min_score=10),
}


class TickerExtractor:
    """Extract stock tickers from text."""

    # Common false positives (words that look like tickers)
    FALSE_POSITIVES = {
        "A",
        "I",
        "AM",
        "PM",
        "CEO",
        "CFO",
        "CTO",
        "COO",
        "IPO",
        "ATH",
        "ATL",
        "DD",
        "EPS",
        "ETF",
        "FDA",
        "FD",
        "FOMO",
        "FUD",
        "GDP",
        "IMO",
        "YOLO",
        "LOL",
        "OMG",
        "WTF",
        "NYSE",
        "SEC",
        "USD",
        "USA",
        "UK",
        "EU",
        "GDP",
        "IT",
        "AI",
        "ML",
        "API",
        "UI",
        "UX",
        "CEO",
        "PR",
        "HR",
        "QA",
        "FAQ",
        "HODL",
        "BTFD",
        "TL",
        "DR",
        "TLDR",
        "OP",
        "OC",
        "EDIT",
        "RIP",
        "WSB",
        "IMO",
        "IMHO",
        "TBH",
        "SMH",
        "IDK",
        "IRL",
        "AMA",
        "ELI5",
        "IIRC",
        "PS",
        "PSA",
        "FYI",
        "ASAP",
        "EOD",
        "EOW",
        "EOM",
        "YTD",
        "QOQ",
        "MOM",
        "IV",
        "DTE",
        "OTM",
        "ITM",
        "ATM",
        "P",
        "C",
        "PE",
        "PB",
        "ROI",
        "ROE",
        "EV",
        "DCF",
        "TA",
        "MA",
        "RSI",
        "MACD",
        "BB",
        "SI",
        "VS",
        "THE",
        "FOR",
    }

    # Known valid tickers (popular ones)
    KNOWN_TICKERS = {
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "AMD",
        "INTC",
        "NFLX",
        "DIS",
        "PYPL",
        "SQ",
        "SHOP",
        "ROKU",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "ARKK",
        "GME",
        "AMC",
        "BB",
        "NOK",
        "PLTR",
        "WISH",
        "CLOV",
        "SPCE",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "C",
        "WFC",
        "XOM",
        "CVX",
        "OXY",
        "BP",
        "PFE",
        "MRNA",
        "JNJ",
        "ABBV",
        "BA",
        "LMT",
        "RTX",
        "GE",
        "F",
        "GM",
        "RIVN",
        "LCID",
        "COIN",
        "HOOD",
        "SOFI",
        "NIO",
        "XPEV",
        "LI",
        "BABA",
        "JD",
        "PDD",
    }

    # Ticker pattern: $TICKER or standalone 2-5 letter uppercase
    TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")

    def extract(self, text: str) -> List[str]:
        """
        Extract stock tickers from text.

        Args:
            text: Text to extract tickers from

        Returns:
            List of ticker symbols
        """
        tickers = set()

        for match in self.TICKER_PATTERN.finditer(text):
            # $TICKER format (group 1) or standalone (group 2)
            ticker = match.group(1) or match.group(2)

            if not ticker:
                continue

            # Skip false positives
            if ticker in self.FALSE_POSITIVES:
                continue

            # Include known tickers always
            if ticker in self.KNOWN_TICKERS:
                tickers.add(ticker)
                continue

            # For unknown tickers, require $ prefix for confidence
            if match.group(1):  # Had $ prefix
                tickers.add(ticker)

        return list(tickers)


class SimpleSentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer for financial text.

    No ML dependencies required.
    """

    # Bullish keywords and phrases
    BULLISH_WORDS = {
        "buy",
        "long",
        "calls",
        "bull",
        "bullish",
        "moon",
        "rocket",
        "undervalued",
        "oversold",
        "breakout",
        "support",
        "accumulate",
        "squeeze",
        "gamma",
        "tendies",
        "gains",
        "profit",
        "winner",
        "strong",
        "growth",
        "beat",
        "upgrade",
        "outperform",
        "buy",
        "yolo",
        "diamond",
        "hands",
        "hodl",
        "hold",
        "dip",
        "discount",
        "opportunity",
        "potential",
        "upside",
        "rally",
        "recovery",
        "positive",
        "optimistic",
        "confident",
        "excited",
        "love",
    }

    # Bearish keywords and phrases
    BEARISH_WORDS = {
        "sell",
        "short",
        "puts",
        "bear",
        "bearish",
        "crash",
        "dump",
        "overvalued",
        "overbought",
        "breakdown",
        "resistance",
        "distribute",
        "baghold",
        "loss",
        "losses",
        "loser",
        "weak",
        "decline",
        "miss",
        "downgrade",
        "underperform",
        "avoid",
        "warning",
        "risk",
        "danger",
        "paper",
        "scared",
        "fear",
        "worried",
        "nervous",
        "panic",
        "hate",
        "terrible",
        "awful",
        "worst",
        "scam",
        "fraud",
        "manipulation",
        "negative",
        "pessimistic",
        "concerned",
        "bubble",
        "overextended",
    }

    # Intensity modifiers
    INTENSIFIERS = {
        "very",
        "extremely",
        "super",
        "incredibly",
        "absolutely",
        "definitely",
        "totally",
        "completely",
        "huge",
        "massive",
    }

    # Negation words
    NEGATIONS = {
        "not",
        "no",
        "never",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "can't",
        "cannot",
        "without",
        "lack",
    }

    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (bearish) to 1 (bullish)
            confidence: 0 to 1
        """
        if not text:
            return 0.0, 0.0

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        if not words:
            return 0.0, 0.0

        bullish_count = 0
        bearish_count = 0
        intensity_boost = 1.0

        # Track negation window (next 3 words after negation)
        negation_window = 0

        for i, word in enumerate(words):
            # Check for negation
            if word in self.NEGATIONS:
                negation_window = 3
                continue

            # Check for intensifiers
            if word in self.INTENSIFIERS:
                intensity_boost = 1.5
                continue

            # Check sentiment
            is_bullish = word in self.BULLISH_WORDS
            is_bearish = word in self.BEARISH_WORDS

            # Apply negation (flip sentiment)
            if negation_window > 0:
                is_bullish, is_bearish = is_bearish, is_bullish
                negation_window -= 1

            if is_bullish:
                bullish_count += intensity_boost
            elif is_bearish:
                bearish_count += intensity_boost

            intensity_boost = 1.0  # Reset

        total = bullish_count + bearish_count

        if total == 0:
            return 0.0, 0.0

        # Calculate sentiment (-1 to 1)
        sentiment = (bullish_count - bearish_count) / total

        # Confidence based on signal strength
        confidence = min(1.0, total / 10)  # More keywords = more confidence

        return sentiment, confidence


class RedditCollector:
    """
    Collector for Reddit financial sentiment.

    Uses public Reddit JSON API (no authentication required).
    """

    BASE_URL = "https://www.reddit.com"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        rate_limit: float = 2.0,  # Reddit rate limit
        user_agent: str = "QuantumAlpha/1.0",
    ):
        """
        Initialize Reddit collector.

        Args:
            cache_dir: Directory for caching data
            rate_limit: Minimum seconds between requests
            user_agent: User agent string for requests
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data_cache", "reddit"
        )
        self.rate_limit = rate_limit
        self.user_agent = user_agent
        self._last_request = 0.0

        self.ticker_extractor = TickerExtractor()
        self.sentiment_analyzer = SimpleSentimentAnalyzer()

        os.makedirs(self.cache_dir, exist_ok=True)

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _make_request(self, url: str) -> Optional[Dict]:
        """
        Make request to Reddit JSON API.

        Args:
            url: Full URL to fetch

        Returns:
            JSON response or None on error
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library required: pip install requests")
            return None

        self._rate_limit_wait()

        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429:
                logger.warning("Reddit rate limited, waiting...")
                time.sleep(60)
                return self._make_request(url)

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Reddit request failed: {e}")
            return None

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 100,
        time_filter: str = "day",
    ) -> List[RedditPost]:
        """
        Fetch posts from a subreddit.

        Args:
            subreddit: Subreddit name
            sort: Sort method (hot, new, top, rising)
            limit: Number of posts to fetch
            time_filter: Time filter for top (hour, day, week, month, year, all)

        Returns:
            List of RedditPost objects
        """
        url = f"{self.BASE_URL}/r/{subreddit}/{sort}.json"
        params = f"?limit={limit}&t={time_filter}&raw_json=1"

        data = self._make_request(url + params)

        if not data or "data" not in data:
            return []

        posts = []

        for child in data["data"].get("children", []):
            try:
                post_data = child["data"]

                # Extract text content
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                full_text = f"{title} {selftext}"

                # Extract tickers
                tickers = self.ticker_extractor.extract(full_text)

                # Analyze sentiment
                sentiment, confidence = self.sentiment_analyzer.analyze(full_text)

                post = RedditPost(
                    id=post_data.get("id", ""),
                    title=title,
                    selftext=selftext[:1000],  # Truncate
                    subreddit=subreddit,
                    score=post_data.get("score", 0),
                    upvote_ratio=post_data.get("upvote_ratio", 0.5),
                    num_comments=post_data.get("num_comments", 0),
                    created_utc=post_data.get("created_utc", 0),
                    author=post_data.get("author", ""),
                    url=post_data.get("url", ""),
                    tickers=tickers,
                    sentiment=sentiment * confidence,  # Weight by confidence
                )

                posts.append(post)

            except Exception as e:
                logger.debug(f"Failed to parse post: {e}")
                continue

        return posts

    def fetch_all_subreddits(
        self, subreddits: Optional[Dict[str, SubredditConfig]] = None, sort: str = "hot"
    ) -> List[RedditPost]:
        """
        Fetch posts from all configured subreddits.

        Args:
            subreddits: Subreddit configurations (defaults to FINANCIAL_SUBREDDITS)
            sort: Sort method

        Returns:
            List of all posts
        """
        if subreddits is None:
            subreddits = FINANCIAL_SUBREDDITS

        all_posts = []

        for name, config in subreddits.items():
            logger.info(f"Fetching r/{name}...")
            posts = self.fetch_subreddit_posts(name, sort=sort, limit=config.post_limit)

            # Filter by minimum score
            posts = [p for p in posts if p.score >= config.min_score]

            all_posts.extend(posts)
            logger.info(f"Got {len(posts)} posts from r/{name}")

        return all_posts

    def aggregate_ticker_sentiment(
        self,
        posts: List[RedditPost],
        subreddits: Optional[Dict[str, SubredditConfig]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate sentiment by ticker.

        Args:
            posts: List of Reddit posts
            subreddits: Subreddit configs for weighting

        Returns:
            DataFrame with ticker sentiment metrics
        """
        if subreddits is None:
            subreddits = FINANCIAL_SUBREDDITS

        ticker_data = defaultdict(
            lambda: {
                "mentions": 0,
                "sentiment_sum": 0.0,
                "score_sum": 0,
                "comments_sum": 0,
                "posts": [],
            }
        )

        for post in posts:
            # Get subreddit weight
            weight = subreddits.get(
                post.subreddit, SubredditConfig(post.subreddit, 0.1)
            ).weight

            # Weight by engagement (log scale)
            engagement_weight = np.log1p(post.score + post.num_comments)

            for ticker in post.tickers:
                data = ticker_data[ticker]
                data["mentions"] += 1
                data["sentiment_sum"] += post.sentiment * weight * engagement_weight
                data["score_sum"] += post.score
                data["comments_sum"] += post.num_comments
                data["posts"].append(post.id)

        # Convert to DataFrame
        records = []
        for ticker, data in ticker_data.items():
            if data["mentions"] > 0:
                avg_sentiment = data["sentiment_sum"] / data["mentions"]
                records.append(
                    {
                        "ticker": ticker,
                        "mentions": data["mentions"],
                        "avg_sentiment": avg_sentiment,
                        "total_score": data["score_sum"],
                        "total_comments": data["comments_sum"],
                        "engagement": data["score_sum"] + data["comments_sum"],
                    }
                )

        df = pd.DataFrame(records)

        if df.empty:
            return df

        # Sort by mentions
        df = df.sort_values("mentions", ascending=False)

        # Add sentiment ranking
        df["sentiment_rank"] = df["avg_sentiment"].rank(pct=True)

        # Add mention z-score
        df["mention_zscore"] = (df["mentions"] - df["mentions"].mean()) / df[
            "mentions"
        ].std()

        return df

    def calculate_market_sentiment(
        self,
        posts: List[RedditPost],
        subreddits: Optional[Dict[str, SubredditConfig]] = None,
    ) -> Dict[str, float]:
        """
        Calculate overall market sentiment from posts.

        Args:
            posts: List of Reddit posts
            subreddits: Subreddit configs for weighting

        Returns:
            Dictionary with sentiment metrics
        """
        if subreddits is None:
            subreddits = FINANCIAL_SUBREDDITS

        if not posts:
            return {
                "overall_sentiment": 0.0,
                "bullish_ratio": 0.5,
                "bearish_ratio": 0.5,
                "neutral_ratio": 0.0,
                "total_posts": 0,
                "avg_engagement": 0.0,
            }

        sentiments = []
        weights = []

        for post in posts:
            weight = subreddits.get(
                post.subreddit, SubredditConfig(post.subreddit, 0.1)
            ).weight

            # Weight by engagement
            engagement = np.log1p(post.score + post.num_comments)

            sentiments.append(post.sentiment)
            weights.append(weight * engagement)

        sentiments = np.array(sentiments)
        weights = np.array(weights)

        # Weighted average sentiment
        if weights.sum() > 0:
            overall = np.average(sentiments, weights=weights)
        else:
            overall = sentiments.mean()

        # Sentiment distribution
        bullish = (sentiments > 0.1).sum() / len(sentiments)
        bearish = (sentiments < -0.1).sum() / len(sentiments)
        neutral = 1 - bullish - bearish

        return {
            "overall_sentiment": float(overall),
            "bullish_ratio": float(bullish),
            "bearish_ratio": float(bearish),
            "neutral_ratio": float(neutral),
            "total_posts": len(posts),
            "avg_engagement": float(np.mean([p.score + p.num_comments for p in posts])),
        }

    def get_trending_tickers(
        self, min_mentions: int = 3, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get currently trending tickers on Reddit.

        Args:
            min_mentions: Minimum mentions to include
            top_n: Number of top tickers to return

        Returns:
            DataFrame with trending tickers
        """
        # Fetch recent posts
        posts = self.fetch_all_subreddits()

        # Aggregate by ticker
        df = self.aggregate_ticker_sentiment(posts)

        if df.empty:
            return df

        # Filter and sort
        df = df[df["mentions"] >= min_mentions]
        df = df.nlargest(top_n, "mentions")

        return df

    def get_ticker_sentiment(
        self, ticker: str, subreddits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get sentiment for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            subreddits: Subreddits to search (defaults to all)

        Returns:
            Dictionary with ticker sentiment data
        """
        if subreddits is None:
            subreddits = list(FINANCIAL_SUBREDDITS.keys())

        all_posts = []

        for sub in subreddits:
            # Search subreddit for ticker
            url = f"{self.BASE_URL}/r/{sub}/search.json"
            params = f"?q={ticker}&restrict_sr=1&sort=relevance&t=week&limit=50"

            data = self._make_request(url + params)

            if data and "data" in data:
                for child in data["data"].get("children", []):
                    post_data = child["data"]

                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    full_text = f"{title} {selftext}"

                    # Verify ticker is actually mentioned
                    tickers = self.ticker_extractor.extract(full_text)
                    if ticker not in tickers:
                        continue

                    sentiment, confidence = self.sentiment_analyzer.analyze(full_text)

                    post = RedditPost(
                        id=post_data.get("id", ""),
                        title=title,
                        selftext=selftext[:500],
                        subreddit=sub,
                        score=post_data.get("score", 0),
                        upvote_ratio=post_data.get("upvote_ratio", 0.5),
                        num_comments=post_data.get("num_comments", 0),
                        created_utc=post_data.get("created_utc", 0),
                        author=post_data.get("author", ""),
                        url=post_data.get("url", ""),
                        tickers=[ticker],
                        sentiment=sentiment * confidence,
                    )

                    all_posts.append(post)

        if not all_posts:
            return {"ticker": ticker, "mentions": 0, "sentiment": 0.0, "posts": []}

        sentiments = [p.sentiment for p in all_posts]

        return {
            "ticker": ticker,
            "mentions": len(all_posts),
            "sentiment": float(np.mean(sentiments)),
            "sentiment_std": float(np.std(sentiments)),
            "total_score": sum(p.score for p in all_posts),
            "total_comments": sum(p.num_comments for p in all_posts),
            "posts": [
                {"title": p.title, "score": p.score, "sentiment": p.sentiment}
                for p in sorted(all_posts, key=lambda x: x.score, reverse=True)[:10]
            ],
        }

    def save_snapshot(self, filename: Optional[str] = None) -> str:
        """
        Save current sentiment snapshot to cache.

        Args:
            filename: Optional filename (defaults to timestamp)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = (
                f"reddit_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        filepath = os.path.join(self.cache_dir, filename)

        # Fetch all data
        posts = self.fetch_all_subreddits()
        ticker_sentiment = self.aggregate_ticker_sentiment(posts)
        market_sentiment = self.calculate_market_sentiment(posts)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": market_sentiment,
            "ticker_sentiment": ticker_sentiment.to_dict("records")
            if not ticker_sentiment.empty
            else [],
            "post_count": len(posts),
        }

        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"Saved snapshot to {filepath}")
        return filepath


class RedditSentimentFallback:
    """
    Fallback when Reddit unavailable.

    Returns neutral sentiment.
    """

    def get_default_sentiment(self) -> Dict[str, float]:
        """Get default neutral sentiment."""
        return {
            "overall_sentiment": 0.0,
            "bullish_ratio": 0.33,
            "bearish_ratio": 0.33,
            "neutral_ratio": 0.34,
            "total_posts": 0,
            "avg_engagement": 0.0,
        }

    def get_ticker_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get default ticker sentiment."""
        return {"ticker": ticker, "mentions": 0, "sentiment": 0.0, "posts": []}


def create_reddit_collector(**kwargs) -> RedditCollector:
    """
    Factory function to create Reddit collector.

    Args:
        **kwargs: Arguments for RedditCollector

    Returns:
        RedditCollector instance
    """
    return RedditCollector(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = RedditCollector()

    # Test trending tickers
    print("Fetching trending tickers...")
    trending = collector.get_trending_tickers(min_mentions=2, top_n=10)
    if not trending.empty:
        print("\nTop Trending Tickers:")
        print(
            trending[["ticker", "mentions", "avg_sentiment", "engagement"]].to_string()
        )

    # Test specific ticker
    print("\n\nFetching NVDA sentiment...")
    nvda = collector.get_ticker_sentiment("NVDA")
    print(f"NVDA: {nvda['mentions']} mentions, sentiment: {nvda['sentiment']:.3f}")

    # Test market sentiment
    print("\n\nCalculating market sentiment...")
    posts = collector.fetch_all_subreddits()
    market = collector.calculate_market_sentiment(posts)
    print(f"Market sentiment: {market['overall_sentiment']:.3f}")
    print(
        f"Bullish: {market['bullish_ratio']:.1%}, Bearish: {market['bearish_ratio']:.1%}"
    )
