"""
Alternative Data Collectors Module.

Free data sources for alternative data:
- SEC EDGAR: Financial filings (10-K, 10-Q, 8-K), insider trades
- FRED: Federal Reserve Economic Data
- Reddit: Social sentiment from investing subreddits
"""

from .sec_edgar import SECEdgarCollector
from .fred_data import FREDCollector
from .reddit_sentiment import RedditCollector, RedditSentimentFallback

__all__ = [
    "SECEdgarCollector",
    "FREDCollector",
    "RedditCollector",
    "RedditSentimentFallback",
]
from .minimal_loaders import (
    load_social_sentiment,
    load_options_sentiment,
    load_insider_trades,
    load_congress_trades,
)

__all__ = [
    "load_social_sentiment",
    "load_options_sentiment",
    "load_insider_trades",
    "load_congress_trades",
]
