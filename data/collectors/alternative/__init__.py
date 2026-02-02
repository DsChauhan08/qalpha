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
