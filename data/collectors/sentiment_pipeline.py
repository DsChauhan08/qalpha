"""
Real News Sentiment Pipeline for Quantum Alpha.

Fetches financial news headlines from multiple FREE sources, scores them
with FinBERT, stores in SQLite, and produces daily sentiment features
for LSTM training.

Sources (all free, no API key):
  1. Google News RSS — ~100 articles/query, up to 1-year lookback
  2. Yahoo Finance (yfinance) — ~10 articles/ticker, recent only
  3. GDELT Project — 250 articles/query, ~3 months lookback (rate-limited)

Features produced per day (to replace price-proxy sentiment):
  - mean_sentiment       : mean FinBERT score across articles [-1, 1]
  - sentiment_std        : disagreement among articles
  - max_positive         : strongest bullish headline
  - max_negative         : strongest bearish headline
  - bullish_ratio        : fraction of positive articles
  - bearish_ratio        : fraction of negative articles
  - neutral_ratio        : fraction of neutral articles
  - n_articles           : article count (news volume)
  - news_volume_surprise : n_articles vs rolling 7-day mean
  - weighted_sentiment   : source-weighted sentiment
  - sentiment_momentum_3d: 3-day sentiment change
  - sentiment_momentum_7d: 7-day sentiment change
  - sentiment_ma3        : 3-day rolling mean sentiment
  - sentiment_ma7        : 7-day rolling mean sentiment
  - sentiment_dispersion : max_positive - max_negative (polarity range)
  - high_confidence_sent : mean sentiment of articles with confidence > 0.7
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_DIR = Path(__file__).parent.parent.parent / "data_store"
DB_PATH = DB_DIR / "sentiment.db"

# Symbol -> search queries for Google News
SYMBOL_QUERIES = {
    "SPY": ['"S&P 500"', '"SPY stock"', '"stock market"'],
    "QQQ": ['"Nasdaq"', '"QQQ stock"', '"tech stocks"'],
    "AAPL": ['"Apple stock"', '"AAPL"', '"Apple Inc"'],
    "MSFT": ['"Microsoft stock"', '"MSFT"', '"Microsoft Corp"'],
    "AMZN": ['"Amazon stock"', '"AMZN"', '"Amazon.com"'],
    "GOOGL": ['"Google stock"', '"GOOGL"', '"Alphabet Inc"'],
    "TSLA": ['"Tesla stock"', '"TSLA"', '"Tesla Inc"'],
    "META": ['"Meta stock"', '"META"', '"Meta Platforms"'],
    "NVDA": ['"Nvidia stock"', '"NVDA"', '"Nvidia Corp"'],
}

# Source credibility weights
SOURCE_WEIGHTS = {
    "Reuters": 1.3,
    "Bloomberg": 1.3,
    "CNBC": 1.1,
    "Wall Street Journal": 1.2,
    "Financial Times": 1.2,
    "MarketWatch": 1.0,
    "Yahoo Finance": 0.9,
    "Benzinga": 0.8,
    "Seeking Alpha": 0.7,
    "Motley Fool": 0.6,
    "InvestorPlace": 0.6,
}

# Real sentiment features that replace price-proxy features
REAL_SENTIMENT_FEATURE_COLS = [
    "mean_sentiment",
    "sentiment_std",
    "max_positive",
    "max_negative",
    "bullish_ratio",
    "bearish_ratio",
    "neutral_ratio",
    "n_articles",
    "news_volume_surprise",
    "weighted_sentiment",
    "sentiment_momentum_3d",
    "sentiment_momentum_7d",
    "sentiment_ma3",
    "sentiment_ma7",
    "sentiment_dispersion",
    "high_confidence_sent",
]

# Price context features (same as before)
PRICE_CONTEXT_COLS = [
    "returns",
    "log_returns",
    "volatility_20d",
    "price_vs_sma20",
    "price_vs_sma50",
]

# Combined feature list for the LSTM
ALL_FEATURE_COLS = REAL_SENTIMENT_FEATURE_COLS + PRICE_CONTEXT_COLS


# ---------------------------------------------------------------------------
# SQLite Database
# ---------------------------------------------------------------------------


class SentimentDB:
    """SQLite storage for scored headlines."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT,
                    published_date TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    url TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    sentiment_confidence REAL,
                    prob_positive REAL,
                    prob_negative REAL,
                    prob_neutral REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_symbol_date
                ON articles(symbol, published_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_date
                ON articles(published_date)
            """)

    def _article_id(self, title: str, source: str, pub_date: str) -> str:
        """Generate deterministic ID for deduplication."""
        key = f"{title.strip().lower()}|{source.lower()}|{pub_date[:10]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def insert_articles(self, articles: List[Dict]) -> int:
        """Insert articles, skipping duplicates. Returns count of new articles."""
        inserted = 0
        with sqlite3.connect(str(self.db_path)) as conn:
            for a in articles:
                aid = self._article_id(
                    a["title"], a.get("source", ""), a.get("published_date", "")
                )
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO articles
                        (id, symbol, title, source, published_date, fetched_at,
                         data_source, url, sentiment_score, sentiment_label,
                         sentiment_confidence, prob_positive, prob_negative, prob_neutral)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            aid,
                            a.get("symbol", ""),
                            a["title"],
                            a.get("source", "unknown"),
                            a.get("published_date", ""),
                            datetime.now().isoformat(),
                            a.get("data_source", "unknown"),
                            a.get("url", ""),
                            a.get("sentiment_score"),
                            a.get("sentiment_label"),
                            a.get("sentiment_confidence"),
                            a.get("prob_positive"),
                            a.get("prob_negative"),
                            a.get("prob_neutral"),
                        ),
                    )
                    if conn.total_changes > 0:
                        inserted += 1
                except sqlite3.IntegrityError:
                    pass
        return inserted

    def get_articles(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Retrieve scored articles for a symbol and date range."""
        query = "SELECT * FROM articles WHERE symbol = ?"
        params: list = [symbol]

        if start_date:
            query += " AND published_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND published_date <= ?"
            params.append(end_date)

        query += " ORDER BY published_date ASC"

        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_unscored_articles(self, limit: int = 500) -> pd.DataFrame:
        """Get articles that haven't been scored yet."""
        query = """
            SELECT * FROM articles
            WHERE sentiment_score IS NULL
            ORDER BY published_date ASC
            LIMIT ?
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            return pd.read_sql_query(query, conn, params=[limit])

    def update_scores(self, scores: List[Dict]):
        """Update sentiment scores for articles by id."""
        with sqlite3.connect(str(self.db_path)) as conn:
            for s in scores:
                conn.execute(
                    """
                    UPDATE articles SET
                        sentiment_score = ?,
                        sentiment_label = ?,
                        sentiment_confidence = ?,
                        prob_positive = ?,
                        prob_negative = ?,
                        prob_neutral = ?
                    WHERE id = ?
                    """,
                    (
                        s["sentiment_score"],
                        s["sentiment_label"],
                        s["sentiment_confidence"],
                        s["prob_positive"],
                        s["prob_negative"],
                        s["prob_neutral"],
                        s["id"],
                    ),
                )

    def count_articles(self, symbol: str = None) -> int:
        """Count total articles, optionally by symbol."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if symbol:
                r = conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE symbol = ?", (symbol,)
                )
            else:
                r = conn.execute("SELECT COUNT(*) FROM articles")
            return r.fetchone()[0]

    def date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Get earliest and latest article dates for a symbol."""
        with sqlite3.connect(str(self.db_path)) as conn:
            r = conn.execute(
                "SELECT MIN(published_date), MAX(published_date) FROM articles WHERE symbol = ?",
                (symbol,),
            )
            row = r.fetchone()
            return (row[0], row[1]) if row else (None, None)


# ---------------------------------------------------------------------------
# Multi-Source News Fetcher
# ---------------------------------------------------------------------------


class MultiSourceNewsFetcher:
    """
    Fetches news headlines from Google News RSS, Yahoo Finance, and GDELT.
    All sources are free and require no API keys.
    """

    def __init__(self, request_delay: float = 1.0):
        self.request_delay = request_delay  # seconds between requests
        self._last_request_time = 0.0

    def _throttle(self):
        """Rate-limit requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Google News RSS
    # ------------------------------------------------------------------

    def fetch_google_news(
        self,
        symbol: str,
        lookback_period: str = "1y",
        max_per_query: int = 100,
    ) -> List[Dict]:
        """
        Fetch headlines from Google News RSS.

        Args:
            symbol: Ticker symbol (used to look up search queries)
            lookback_period: e.g. "1y", "6m", "3m", "1m", "7d"
            max_per_query: Max articles per query (Google caps at ~100)

        Returns:
            List of article dicts
        """
        import urllib.request

        queries = SYMBOL_QUERIES.get(symbol, [f'"{symbol} stock"'])
        all_articles = []

        for query in queries:
            self._throttle()
            encoded = quote_plus(query + f" when:{lookback_period}")
            url = (
                f"https://news.google.com/rss/search?q={encoded}"
                f"&hl=en-US&gl=US&ceid=US:en"
            )

            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; QuantumAlpha/1.0)"
                    },
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    xml_data = resp.read()

                root = ET.fromstring(xml_data)
                items = root.findall(".//item")

                for item in items[:max_per_query]:
                    title = item.findtext("title", "").strip()
                    if not title:
                        continue

                    # Parse source from title (Google News format: "Title - Source")
                    source = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        if len(parts) == 2 and len(parts[1]) < 50:
                            title = parts[0].strip()
                            source = parts[1].strip()

                    pub_date = item.findtext("pubDate", "")
                    pub_date_parsed = self._parse_rss_date(pub_date)

                    all_articles.append(
                        {
                            "title": title,
                            "source": source,
                            "published_date": pub_date_parsed,
                            "url": item.findtext("link", ""),
                            "symbol": symbol,
                            "data_source": "google_news",
                        }
                    )

                logger.info(
                    "Google News: %d articles for query '%s'", len(items), query
                )

            except Exception as e:
                logger.warning("Google News fetch failed for '%s': %s", query, e)

        return all_articles

    # ------------------------------------------------------------------
    # Yahoo Finance
    # ------------------------------------------------------------------

    def fetch_yahoo_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Fetch recent news from Yahoo Finance via yfinance.

        Returns:
            List of article dicts
        """
        try:
            import yfinance as yf

            self._throttle()
            ticker = yf.Ticker(symbol)
            news = ticker.news or []

            articles = []
            for item in news[:max_articles]:
                # yfinance 0.2.66 uses nested content structure
                content = item.get("content", item)
                if isinstance(content, dict):
                    title = content.get("title", item.get("title", ""))
                    summary = content.get("summary", "")
                    pub_date_raw = content.get("pubDate", "")
                    provider = content.get("provider", {})
                    if isinstance(provider, dict):
                        source = provider.get("displayName", "Yahoo Finance")
                    else:
                        source = "Yahoo Finance"
                else:
                    title = item.get("title", "")
                    summary = item.get("summary", "")
                    pub_date_raw = item.get("providerPublishTime", "")
                    source = item.get("publisher", "Yahoo Finance")

                if not title:
                    continue

                # Handle epoch timestamps
                if isinstance(pub_date_raw, (int, float)):
                    pub_date = datetime.fromtimestamp(pub_date_raw).strftime("%Y-%m-%d")
                elif isinstance(pub_date_raw, str) and pub_date_raw:
                    pub_date = self._parse_iso_date(pub_date_raw)
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d")

                articles.append(
                    {
                        "title": title,
                        "source": source,
                        "published_date": pub_date,
                        "url": item.get("link", item.get("url", "")),
                        "symbol": symbol,
                        "data_source": "yahoo_finance",
                    }
                )

            logger.info("Yahoo Finance: %d articles for %s", len(articles), symbol)
            return articles

        except Exception as e:
            logger.warning("Yahoo Finance fetch failed for %s: %s", symbol, e)
            return []

    # ------------------------------------------------------------------
    # GDELT
    # ------------------------------------------------------------------

    def fetch_gdelt_news(
        self,
        symbol: str,
        timespan: str = "3m",
        max_records: int = 250,
    ) -> List[Dict]:
        """
        Fetch news from GDELT Project API.

        Args:
            symbol: Ticker symbol
            timespan: GDELT timespan (e.g. "3m" for 3 months)
            max_records: Max articles (GDELT caps at 250)

        Returns:
            List of article dicts
        """
        import urllib.request
        import json

        # Build query terms
        company_names = {
            "SPY": "S&P 500",
            "QQQ": "Nasdaq",
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "GOOGL": "Google",
            "TSLA": "Tesla",
            "META": "Meta",
            "NVDA": "Nvidia",
        }
        query_term = company_names.get(symbol, symbol)
        query_encoded = quote_plus(f'"{query_term}" stock market')

        url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc?"
            f"query={query_encoded}&mode=artlist"
            f"&maxrecords={max_records}&format=json&timespan={timespan}"
        )

        try:
            self._throttle()
            # Extra delay for GDELT (rate-limited)
            time.sleep(2.0)

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; QuantumAlpha/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            gdelt_articles = data.get("articles", [])
            articles = []

            for item in gdelt_articles:
                title = item.get("title", "").strip()
                if not title:
                    continue

                # GDELT seendate format: "20240115T120000Z"
                seen_date = item.get("seendate", "")
                if seen_date:
                    try:
                        pub_date = datetime.strptime(seen_date[:8], "%Y%m%d").strftime(
                            "%Y-%m-%d"
                        )
                    except ValueError:
                        pub_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d")

                articles.append(
                    {
                        "title": title,
                        "source": item.get("domain", "GDELT"),
                        "published_date": pub_date,
                        "url": item.get("url", ""),
                        "symbol": symbol,
                        "data_source": "gdelt",
                    }
                )

            logger.info("GDELT: %d articles for %s", len(articles), symbol)
            return articles

        except Exception as e:
            logger.warning("GDELT fetch failed for %s: %s", symbol, e)
            return []

    # ------------------------------------------------------------------
    # Combined fetch
    # ------------------------------------------------------------------

    def fetch_all_sources(
        self,
        symbol: str,
        include_gdelt: bool = True,
        google_lookback: str = "1y",
    ) -> List[Dict]:
        """
        Fetch from all sources and return unified article list.

        Args:
            symbol: Ticker symbol
            include_gdelt: Whether to include GDELT (slow, rate-limited)
            google_lookback: Google News lookback period

        Returns:
            Deduplicated list of article dicts
        """
        all_articles = []

        # Google News (primary — most articles, longest lookback)
        print(f"  Fetching Google News for {symbol}...")
        google_articles = self.fetch_google_news(
            symbol, lookback_period=google_lookback
        )
        all_articles.extend(google_articles)
        print(f"    -> {len(google_articles)} articles")

        # Yahoo Finance (supplementary — recent, high quality)
        print(f"  Fetching Yahoo Finance news for {symbol}...")
        yahoo_articles = self.fetch_yahoo_news(symbol)
        all_articles.extend(yahoo_articles)
        print(f"    -> {len(yahoo_articles)} articles")

        # GDELT (supplementary — rate-limited)
        if include_gdelt:
            print(f"  Fetching GDELT news for {symbol}...")
            gdelt_articles = self.fetch_gdelt_news(symbol)
            all_articles.extend(gdelt_articles)
            print(f"    -> {len(gdelt_articles)} articles")

        # Deduplicate by title similarity
        all_articles = self._deduplicate(all_articles)
        print(f"  Total after dedup: {len(all_articles)} articles")

        return all_articles

    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove near-duplicate articles by normalized title."""
        seen = set()
        unique = []
        for a in articles:
            # Normalize: lowercase, remove punctuation, collapse whitespace
            norm = re.sub(r"[^\w\s]", "", a["title"].lower())
            norm = re.sub(r"\s+", " ", norm).strip()
            # Use first 80 chars as dedup key (same headline from different sources)
            key = norm[:80]
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    # ------------------------------------------------------------------
    # Date parsing helpers
    # ------------------------------------------------------------------

    def _parse_rss_date(self, date_str: str) -> str:
        """Parse RSS pubDate to YYYY-MM-DD."""
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")
        # RFC 2822: "Thu, 15 Feb 2024 12:00:00 GMT"
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Fallback: try to extract date pattern
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if match:
            return match.group(0)
        return datetime.now().strftime("%Y-%m-%d")

    def _parse_iso_date(self, date_str: str) -> str:
        """Parse ISO date string to YYYY-MM-DD."""
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_str.strip()[:26], fmt).strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                continue
        return (
            date_str[:10]
            if len(date_str) >= 10
            else datetime.now().strftime("%Y-%m-%d")
        )


# ---------------------------------------------------------------------------
# FinBERT Scorer
# ---------------------------------------------------------------------------


class SentimentScorer:
    """
    Scores articles using FinBERT.

    Lazily loads the model on first use (takes ~30s on CPU).
    """

    def __init__(self):
        self._analyzer = None

    def _load_analyzer(self):
        """Load FinBERT analyzer (lazy)."""
        if self._analyzer is not None:
            return

        try:
            from quantum_alpha.models.sentiment.finbert_analyzer import (
                FinBERTSentimentAnalyzer,
                HAS_TRANSFORMERS,
            )
        except ModuleNotFoundError:
            import sys
            from pathlib import Path

            _root = str(Path(__file__).parent.parent.parent.parent)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from quantum_alpha.models.sentiment.finbert_analyzer import (
                FinBERTSentimentAnalyzer,
                HAS_TRANSFORMERS,
            )

        if not HAS_TRANSFORMERS:
            logger.warning("transformers not installed — using rule-based fallback")

        self._analyzer = FinBERTSentimentAnalyzer(device="cpu")

    def score_articles(
        self,
        articles: List[Dict],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Score articles with FinBERT.

        Args:
            articles: List of article dicts (must have 'title' key)
            batch_size: Inference batch size

        Returns:
            Same articles with sentiment fields added
        """
        self._load_analyzer()

        titles = [a["title"] for a in articles]
        if not titles:
            return articles

        # Score in batches
        results = self._analyzer.analyze(titles)

        scored = []
        for article, result in zip(articles, results):
            article_copy = article.copy()
            article_copy.update(
                {
                    "sentiment_score": result.sentiment_score,
                    "sentiment_label": result.label,
                    "sentiment_confidence": result.confidence,
                    "prob_positive": result.probabilities["positive"],
                    "prob_negative": result.probabilities["negative"],
                    "prob_neutral": result.probabilities["neutral"],
                }
            )
            scored.append(article_copy)

        return scored


# ---------------------------------------------------------------------------
# Daily Sentiment Feature Builder
# ---------------------------------------------------------------------------


def build_daily_sentiment_features(
    articles_df: pd.DataFrame,
    date_index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Aggregate per-article sentiment scores into daily features.

    Args:
        articles_df: DataFrame with columns:
            published_date, sentiment_score, sentiment_confidence,
            sentiment_label, source, prob_positive, prob_negative, prob_neutral
        date_index: Optional DatetimeIndex to align features to (fills gaps)

    Returns:
        DataFrame indexed by date with 16 sentiment features
    """
    if articles_df.empty:
        if date_index is not None:
            return pd.DataFrame(
                0.0,
                index=date_index,
                columns=REAL_SENTIMENT_FEATURE_COLS,
            )
        return pd.DataFrame(columns=REAL_SENTIMENT_FEATURE_COLS)

    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["published_date"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Source weights
    df["source_weight"] = df["source"].map(SOURCE_WEIGHTS).fillna(1.0)

    # Group by date
    daily = df.groupby("date").agg(
        mean_sentiment=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        max_positive=("sentiment_score", "max"),
        max_negative=("sentiment_score", "min"),
        n_articles=("sentiment_score", "count"),
        bullish_count=("sentiment_label", lambda x: (x == "positive").sum()),
        bearish_count=("sentiment_label", lambda x: (x == "negative").sum()),
        neutral_count=("sentiment_label", lambda x: (x == "neutral").sum()),
    )

    # Weighted sentiment
    weighted = df.groupby("date").apply(
        lambda g: np.average(g["sentiment_score"], weights=g["source_weight"])
        if len(g) > 0
        else 0.0,
        include_groups=False,
    )
    daily["weighted_sentiment"] = weighted

    # High-confidence sentiment (articles with confidence > 0.7)
    high_conf = (
        df[df["sentiment_confidence"] > 0.7].groupby("date")["sentiment_score"].mean()
    )
    daily["high_confidence_sent"] = high_conf

    # Ratios
    daily["bullish_ratio"] = daily["bullish_count"] / daily["n_articles"]
    daily["bearish_ratio"] = daily["bearish_count"] / daily["n_articles"]
    daily["neutral_ratio"] = daily["neutral_count"] / daily["n_articles"]

    # Fill NaN std (single-article days)
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily["high_confidence_sent"] = daily["high_confidence_sent"].fillna(
        daily["mean_sentiment"]
    )

    # Drop intermediate columns
    daily = daily.drop(columns=["bullish_count", "bearish_count", "neutral_count"])

    # Sort by date
    daily = daily.sort_index()

    # Rolling/momentum features (require sorted temporal data)
    daily["news_volume_surprise"] = (
        daily["n_articles"] / daily["n_articles"].rolling(7, min_periods=1).mean()
    ) - 1.0

    daily["sentiment_ma3"] = daily["mean_sentiment"].rolling(3, min_periods=1).mean()
    daily["sentiment_ma7"] = daily["mean_sentiment"].rolling(7, min_periods=1).mean()
    daily["sentiment_momentum_3d"] = daily["mean_sentiment"] - daily[
        "mean_sentiment"
    ].shift(3)
    daily["sentiment_momentum_7d"] = daily["mean_sentiment"] - daily[
        "mean_sentiment"
    ].shift(7)
    daily["sentiment_dispersion"] = daily["max_positive"] - daily["max_negative"]

    # Reorder columns to match REAL_SENTIMENT_FEATURE_COLS
    for col in REAL_SENTIMENT_FEATURE_COLS:
        if col not in daily.columns:
            daily[col] = 0.0

    daily = daily[REAL_SENTIMENT_FEATURE_COLS]

    # Align to provided date index if given (forward-fill for market holidays, etc.)
    if date_index is not None:
        # Normalize both indices to tz-naive date-only for alignment
        # (price data may be tz-aware, article dates are tz-naive)
        daily.index = pd.to_datetime(daily.index).normalize().tz_localize(None)
        norm_index = pd.to_datetime(date_index).normalize()
        if norm_index.tz is not None:
            norm_index = norm_index.tz_localize(None)

        daily = daily.reindex(norm_index)
        # Forward-fill for weekends/holidays (last known sentiment persists)
        daily = daily.ffill()
        # Fill any remaining NaN at the start with 0
        daily = daily.fillna(0.0)

        # Restore original index (preserve original tz for downstream compatibility)
        daily.index = date_index

    # Replace any remaining NaN
    daily = daily.fillna(0.0)

    return daily


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------


class SentimentPipeline:
    """
    End-to-end pipeline: fetch -> score -> store -> aggregate features.

    Usage:
        pipeline = SentimentPipeline()
        pipeline.collect_and_score("SPY")
        features = pipeline.get_features("SPY", price_df.index)
    """

    def __init__(self, db_path: str = None):
        self.db = SentimentDB(db_path)
        self.fetcher = MultiSourceNewsFetcher(request_delay=1.5)
        self.scorer = SentimentScorer()

    def collect_and_score(
        self,
        symbol: str,
        include_gdelt: bool = True,
        google_lookback: str = "1y",
        score_batch_size: int = 32,
    ) -> int:
        """
        Fetch news from all sources, score with FinBERT, store in DB.

        Args:
            symbol: Ticker symbol
            include_gdelt: Include GDELT source
            google_lookback: Google News lookback period
            score_batch_size: FinBERT batch size

        Returns:
            Number of new articles added
        """
        print(f"\n--- Collecting news for {symbol} ---")

        # Fetch from all sources
        articles = self.fetcher.fetch_all_sources(
            symbol,
            include_gdelt=include_gdelt,
            google_lookback=google_lookback,
        )

        if not articles:
            print(f"  No articles found for {symbol}")
            return 0

        # Score with FinBERT
        print(f"  Scoring {len(articles)} articles with FinBERT...")
        t0 = time.time()
        scored = self.scorer.score_articles(articles, batch_size=score_batch_size)
        t1 = time.time()
        print(f"    Scored in {t1 - t0:.1f}s")

        # Store in DB
        n_new = self.db.insert_articles(scored)
        total = self.db.count_articles(symbol)
        print(f"  Stored: {n_new} new articles ({total} total for {symbol})")

        return n_new

    def score_unscored(self, batch_size: int = 32) -> int:
        """Score any articles in DB that haven't been scored yet."""
        unscored = self.db.get_unscored_articles(limit=500)
        if unscored.empty:
            return 0

        print(f"  Scoring {len(unscored)} unscored articles...")
        titles = unscored["title"].tolist()

        self.scorer._load_analyzer()
        results = self.scorer._analyzer.analyze(titles)

        updates = []
        for idx, result in zip(unscored.index, results):
            updates.append(
                {
                    "id": unscored.loc[idx, "id"],
                    "sentiment_score": result.sentiment_score,
                    "sentiment_label": result.label,
                    "sentiment_confidence": result.confidence,
                    "prob_positive": result.probabilities["positive"],
                    "prob_negative": result.probabilities["negative"],
                    "prob_neutral": result.probabilities["neutral"],
                }
            )

        self.db.update_scores(updates)
        return len(updates)

    def get_features(
        self,
        symbol: str,
        date_index: pd.DatetimeIndex = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Get daily sentiment features for a symbol.

        Args:
            symbol: Ticker symbol
            date_index: DatetimeIndex to align to (e.g. from price data)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with 16 sentiment features, indexed by date
        """
        articles_df = self.db.get_articles(symbol, start_date, end_date)

        if articles_df.empty:
            logger.warning("No articles in DB for %s — returning zero features", symbol)
            if date_index is not None:
                return pd.DataFrame(
                    0.0, index=date_index, columns=REAL_SENTIMENT_FEATURE_COLS
                )
            return pd.DataFrame(columns=REAL_SENTIMENT_FEATURE_COLS)

        # Filter to only scored articles
        scored = articles_df[articles_df["sentiment_score"].notna()]
        if scored.empty:
            logger.warning("No scored articles for %s", symbol)
            if date_index is not None:
                return pd.DataFrame(
                    0.0, index=date_index, columns=REAL_SENTIMENT_FEATURE_COLS
                )
            return pd.DataFrame(columns=REAL_SENTIMENT_FEATURE_COLS)

        return build_daily_sentiment_features(scored, date_index)

    def build_training_features(
        self,
        price_df: pd.DataFrame,
        symbol: str = "SPY",
        forward_period: int = 1,
        use_real_sentiment: bool = True,
    ) -> pd.DataFrame:
        """
        Build complete feature set for LSTM training.

        Combines:
        - Real FinBERT sentiment features (16) OR price-proxy fallback
        - Minimal price context (5 features)
        - Target variables

        All sentiment features are LAGGED by 1 day to prevent lookahead.

        Args:
            price_df: DataFrame with OHLCV columns (DatetimeIndex)
            symbol: Ticker symbol
            forward_period: Days ahead for target return
            use_real_sentiment: If True, use real FinBERT features from DB

        Returns:
            DataFrame ready for LSTM windowing
        """
        df = price_df.copy()

        if use_real_sentiment:
            # Get real sentiment features from DB
            start_str = df.index[0].strftime("%Y-%m-%d")
            end_str = df.index[-1].strftime("%Y-%m-%d")
            sentiment_df = self.get_features(
                symbol,
                date_index=df.index,
                start_date=start_str,
                end_date=end_str,
            )

            # Check coverage
            nonzero_days = (sentiment_df["n_articles"] > 0).sum()
            total_days = len(sentiment_df)
            coverage = nonzero_days / total_days if total_days > 0 else 0

            print(
                f"  Sentiment coverage for {symbol}: {nonzero_days}/{total_days} "
                f"days ({coverage:.1%})"
            )

            if coverage < 0.1:
                logger.warning(
                    "Very low sentiment coverage (%.1f%%) for %s — "
                    "consider fetching more news first",
                    coverage * 100,
                    symbol,
                )
        else:
            # Fallback to price-proxy features
            from quantum_alpha.data.collectors.news_collector import NewsCollector

            collector = NewsCollector()
            proxy_df = collector.build_historical_sentiment(df, symbol)
            # Rename proxy columns to match real sentiment columns
            # Map proxy -> real feature names
            sentiment_df = pd.DataFrame(
                index=df.index, columns=REAL_SENTIMENT_FEATURE_COLS
            )
            sentiment_df["mean_sentiment"] = proxy_df.get("sentiment_proxy", 0)
            sentiment_df["sentiment_std"] = proxy_df.get("fear_greed", 0).abs() * 0.5
            sentiment_df["max_positive"] = proxy_df.get("sentiment_proxy", 0).clip(
                lower=0
            )
            sentiment_df["max_negative"] = proxy_df.get("sentiment_proxy", 0).clip(
                upper=0
            )
            sentiment_df["bullish_ratio"] = (
                proxy_df.get("sentiment_proxy", 0) > 0
            ).astype(float) * 0.5 + 0.25
            sentiment_df["bearish_ratio"] = (
                proxy_df.get("sentiment_proxy", 0) < 0
            ).astype(float) * 0.5 + 0.25
            sentiment_df["neutral_ratio"] = (
                1.0 - sentiment_df["bullish_ratio"] - sentiment_df["bearish_ratio"]
            )
            sentiment_df["n_articles"] = 5.0  # fake
            sentiment_df["news_volume_surprise"] = proxy_df.get("volume_surprise", 0)
            sentiment_df["weighted_sentiment"] = proxy_df.get("sentiment_proxy", 0)
            sentiment_df["sentiment_momentum_3d"] = proxy_df.get(
                "sentiment_momentum", 0
            )
            sentiment_df["sentiment_momentum_7d"] = proxy_df.get("sentiment_ma7", 0)
            sentiment_df["sentiment_ma3"] = proxy_df.get("sentiment_ma3", 0)
            sentiment_df["sentiment_ma7"] = proxy_df.get("sentiment_ma7", 0)
            sentiment_df["sentiment_dispersion"] = proxy_df.get("range_surprise", 0)
            sentiment_df["high_confidence_sent"] = proxy_df.get("sentiment_proxy", 0)
            sentiment_df = sentiment_df.fillna(0.0)

        # Price context features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = df["returns"].rolling(20).std()
        df["price_vs_sma20"] = df["close"] / df["close"].rolling(20).mean() - 1
        df["price_vs_sma50"] = df["close"] / df["close"].rolling(50).mean() - 1

        price_context = df[PRICE_CONTEXT_COLS].copy()

        # Combine
        features = pd.concat([sentiment_df, price_context], axis=1)

        # Targets
        features["target_1d"] = df["close"].pct_change().shift(-1)
        features["target_5d"] = df["close"].pct_change(5).shift(-5)
        features["close"] = df["close"]

        # LAG sentiment features by 1 day (prevent lookahead)
        for col in REAL_SENTIMENT_FEATURE_COLS:
            features[col] = features[col].shift(1)

        # Drop NaN
        features = features.dropna()

        return features

    def summary(self, symbol: str = None) -> str:
        """Print summary of DB contents."""
        lines = ["=== Sentiment Database Summary ==="]

        if symbol:
            symbols = [symbol]
        else:
            with sqlite3.connect(str(self.db.db_path)) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT symbol FROM articles ORDER BY symbol"
                ).fetchall()
                symbols = [r[0] for r in rows]

        total = 0
        for sym in symbols:
            count = self.db.count_articles(sym)
            total += count
            date_min, date_max = self.db.date_range(sym)
            lines.append(
                f"  {sym}: {count} articles "
                f"({date_min or 'N/A'} to {date_max or 'N/A'})"
            )

        lines.append(f"\n  Total: {total} articles across {len(symbols)} symbols")
        lines.append(f"  DB: {self.db.db_path}")
        return "\n".join(lines)
