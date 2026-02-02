"""
Financial Text Preprocessor.

Specialized text preprocessing for financial documents:
- News headlines and articles
- SEC filings (10-K, 10-Q, 8-K)
- Earnings call transcripts
- Social media (Reddit, Twitter)

Handles domain-specific cleaning and normalization.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Container for processed text with metadata."""

    original: str
    cleaned: str
    entities: List[str]
    tickers: List[str]
    numbers: List[Tuple[str, float]]
    sentiment_hints: Dict[str, int]


class FinancialTextPreprocessor:
    """
    Text preprocessor optimized for financial documents.

    Features:
    - Ticker symbol extraction
    - Financial number normalization
    - Domain-specific stopword handling
    - Entity recognition (companies, metrics)
    - Sentiment hint extraction (buy/sell signals)
    """

    # Financial sentiment keywords
    BULLISH_KEYWORDS = {
        "buy",
        "upgrade",
        "outperform",
        "overweight",
        "bullish",
        "beat",
        "exceeded",
        "strong",
        "growth",
        "positive",
        "momentum",
        "breakout",
        "rally",
        "surge",
        "soar",
        "record",
        "optimistic",
        "upside",
        "accumulate",
        "conviction",
    }

    BEARISH_KEYWORDS = {
        "sell",
        "downgrade",
        "underperform",
        "underweight",
        "bearish",
        "miss",
        "missed",
        "weak",
        "decline",
        "negative",
        "slowdown",
        "breakdown",
        "crash",
        "plunge",
        "drop",
        "warning",
        "pessimistic",
        "downside",
        "reduce",
        "risk",
    }

    # Common financial abbreviations
    FINANCIAL_ABBREVS = {
        "eps": "earnings per share",
        "pe": "price to earnings",
        "ebitda": "earnings before interest taxes depreciation amortization",
        "yoy": "year over year",
        "qoq": "quarter over quarter",
        "mom": "month over month",
        "cagr": "compound annual growth rate",
        "roi": "return on investment",
        "roe": "return on equity",
        "roa": "return on assets",
        "dcf": "discounted cash flow",
        "fcf": "free cash flow",
        "nav": "net asset value",
        "aum": "assets under management",
        "etf": "exchange traded fund",
        "ipo": "initial public offering",
        "m&a": "mergers and acquisitions",
        "sec": "securities exchange commission",
        "fed": "federal reserve",
        "fomc": "federal open market committee",
    }

    # Ticker pattern (1-5 uppercase letters, optionally with exchange prefix)
    TICKER_PATTERN = re.compile(
        r"(?:^|\s|\$)([A-Z]{1,5})(?:\s|$|[,.\)])|"  # Standard tickers
        r"(?:NYSE|NASDAQ|AMEX|OTC):\s*([A-Z]{1,5})"  # Exchange-prefixed
    )

    # Number patterns (handles millions, billions, percentages)
    NUMBER_PATTERN = re.compile(
        r"(?<!\w)(\$?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(million|billion|trillion|mn|bn|tn|m|b|%)?(?!\w)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_numbers: bool = True,
        expand_abbreviations: bool = False,
        max_length: int = 512,
        extract_tickers: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            normalize_numbers: Normalize financial numbers
            expand_abbreviations: Expand financial abbreviations
            max_length: Maximum text length (for model input)
            extract_tickers: Extract and tag ticker symbols
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_numbers = normalize_numbers
        self.expand_abbreviations = expand_abbreviations
        self.max_length = max_length
        self.extract_tickers = extract_tickers

    def preprocess(self, text: str) -> ProcessedText:
        """
        Preprocess financial text.

        Args:
            text: Raw text

        Returns:
            ProcessedText with cleaned text and extracted entities
        """
        original = text

        # Extract tickers before lowercasing
        tickers = self._extract_tickers(text) if self.extract_tickers else []

        # Extract numbers
        numbers = self._extract_numbers(text) if self.normalize_numbers else []

        # Count sentiment hints
        sentiment_hints = self._count_sentiment_hints(text)

        # Clean text
        cleaned = text

        # Remove URLs
        if self.remove_urls:
            cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

        # Remove emails
        if self.remove_emails:
            cleaned = re.sub(r"\S+@\S+\.\S+", " ", cleaned)

        # Remove special characters but keep financial symbols
        cleaned = re.sub(r"[^\w\s$%.,\-+()]", " ", cleaned)

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        # Expand abbreviations
        if self.expand_abbreviations:
            cleaned = self._expand_abbreviations(cleaned)

        # Lowercase
        if self.lowercase:
            cleaned = cleaned.lower()

        # Truncate
        if len(cleaned) > self.max_length * 4:  # Rough char to token estimate
            cleaned = cleaned[: self.max_length * 4]

        # Extract entities (companies, metrics)
        entities = self._extract_entities(original)

        return ProcessedText(
            original=original,
            cleaned=cleaned,
            entities=entities,
            tickers=tickers,
            numbers=numbers,
            sentiment_hints=sentiment_hints,
        )

    def preprocess_batch(self, texts: List[str]) -> List[ProcessedText]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock ticker symbols from text."""
        matches = self.TICKER_PATTERN.findall(text)
        # Flatten and filter empty strings
        tickers = [m for match in matches for m in match if m]

        # Filter out common words that look like tickers
        common_words = {"A", "I", "AM", "PM", "CEO", "CFO", "CTO", "USA", "UK", "EU"}
        tickers = [t for t in tickers if t not in common_words]

        return list(set(tickers))

    def _extract_numbers(self, text: str) -> List[Tuple[str, float]]:
        """Extract and normalize financial numbers."""
        matches = self.NUMBER_PATTERN.findall(text)
        results = []

        for num_str, suffix in matches:
            try:
                # Remove commas and dollar signs
                num = float(num_str.replace(",", "").replace("$", ""))

                # Apply multiplier
                suffix_lower = suffix.lower() if suffix else ""
                if suffix_lower in ("million", "mn", "m"):
                    num *= 1e6
                elif suffix_lower in ("billion", "bn", "b"):
                    num *= 1e9
                elif suffix_lower in ("trillion", "tn"):
                    num *= 1e12
                elif suffix_lower == "%":
                    num /= 100  # Convert to decimal

                results.append((num_str + (suffix or ""), num))

            except ValueError:
                continue

        return results

    def _count_sentiment_hints(self, text: str) -> Dict[str, int]:
        """Count bullish and bearish keywords."""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        bullish = len(words & self.BULLISH_KEYWORDS)
        bearish = len(words & self.BEARISH_KEYWORDS)

        return {
            "bullish": bullish,
            "bearish": bearish,
            "net_sentiment": bullish - bearish,
        }

    def _expand_abbreviations(self, text: str) -> str:
        """Expand financial abbreviations."""
        text_lower = text.lower()

        for abbrev, expansion in self.FINANCIAL_ABBREVS.items():
            # Word boundary matching
            pattern = rf"\b{abbrev}\b"
            text_lower = re.sub(pattern, expansion, text_lower, flags=re.IGNORECASE)

        return text_lower

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simple rule-based)."""
        entities = []

        # Find capitalized sequences (potential company names)
        cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        entities.extend(cap_pattern.findall(text))

        # Find common financial metrics
        metric_pattern = re.compile(
            r"\b(revenue|earnings|profit|loss|sales|income|margin|"
            r"guidance|forecast|estimate|target|price)\b",
            re.IGNORECASE,
        )
        entities.extend([m.group(1) for m in metric_pattern.finditer(text)])

        return list(set(entities))

    def clean_for_model(self, text: str) -> str:
        """
        Quick clean for model input (no metadata extraction).

        Args:
            text: Raw text

        Returns:
            Cleaned text ready for model
        """
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove special characters but keep financial symbols
        text = re.sub(r"[^\w\s$%.,\-]", " ", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Truncate
        if len(text) > self.max_length * 4:
            text = text[: self.max_length * 4]

        return text
