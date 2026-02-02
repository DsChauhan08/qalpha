"""
FinBERT Sentiment Analyzer.

Financial sentiment analysis using the free ProsusAI/finbert model from HuggingFace.

Based on agent.md Section 3.2:
- Model: ProsusAI/finbert (fine-tuned on Financial PhraseBank)
- Labels: positive, negative, neutral
- Output: Sentiment score [-1, 1] with confidence

Falls back to rule-based sentiment if transformers not available.
"""

import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
import warnings
import re

from .text_preprocessor import FinancialTextPreprocessor

# Try to import transformers
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn(
        "transformers/torch not available. Using rule-based sentiment fallback. "
        "Install with: pip install transformers torch"
    )


@dataclass
class SentimentResult:
    """Container for sentiment analysis result."""

    text: str
    label: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0 to 1
    sentiment_score: float  # -1 to 1
    probabilities: Dict[str, float]


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.

    Model: ProsusAI/finbert (fine-tuned on Financial PhraseBank)
    Labels: positive (0), negative (1), neutral (2)
    Output: Sentiment score [-1, 1] with confidence

    Falls back to rule-based analysis if transformers unavailable.
    """

    # Default model from HuggingFace (FREE)
    DEFAULT_MODEL = "ProsusAI/finbert"

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        use_preprocessing: bool = True,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier (default: ProsusAI/finbert)
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Inference batch size
            max_length: Maximum token length
            use_preprocessing: Whether to preprocess text
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_preprocessing = use_preprocessing

        self.preprocessor = FinancialTextPreprocessor(
            lowercase=False,  # FinBERT handles casing
            max_length=max_length,
        )

        # Label mapping for FinBERT
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        self.label2id = {"positive": 0, "negative": 1, "neutral": 2}

        # Initialize model if available
        self.model = None
        self.tokenizer = None
        self.device = None

        if HAS_TRANSFORMERS:
            self._load_model(device)
        else:
            warnings.warn("Using rule-based sentiment (transformers not available)")

    def _load_model(self, device: Optional[str] = None):
        """Load the FinBERT model."""
        if not HAS_TRANSFORMERS:
            return

        # Detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            print(f"Loading {self.model_name} on {self.device}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded successfully")

        except Exception as e:
            warnings.warn(f"Failed to load model: {e}. Using rule-based fallback.")
            self.model = None
            self.tokenizer = None

    def analyze(self, texts: Union[str, List[str]]) -> List[SentimentResult]:
        """
        Analyze sentiment of financial texts.

        Args:
            texts: Single text or list of texts

        Returns:
            List of SentimentResult objects
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Use model if available, otherwise fall back to rules
        if self.model is not None and self.tokenizer is not None:
            return self._analyze_with_model(texts)
        else:
            return self._analyze_with_rules(texts)

    def _analyze_with_model(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze using FinBERT model."""
        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results

    def _process_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Process a batch of texts with the model."""
        # Preprocess
        if self.use_preprocessing:
            cleaned = [self.preprocessor.clean_for_model(t) for t in texts]
        else:
            cleaned = texts

        # Tokenize
        inputs = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        # Convert to results
        results = []
        for i, text in enumerate(texts):
            prob = probs[i].cpu().numpy()
            pred_id = int(np.argmax(prob))

            # Calculate sentiment score
            sentiment_score = self._probability_to_score(prob)

            # Truncate text for display
            display_text = text[:100] + "..." if len(text) > 100 else text

            results.append(
                SentimentResult(
                    text=display_text,
                    label=self.id2label[pred_id],
                    confidence=float(prob[pred_id]),
                    sentiment_score=float(sentiment_score),
                    probabilities={
                        "positive": float(prob[0]),
                        "negative": float(prob[1]),
                        "neutral": float(prob[2]),
                    },
                )
            )

        return results

    def _analyze_with_rules(self, texts: List[str]) -> List[SentimentResult]:
        """Rule-based sentiment analysis fallback."""
        results = []

        for text in texts:
            # Use preprocessor to get sentiment hints
            processed = self.preprocessor.preprocess(text)
            hints = processed.sentiment_hints

            # Calculate score from keyword counts
            bullish = hints["bullish"]
            bearish = hints["bearish"]
            total = bullish + bearish + 1  # +1 to avoid division by zero

            # Sentiment score: (bullish - bearish) / total, scaled
            raw_score = (bullish - bearish) / total
            sentiment_score = np.clip(raw_score * 2, -1, 1)  # Scale and clip

            # Determine label
            if sentiment_score > 0.2:
                label = "positive"
            elif sentiment_score < -0.2:
                label = "negative"
            else:
                label = "neutral"

            # Confidence based on keyword presence
            confidence = min(0.3 + (bullish + bearish) * 0.1, 0.9)

            # Calculate pseudo-probabilities
            if label == "positive":
                probs = {
                    "positive": confidence,
                    "negative": 0.1,
                    "neutral": 1 - confidence - 0.1,
                }
            elif label == "negative":
                probs = {
                    "positive": 0.1,
                    "negative": confidence,
                    "neutral": 1 - confidence - 0.1,
                }
            else:
                probs = {"positive": 0.2, "negative": 0.2, "neutral": 0.6}

            display_text = text[:100] + "..." if len(text) > 100 else text

            results.append(
                SentimentResult(
                    text=display_text,
                    label=label,
                    confidence=confidence,
                    sentiment_score=sentiment_score,
                    probabilities=probs,
                )
            )

        return results

    def _probability_to_score(self, probs: np.ndarray) -> float:
        """
        Convert probability distribution to sentiment score.

        Formula: (pos - neg) weighted by confidence
        Range: [-1, 1] where -1 = very negative, 1 = very positive
        """
        pos, neg, neutral = probs[0], probs[1], probs[2]

        # Basic score
        score = pos - neg

        # Weight by confidence (how much above uniform)
        confidence = max(pos, neg, neutral)
        confidence_weight = (confidence - 1 / 3) / (2 / 3)  # 0 at uniform, 1 at certain

        # Apply confidence weighting
        score *= 0.5 + 0.5 * confidence_weight

        return np.clip(score, -1, 1)

    def aggregate_sentiment(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None,
        decay_factor: float = 0.95,
    ) -> Dict:
        """
        Aggregate sentiment from multiple texts with time decay.

        Args:
            texts: List of texts (chronological order, oldest first)
            weights: Optional per-text weights (e.g., based on source credibility)
            decay_factor: Exponential decay for older texts (1.0 = no decay)

        Returns:
            Aggregated sentiment metrics
        """
        if not texts:
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "label": "neutral",
                "n_sources": 0,
            }

        # Analyze all texts
        results = self.analyze(texts)

        # Apply time decay (more recent = higher weight)
        n = len(results)
        if weights is None:
            weights = [decay_factor ** (n - 1 - i) for i in range(n)]

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Weighted aggregation
        sentiments = np.array([r.sentiment_score for r in results])
        confidences = np.array([r.confidence for r in results])

        weighted_sentiment = np.sum(sentiments * weights)
        weighted_confidence = np.sum(confidences * weights)

        # Measure disagreement
        sentiment_std = np.std(sentiments)

        # Determine aggregate label
        if weighted_sentiment > 0.2:
            label = "bullish"
        elif weighted_sentiment < -0.2:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "sentiment": float(weighted_sentiment),
            "confidence": float(weighted_confidence),
            "label": label,
            "sentiment_std": float(sentiment_std),
            "n_sources": len(texts),
            "individual_scores": [r.sentiment_score for r in results],
            "individual_labels": [r.label for r in results],
        }

    def analyze_news_batch(
        self,
        headlines: List[Dict],
        symbol: Optional[str] = None,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Analyze batch of news headlines with relevance filtering.

        Args:
            headlines: List of dicts with 'title', 'summary', 'source', 'published_at'
            symbol: Filter for symbol mentions (e.g., 'AAPL')
            source_weights: Weights by source (e.g., {'Reuters': 1.2, 'Twitter': 0.8})

        Returns:
            Aggregated sentiment with metadata
        """
        source_weights = source_weights or {}

        # Filter relevant headlines
        relevant_texts = []
        weights = []

        for h in headlines:
            text = h.get("title", "") + " " + h.get("summary", "")

            # Check symbol mention if specified
            if symbol:
                symbol_upper = symbol.upper()
                if (
                    symbol_upper not in text.upper()
                    and f"${symbol_upper}" not in text.upper()
                ):
                    continue

            relevant_texts.append(text)

            # Get source weight
            source = h.get("source", "unknown")
            weight = source_weights.get(source, 1.0)
            weights.append(weight)

        if not relevant_texts:
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "label": "neutral",
                "n_relevant": 0,
                "n_total": len(headlines),
                "note": "No relevant headlines found",
            }

        # Aggregate sentiment
        result = self.aggregate_sentiment(relevant_texts, weights=weights)
        result["n_relevant"] = len(relevant_texts)
        result["n_total"] = len(headlines)

        return result

    def get_trading_signal(
        self,
        sentiment_result: Dict,
        threshold: float = 0.3,
        min_confidence: float = 0.5,
        min_sources: int = 3,
    ) -> Dict:
        """
        Convert sentiment analysis to trading signal.

        Args:
            sentiment_result: Result from aggregate_sentiment or analyze_news_batch
            threshold: Minimum absolute sentiment for signal
            min_confidence: Minimum confidence to generate signal
            min_sources: Minimum number of sources required

        Returns:
            Trading signal dict
        """
        sentiment = sentiment_result.get("sentiment", 0)
        confidence = sentiment_result.get("confidence", 0)
        n_sources = sentiment_result.get("n_sources", 0)

        # Check thresholds
        if confidence < min_confidence:
            return {
                "signal": 0,
                "strength": 0,
                "reason": f"Low confidence ({confidence:.2f} < {min_confidence})",
            }

        if n_sources < min_sources:
            return {
                "signal": 0,
                "strength": 0,
                "reason": f"Insufficient sources ({n_sources} < {min_sources})",
            }

        if abs(sentiment) < threshold:
            return {
                "signal": 0,
                "strength": abs(sentiment),
                "reason": f"Sentiment below threshold ({sentiment:.2f})",
            }

        # Generate signal
        signal = 1 if sentiment > 0 else -1
        strength = min(abs(sentiment) * confidence, 1.0)

        return {
            "signal": signal,
            "strength": float(strength),
            "sentiment": float(sentiment),
            "confidence": float(confidence),
            "direction": "bullish" if signal > 0 else "bearish",
        }
