"""
Signal Blending Module

Combines signals from multiple models/strategies into unified trading signals.
Handles different signal types, timeframes, and confidence levels.

Methods:
- SignalBlender: Main blending interface
- RegimeAwareBlender: Adjusts blending based on market regime
- ConfidenceWeightedBlender: Weights by signal confidence
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""

    DIRECTIONAL = "directional"  # -1 to 1 (short to long)
    PROBABILITY = "probability"  # 0 to 1 (probability of up move)
    CATEGORICAL = "categorical"  # -1, 0, 1 (sell, hold, buy)
    STRENGTH = "strength"  # 0 to 1 (signal strength, unsigned)


class TimeHorizon(Enum):
    """Signal time horizons."""

    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TradingSignal:
    """Container for a trading signal."""

    name: str
    signal: Union[float, np.ndarray]  # Single value or time series
    signal_type: SignalType = SignalType.DIRECTIONAL
    horizon: TimeHorizon = TimeHorizon.DAILY
    confidence: float = 1.0  # 0 to 1
    timestamp: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_directional(self) -> float:
        """Convert signal to directional (-1 to 1)."""
        val = self.signal if np.isscalar(self.signal) else self.signal[-1]

        if self.signal_type == SignalType.DIRECTIONAL:
            return float(np.clip(val, -1, 1))
        elif self.signal_type == SignalType.PROBABILITY:
            return float(2 * val - 1)  # Map [0,1] to [-1,1]
        elif self.signal_type == SignalType.CATEGORICAL:
            return float(val)
        elif self.signal_type == SignalType.STRENGTH:
            # Strength is unsigned, use metadata for direction
            direction = self.metadata.get("direction", 1)
            return float(val * direction)

        return 0.0


@dataclass
class BlendedSignal:
    """Result of signal blending."""

    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    component_weights: Dict[str, float]
    component_signals: Dict[str, float]
    timestamp: Optional[pd.Timestamp] = None
    regime: Optional[str] = None

    @property
    def position_size_factor(self) -> float:
        """Suggested position size factor based on confidence."""
        return self.confidence * abs(self.signal)

    @property
    def direction(self) -> int:
        """Signal direction: -1, 0, or 1."""
        if self.signal > 0.1:
            return 1
        elif self.signal < -0.1:
            return -1
        return 0


class SignalBlender:
    """
    Main signal blending class.

    Combines multiple trading signals into a single unified signal.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.3,
        agreement_bonus: float = 0.2,
        conflict_penalty: float = 0.3,
        normalize_output: bool = True,
    ):
        """
        Initialize signal blender.

        Args:
            weights: Fixed weights per signal source
            confidence_threshold: Minimum confidence to include signal
            agreement_bonus: Bonus when signals agree
            conflict_penalty: Penalty when signals conflict
            normalize_output: Whether to normalize final signal to [-1, 1]
        """
        self.fixed_weights = weights or {}
        self.confidence_threshold = confidence_threshold
        self.agreement_bonus = agreement_bonus
        self.conflict_penalty = conflict_penalty
        self.normalize_output = normalize_output

        self._learned_weights: Dict[str, float] = {}

    def _get_weight(self, signal_name: str, n_signals: int) -> float:
        """Get weight for a signal source."""
        if signal_name in self.fixed_weights:
            return self.fixed_weights[signal_name]
        if signal_name in self._learned_weights:
            return self._learned_weights[signal_name]
        return 1.0 / n_signals

    def _calculate_agreement_factor(self, signals: List[TradingSignal]) -> float:
        """
        Calculate agreement factor among signals.

        Returns bonus/penalty based on signal agreement.
        """
        if len(signals) < 2:
            return 1.0

        directions = [np.sign(s.to_directional()) for s in signals]

        # Count agreements
        n_positive = sum(1 for d in directions if d > 0)
        n_negative = sum(1 for d in directions if d < 0)
        n_neutral = sum(1 for d in directions if d == 0)

        total_directional = n_positive + n_negative
        if total_directional == 0:
            return 1.0

        # Agreement ratio
        max_agreement = max(n_positive, n_negative)
        agreement_ratio = max_agreement / total_directional

        if agreement_ratio > 0.7:
            return 1.0 + self.agreement_bonus * agreement_ratio
        elif agreement_ratio < 0.4:
            return 1.0 - self.conflict_penalty * (1 - agreement_ratio)

        return 1.0

    def blend(
        self, signals: List[TradingSignal], timestamp: Optional[pd.Timestamp] = None
    ) -> BlendedSignal:
        """
        Blend multiple signals into unified signal.

        Args:
            signals: List of trading signals
            timestamp: Optional timestamp for the signal

        Returns:
            Blended signal result
        """
        if not signals:
            return BlendedSignal(
                signal=0.0,
                confidence=0.0,
                component_weights={},
                component_signals={},
                timestamp=timestamp,
            )

        # Filter by confidence threshold
        valid_signals = [
            s for s in signals if s.confidence >= self.confidence_threshold
        ]

        if not valid_signals:
            valid_signals = signals  # Use all if none pass threshold

        n_signals = len(valid_signals)

        # Get weights and normalize
        weights = {
            s.name: self._get_weight(s.name, n_signals) * s.confidence
            for s in valid_signals
        }
        total_weight = sum(weights.values())

        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Convert signals to directional
        directional_signals = {s.name: s.to_directional() for s in valid_signals}

        # Calculate weighted sum
        blended = sum(weights[name] * directional_signals[name] for name in weights)

        # Apply agreement factor
        agreement_factor = self._calculate_agreement_factor(valid_signals)
        blended *= agreement_factor

        # Calculate confidence
        avg_confidence = np.mean([s.confidence for s in valid_signals])
        signal_alignment = 1.0 - np.std([s.to_directional() for s in valid_signals])
        confidence = avg_confidence * signal_alignment * min(1.0, agreement_factor)

        # Normalize if requested
        if self.normalize_output:
            blended = np.clip(blended, -1, 1)

        return BlendedSignal(
            signal=float(blended),
            confidence=float(confidence),
            component_weights=weights,
            component_signals=directional_signals,
            timestamp=timestamp,
        )

    def learn_weights(
        self,
        signal_history: List[List[TradingSignal]],
        returns: np.ndarray,
        method: str = "correlation",
    ) -> Dict[str, float]:
        """
        Learn optimal weights from historical data.

        Args:
            signal_history: List of signal lists for each period
            returns: Actual returns for each period
            method: 'correlation' or 'regression'

        Returns:
            Learned weights
        """
        if len(signal_history) != len(returns):
            raise ValueError("Signal history and returns must have same length")

        # Extract signal names
        all_names = set()
        for signals in signal_history:
            all_names.update(s.name for s in signals)

        signal_names = list(all_names)
        n_signals = len(signal_names)

        # Build signal matrix
        n_periods = len(returns)
        signal_matrix = np.zeros((n_periods, n_signals))

        for t, signals in enumerate(signal_history):
            signal_dict = {s.name: s.to_directional() for s in signals}
            for i, name in enumerate(signal_names):
                signal_matrix[t, i] = signal_dict.get(name, 0)

        if method == "correlation":
            # Weight by correlation with returns
            correlations = np.zeros(n_signals)
            for i in range(n_signals):
                corr = np.corrcoef(signal_matrix[:, i], returns)[0, 1]
                correlations[i] = max(0, corr)  # Only positive correlations

            # Normalize
            if correlations.sum() > 0:
                weights = correlations / correlations.sum()
            else:
                weights = np.ones(n_signals) / n_signals

        else:  # regression
            # Ridge regression weights
            alpha = 1.0
            XtX = signal_matrix.T @ signal_matrix + alpha * np.eye(n_signals)
            Xty = signal_matrix.T @ returns

            try:
                weights = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(XtX) @ Xty

            # Convert to positive weights
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-10)

        self._learned_weights = {
            name: float(w) for name, w in zip(signal_names, weights)
        }

        logger.info(f"Learned signal weights: {self._learned_weights}")
        return self._learned_weights


class RegimeAwareBlender(SignalBlender):
    """
    Signal blender that adjusts based on market regime.

    Different signal sources may perform better in different regimes.
    """

    # Default regime-specific weights
    DEFAULT_REGIME_WEIGHTS = {
        "bull": {
            "momentum": 1.2,
            "trend": 1.1,
            "mean_reversion": 0.7,
            "volatility": 0.8,
            "sentiment": 1.0,
        },
        "bear": {
            "momentum": 0.8,
            "trend": 0.9,
            "mean_reversion": 0.9,
            "volatility": 1.2,
            "sentiment": 0.7,
        },
        "sideways": {
            "momentum": 0.6,
            "trend": 0.7,
            "mean_reversion": 1.3,
            "volatility": 0.9,
            "sentiment": 0.8,
        },
        "crisis": {
            "momentum": 0.5,
            "trend": 0.6,
            "mean_reversion": 0.5,
            "volatility": 1.5,
            "sentiment": 0.6,
        },
    }

    def __init__(
        self, regime_weights: Optional[Dict[str, Dict[str, float]]] = None, **kwargs
    ):
        """
        Initialize regime-aware blender.

        Args:
            regime_weights: Weights per regime per signal type
            **kwargs: Arguments for base SignalBlender
        """
        super().__init__(**kwargs)
        self.regime_weights = regime_weights or self.DEFAULT_REGIME_WEIGHTS

    def _get_regime_multiplier(self, signal_name: str, regime: str) -> float:
        """Get regime-specific weight multiplier."""
        regime_lower = regime.lower()

        if regime_lower not in self.regime_weights:
            return 1.0

        regime_w = self.regime_weights[regime_lower]

        # Check for exact match
        if signal_name in regime_w:
            return regime_w[signal_name]

        # Check for partial match (e.g., "momentum_20d" matches "momentum")
        for key, value in regime_w.items():
            if key in signal_name.lower():
                return value

        return 1.0

    def blend(
        self,
        signals: List[TradingSignal],
        regime: str = "unknown",
        timestamp: Optional[pd.Timestamp] = None,
    ) -> BlendedSignal:
        """
        Blend signals with regime awareness.

        Args:
            signals: List of trading signals
            regime: Current market regime
            timestamp: Optional timestamp

        Returns:
            Blended signal with regime adjustment
        """
        if not signals:
            return BlendedSignal(
                signal=0.0,
                confidence=0.0,
                component_weights={},
                component_signals={},
                timestamp=timestamp,
                regime=regime,
            )

        # Apply regime multipliers to confidence
        adjusted_signals = []
        for s in signals:
            multiplier = self._get_regime_multiplier(s.name, regime)
            adjusted_signal = TradingSignal(
                name=s.name,
                signal=s.signal,
                signal_type=s.signal_type,
                horizon=s.horizon,
                confidence=s.confidence * multiplier,
                timestamp=s.timestamp,
                metadata=s.metadata,
            )
            adjusted_signals.append(adjusted_signal)

        # Use base blending
        result = super().blend(adjusted_signals, timestamp)
        result.regime = regime

        # Additional regime-based adjustments
        if regime.lower() == "crisis":
            # Reduce overall exposure in crisis
            result = BlendedSignal(
                signal=result.signal * 0.5,
                confidence=result.confidence * 0.7,
                component_weights=result.component_weights,
                component_signals=result.component_signals,
                timestamp=result.timestamp,
                regime=regime,
            )

        return result


class TimeframeBlender(SignalBlender):
    """
    Signal blender that handles multiple timeframes.

    Aligns signals from different horizons into coherent signal.
    """

    # Timeframe weights (longer = more weight for position sizing)
    DEFAULT_TIMEFRAME_WEIGHTS = {
        TimeHorizon.INTRADAY: 0.15,
        TimeHorizon.DAILY: 0.35,
        TimeHorizon.WEEKLY: 0.30,
        TimeHorizon.MONTHLY: 0.20,
    }

    def __init__(
        self,
        timeframe_weights: Optional[Dict[TimeHorizon, float]] = None,
        require_alignment: bool = True,
        alignment_threshold: float = 0.6,
        **kwargs,
    ):
        """
        Initialize timeframe blender.

        Args:
            timeframe_weights: Weights per timeframe
            require_alignment: Require timeframe alignment for strong signal
            alignment_threshold: Minimum alignment ratio
            **kwargs: Arguments for base SignalBlender
        """
        super().__init__(**kwargs)
        self.timeframe_weights = timeframe_weights or self.DEFAULT_TIMEFRAME_WEIGHTS
        self.require_alignment = require_alignment
        self.alignment_threshold = alignment_threshold

    def _check_timeframe_alignment(
        self, signals: List[TradingSignal]
    ) -> Tuple[bool, float]:
        """
        Check if signals across timeframes are aligned.

        Returns:
            Tuple of (is_aligned, alignment_ratio)
        """
        if len(signals) < 2:
            return True, 1.0

        directions = [np.sign(s.to_directional()) for s in signals]

        # Count by direction
        n_positive = sum(1 for d in directions if d > 0)
        n_negative = sum(1 for d in directions if d < 0)

        total = n_positive + n_negative
        if total == 0:
            return True, 1.0

        alignment_ratio = max(n_positive, n_negative) / total
        is_aligned = alignment_ratio >= self.alignment_threshold

        return is_aligned, alignment_ratio

    def blend(
        self, signals: List[TradingSignal], timestamp: Optional[pd.Timestamp] = None
    ) -> BlendedSignal:
        """
        Blend signals across timeframes.

        Args:
            signals: List of trading signals with different horizons
            timestamp: Optional timestamp

        Returns:
            Blended signal
        """
        if not signals:
            return BlendedSignal(
                signal=0.0,
                confidence=0.0,
                component_weights={},
                component_signals={},
                timestamp=timestamp,
            )

        # Group by timeframe
        by_timeframe: Dict[TimeHorizon, List[TradingSignal]] = {}
        for s in signals:
            horizon = s.horizon
            if horizon not in by_timeframe:
                by_timeframe[horizon] = []
            by_timeframe[horizon].append(s)

        # Blend within each timeframe first
        timeframe_signals = []
        for horizon, tf_signals in by_timeframe.items():
            tf_blend = super().blend(tf_signals, timestamp)

            # Create synthetic signal for this timeframe
            tf_weight = self.timeframe_weights.get(horizon, 0.25)
            timeframe_signals.append(
                TradingSignal(
                    name=f"tf_{horizon.value}",
                    signal=tf_blend.signal,
                    signal_type=SignalType.DIRECTIONAL,
                    horizon=horizon,
                    confidence=tf_blend.confidence * tf_weight,
                    timestamp=timestamp,
                )
            )

        # Check alignment
        is_aligned, alignment_ratio = self._check_timeframe_alignment(timeframe_signals)

        # Final blend
        result = super().blend(timeframe_signals, timestamp)

        # Adjust confidence based on alignment
        if self.require_alignment and not is_aligned:
            result = BlendedSignal(
                signal=result.signal * alignment_ratio,
                confidence=result.confidence * alignment_ratio,
                component_weights=result.component_weights,
                component_signals=result.component_signals,
                timestamp=result.timestamp,
            )

        return result


class ConfidenceWeightedBlender(SignalBlender):
    """
    Signal blender that heavily weights by confidence.

    Uses confidence-squared weighting to emphasize high-confidence signals.
    """

    def __init__(
        self,
        confidence_power: float = 2.0,
        min_confidence_weight: float = 0.01,
        **kwargs,
    ):
        """
        Initialize confidence-weighted blender.

        Args:
            confidence_power: Power to raise confidence (higher = more emphasis)
            min_confidence_weight: Minimum weight even for low confidence
            **kwargs: Arguments for base SignalBlender
        """
        super().__init__(**kwargs)
        self.confidence_power = confidence_power
        self.min_confidence_weight = min_confidence_weight

    def blend(
        self, signals: List[TradingSignal], timestamp: Optional[pd.Timestamp] = None
    ) -> BlendedSignal:
        """
        Blend signals with confidence-weighted approach.

        Args:
            signals: List of trading signals
            timestamp: Optional timestamp

        Returns:
            Blended signal
        """
        if not signals:
            return BlendedSignal(
                signal=0.0,
                confidence=0.0,
                component_weights={},
                component_signals={},
                timestamp=timestamp,
            )

        # Calculate confidence-based weights
        weights = {}
        for s in signals:
            conf_weight = max(
                self.min_confidence_weight, s.confidence**self.confidence_power
            )
            weights[s.name] = conf_weight

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Convert to directional
        directional_signals = {s.name: s.to_directional() for s in signals}

        # Weighted sum
        blended = sum(weights[name] * directional_signals[name] for name in weights)

        # Confidence is weighted average
        confidence = sum(weights[s.name] * s.confidence for s in signals)

        # Normalize
        if self.normalize_output:
            blended = np.clip(blended, -1, 1)

        return BlendedSignal(
            signal=float(blended),
            confidence=float(confidence),
            component_weights=weights,
            component_signals=directional_signals,
            timestamp=timestamp,
        )


class HierarchicalBlender:
    """
    Hierarchical signal blending.

    First groups signals by category, blends within groups,
    then blends group signals.
    """

    def __init__(
        self,
        group_weights: Optional[Dict[str, float]] = None,
        intra_group_blender: Optional[SignalBlender] = None,
        inter_group_blender: Optional[SignalBlender] = None,
    ):
        """
        Initialize hierarchical blender.

        Args:
            group_weights: Weights for each signal group
            intra_group_blender: Blender for within-group signals
            inter_group_blender: Blender for between-group signals
        """
        self.group_weights = group_weights or {}
        self.intra_group_blender = intra_group_blender or SignalBlender()
        self.inter_group_blender = inter_group_blender or SignalBlender()

    def blend(
        self,
        signals: List[TradingSignal],
        groups: Dict[str, List[str]],  # group_name -> [signal_names]
        timestamp: Optional[pd.Timestamp] = None,
    ) -> BlendedSignal:
        """
        Hierarchically blend signals.

        Args:
            signals: List of all trading signals
            groups: Mapping of group names to signal names
            timestamp: Optional timestamp

        Returns:
            Blended signal
        """
        signal_dict = {s.name: s for s in signals}

        # Blend within each group
        group_signals = []
        for group_name, signal_names in groups.items():
            group_sigs = [
                signal_dict[name] for name in signal_names if name in signal_dict
            ]

            if group_sigs:
                group_blend = self.intra_group_blender.blend(group_sigs, timestamp)

                # Create group-level signal
                group_weight = self.group_weights.get(group_name, 1.0)
                group_signals.append(
                    TradingSignal(
                        name=f"group_{group_name}",
                        signal=group_blend.signal,
                        signal_type=SignalType.DIRECTIONAL,
                        horizon=TimeHorizon.DAILY,
                        confidence=group_blend.confidence * group_weight,
                        timestamp=timestamp,
                    )
                )

        # Blend groups
        return self.inter_group_blender.blend(group_signals, timestamp)


def create_signal_blender(**kwargs) -> SignalBlender:
    """Factory for signal blender."""
    return SignalBlender(**kwargs)


def create_regime_aware_blender(**kwargs) -> RegimeAwareBlender:
    """Factory for regime-aware blender."""
    return RegimeAwareBlender(**kwargs)


def create_timeframe_blender(**kwargs) -> TimeframeBlender:
    """Factory for timeframe blender."""
    return TimeframeBlender(**kwargs)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample signals
    signals = [
        TradingSignal(
            name="momentum_20d",
            signal=0.6,
            signal_type=SignalType.DIRECTIONAL,
            horizon=TimeHorizon.DAILY,
            confidence=0.8,
        ),
        TradingSignal(
            name="mean_reversion",
            signal=-0.3,
            signal_type=SignalType.DIRECTIONAL,
            horizon=TimeHorizon.DAILY,
            confidence=0.6,
        ),
        TradingSignal(
            name="sentiment",
            signal=0.75,
            signal_type=SignalType.PROBABILITY,  # Will be converted
            horizon=TimeHorizon.DAILY,
            confidence=0.5,
        ),
        TradingSignal(
            name="lstm_weekly",
            signal=0.4,
            signal_type=SignalType.DIRECTIONAL,
            horizon=TimeHorizon.WEEKLY,
            confidence=0.7,
        ),
    ]

    # Test basic blender
    print("Basic Signal Blender:")
    blender = SignalBlender()
    result = blender.blend(signals)
    print(f"  Blended signal: {result.signal:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Weights: {result.component_weights}")

    # Test regime-aware blender
    print("\nRegime-Aware Blender (Bull):")
    regime_blender = RegimeAwareBlender()
    result_bull = regime_blender.blend(signals, regime="bull")
    print(f"  Blended signal: {result_bull.signal:.3f}")
    print(f"  Confidence: {result_bull.confidence:.3f}")

    print("\nRegime-Aware Blender (Bear):")
    result_bear = regime_blender.blend(signals, regime="bear")
    print(f"  Blended signal: {result_bear.signal:.3f}")
    print(f"  Confidence: {result_bear.confidence:.3f}")

    # Test timeframe blender
    print("\nTimeframe Blender:")
    tf_blender = TimeframeBlender()
    result_tf = tf_blender.blend(signals)
    print(f"  Blended signal: {result_tf.signal:.3f}")
    print(f"  Confidence: {result_tf.confidence:.3f}")

    # Test confidence-weighted blender
    print("\nConfidence-Weighted Blender:")
    conf_blender = ConfidenceWeightedBlender(confidence_power=2.0)
    result_conf = conf_blender.blend(signals)
    print(f"  Blended signal: {result_conf.signal:.3f}")
    print(f"  Confidence: {result_conf.confidence:.3f}")
    print(f"  Weights: {result_conf.component_weights}")
