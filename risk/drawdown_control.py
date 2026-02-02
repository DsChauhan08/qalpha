"""
Drawdown Control Module - Quantum Alpha V1
Dynamic exposure reduction based on drawdown per agent.md Section 5.

Implements:
- Real-time drawdown monitoring
- Dynamic position scaling based on drawdown
- Drawdown-based circuit breakers
- Recovery tracking and gradual re-entry
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DrawdownState(Enum):
    """Current drawdown state."""

    NORMAL = "normal"  # Below warning threshold
    WARNING = "warning"  # Approaching limits
    CRITICAL = "critical"  # Near max drawdown
    CIRCUIT_BREAKER = "circuit_breaker"  # Trading halted
    RECOVERY = "recovery"  # Recovering from drawdown


@dataclass
class DrawdownMetrics:
    """Current drawdown metrics."""

    current_drawdown: float  # Current drawdown percentage
    max_drawdown: float  # Maximum drawdown in period
    drawdown_duration: int  # Days in current drawdown
    recovery_needed: float  # % gain needed to recover
    state: DrawdownState
    exposure_multiplier: float  # Current exposure scaling
    time_to_recovery: Optional[int] = None  # Estimated days
    peak_value: float = 0.0
    trough_value: float = 0.0


@dataclass
class DrawdownEvent:
    """Record of a drawdown event."""

    start_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    max_drawdown: float
    duration_days: int
    recovered: bool


class DrawdownController:
    """
    Dynamic drawdown control with exposure scaling.
    Reduces risk as drawdown increases to protect capital.
    """

    def __init__(
        self,
        warning_threshold: float = 0.05,  # 5% drawdown warning
        critical_threshold: float = 0.10,  # 10% drawdown critical
        circuit_breaker_threshold: float = 0.15,  # 15% halt trading
        max_drawdown_limit: float = 0.20,  # 20% absolute max
        recovery_threshold: float = 0.50,  # 50% recovery to resume
        scaling_method: str = "linear",  # linear, exponential, stepped
        min_exposure: float = 0.10,  # Minimum 10% exposure
        cooldown_days: int = 5,  # Days before re-entry after circuit breaker
        history_size: int = 1000,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.max_drawdown_limit = max_drawdown_limit
        self.recovery_threshold = recovery_threshold
        self.scaling_method = scaling_method
        self.min_exposure = min_exposure
        self.cooldown_days = cooldown_days

        # State tracking
        self.peak_value: float = 0.0
        self.trough_value: float = float("inf")
        self.current_value: float = 0.0
        self.current_state: DrawdownState = DrawdownState.NORMAL
        self.drawdown_start: Optional[datetime] = None
        self.circuit_breaker_triggered: Optional[datetime] = None

        # History
        self.value_history: deque = deque(maxlen=history_size)
        self.drawdown_history: deque = deque(maxlen=history_size)
        self.drawdown_events: List[DrawdownEvent] = []

        # Current event tracking
        self.current_event: Optional[DrawdownEvent] = None

    def update(
        self, portfolio_value: float, timestamp: Optional[datetime] = None
    ) -> DrawdownMetrics:
        """
        Update with new portfolio value.

        Args:
            portfolio_value: Current portfolio value
            timestamp: Optional timestamp

        Returns:
            DrawdownMetrics with current state
        """
        timestamp = timestamp or datetime.now()
        self.current_value = portfolio_value

        # Update peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self._end_drawdown_event(timestamp)

        # Calculate current drawdown
        current_dd = self._calculate_drawdown(portfolio_value)

        # Update trough during drawdown
        if current_dd > 0:
            if portfolio_value < self.trough_value:
                self.trough_value = portfolio_value

            # Start drawdown event if new
            if self.current_event is None and current_dd >= self.warning_threshold:
                self._start_drawdown_event(timestamp)

        # Update state
        new_state = self._determine_state(current_dd, timestamp)
        self._handle_state_transition(self.current_state, new_state, timestamp)
        self.current_state = new_state

        # Calculate exposure multiplier
        exposure_mult = self._calculate_exposure_multiplier(current_dd)

        # Store history
        self.value_history.append(
            {
                "timestamp": timestamp,
                "value": portfolio_value,
                "drawdown": current_dd,
                "state": new_state.value,
                "exposure_multiplier": exposure_mult,
            }
        )

        self.drawdown_history.append(current_dd)

        # Calculate metrics
        max_dd = max(self.drawdown_history) if self.drawdown_history else 0
        duration = self._calculate_duration(timestamp)
        recovery_needed = self._calculate_recovery_needed(current_dd)

        return DrawdownMetrics(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            drawdown_duration=duration,
            recovery_needed=recovery_needed,
            state=new_state,
            exposure_multiplier=exposure_mult,
            time_to_recovery=self._estimate_recovery_time(recovery_needed),
            peak_value=self.peak_value,
            trough_value=self.trough_value
            if self.trough_value != float("inf")
            else portfolio_value,
        )

    def _calculate_drawdown(self, value: float) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - value) / self.peak_value

    def _determine_state(self, drawdown: float, timestamp: datetime) -> DrawdownState:
        """Determine current drawdown state."""
        # Check if in cooldown after circuit breaker
        if self.circuit_breaker_triggered:
            days_since = (timestamp - self.circuit_breaker_triggered).days
            if days_since < self.cooldown_days:
                return DrawdownState.CIRCUIT_BREAKER

        # Check thresholds
        if drawdown >= self.circuit_breaker_threshold:
            return DrawdownState.CIRCUIT_BREAKER
        elif drawdown >= self.critical_threshold:
            return DrawdownState.CRITICAL
        elif drawdown >= self.warning_threshold:
            return DrawdownState.WARNING
        elif self.current_state in [DrawdownState.CRITICAL, DrawdownState.WARNING]:
            # Check if recovering
            if drawdown > 0:
                return DrawdownState.RECOVERY

        return DrawdownState.NORMAL

    def _handle_state_transition(
        self, old_state: DrawdownState, new_state: DrawdownState, timestamp: datetime
    ):
        """Handle state transitions."""
        if old_state == new_state:
            return

        logger.info(
            f"Drawdown state transition: {old_state.value} -> {new_state.value}"
        )

        # Circuit breaker triggered
        if new_state == DrawdownState.CIRCUIT_BREAKER:
            self.circuit_breaker_triggered = timestamp
            logger.warning(f"CIRCUIT BREAKER TRIGGERED at {timestamp}")

        # Recovery from circuit breaker
        elif (
            old_state == DrawdownState.CIRCUIT_BREAKER
            and new_state != DrawdownState.CIRCUIT_BREAKER
        ):
            self.circuit_breaker_triggered = None
            logger.info("Circuit breaker released")

    def _calculate_exposure_multiplier(self, drawdown: float) -> float:
        """
        Calculate exposure multiplier based on drawdown.
        Returns value between min_exposure and 1.0.
        """
        if drawdown <= 0:
            return 1.0

        if self.current_state == DrawdownState.CIRCUIT_BREAKER:
            return 0.0  # No trading during circuit breaker

        if self.scaling_method == "linear":
            # Linear scaling from 1.0 at 0% to min_exposure at circuit_breaker
            scale_range = self.circuit_breaker_threshold
            scale_factor = max(0, 1 - (drawdown / scale_range))
            return max(self.min_exposure, scale_factor)

        elif self.scaling_method == "exponential":
            # Exponential decay - faster reduction as drawdown increases
            decay_rate = 5.0  # Steepness of decay
            scale_factor = np.exp(-decay_rate * drawdown)
            return max(self.min_exposure, scale_factor)

        elif self.scaling_method == "stepped":
            # Step-wise reduction at thresholds
            if drawdown >= self.critical_threshold:
                return self.min_exposure
            elif drawdown >= self.warning_threshold:
                return 0.5
            else:
                return 1.0

        return 1.0

    def _calculate_duration(self, timestamp: datetime) -> int:
        """Calculate duration of current drawdown in days."""
        if self.drawdown_start is None:
            return 0
        return (timestamp - self.drawdown_start).days

    def _calculate_recovery_needed(self, drawdown: float) -> float:
        """Calculate percentage gain needed to recover from drawdown."""
        if drawdown <= 0:
            return 0.0
        # If down 20%, need 25% gain to recover (1/0.8 - 1)
        return 1 / (1 - drawdown) - 1

    def _estimate_recovery_time(
        self, recovery_needed: float, expected_daily_return: float = 0.0003
    ) -> Optional[int]:
        """Estimate days to recovery based on expected returns."""
        if recovery_needed <= 0:
            return 0
        if expected_daily_return <= 0:
            return None

        # Simple estimation: days = ln(1 + recovery_needed) / ln(1 + daily_return)
        days = np.log(1 + recovery_needed) / np.log(1 + expected_daily_return)
        return int(days)

    def _start_drawdown_event(self, timestamp: datetime):
        """Start tracking a drawdown event."""
        self.drawdown_start = timestamp
        self.trough_value = self.current_value

        self.current_event = DrawdownEvent(
            start_date=timestamp,
            end_date=None,
            peak_value=self.peak_value,
            trough_value=self.current_value,
            max_drawdown=0,
            duration_days=0,
            recovered=False,
        )

    def _end_drawdown_event(self, timestamp: datetime):
        """End current drawdown event (new high reached)."""
        if self.current_event is not None:
            self.current_event.end_date = timestamp
            self.current_event.trough_value = self.trough_value
            self.current_event.max_drawdown = (
                self.current_event.peak_value - self.trough_value
            ) / self.current_event.peak_value
            self.current_event.duration_days = (
                timestamp - self.current_event.start_date
            ).days
            self.current_event.recovered = True

            self.drawdown_events.append(self.current_event)
            self.current_event = None

        # Reset tracking
        self.drawdown_start = None
        self.trough_value = float("inf")

    def get_position_scale(self, base_position: float) -> float:
        """
        Get scaled position size based on current drawdown.

        Args:
            base_position: Original position size

        Returns:
            Scaled position size
        """
        if not self.value_history:
            return base_position

        last_metrics = self.value_history[-1]
        multiplier = last_metrics.get("exposure_multiplier", 1.0)

        return base_position * multiplier

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        if self.current_state == DrawdownState.CIRCUIT_BREAKER:
            return False, "Circuit breaker active - trading halted"

        if not self.value_history:
            return True, "No drawdown history"

        last_dd = self.value_history[-1].get("drawdown", 0)
        if last_dd >= self.max_drawdown_limit:
            return False, f"Max drawdown limit reached: {last_dd:.1%}"

        return True, "Trading allowed"

    def get_drawdown_events(self, min_drawdown: float = 0.05) -> List[DrawdownEvent]:
        """Get historical drawdown events above threshold."""
        return [e for e in self.drawdown_events if e.max_drawdown >= min_drawdown]

    def get_statistics(self) -> Dict[str, Any]:
        """Get drawdown statistics."""
        if not self.drawdown_events:
            return {
                "total_events": 0,
                "avg_drawdown": 0,
                "max_drawdown": 0,
                "avg_duration": 0,
                "recovery_rate": 0,
            }

        drawdowns = [e.max_drawdown for e in self.drawdown_events]
        durations = [e.duration_days for e in self.drawdown_events]
        recovered = [e for e in self.drawdown_events if e.recovered]

        return {
            "total_events": len(self.drawdown_events),
            "avg_drawdown": np.mean(drawdowns),
            "max_drawdown": max(drawdowns),
            "avg_duration": np.mean(durations),
            "max_duration": max(durations),
            "recovery_rate": len(recovered) / len(self.drawdown_events),
            "current_state": self.current_state.value,
        }

    def reset(self, initial_value: float):
        """Reset controller with new initial value."""
        self.peak_value = initial_value
        self.trough_value = float("inf")
        self.current_value = initial_value
        self.current_state = DrawdownState.NORMAL
        self.drawdown_start = None
        self.circuit_breaker_triggered = None
        self.value_history.clear()
        self.drawdown_history.clear()
        self.current_event = None


class AdaptiveDrawdownController(DrawdownController):
    """
    Adaptive drawdown controller that adjusts thresholds based on
    market regime and strategy performance.
    """

    def __init__(
        self,
        base_warning: float = 0.05,
        base_critical: float = 0.10,
        base_circuit_breaker: float = 0.15,
        volatility_adjustment: bool = True,
        regime_adjustment: bool = True,
        **kwargs,
    ):
        super().__init__(
            warning_threshold=base_warning,
            critical_threshold=base_critical,
            circuit_breaker_threshold=base_circuit_breaker,
            **kwargs,
        )

        self.base_warning = base_warning
        self.base_critical = base_critical
        self.base_circuit_breaker = base_circuit_breaker
        self.volatility_adjustment = volatility_adjustment
        self.regime_adjustment = regime_adjustment

        # Volatility tracking
        self.returns_buffer: deque = deque(maxlen=60)
        self.current_volatility: float = 0.0
        self.long_term_volatility: float = 0.15  # Annualized default

    def update_with_regime(
        self,
        portfolio_value: float,
        regime: Optional[str] = None,
        market_volatility: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> DrawdownMetrics:
        """
        Update with regime and volatility information.

        Args:
            portfolio_value: Current portfolio value
            regime: Current market regime (bull, bear, crisis, etc.)
            market_volatility: Current market volatility
            timestamp: Optional timestamp

        Returns:
            DrawdownMetrics
        """
        # Track returns for volatility
        if self.value_history:
            last_value = self.value_history[-1]["value"]
            if last_value > 0:
                ret = (portfolio_value - last_value) / last_value
                self.returns_buffer.append(ret)

        # Update volatility estimate
        if len(self.returns_buffer) >= 20:
            self.current_volatility = np.std(list(self.returns_buffer)) * np.sqrt(252)

        if market_volatility:
            self.current_volatility = market_volatility

        # Adjust thresholds
        self._adjust_thresholds(regime)

        # Call parent update
        return self.update(portfolio_value, timestamp)

    def _adjust_thresholds(self, regime: Optional[str] = None):
        """Adjust thresholds based on conditions."""
        multiplier = 1.0

        # Volatility adjustment
        if self.volatility_adjustment and self.current_volatility > 0:
            vol_ratio = self.current_volatility / self.long_term_volatility
            # Widen thresholds when volatility is high
            multiplier *= min(1.5, max(0.75, vol_ratio))

        # Regime adjustment
        if self.regime_adjustment and regime:
            regime_multipliers = {
                "bull": 1.2,  # More lenient in bull markets
                "bear": 0.8,  # Tighter in bear markets
                "crisis": 0.6,  # Much tighter in crisis
                "sideways": 1.0,
            }
            multiplier *= regime_multipliers.get(regime.lower(), 1.0)

        # Apply adjustments
        self.warning_threshold = self.base_warning * multiplier
        self.critical_threshold = self.base_critical * multiplier
        self.circuit_breaker_threshold = self.base_circuit_breaker * multiplier


class DrawdownProtection:
    """
    Portfolio protection strategies based on drawdown.
    """

    @staticmethod
    def calculate_protective_put_coverage(
        drawdown: float, portfolio_value: float, max_coverage: float = 0.20
    ) -> float:
        """
        Calculate suggested protective put coverage based on drawdown.
        As drawdown increases, suggest more protection.
        """
        if drawdown <= 0.02:
            return 0.0  # No protection needed

        # Scale coverage with drawdown
        coverage = min(max_coverage, drawdown * 2)
        return coverage * portfolio_value

    @staticmethod
    def calculate_stop_loss_level(
        entry_price: float,
        current_drawdown: float,
        atr: float,
        base_stop_atr: float = 2.0,
    ) -> float:
        """
        Calculate adaptive stop loss based on drawdown.
        Tighter stops as drawdown increases.
        """
        # Tighten stop as drawdown increases
        if current_drawdown >= 0.10:
            atr_multiplier = base_stop_atr * 0.5
        elif current_drawdown >= 0.05:
            atr_multiplier = base_stop_atr * 0.75
        else:
            atr_multiplier = base_stop_atr

        return entry_price - (atr * atr_multiplier)

    @staticmethod
    def get_hedging_recommendation(
        drawdown: float, portfolio_beta: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get hedging recommendation based on drawdown.
        """
        if drawdown < 0.05:
            return {
                "action": "none",
                "hedge_ratio": 0.0,
                "description": "No hedging needed",
            }
        elif drawdown < 0.10:
            hedge_ratio = 0.25 * portfolio_beta
            return {
                "action": "partial_hedge",
                "hedge_ratio": hedge_ratio,
                "description": f"Consider hedging {hedge_ratio:.0%} of portfolio beta",
                "instruments": ["SPY puts", "VIX calls"],
            }
        else:
            hedge_ratio = 0.50 * portfolio_beta
            return {
                "action": "aggressive_hedge",
                "hedge_ratio": hedge_ratio,
                "description": f"Recommend hedging {hedge_ratio:.0%} of portfolio beta",
                "instruments": ["SPY puts", "VIX calls", "Reduce long exposure"],
            }


def calculate_underwater_curve(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate underwater (drawdown) curve from equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Series of drawdown percentages (negative values)
    """
    running_max = equity_curve.expanding().max()
    underwater = (equity_curve - running_max) / running_max
    return underwater


def analyze_drawdowns(equity_curve: pd.Series, threshold: float = 0.05) -> pd.DataFrame:
    """
    Analyze all drawdowns in an equity curve.

    Args:
        equity_curve: Series of portfolio values
        threshold: Minimum drawdown to consider

    Returns:
        DataFrame with drawdown analysis
    """
    underwater = calculate_underwater_curve(equity_curve)

    # Find drawdown periods
    in_drawdown = underwater < -threshold

    # Identify drawdown starts and ends
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

    events = []
    start_idx = None

    for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
        if is_start:
            start_idx = i
        elif is_end and start_idx is not None:
            period = underwater.iloc[start_idx:i]
            events.append(
                {
                    "start": equity_curve.index[start_idx],
                    "end": equity_curve.index[i],
                    "max_drawdown": period.min(),
                    "duration_days": i - start_idx,
                    "peak_value": equity_curve.iloc[start_idx],
                    "trough_value": equity_curve.iloc[start_idx:i].min(),
                }
            )
            start_idx = None

    return pd.DataFrame(events)
