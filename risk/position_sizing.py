"""
Risk Management Module - V1
Position sizing with Kelly Criterion, VaR, and drawdown control.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
from scipy import stats


@dataclass
class KellyResult:
    """Kelly Criterion calculation result."""

    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    win_rate: float
    win_loss_ratio: float
    expected_value: float
    confidence: float


class KellyCriterion:
    """
    Kelly Criterion position sizing.

    Kelly Formula: f* = (p(b+1) - 1) / b
    where:
        f* = optimal fraction of capital
        p  = probability of win
        b  = win/loss ratio (odds)
    """

    def __init__(self, max_position: float = 0.25, confidence_threshold: float = 0.5):
        self.max_position = max_position
        self.confidence_threshold = confidence_threshold

    def calculate(
        self, trade_returns: np.ndarray, signal_confidence: float = 0.5
    ) -> KellyResult:
        """
        Calculate Kelly fraction from trade history.

        Args:
            trade_returns: Array of trade returns
            signal_confidence: Model confidence (0-1)

        Returns:
            KellyResult with position sizing recommendations
        """
        if len(trade_returns) < 10:
            return KellyResult(0, 0, 0, 0, 0, 0, 0)

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        if len(losses) == 0 or len(wins) == 0:
            return KellyResult(0, 0, 0, 0, 0, 0, 0)

        # Core parameters
        win_rate = len(wins) / len(trade_returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return KellyResult(0, 0, 0, win_rate, 0, 0, 0)

        win_loss_ratio = avg_win / avg_loss

        # Kelly formula
        kelly = (win_rate * (win_loss_ratio + 1) - 1) / win_loss_ratio

        # Adjust for signal confidence
        kelly *= signal_confidence

        # Apply safety caps
        kelly = np.clip(kelly, 0, self.max_position)

        # Expected value
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Confidence based on sample size
        sample_confidence = min(len(trade_returns) / 100, 1.0)

        return KellyResult(
            full_kelly=kelly,
            half_kelly=kelly / 2,
            quarter_kelly=kelly / 4,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            expected_value=ev,
            confidence=sample_confidence,
        )


class VaRCalculator:
    """
    Value-at-Risk calculator with multiple methods.
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def historical(self, returns: np.ndarray, portfolio_value: float = 1.0) -> float:
        """Historical VaR using empirical quantile."""
        var_pct = np.percentile(returns, (1 - self.confidence) * 100)
        return abs(var_pct * portfolio_value)

    def parametric(self, returns: np.ndarray, portfolio_value: float = 1.0) -> float:
        """Parametric VaR assuming normal distribution."""
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(1 - self.confidence)
        var_ret = mu + z * sigma
        return abs(var_ret * portfolio_value)

    def cornish_fisher(
        self, returns: np.ndarray, portfolio_value: float = 1.0
    ) -> float:
        """Cornish-Fisher VaR adjusting for skewness/kurtosis."""
        mu = returns.mean()
        sigma = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        z = stats.norm.ppf(1 - self.confidence)

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )

        var_ret = mu + z_cf * sigma
        return abs(var_ret * portfolio_value)

    def calculate_all(
        self, returns: np.ndarray, portfolio_value: float = 1.0
    ) -> Dict[str, float]:
        """Calculate VaR using all methods."""
        return {
            "historical": self.historical(returns, portfolio_value),
            "parametric": self.parametric(returns, portfolio_value),
            "cornish_fisher": self.cornish_fisher(returns, portfolio_value),
            "confidence": self.confidence,
        }


class CVaRCalculator:
    """Conditional VaR (Expected Shortfall)."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def calculate(self, returns: np.ndarray, portfolio_value: float = 1.0) -> float:
        """
        Calculate CVaR - expected loss given VaR is exceeded.
        """
        var_threshold = np.percentile(returns, (1 - self.confidence) * 100)
        tail = returns[returns <= var_threshold]

        if len(tail) == 0:
            return abs(var_threshold * portfolio_value)

        return abs(tail.mean() * portfolio_value)


class DrawdownController:
    """
    Dynamic position scaling based on drawdown.
    """

    def __init__(self, max_drawdown: float = 0.10):
        self.max_dd = max_drawdown
        self.steps = [
            (0.05, 1.0),  # 0-5% DD: full exposure
            (0.07, 0.75),  # 5-7% DD: 75%
            (0.10, 0.50),  # 7-10% DD: 50%
            (0.15, 0.25),  # 10-15% DD: 25%
            (float("inf"), 0.0),
        ]

    def get_multiplier(self, current_drawdown: float) -> float:
        """Get exposure multiplier based on drawdown."""
        dd = abs(current_drawdown)
        for threshold, mult in self.steps:
            if dd <= threshold:
                return mult
        return 0.0

    def should_halt(self, current_drawdown: float) -> bool:
        """Check if trading should stop."""
        return abs(current_drawdown) >= self.max_dd * 1.5


class RiskParity:
    """
    Risk parity position sizing - equal risk contribution.
    """

    def __init__(self, target_volatility: float = 0.10):
        self.target_vol = target_volatility

    def calculate_weights(
        self, volatilities: Dict[str, float], correlations: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights.

        Args:
            volatilities: Dict of asset -> annualized volatility
            correlations: Correlation matrix (identity if None)

        Returns:
            Dict of asset -> weight
        """
        assets = list(volatilities.keys())
        n = len(assets)

        if correlations is None:
            correlations = np.eye(n)

        # Inverse volatility weights
        vol_array = np.array([volatilities[a] for a in assets])
        inv_vol = 1 / np.where(vol_array > 0, vol_array, 1)
        weights = inv_vol / inv_vol.sum()

        # Scale to target vol
        cov = np.outer(vol_array, vol_array) * correlations
        port_vol = np.sqrt(weights @ cov @ weights)

        if port_vol > 0:
            scale = self.target_vol / port_vol
            weights = weights * scale

        return {asset: float(weights[i]) for i, asset in enumerate(assets)}


class PositionSizer:
    """
    Unified position sizer combining multiple methods.
    """

    def __init__(
        self,
        kelly_weight: float = 0.4,
        vol_weight: float = 0.4,
        max_position: float = 0.25,
    ):
        self.kelly = KellyCriterion(max_position)
        self.var = VaRCalculator()
        self.dd_control = DrawdownController()
        self.kelly_weight = kelly_weight
        self.vol_weight = vol_weight
        self.max_position = max_position

    def calculate(
        self,
        trade_history: np.ndarray,
        current_volatility: float,
        current_drawdown: float,
        signal_strength: float,
        signal_confidence: float,
    ) -> Dict:
        """
        Calculate position size combining multiple methods.

        Args:
            trade_history: Historical trade returns
            current_volatility: Current annualized volatility
            current_drawdown: Current portfolio drawdown
            signal_strength: Signal strength (-1 to 1)
            signal_confidence: Confidence in signal (0-1)

        Returns:
            Dict with position size and components
        """
        # Kelly component
        kelly_result = self.kelly.calculate(trade_history, signal_confidence)
        kelly_size = kelly_result.half_kelly * abs(signal_strength)

        # Volatility scaling
        vol_scale = 0.20 / current_volatility if current_volatility > 0 else 1.0
        vol_size = abs(signal_strength) * vol_scale * 0.10

        # Drawdown adjustment
        dd_mult = self.dd_control.get_multiplier(current_drawdown)

        # Combine
        raw_size = kelly_size * self.kelly_weight + vol_size * self.vol_weight

        # Apply drawdown control
        adjusted_size = raw_size * dd_mult

        # Apply direction from signal
        final_size = adjusted_size * np.sign(signal_strength)

        # Cap at max
        final_size = np.clip(final_size, -self.max_position, self.max_position)

        return {
            "position_size": float(final_size),
            "kelly_component": float(kelly_size),
            "vol_component": float(vol_size),
            "dd_multiplier": float(dd_mult),
            "signal_strength": float(signal_strength),
            "signal_confidence": float(signal_confidence),
            "halt_trading": self.dd_control.should_halt(current_drawdown),
        }
