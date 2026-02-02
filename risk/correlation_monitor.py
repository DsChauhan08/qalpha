"""
Correlation Monitor Module - Quantum Alpha V1
Real-time correlation tracking and diversification metrics per agent.md Section 5.

Implements:
- Rolling correlation matrices
- Correlation regime detection
- Diversification ratio calculation
- Concentration risk metrics
- Correlation breakdown alerts
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import logging
from collections import deque

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Correlation regime states."""

    LOW = "low"  # Correlations below normal
    NORMAL = "normal"  # Correlations at historical average
    ELEVATED = "elevated"  # Correlations above normal
    CRISIS = "crisis"  # Correlations near 1 (risk-off)


@dataclass
class CorrelationAlert:
    """Alert for correlation changes."""

    timestamp: datetime
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics."""

    diversification_ratio: float  # >1 means diversified
    effective_n: float  # Effective number of independent bets
    concentration_ratio: float  # HHI-like concentration
    max_correlation: float
    avg_correlation: float
    correlation_regime: CorrelationRegime
    marginal_contributions: Dict[str, float] = field(default_factory=dict)


class CorrelationMonitor:
    """
    Real-time correlation monitoring and diversification analysis.
    """

    def __init__(
        self,
        lookback_short: int = 20,
        lookback_medium: int = 60,
        lookback_long: int = 252,
        correlation_threshold_high: float = 0.70,
        correlation_threshold_crisis: float = 0.85,
        update_frequency: int = 1,  # Update every N observations
        history_size: int = 1000,
    ):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.correlation_threshold_high = correlation_threshold_high
        self.correlation_threshold_crisis = correlation_threshold_crisis
        self.update_frequency = update_frequency

        # State tracking
        self.returns_buffer: deque = deque(maxlen=history_size)
        self.correlation_history: deque = deque(maxlen=history_size)
        self.alerts: List[CorrelationAlert] = []
        self.observation_count = 0

        # Cached calculations
        self._cached_corr_matrix: Optional[pd.DataFrame] = None
        self._cached_regime: Optional[CorrelationRegime] = None
        self._last_update: Optional[datetime] = None

    def update(self, returns: pd.Series) -> Optional[DiversificationMetrics]:
        """
        Update with new return observation.

        Args:
            returns: Series of returns for all assets

        Returns:
            DiversificationMetrics if update performed, None otherwise
        """
        self.returns_buffer.append(returns)
        self.observation_count += 1

        # Check if we should update calculations
        if self.observation_count % self.update_frequency != 0:
            return None

        if len(self.returns_buffer) < self.lookback_short:
            return None

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Check for alerts
        self._check_alerts(metrics)

        return metrics

    def calculate_metrics(
        self, returns: Optional[pd.DataFrame] = None
    ) -> DiversificationMetrics:
        """
        Calculate comprehensive diversification metrics.

        Args:
            returns: Optional DataFrame of returns (uses buffer if None)

        Returns:
            DiversificationMetrics
        """
        if returns is None:
            if len(self.returns_buffer) < self.lookback_short:
                raise ValueError("Insufficient data in buffer")
            returns = pd.DataFrame(list(self.returns_buffer))

        # Calculate correlation matrix (short-term)
        corr_matrix = returns.tail(self.lookback_short).corr()
        self._cached_corr_matrix = corr_matrix

        # Get correlation statistics
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, np.nan)

        avg_corr = np.nanmean(corr_values)
        max_corr = np.nanmax(np.abs(corr_values))

        # Determine regime
        regime = self._determine_regime(avg_corr, max_corr)
        self._cached_regime = regime

        # Calculate diversification ratio
        div_ratio = self._calculate_diversification_ratio(returns)

        # Calculate effective N (number of independent bets)
        effective_n = self._calculate_effective_n(corr_matrix)

        # Calculate concentration ratio
        concentration = self._calculate_concentration(corr_matrix)

        # Store in history
        self.correlation_history.append(
            {
                "timestamp": datetime.now(),
                "avg_correlation": avg_corr,
                "max_correlation": max_corr,
                "regime": regime,
                "diversification_ratio": div_ratio,
            }
        )

        return DiversificationMetrics(
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            concentration_ratio=concentration,
            max_correlation=max_corr,
            avg_correlation=avg_corr,
            correlation_regime=regime,
        )

    def _determine_regime(self, avg_corr: float, max_corr: float) -> CorrelationRegime:
        """Determine current correlation regime."""
        if max_corr >= self.correlation_threshold_crisis:
            return CorrelationRegime.CRISIS
        elif avg_corr >= self.correlation_threshold_high:
            return CorrelationRegime.ELEVATED
        elif avg_corr >= 0.30:
            return CorrelationRegime.NORMAL
        else:
            return CorrelationRegime.LOW

    def _calculate_diversification_ratio(self, returns: pd.DataFrame) -> float:
        """
        Calculate diversification ratio.
        DR = (sum of individual volatilities) / (portfolio volatility)
        DR > 1 indicates diversification benefit.
        """
        # Individual volatilities
        individual_vols = returns.std()

        # Equal-weighted portfolio volatility
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        cov_matrix = returns.cov()

        portfolio_var = weights @ cov_matrix.values @ weights
        portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 1e-8

        # Diversification ratio
        weighted_vol_sum = np.sum(individual_vols * weights)
        div_ratio = weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 1.0

        return float(div_ratio)

    def _calculate_effective_n(self, corr_matrix: pd.DataFrame) -> float:
        """
        Calculate effective number of independent bets.
        Uses eigenvalue decomposition of correlation matrix.
        """
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

        # Normalize eigenvalues
        total = np.sum(eigenvalues)
        if total <= 0:
            return len(corr_matrix)

        proportions = eigenvalues / total

        # Effective N using entropy-based measure
        # Higher entropy = more independent factors
        proportions = proportions[proportions > 1e-10]
        entropy = -np.sum(proportions * np.log(proportions))
        effective_n = np.exp(entropy)

        return float(effective_n)

    def _calculate_concentration(self, corr_matrix: pd.DataFrame) -> float:
        """
        Calculate concentration ratio based on correlation structure.
        Similar to Herfindahl-Hirschman Index.
        """
        n = len(corr_matrix)
        if n <= 1:
            return 1.0

        # Use average absolute correlation as proxy for concentration
        corr_values = corr_matrix.values.copy()
        np.fill_diagonal(corr_values, 0)

        avg_abs_corr = np.mean(np.abs(corr_values))

        # Scale to 0-1 range (0 = fully diversified, 1 = fully concentrated)
        concentration = avg_abs_corr

        return float(concentration)

    def _check_alerts(self, metrics: DiversificationMetrics):
        """Check for correlation-related alerts."""
        now = datetime.now()

        # Crisis regime alert
        if metrics.correlation_regime == CorrelationRegime.CRISIS:
            self.alerts.append(
                CorrelationAlert(
                    timestamp=now,
                    alert_type="correlation_crisis",
                    severity="critical",
                    message=f"Correlation crisis detected! Avg: {metrics.avg_correlation:.2f}, Max: {metrics.max_correlation:.2f}",
                    details={
                        "avg_correlation": metrics.avg_correlation,
                        "max_correlation": metrics.max_correlation,
                        "diversification_ratio": metrics.diversification_ratio,
                    },
                )
            )

        # Low diversification alert
        elif metrics.diversification_ratio < 1.1:
            self.alerts.append(
                CorrelationAlert(
                    timestamp=now,
                    alert_type="low_diversification",
                    severity="high",
                    message=f"Low diversification ratio: {metrics.diversification_ratio:.2f}",
                    details={
                        "diversification_ratio": metrics.diversification_ratio,
                        "effective_n": metrics.effective_n,
                    },
                )
            )

        # Elevated correlation alert
        elif metrics.correlation_regime == CorrelationRegime.ELEVATED:
            self.alerts.append(
                CorrelationAlert(
                    timestamp=now,
                    alert_type="elevated_correlation",
                    severity="medium",
                    message=f"Elevated correlations detected: {metrics.avg_correlation:.2f}",
                    details={
                        "avg_correlation": metrics.avg_correlation,
                        "regime": metrics.correlation_regime.value,
                    },
                )
            )

    def get_correlation_matrix(self, lookback: Optional[int] = None) -> pd.DataFrame:
        """Get correlation matrix for specified lookback."""
        if len(self.returns_buffer) == 0:
            return pd.DataFrame()

        returns = pd.DataFrame(list(self.returns_buffer))
        lookback = lookback or self.lookback_short

        return returns.tail(lookback).corr()

    def get_rolling_correlations(
        self, asset1: str, asset2: str, window: int = 20
    ) -> pd.Series:
        """Get rolling correlation between two assets."""
        if len(self.returns_buffer) < window:
            return pd.Series()

        returns = pd.DataFrame(list(self.returns_buffer))

        if asset1 not in returns.columns or asset2 not in returns.columns:
            return pd.Series()

        return returns[asset1].rolling(window).corr(returns[asset2])

    def get_correlation_changes(
        self, short_window: int = 20, long_window: int = 60
    ) -> Dict[str, float]:
        """
        Detect significant correlation changes.
        Compares short-term vs long-term correlations.
        """
        if len(self.returns_buffer) < long_window:
            return {}

        returns = pd.DataFrame(list(self.returns_buffer))

        corr_short = returns.tail(short_window).corr()
        corr_long = returns.tail(long_window).corr()

        # Calculate changes
        changes = {}
        for i in range(len(corr_short.columns)):
            for j in range(i + 1, len(corr_short.columns)):
                asset1 = corr_short.columns[i]
                asset2 = corr_short.columns[j]

                short_corr = corr_short.iloc[i, j]
                long_corr = corr_long.iloc[i, j]

                change = short_corr - long_corr
                if abs(change) > 0.2:  # Significant change threshold
                    changes[f"{asset1}/{asset2}"] = change

        return changes

    def get_highly_correlated_pairs(
        self, threshold: float = 0.70
    ) -> List[Tuple[str, str, float]]:
        """Get pairs with correlation above threshold."""
        if self._cached_corr_matrix is None:
            return []

        corr = self._cached_corr_matrix
        pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                correlation = corr.iloc[i, j]
                if abs(correlation) >= threshold:
                    pairs.append((corr.columns[i], corr.columns[j], correlation))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def get_regime_history(self) -> pd.DataFrame:
        """Get history of correlation regimes."""
        if not self.correlation_history:
            return pd.DataFrame()

        return pd.DataFrame(list(self.correlation_history))

    def get_recent_alerts(
        self, n: int = 10, severity: Optional[str] = None
    ) -> List[CorrelationAlert]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self.alerts[-n:] if n else self.alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()


class SectorCorrelationAnalyzer:
    """
    Analyze correlations at sector level for better diversification.
    """

    # Default sector mappings
    SECTOR_MAPPINGS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
        # Individual stocks
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Consumer Discretionary",
        "META": "Technology",
        "NVDA": "Technology",
        "JPM": "Financials",
        "BAC": "Financials",
        "GS": "Financials",
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "PFE": "Healthcare",
        "XOM": "Energy",
        "CVX": "Energy",
        "WMT": "Consumer Staples",
        "PG": "Consumer Staples",
        "KO": "Consumer Staples",
        # ETFs
        "SPY": "Broad Market",
        "QQQ": "Technology",
        "IWM": "Small Cap",
        "TLT": "Bonds",
        "IEF": "Bonds",
        "AGG": "Bonds",
        "GLD": "Commodities",
        "SLV": "Commodities",
        "USO": "Commodities",
    }

    def __init__(self, sector_mappings: Optional[Dict[str, str]] = None):
        self.sector_mappings = sector_mappings or self.SECTOR_MAPPINGS

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.sector_mappings.get(symbol.upper(), "Unknown")

    def calculate_sector_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations aggregated by sector."""
        # Map columns to sectors
        sector_returns = {}

        for col in returns.columns:
            sector = self.get_sector(col)
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(returns[col])

        # Average returns within each sector
        sector_avg_returns = {}
        for sector, ret_list in sector_returns.items():
            sector_avg_returns[sector] = pd.concat(ret_list, axis=1).mean(axis=1)

        sector_df = pd.DataFrame(sector_avg_returns)

        return sector_df.corr()

    def get_sector_exposures(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate exposure to each sector."""
        total_value = sum(positions.values())
        if total_value == 0:
            return {}

        sector_exposure = {}
        for symbol, value in positions.items():
            sector = self.get_sector(symbol)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value

        # Convert to percentages
        return {s: v / total_value for s, v in sector_exposure.items()}

    def get_diversification_suggestions(
        self, positions: Dict[str, float], returns: pd.DataFrame
    ) -> List[str]:
        """Get suggestions for improving diversification."""
        suggestions = []

        # Check sector concentration
        sector_exposure = self.get_sector_exposures(positions)
        for sector, exposure in sector_exposure.items():
            if exposure > 0.30:
                suggestions.append(
                    f"High concentration in {sector}: {exposure:.1%}. "
                    f"Consider reducing exposure."
                )

        # Check for missing sectors
        present_sectors = set(sector_exposure.keys())
        all_sectors = set(self.sector_mappings.values())
        missing = all_sectors - present_sectors - {"Unknown"}

        if missing:
            suggestions.append(
                f"Missing sector exposure: {', '.join(list(missing)[:3])}. "
                f"Consider adding for better diversification."
            )

        # Check sector correlations
        sector_corr = self.calculate_sector_correlations(returns)
        high_corr_sectors = []

        for i, sector1 in enumerate(sector_corr.columns):
            for j, sector2 in enumerate(sector_corr.columns):
                if i < j and sector_corr.iloc[i, j] > 0.7:
                    if sector1 in sector_exposure and sector2 in sector_exposure:
                        high_corr_sectors.append(
                            (sector1, sector2, sector_corr.iloc[i, j])
                        )

        for s1, s2, corr in high_corr_sectors[:3]:
            suggestions.append(
                f"{s1} and {s2} highly correlated ({corr:.2f}). "
                f"Exposure to both may reduce diversification."
            )

        return suggestions


def calculate_portfolio_diversification(
    returns: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenience function for quick diversification analysis.

    Args:
        returns: DataFrame of asset returns
        weights: Optional position weights (equal weight if None)

    Returns:
        Dict with diversification metrics
    """
    monitor = CorrelationMonitor()

    # Feed all returns
    for _, row in returns.iterrows():
        monitor.update(row)

    metrics = monitor.calculate_metrics(returns)

    # Get highly correlated pairs
    high_corr_pairs = monitor.get_highly_correlated_pairs(threshold=0.70)

    return {
        "diversification_ratio": metrics.diversification_ratio,
        "effective_n": metrics.effective_n,
        "concentration_ratio": metrics.concentration_ratio,
        "avg_correlation": metrics.avg_correlation,
        "max_correlation": metrics.max_correlation,
        "regime": metrics.correlation_regime.value,
        "highly_correlated_pairs": high_corr_pairs[:5],
        "n_assets": len(returns.columns),
    }
