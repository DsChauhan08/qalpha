"""
Position Limits Module - Quantum Alpha V1
Exposure constraints per asset, sector, and portfolio per agent.md Section 5.

Implements:
- Position size limits
- Sector exposure limits
- Correlation-based limits
- Gross/net exposure limits
- Concentration limits
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of position limits."""

    POSITION = "position"  # Single position
    SECTOR = "sector"  # Sector exposure
    ASSET_CLASS = "asset_class"  # Asset class exposure
    GROSS = "gross"  # Total gross exposure
    NET = "net"  # Net exposure (long - short)
    CONCENTRATION = "concentration"  # Top N concentration
    CORRELATION = "correlation"  # Correlated positions
    LIQUIDITY = "liquidity"  # Liquidity-based


@dataclass
class LimitBreach:
    """Record of a limit breach."""

    timestamp: datetime
    limit_type: LimitType
    limit_name: str
    current_value: float
    limit_value: float
    excess: float
    severity: str  # warning, breach, critical
    positions_involved: List[str] = field(default_factory=list)


@dataclass
class PositionLimits:
    """Position limit configuration."""

    # Single position limits
    max_position_pct: float = 0.10  # Max 10% in any single position
    min_position_pct: float = 0.01  # Min 1% position size
    max_position_value: Optional[float] = None  # Absolute max value

    # Sector limits
    max_sector_pct: float = 0.30  # Max 30% in any sector

    # Asset class limits
    max_equity_pct: float = 0.80
    max_bond_pct: float = 0.50
    max_commodity_pct: float = 0.20
    max_cash_pct: float = 1.00

    # Gross/net limits
    max_gross_exposure: float = 1.50  # 150% gross (with leverage)
    min_gross_exposure: float = 0.50  # Min 50% invested
    max_net_long: float = 1.20  # Max 120% net long
    max_net_short: float = 0.50  # Max 50% net short

    # Concentration limits
    max_top5_concentration: float = 0.50  # Top 5 positions max 50%
    max_top10_concentration: float = 0.70  # Top 10 positions max 70%

    # Correlation limits
    max_correlated_exposure: float = 0.40  # Max 40% in highly correlated positions
    correlation_threshold: float = 0.70  # Define "highly correlated"

    # Liquidity limits
    max_illiquid_pct: float = 0.20  # Max 20% in illiquid positions
    min_avg_volume_days: float = 5.0  # Min 5 days avg volume to liquidate


class PositionLimitChecker:
    """
    Check positions against various limits.
    """

    # Sector mappings
    SECTOR_MAPPINGS = {
        "XLK": "Technology",
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Consumer Discretionary",
        "META": "Technology",
        "NVDA": "Technology",
        "QQQ": "Technology",
        "XLF": "Financials",
        "JPM": "Financials",
        "BAC": "Financials",
        "GS": "Financials",
        "XLV": "Healthcare",
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "PFE": "Healthcare",
        "XLE": "Energy",
        "XOM": "Energy",
        "CVX": "Energy",
        "XLI": "Industrials",
        "CAT": "Industrials",
        "BA": "Industrials",
        "XLP": "Consumer Staples",
        "WMT": "Consumer Staples",
        "PG": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "VNQ": "Real Estate",
        "XLC": "Communication Services",
        "TLT": "Bonds",
        "IEF": "Bonds",
        "AGG": "Bonds",
        "LQD": "Bonds",
        "HYG": "Bonds",
        "GLD": "Commodities",
        "SLV": "Commodities",
        "USO": "Commodities",
        "SPY": "Broad Market",
        "IWM": "Small Cap",
    }

    # Asset class mappings
    ASSET_CLASS_MAPPINGS = {
        "equity": [
            "XLK",
            "XLF",
            "XLV",
            "XLE",
            "XLI",
            "XLP",
            "XLY",
            "XLB",
            "XLU",
            "XLRE",
            "XLC",
            "SPY",
            "QQQ",
            "IWM",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "JPM",
            "BAC",
            "GS",
            "JNJ",
            "UNH",
            "PFE",
            "XOM",
            "CVX",
            "CAT",
            "BA",
            "WMT",
            "PG",
        ],
        "bonds": ["TLT", "IEF", "AGG", "LQD", "HYG", "BND", "GOVT"],
        "commodities": ["GLD", "SLV", "USO", "DBA", "DBC"],
        "cash": ["SHY", "BIL", "SGOV"],
    }

    def __init__(
        self,
        limits: Optional[PositionLimits] = None,
        sector_mappings: Optional[Dict[str, str]] = None,
        asset_class_mappings: Optional[Dict[str, List[str]]] = None,
    ):
        self.limits = limits or PositionLimits()
        self.sector_mappings = sector_mappings or self.SECTOR_MAPPINGS
        self.asset_class_mappings = asset_class_mappings or self.ASSET_CLASS_MAPPINGS

        # Reverse asset class mapping for quick lookup
        self._asset_to_class = {}
        for asset_class, symbols in self.asset_class_mappings.items():
            for symbol in symbols:
                self._asset_to_class[symbol.upper()] = asset_class

        # Breach history
        self.breach_history: List[LimitBreach] = []

    def check_all_limits(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        correlation_matrix: Optional[pd.DataFrame] = None,
        avg_volumes: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, List[LimitBreach]]:
        """
        Check all position limits.

        Args:
            positions: Symbol -> position value mapping
            portfolio_value: Total portfolio value
            correlation_matrix: Optional correlation matrix
            avg_volumes: Optional average daily volumes

        Returns:
            Tuple of (all_passed, list of breaches)
        """
        breaches = []

        # Single position checks
        breaches.extend(self._check_position_limits(positions, portfolio_value))

        # Sector checks
        breaches.extend(self._check_sector_limits(positions, portfolio_value))

        # Asset class checks
        breaches.extend(self._check_asset_class_limits(positions, portfolio_value))

        # Gross/net exposure checks
        breaches.extend(self._check_exposure_limits(positions, portfolio_value))

        # Concentration checks
        breaches.extend(self._check_concentration_limits(positions, portfolio_value))

        # Correlation checks
        if correlation_matrix is not None:
            breaches.extend(
                self._check_correlation_limits(
                    positions, portfolio_value, correlation_matrix
                )
            )

        # Liquidity checks
        if avg_volumes is not None:
            breaches.extend(
                self._check_liquidity_limits(positions, portfolio_value, avg_volumes)
            )

        # Store breaches
        self.breach_history.extend(breaches)

        all_passed = len([b for b in breaches if b.severity == "breach"]) == 0

        return all_passed, breaches

    def _check_position_limits(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[LimitBreach]:
        """Check individual position size limits."""
        breaches = []
        now = datetime.now()

        for symbol, value in positions.items():
            pct = abs(value) / portfolio_value if portfolio_value > 0 else 0

            # Check max position
            if pct > self.limits.max_position_pct:
                severity = (
                    "critical" if pct > self.limits.max_position_pct * 1.5 else "breach"
                )
                breaches.append(
                    LimitBreach(
                        timestamp=now,
                        limit_type=LimitType.POSITION,
                        limit_name=f"max_position_{symbol}",
                        current_value=pct,
                        limit_value=self.limits.max_position_pct,
                        excess=pct - self.limits.max_position_pct,
                        severity=severity,
                        positions_involved=[symbol],
                    )
                )

            # Check absolute max if set
            if self.limits.max_position_value:
                if abs(value) > self.limits.max_position_value:
                    breaches.append(
                        LimitBreach(
                            timestamp=now,
                            limit_type=LimitType.POSITION,
                            limit_name=f"max_position_value_{symbol}",
                            current_value=abs(value),
                            limit_value=self.limits.max_position_value,
                            excess=abs(value) - self.limits.max_position_value,
                            severity="breach",
                            positions_involved=[symbol],
                        )
                    )

        return breaches

    def _check_sector_limits(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[LimitBreach]:
        """Check sector exposure limits."""
        breaches = []
        now = datetime.now()

        # Calculate sector exposures
        sector_exposure: Dict[str, float] = {}
        sector_positions: Dict[str, List[str]] = {}

        for symbol, value in positions.items():
            sector = self.sector_mappings.get(symbol.upper(), "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(value)
            if sector not in sector_positions:
                sector_positions[sector] = []
            sector_positions[sector].append(symbol)

        # Check limits
        for sector, exposure in sector_exposure.items():
            pct = exposure / portfolio_value if portfolio_value > 0 else 0

            if pct > self.limits.max_sector_pct:
                severity = (
                    "critical" if pct > self.limits.max_sector_pct * 1.5 else "breach"
                )
                breaches.append(
                    LimitBreach(
                        timestamp=now,
                        limit_type=LimitType.SECTOR,
                        limit_name=f"max_sector_{sector}",
                        current_value=pct,
                        limit_value=self.limits.max_sector_pct,
                        excess=pct - self.limits.max_sector_pct,
                        severity=severity,
                        positions_involved=sector_positions[sector],
                    )
                )

        return breaches

    def _check_asset_class_limits(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[LimitBreach]:
        """Check asset class exposure limits."""
        breaches = []
        now = datetime.now()

        # Calculate asset class exposures
        class_exposure = {"equity": 0.0, "bonds": 0.0, "commodities": 0.0, "cash": 0.0}
        class_positions: Dict[str, List[str]] = {k: [] for k in class_exposure}

        for symbol, value in positions.items():
            asset_class = self._asset_to_class.get(symbol.upper(), "equity")
            class_exposure[asset_class] = class_exposure.get(asset_class, 0) + abs(
                value
            )
            if asset_class in class_positions:
                class_positions[asset_class].append(symbol)

        # Check limits
        limits_map = {
            "equity": self.limits.max_equity_pct,
            "bonds": self.limits.max_bond_pct,
            "commodities": self.limits.max_commodity_pct,
            "cash": self.limits.max_cash_pct,
        }

        for asset_class, exposure in class_exposure.items():
            pct = exposure / portfolio_value if portfolio_value > 0 else 0
            limit = limits_map.get(asset_class, 1.0)

            if pct > limit:
                breaches.append(
                    LimitBreach(
                        timestamp=now,
                        limit_type=LimitType.ASSET_CLASS,
                        limit_name=f"max_{asset_class}",
                        current_value=pct,
                        limit_value=limit,
                        excess=pct - limit,
                        severity="breach",
                        positions_involved=class_positions.get(asset_class, []),
                    )
                )

        return breaches

    def _check_exposure_limits(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[LimitBreach]:
        """Check gross and net exposure limits."""
        breaches = []
        now = datetime.now()

        if portfolio_value <= 0:
            return breaches

        # Calculate exposures
        long_exposure = sum(v for v in positions.values() if v > 0)
        short_exposure = sum(abs(v) for v in positions.values() if v < 0)
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        gross_pct = gross_exposure / portfolio_value
        net_long_pct = net_exposure / portfolio_value if net_exposure > 0 else 0
        net_short_pct = abs(net_exposure) / portfolio_value if net_exposure < 0 else 0

        # Check gross exposure max
        if gross_pct > self.limits.max_gross_exposure:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.GROSS,
                    limit_name="max_gross_exposure",
                    current_value=gross_pct,
                    limit_value=self.limits.max_gross_exposure,
                    excess=gross_pct - self.limits.max_gross_exposure,
                    severity="breach",
                    positions_involved=list(positions.keys()),
                )
            )

        # Check gross exposure min
        if gross_pct < self.limits.min_gross_exposure:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.GROSS,
                    limit_name="min_gross_exposure",
                    current_value=gross_pct,
                    limit_value=self.limits.min_gross_exposure,
                    excess=self.limits.min_gross_exposure - gross_pct,
                    severity="warning",
                    positions_involved=list(positions.keys()),
                )
            )

        # Check net long
        if net_long_pct > self.limits.max_net_long:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.NET,
                    limit_name="max_net_long",
                    current_value=net_long_pct,
                    limit_value=self.limits.max_net_long,
                    excess=net_long_pct - self.limits.max_net_long,
                    severity="breach",
                    positions_involved=[s for s, v in positions.items() if v > 0],
                )
            )

        # Check net short
        if net_short_pct > self.limits.max_net_short:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.NET,
                    limit_name="max_net_short",
                    current_value=net_short_pct,
                    limit_value=self.limits.max_net_short,
                    excess=net_short_pct - self.limits.max_net_short,
                    severity="breach",
                    positions_involved=[s for s, v in positions.items() if v < 0],
                )
            )

        return breaches

    def _check_concentration_limits(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[LimitBreach]:
        """Check concentration limits."""
        breaches = []
        now = datetime.now()

        if portfolio_value <= 0 or not positions:
            return breaches

        # Sort positions by absolute value
        sorted_positions = sorted(
            positions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Top 5 concentration
        top5_value = sum(abs(v) for _, v in sorted_positions[:5])
        top5_pct = top5_value / portfolio_value

        if top5_pct > self.limits.max_top5_concentration:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.CONCENTRATION,
                    limit_name="max_top5_concentration",
                    current_value=top5_pct,
                    limit_value=self.limits.max_top5_concentration,
                    excess=top5_pct - self.limits.max_top5_concentration,
                    severity="breach",
                    positions_involved=[s for s, _ in sorted_positions[:5]],
                )
            )

        # Top 10 concentration
        top10_value = sum(abs(v) for _, v in sorted_positions[:10])
        top10_pct = top10_value / portfolio_value

        if top10_pct > self.limits.max_top10_concentration:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.CONCENTRATION,
                    limit_name="max_top10_concentration",
                    current_value=top10_pct,
                    limit_value=self.limits.max_top10_concentration,
                    excess=top10_pct - self.limits.max_top10_concentration,
                    severity="warning",
                    positions_involved=[s for s, _ in sorted_positions[:10]],
                )
            )

        return breaches

    def _check_correlation_limits(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        correlation_matrix: pd.DataFrame,
    ) -> List[LimitBreach]:
        """Check correlation-based limits."""
        breaches = []
        now = datetime.now()

        if portfolio_value <= 0:
            return breaches

        # Find highly correlated position groups
        symbols = [s for s in positions.keys() if s in correlation_matrix.columns]

        if len(symbols) < 2:
            return breaches

        # Build correlation groups
        correlated_groups: List[Set[str]] = []
        visited = set()

        for i, sym1 in enumerate(symbols):
            if sym1 in visited:
                continue

            group = {sym1}
            for j, sym2 in enumerate(symbols):
                if i != j and sym2 not in visited:
                    try:
                        corr = correlation_matrix.loc[sym1, sym2]
                        if abs(corr) >= self.limits.correlation_threshold:
                            group.add(sym2)
                    except KeyError:
                        continue

            if len(group) > 1:
                correlated_groups.append(group)
                visited.update(group)

        # Check exposure in each correlated group
        for group in correlated_groups:
            group_exposure = sum(abs(positions.get(s, 0)) for s in group)
            group_pct = group_exposure / portfolio_value

            if group_pct > self.limits.max_correlated_exposure:
                breaches.append(
                    LimitBreach(
                        timestamp=now,
                        limit_type=LimitType.CORRELATION,
                        limit_name="max_correlated_exposure",
                        current_value=group_pct,
                        limit_value=self.limits.max_correlated_exposure,
                        excess=group_pct - self.limits.max_correlated_exposure,
                        severity="warning",
                        positions_involved=list(group),
                    )
                )

        return breaches

    def _check_liquidity_limits(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        avg_volumes: Dict[str, float],
    ) -> List[LimitBreach]:
        """Check liquidity-based limits."""
        breaches = []
        now = datetime.now()

        if portfolio_value <= 0:
            return breaches

        illiquid_exposure = 0.0
        illiquid_positions = []

        for symbol, value in positions.items():
            avg_vol = avg_volumes.get(symbol, 0)

            if avg_vol > 0:
                # Days to liquidate = position size / avg daily volume
                days_to_liquidate = abs(value) / avg_vol

                if days_to_liquidate > self.limits.min_avg_volume_days:
                    illiquid_exposure += abs(value)
                    illiquid_positions.append(symbol)

        illiquid_pct = illiquid_exposure / portfolio_value

        if illiquid_pct > self.limits.max_illiquid_pct:
            breaches.append(
                LimitBreach(
                    timestamp=now,
                    limit_type=LimitType.LIQUIDITY,
                    limit_name="max_illiquid_exposure",
                    current_value=illiquid_pct,
                    limit_value=self.limits.max_illiquid_pct,
                    excess=illiquid_pct - self.limits.max_illiquid_pct,
                    severity="warning",
                    positions_involved=illiquid_positions,
                )
            )

        return breaches

    def get_position_capacity(
        self, symbol: str, current_positions: Dict[str, float], portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate remaining capacity for a position.

        Returns:
            Dict with max_additional_long and max_additional_short
        """
        current_value = current_positions.get(symbol, 0)
        current_pct = abs(current_value) / portfolio_value if portfolio_value > 0 else 0

        # Position limit capacity
        position_capacity = (
            self.limits.max_position_pct - current_pct
        ) * portfolio_value

        # Sector capacity
        sector = self.sector_mappings.get(symbol.upper(), "Unknown")
        sector_exposure = sum(
            abs(v)
            for s, v in current_positions.items()
            if self.sector_mappings.get(s.upper(), "Unknown") == sector
        )
        sector_pct = sector_exposure / portfolio_value if portfolio_value > 0 else 0
        sector_capacity = (self.limits.max_sector_pct - sector_pct) * portfolio_value

        # Gross exposure capacity
        gross = sum(abs(v) for v in current_positions.values())
        gross_pct = gross / portfolio_value if portfolio_value > 0 else 0
        gross_capacity = (self.limits.max_gross_exposure - gross_pct) * portfolio_value

        # Take minimum of all limits
        max_additional = min(position_capacity, sector_capacity, gross_capacity)

        return {
            "max_additional_long": max(0, max_additional),
            "max_additional_short": max(0, max_additional),
            "position_capacity": position_capacity,
            "sector_capacity": sector_capacity,
            "gross_capacity": gross_capacity,
            "limiting_factor": "position"
            if position_capacity <= sector_capacity
            and position_capacity <= gross_capacity
            else "sector"
            if sector_capacity <= gross_capacity
            else "gross",
        }

    def suggest_rebalance(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        breaches: List[LimitBreach],
    ) -> List[Dict[str, Any]]:
        """
        Suggest trades to fix limit breaches.

        Returns:
            List of suggested trades
        """
        suggestions = []

        for breach in breaches:
            if breach.severity not in ["breach", "critical"]:
                continue

            if breach.limit_type == LimitType.POSITION:
                # Reduce oversized position
                symbol = (
                    breach.positions_involved[0] if breach.positions_involved else None
                )
                if symbol:
                    current_value = positions.get(symbol, 0)
                    target_value = self.limits.max_position_pct * portfolio_value
                    reduce_by = abs(current_value) - target_value

                    suggestions.append(
                        {
                            "action": "reduce",
                            "symbol": symbol,
                            "current_value": current_value,
                            "target_value": target_value * np.sign(current_value),
                            "trade_value": -reduce_by * np.sign(current_value),
                            "reason": f"Position exceeds {self.limits.max_position_pct:.0%} limit",
                        }
                    )

            elif breach.limit_type == LimitType.SECTOR:
                # Reduce sector exposure
                excess_value = breach.excess * portfolio_value

                # Suggest reducing largest position in sector
                sector_positions = [
                    (s, v)
                    for s, v in positions.items()
                    if s in breach.positions_involved
                ]
                if sector_positions:
                    largest = max(sector_positions, key=lambda x: abs(x[1]))
                    suggestions.append(
                        {
                            "action": "reduce_sector",
                            "symbol": largest[0],
                            "sector": breach.limit_name.replace("max_sector_", ""),
                            "current_value": largest[1],
                            "reduce_by": min(abs(largest[1]), excess_value),
                            "reason": f"Sector exposure exceeds {self.limits.max_sector_pct:.0%} limit",
                        }
                    )

            elif breach.limit_type == LimitType.GROSS:
                # Reduce gross exposure
                excess_value = breach.excess * portfolio_value

                suggestions.append(
                    {
                        "action": "reduce_gross",
                        "reduce_by": excess_value,
                        "positions": breach.positions_involved,
                        "reason": f"Gross exposure exceeds {self.limits.max_gross_exposure:.0%} limit",
                    }
                )

        return suggestions

    def get_limit_summary(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> Dict[str, Any]:
        """Get summary of current limit utilization."""
        if portfolio_value <= 0:
            return {}

        # Calculate exposures
        long_exp = sum(v for v in positions.values() if v > 0)
        short_exp = sum(abs(v) for v in positions.values() if v < 0)
        gross_exp = long_exp + short_exp
        net_exp = long_exp - short_exp

        # Sector exposures
        sector_exp = {}
        for symbol, value in positions.items():
            sector = self.sector_mappings.get(symbol.upper(), "Unknown")
            sector_exp[sector] = sector_exp.get(sector, 0) + abs(value)

        # Largest positions
        sorted_pos = sorted(positions.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "gross_exposure": {
                "current": gross_exp / portfolio_value,
                "limit": self.limits.max_gross_exposure,
                "utilization": (gross_exp / portfolio_value)
                / self.limits.max_gross_exposure,
            },
            "net_exposure": {
                "current": net_exp / portfolio_value,
                "max_long": self.limits.max_net_long,
                "max_short": self.limits.max_net_short,
            },
            "largest_position": {
                "symbol": sorted_pos[0][0] if sorted_pos else None,
                "pct": abs(sorted_pos[0][1]) / portfolio_value if sorted_pos else 0,
                "limit": self.limits.max_position_pct,
            },
            "sector_exposures": {
                s: {"pct": v / portfolio_value, "limit": self.limits.max_sector_pct}
                for s, v in sector_exp.items()
            },
            "top5_concentration": {
                "pct": sum(abs(v) for _, v in sorted_pos[:5]) / portfolio_value,
                "limit": self.limits.max_top5_concentration,
            },
        }


def validate_new_position(
    symbol: str,
    trade_value: float,
    current_positions: Dict[str, float],
    portfolio_value: float,
    limits: Optional[PositionLimits] = None,
) -> Tuple[bool, str]:
    """
    Quick validation of a new position.

    Returns:
        Tuple of (is_valid, reason)
    """
    checker = PositionLimitChecker(limits)

    # Simulate new positions
    new_positions = current_positions.copy()
    new_positions[symbol] = new_positions.get(symbol, 0) + trade_value

    # Check limits
    passed, breaches = checker.check_all_limits(new_positions, portfolio_value)

    if passed:
        return True, "Position within all limits"

    # Find the breach
    critical_breaches = [b for b in breaches if b.severity in ["breach", "critical"]]
    if critical_breaches:
        breach = critical_breaches[0]
        return (
            False,
            f"{breach.limit_name}: {breach.current_value:.1%} exceeds {breach.limit_value:.1%}",
        )

    return True, "Position within limits (warnings only)"
