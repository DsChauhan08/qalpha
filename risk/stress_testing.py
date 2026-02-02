"""
Stress Testing Module - Quantum Alpha V1
Scenario analysis and historical stress tests per agent.md Section 5.

Implements:
- Historical scenario replay (2008, 2020, Flash Crash, etc.)
- Hypothetical scenario generation
- Monte Carlo stress simulation
- Portfolio impact analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of stress scenarios."""

    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    REVERSE = "reverse"  # Find scenario that causes X% loss


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    scenario_type: ScenarioType
    description: str

    # Market shocks (asset -> return shock)
    equity_shock: float = 0.0  # e.g., -0.20 for 20% drop
    bond_shock: float = 0.0
    commodity_shock: float = 0.0
    fx_shock: float = 0.0
    volatility_shock: float = 0.0  # VIX multiplier

    # Correlation shock (how much correlations increase)
    correlation_shock: float = 0.0

    # Liquidity shock (spread multiplier)
    liquidity_shock: float = 1.0

    # Duration in days
    duration_days: int = 1

    # Specific asset shocks (symbol -> return)
    asset_shocks: Dict[str, float] = field(default_factory=dict)

    # Historical reference period (for historical scenarios)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class StressTestResult:
    """Results from a stress test."""

    scenario_name: str
    portfolio_loss: float
    portfolio_loss_pct: float
    max_drawdown: float
    var_breach: bool  # Did loss exceed VaR?
    positions_at_risk: List[str]
    recovery_days: int  # Estimated days to recover
    margin_call_risk: bool
    details: Dict[str, Any] = field(default_factory=dict)


class HistoricalScenarios:
    """Pre-defined historical stress scenarios."""

    # 2008 Financial Crisis (Sep-Nov 2008)
    GFC_2008 = StressScenario(
        name="2008_financial_crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="2008 Global Financial Crisis - Lehman collapse",
        equity_shock=-0.45,
        bond_shock=0.05,  # Flight to quality
        commodity_shock=-0.35,
        volatility_shock=3.5,  # VIX spiked to 80+
        correlation_shock=0.4,
        liquidity_shock=3.0,
        duration_days=60,
        start_date="2008-09-15",
        end_date="2008-11-20",
    )

    # COVID Crash (Feb-Mar 2020)
    COVID_2020 = StressScenario(
        name="covid_crash_2020",
        scenario_type=ScenarioType.HISTORICAL,
        description="COVID-19 market crash - fastest 30% decline",
        equity_shock=-0.34,
        bond_shock=0.02,
        commodity_shock=-0.50,  # Oil went negative
        volatility_shock=4.0,  # VIX hit 82
        correlation_shock=0.5,
        liquidity_shock=2.5,
        duration_days=23,
        start_date="2020-02-19",
        end_date="2020-03-23",
    )

    # Flash Crash (May 2010)
    FLASH_CRASH_2010 = StressScenario(
        name="flash_crash_2010",
        scenario_type=ScenarioType.HISTORICAL,
        description="2010 Flash Crash - intraday 9% drop",
        equity_shock=-0.09,
        volatility_shock=2.0,
        correlation_shock=0.3,
        liquidity_shock=10.0,  # Extreme liquidity drought
        duration_days=1,
        start_date="2010-05-06",
        end_date="2010-05-06",
    )

    # 2022 Rate Shock
    RATE_SHOCK_2022 = StressScenario(
        name="rate_shock_2022",
        scenario_type=ScenarioType.HISTORICAL,
        description="2022 Fed rate hiking cycle",
        equity_shock=-0.25,
        bond_shock=-0.15,  # Both stocks and bonds fell
        volatility_shock=1.5,
        correlation_shock=0.2,
        duration_days=280,
        start_date="2022-01-03",
        end_date="2022-10-12",
    )

    # Black Monday 1987
    BLACK_MONDAY_1987 = StressScenario(
        name="black_monday_1987",
        scenario_type=ScenarioType.HISTORICAL,
        description="Black Monday - 22% single day drop",
        equity_shock=-0.22,
        volatility_shock=5.0,
        correlation_shock=0.6,
        liquidity_shock=5.0,
        duration_days=1,
        start_date="1987-10-19",
        end_date="1987-10-19",
    )

    @classmethod
    def get_all(cls) -> List[StressScenario]:
        """Get all predefined historical scenarios."""
        return [
            cls.GFC_2008,
            cls.COVID_2020,
            cls.FLASH_CRASH_2010,
            cls.RATE_SHOCK_2022,
            cls.BLACK_MONDAY_1987,
        ]


class HypotheticalScenarios:
    """Hypothetical stress scenarios for forward-looking analysis."""

    # Severe recession
    SEVERE_RECESSION = StressScenario(
        name="severe_recession",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Severe economic recession scenario",
        equity_shock=-0.40,
        bond_shock=0.10,
        commodity_shock=-0.30,
        volatility_shock=3.0,
        correlation_shock=0.4,
        liquidity_shock=2.0,
        duration_days=180,
    )

    # Stagflation
    STAGFLATION = StressScenario(
        name="stagflation",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Stagflation - high inflation + recession",
        equity_shock=-0.30,
        bond_shock=-0.20,  # Inflation hurts bonds
        commodity_shock=0.25,  # Commodities rise
        volatility_shock=2.0,
        correlation_shock=0.3,
        duration_days=365,
    )

    # Liquidity crisis
    LIQUIDITY_CRISIS = StressScenario(
        name="liquidity_crisis",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Severe market liquidity crisis",
        equity_shock=-0.15,
        volatility_shock=2.5,
        correlation_shock=0.5,
        liquidity_shock=5.0,
        duration_days=30,
    )

    # Geopolitical shock
    GEOPOLITICAL_SHOCK = StressScenario(
        name="geopolitical_shock",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Major geopolitical event",
        equity_shock=-0.20,
        commodity_shock=0.30,  # Oil spikes
        fx_shock=0.10,
        volatility_shock=2.5,
        correlation_shock=0.3,
        duration_days=60,
    )

    # Tech sector collapse
    TECH_COLLAPSE = StressScenario(
        name="tech_collapse",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="Technology sector specific collapse",
        equity_shock=-0.15,
        volatility_shock=2.0,
        correlation_shock=0.2,
        duration_days=90,
        asset_shocks={
            "QQQ": -0.40,
            "AAPL": -0.35,
            "MSFT": -0.35,
            "GOOGL": -0.40,
            "AMZN": -0.45,
            "NVDA": -0.50,
            "META": -0.45,
        },
    )

    @classmethod
    def get_all(cls) -> List[StressScenario]:
        """Get all hypothetical scenarios."""
        return [
            cls.SEVERE_RECESSION,
            cls.STAGFLATION,
            cls.LIQUIDITY_CRISIS,
            cls.GEOPOLITICAL_SHOCK,
            cls.TECH_COLLAPSE,
        ]


class StressTester:
    """
    Main stress testing engine.
    Applies scenarios to portfolios and calculates impact.
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        monte_carlo_sims: int = 10000,
        seed: int = 42,
    ):
        self.confidence_level = confidence_level
        self.monte_carlo_sims = monte_carlo_sims
        self.rng = np.random.default_rng(seed)

        # Asset class mappings for sector shocks
        self.sector_mappings = {
            "technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "QQQ"],
            "financials": ["JPM", "BAC", "GS", "MS", "XLF"],
            "energy": ["XOM", "CVX", "XLE", "USO"],
            "healthcare": ["JNJ", "UNH", "PFE", "XLV"],
            "consumer": ["WMT", "PG", "KO", "XLP", "XLY"],
            "industrials": ["CAT", "BA", "GE", "XLI"],
            "utilities": ["NEE", "DUK", "XLU"],
            "materials": ["LIN", "APD", "XLB"],
            "real_estate": ["VNQ", "AMT", "PLD"],
            "bonds": ["TLT", "IEF", "LQD", "HYG", "AGG"],
            "commodities": ["GLD", "SLV", "USO", "DBA"],
        }

    def run_scenario(
        self,
        scenario: StressScenario,
        positions: Dict[str, float],  # symbol -> position value
        portfolio_value: float,
        historical_returns: Optional[pd.DataFrame] = None,
    ) -> StressTestResult:
        """
        Run a stress scenario against a portfolio.

        Args:
            scenario: Stress scenario to apply
            positions: Current positions (symbol -> value)
            portfolio_value: Total portfolio value
            historical_returns: Historical returns for correlation adjustment

        Returns:
            StressTestResult with impact analysis
        """
        logger.info(f"Running stress test: {scenario.name}")

        # Calculate position shocks
        position_losses = {}
        total_loss = 0.0

        for symbol, value in positions.items():
            shock = self._get_asset_shock(symbol, scenario)

            # Apply correlation shock (losses are more correlated in stress)
            if historical_returns is not None and scenario.correlation_shock > 0:
                shock = self._apply_correlation_shock(shock, scenario.correlation_shock)

            # Apply liquidity shock (wider spreads = worse execution)
            if scenario.liquidity_shock > 1.0:
                shock *= 1 + (scenario.liquidity_shock - 1) * 0.1

            loss = value * shock
            position_losses[symbol] = loss
            total_loss += loss

        # Calculate metrics
        portfolio_loss_pct = total_loss / portfolio_value if portfolio_value > 0 else 0

        # Identify positions at highest risk
        sorted_losses = sorted(position_losses.items(), key=lambda x: x[1])
        positions_at_risk = [s for s, l in sorted_losses[:5] if l < 0]

        # Estimate recovery days (rough heuristic)
        avg_daily_return = 0.0003  # ~7% annual
        recovery_days = (
            int(abs(portfolio_loss_pct) / avg_daily_return)
            if avg_daily_return > 0
            else 999
        )

        # Check VaR breach (simplified)
        var_99 = portfolio_value * 0.05  # Assume 5% VaR
        var_breach = abs(total_loss) > var_99

        # Margin call risk (if loss > 25% with leverage)
        margin_call_risk = portfolio_loss_pct < -0.25

        # Calculate max drawdown during scenario
        max_drawdown = self._estimate_max_drawdown(scenario, portfolio_loss_pct)

        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=total_loss,
            portfolio_loss_pct=portfolio_loss_pct,
            max_drawdown=max_drawdown,
            var_breach=var_breach,
            positions_at_risk=positions_at_risk,
            recovery_days=recovery_days,
            margin_call_risk=margin_call_risk,
            details={
                "position_losses": position_losses,
                "scenario_duration": scenario.duration_days,
                "liquidity_impact": scenario.liquidity_shock,
                "correlation_impact": scenario.correlation_shock,
            },
        )

    def _get_asset_shock(self, symbol: str, scenario: StressScenario) -> float:
        """Get the shock for a specific asset."""
        # Check for specific asset shock first
        if symbol in scenario.asset_shocks:
            return scenario.asset_shocks[symbol]

        # Determine asset class and apply appropriate shock
        symbol_upper = symbol.upper()

        # Check sector mappings
        for sector, symbols in self.sector_mappings.items():
            if symbol_upper in symbols:
                if sector == "bonds":
                    return scenario.bond_shock
                elif sector == "commodities":
                    return scenario.commodity_shock
                elif sector in [
                    "technology",
                    "financials",
                    "healthcare",
                    "consumer",
                    "industrials",
                    "utilities",
                    "materials",
                    "real_estate",
                ]:
                    return scenario.equity_shock

        # Default to equity shock for unknown symbols
        return scenario.equity_shock

    def _apply_correlation_shock(
        self, base_shock: float, correlation_increase: float
    ) -> float:
        """Apply correlation shock - losses become more correlated."""
        # In stress, correlations go to 1, amplifying losses
        if base_shock < 0:
            return base_shock * (1 + correlation_increase * 0.5)
        return base_shock

    def _estimate_max_drawdown(
        self, scenario: StressScenario, total_loss_pct: float
    ) -> float:
        """Estimate max drawdown during scenario."""
        # Max DD is typically worse than end-to-end loss
        # due to volatility clustering
        vol_multiplier = 1 + (scenario.volatility_shock - 1) * 0.2
        return min(total_loss_pct * vol_multiplier, -0.99)

    def run_monte_carlo_stress(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        historical_returns: pd.DataFrame,
        tail_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo stress simulation focusing on tail scenarios.

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            historical_returns: Historical returns DataFrame
            tail_threshold: Percentile for tail focus (0.05 = 5% worst)

        Returns:
            Dict with Monte Carlo stress results
        """
        logger.info(
            f"Running Monte Carlo stress test ({self.monte_carlo_sims} simulations)"
        )

        symbols = [s for s in positions.keys() if s in historical_returns.columns]
        if not symbols:
            logger.warning("No matching symbols for Monte Carlo stress test")
            return {}

        returns = historical_returns[symbols].dropna()
        if len(returns) < 30:
            logger.warning("Insufficient data for Monte Carlo stress test")
            return {}

        # Calculate return statistics
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        # Add stress to covariance (increase volatility and correlation)
        stress_multiplier = 2.0  # Double volatility in stress
        stressed_cov = cov_matrix * stress_multiplier

        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(stressed_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        stressed_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Generate stressed scenarios
        try:
            scenarios = self.rng.multivariate_normal(
                mean_returns * 0.5,  # Reduce expected returns in stress
                stressed_cov,
                size=self.monte_carlo_sims,
            )
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix issue, using diagonal")
            scenarios = self.rng.normal(
                mean_returns * 0.5,
                np.sqrt(np.diag(stressed_cov)) * stress_multiplier,
                size=(self.monte_carlo_sims, len(symbols)),
            )

        # Calculate portfolio returns for each scenario
        weights = np.array([positions.get(s, 0) for s in symbols])
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        portfolio_returns = scenarios @ weights

        # Focus on tail scenarios
        tail_cutoff = np.percentile(portfolio_returns, tail_threshold * 100)
        tail_scenarios = portfolio_returns[portfolio_returns <= tail_cutoff]

        # Calculate statistics
        results = {
            "mean_tail_loss": float(np.mean(tail_scenarios)),
            "worst_case_loss": float(np.min(portfolio_returns)),
            "var_99": float(np.percentile(portfolio_returns, 1)),
            "var_95": float(np.percentile(portfolio_returns, 5)),
            "cvar_99": float(
                np.mean(
                    portfolio_returns[
                        portfolio_returns <= np.percentile(portfolio_returns, 1)
                    ]
                )
            ),
            "cvar_95": float(
                np.mean(
                    portfolio_returns[
                        portfolio_returns <= np.percentile(portfolio_returns, 5)
                    ]
                )
            ),
            "prob_loss_10pct": float(np.mean(portfolio_returns < -0.10)),
            "prob_loss_20pct": float(np.mean(portfolio_returns < -0.20)),
            "prob_loss_30pct": float(np.mean(portfolio_returns < -0.30)),
            "tail_scenarios_count": len(tail_scenarios),
            "total_simulations": self.monte_carlo_sims,
        }

        # Convert to dollar amounts
        results["dollar_var_99"] = results["var_99"] * portfolio_value
        results["dollar_cvar_99"] = results["cvar_99"] * portfolio_value
        results["dollar_worst_case"] = results["worst_case_loss"] * portfolio_value

        return results

    def run_reverse_stress_test(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        target_loss_pct: float = -0.25,
        historical_returns: Optional[pd.DataFrame] = None,
    ) -> List[StressScenario]:
        """
        Reverse stress test: find scenarios that cause target loss.

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            target_loss_pct: Target loss percentage (negative)
            historical_returns: Historical returns for calibration

        Returns:
            List of scenarios that could cause target loss
        """
        logger.info(f"Running reverse stress test for {target_loss_pct:.1%} loss")

        scenarios_causing_loss = []

        # Test combinations of shocks
        equity_shocks = np.arange(-0.50, 0.0, 0.05)
        vol_shocks = [1.0, 1.5, 2.0, 2.5, 3.0]
        correlation_shocks = [0.0, 0.2, 0.4]

        for eq_shock in equity_shocks:
            for vol_shock in vol_shocks:
                for corr_shock in correlation_shocks:
                    scenario = StressScenario(
                        name=f"reverse_eq{eq_shock:.0%}_vol{vol_shock:.1f}_corr{corr_shock:.1f}",
                        scenario_type=ScenarioType.REVERSE,
                        description=f"Reverse stress test scenario",
                        equity_shock=eq_shock,
                        volatility_shock=vol_shock,
                        correlation_shock=corr_shock,
                        duration_days=1,
                    )

                    result = self.run_scenario(scenario, positions, portfolio_value)

                    # Check if this scenario causes target loss
                    if result.portfolio_loss_pct <= target_loss_pct:
                        scenarios_causing_loss.append(scenario)

        # Sort by severity (least severe first - minimum shock needed)
        scenarios_causing_loss.sort(
            key=lambda s: abs(s.equity_shock) + s.volatility_shock
        )

        logger.info(
            f"Found {len(scenarios_causing_loss)} scenarios causing {target_loss_pct:.1%} loss"
        )

        return scenarios_causing_loss[:10]  # Return top 10

    def run_all_historical(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[StressTestResult]:
        """Run all predefined historical scenarios."""
        results = []
        for scenario in HistoricalScenarios.get_all():
            result = self.run_scenario(scenario, positions, portfolio_value)
            results.append(result)
        return results

    def run_all_hypothetical(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> List[StressTestResult]:
        """Run all predefined hypothetical scenarios."""
        results = []
        for scenario in HypotheticalScenarios.get_all():
            result = self.run_scenario(scenario, positions, portfolio_value)
            results.append(result)
        return results

    def generate_stress_report(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        historical_returns: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive stress test report.

        Returns:
            Dict with complete stress analysis
        """
        logger.info("Generating comprehensive stress test report")

        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "position_count": len(positions),
            "historical_scenarios": [],
            "hypothetical_scenarios": [],
            "monte_carlo": {},
            "reverse_stress": [],
            "summary": {},
        }

        # Run historical scenarios
        historical_results = self.run_all_historical(positions, portfolio_value)
        report["historical_scenarios"] = [
            {
                "name": r.scenario_name,
                "loss": r.portfolio_loss,
                "loss_pct": r.portfolio_loss_pct,
                "max_drawdown": r.max_drawdown,
                "var_breach": r.var_breach,
                "margin_call_risk": r.margin_call_risk,
                "positions_at_risk": r.positions_at_risk,
            }
            for r in historical_results
        ]

        # Run hypothetical scenarios
        hypothetical_results = self.run_all_hypothetical(positions, portfolio_value)
        report["hypothetical_scenarios"] = [
            {
                "name": r.scenario_name,
                "loss": r.portfolio_loss,
                "loss_pct": r.portfolio_loss_pct,
                "max_drawdown": r.max_drawdown,
                "var_breach": r.var_breach,
                "margin_call_risk": r.margin_call_risk,
                "positions_at_risk": r.positions_at_risk,
            }
            for r in hypothetical_results
        ]

        # Monte Carlo stress (if historical returns available)
        if historical_returns is not None:
            report["monte_carlo"] = self.run_monte_carlo_stress(
                positions, portfolio_value, historical_returns
            )

        # Reverse stress test
        reverse_scenarios = self.run_reverse_stress_test(
            positions, portfolio_value, target_loss_pct=-0.25
        )
        report["reverse_stress"] = [
            {
                "equity_shock": s.equity_shock,
                "volatility_shock": s.volatility_shock,
                "correlation_shock": s.correlation_shock,
            }
            for s in reverse_scenarios[:5]
        ]

        # Summary statistics
        all_results = historical_results + hypothetical_results
        losses = [r.portfolio_loss_pct for r in all_results]

        report["summary"] = {
            "worst_case_scenario": min(
                all_results, key=lambda r: r.portfolio_loss_pct
            ).scenario_name,
            "worst_case_loss_pct": min(losses),
            "average_stress_loss_pct": np.mean(losses),
            "scenarios_with_var_breach": sum(1 for r in all_results if r.var_breach),
            "scenarios_with_margin_risk": sum(
                1 for r in all_results if r.margin_call_risk
            ),
            "total_scenarios_tested": len(all_results),
        }

        return report


# Convenience function for quick stress testing
def quick_stress_test(
    positions: Dict[str, float],
    portfolio_value: float,
    scenarios: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Quick stress test with common scenarios.

    Args:
        positions: Symbol -> value mapping
        portfolio_value: Total portfolio value
        scenarios: List of scenario names (None for all)

    Returns:
        DataFrame with stress test results
    """
    tester = StressTester()

    all_scenarios = {
        **{s.name: s for s in HistoricalScenarios.get_all()},
        **{s.name: s for s in HypotheticalScenarios.get_all()},
    }

    if scenarios is None:
        scenarios = list(all_scenarios.keys())

    results = []
    for name in scenarios:
        if name in all_scenarios:
            result = tester.run_scenario(
                all_scenarios[name], positions, portfolio_value
            )
            results.append(
                {
                    "Scenario": result.scenario_name,
                    "Loss ($)": f"${result.portfolio_loss:,.0f}",
                    "Loss (%)": f"{result.portfolio_loss_pct:.1%}",
                    "Max DD": f"{result.max_drawdown:.1%}",
                    "VaR Breach": "Yes" if result.var_breach else "No",
                    "Margin Risk": "Yes" if result.margin_call_risk else "No",
                }
            )

    return pd.DataFrame(results)
