"""
Performance gating based on fixed metric list and benchmark comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


METRIC_RULES = {
    "total_return": "higher",
    "cagr": "higher",
    "annual_volatility": "lower",
    "beta": "lower",
    "sharpe_ratio": "higher",
    "sortino_ratio": "higher",
    "treynor_ratio": "higher",
    "jensen_alpha": "higher",
    "information_ratio": "higher",
    "tracking_error": "lower",
    "r_squared": "higher",
    "max_drawdown": "lower",
    "calmar_ratio": "higher",
    "ulcer_index": "lower",
    "downside_deviation": "lower",
    "var": "lower",
    "cvar": "lower",
    "omega_ratio": "higher",
    "upside_capture": "higher",
    "downside_capture": "higher",
    "capture_ratio": "higher",
    "m2": "higher",
    "sterling_ratio": "higher",
    "recovery_time": "lower",
    "win_rate": "higher",
    "payoff_ratio": "higher",
    "profit_factor": "higher",
    "max_consecutive_losses": "lower",
    "max_consecutive_wins": "higher",
    "price_to_earnings": "lower",
    "price_to_book": "lower",
    "ev_to_ebitda": "lower",
    "ev_to_sales": "lower",
    "peg_ratio": "lower",
    "free_cashflow_yield": "higher",
    "price_to_cashflow": "lower",
    "dividend_yield": "higher",
    "dividend_payout_ratio": "lower",
    "dividend_growth": "higher",
    "eps_growth": "higher",
    "revenue_growth": "higher",
    "roe": "higher",
    "roa": "higher",
    "roic": "higher",
    "gross_margin": "higher",
    "operating_margin": "higher",
    "net_profit_margin": "higher",
    "ebitda_margin": "higher",
    "interest_coverage": "higher",
    "debt_to_equity": "lower",
    "net_debt_to_ebitda": "lower",
}


@dataclass
class GateResult:
    passed: bool
    coverage: int
    good_count: int
    ratio_good: float
    details: Dict[str, Dict[str, object]]


def _compare(metric: str, value: float, benchmark: float, rule: str) -> bool:
    if rule == "higher":
        return value >= benchmark
    if rule == "lower":
        return value <= benchmark
    return False


def evaluate_gate(
    metrics: Dict[str, float],
    market_metrics: Dict[str, float],
    quant_metrics: Dict[str, float],
    min_ratio: float = 0.9,
    min_metrics: int = 50,
) -> GateResult:
    details: Dict[str, Dict[str, object]] = {}
    good = 0
    coverage = 0

    for metric, rule in METRIC_RULES.items():
        value = metrics.get(metric)
        market_val = market_metrics.get(metric)
        quant_val = quant_metrics.get(metric)

        if value is None or market_val is None or quant_val is None:
            details[metric] = {
                "value": value,
                "market": market_val,
                "quant": quant_val,
                "rule": rule,
                "good": False,
                "reason": "missing",
            }
            continue

        coverage += 1
        beat_market = _compare(metric, value, market_val, rule)
        beat_quant = _compare(metric, value, quant_val, rule)
        is_good = beat_market and beat_quant

        details[metric] = {
            "value": value,
            "market": market_val,
            "quant": quant_val,
            "rule": rule,
            "good": is_good,
        }

        if is_good:
            good += 1

    ratio_good = good / coverage if coverage > 0 else 0.0
    passed = coverage >= min_metrics and ratio_good >= min_ratio

    return GateResult(
        passed=passed,
        coverage=coverage,
        good_count=good,
        ratio_good=ratio_good,
        details=details,
    )


def aggregate_fundamentals(fundamentals: List[Dict[str, object]]) -> Dict[str, float]:
    if not fundamentals:
        return {}

    def avg(key: str) -> Optional[float]:
        values = [f.get(key) for f in fundamentals]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return float(np.mean(values))

    results: Dict[str, Optional[float]] = {
        "price_to_earnings": avg("pe_ratio"),
        "price_to_book": avg("price_to_book"),
        "ev_to_ebitda": avg("enterprise_to_ebitda"),
        "ev_to_sales": avg("enterprise_to_revenue"),
        "peg_ratio": avg("peg_ratio"),
        "dividend_yield": avg("dividend_yield"),
        "dividend_payout_ratio": avg("payout_ratio"),
        "eps_growth": avg("earnings_growth"),
        "revenue_growth": avg("revenue_growth"),
        "roe": avg("return_on_equity"),
        "roa": avg("return_on_assets"),
        "roic": avg("return_on_investment"),
        "gross_margin": avg("gross_margins"),
        "operating_margin": avg("operating_margins"),
        "net_profit_margin": avg("profit_margins"),
        "ebitda_margin": avg("ebitda_margins"),
        "debt_to_equity": avg("debt_to_equity"),
    }

    # Derived metrics
    market_caps = [f.get("market_cap") for f in fundamentals if f.get("market_cap")]
    free_cashflows = [f.get("free_cashflow") for f in fundamentals if f.get("free_cashflow")]
    operating_cashflows = [f.get("operating_cashflow") for f in fundamentals if f.get("operating_cashflow")]
    total_debts = [f.get("total_debt") for f in fundamentals if f.get("total_debt")]
    total_cash = [f.get("total_cash") for f in fundamentals if f.get("total_cash")]
    ebitdas = [f.get("ebitda") for f in fundamentals if f.get("ebitda")]

    if market_caps and free_cashflows:
        results["free_cashflow_yield"] = float(np.mean(free_cashflows) / np.mean(market_caps))
    else:
        results["free_cashflow_yield"] = None

    if market_caps and operating_cashflows:
        results["price_to_cashflow"] = float(np.mean(market_caps) / np.mean(operating_cashflows))
    else:
        results["price_to_cashflow"] = None

    if total_debts and total_cash and ebitdas:
        net_debt = np.mean(total_debts) - np.mean(total_cash)
        results["net_debt_to_ebitda"] = float(net_debt / np.mean(ebitdas)) if np.mean(ebitdas) else None
    else:
        results["net_debt_to_ebitda"] = None

    results["interest_coverage"] = None
    results["dividend_growth"] = None

    return {k: v for k, v in results.items()}
