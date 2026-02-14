#!/usr/bin/env python3
"""
Backtest Meta-Ensemble Strategy
=================================
CLI wrapper around run_backtest() for meta-ensemble strategy.

Usage:
  python backtest_meta_ensemble.py                          # SPY, 5 years
  python backtest_meta_ensemble.py --symbols SPY AAPL MSFT  # Multi-symbol
  python backtest_meta_ensemble.py --years 10               # 10 years
  python backtest_meta_ensemble.py --validate               # With MCPT validation
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/regulus/Trade")

from quantum_alpha.main import run_backtest


def main():
    parser = argparse.ArgumentParser(description="Backtest Meta-Ensemble Strategy")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols to trade (default: SPY)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=5,
        help="Number of years to backtest (default: 5)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD), overrides --years",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run MCPT validation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config directory or settings.yaml",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.10,
        help="Minimum confidence to trade (default: 0.10, range: 0-1)",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.52,
        help="Minimum probability for buy/sell (default: 0.52)",
    )
    parser.add_argument(
        "--proportional-sizing",
        action="store_true",
        default=True,
        help="Scale position size by confidence (default: True)",
    )
    parser.add_argument(
        "--binary-sizing",
        action="store_true",
        help="Use binary (0/1) position sizing instead of proportional",
    )

    args = parser.parse_args()

    # Date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=int(args.years * 365.25))

    print(f"\n{'=' * 60}")
    print("META-ENSEMBLE STRATEGY BACKTEST")
    print(f"{'=' * 60}")
    print(f"Symbols:    {', '.join(args.symbols)}")
    print(f"Period:     {start_date.date()} to {end_date.date()}")
    print(f"Capital:    ${args.capital:,.0f}")
    print(f"Confidence: >= {args.confidence_threshold:.0%}")
    print(f"Signal thr: >= {args.signal_threshold:.2f}")
    proportional = not args.binary_sizing
    print(f"Sizing:     {'proportional' if proportional else 'binary'}")
    print(f"Validate:   {args.validate}")
    print(f"{'=' * 60}\n")

    # Strategy kwargs
    strategy_kwargs = {
        "confidence_threshold": args.confidence_threshold,
        "signal_threshold": args.signal_threshold,
        "proportional_sizing": proportional,
    }

    config_path = args.config
    if config_path is None:
        # Use meta-ensemble-specific config (daily rebalance, no momentum filter)
        config_path = str(
            Path(__file__).parent / "config" / "meta_ensemble_settings.yaml"
        )
        print(f"Using meta-ensemble config (daily rebalance, no momentum filter)")

    # Run backtest
    results = run_backtest(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        strategy_type="meta_ensemble",
        validate=args.validate,
        verbose=True,
        config_path=config_path,
        strategy_kwargs=strategy_kwargs,
    )

    if not results:
        print("\nBacktest returned no results.")
        return

    # Print summary
    metrics = results.get("metrics", {})
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    summary_keys = [
        ("total_return", "Total Return", ".2%"),
        ("annual_return", "Annual Return", ".2%"),
        ("volatility", "Volatility", ".2%"),
        ("sharpe_ratio", "Sharpe Ratio", ".3f"),
        ("sortino_ratio", "Sortino Ratio", ".3f"),
        ("max_drawdown", "Max Drawdown", ".2%"),
        ("calmar_ratio", "Calmar Ratio", ".3f"),
        ("win_rate", "Win Rate", ".2%"),
        ("profit_factor", "Profit Factor", ".3f"),
        ("total_trades", "Total Trades", "d"),
        ("final_equity", "Final Equity", ",.0f"),
    ]

    for key, label, fmt in summary_keys:
        val = metrics.get(key)
        if val is not None:
            if fmt == "d":
                print(f"  {label:.<30} {int(val):{fmt}}")
            elif fmt == ",.0f":
                print(f"  {label:.<30} ${val:{fmt}}")
            else:
                print(f"  {label:.<30} {val:{fmt}}")

    # Gate results
    gate = results.get("gate_results", {})
    if gate:
        print(f"\n{'=' * 60}")
        print("PERFORMANCE GATE")
        print(f"{'=' * 60}")
        passed = gate.get("passed", False)
        print(f"  Gate Passed: {'YES' if passed else 'NO'}")
        for check_name, check_val in gate.get("checks", {}).items():
            print(f"    {check_name}: {check_val}")

    # MCPT p-value
    mcpt = results.get("mcpt_pvalue")
    if mcpt is not None:
        print(
            f"\n  MCPT p-value: {mcpt:.4f} ({'SIGNIFICANT' if mcpt < 0.05 else 'NOT significant'})"
        )

    # Verdict
    total_ret = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = abs(metrics.get("max_drawdown", 1))

    print(f"\n{'=' * 60}")
    if total_ret > 0 and sharpe > 0.5 and max_dd < 0.30:
        print("VERDICT: PROMISING - Positive returns with acceptable risk")
    elif total_ret > 0:
        print("VERDICT: MARGINAL - Positive returns but risk metrics need improvement")
    else:
        print("VERDICT: UNPROFITABLE - Negative returns")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
