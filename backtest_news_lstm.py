#!/usr/bin/env python3
"""
News-Driven LSTM Backtest Harness.

Runs a complete backtest using the trained NewsDrivenLSTM model
through the standard Quantum Alpha backtesting engine.

Usage:
    python backtest_news_lstm.py
    python backtest_news_lstm.py --checkpoint news_lstm_20260212 --symbols SPY
    python backtest_news_lstm.py --symbols SPY QQQ AAPL --years 5 --validate
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_alpha.main import run_backtest as _run_backtest_original


def run_news_lstm_backtest(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    checkpoint_name: str = None,
    validate: bool = False,
    verbose: bool = True,
    config_path: str = None,
) -> dict:
    """
    Run a backtest using the news_lstm strategy type.

    This calls the standard run_backtest() from main.py with
    strategy_type="news_lstm".
    """
    return _run_backtest_original(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        strategy_type="news_lstm",
        validate=validate,
        verbose=verbose,
        config_path=config_path,
        checkpoint_name=checkpoint_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the News-Driven LSTM strategy"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols to trade (default: SPY)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of backtest history (default: 5)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --years.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name (default: auto-detect latest)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run MCPT validation (p < 0.05 required)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config directory or settings.yaml",
    )

    args = parser.parse_args()

    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.years * 365)

    print(f"\n{'#' * 60}")
    print("# NEWS-DRIVEN LSTM BACKTEST")
    print(f"{'#' * 60}")
    print(f"Symbols:    {args.symbols}")
    print(f"Period:     {start_date.date()} to {end_date.date()}")
    print(f"Capital:    ${args.capital:,.0f}")
    print(f"Checkpoint: {args.checkpoint or 'auto-detect'}")
    print(f"Validate:   {args.validate}")

    # Run backtest
    results = run_news_lstm_backtest(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        checkpoint_name=args.checkpoint,
        validate=args.validate,
        verbose=True,
        config_path=args.config,
    )

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    metrics = results
    key_metrics = [
        ("Total Return", "total_return", "%"),
        ("Annual Return", "annual_return", "%"),
        ("Volatility", "volatility", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Sortino Ratio", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar Ratio", "calmar_ratio", ""),
        ("Win Rate", "win_rate", "%"),
        ("Profit Factor", "profit_factor", ""),
        ("Total Trades", "n_trades", ""),
        ("Final Equity", "final_equity", "$"),
    ]

    for label, key, fmt in key_metrics:
        val = metrics.get(key)
        if val is None:
            continue
        if fmt == "%":
            print(f"  {label:<20} {val * 100:>10.2f}%")
        elif fmt == "$":
            print(f"  {label:<20} ${val:>10,.2f}")
        else:
            if isinstance(val, float):
                print(f"  {label:<20} {val:>10.2f}")
            else:
                print(f"  {label:<20} {val:>10}")

    # Gate result
    if "gate_passed" in metrics:
        passed = metrics["gate_passed"]
        print(f"\n  Performance Gate:  {'PASSED' if passed else 'FAILED'}")
        print(f"  Gate Coverage:     {metrics.get('gate_coverage', 'N/A')}")

    # MCPT result
    if "mcpt_p_value" in metrics:
        p = metrics["mcpt_p_value"]
        print(
            f"\n  MCPT p-value:      {p:.4f} ({'SIGNIFICANT' if p < 0.05 else 'NOT significant'})"
        )

    # Verdict
    total_return = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = abs(metrics.get("max_drawdown", 1))

    print(f"\n{'=' * 60}")
    if total_return > 0 and sharpe > 0.5 and max_dd < 0.30:
        print("VERDICT: PROMISING - Worth further investigation")
    elif total_return > 0:
        print("VERDICT: MARGINAL - Positive return but weak risk-adjusted metrics")
    else:
        print("VERDICT: UNPROFITABLE - Needs retraining or architecture changes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
