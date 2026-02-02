"""
Quantum Alpha V1 - Main Entry Point
Single command deployment for backtesting.
"""

import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.strategy.signals import MomentumStrategy, CompositeStrategy
from quantum_alpha.backtesting.engine import Backtester, OrderSide, OrderType
from quantum_alpha.backtesting.validation import MCPT, BootstrapAnalysis
from quantum_alpha.risk.position_sizing import PositionSizer, VaRCalculator


def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "quantum_alpha" / "config" / "settings.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_backtest(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "momentum",
    validate: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run a complete backtest.

    Args:
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        strategy_type: 'momentum', 'mean_reversion', 'composite'
        validate: Whether to run MCPT validation
        verbose: Print progress

    Returns:
        Dict with backtest results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("QUANTUM ALPHA V1 - BACKTEST")
        print(f"{'=' * 60}")
        print(f"Symbols: {symbols}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Capital: ${initial_capital:,.0f}")
        print(f"Strategy: {strategy_type}")
        print(f"{'=' * 60}\n")

    # Initialize components
    collector = DataCollector()
    feature_gen = TechnicalFeatureGenerator()

    if strategy_type == "momentum":
        strategy = MomentumStrategy()
    elif strategy_type == "composite":
        strategy = CompositeStrategy()
    else:
        strategy = MomentumStrategy()

    position_sizer = PositionSizer()
    backtester = Backtester(initial_capital=initial_capital)

    # Collect data
    if verbose:
        print("Collecting price data...")

    data = {}
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = feature_gen.generate(df)
            df = strategy.generate_signals(df)
            data[symbol] = df
            if verbose:
                print(f"  {symbol}: {len(df)} bars")
        except Exception as e:
            if verbose:
                print(f"  {symbol}: FAILED - {e}")

    if not data:
        return {"error": "No data collected"}

    if verbose:
        print(f"\nRunning backtest...")

    # Track state for strategy
    state = {
        "positions": {},
        "trade_history": [],
        "current_drawdown": 0,
        "peak_equity": initial_capital,
    }

    def trading_strategy(timestamp, bars, bt):
        """Strategy execution function."""
        for symbol, bar in bars.items():
            if symbol not in data:
                continue

            df = data[symbol]
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            signal = row.get("signal", 0)
            confidence = row.get("signal_confidence", 0.5)

            # Get current position
            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0

            # Calculate position size
            trade_history = (
                np.array([t["pnl"] for t in bt.trades]) if bt.trades else np.array([0])
            )
            volatility = row.get("atr_pct", 0.02) * np.sqrt(252)

            sizing = position_sizer.calculate(
                trade_history=trade_history,
                current_volatility=max(volatility, 0.01),
                current_drawdown=state["current_drawdown"],
                signal_strength=signal,
                signal_confidence=confidence,
            )

            if sizing["halt_trading"]:
                return

            target_position = sizing["position_size"]
            equity = bt._total_equity()
            target_value = equity * target_position
            target_qty = target_value / bar["close"] if bar["close"] > 0 else 0

            # Generate orders
            qty_diff = target_qty - current_qty

            if abs(qty_diff) > 0.01 * equity / bar["close"]:
                if qty_diff > 0:
                    bt.submit_order(
                        symbol, OrderSide.BUY, abs(qty_diff), OrderType.MARKET
                    )
                else:
                    bt.submit_order(
                        symbol, OrderSide.SELL, abs(qty_diff), OrderType.MARKET
                    )

        # Update drawdown tracking
        equity = bt._total_equity()
        state["peak_equity"] = max(state["peak_equity"], equity)
        state["current_drawdown"] = (equity - state["peak_equity"]) / state[
            "peak_equity"
        ]

    # Run backtest
    backtester.run(data, trading_strategy)

    # Get results
    metrics = backtester.get_metrics()

    if verbose:
        print(f"\n{'=' * 60}")
        print("BACKTEST RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Return:    {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annual Return:   {metrics['annual_return'] * 100:>10.2f}%")
        print(f"Volatility:      {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:   {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:    {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Calmar Ratio:    {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:        {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Profit Factor:   {metrics['profit_factor']:>10.2f}")
        print(f"Total Trades:    {metrics['n_trades']:>10d}")
        print(f"Final Equity:    ${metrics['final_equity']:>10,.2f}")
        print(f"{'=' * 60}\n")

    results = {
        "metrics": metrics,
        "equity_curve": backtester.equity_curve,
        "trades": backtester.trades,
        "fills": backtester.fills,
    }

    # Run validation if requested
    if validate and len(data) > 0:
        if verbose:
            print("Running MCPT validation...")

        # Use first symbol for MCPT
        symbol = list(data.keys())[0]
        df = data[symbol]

        def strategy_func(price_df):
            feat_df = feature_gen.generate(price_df)
            sig_df = strategy.generate_signals(feat_df)
            return sig_df["signal"].fillna(0).values

        mcpt = MCPT(n_permutations=500, test_statistic="sharpe")
        mcpt_results = mcpt.run(df, strategy_func)

        results["mcpt"] = mcpt_results

        if verbose:
            print(f"\nMCPT Results:")
            print(f"  P-Value: {mcpt_results['p_value']:.4f}")
            print(f"  Significant: {'YES' if mcpt_results['is_significant'] else 'NO'}")
            print(f"  Percentile: {mcpt_results['percentile']:.1f}%")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantum Alpha V1 - Algorithmic Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Operating mode",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["SPY"], help="Symbols to trade"
    )
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument(
        "--start", type=str, default=None, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean_reversion", "composite"],
        default="momentum",
        help="Strategy type",
    )
    parser.add_argument("--validate", action="store_true", help="Run MCPT validation")
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=365 * 2)  # 2 years default

    if args.mode == "backtest":
        results = run_backtest(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            strategy_type=args.strategy,
            validate=args.validate,
            verbose=True,
        )
        return results

    elif args.mode == "paper":
        print("Paper trading mode not yet implemented in V1")
        print("Coming in Phase 1...")
        return None

    elif args.mode == "live":
        print("Live trading requires additional safety checks")
        print("Coming in Phase 2...")
        return None


if __name__ == "__main__":
    main()
