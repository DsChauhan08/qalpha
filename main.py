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
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.strategy.signals import MomentumStrategy, CompositeStrategy
from quantum_alpha.backtesting.engine import Backtester, OrderSide, OrderType
from quantum_alpha.backtesting.validation import MCPT, BootstrapAnalysis
from quantum_alpha.backtesting.performance_metrics import (
    compute_metrics,
    compute_metrics_from_returns,
)
from quantum_alpha.backtesting.performance_gate import evaluate_gate, aggregate_fundamentals
from quantum_alpha.risk.position_sizing import PositionSizer, VaRCalculator
from quantum_alpha.execution.paper_trader import PaperTrader
from quantum_alpha.config.validator import (
    validate_settings,
    validate_strategies,
    validate_risk_limits,
    validate_data_sources,
)
from quantum_alpha.monitoring.logging import configure_logging
from quantum_alpha.monitoring.alert_system import AlertManager, build_default_rules
from quantum_alpha.plugins import load_plugins


def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "quantum_alpha" / "config" / "settings.yaml"

    config_path = Path(config_path)
    if config_path.is_dir():
        config_path = config_path / "settings.yaml"

    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    issues = validate_settings(settings)
    config_dir = config_path.parent

    strategies_path = config_dir / "strategies.yaml"
    if strategies_path.exists():
        with open(strategies_path, "r") as f:
            strategies_cfg = yaml.safe_load(f)
        issues.extend(validate_strategies(strategies_cfg))

    risk_limits_path = config_dir / "risk_limits.yaml"
    if risk_limits_path.exists():
        with open(str(risk_limits_path), "r") as f:
            risk_cfg = yaml.safe_load(f)
        issues.extend(validate_risk_limits(risk_cfg))

    data_sources_path = config_dir / "data_sources.yaml"
    if data_sources_path.exists():
        with open(str(data_sources_path), "r") as f:
            data_cfg = yaml.safe_load(f)
        issues.extend(validate_data_sources(data_cfg))

    if issues:
        raise ValueError(f"Config validation failed: {', '.join(issues)}")

    return settings


def _resolve_config_dir(config_path: Optional[str]) -> Path:
    if config_path is None:
        return PROJECT_ROOT / "quantum_alpha" / "config"

    config_path = Path(config_path)
    return config_path if config_path.is_dir() else config_path.parent


def _load_optional_yaml(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_backtest(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "momentum",
    validate: bool = False,
    verbose: bool = True,
    config_path: Optional[str] = None,
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
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

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
            df = cleaner.clean(df)
            df = imputer.impute(df)
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

    # Extended metrics and gating
    settings = load_config(config_path)
    bench_cfg = settings.get("benchmarks", {})
    market_benchmark = bench_cfg.get("market", "SPY")
    quant_benchmark = bench_cfg.get("quant_composite", ["QQQ", "IWM"])

    def _returns(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change().dropna()

    market_returns = None
    quant_returns = None

    try:
        market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        market_returns = _returns(market_df)
    except Exception:
        market_returns = None

    if isinstance(quant_benchmark, list) and quant_benchmark:
        returns_list = []
        for sym in quant_benchmark:
            try:
                qdf = collector.fetch_ohlcv(sym, start_date, end_date)
                returns_list.append(_returns(qdf))
            except Exception:
                continue
        if returns_list:
            quant_returns = pd.concat(returns_list, axis=1).mean(axis=1).dropna()

    extended_metrics = compute_metrics(
        backtester.equity_curve,
        trades=backtester.trades,
        benchmark_returns=market_returns,
    )
    metrics.update(extended_metrics)

    fundamentals = []
    for symbol in symbols:
        try:
            fundamentals.append(collector.fetch_fundamentals(symbol))
        except Exception:
            continue
    metrics.update(aggregate_fundamentals(fundamentals))

    market_metrics = (
        compute_metrics_from_returns(market_returns, benchmark_returns=market_returns)
        if market_returns is not None
        else {}
    )
    quant_metrics = (
        compute_metrics_from_returns(quant_returns, benchmark_returns=market_returns)
        if quant_returns is not None
        else {}
    )

    try:
        market_fund = collector.fetch_fundamentals(market_benchmark)
        market_metrics.update(aggregate_fundamentals([market_fund]))
    except Exception:
        pass

    if isinstance(quant_benchmark, list) and quant_benchmark:
        quant_funds = []
        for sym in quant_benchmark:
            try:
                quant_funds.append(collector.fetch_fundamentals(sym))
            except Exception:
                continue
        if quant_funds:
            quant_metrics.update(aggregate_fundamentals(quant_funds))

    gate_details = None
    if market_returns is not None and quant_returns is not None:
        gate = evaluate_gate(metrics, market_metrics, quant_metrics)
        metrics["gate_passed"] = gate.passed
        metrics["gate_ratio_good"] = gate.ratio_good
        metrics["gate_coverage"] = gate.coverage
        metrics["gate_good_count"] = gate.good_count
        gate_details = gate.details

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
        if "gate_passed" in metrics:
            print(f"Gate Passed:     {str(metrics['gate_passed']).upper():>10}")
            print(f"Gate Coverage:   {metrics['gate_coverage']:>10d}")
            print(f"Gate Good:       {metrics['gate_good_count']:>10d}")
            print(f"Gate Ratio:      {metrics['gate_ratio_good'] * 100:>9.2f}%")
        print(f"{'=' * 60}\n")

    results = {
        "metrics": metrics,
        "gate_details": gate_details,
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


def run_paper(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "momentum",
    paper_bars: int = 30,
    verbose: bool = True,
) -> Dict:
    """
    Run a paper trading simulation on the most recent bars.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("QUANTUM ALPHA V1 - PAPER TRADING")
        print(f"{'=' * 60}")
        print(f"Symbols: {symbols}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Capital: ${initial_capital:,.0f}")
        print(f"Strategy: {strategy_type}")
        print(f"Paper Bars: {paper_bars}")
        print(f"{'=' * 60}\n")

    collector = DataCollector()
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    if strategy_type == "momentum":
        strategy = MomentumStrategy()
    elif strategy_type == "composite":
        strategy = CompositeStrategy()
    else:
        strategy = MomentumStrategy()

    position_sizer = PositionSizer()
    paper_trader = PaperTrader(
        initial_capital=initial_capital, paper_bars=paper_bars
    )

    if verbose:
        print("Collecting price data...")

    data = {}
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = cleaner.clean(df)
            df = imputer.impute(df)
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
        print(f"\nRunning paper trading simulation...")

    state = {
        "positions": {},
        "trade_history": [],
        "current_drawdown": 0,
        "peak_equity": initial_capital,
    }

    def trading_strategy(timestamp, bars, bt):
        for symbol, bar in bars.items():
            if symbol not in data:
                continue

            df = data[symbol]
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            signal = row.get("signal", 0)
            confidence = row.get("signal_confidence", 0.5)

            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0

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

        equity = bt._total_equity()
        state["peak_equity"] = max(state["peak_equity"], equity)
        state["current_drawdown"] = (equity - state["peak_equity"]) / state[
            "peak_equity"
        ]

    metrics, paper_start = paper_trader.run(data, trading_strategy)

    if verbose and "error" not in metrics:
        print(f"\n{'=' * 60}")
        print("PAPER RESULTS")
        print(f"{'=' * 60}")
        print(f"Paper Start:    {paper_start.date()}")
        print(f"Total Return:   {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annual Return:  {metrics['annual_return'] * 100:>10.2f}%")
        print(f"Volatility:     {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:  {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:   {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Calmar Ratio:   {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:       {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Profit Factor:  {metrics['profit_factor']:>10.2f}")
        print(f"Total Trades:   {metrics['n_trades']:>10d}")
        print(f"Final Equity:   ${metrics['final_equity']:>10,.2f}")
        print(f"{'=' * 60}\n")

    return {
        "metrics": metrics,
        "paper_start": paper_start,
        "equity_curve": paper_trader.backtester.equity_curve,
        "trades": paper_trader.backtester.trades,
        "fills": paper_trader.backtester.fills,
    }


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
    parser.add_argument(
        "--firm-mode",
        action="store_true",
        help="Enable firm-grade execution safeguards",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch local dashboard (if available)",
    )
    parser.add_argument(
        "--paper-bars",
        type=int,
        default=30,
        help="Number of most recent bars to simulate in paper mode",
    )
    parser.add_argument("--validate", action="store_true", help="Run MCPT validation")
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    try:
        settings = load_config(args.config)
    except Exception as e:
        print(f"Config error: {e}")
        return None

    load_plugins()

    config_dir = _resolve_config_dir(args.config)
    log_cfg = settings.get("logging", {}) if settings else {}
    configure_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "quantum_alpha.log"),
    )
    thresholds = {}
    risk_cfg = _load_optional_yaml(config_dir / "risk_limits.yaml")
    if risk_cfg and "limits" in risk_cfg:
        limits = risk_cfg["limits"]
        if "max_drawdown" in limits:
            thresholds["max_drawdown"] = limits["max_drawdown"]
        if "min_sharpe" in limits:
            thresholds["min_sharpe"] = limits["min_sharpe"]
        if "min_win_rate" in limits:
            thresholds["min_win_rate"] = limits["min_win_rate"]

    alert_manager = AlertManager()
    for rule in build_default_rules(thresholds):
        alert_manager.add_rule(rule)

    if args.firm_mode and args.mode != "live":
        print("Firm mode requested, but live execution is not enabled. Firm mode disabled.")
        args.firm_mode = False

    if args.firm_mode and args.mode == "live":
        print("Firm mode requested, but live execution is not implemented in this phase.")
        return None

    if args.dashboard:
        print("Dashboard flag enabled. Local dashboard is not implemented in V1.")

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
            config_path=args.config,
        )
        if isinstance(results, dict) and "metrics" in results:
            alert_manager.evaluate(results["metrics"])
        return results

    elif args.mode == "paper":
        results = run_paper(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            strategy_type=args.strategy,
            paper_bars=args.paper_bars,
            verbose=True,
        )
        if isinstance(results, dict) and "metrics" in results:
            alert_manager.evaluate(results["metrics"])
        return results

    elif args.mode == "live":
        print("Live trading requires additional safety checks")
        print("Coming in Phase 2...")
        return None


if __name__ == "__main__":
    main()
