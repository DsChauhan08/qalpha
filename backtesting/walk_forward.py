"""
Walk-Forward Optimization Framework.

Implements rolling window optimization with out-of-sample validation.

Key features:
- Rolling/expanding window analysis
- Parameter optimization in-sample
- Out-of-sample validation
- Aggregated performance metrics
- Prevents overfitting through proper train/test separation

Based on agent.md specification for robust backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    # Window settings
    in_sample_size: int = 252  # 1 year of trading days
    out_of_sample_size: int = 63  # 3 months
    step_size: int = 21  # 1 month step

    # Optimization settings
    n_trials: int = 50  # Number of parameter combinations to try
    metric: str = "sharpe"  # Optimization metric

    # Validation settings
    min_trades: int = 10  # Minimum trades for valid period

    # Window type
    expanding: bool = False  # True = expanding window, False = rolling


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward period."""

    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime

    # Optimal parameters found in-sample
    optimal_params: Dict

    # In-sample metrics (for reference)
    in_sample_metrics: Dict

    # Out-of-sample metrics (the real test)
    out_of_sample_metrics: Dict

    # Trades
    n_trades: int


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for strategy validation.

    Walk-forward analysis is the gold standard for strategy validation:
    1. Optimize parameters on in-sample data
    2. Test on out-of-sample data
    3. Roll window forward and repeat
    4. Aggregate OOS results for true performance estimate

    This prevents overfitting by ensuring all performance metrics
    come from out-of-sample periods.
    """

    def __init__(
        self,
        strategy_factory: Callable,
        data: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_factory: Function that creates strategy given parameters
            data: Full historical data (DataFrame with OHLCV + features)
            config: Walk-forward configuration
        """
        self.strategy_factory = strategy_factory
        self.data = data
        self.config = config or WalkForwardConfig()

        self.results: List[WalkForwardResult] = []
        self.aggregated_metrics: Dict = {}

    def generate_windows(self) -> List[Tuple[int, int, int, int]]:
        """
        Generate train/test window indices.

        Returns:
            List of (is_start, is_end, oos_start, oos_end) tuples
        """
        n_samples = len(self.data)
        windows = []

        is_size = self.config.in_sample_size
        oos_size = self.config.out_of_sample_size
        step = self.config.step_size

        if self.config.expanding:
            # Expanding window: IS grows, OOS stays same size
            is_start = 0
            current_end = is_size

            while current_end + oos_size <= n_samples:
                is_end = current_end
                oos_start = current_end
                oos_end = oos_start + oos_size

                windows.append((is_start, is_end, oos_start, oos_end))

                current_end += step
        else:
            # Rolling window: both IS and OOS slide forward
            current_start = 0

            while current_start + is_size + oos_size <= n_samples:
                is_start = current_start
                is_end = is_start + is_size
                oos_start = is_end
                oos_end = oos_start + oos_size

                windows.append((is_start, is_end, oos_start, oos_end))

                current_start += step

        return windows

    def optimize_parameters(
        self,
        train_data: pd.DataFrame,
        param_space: Dict[str, List],
        n_trials: Optional[int] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Optimize strategy parameters on training data.

        Args:
            train_data: In-sample data
            param_space: Dictionary of parameter name -> list of values
            n_trials: Max number of combinations to try

        Returns:
            (optimal_params, metrics)
        """
        n_trials = n_trials or self.config.n_trials

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_space, n_trials)

        best_params = None
        best_metric = float("-inf")
        best_metrics = {}

        for params in param_combinations:
            try:
                # Create strategy with these parameters
                strategy = self.strategy_factory(**params)

                # Run backtest
                metrics = self._run_backtest(strategy, train_data)

                # Check if valid
                if metrics.get("n_trades", 0) < self.config.min_trades:
                    continue

                # Check if best
                metric_value = metrics.get(self.config.metric, float("-inf"))

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_metrics = metrics

            except Exception as e:
                warnings.warn(f"Parameter combination failed: {params}, Error: {e}")
                continue

        if best_params is None:
            # Return default parameters if no valid combination found
            best_params = {k: v[0] for k, v in param_space.items()}
            best_metrics = {"error": "No valid parameter combination found"}

        return best_params, best_metrics

    def _generate_param_combinations(
        self, param_space: Dict[str, List], max_combinations: int
    ) -> List[Dict]:
        """Generate parameter combinations to try."""
        from itertools import product

        keys = list(param_space.keys())
        values = list(param_space.values())

        # Generate all combinations
        all_combinations = list(product(*values))

        # Sample if too many
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(
                len(all_combinations), size=max_combinations, replace=False
            )
            all_combinations = [all_combinations[i] for i in indices]

        # Convert to dicts
        return [dict(zip(keys, combo)) for combo in all_combinations]

    def _run_backtest(self, strategy: Any, data: pd.DataFrame) -> Dict:
        """
        Run backtest and return metrics.

        Args:
            strategy: Strategy instance with generate_signals method
            data: Data to backtest on

        Returns:
            Performance metrics
        """
        # Generate signals
        if hasattr(strategy, "generate_signals"):
            signals_df = strategy.generate_signals(data)
        else:
            # Assume strategy is callable
            signals_df = strategy(data)

        # Extract signals
        if "position_signal" in signals_df.columns:
            signals = signals_df["position_signal"].values
        elif "signal" in signals_df.columns:
            signals = signals_df["signal"].values
        else:
            signals = np.zeros(len(data))

        # Calculate returns
        if "returns" in data.columns:
            returns = data["returns"].values
        else:
            returns = data["close"].pct_change().fillna(0).values

        # Strategy returns (signal * next period return)
        strategy_returns = signals[:-1] * returns[1:]

        # Calculate metrics
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return {"sharpe": 0, "total_return": 0, "n_trades": 0}

        total_return = np.sum(strategy_returns)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)

        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Count trades (signal changes)
        n_trades = np.sum(np.diff(signals) != 0)

        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Win rate
        winning = strategy_returns > 0
        win_rate = np.mean(winning) if len(winning) > 0 else 0

        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "n_trades": int(n_trades),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
        }

    def run(self, param_space: Dict[str, List], verbose: int = 1) -> Dict:
        """
        Run full walk-forward optimization.

        Args:
            param_space: Parameter space to search
            verbose: Verbosity level

        Returns:
            Aggregated results
        """
        windows = self.generate_windows()

        if verbose:
            print(f"Walk-Forward Optimization")
            print(f"  Periods: {len(windows)}")
            print(f"  IS Size: {self.config.in_sample_size}")
            print(f"  OOS Size: {self.config.out_of_sample_size}")
            print()

        self.results = []

        for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            # Get data slices
            train_data = self.data.iloc[is_start:is_end].copy()
            test_data = self.data.iloc[oos_start:oos_end].copy()

            # Get timestamps
            is_start_dt = (
                train_data.index[0]
                if hasattr(train_data.index, "__getitem__")
                else datetime.now()
            )
            is_end_dt = (
                train_data.index[-1]
                if hasattr(train_data.index, "__getitem__")
                else datetime.now()
            )
            oos_start_dt = (
                test_data.index[0]
                if hasattr(test_data.index, "__getitem__")
                else datetime.now()
            )
            oos_end_dt = (
                test_data.index[-1]
                if hasattr(test_data.index, "__getitem__")
                else datetime.now()
            )

            if verbose:
                print(f"Period {i + 1}/{len(windows)}")

            # Optimize on in-sample
            optimal_params, is_metrics = self.optimize_parameters(
                train_data, param_space
            )

            if verbose >= 2:
                print(f"  Optimal params: {optimal_params}")
                print(f"  IS Sharpe: {is_metrics.get('sharpe', 0):.2f}")

            # Test on out-of-sample
            strategy = self.strategy_factory(**optimal_params)
            oos_metrics = self._run_backtest(strategy, test_data)

            if verbose:
                print(f"  OOS Return: {oos_metrics.get('total_return', 0):.2%}")
                print(f"  OOS Sharpe: {oos_metrics.get('sharpe', 0):.2f}")
                print()

            # Store result
            result = WalkForwardResult(
                period_id=i,
                in_sample_start=is_start_dt,
                in_sample_end=is_end_dt,
                out_of_sample_start=oos_start_dt,
                out_of_sample_end=oos_end_dt,
                optimal_params=optimal_params,
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                n_trades=oos_metrics.get("n_trades", 0),
            )

            self.results.append(result)

        # Aggregate results
        self.aggregated_metrics = self._aggregate_results()

        if verbose:
            print("=" * 50)
            print("Walk-Forward Results (Out-of-Sample)")
            print(f"  Total Return: {self.aggregated_metrics['total_return']:.2%}")
            print(f"  Avg Sharpe: {self.aggregated_metrics['avg_sharpe']:.2f}")
            print(f"  Win Rate: {self.aggregated_metrics['win_rate']:.2%}")
            print(
                f"  Periods Profitable: {self.aggregated_metrics['periods_profitable']}/{len(self.results)}"
            )

        return self.aggregated_metrics

    def _aggregate_results(self) -> Dict:
        """Aggregate results across all periods."""
        if not self.results:
            return {"error": "No results to aggregate"}

        # Extract OOS metrics
        oos_returns = [
            r.out_of_sample_metrics.get("total_return", 0) for r in self.results
        ]
        oos_sharpes = [r.out_of_sample_metrics.get("sharpe", 0) for r in self.results]
        oos_drawdowns = [
            r.out_of_sample_metrics.get("max_drawdown", 0) for r in self.results
        ]
        n_trades = [r.n_trades for r in self.results]

        # Aggregate
        total_return = np.sum(oos_returns)  # Cumulative
        avg_sharpe = np.mean(oos_sharpes)
        avg_drawdown = np.mean(oos_drawdowns)
        periods_profitable = sum(1 for r in oos_returns if r > 0)

        # Compute overall Sharpe from concatenated returns
        all_returns = []
        for r in self.results:
            period_return = r.out_of_sample_metrics.get("total_return", 0)
            period_std = r.out_of_sample_metrics.get("std_return", 0.01)
            n_days = self.config.out_of_sample_size

            # Approximate daily returns
            daily_return = period_return / n_days
            all_returns.extend([daily_return] * n_days)

        if all_returns and np.std(all_returns) > 0:
            overall_sharpe = (np.mean(all_returns) / np.std(all_returns)) * np.sqrt(252)
        else:
            overall_sharpe = 0

        # IS vs OOS comparison (efficiency ratio)
        is_sharpes = [r.in_sample_metrics.get("sharpe", 0) for r in self.results]
        if np.mean(is_sharpes) != 0:
            efficiency_ratio = avg_sharpe / np.mean(is_sharpes)
        else:
            efficiency_ratio = 0

        return {
            "total_return": float(total_return),
            "avg_sharpe": float(avg_sharpe),
            "overall_sharpe": float(overall_sharpe),
            "avg_drawdown": float(avg_drawdown),
            "max_drawdown": float(min(oos_drawdowns)),
            "win_rate": float(np.mean([1 if r > 0 else 0 for r in oos_returns])),
            "periods_profitable": periods_profitable,
            "total_periods": len(self.results),
            "total_trades": sum(n_trades),
            "efficiency_ratio": float(efficiency_ratio),
            "is_avg_sharpe": float(np.mean(is_sharpes)),
            "oos_avg_sharpe": float(avg_sharpe),
        }

    def get_parameter_stability(self) -> pd.DataFrame:
        """
        Analyze parameter stability across periods.

        Returns:
            DataFrame showing parameter values over time
        """
        if not self.results:
            return pd.DataFrame()

        param_history = []
        for r in self.results:
            row = {"period": r.period_id}
            row.update(r.optimal_params)
            row["oos_sharpe"] = r.out_of_sample_metrics.get("sharpe", 0)
            param_history.append(row)

        return pd.DataFrame(param_history)

    def get_equity_curve(self) -> pd.Series:
        """
        Get aggregated OOS equity curve.

        Returns:
            Series of cumulative returns
        """
        returns = []
        for r in self.results:
            ret = r.out_of_sample_metrics.get("total_return", 0)
            returns.append(ret)

        cumulative = np.cumsum(returns)

        return pd.Series(cumulative, name="equity_curve")
