"""
Monte Carlo Permutation Test (MCPT) - V1
Statistical validation of strategy performance.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from tqdm import tqdm


class MCPT:
    """
    Monte Carlo Permutation Test for strategy validation.

    Tests whether strategy performance is statistically significant
    or could arise from random chance.

    Based on: neurotrader888/mcpt
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        test_statistic: str = "sharpe",
        random_seed: int = 42,
    ):
        """
        Args:
            n_permutations: Number of random permutations
            test_statistic: 'sharpe', 'profit_factor', or 'return'
            random_seed: Random seed for reproducibility
        """
        self.n_perm = n_permutations
        self.statistic = test_statistic
        np.random.seed(random_seed)

    def _profit_factor(self, returns: np.ndarray) -> float:
        """Gross profit / gross loss."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0
        return gross_profit / gross_loss

    def _sharpe(self, returns: np.ndarray) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def _total_return(self, returns: np.ndarray) -> float:
        """Total cumulative return."""
        return returns.sum()

    def _calc_statistic(self, returns: np.ndarray) -> float:
        """Calculate chosen test statistic."""
        if self.statistic == "profit_factor":
            return self._profit_factor(returns)
        elif self.statistic == "sharpe":
            return self._sharpe(returns)
        elif self.statistic == "return":
            return self._total_return(returns)
        else:
            return self._sharpe(returns)

    def permute_bars(self, df: pd.DataFrame, start_idx: int = 0) -> pd.DataFrame:
        """
        Randomly permute price bars while preserving OHLC relationships.

        Args:
            df: DataFrame with OHLC columns
            start_idx: Index to start permutation from

        Returns:
            Permuted DataFrame
        """
        permuted = df.copy()
        n_bars = len(permuted) - start_idx

        perm_indices = np.random.permutation(n_bars) + start_idx
        permuted.iloc[start_idx:] = permuted.iloc[perm_indices].values

        # Recalculate returns
        permuted["returns"] = permuted["close"].pct_change()

        return permuted

    def run(
        self,
        price_data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], np.ndarray],
        show_progress: bool = True,
    ) -> Dict:
        """
        Run MCPT on a strategy.

        Args:
            price_data: DataFrame with OHLCV and returns
            strategy_func: Function that takes price_data and returns signals array
            show_progress: Whether to show progress bar

        Returns:
            Dict with test results
        """
        # Calculate actual strategy returns
        signals = strategy_func(price_data)
        actual_returns = price_data["returns"].values * signals
        actual_stat = self._calc_statistic(actual_returns)

        print(f"Actual {self.statistic}: {actual_stat:.4f}")

        # Run permutations
        perm_stats = []
        better_count = 1  # Start at 1 for conservative estimate

        iterator = range(self.n_perm)
        if show_progress:
            iterator = tqdm(iterator, desc="Running MCPT")

        for _ in iterator:
            # Permute price data
            perm_data = self.permute_bars(price_data)

            # Generate signals on permuted data
            perm_signals = strategy_func(perm_data)
            perm_returns = perm_data["returns"].values * perm_signals

            # Calculate statistic
            perm_stat = self._calc_statistic(perm_returns)
            perm_stats.append(perm_stat)

            if perm_stat >= actual_stat:
                better_count += 1

        # Calculate p-value
        p_value = better_count / (self.n_perm + 1)

        return {
            "actual_statistic": actual_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "permutation_stats": perm_stats,
            "percentile": 100 * (1 - p_value),
            "n_permutations": self.n_perm,
            "statistic_type": self.statistic,
        }

    def run_on_returns(
        self, returns: np.ndarray, show_progress: bool = True
    ) -> Dict:
        """
        Run MCPT directly on a vector of realized returns (e.g., equity
        curve pct changes). This avoids re-generating signals and uses
        the full backtester path.

        Args:
            returns: 1D array of strategy returns
            show_progress: Whether to display progress bar

        Returns:
            Dict with permutation p-value and statistics
        """
        clean_returns = np.nan_to_num(returns.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        actual_stat = self._calc_statistic(clean_returns)

        iterator = range(self.n_perm)
        if show_progress:
            iterator = tqdm(iterator, desc="Running MCPT")

        perm_stats = []
        better_count = 1  # Conservative: count actual path

        for _ in iterator:
            perm = np.random.permutation(clean_returns)
            perm_stat = self._calc_statistic(perm)
            perm_stats.append(perm_stat)
            if perm_stat >= actual_stat:
                better_count += 1

        p_value = better_count / (self.n_perm + 1)

        return {
            "actual_statistic": actual_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "permutation_stats": perm_stats,
            "percentile": 100 * (1 - p_value),
            "n_permutations": self.n_perm,
            "statistic_type": self.statistic,
        }


class BootstrapAnalysis:
    """
    Bootstrap resampling for confidence interval estimation.
    """

    def __init__(
        self, n_bootstrap: int = 10000, confidence: float = 0.95, random_seed: int = 42
    ):
        self.n_boot = n_bootstrap
        self.confidence = confidence
        np.random.seed(random_seed)

    def run(self, returns: np.ndarray) -> Dict:
        """
        Run bootstrap analysis.

        Args:
            returns: Array of returns

        Returns:
            Dict with statistics and confidence intervals
        """
        # Actual statistics
        actual_sharpe = self._sharpe(returns)
        actual_cagr = self._cagr(returns)
        actual_mdd = self._max_drawdown(returns)

        # Bootstrap
        sharpe_boot = []
        cagr_boot = []
        mdd_boot = []

        for _ in range(self.n_boot):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            sharpe_boot.append(self._sharpe(sample))
            cagr_boot.append(self._cagr(sample))
            mdd_boot.append(self._max_drawdown(sample))

        alpha = (1 - self.confidence) / 2

        return {
            "sharpe": {
                "actual": actual_sharpe,
                "mean": np.mean(sharpe_boot),
                "ci_lower": np.percentile(sharpe_boot, 100 * alpha),
                "ci_upper": np.percentile(sharpe_boot, 100 * (1 - alpha)),
            },
            "cagr": {
                "actual": actual_cagr,
                "mean": np.mean(cagr_boot),
                "ci_lower": np.percentile(cagr_boot, 100 * alpha),
                "ci_upper": np.percentile(cagr_boot, 100 * (1 - alpha)),
            },
            "max_drawdown": {
                "actual": actual_mdd,
                "mean": np.mean(mdd_boot),
                "ci_lower": np.percentile(mdd_boot, 100 * alpha),
                "ci_upper": np.percentile(mdd_boot, 100 * (1 - alpha)),
            },
        }

    def _sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def _cagr(self, returns: np.ndarray) -> float:
        total = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        return (1 + total) ** (1 / max(n_years, 0.01)) - 1

    def _max_drawdown(self, returns: np.ndarray) -> float:
        equity = (1 + returns).cumprod()
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return dd.min()


def walk_forward_mcpt(
    price_data: pd.DataFrame,
    strategy_func: Callable,
    train_window: int,
    test_window: int,
    n_permutations: int = 500,
) -> Dict:
    """
    Walk-forward MCPT for out-of-sample validation.

    This is the gold standard test - passing indicates genuine predictive power.

    Args:
        price_data: Full price history
        strategy_func: Strategy function(train_data, test_data) -> signals
        train_window: Training window size
        test_window: Test window size
        n_permutations: Number of permutations

    Returns:
        Dict with test results
    """
    np.random.seed(42)

    # Calculate walk-forward returns on actual data
    def get_wf_returns(data: pd.DataFrame) -> np.ndarray:
        all_returns = []

        for i in range(train_window, len(data) - test_window, test_window):
            train = data.iloc[i - train_window : i]
            test = data.iloc[i : i + test_window]

            signals = strategy_func(train, test)
            test_returns = test["returns"].values * signals
            all_returns.extend(test_returns)

        return np.array(all_returns)

    actual_returns = get_wf_returns(price_data)
    actual_sharpe = (
        (actual_returns.mean() / actual_returns.std()) * np.sqrt(252)
        if actual_returns.std() > 0
        else 0
    )

    print(f"Walk-forward Sharpe: {actual_sharpe:.4f}")

    # Run permutations
    perm_stats = []
    better_count = 1

    for _ in tqdm(range(n_permutations), desc="WF-MCPT"):
        # Permute from train_window onwards
        perm_data = price_data.copy()
        n_to_perm = len(perm_data) - train_window
        perm_idx = np.random.permutation(n_to_perm) + train_window
        perm_data.iloc[train_window:] = perm_data.iloc[perm_idx].values
        perm_data["returns"] = perm_data["close"].pct_change()

        # Calculate walk-forward returns
        perm_returns = get_wf_returns(perm_data)
        perm_sharpe = (
            (perm_returns.mean() / perm_returns.std()) * np.sqrt(252)
            if perm_returns.std() > 0
            else 0
        )

        perm_stats.append(perm_sharpe)
        if perm_sharpe >= actual_sharpe:
            better_count += 1

    p_value = better_count / (n_permutations + 1)

    return {
        "actual_sharpe": actual_sharpe,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "permutation_sharpes": perm_stats,
        "percentile": 100 * (1 - p_value),
    }
