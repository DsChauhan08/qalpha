"""
Statistical Arbitrage Module

Implements pairs trading strategies using cointegration analysis:
- PairsTradingStrategy: Engle-Granger cointegration-based pairs trading
- KalmanPairsTrader: Adaptive hedge ratio via Kalman filter
- MultiPairsPortfolio: Diversified portfolio of cointegrated pairs

Key improvements over basic implementations:
- Uses statsmodels ADF test with numpy fallback for robust p-values
- Kalman filter handles both scalar and matrix residuals
- Half-life estimation for mean reversion speed
- Proper logging throughout (no print statements)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class PairsTrade:
    """Represents a pairs trade signal."""

    symbol_long: str
    symbol_short: str
    hedge_ratio: float
    z_score: float
    signal_strength: float
    entry_threshold: float
    exit_threshold: float


class PairsTradingStrategy:
    """
    Statistical arbitrage using cointegrated pairs.

    Strategy:
    1. Find cointegrated pairs via Engle-Granger test
    2. Calculate spread and z-score
    3. Enter when z-score exceeds threshold
    4. Exit when spread reverts to mean

    Uses statsmodels.tsa.stattools.adfuller when available for
    accurate ADF critical values; falls back to numpy-only
    approximation otherwise.
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        adf_threshold: float = 0.05,
    ):
        """
        Args:
            lookback: Lookback window for spread calculations
            entry_z: Z-score entry threshold (absolute value)
            exit_z: Z-score exit threshold for mean reversion
            adf_threshold: ADF test significance level
        """
        if lookback < 10:
            raise ValueError("lookback must be >= 10")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.adf_threshold = adf_threshold
        self.pairs: List[Dict] = []

    def find_cointegrated_pairs(
        self, prices: pd.DataFrame, max_pairs: int = 10
    ) -> pd.DataFrame:
        """
        Find cointegrated pairs from price data.

        Args:
            prices: DataFrame with price series as columns
            max_pairs: Maximum number of pairs to return

        Returns:
            DataFrame of cointegrated pairs sorted by composite score
        """
        symbols = prices.columns.tolist()
        n = len(symbols)

        if n < 2:
            logger.warning("Need at least 2 symbols to find pairs")
            return pd.DataFrame()

        results = []

        for i in range(n):
            for j in range(i + 1, n):
                sym1, sym2 = symbols[i], symbols[j]

                y = prices[sym1].dropna()
                x = prices[sym2].dropna()

                # Align on common index
                common_idx = y.index.intersection(x.index)
                if len(common_idx) < self.lookback:
                    continue

                y = y.loc[common_idx]
                x = x.loc[common_idx]

                # Test cointegration
                coint_result = self._engle_granger(y, x)

                if coint_result["cointegrated"]:
                    results.append(
                        {
                            "symbol_1": sym1,
                            "symbol_2": sym2,
                            "adf_pvalue": coint_result["adf_pvalue"],
                            "hedge_ratio": coint_result["hedge_ratio"],
                            "half_life": coint_result.get("half_life", np.nan),
                            "correlation": float(y.corr(x)),
                        }
                    )

        if not results:
            logger.info("No cointegrated pairs found")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Composite score: lower p-value, shorter half-life, higher correlation
        df["score"] = (
            (1 - df["adf_pvalue"]) * 0.4
            + (1 / (1 + df["half_life"].fillna(100))) * 0.4
            + df["correlation"].abs() * 0.2
        )

        result_df = df.nlargest(max_pairs, "score")
        logger.info(
            "Found %d cointegrated pairs from %d candidates",
            len(result_df),
            n * (n - 1) // 2,
        )
        return result_df

    def _engle_granger(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Engle-Granger cointegration test.

        Steps:
        1. OLS regression of y on x
        2. ADF test on residuals
        3. Half-life estimation of mean reversion

        Args:
            y: First (dependent) price series
            x: Second (independent) price series

        Returns:
            dict with keys: cointegrated, adf_statistic, adf_pvalue,
            hedge_ratio, intercept, half_life, residuals
        """
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Residuals (spread)
        residuals = y - (intercept + slope * x)

        # ADF test on residuals
        adf_result = self._adf_test(residuals.values)

        # Half-life of mean reversion
        half_life = self._estimate_half_life(residuals)

        return {
            "cointegrated": adf_result["p_value"] < self.adf_threshold,
            "adf_statistic": adf_result["statistic"],
            "adf_pvalue": adf_result["p_value"],
            "hedge_ratio": slope,
            "intercept": intercept,
            "half_life": half_life,
            "residuals": residuals,
        }

    def _adf_test(self, series: np.ndarray, maxlag: Optional[int] = None) -> Dict:
        """
        Augmented Dickey-Fuller test.

        Uses statsmodels.tsa.stattools.adfuller when available for
        proper MacKinnon critical values. Falls back to a numpy-only
        OLS regression with approximate p-value from normal distribution.

        Args:
            series: Time series to test for stationarity
            maxlag: Maximum lag (auto-selected if None)

        Returns:
            dict with keys: statistic, p_value
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            lag = maxlag if maxlag is not None else None
            result = adfuller(
                series, maxlag=lag, autolag="AIC" if lag is None else None
            )
            return {"statistic": float(result[0]), "p_value": float(result[1])}
        except ImportError:
            logger.debug("statsmodels not available, using numpy ADF fallback")

        # Numpy fallback
        diff = np.diff(series)
        lagged = series[:-1]

        X = np.column_stack([np.ones(len(lagged)), lagged])
        y = diff

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        mse = np.mean(residuals**2)
        try:
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta1 = np.sqrt(var_beta[1, 1])
        except np.linalg.LinAlgError:
            return {"statistic": 0.0, "p_value": 1.0}

        t_stat = beta[1] / se_beta1 if se_beta1 > 0 else 0.0

        # Approximate p-value (conservative for Dickey-Fuller distribution)
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(abs(t_stat)))

        return {"statistic": float(t_stat), "p_value": float(p_value)}

    def _estimate_half_life(self, residuals: pd.Series) -> float:
        """
        Estimate half-life of mean reversion from OU process.

        Uses regression: delta_spread = alpha + beta * lagged_spread
        Half-life = -ln(2) / beta

        Args:
            residuals: Spread residuals

        Returns:
            Half-life in periods, or nan if not mean-reverting
        """
        lagged = residuals.shift(1).dropna()
        delta = residuals.diff().dropna()

        common_idx = lagged.index.intersection(delta.index)
        lagged = lagged.loc[common_idx]
        delta = delta.loc[common_idx]

        if len(lagged) < 10:
            return np.nan

        beta = np.polyfit(lagged, delta, 1)[0]
        if beta < 0:
            return float(-np.log(2) / beta)
        return np.nan

    def generate_signals(self, pair: Dict, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for a cointegrated pair.

        Args:
            pair: Pair information dict with symbol_1, symbol_2, hedge_ratio
            prices: Price data DataFrame

        Returns:
            DataFrame with z_score, spread, long_signal, short_signal,
            exit_long, exit_short columns
        """
        sym1, sym2 = pair["symbol_1"], pair["symbol_2"]
        hedge_ratio = pair["hedge_ratio"]

        y = prices[sym1]
        x = prices[sym2]

        # Calculate spread
        spread = y - hedge_ratio * x

        # Rolling z-score
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / spread_std.clip(lower=1e-8)

        signals = pd.DataFrame(index=prices.index)
        signals["z_score"] = z_score
        signals["spread"] = spread

        # Long spread when z-score is very negative (buy sym1, sell sym2)
        signals["long_signal"] = z_score < -self.entry_z
        # Short spread when z-score is very positive (sell sym1, buy sym2)
        signals["short_signal"] = z_score > self.entry_z

        # Exit signals
        signals["exit_long"] = z_score > -self.exit_z
        signals["exit_short"] = z_score < self.exit_z

        return signals

    def backtest_pair(
        self,
        pair: Dict,
        prices: pd.DataFrame,
        initial_capital: float = 100000.0,
    ) -> Dict:
        """
        Backtest a pairs trading strategy on historical data.

        Args:
            pair: Pair information dict
            prices: Price data
            initial_capital: Starting capital

        Returns:
            dict with total_return, sharpe_ratio, max_drawdown,
            n_trades, trades, equity_curve
        """
        signals = self.generate_signals(pair, prices)

        position = 0  # 0=flat, 1=long spread, -1=short spread
        capital = initial_capital
        trades: List[Dict] = []
        equity = [initial_capital]
        entry_spread = 0.0

        for i in range(1, len(signals)):
            date = signals.index[i]
            z = signals["z_score"].iloc[i]
            spread_val = signals["spread"].iloc[i]

            if np.isnan(z) or np.isnan(spread_val):
                equity.append(capital)
                continue

            if position == 0:
                if signals["long_signal"].iloc[i]:
                    position = 1
                    entry_spread = spread_val
                    trades.append(
                        {"date": date, "action": "enter_long", "spread": entry_spread}
                    )
                elif signals["short_signal"].iloc[i]:
                    position = -1
                    entry_spread = spread_val
                    trades.append(
                        {"date": date, "action": "enter_short", "spread": entry_spread}
                    )

            elif position == 1:
                if signals["exit_long"].iloc[i]:
                    pnl = spread_val - entry_spread
                    capital += pnl
                    trades.append(
                        {
                            "date": date,
                            "action": "exit_long",
                            "spread": spread_val,
                            "pnl": pnl,
                        }
                    )
                    position = 0

            elif position == -1:
                if signals["exit_short"].iloc[i]:
                    pnl = entry_spread - spread_val
                    capital += pnl
                    trades.append(
                        {
                            "date": date,
                            "action": "exit_short",
                            "spread": spread_val,
                            "pnl": pnl,
                        }
                    )
                    position = 0

            equity.append(capital)

        # Metrics
        equity_series = pd.Series(equity, index=signals.index[: len(equity)])
        returns = equity_series.pct_change().dropna()

        total_return = (capital - initial_capital) / initial_capital

        if len(returns) > 0 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
        else:
            sharpe = 0.0

        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak.clip(lower=1e-8)
        max_dd = float(drawdown.min())

        completed_trades = [t for t in trades if "pnl" in t]
        logger.info(
            "Pair %s-%s: return=%.2f%%, sharpe=%.2f, maxDD=%.2f%%, trades=%d",
            pair["symbol_1"],
            pair["symbol_2"],
            total_return * 100,
            sharpe,
            max_dd * 100,
            len(completed_trades),
        )

        return {
            "pair": pair,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": len(completed_trades),
            "trades": trades,
            "equity_curve": equity_series,
        }


class KalmanPairsTrader:
    """
    Pairs trading with Kalman filter for adaptive hedge ratio.

    The Kalman filter continuously updates the hedge ratio,
    adapting to changing market relationships. This is superior
    to static OLS-based hedge ratios for non-stationary pairs.

    State vector: [hedge_ratio, intercept]
    Measurement: y = hedge_ratio * x + intercept + noise
    """

    def __init__(
        self,
        delta: float = 1e-4,
        R: float = 1e-3,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        """
        Args:
            delta: Transition covariance (process/system noise)
            R: Measurement noise variance
            entry_z: Entry z-score threshold
            exit_z: Exit z-score threshold
        """
        if delta <= 0 or R <= 0:
            raise ValueError("delta and R must be positive")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")

        self.delta = delta
        self.R = R
        self.entry_z = entry_z
        self.exit_z = exit_z

        # State variables (initialized in initialize_filter)
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.F: np.ndarray = np.eye(2)
        self.Q: Optional[np.ndarray] = None

    def initialize_filter(self, y: np.ndarray, x: np.ndarray) -> None:
        """
        Initialize Kalman filter with OLS estimate from initial data.

        Args:
            y: Target price series (dependent)
            x: Hedge price series (independent)
        """
        n = min(len(y), 100)
        X = np.column_stack([np.ones(n), x[:n]])

        beta = np.linalg.lstsq(X, y[:n], rcond=None)[0]

        # State: [intercept, hedge_ratio] -> reorder to [hedge_ratio, intercept]
        self.x = np.array([[beta[1]], [beta[0]]])  # [hedge_ratio, intercept]
        self.P = np.eye(2)
        self.Q = np.eye(2) * self.delta

        logger.debug(
            "Kalman initialized: hedge_ratio=%.4f, intercept=%.4f",
            self.x[0, 0],
            self.x[1, 0],
        )

    def update(self, y: float, x: float) -> Dict:
        """
        Update Kalman filter with new observation.

        Args:
            y: New target price
            x: New hedge price

        Returns:
            dict with hedge_ratio, intercept, spread, spread_std,
            z_score, residual
        """
        if self.x is None or self.P is None or self.Q is None:
            raise RuntimeError("Filter not initialized. Call initialize_filter first.")

        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Measurement matrix: y = [x, 1] @ [hedge_ratio, intercept]
        H = np.array([[x, 1.0]])
        y_pred = H @ self.x

        # Innovation (residual)
        raw_residual = y - y_pred
        # Handle both scalar and matrix residual
        if isinstance(raw_residual, np.ndarray):
            residual_scalar = float(raw_residual.flat[0])
        else:
            residual_scalar = float(raw_residual)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        S_scalar = float(S.flat[0]) if isinstance(S, np.ndarray) else float(S)

        # Kalman gain
        if S_scalar > 0:
            K = self.P @ H.T / S_scalar
        else:
            K = np.zeros((2, 1))

        # Update step
        self.x = self.x + K * residual_scalar
        self.P = (np.eye(2) - K @ H) @ self.P

        # Extract parameters
        hedge_ratio = float(self.x[0, 0])
        intercept = float(self.x[1, 0])

        # Spread and z-score
        spread = y - (hedge_ratio * x + intercept)
        spread_std = np.sqrt(abs(self.P[0, 0]) * x**2 + abs(self.P[1, 1]) + self.R)
        z_score = spread / spread_std if spread_std > 1e-10 else 0.0

        return {
            "hedge_ratio": hedge_ratio,
            "intercept": intercept,
            "spread": float(spread),
            "spread_std": float(spread_std),
            "z_score": float(z_score),
            "residual": residual_scalar,
        }

    def backtest(
        self,
        y: pd.Series,
        x: pd.Series,
        initial_capital: float = 100000.0,
        warmup: int = 30,
    ) -> Dict:
        """
        Backtest Kalman pairs trading strategy.

        Args:
            y: Target price series
            x: Hedge price series
            initial_capital: Starting capital
            warmup: Number of periods for Kalman warmup

        Returns:
            dict with total_return, sharpe_ratio, max_drawdown,
            n_trades, trades, equity_curve
        """
        if len(y) != len(x):
            raise ValueError("y and x must have same length")
        if len(y) < warmup + 10:
            raise ValueError("Insufficient data for backtest")

        self.initialize_filter(y.values, x.values)

        # Warmup
        for i in range(warmup):
            self.update(float(y.iloc[i]), float(x.iloc[i]))

        # Trading
        position = 0
        capital = initial_capital
        trades: List[Dict] = []
        equity = [initial_capital] * warmup
        entry_z = 0.0

        for i in range(warmup, len(y)):
            result = self.update(float(y.iloc[i]), float(x.iloc[i]))
            z_score = result["z_score"]

            if position == 0:
                if z_score < -self.entry_z:
                    position = 1
                    entry_z = z_score
                    trades.append({"index": i, "action": "long", "z_score": z_score})
                elif z_score > self.entry_z:
                    position = -1
                    entry_z = z_score
                    trades.append({"index": i, "action": "short", "z_score": z_score})

            elif position == 1 and z_score > -self.exit_z:
                pnl = (entry_z - z_score) * 1000  # Simplified P&L
                capital += pnl
                trades.append(
                    {"index": i, "action": "exit_long", "z_score": z_score, "pnl": pnl}
                )
                position = 0

            elif position == -1 and z_score < self.exit_z:
                pnl = (z_score - entry_z) * 1000
                capital += pnl
                trades.append(
                    {"index": i, "action": "exit_short", "z_score": z_score, "pnl": pnl}
                )
                position = 0

            equity.append(capital)

        # Metrics
        equity_series = pd.Series(equity, index=y.index[: len(equity)])
        returns = equity_series.pct_change().dropna()

        total_return = (capital - initial_capital) / initial_capital
        sharpe = (
            float(returns.mean() / returns.std() * np.sqrt(252))
            if len(returns) > 0 and returns.std() > 0
            else 0.0
        )

        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak.clip(lower=1e-8)
        max_dd = float(drawdown.min())

        completed_trades = [t for t in trades if "pnl" in t]
        logger.info(
            "Kalman backtest: return=%.2f%%, sharpe=%.2f, maxDD=%.2f%%, trades=%d",
            total_return * 100,
            sharpe,
            max_dd * 100,
            len(completed_trades),
        )

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": len(completed_trades),
            "trades": trades,
            "equity_curve": equity_series,
        }


class MultiPairsPortfolio:
    """
    Portfolio of multiple pairs trading strategies.

    Diversifies across multiple cointegrated pairs using
    Kalman-filtered hedge ratios. Manages capital allocation
    and generates portfolio-level signals.
    """

    def __init__(
        self,
        max_pairs: int = 5,
        capital_per_pair: float = 20000.0,
        correlation_threshold: float = 0.7,
    ):
        """
        Args:
            max_pairs: Maximum number of pairs to trade simultaneously
            capital_per_pair: Capital allocated to each pair
            correlation_threshold: Minimum correlation for pair candidates
        """
        if max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")

        self.max_pairs = max_pairs
        self.capital_per_pair = capital_per_pair
        self.correlation_threshold = correlation_threshold
        self.pairs: List[str] = []
        self.traders: Dict[str, Dict] = {}

    def select_pairs(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Select best pairs for trading based on correlation and cointegration.

        Args:
            prices: Price data for all candidate symbols

        Returns:
            DataFrame of selected pairs with scores
        """
        returns = prices.pct_change().dropna()
        corr = returns.corr()

        candidates = []
        symbols = corr.columns.tolist()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if abs(corr.iloc[i, j]) > self.correlation_threshold:
                    candidates.append(
                        {
                            "symbol_1": symbols[i],
                            "symbol_2": symbols[j],
                            "correlation": float(corr.iloc[i, j]),
                        }
                    )

        if not candidates:
            logger.warning(
                "No pairs exceed correlation threshold %.2f",
                self.correlation_threshold,
            )
            return pd.DataFrame()

        candidates_df = pd.DataFrame(candidates)

        # Test cointegration for each candidate
        coint_results = []
        strategy = PairsTradingStrategy()

        for _, pair in candidates_df.iterrows():
            y = prices[pair["symbol_1"]]
            x = prices[pair["symbol_2"]]

            common_idx = y.index.intersection(x.index)
            y = y.loc[common_idx].dropna()
            x = x.loc[common_idx].dropna()

            common_idx = y.index.intersection(x.index)
            y = y.loc[common_idx]
            x = x.loc[common_idx]

            if len(y) < 60:
                continue

            try:
                result = strategy._engle_granger(y, x)
                if result["cointegrated"]:
                    coint_results.append(
                        {
                            **pair.to_dict(),
                            "adf_pvalue": result["adf_pvalue"],
                            "hedge_ratio": result["hedge_ratio"],
                            "half_life": result.get("half_life", np.nan),
                        }
                    )
            except Exception as e:
                logger.debug(
                    "Cointegration test failed for %s-%s: %s",
                    pair["symbol_1"],
                    pair["symbol_2"],
                    e,
                )

        if not coint_results:
            logger.info(
                "No cointegrated pairs found among %d candidates", len(candidates)
            )
            return pd.DataFrame()

        result_df = pd.DataFrame(coint_results)

        result_df["score"] = (1 - result_df["adf_pvalue"]) * 0.5 + (
            1 / (1 + result_df["half_life"].fillna(100))
        ) * 0.5

        selected = result_df.nlargest(self.max_pairs, "score")
        logger.info(
            "Selected %d pairs from %d cointegrated candidates",
            len(selected),
            len(coint_results),
        )
        return selected

    def initialize_traders(self, selected_pairs: pd.DataFrame) -> None:
        """
        Initialize Kalman traders for selected pairs.

        Args:
            selected_pairs: DataFrame of selected pairs from select_pairs()
        """
        self.pairs = []
        self.traders = {}

        for _, pair in selected_pairs.iterrows():
            pair_key = f"{pair['symbol_1']}_{pair['symbol_2']}"
            self.pairs.append(pair_key)
            self.traders[pair_key] = {
                "pair": pair.to_dict(),
                "trader": KalmanPairsTrader(),
                "capital": self.capital_per_pair,
                "initialized": False,
            }

        logger.info("Initialized %d Kalman traders", len(self.traders))

    def generate_portfolio_signals(self, prices: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate signals for all pairs in portfolio.

        Args:
            prices: Current price data (must contain all pair symbols)

        Returns:
            dict mapping pair_key to signal info
        """
        signals = {}

        for pair_key, trader_info in self.traders.items():
            pair = trader_info["pair"]
            trader: KalmanPairsTrader = trader_info["trader"]

            sym1 = pair["symbol_1"]
            sym2 = pair["symbol_2"]

            if sym1 not in prices.columns or sym2 not in prices.columns:
                logger.warning("Missing price data for pair %s", pair_key)
                continue

            y_series = prices[sym1].dropna()
            x_series = prices[sym2].dropna()

            if len(y_series) == 0 or len(x_series) == 0:
                continue

            # Initialize if needed
            if not trader_info["initialized"]:
                try:
                    trader.initialize_filter(y_series.values, x_series.values)
                    trader_info["initialized"] = True
                    # Warmup
                    n_warmup = min(30, len(y_series) - 1)
                    for i in range(n_warmup):
                        trader.update(float(y_series.iloc[i]), float(x_series.iloc[i]))
                except Exception as e:
                    logger.warning(
                        "Failed to initialize trader for %s: %s", pair_key, e
                    )
                    continue

            # Update with latest observation
            try:
                result = trader.update(
                    float(y_series.iloc[-1]),
                    float(x_series.iloc[-1]),
                )

                z = result["z_score"]
                signal = (
                    "long"
                    if z < -trader.entry_z
                    else ("short" if z > trader.entry_z else "none")
                )

                signals[pair_key] = {
                    "z_score": z,
                    "hedge_ratio": result["hedge_ratio"],
                    "spread": result["spread"],
                    "signal": signal,
                }
            except Exception as e:
                logger.warning("Failed to update trader for %s: %s", pair_key, e)

        return signals
