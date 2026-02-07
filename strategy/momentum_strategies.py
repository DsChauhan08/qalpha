"""
Momentum Strategies Module

Implements multiple momentum-based trading strategies:
- MomentumStrategy: Time-series, cross-sectional, and dual momentum
- TrendFollowingStrategy: MA crossover, Donchian breakout, Turtle trading
- MomentumRegimeStrategy: Regime-adaptive momentum with variable lookbacks

Key improvements over basic implementations:
- Proper Optional type annotations (no mutable default args)
- Vectorized operations where possible (Turtle still requires state tracking)
- ATR-based position sizing
- Volatility filtering for momentum strategies
- Proper logging throughout
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """
    Multi-horizon momentum trading strategies.

    Includes:
    - Time-series momentum: long/short based on own past returns
    - Cross-sectional momentum: rank assets by returns, long top N, short bottom N
    - Dual momentum: absolute + relative momentum filter
    - Volatility-filtered momentum: reduce exposure in high-vol regimes
    """

    def __init__(
        self,
        lookback: int = 252,
        holding_period: int = 21,
        n_top: int = 10,
    ):
        """
        Args:
            lookback: Lookback period for momentum calculation (trading days)
            holding_period: Holding period for positions
            n_top: Number of top/bottom performers to select
        """
        if lookback < 5:
            raise ValueError("lookback must be >= 5")
        if n_top < 1:
            raise ValueError("n_top must be >= 1")

        self.lookback = lookback
        self.holding_period = holding_period
        self.n_top = n_top

    def time_series_momentum(self, prices: pd.Series) -> pd.Series:
        """
        Calculate time-series momentum signal.

        Goes long when past return is positive, short when negative.

        Args:
            prices: Price series

        Returns:
            pd.Series: Momentum signals (-1, 0, 1)
        """
        momentum = prices.pct_change(self.lookback)

        signals = pd.Series(0, index=prices.index, dtype=float)
        signals[momentum > 0] = 1.0
        signals[momentum < 0] = -1.0

        return signals

    def cross_sectional_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional momentum signals.

        Ranks assets by past returns. Longs top N, shorts bottom N.

        Args:
            prices: DataFrame of price series (columns = assets)

        Returns:
            DataFrame of position signals for each asset
        """
        momentum = prices.pct_change(self.lookback)

        # Rank assets each period (1 = best performer)
        ranks = momentum.rank(axis=1, ascending=False)

        n_assets = len(prices.columns)
        n_bottom = min(self.n_top, n_assets // 2)
        n_top = min(self.n_top, n_assets // 2)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[ranks <= n_top] = 1.0
        signals[ranks > n_assets - n_bottom] = -1.0

        return signals

    def dual_momentum(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Dual momentum: absolute + relative momentum.

        Only invests in assets that beat the risk-free rate (absolute)
        AND rank in the top N (relative).

        Args:
            prices: Price data
            risk_free_rate: Annual risk-free rate

        Returns:
            DataFrame of signals
        """
        returns = prices.pct_change(self.lookback)
        risk_free_return = risk_free_rate * (self.lookback / 252)

        # Absolute momentum filter
        positive_momentum = returns > risk_free_return

        # Relative momentum ranking
        ranks = returns.rank(axis=1, ascending=False)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Long top N with positive absolute momentum
        n_top = min(self.n_top, len(prices.columns))
        long_condition = (ranks <= n_top) & positive_momentum
        signals[long_condition] = 1.0

        return signals

    def momentum_with_volatility_filter(
        self,
        prices: pd.DataFrame,
        vol_lookback: int = 63,
        vol_threshold: float = 0.20,
    ) -> pd.DataFrame:
        """
        Cross-sectional momentum with volatility filter.

        Reduces position sizes when market volatility exceeds threshold.

        Args:
            prices: Price data
            vol_lookback: Volatility calculation window
            vol_threshold: Annualized volatility threshold

        Returns:
            DataFrame of volatility-adjusted signals
        """
        signals = self.cross_sectional_momentum(prices)

        # Market volatility (equal-weight portfolio proxy)
        market_returns = prices.mean(axis=1).pct_change()
        market_vol = market_returns.rolling(vol_lookback).std() * np.sqrt(252)

        # Scale down signals in high-vol regime
        high_vol_mask = market_vol > vol_threshold
        vol_scalar = pd.Series(1.0, index=prices.index)
        vol_scalar[high_vol_mask] = 0.5

        # Apply scalar to all columns
        for col in signals.columns:
            signals[col] = signals[col] * vol_scalar

        return signals


class TrendFollowingStrategy:
    """
    Trend following strategies using moving averages and breakouts.

    Includes:
    - Moving average crossover (golden/death cross)
    - Donchian channel breakout
    - Classic Turtle Trading system
    - ATR-based position sizing
    """

    def __init__(
        self,
        fast_ma: int = 50,
        slow_ma: int = 200,
        atr_period: int = 14,
    ):
        """
        Args:
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period
            atr_period: ATR period for position sizing
        """
        if fast_ma >= slow_ma:
            raise ValueError("fast_ma must be < slow_ma")

        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.atr_period = atr_period

    def moving_average_crossover(self, prices: pd.Series) -> pd.Series:
        """
        Generate signals from MA crossover.

        Long when fast MA > slow MA, short when fast MA < slow MA.

        Args:
            prices: Price series

        Returns:
            pd.Series: Signals (1=long, -1=short, 0=flat)
        """
        fast = prices.rolling(self.fast_ma).mean()
        slow = prices.rolling(self.slow_ma).mean()

        signals = pd.Series(0, index=prices.index, dtype=float)
        signals[fast > slow] = 1.0
        signals[fast < slow] = -1.0

        return signals

    def donchian_channel_breakout(
        self, prices: pd.Series, channel_period: int = 20
    ) -> pd.Series:
        """
        Donchian channel breakout strategy.

        Long on breakout above prior highest high.
        Short on breakdown below prior lowest low.

        Args:
            prices: Price series
            channel_period: Lookback for channel boundaries

        Returns:
            pd.Series: Signals
        """
        highest_high = prices.rolling(channel_period).max().shift(1)
        lowest_low = prices.rolling(channel_period).min().shift(1)

        signals = pd.Series(0, index=prices.index, dtype=float)
        signals[prices > highest_high] = 1.0
        signals[prices < lowest_low] = -1.0

        return signals

    def turtle_trading(
        self,
        prices: pd.Series,
        entry_period: int = 20,
        exit_period: int = 10,
    ) -> pd.Series:
        """
        Classic Turtle Trading system.

        Enter on N-day breakout, exit on shorter-period breakout in
        opposite direction. Maintains state (position tracking).

        Args:
            prices: Price series
            entry_period: Entry breakout lookback period
            exit_period: Exit breakout lookback period

        Returns:
            pd.Series: Signals with position state
        """
        entry_high = prices.rolling(entry_period).max().shift(1)
        entry_low = prices.rolling(entry_period).min().shift(1)
        exit_high = prices.rolling(exit_period).max().shift(1)
        exit_low = prices.rolling(exit_period).min().shift(1)

        signals = pd.Series(0.0, index=prices.index)
        position = 0

        for i in range(max(entry_period, exit_period) + 1, len(prices)):
            price = prices.iloc[i]
            e_high = entry_high.iloc[i]
            e_low = entry_low.iloc[i]
            x_high = exit_high.iloc[i]
            x_low = exit_low.iloc[i]

            # Skip NaN values
            if (
                np.isnan(e_high)
                or np.isnan(e_low)
                or np.isnan(x_high)
                or np.isnan(x_low)
            ):
                signals.iloc[i] = position
                continue

            if position == 0:
                if price > e_high:
                    position = 1
                elif price < e_low:
                    position = -1
            elif position == 1:
                if price < x_low:
                    position = 0
            elif position == -1:
                if price > x_high:
                    position = 0

            signals.iloc[i] = position

        return signals

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            pd.Series: ATR values
        """
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(self.atr_period).mean()

        return atr

    def position_sizing_by_atr(
        self,
        capital: float,
        atr: float,
        risk_per_trade: float = 0.01,
        atr_multiple: float = 2.0,
    ) -> float:
        """
        Calculate position size based on ATR risk.

        Position size = risk_amount / stop_distance
        where stop_distance = ATR * multiple.

        Args:
            capital: Account capital
            atr: Current ATR value
            risk_per_trade: Risk per trade as fraction of capital
            atr_multiple: ATR multiple for stop distance

        Returns:
            Position size (number of shares)
        """
        if atr <= 0 or capital <= 0:
            return 0.0

        risk_amount = capital * risk_per_trade
        stop_distance = atr * atr_multiple

        if stop_distance <= 0:
            return 0.0

        return risk_amount / stop_distance


class MomentumRegimeStrategy:
    """
    Momentum strategy that adapts to market regime.

    Detects the current regime (trending, mean_reverting, volatile)
    and adjusts momentum lookback and thresholds accordingly.

    Regime detection uses:
    - Annualized volatility for volatile regime
    - Linear regression R-squared for trend strength
    """

    def __init__(self, regimes: Optional[Dict[str, Dict]] = None):
        """
        Args:
            regimes: Dict of regime name -> parameter dict.
                     Each parameter dict should have 'lookback' and 'threshold'.
                     If None, uses sensible defaults.
        """
        if regimes is None:
            self.regimes = {
                "trending": {"lookback": 252, "threshold": 0.10},
                "mean_reverting": {"lookback": 63, "threshold": 0.05},
                "volatile": {"lookback": 21, "threshold": 0.02},
            }
        else:
            self.regimes = regimes

    def detect_regime(
        self,
        prices: pd.Series,
        vol_window: int = 63,
        trend_window: int = 252,
    ) -> str:
        """
        Detect current market regime.

        Classification rules:
        - volatile: annualized vol > 25%
        - trending: R² of linear fit > 0.7
        - mean_reverting: default (neither trending nor volatile)

        Args:
            prices: Price series
            vol_window: Volatility lookback window
            trend_window: Trend detection lookback window

        Returns:
            Regime name: 'volatile', 'trending', or 'mean_reverting'
        """
        returns = prices.pct_change().dropna()

        if len(returns) < vol_window:
            return "mean_reverting"

        # Annualized volatility
        volatility = float(returns.iloc[-vol_window:].std() * np.sqrt(252))

        if volatility > 0.25:
            return "volatile"

        # Trend strength via linear regression R²
        n_points = min(trend_window, len(prices))
        x = np.arange(n_points)
        y = prices.iloc[-n_points:].values

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend_strength = abs(r_value)
        except Exception:
            trend_strength = 0.0

        if trend_strength > 0.7:
            return "trending"

        return "mean_reverting"

    def generate_signals(
        self,
        prices: pd.DataFrame,
        n_long: int = 5,
        n_short: int = 5,
    ) -> pd.DataFrame:
        """
        Generate regime-adaptive momentum signals.

        For each time step (after warmup), detects regime and
        applies appropriate momentum lookback. Ranks assets
        and goes long top N, short bottom N.

        Args:
            prices: Price data (columns = assets)
            n_long: Number of assets to go long
            n_short: Number of assets to go short

        Returns:
            DataFrame of signals
        """
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_assets = len(prices.columns)

        n_long = min(n_long, n_assets // 2) if n_assets > 1 else 1
        n_short = min(n_short, n_assets // 2) if n_assets > 1 else 0

        # Need at least 252 periods for regime detection
        start_idx = min(252, len(prices) - 1)

        for i in range(start_idx, len(prices)):
            # Detect regime using equal-weight market average
            market_price = prices.iloc[:i].mean(axis=1)
            regime = self.detect_regime(market_price)

            params = self.regimes[regime]
            lookback = params["lookback"]

            if i < lookback:
                continue

            # Calculate momentum for each asset
            momentum = prices.iloc[i] / prices.iloc[i - lookback] - 1

            # Rank and signal
            ranks = momentum.rank(ascending=False)

            for col in prices.columns:
                if ranks[col] <= n_long:
                    signals.iloc[i, signals.columns.get_loc(col)] = 1.0
                elif ranks[col] > n_assets - n_short:
                    signals.iloc[i, signals.columns.get_loc(col)] = -1.0

        logger.info(
            "Generated regime-adaptive momentum signals for %d assets over %d periods",
            n_assets,
            len(prices),
        )

        return signals
