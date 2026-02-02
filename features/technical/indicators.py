"""
Technical Indicators Module - V1
Efficient numpy-based technical analysis indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    return pd.Series(data).rolling(window=period).mean().values


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        close: Closing prices
        period: RSI period

    Returns:
        RSI values (0-100)
    """
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal)
    Histogram = MACD - Signal

    Args:
        close: Closing prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    close: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Middle = SMA(period)
    Upper = Middle + std_dev * StdDev
    Lower = Middle - std_dev * StdDev

    Args:
        close: Closing prices
        period: SMA period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (upper, middle, lower)
    """
    middle = _sma(close, period)
    rolling_std = pd.Series(close).rolling(window=period).std().values

    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    return upper, middle, lower


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Average True Range.

    TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA(TR, period)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values
    """
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_values = _ema(true_range, period)

    return atr_values


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator.

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K lookback period
        d_period: %D smoothing period

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = pd.Series(low).rolling(window=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period).max().values

    k_values = (
        np.divide(
            (close - lowest_low),
            (highest_high - lowest_low),
            out=np.zeros_like(close),
            where=(highest_high - lowest_low) != 0,
        )
        * 100
    )

    d_values = _sma(k_values, d_period)

    return k_values, d_values


def adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Average Directional Index.

    Measures trend strength regardless of direction.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        ADX values (0-100)
    """
    atr_values = atr(high, low, close, period)

    # Directional Movement
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Directional Indicators
    plus_di = 100 * _ema(plus_dm, period) / np.where(atr_values > 0, atr_values, 1)
    minus_di = 100 * _ema(minus_dm, period) / np.where(atr_values > 0, atr_values, 1)

    # ADX
    dx = (
        100
        * np.abs(plus_di - minus_di)
        / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    )
    adx_values = _ema(dx, period)

    return adx_values


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume.

    Cumulative volume indicator based on price direction.

    Args:
        close: Close prices
        volume: Volume

    Returns:
        OBV values
    """
    direction = np.sign(np.diff(close, prepend=close[0]))
    obv_values = np.cumsum(direction * volume)
    return obv_values


def vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Volume Weighted Average Price.

    VWAP = cumsum(Typical Price * Volume) / cumsum(Volume)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume

    Returns:
        VWAP values
    """
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)

    vwap_values = np.divide(
        cumulative_tp_vol,
        cumulative_vol,
        out=np.zeros_like(cumulative_tp_vol),
        where=cumulative_vol != 0,
    )

    return vwap_values


class TechnicalFeatureGenerator:
    """
    Generates technical features for a DataFrame.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features for a DataFrame.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added feature columns
        """
        result = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # RSI
        result["rsi"] = rsi(close, self.rsi_period)

        # MACD
        macd_line, signal_line, hist = macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_hist"] = hist

        # Bollinger Bands
        upper, middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower
        result["bb_position"] = (close - lower) / np.where(
            (upper - lower) > 0, upper - lower, 1
        )

        # ATR
        result["atr"] = atr(high, low, close, self.atr_period)
        result["atr_pct"] = result["atr"] / close

        # Stochastic
        k, d = stochastic(high, low, close)
        result["stoch_k"] = k
        result["stoch_d"] = d

        # ADX
        result["adx"] = adx(high, low, close)

        # OBV
        result["obv"] = obv(close, volume)
        result["obv_sma"] = _sma(result["obv"].values, 20)

        # VWAP (for intraday, less meaningful for daily)
        result["vwap"] = vwap(high, low, close, volume)

        # Normalized signals for model input
        result["rsi_signal"] = (result["rsi"] - 50) / 50  # -1 to 1
        result["bb_signal"] = 2 * (result["bb_position"] - 0.5)  # -1 to 1
        result["macd_signal_norm"] = np.tanh(result["macd_hist"] / result["atr"])

        return result
