"""
Mean Reversion Strategies Module

Implements mean reversion trading strategies:
- MeanReversionStrategy: Bollinger Bands and RSI-based mean reversion
- StatisticalArbitrage: PCA-based factor model stat arb

Key features:
- Bollinger Band mean reversion with position state tracking
- RSI mean reversion with oversold/overbought thresholds
- PCA factor model for isolating idiosyncratic residual returns
- Z-score based signal generation on residuals
- Proper logging throughout
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Mean reversion trading strategies.

    Based on the principle that prices tend to revert to their
    statistical mean. Implements two approaches:
    - Bollinger Bands: price relative to rolling mean +/- std bands
    - RSI mean reversion: buy oversold, sell overbought
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        """
        Args:
            lookback: Lookback for rolling mean and std calculation
            entry_z: Z-score entry threshold (absolute value)
            exit_z: Z-score exit threshold (reversion target)
        """
        if lookback < 5:
            raise ValueError("lookback must be >= 5")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def bollinger_bands(self, prices: pd.Series, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and z-score.

        Args:
            prices: Price series
            num_std: Number of standard deviations for bands

        Returns:
            DataFrame with middle, upper, lower, z_score columns
        """
        middle = prices.rolling(self.lookback).mean()
        std = prices.rolling(self.lookback).std()

        upper = middle + num_std * std
        lower = middle - num_std * std

        z_score = (prices - middle) / std.clip(lower=1e-8)

        return pd.DataFrame(
            {
                "middle": middle,
                "upper": upper,
                "lower": lower,
                "z_score": z_score,
            }
        )

    def generate_bb_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate Bollinger Bands mean reversion signals.

        Buy when price touches lower band (oversold).
        Sell when price touches upper band (overbought).
        Exit long when price reverts to middle band.
        Exit short when price reverts to middle band.

        Args:
            prices: Price series

        Returns:
            pd.Series: Position signals (-1, 0, 1)
        """
        bb = self.bollinger_bands(prices)

        signals = pd.Series(0.0, index=prices.index)
        position = 0

        for i in range(1, len(prices)):
            lower = bb["lower"].iloc[i]
            upper = bb["upper"].iloc[i]
            mid = bb["middle"].iloc[i]

            if np.isnan(lower) or np.isnan(upper) or np.isnan(mid):
                signals.iloc[i] = position
                continue

            price = prices.iloc[i]

            if position == 0:
                if price < lower:
                    position = 1  # Buy (expect reversion up)
                elif price > upper:
                    position = -1  # Sell (expect reversion down)
            elif position == 1:
                if price > mid:
                    position = 0  # Exit long at mean
            elif position == -1:
                if price < mid:
                    position = 0  # Exit short at mean

            signals.iloc[i] = position

        return signals

    def rsi_mean_reversion(
        self,
        prices: pd.Series,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> pd.Series:
        """
        RSI-based mean reversion strategy.

        Buy when RSI drops below oversold level.
        Sell when RSI rises above overbought level.
        Exit at RSI = 50 (neutral).

        Args:
            prices: Price series
            period: RSI calculation period
            oversold: RSI oversold threshold
            overbought: RSI overbought threshold

        Returns:
            pd.Series: Position signals (-1, 0, 1)
        """
        # Calculate RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss.clip(lower=1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Generate signals with state tracking
        signals = pd.Series(0.0, index=prices.index)
        position = 0

        for i in range(period + 1, len(prices)):
            rsi_val = rsi.iloc[i]

            if np.isnan(rsi_val):
                signals.iloc[i] = position
                continue

            if position == 0:
                if rsi_val < oversold:
                    position = 1  # Buy oversold
                elif rsi_val > overbought:
                    position = -1  # Sell overbought
            elif position == 1:
                if rsi_val > 50:
                    position = 0  # Exit long at neutral
            elif position == -1:
                if rsi_val < 50:
                    position = 0  # Exit short at neutral

            signals.iloc[i] = position

        return signals

    def z_score_mean_reversion(self, prices: pd.Series) -> pd.DataFrame:
        """
        Z-score based mean reversion with full signal info.

        Provides z-score, position signals, exit signals,
        and signal confidence.

        Args:
            prices: Price series

        Returns:
            DataFrame with zscore, signal, exit_signal, signal_confidence
        """
        rolling_mean = prices.rolling(self.lookback).mean()
        rolling_std = prices.rolling(self.lookback).std().clip(lower=1e-8)
        z_score = (prices - rolling_mean) / rolling_std

        result = pd.DataFrame(index=prices.index)
        result["zscore"] = z_score

        # Entry signals
        result["signal"] = 0.0
        result.loc[z_score > self.entry_z, "signal"] = -1.0
        result.loc[z_score < -self.entry_z, "signal"] = 1.0

        # Exit signals
        result["exit_signal"] = (z_score.abs() < self.exit_z).astype(float)

        # Confidence based on z-score magnitude
        result["signal_confidence"] = np.clip(z_score.abs() / 3.0, 0, 1)

        return result


class StatisticalArbitrage:
    """
    Statistical arbitrage using PCA factor models.

    Decomposes asset returns into systematic (factor) and
    idiosyncratic (residual) components using PCA. Trades
    on mean reversion of residual returns.

    Steps:
    1. Fit PCA to standardized returns
    2. Project returns onto factor space
    3. Calculate residuals (actual - predicted)
    4. Trade on z-scored residuals
    """

    def __init__(self, n_factors: int = 5):
        """
        Args:
            n_factors: Number of principal components to extract
        """
        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")

        self.n_factors = n_factors
        self.factor_loadings: Optional[np.ndarray] = None
        self.factor_returns: Optional[np.ndarray] = None
        self._return_mean: Optional[pd.Series] = None
        self._return_std: Optional[pd.Series] = None

    def fit_factor_model(self, returns: pd.DataFrame) -> Dict:
        """
        Fit PCA factor model to return data.

        Args:
            returns: Return data (rows = time, columns = assets)

        Returns:
            dict with explained_variance, factor_loadings, factor_returns
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn required for PCA-based stat arb")

        # Store standardization parameters
        self._return_mean = returns.mean()
        self._return_std = returns.std().clip(lower=1e-8)

        # Standardize
        standardized = (returns - self._return_mean) / self._return_std

        # Drop rows with NaN
        clean_data = standardized.dropna()
        if len(clean_data) < self.n_factors:
            raise ValueError(
                f"Insufficient clean data ({len(clean_data)} rows) "
                f"for {self.n_factors} factors"
            )

        # Clamp n_factors to number of assets
        actual_factors = min(self.n_factors, len(returns.columns), len(clean_data))

        pca = PCA(n_components=actual_factors)
        factor_returns = pca.fit_transform(clean_data)

        self.factor_loadings = pca.components_.T  # (n_assets, n_factors)
        self.factor_returns = factor_returns  # (n_obs, n_factors)

        explained = pca.explained_variance_ratio_
        logger.info(
            "PCA factor model: %d factors explain %.1f%% of variance",
            actual_factors,
            float(np.sum(explained) * 100),
        )

        return {
            "explained_variance": explained,
            "factor_loadings": self.factor_loadings,
            "factor_returns": self.factor_returns,
        }

    def calculate_residual_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate residual returns after removing factor exposure.

        residual = actual - factor_loadings @ factor_returns.T

        Args:
            returns: Return data (must have same columns as training data)

        Returns:
            DataFrame of residual (idiosyncratic) returns
        """
        if self.factor_loadings is None or self.factor_returns is None:
            raise ValueError("Factor model not fitted. Call fit_factor_model first.")

        # Predicted returns from factor model
        predicted = self.factor_returns @ self.factor_loadings.T

        # Align dimensions
        n_obs = min(len(returns), predicted.shape[0])

        # Use the last n_obs rows from returns to match factor_returns
        actual = returns.iloc[-n_obs:].values
        pred = predicted[-n_obs:]

        residuals = actual - pred

        return pd.DataFrame(
            residuals,
            index=returns.index[-n_obs:],
            columns=returns.columns,
        )

    def generate_signals(
        self,
        returns: pd.DataFrame,
        residual_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Generate signals based on residual return z-scores.

        Long assets with negative residual z-score (underperformed factors).
        Short assets with positive residual z-score (outperformed factors).

        Args:
            returns: Return data
            residual_threshold: Z-score threshold for entry

        Returns:
            DataFrame of signals (-1, 0, 1)
        """
        residuals = self.calculate_residual_returns(returns)

        # Z-score the residuals cross-sectionally
        residual_mean = residuals.mean()
        residual_std = residuals.std().clip(lower=1e-8)
        residual_z = (residuals - residual_mean) / residual_std

        # Generate signals
        signals = pd.DataFrame(0.0, index=residuals.index, columns=residuals.columns)
        signals[residual_z < -residual_threshold] = 1.0  # Long underperformers
        signals[residual_z > residual_threshold] = -1.0  # Short outperformers

        n_long = int((signals == 1.0).sum().sum())
        n_short = int((signals == -1.0).sum().sum())
        logger.info(
            "Stat arb signals: %d long entries, %d short entries",
            n_long,
            n_short,
        )

        return signals
