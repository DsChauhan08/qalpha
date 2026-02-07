"""
Advanced Time Series Models for Financial Analysis

Implements:
1. ARMA - Autoregressive Moving Average
2. GARCH - Generalized Autoregressive Conditional Heteroskedasticity
3. CointegrationTest - Engle-Granger and ADF tests

Uses statsmodels where available for production-quality tests,
with pure-numpy fallbacks.

Applications:
- Return forecasting (ARMA)
- Volatility modeling and forecasting (GARCH)
- Pairs trading pair selection (Cointegration)
- Regime detection via volatility clustering
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ARMA:
    """
    Autoregressive Moving Average model.

    X_t = c + sum(phi_i * X_{t-i}) + sum(theta_j * eps_{t-j}) + eps_t

    Fitted via conditional maximum likelihood.
    """

    def __init__(self, p: int = 1, q: int = 0):
        """
        Args:
            p: AR order
            q: MA order
        """
        if p < 0 or q < 0:
            raise ValueError("Orders p and q must be non-negative")
        self.p = p
        self.q = q
        self.params: Optional[np.ndarray] = None
        self.sigma2: Optional[float] = None
        self._fitted_data: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ARMA":
        """
        Fit ARMA model using maximum likelihood.

        Args:
            X: Time series data

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n < max(self.p, self.q) + 10:
            raise ValueError(
                f"Need at least {max(self.p, self.q) + 10} observations, got {n}"
            )

        self._fitted_data = X.copy()

        # Initialize parameters: c, phi_1..phi_p, theta_1..theta_q
        n_params = 1 + self.p + self.q
        init_params = np.zeros(n_params)
        init_params[0] = float(np.mean(X))

        def neg_log_lik(params: np.ndarray) -> float:
            c = params[0]
            phi = params[1 : 1 + self.p]
            theta = params[1 + self.p :]

            residuals = self._compute_residuals(X, c, phi, theta)
            if len(residuals) == 0:
                return 1e10

            sigma2 = float(np.mean(residuals**2))
            if sigma2 <= 0:
                return 1e10

            nll = 0.5 * len(residuals) * np.log(2 * np.pi * sigma2)
            nll += 0.5 * np.sum(residuals**2) / sigma2

            return float(nll)

        result = minimize(neg_log_lik, init_params, method="L-BFGS-B")

        self.params = result.x
        residuals = self._compute_residuals(
            X,
            self.params[0],
            self.params[1 : 1 + self.p],
            self.params[1 + self.p :],
        )
        self.sigma2 = float(np.var(residuals)) if len(residuals) > 0 else 1e-6

        return self

    def _compute_residuals(
        self,
        X: np.ndarray,
        c: float,
        phi: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Compute residuals given parameters."""
        n = len(X)
        residuals = np.zeros(n)

        max_lag = max(self.p, self.q, 1)

        for t in range(max_lag, n):
            # AR component
            ar_term = sum(phi[i] * X[t - i - 1] for i in range(min(self.p, t)))

            # MA component
            ma_term = sum(
                theta[j] * residuals[t - j - 1] for j in range(min(self.q, t))
            )

            X_pred = c + ar_term + ma_term
            residuals[t] = X[t] - X_pred

        return residuals[max_lag:]

    def forecast(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts.

        Args:
            steps: Number of steps to forecast

        Returns:
            Tuple of (forecasts, standard_errors)
        """
        if self.params is None or self.sigma2 is None:
            raise ValueError("Model not fitted")

        c = self.params[0]
        phi = self.params[1 : 1 + self.p]

        forecasts = np.zeros(steps)

        for h in range(steps):
            forecast = c
            for i in range(min(self.p, h)):
                forecast += phi[i] * forecasts[h - i - 1]
            # For the first few steps, use last observed values
            if h < self.p and self._fitted_data is not None:
                for i in range(h, self.p):
                    forecast += phi[i] * self._fitted_data[-(i - h + 1)]
            forecasts[h] = forecast

        # Standard errors grow with horizon
        ses = np.sqrt(self.sigma2 * np.arange(1, steps + 1))

        return forecasts, ses

    def aic(self) -> float:
        """Akaike Information Criterion."""
        if self.params is None or self.sigma2 is None or self._fitted_data is None:
            raise ValueError("Model not fitted")
        n = len(self._fitted_data)
        k = len(self.params) + 1  # +1 for sigma2
        nll = 0.5 * n * np.log(2 * np.pi * self.sigma2) + 0.5 * n
        return float(2 * k + 2 * nll)

    def bic(self) -> float:
        """Bayesian Information Criterion."""
        if self.params is None or self.sigma2 is None or self._fitted_data is None:
            raise ValueError("Model not fitted")
        n = len(self._fitted_data)
        k = len(self.params) + 1
        nll = 0.5 * n * np.log(2 * np.pi * self.sigma2) + 0.5 * n
        return float(k * np.log(n) + 2 * nll)


class GARCH:
    """
    Generalized Autoregressive Conditional Heteroskedasticity.

    Models time-varying volatility:
    sigma_t^2 = omega + sum(alpha_i * eps_{t-i}^2) + sum(beta_j * sigma_{t-j}^2)

    Fitted via quasi-maximum likelihood.
    """

    def __init__(self, p: int = 1, q: int = 1):
        """
        Args:
            p: GARCH order (beta terms)
            q: ARCH order (alpha terms)
        """
        if p < 0 or q < 0:
            raise ValueError("Orders p and q must be non-negative")
        self.p = p
        self.q = q
        self.params: Optional[np.ndarray] = None
        self.omega: Optional[float] = None
        self.alpha: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self._fitted_returns: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray) -> "GARCH":
        """
        Fit GARCH model.

        Args:
            returns: Return series (demeaned)

        Returns:
            self
        """
        returns = np.asarray(returns, dtype=float)
        self._fitted_returns = returns.copy()

        n_params = 1 + self.q + self.p  # omega, alpha_1..q, beta_1..p

        # Initial guess
        var_returns = float(np.var(returns))
        init_params = np.zeros(n_params)
        init_params[0] = var_returns * 0.1  # omega
        if self.q > 0:
            init_params[1 : 1 + self.q] = 0.1  # alpha
        if self.p > 0:
            init_params[1 + self.q :] = 0.8  # beta

        # Bounds: omega > 0, alpha >= 0, beta >= 0
        bounds = [(1e-8, None)] + [(1e-8, 0.999)] * (n_params - 1)

        def neg_log_lik(params: np.ndarray) -> float:
            sigma2 = self._compute_variance(returns, params)
            # Avoid log of non-positive values
            sigma2 = np.maximum(sigma2, 1e-10)

            nll = 0.5 * np.sum(np.log(2 * np.pi * sigma2))
            nll += 0.5 * np.sum(returns**2 / sigma2)

            if not np.isfinite(nll):
                return 1e10
            return float(nll)

        result = minimize(neg_log_lik, init_params, bounds=bounds, method="L-BFGS-B")

        self.params = result.x
        self.omega = float(self.params[0])
        self.alpha = self.params[1 : 1 + self.q]
        self.beta = self.params[1 + self.q :]

        return self

    def _compute_variance(self, returns: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute conditional variance series."""
        n = len(returns)
        omega = params[0]
        alpha = params[1 : 1 + self.q]
        beta = params[1 + self.q :]

        sigma2 = np.zeros(n)
        unconditional_var = float(np.var(returns))
        start = max(self.p, self.q)
        sigma2[:start] = unconditional_var

        for t in range(start, n):
            sigma2[t] = omega

            for i in range(self.q):
                if t - i - 1 >= 0:
                    sigma2[t] += alpha[i] * returns[t - i - 1] ** 2

            for j in range(self.p):
                if t - j - 1 >= 0:
                    sigma2[t] += beta[j] * sigma2[t - j - 1]

        return sigma2

    def forecast_volatility(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future volatility.

        Args:
            steps: Number of steps ahead

        Returns:
            np.ndarray: Forecasted volatilities (standard deviations)
        """
        if self.params is None or self._fitted_returns is None:
            raise ValueError("Model not fitted")

        persistence = float(np.sum(self.alpha) + np.sum(self.beta))
        long_term_var = (
            self.omega / (1 - persistence) if persistence < 1 else self.omega
        )

        # Start from last fitted variance
        sigma2_series = self._compute_variance(self._fitted_returns, self.params)
        last_var = sigma2_series[-1]
        last_eps2 = self._fitted_returns[-1] ** 2

        forecasts = np.zeros(steps)

        for h in range(steps):
            if h == 0:
                fv = self.omega
                for i in range(self.q):
                    idx = -(i + 1)
                    if abs(idx) <= len(self._fitted_returns):
                        fv += self.alpha[i] * self._fitted_returns[idx] ** 2
                for j in range(self.p):
                    idx = -(j + 1)
                    if abs(idx) <= len(sigma2_series):
                        fv += self.beta[j] * sigma2_series[idx]
                forecasts[h] = fv
            else:
                # Multi-step: sigma2_h = omega + (alpha+beta)*sigma2_{h-1}
                # converges to long_term_var
                forecasts[h] = self.omega + persistence * forecasts[h - 1]

        return np.sqrt(np.maximum(forecasts, 0))

    def get_conditional_volatility(self) -> np.ndarray:
        """Get fitted conditional volatility series."""
        if self.params is None or self._fitted_returns is None:
            raise ValueError("Model not fitted")

        sigma2 = self._compute_variance(self._fitted_returns, self.params)
        return np.sqrt(np.maximum(sigma2, 0))

    def persistence(self) -> float:
        """Sum of alpha + beta. Values close to 1 indicate high persistence."""
        if self.alpha is None or self.beta is None:
            raise ValueError("Model not fitted")
        return float(np.sum(self.alpha) + np.sum(self.beta))

    def unconditional_variance(self) -> float:
        """Long-run unconditional variance."""
        if self.omega is None:
            raise ValueError("Model not fitted")
        p = self.persistence()
        if p >= 1:
            logger.warning("Persistence >= 1, unconditional variance is infinite")
            return float("inf")
        return self.omega / (1 - p)


class CointegrationTest:
    """
    Cointegration testing for pairs/basket identification.

    Provides:
    - Engle-Granger two-step test
    - ADF unit root test (statsmodels if available, else fallback)
    """

    def __init__(self, significance: float = 0.05):
        """
        Args:
            significance: Significance level for tests
        """
        self.significance = significance

    def engle_granger(self, y: np.ndarray, x: np.ndarray) -> Dict:
        """
        Engle-Granger two-step cointegration test.

        Step 1: OLS regression y = alpha + beta*x + epsilon
        Step 2: ADF test on residuals epsilon

        Args:
            y: First time series
            x: Second time series

        Returns:
            dict: Test results including cointegration boolean, hedge ratio, etc.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if len(y) != len(x):
            raise ValueError("y and x must have the same length")

        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        residuals = y - (intercept + slope * x)

        adf_result = self._adf_test(residuals)

        # Calculate half-life of mean reversion
        half_life = self._compute_half_life(residuals)

        return {
            "cointegrated": adf_result["p_value"] < self.significance,
            "adf_statistic": adf_result["statistic"],
            "adf_pvalue": adf_result["p_value"],
            "hedge_ratio": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "half_life": half_life,
            "residuals": residuals,
        }

    def _adf_test(self, series: np.ndarray, maxlag: int = 1) -> Dict:
        """
        Augmented Dickey-Fuller test for unit root.

        Uses statsmodels if available for accurate critical values;
        otherwise falls back to simplified OLS-based test.
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(series, maxlag=maxlag, autolag=None)
            return {
                "statistic": float(result[0]),
                "p_value": float(result[1]),
                "used_lag": result[2],
                "n_obs": result[3],
                "critical_values": result[4],
            }
        except ImportError:
            logger.warning("statsmodels not available, using simplified ADF test")

        # Fallback: simplified ADF
        diff = np.diff(series)
        lagged = series[:-1]

        X = np.column_stack([np.ones(len(lagged)), lagged])
        y_vec = diff

        beta = np.linalg.lstsq(X, y_vec, rcond=None)[0]
        residuals = y_vec - X @ beta

        mse = float(np.mean(residuals**2))
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(var_beta[1, 1])

        t_stat = beta[1] / se_beta1

        # Use Dickey-Fuller critical value approximation
        # For n > 100: 1% ~ -3.43, 5% ~ -2.86, 10% ~ -2.57
        # Approximate p-value using standard normal (conservative)
        p_value = float(2 * (1 - norm.cdf(abs(t_stat))))

        return {
            "statistic": float(t_stat),
            "p_value": p_value,
            "beta": beta,
        }

    @staticmethod
    def _compute_half_life(residuals: np.ndarray) -> float:
        """Compute half-life of mean reversion from residual series."""
        lagged = residuals[:-1]
        delta = np.diff(residuals)

        if len(lagged) < 10:
            return float("nan")

        # OLS: delta = a + b * lagged
        X = np.column_stack([np.ones(len(lagged)), lagged])
        beta = np.linalg.lstsq(X, delta, rcond=None)[0]

        b = beta[1]
        if b < 0:
            return float(-np.log(2) / b)
        return float("nan")
