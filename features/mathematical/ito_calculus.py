"""
Ito Calculus Utilities for Derivatives Pricing and Hedging

Implements:
1. Ito's Lemma (numerical) - transform stochastic processes
2. Black-Scholes pricing - call and put options
3. Greeks computation - delta, gamma, theta, vega, rho
4. Implied volatility - Newton-Raphson / Brent solver

Applications:
- Options pricing and hedging
- Risk management via Greeks
- Volatility surface construction
- Portfolio delta/gamma neutralization
"""

import logging
from typing import Callable, Dict

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ItoCalculus:
    """
    Ito calculus utilities for derivatives pricing and hedging.
    """

    @staticmethod
    def ito_lemma_1d(
        f: Callable[[float, float], float],
        S: float,
        t: float,
        mu: float,
        sigma: float,
    ) -> Dict[str, float]:
        """
        Apply Ito's lemma to f(S, t) where S follows GBM.

        df = (df/dt) dt + (df/dS) dS + (1/2)(d^2f/dS^2)(dS)^2

        For GBM: dS = mu*S*dt + sigma*S*dW

        Args:
            f: Function f(S, t) - must accept (float, float)
            S: Current price
            t: Current time
            mu: Drift
            sigma: Volatility

        Returns:
            dict: Ito decomposition components
        """
        eps_S = max(S * 1e-5, 1e-8)
        eps_t = 1e-5

        # Partial derivatives (central differences)
        df_dS = (f(S + eps_S, t) - f(S - eps_S, t)) / (2 * eps_S)
        df_dt = (f(S, t + eps_t) - f(S, t - eps_t)) / (2 * eps_t)
        d2f_dS2 = (f(S + eps_S, t) - 2 * f(S, t) + f(S - eps_S, t)) / (eps_S**2)

        # Ito terms
        drift = df_dt + mu * S * df_dS + 0.5 * sigma**2 * S**2 * d2f_dS2
        diffusion = sigma * S * df_dS

        return {
            "drift": float(drift),
            "diffusion": float(diffusion),
            "df_dS": float(df_dS),
            "df_dt": float(df_dt),
            "d2f_dS2": float(d2f_dS2),
        }

    @staticmethod
    def delta_hedge(
        option_price: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Calculate delta for an option.

        Delta = dV/dS (central difference)

        Args:
            option_price: Option pricing function(S, K, T, r, sigma)
            S: Current price
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility

        Returns:
            float: Delta
        """
        eps = S * 0.001
        V_up = option_price(S + eps, K, T, r, sigma)
        V_down = option_price(S - eps, K, T, r, sigma)

        return float((V_up - V_down) / (2 * eps))

    @staticmethod
    def gamma(
        option_price: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate gamma (d^2V/dS^2)."""
        eps = S * 0.001
        delta_up = ItoCalculus.delta_hedge(option_price, S + eps, K, T, r, sigma)
        delta_down = ItoCalculus.delta_hedge(option_price, S - eps, K, T, r, sigma)

        return float((delta_up - delta_down) / (2 * eps))

    @staticmethod
    def theta(
        option_price: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate theta (dV/dt, time decay per day)."""
        eps = 1.0 / 365.0  # One day
        if T - eps <= 0:
            eps = T / 2.0
            if eps <= 0:
                return 0.0

        V_now = option_price(S, K, T, r, sigma)
        V_later = option_price(S, K, T - eps, r, sigma)

        return float((V_later - V_now) / eps)

    @staticmethod
    def vega(
        option_price: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate vega (dV/d_sigma)."""
        eps = 0.01
        V_up = option_price(S, K, T, r, sigma + eps)
        V_down = option_price(S, K, T, r, max(sigma - eps, 1e-6))

        return float((V_up - V_down) / (2 * eps))

    @staticmethod
    def rho(
        option_price: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate rho (dV/dr)."""
        eps = 0.001
        V_up = option_price(S, K, T, r + eps, sigma)
        V_down = option_price(S, K, T, r - eps, sigma)

        return float((V_up - V_down) / (2 * eps))

    @staticmethod
    def all_greeks(
        option_price: Callable[..., float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> Dict[str, float]:
        """
        Compute all Greeks for an option.

        Args:
            option_price: Pricing function(S, K, T, r, sigma)
            S, K, T, r, sigma: Option parameters

        Returns:
            dict: All Greeks
        """
        return {
            "price": float(option_price(S, K, T, r, sigma)),
            "delta": ItoCalculus.delta_hedge(option_price, S, K, T, r, sigma),
            "gamma": ItoCalculus.gamma(option_price, S, K, T, r, sigma),
            "theta": ItoCalculus.theta(option_price, S, K, T, r, sigma),
            "vega": ItoCalculus.vega(option_price, S, K, T, r, sigma),
            "rho": ItoCalculus.rho(option_price, S, K, T, r, sigma),
        }


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.

    C = S * N(d1) - K * exp(-rT) * N(d2)

    Where:
        d1 = (ln(S/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

    Args:
        S: Current price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        float: Call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price via put-call parity."""
    call = black_scholes_call(S, K, T, r, sigma)
    return float(call + K * np.exp(-r * T) - S)


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Calculate implied volatility from option price.

    Uses Brent's method for robust root-finding.

    Args:
        option_price: Observed market price
        S: Current price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        option_type: 'call' or 'put'
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        float: Implied volatility (NaN if no solution)
    """
    from scipy.optimize import brentq

    pricing_fn = black_scholes_call if option_type == "call" else black_scholes_put

    def objective(sigma: float) -> float:
        return pricing_fn(S, K, T, r, sigma) - option_price

    try:
        return float(brentq(objective, 1e-4, 5.0, xtol=tol, maxiter=max_iter))
    except ValueError:
        logger.warning(
            "Could not find implied volatility for price=%.4f, S=%.2f, K=%.2f, T=%.4f",
            option_price,
            S,
            K,
            T,
        )
        return float("nan")
