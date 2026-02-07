"""
Stochastic Processes for Price Simulation and Risk Analysis

Implements key stochastic models:
1. Geometric Brownian Motion (GBM) - standard price dynamics
2. Jump Diffusion Process (Merton) - GBM + Poisson jumps for fat tails
3. Ornstein-Uhlenbeck Process - mean-reverting dynamics (spreads, vol)
4. Heston Stochastic Volatility - price + volatility co-dynamics

Applications:
- Monte Carlo option pricing
- Risk scenario generation
- Strategy stress testing
- Pairs trading spread modeling
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion for price simulation.

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Solution:
    S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*W_t)
    """

    def __init__(self, mu: float, sigma: float, S0: float):
        """
        Args:
            mu: Annual drift (expected return)
            sigma: Annual volatility
            S0: Initial price
        """
        if sigma < 0:
            raise ValueError("Volatility sigma must be non-negative")
        if S0 <= 0:
            raise ValueError("Initial price S0 must be positive")
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate GBM paths.

        Args:
            T: Time horizon in years
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            antithetic: Use antithetic variates for variance reduction

        Returns:
            np.ndarray: Shape (n_paths, n_steps+1)
        """
        dt = T / n_steps
        n_total = max(n_paths // 2, 1) if antithetic else n_paths

        # Generate random increments
        dW = np.random.randn(n_total, n_steps) * np.sqrt(dt)

        if antithetic:
            dW = np.concatenate([dW, -dW], axis=0)
            # Trim to exact n_paths if n_paths was odd
            dW = dW[:n_paths]

        # Cumulative sum for Brownian motion
        W = np.cumsum(dW, axis=1)
        W = np.concatenate([np.zeros((len(W), 1)), W], axis=1)

        # GBM formula (vectorized)
        t = np.linspace(0, T, n_steps + 1)
        exponent = (self.mu - 0.5 * self.sigma**2) * t + self.sigma * W
        S = self.S0 * np.exp(exponent)

        return S

    def transition_density(self, S_t: float, t: float, S_T: float, T: float) -> float:
        """
        Probability density of transitioning from S_t to S_T.

        Args:
            S_t: Price at time t
            t: Current time
            S_T: Target price
            T: Target time

        Returns:
            float: Probability density
        """
        tau = T - t
        if tau <= 0:
            return 1.0 if np.isclose(S_t, S_T) else 0.0

        log_return = np.log(S_T / S_t)
        mean = (self.mu - 0.5 * self.sigma**2) * tau
        std = self.sigma * np.sqrt(tau)

        return float(norm.pdf(log_return, mean, std))

    def probability_above(self, S_t: float, t: float, K: float, T: float) -> float:
        """
        Probability of price being above K at time T.

        Args:
            S_t: Current price
            t: Current time
            K: Strike/target price
            T: Target time

        Returns:
            float: Probability
        """
        tau = T - t
        if tau <= 0:
            return 1.0 if S_t > K else 0.0

        d2 = (np.log(S_t / K) + (self.mu - 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        return float(norm.cdf(d2))

    def expected_price(self, t: float) -> float:
        """Expected price at time t."""
        return self.S0 * np.exp(self.mu * t)

    def variance_at(self, t: float) -> float:
        """Variance of price at time t."""
        return (self.S0**2) * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1)


class JumpDiffusionProcess:
    """
    Merton's Jump Diffusion model.

    Adds Poisson jumps to GBM for more realistic price dynamics
    (fat tails, sudden moves).

    dS_t = mu * S_t * dt + sigma * S_t * dW_t + S_t * dJ_t

    Where J_t is a compound Poisson process with lognormal jump sizes.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lambda_jump: float,
        mu_jump: float,
        sigma_jump: float,
        S0: float,
    ):
        """
        Args:
            mu: Drift
            sigma: Diffusion volatility
            lambda_jump: Jump intensity (jumps per year)
            mu_jump: Mean jump size (lognormal)
            sigma_jump: Jump size volatility
            S0: Initial price
        """
        if S0 <= 0:
            raise ValueError("Initial price S0 must be positive")
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.S0 = S0

    def simulate(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulate jump diffusion paths.

        Args:
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths

        Returns:
            np.ndarray: Simulated paths, shape (n_paths, n_steps+1)
        """
        dt = T / n_steps

        # Diffusion component via GBM
        gbm = GeometricBrownianMotion(self.mu, self.sigma, self.S0)
        paths = gbm.simulate(T, n_steps, n_paths, antithetic=False)

        # Add jumps (vectorized Poisson draws)
        jump_counts = np.random.poisson(self.lambda_jump * dt, (n_paths, n_steps))

        for i in range(n_paths):
            for j in range(n_steps):
                n_jumps = jump_counts[i, j]
                if n_jumps > 0:
                    # Lognormal jump sizes
                    jump_sizes = np.exp(
                        np.random.randn(n_jumps) * self.sigma_jump + self.mu_jump
                    )
                    total_jump = np.prod(jump_sizes)
                    paths[i, j + 1 :] *= total_jump

        return paths

    def total_volatility(self) -> float:
        """Total annualized volatility including jump component."""
        jump_var = self.lambda_jump * (self.sigma_jump**2 + self.mu_jump**2)
        return np.sqrt(self.sigma**2 + jump_var)


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck process for mean-reverting quantities.

    dx_t = theta * (mu - x_t) * dt + sigma * dW_t

    Applications:
    - Interest rate modeling (Vasicek model)
    - Volatility modeling
    - Pairs trading spread dynamics
    """

    def __init__(self, theta: float, mu: float, sigma: float, x0: float):
        """
        Args:
            theta: Speed of mean reversion (> 0)
            mu: Long-term mean
            sigma: Volatility
            x0: Initial value
        """
        if theta <= 0:
            raise ValueError("Mean reversion speed theta must be positive")
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0

    def simulate(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simulate OU paths using exact discretization."""
        dt = T / n_steps

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.x0

        # Exact discretization parameters
        exp_minus_theta_dt = np.exp(-self.theta * dt)
        sqrt_var = self.sigma * np.sqrt(
            (1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta)
        )

        # Vectorized simulation
        noise = np.random.randn(n_paths, n_steps) * sqrt_var

        for j in range(n_steps):
            paths[:, j + 1] = (
                self.mu + (paths[:, j] - self.mu) * exp_minus_theta_dt + noise[:, j]
            )

        return paths

    def stationary_variance(self) -> float:
        """Long-term variance of the process."""
        return self.sigma**2 / (2 * self.theta)

    def stationary_std(self) -> float:
        """Long-term standard deviation."""
        return np.sqrt(self.stationary_variance())

    def half_life(self) -> float:
        """Time to revert halfway to mean."""
        return np.log(2) / self.theta

    def expected_value(self, t: float) -> float:
        """Expected value at time t."""
        return self.mu + (self.x0 - self.mu) * np.exp(-self.theta * t)

    @staticmethod
    def fit_from_data(data: np.ndarray, dt: float = 1.0) -> "OrnsteinUhlenbeckProcess":
        """
        Estimate OU parameters from time series data.

        Uses OLS on the discretized OU equation:
        x_{t+1} - x_t = theta*(mu - x_t)*dt + sigma*sqrt(dt)*epsilon

        Args:
            data: Time series array
            dt: Time step size

        Returns:
            Fitted OrnsteinUhlenbeckProcess instance
        """
        x = data[:-1]
        dx = np.diff(data)

        # OLS: dx = a + b*x + residual => theta = -b/dt, mu = -a/(b)
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, dx, rcond=None)[0]

        a, b = beta[0], beta[1]
        theta = -b / dt
        mu = -a / b if abs(b) > 1e-12 else float(np.mean(data))
        residuals = dx - X @ beta
        sigma = float(np.std(residuals) / np.sqrt(dt))
        x0 = float(data[0])

        if theta <= 0:
            logger.warning(
                "Estimated theta=%.4f is non-positive; data may not be mean-reverting. "
                "Clamping to 0.001.",
                theta,
            )
            theta = 0.001

        return OrnsteinUhlenbeckProcess(theta=theta, mu=mu, sigma=sigma, x0=x0)


class HestonModel:
    """
    Heston stochastic volatility model.

    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^S
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v

    Where dW_t^S * dW_t^v = rho * dt

    Captures:
    - Volatility clustering
    - Leverage effect (negative rho)
    - Fat-tailed return distributions
    """

    def __init__(
        self,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        S0: float,
    ):
        """
        Args:
            mu: Drift
            v0: Initial variance
            kappa: Mean reversion speed for variance
            theta: Long-term variance
            xi: Volatility of volatility (vol-of-vol)
            rho: Correlation between price and variance Brownian motions
            S0: Initial price
        """
        if S0 <= 0:
            raise ValueError("Initial price S0 must be positive")
        if v0 < 0:
            raise ValueError("Initial variance v0 must be non-negative")
        if abs(rho) > 1:
            raise ValueError("Correlation rho must be in [-1, 1]")
        self.mu = mu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.S0 = S0

    def simulate(
        self, T: float, n_steps: int, n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using Euler-Maruyama with truncation.

        Args:
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths

        Returns:
            Tuple of (price_paths, variance_paths), each (n_paths, n_steps+1)
        """
        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        sqrt_dt = np.sqrt(dt)
        sqrt_one_minus_rho2 = np.sqrt(1 - self.rho**2)

        for j in range(n_steps):
            # Correlated Brownian motions (vectorized over paths)
            Z1 = np.random.randn(n_paths)
            Z2 = self.rho * Z1 + sqrt_one_minus_rho2 * np.random.randn(n_paths)

            v_pos = np.maximum(v[:, j], 0)
            sqrt_v = np.sqrt(v_pos)

            # Variance process (truncated at 0)
            v[:, j + 1] = np.maximum(
                v[:, j]
                + self.kappa * (self.theta - v[:, j]) * dt
                + self.xi * sqrt_v * sqrt_dt * Z2,
                0,
            )

            # Price process (log-Euler scheme)
            S[:, j + 1] = S[:, j] * np.exp(
                (self.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1
            )

        return S, v

    def feller_condition(self) -> bool:
        """
        Check Feller condition: 2*kappa*theta > xi^2

        Ensures variance process stays strictly positive.
        """
        return 2 * self.kappa * self.theta > self.xi**2

    def long_term_volatility(self) -> float:
        """Long-term average volatility."""
        return np.sqrt(self.theta)
