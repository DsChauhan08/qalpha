"""
Copula Models for Dependence Structure Analysis

Implements:
1. Gaussian Copula - linear correlation in normal space
2. Clayton Copula - lower tail dependence (crash correlation)
3. Frank Copula - symmetric tail dependence
4. Gumbel Copula - upper tail dependence

Applications:
- Portfolio risk: tail dependence during crises
- Pairs trading: non-linear dependence modeling
- Risk management: joint extreme event probability
- Multi-asset option pricing

All copulas are parameterized by theta and support:
- PDF/CDF evaluation
- Sampling
- Tail dependence coefficients
- Parameter fitting via maximum likelihood
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, norm

logger = logging.getLogger(__name__)


class Copula(ABC):
    """Abstract base class for bivariate copulas."""

    @abstractmethod
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Copula CDF C(u, v)."""

    @abstractmethod
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Copula density c(u, v) = d^2 C / (du dv)."""

    @abstractmethod
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n samples from the copula."""

    @abstractmethod
    def lower_tail_dependence(self) -> float:
        """Lower tail dependence coefficient lambda_L."""

    @abstractmethod
    def upper_tail_dependence(self) -> float:
        """Upper tail dependence coefficient lambda_U."""

    def log_likelihood(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute log-likelihood for observations (u, v) in [0,1]^2."""
        density = self.pdf(u, v)
        density = np.maximum(density, 1e-20)
        return float(np.sum(np.log(density)))

    def aic(self, u: np.ndarray, v: np.ndarray, k: int = 1) -> float:
        """Akaike Information Criterion."""
        return float(-2 * self.log_likelihood(u, v) + 2 * k)


class GaussianCopula(Copula):
    """
    Gaussian copula parameterized by correlation rho.

    C(u, v) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)

    Properties:
    - Symmetric: lambda_L = lambda_U = 0 (no tail dependence)
    - rho in (-1, 1)
    """

    def __init__(self, rho: float = 0.5):
        if abs(rho) >= 1:
            raise ValueError("rho must be in (-1, 1)")
        self.rho = rho

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gaussian copula CDF via bivariate normal."""
        from scipy.stats import multivariate_normal

        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        x = norm.ppf(u)
        y = norm.ppf(v)

        cov = np.array([[1, self.rho], [self.rho, 1]])
        mvn = multivariate_normal(mean=[0, 0], cov=cov)

        result = np.array([mvn.cdf([xi, yi]) for xi, yi in zip(x, y)])
        return result

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gaussian copula density."""
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        x = norm.ppf(u)
        y = norm.ppf(v)

        rho2 = self.rho**2
        denom = 1 - rho2

        exponent = -(rho2 * (x**2 + y**2) - 2 * self.rho * x * y) / (2 * denom)
        density = np.exp(exponent) / np.sqrt(denom)
        return density

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from Gaussian copula."""
        mean = [0, 0]
        cov = [[1, self.rho], [self.rho, 1]]
        Z = np.random.multivariate_normal(mean, cov, n)
        u = norm.cdf(Z[:, 0])
        v = norm.cdf(Z[:, 1])
        return u, v

    def lower_tail_dependence(self) -> float:
        return 0.0

    def upper_tail_dependence(self) -> float:
        return 0.0

    @classmethod
    def fit(cls, u: np.ndarray, v: np.ndarray) -> "GaussianCopula":
        """Fit Gaussian copula to pseudo-observations."""
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        x = norm.ppf(u)
        y = norm.ppf(v)
        rho = float(np.corrcoef(x, y)[0, 1])
        rho = np.clip(rho, -0.999, 0.999)
        return cls(rho=rho)


class ClaytonCopula(Copula):
    """
    Clayton copula.

    C(u, v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}

    Properties:
    - Lower tail dependence: lambda_L = 2^{-1/theta}
    - No upper tail dependence: lambda_U = 0
    - theta in (0, inf)
    - Good for modeling crash dependence
    """

    def __init__(self, theta: float = 1.0):
        if theta <= 0:
            raise ValueError("Clayton theta must be > 0")
        self.theta = theta

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        val = u ** (-self.theta) + v ** (-self.theta) - 1
        val = np.maximum(val, 1e-10)
        return val ** (-1 / self.theta)

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        t = self.theta
        A = u ** (-t) + v ** (-t) - 1
        A = np.maximum(A, 1e-10)

        density = (1 + t) * (u * v) ** (-(t + 1)) * A ** (-(2 + 1 / t))
        return np.maximum(density, 1e-20)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample via conditional method (Marshall-Olkin)."""
        u = np.random.uniform(0, 1, n)
        w = np.random.uniform(0, 1, n)

        # Conditional inverse: v = ((u^{-theta}*(w^{-theta/(1+theta)} - 1) + 1))^{-1/theta}
        t = self.theta
        v = (u ** (-t) * (w ** (-t / (1 + t)) - 1) + 1) ** (-1 / t)
        v = np.clip(v, 0, 1)

        return u, v

    def lower_tail_dependence(self) -> float:
        return float(2 ** (-1 / self.theta))

    def upper_tail_dependence(self) -> float:
        return 0.0

    @classmethod
    def fit(cls, u: np.ndarray, v: np.ndarray) -> "ClaytonCopula":
        """Fit Clayton copula using Kendall's tau inversion."""
        tau, _ = kendalltau(u, v)
        tau = max(tau, 0.01)  # Clayton requires positive dependence
        theta = 2 * tau / (1 - tau)
        theta = max(theta, 0.01)
        return cls(theta=theta)


class FrankCopula(Copula):
    """
    Frank copula.

    C(u, v) = -1/theta * log(1 + (e^{-theta*u} - 1)(e^{-theta*v} - 1) / (e^{-theta} - 1))

    Properties:
    - Symmetric: lambda_L = lambda_U = 0
    - theta in R \\ {0}
    - Captures both positive and negative dependence
    """

    def __init__(self, theta: float = 2.0):
        if abs(theta) < 1e-10:
            raise ValueError("Frank theta must be nonzero")
        self.theta = theta

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        t = self.theta
        num = (np.exp(-t * u) - 1) * (np.exp(-t * v) - 1)
        denom = np.exp(-t) - 1

        arg = 1 + num / denom
        arg = np.maximum(arg, 1e-20)
        return -np.log(arg) / t

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        t = self.theta
        e_tu = np.exp(-t * u)
        e_tv = np.exp(-t * v)
        e_t = np.exp(-t)

        num = -t * (e_t - 1) * np.exp(-t * (u + v))
        denom = ((e_t - 1) + (e_tu - 1) * (e_tv - 1)) ** 2

        density = num / denom
        return np.maximum(np.abs(density), 1e-20)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample via conditional inverse."""
        u = np.random.uniform(0, 1, n)
        w = np.random.uniform(0, 1, n)

        t = self.theta
        # Conditional inverse
        v = (
            -np.log(
                1
                + (np.exp(-t) - 1)
                / (
                    (np.exp(-t * u) - 1) / (w * (np.exp(-t) - 1) - (np.exp(-t * u) - 1))
                    + 1
                )
            )
            / t
        )

        v = np.clip(v, 0, 1)
        return u, v

    def lower_tail_dependence(self) -> float:
        return 0.0

    def upper_tail_dependence(self) -> float:
        return 0.0

    @classmethod
    def fit(cls, u: np.ndarray, v: np.ndarray) -> "FrankCopula":
        """Fit Frank copula via MLE."""

        def neg_ll(theta: float) -> float:
            try:
                cop = cls(theta=theta)
                return -cop.log_likelihood(u, v)
            except (ValueError, FloatingPointError):
                return 1e10

        result = minimize_scalar(neg_ll, bounds=(-30, 30), method="bounded")
        theta = result.x
        if abs(theta) < 0.01:
            theta = 0.01
        return cls(theta=theta)


class GumbelCopula(Copula):
    """
    Gumbel copula.

    C(u, v) = exp(-((-log u)^theta + (-log v)^theta)^{1/theta})

    Properties:
    - Upper tail dependence: lambda_U = 2 - 2^{1/theta}
    - No lower tail dependence: lambda_L = 0
    - theta in [1, inf)
    - Good for modeling joint rallies / upper tail events
    """

    def __init__(self, theta: float = 2.0):
        if theta < 1:
            raise ValueError("Gumbel theta must be >= 1")
        self.theta = theta

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        t = self.theta
        A = (-np.log(u)) ** t + (-np.log(v)) ** t
        return np.exp(-(A ** (1 / t)))

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        t = self.theta
        lu = -np.log(u)
        lv = -np.log(v)

        A = lu**t + lv**t
        A_inv_t = A ** (1 / t)

        C = np.exp(-A_inv_t)

        # Density formula for Gumbel copula
        term1 = C / (u * v)
        term2 = (lu * lv) ** (t - 1)
        term3 = A ** (1 / t - 2)
        term4 = t - 1 + A_inv_t

        density = term1 * term2 * term3 * term4
        return np.maximum(density, 1e-20)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample via stable distribution method (Marshall-Olkin)."""
        # Generate stable(1/theta, 1, cos(pi/(2*theta))^theta, 0; 1) random variable
        # Simplified approach using exponential frailty
        t = self.theta

        # Generate frailty variable from stable distribution
        # Using Chambers-Mallows-Stuck method for positive stable
        W = np.random.exponential(1, n)
        U_angle = np.random.uniform(0, np.pi, n)

        if t == 1:
            V = np.ones(n)
        else:
            # Positive stable with parameter 1/theta
            alpha = 1 / t
            zeta = 1.0

            term1 = np.sin(alpha * U_angle) / (np.sin(U_angle) ** (1 / alpha))
            term2 = (np.sin((1 - alpha) * U_angle) / W) ** ((1 - alpha) / alpha)
            V = term1 * term2

        # Generate uniform RVs and apply inverse conditional
        E1 = np.random.exponential(1, n)
        E2 = np.random.exponential(1, n)

        u = np.exp(-((E1 / V) ** (1 / t)))
        v = np.exp(-((E2 / V) ** (1 / t)))

        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        return u, v

    def lower_tail_dependence(self) -> float:
        return 0.0

    def upper_tail_dependence(self) -> float:
        return float(2 - 2 ** (1 / self.theta))

    @classmethod
    def fit(cls, u: np.ndarray, v: np.ndarray) -> "GumbelCopula":
        """Fit Gumbel copula using Kendall's tau inversion."""
        tau, _ = kendalltau(u, v)
        tau = max(tau, 0.01)  # Gumbel requires positive dependence
        theta = 1 / (1 - tau)
        theta = max(theta, 1.0)
        return cls(theta=theta)


class CopulaAnalyzer:
    """
    High-level copula analysis for financial time series.

    Fits multiple copula families and selects the best fit.
    """

    COPULA_FAMILIES = {
        "gaussian": GaussianCopula,
        "clayton": ClaytonCopula,
        "frank": FrankCopula,
        "gumbel": GumbelCopula,
    }

    @staticmethod
    def to_pseudo_observations(data: np.ndarray) -> np.ndarray:
        """
        Convert data to pseudo-observations (empirical CDF values).

        Uses rank/(n+1) transformation to avoid boundary issues.

        Args:
            data: Raw data array of shape (n,)

        Returns:
            np.ndarray: Pseudo-observations in (0, 1)
        """
        from scipy.stats import rankdata

        n = len(data)
        return rankdata(data) / (n + 1)

    def fit_all(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Dict]:
        """
        Fit all copula families and compare via AIC.

        Args:
            u: First variable pseudo-observations
            v: Second variable pseudo-observations

        Returns:
            dict: family -> {copula, log_likelihood, aic, tail_dependence}
        """
        results = {}

        for name, CopulaClass in self.COPULA_FAMILIES.items():
            try:
                copula = CopulaClass.fit(u, v)
                ll = copula.log_likelihood(u, v)
                aic_val = copula.aic(u, v)

                results[name] = {
                    "copula": copula,
                    "log_likelihood": ll,
                    "aic": aic_val,
                    "lower_tail_dep": copula.lower_tail_dependence(),
                    "upper_tail_dep": copula.upper_tail_dependence(),
                }
            except Exception as e:
                logger.warning("Failed to fit %s copula: %s", name, e)

        return results

    def select_best(self, u: np.ndarray, v: np.ndarray) -> Tuple[str, Copula]:
        """
        Select best copula family by AIC.

        Args:
            u: First variable pseudo-observations
            v: Second variable pseudo-observations

        Returns:
            Tuple of (family_name, fitted_copula)
        """
        results = self.fit_all(u, v)

        if not results:
            raise ValueError("No copula could be fitted")

        best_name = min(results, key=lambda k: results[k]["aic"])
        return best_name, results[best_name]["copula"]

    def tail_dependence_analysis(
        self, u: np.ndarray, v: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze tail dependence structure.

        Args:
            u, v: Pseudo-observations

        Returns:
            dict: Tail dependence estimates from best-fitting copulas
        """
        results = self.fit_all(u, v)

        # Use Clayton for lower tail, Gumbel for upper tail
        lower_tail = 0.0
        upper_tail = 0.0

        if "clayton" in results:
            lower_tail = results["clayton"]["lower_tail_dep"]
        if "gumbel" in results:
            upper_tail = results["gumbel"]["upper_tail_dep"]

        # Overall from best model
        best_name, best_cop = self.select_best(u, v)

        return {
            "best_model": best_name,
            "lower_tail_dependence": lower_tail,
            "upper_tail_dependence": upper_tail,
            "best_model_lower": best_cop.lower_tail_dependence(),
            "best_model_upper": best_cop.upper_tail_dependence(),
            "kendall_tau": float(kendalltau(u, v)[0]),
        }

    def generate_trading_signal(
        self,
        returns_x: np.ndarray,
        returns_y: np.ndarray,
        window: int = 252,
    ) -> Dict:
        """
        Generate trading signal based on copula dependence structure.

        High tail dependence during stress -> reduce paired exposure.
        Low tail dependence -> diversification benefit is real.

        Args:
            returns_x: Returns of asset X
            returns_y: Returns of asset Y
            window: Lookback window for fitting

        Returns:
            dict: Signal, confidence, dependence metrics
        """
        # Use recent window
        rx = returns_x[-window:]
        ry = returns_y[-window:]

        u = self.to_pseudo_observations(rx)
        v = self.to_pseudo_observations(ry)

        tail_info = self.tail_dependence_analysis(u, v)

        # Signal logic:
        # High lower tail dep -> correlated in crashes -> reduce hedge ratio confidence
        # Low tail dep -> good diversification
        crash_risk = tail_info["lower_tail_dependence"]
        diversification = 1 - crash_risk

        return {
            "crash_correlation": crash_risk,
            "diversification_benefit": diversification,
            "best_copula": tail_info["best_model"],
            "kendall_tau": tail_info["kendall_tau"],
            "signal": "reduce_exposure" if crash_risk > 0.5 else "normal",
            "confidence": min(abs(crash_risk - 0.5) * 2, 1.0),
        }
