"""
Monte Carlo Simulation & Padé Approximation Feature Engine
==========================================================
Generates forward-looking probabilistic features for trading models:

1. Monte Carlo (MC) Simulation:
   - Geometric Brownian Motion (GBM) paths
   - Jump-Diffusion (Merton model) paths
   - Regime-switching volatility paths
   - Extracts: probability of up/down, expected return, VaR, CVaR,
     skewness of simulated distribution, path-dependent features

2. Padé Approximation:
   - Rational function approximation of the moment generating function
   - Fast estimation of return distribution tail probabilities
   - Extracts: tail risk ratios, distribution shape parameters,
     expected shortfall estimates

These features tell the model "what COULD happen" based on current
market conditions — fundamentally different from backward-looking indicators.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)


def _safe_clip(x, lo=None, hi=None):
    """Clip that works on both numpy arrays and pandas Series."""
    return np.clip(x, lo, hi)


# =====================================================================
# Monte Carlo Simulation Engine
# =====================================================================


class MonteCarloEngine:
    """
    Fast Monte Carlo price path simulator with multiple models.

    Designed for batch feature generation: simulates thousands of paths
    per data point, extracts distributional features efficiently.
    """

    def __init__(
        self,
        n_paths: int = 500,
        horizons: List[int] = None,
        vol_lookback: int = 21,
        jump_intensity: float = 0.05,
        jump_mean: float = -0.02,
        jump_std: float = 0.03,
        seed: int = 42,
    ):
        """
        Args:
            n_paths: Number of MC paths to simulate per point
            horizons: Forward horizons in days [1, 5, 10, 21]
            vol_lookback: Lookback for realized volatility estimation
            jump_intensity: Poisson jump rate (jumps per day)
            jump_mean: Mean of log-normal jump size
            jump_std: Std of log-normal jump size
            seed: Random seed for reproducibility
        """
        self.n_paths = n_paths
        self.horizons = horizons or [1, 5, 10, 21]
        self.vol_lookback = vol_lookback
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.rng = np.random.RandomState(seed)

    def _estimate_params(
        self,
        returns: np.ndarray,
        lookback: int = None,
    ) -> Dict[str, float]:
        """Estimate GBM parameters from recent returns."""
        if lookback is None:
            lookback = self.vol_lookback

        recent = returns[-lookback:]
        recent = recent[~np.isnan(recent)]

        if len(recent) < 5:
            return {"mu": 0.0, "sigma": 0.2 / np.sqrt(252), "skew": 0.0, "kurt": 3.0}

        mu = np.mean(recent)
        sigma = np.std(recent, ddof=1)

        if sigma < 1e-8:
            sigma = 0.2 / np.sqrt(252)

        try:
            skew = float(stats.skew(recent))
        except Exception:
            skew = 0.0

        try:
            kurt = float(stats.kurtosis(recent, fisher=False))
        except Exception:
            kurt = 3.0

        return {"mu": float(mu), "sigma": float(sigma), "skew": skew, "kurt": kurt}

    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Simulate GBM paths.

        dS = mu*S*dt + sigma*S*dW

        Returns:
            Array of shape (n_paths, horizon+1) with price paths
        """
        dt = 1.0  # Daily
        Z = self.rng.standard_normal((self.n_paths, horizon))

        # Log-space simulation (exact)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        log_prices = np.zeros((self.n_paths, horizon + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

        return np.exp(log_prices)

    def simulate_jump_diffusion(
        self,
        S0: float,
        mu: float,
        sigma: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Simulate Merton jump-diffusion paths.

        dS/S = (mu - lambda*k)*dt + sigma*dW + J*dN
        where N is Poisson, J is log-normal

        Returns:
            Array of shape (n_paths, horizon+1) with price paths
        """
        dt = 1.0
        lam = self.jump_intensity

        # Expected jump compensation
        k = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1

        Z = self.rng.standard_normal((self.n_paths, horizon))
        N = self.rng.poisson(lam * dt, (self.n_paths, horizon))
        J = self.rng.normal(self.jump_mean, self.jump_std, (self.n_paths, horizon))

        # Log returns with jumps
        drift = (mu - lam * k - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        jumps = N * J  # Jump size * number of jumps

        log_returns = drift + diffusion + jumps
        log_prices = np.zeros((self.n_paths, horizon + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

        return np.exp(log_prices)

    def simulate_regime_switching(
        self,
        S0: float,
        mu: float,
        sigma: float,
        horizon: int,
        vol_ratio_high: float = 2.0,
        regime_persistence: float = 0.95,
    ) -> np.ndarray:
        """
        Simulate 2-regime switching model.

        Low-vol regime: sigma
        High-vol regime: sigma * vol_ratio_high
        Switches with probability (1 - regime_persistence)

        Returns:
            Array of shape (n_paths, horizon+1) with price paths
        """
        dt = 1.0
        Z = self.rng.standard_normal((self.n_paths, horizon))
        U = self.rng.uniform(0, 1, (self.n_paths, horizon))

        # Track regime per path
        in_high_vol = self.rng.uniform(0, 1, self.n_paths) > 0.7  # Start mostly low-vol

        log_prices = np.zeros((self.n_paths, horizon + 1))
        log_prices[:, 0] = np.log(S0)

        for t in range(horizon):
            # Current volatility
            current_sigma = np.where(in_high_vol, sigma * vol_ratio_high, sigma)

            drift = (mu - 0.5 * current_sigma**2) * dt
            diffusion = current_sigma * np.sqrt(dt) * Z[:, t]

            log_prices[:, t + 1] = log_prices[:, t] + drift + diffusion

            # Regime transition
            switch = U[:, t] > regime_persistence
            in_high_vol = np.where(switch, ~in_high_vol, in_high_vol)

        return np.exp(log_prices)

    def extract_path_features(
        self,
        paths: np.ndarray,
        S0: float,
        horizon: int,
    ) -> Dict[str, float]:
        """
        Extract distributional features from simulated paths.

        Args:
            paths: (n_paths, horizon+1) price paths
            S0: Starting price
            horizon: Horizon in days

        Returns:
            Dict of feature name -> value
        """
        # Terminal returns
        terminal_prices = paths[:, -1]
        terminal_returns = (terminal_prices / S0) - 1

        prefix = f"mc_{horizon}d"
        features = {}

        # Central tendency
        features[f"{prefix}_mean_return"] = float(np.mean(terminal_returns))
        features[f"{prefix}_median_return"] = float(np.median(terminal_returns))

        # Probability of direction
        features[f"{prefix}_prob_up"] = float(np.mean(terminal_returns > 0))
        features[f"{prefix}_prob_up_1pct"] = float(np.mean(terminal_returns > 0.01))
        features[f"{prefix}_prob_down_1pct"] = float(np.mean(terminal_returns < -0.01))
        features[f"{prefix}_prob_up_5pct"] = float(np.mean(terminal_returns > 0.05))
        features[f"{prefix}_prob_down_5pct"] = float(np.mean(terminal_returns < -0.05))

        # Risk metrics
        features[f"{prefix}_std"] = float(np.std(terminal_returns))

        # VaR and CVaR (5th percentile)
        var_5 = np.percentile(terminal_returns, 5)
        cvar_5 = (
            np.mean(terminal_returns[terminal_returns <= var_5])
            if np.sum(terminal_returns <= var_5) > 0
            else var_5
        )
        features[f"{prefix}_var_5"] = float(var_5)
        features[f"{prefix}_cvar_5"] = float(cvar_5)

        # Upside potential (95th percentile)
        var_95 = np.percentile(terminal_returns, 95)
        features[f"{prefix}_upside_95"] = float(var_95)

        # Skewness of simulated distribution
        features[f"{prefix}_skew"] = float(stats.skew(terminal_returns))

        # Kurtosis
        features[f"{prefix}_kurt"] = float(
            stats.kurtosis(terminal_returns, fisher=True)
        )

        # Risk/reward ratio
        expected_gain = (
            np.mean(terminal_returns[terminal_returns > 0])
            if np.sum(terminal_returns > 0) > 0
            else 0
        )
        expected_loss = (
            np.mean(terminal_returns[terminal_returns < 0])
            if np.sum(terminal_returns < 0) > 0
            else 0
        )
        features[f"{prefix}_gain_loss_ratio"] = float(
            abs(expected_gain / expected_loss) if abs(expected_loss) > 1e-8 else 1.0
        )

        # Path-dependent features (max drawdown, max run-up along paths)
        cumulative_returns = paths / S0 - 1

        # Average max drawdown across paths
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / (running_max + 1e-8)
        max_drawdowns = np.min(drawdowns, axis=1)
        features[f"{prefix}_avg_max_dd"] = float(np.mean(max_drawdowns))

        # Average max run-up
        running_min = np.minimum.accumulate(paths, axis=1)
        runups = (paths - running_min) / (running_min + 1e-8)
        max_runups = np.max(runups, axis=1)
        features[f"{prefix}_avg_max_runup"] = float(np.mean(max_runups))

        # Probability of touching a barrier
        features[f"{prefix}_prob_touch_up_3pct"] = float(
            np.mean(np.any(cumulative_returns > 0.03, axis=1))
        )
        features[f"{prefix}_prob_touch_down_3pct"] = float(
            np.mean(np.any(cumulative_returns < -0.03, axis=1))
        )

        return features

    def generate_features_single(
        self,
        close_prices: np.ndarray,
        current_idx: int,
    ) -> Dict[str, float]:
        """
        Generate all MC features for a single time step.

        Args:
            close_prices: Full close price array
            current_idx: Current index in the array

        Returns:
            Dict of all MC features
        """
        if current_idx < self.vol_lookback + 10:
            return {}

        S0 = close_prices[current_idx]
        if S0 <= 0 or np.isnan(S0):
            return {}

        # Compute returns
        returns = np.diff(np.log(close_prices[: current_idx + 1]))

        # Estimate parameters
        params = self._estimate_params(returns)
        mu, sigma = params["mu"], params["sigma"]

        all_features = {}

        for horizon in self.horizons:
            # GBM simulation
            gbm_paths = self.simulate_gbm(S0, mu, sigma, horizon)
            gbm_features = self.extract_path_features(gbm_paths, S0, horizon)

            # Jump-diffusion (only for longer horizons to save time)
            if horizon >= 5:
                jd_paths = self.simulate_jump_diffusion(S0, mu, sigma, horizon)
                jd_features = self.extract_path_features(jd_paths, S0, horizon)
                # Rename with jd_ prefix
                for k, v in jd_features.items():
                    all_features[k.replace(f"mc_{horizon}d", f"jd_{horizon}d")] = v

            # Regime-switching (only for 21d)
            if horizon >= 21:
                rs_paths = self.simulate_regime_switching(S0, mu, sigma, horizon)
                rs_features = self.extract_path_features(rs_paths, S0, horizon)
                for k, v in rs_features.items():
                    all_features[k.replace(f"mc_{horizon}d", f"rs_{horizon}d")] = v

            all_features.update(gbm_features)

        # Add the estimated parameters as features too
        all_features["mc_est_mu"] = params["mu"]
        all_features["mc_est_sigma"] = params["sigma"]
        all_features["mc_est_skew"] = params["skew"]
        all_features["mc_est_kurt"] = params["kurt"]

        return all_features


# =====================================================================
# Padé Approximation Engine
# =====================================================================


class PadeApproximant:
    """
    Padé approximation for fast distribution estimation.

    Uses rational function approximation of the moment generating function (MGF)
    to estimate tail probabilities and distribution characteristics faster
    than full MC simulation.

    The Padé approximant P[m/n](x) = sum(a_i * x^i) / sum(b_i * x^i)
    provides better convergence than Taylor series for estimating
    the characteristic function of return distributions.
    """

    def __init__(
        self,
        order: Tuple[int, int] = (3, 3),
        moment_lookback: int = 63,
    ):
        """
        Args:
            order: (m, n) order of numerator/denominator polynomials
            moment_lookback: Days of history for moment estimation
        """
        self.m_order, self.n_order = order
        self.moment_lookback = moment_lookback

    def _estimate_moments(self, returns: np.ndarray) -> Dict[str, float]:
        """Estimate raw and central moments from returns."""
        n = len(returns)
        if n < 10:
            return {
                "mean": 0.0,
                "var": 0.0001,
                "skew": 0.0,
                "kurt": 3.0,
                "m5": 0.0,
                "m6": 0.0,
            }

        mean = np.mean(returns)
        var = np.var(returns, ddof=1)

        if var < 1e-12:
            var = 1e-8

        centered = returns - mean
        std = np.sqrt(var)

        standardized = centered / std

        skew = np.mean(standardized**3)
        kurt = np.mean(standardized**4)
        m5 = np.mean(standardized**5)
        m6 = np.mean(standardized**6)

        return {
            "mean": float(mean),
            "var": float(var),
            "skew": float(skew),
            "kurt": float(kurt),
            "m5": float(np.clip(m5, -50, 50)),
            "m6": float(np.clip(m6, 0, 200)),
        }

    def _build_pade_coefficients(
        self, moments: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Padé approximant coefficients from moments.

        Uses the moment sequence to construct [m/n] Padé approximant
        of the moment generating function: M(t) = E[exp(tX)]

        Taylor coefficients of MGF: c_k = E[X^k] / k!
        """
        mean, var, skew, kurt = (
            moments["mean"],
            moments["var"],
            moments["skew"],
            moments["kurt"],
        )
        std = np.sqrt(var)

        # Taylor coefficients of the MGF around t=0
        # M(t) = 1 + mean*t + (mean^2 + var)/2 * t^2 + ...
        raw_moments = [
            1.0,  # M(0) = 1
            mean,  # M'(0) = E[X]
            mean**2 + var,  # E[X^2]
            mean**3 + 3 * mean * var + skew * std**3,  # E[X^3]
            mean**4
            + 6 * mean**2 * var
            + 4 * mean * skew * std**3
            + (kurt - 3) * var**2
            + 3 * var**2,  # E[X^4]
        ]

        # Pad with estimates for higher moments
        while len(raw_moments) < self.m_order + self.n_order + 1:
            k = len(raw_moments)
            # Gaussian approximation for higher raw moments
            if k % 2 == 0:
                raw_moments.append(var ** (k // 2) * np.prod(np.arange(1, k, 2)))
            else:
                raw_moments.append(0.0)

        # Taylor coefficients c_k = raw_moment_k / k!
        c = np.array(
            [raw_moments[k] / math.factorial(k) for k in range(len(raw_moments))]
        )

        # Build Padé table using the Padé algorithm
        # For [m/n] approximant, solve the linear system
        m, n = self.m_order, self.n_order

        try:
            if n == 0:
                return c[: m + 1], np.array([1.0])

            # Build the system for denominator coefficients
            # Using the Padé equations: sum_j b_j * c_{i-j} = 0 for i = m+1, ..., m+n
            A = np.zeros((n, n))
            rhs = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    idx = m + 1 + i - j - 1
                    if 0 <= idx < len(c):
                        A[i, j] = c[idx]
                rhs[i] = -c[m + 1 + i] if m + 1 + i < len(c) else 0.0

            # Solve for b coefficients (b_0 = 1)
            try:
                b_coeffs = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                b_coeffs = np.linalg.lstsq(A, rhs, rcond=None)[0]

            b = np.concatenate([[1.0], b_coeffs])

            # Compute numerator coefficients
            a = np.zeros(m + 1)
            for i in range(m + 1):
                a[i] = sum(
                    b[j] * c[i - j]
                    for j in range(min(i + 1, len(b)))
                    if 0 <= i - j < len(c)
                )

            return a, b

        except Exception:
            # Fallback: simple [1/1] Padé
            a = np.array([1.0, c[1] if len(c) > 1 else 0])
            b = np.array([1.0, -c[1] if len(c) > 1 else 0])
            return a, b

    def _evaluate_pade(
        self,
        a: np.ndarray,
        b: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """Evaluate P[m/n](t) = sum(a_i * t^i) / sum(b_i * t^i)."""
        num = np.zeros_like(t)
        den = np.zeros_like(t)

        for i, ai in enumerate(a):
            num += ai * t**i
        for i, bi in enumerate(b):
            den += bi * t**i

        # Avoid division by zero
        den = np.where(np.abs(den) < 1e-12, 1e-12 * np.sign(den + 1e-15), den)

        return num / den

    def estimate_tail_probabilities(
        self,
        returns: np.ndarray,
        thresholds: List[float] = None,
    ) -> Dict[str, float]:
        """
        Estimate tail probabilities using Padé-approximated MGF.

        Uses the Chernoff bound with Padé-enhanced MGF:
        P(X > a) <= inf_t>0 exp(-t*a) * M(t)

        This gives tighter bounds than simple Gaussian estimates.
        """
        if thresholds is None:
            thresholds = [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05]

        moments = self._estimate_moments(returns)
        a_coeffs, b_coeffs = self._build_pade_coefficients(moments)

        features = {}
        std = np.sqrt(moments["var"])
        mean = moments["mean"]

        for thresh in thresholds:
            # Chernoff bound optimization
            # P(X > a) <= min_t>0 exp(-t*a) * M_pade(t)
            t_values = np.linspace(0.01, 5.0 / (std + 1e-8), 50)

            try:
                mgf_values = self._evaluate_pade(a_coeffs, b_coeffs, t_values)

                if thresh > mean:
                    # Upper tail: P(X > thresh)
                    bounds = np.exp(-t_values * thresh) * np.abs(mgf_values)
                    valid = bounds[np.isfinite(bounds) & (bounds > 0)]
                    if len(valid) > 0:
                        prob = float(np.clip(np.min(valid), 0, 1))
                    else:
                        prob = float(1 - stats.norm.cdf(thresh, mean, std))
                else:
                    # Lower tail: P(X < thresh) = P(-X > -thresh)
                    bounds = np.exp(t_values * thresh) * np.abs(
                        self._evaluate_pade(a_coeffs, b_coeffs, -t_values)
                    )
                    valid = bounds[np.isfinite(bounds) & (bounds > 0)]
                    if len(valid) > 0:
                        prob = float(np.clip(np.min(valid), 0, 1))
                    else:
                        prob = float(stats.norm.cdf(thresh, mean, std))
            except Exception:
                # Fallback to normal approximation
                prob = (
                    float(stats.norm.cdf(thresh, mean, std))
                    if thresh < mean
                    else float(1 - stats.norm.cdf(thresh, mean, std))
                )

            # Label
            direction = "up" if thresh > 0 else "down"
            pct = abs(int(thresh * 100))
            features[f"pade_prob_{direction}_{pct}pct"] = prob

        # Distribution shape features from Padé
        features["pade_mean"] = moments["mean"]
        features["pade_var"] = moments["var"]
        features["pade_skew"] = moments["skew"]
        features["pade_kurt"] = moments["kurt"]
        features["pade_m5"] = moments["m5"]
        features["pade_m6"] = moments["m6"]

        # Tail ratio: right tail / left tail probability
        right_tail = features.get("pade_prob_up_2pct", 0.5)
        left_tail = features.get("pade_prob_down_2pct", 0.5)
        features["pade_tail_ratio"] = float(right_tail / (left_tail + 1e-8))

        # Expected shortfall estimate (Padé-enhanced)
        # ES = mean - std * phi(z_alpha) / (1 - Phi(z_alpha))
        alpha = 0.05
        z_alpha = stats.norm.ppf(alpha)
        phi_z = stats.norm.pdf(z_alpha)

        # Cornish-Fisher expansion for non-normal adjustment
        cf_z = z_alpha + (z_alpha**2 - 1) * moments["skew"] / 6
        cf_z += (z_alpha**3 - 3 * z_alpha) * (moments["kurt"] - 3) / 24
        cf_z -= (2 * z_alpha**3 - 5 * z_alpha) * moments["skew"] ** 2 / 36

        features["pade_es_5pct"] = float(mean + std * cf_z)

        # Padé convergence quality (ratio of coefficients)
        if len(b_coeffs) > 1:
            features["pade_convergence"] = float(
                np.abs(a_coeffs[-1]) / (np.abs(b_coeffs[-1]) + 1e-8)
            )
        else:
            features["pade_convergence"] = 1.0

        return features

    def generate_features_single(
        self,
        returns: np.ndarray,
        current_idx: int,
    ) -> Dict[str, float]:
        """Generate all Padé features for a single time step."""
        if current_idx < self.moment_lookback + 5:
            return {}

        recent = returns[max(0, current_idx - self.moment_lookback) : current_idx]
        recent = recent[~np.isnan(recent)]

        if len(recent) < 20:
            return {}

        return self.estimate_tail_probabilities(recent)


# =====================================================================
# Combined Feature Generator
# =====================================================================


class MCPadeFeatureGenerator:
    """
    Combined Monte Carlo + Padé feature generator.

    Generates ~80+ probabilistic features per data point:
    - MC features at multiple horizons (1d, 5d, 10d, 21d)
    - Jump-diffusion features at 5d, 10d, 21d
    - Regime-switching features at 21d
    - Padé tail probability estimates
    - Distribution shape parameters
    """

    def __init__(
        self,
        n_paths: int = 500,
        horizons: List[int] = None,
        mc_vol_lookback: int = 21,
        pade_lookback: int = 63,
        pade_order: Tuple[int, int] = (3, 3),
        seed: int = 42,
    ):
        self.mc = MonteCarloEngine(
            n_paths=n_paths,
            horizons=horizons or [1, 5, 10, 21],
            vol_lookback=mc_vol_lookback,
            seed=seed,
        )
        self.pade = PadeApproximant(
            order=pade_order,
            moment_lookback=pade_lookback,
        )

    def generate_features(
        self,
        df: pd.DataFrame,
        step: int = 1,
        progress_every: int = 500,
    ) -> pd.DataFrame:
        """
        Generate MC + Padé features for entire DataFrame.
        Uses row-by-row MC simulation (slow but exact).

        For large datasets, use generate_features_fast() instead.

        Args:
            df: DataFrame with 'close' column
            step: Compute every N-th row (1 = all, for speed use 1)
            progress_every: Log progress every N rows

        Returns:
            DataFrame with MC/Padé feature columns added
        """
        close = df["close"].values
        returns = np.diff(np.log(close), prepend=np.log(close[0]))

        n = len(df)
        all_features = []

        min_lookback = max(self.mc.vol_lookback, self.pade.moment_lookback) + 10

        for i in range(n):
            if i < min_lookback:
                all_features.append({})
                continue

            if step > 1 and i % step != 0:
                all_features.append({})
                continue

            # MC features
            mc_feats = self.mc.generate_features_single(close, i)

            # Padé features
            pade_feats = self.pade.generate_features_single(returns, i)

            combined = {**mc_feats, **pade_feats}
            all_features.append(combined)

            if progress_every and (i + 1) % progress_every == 0:
                logger.debug(f"  MC/Padé features: {i + 1}/{n}")

        # Convert to DataFrame
        feat_df = pd.DataFrame(all_features, index=df.index)

        # Fill NaN with 0 for missing early rows
        feat_df = feat_df.fillna(0.0)

        # Replace inf
        feat_df = feat_df.replace([np.inf, -np.inf], 0.0)

        # Merge with original
        result = pd.concat([df, feat_df], axis=1)

        return result

    def generate_features_fast(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Vectorized analytical MC + Padé feature computation.

        Instead of running MC simulation per row, uses the fact that GBM
        terminal returns are exactly log-normal with known parameters:
          log(S_T/S_0) ~ N((mu - sigma^2/2)*T, sigma^2*T)

        All features (probabilities, VaR, CVaR, etc.) can be computed
        analytically from rolling mu and sigma. ~1000x faster than
        row-by-row simulation.

        Jump-diffusion and regime-switching features are approximated
        using analytical corrections to the GBM base.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with MC/Padé feature columns added
        """
        close = df["close"]
        log_returns = np.log(close / close.shift(1))

        vol_lb = self.mc.vol_lookback
        pade_lb = self.pade.moment_lookback
        min_lb = max(vol_lb, pade_lb) + 10

        # Rolling parameters
        mu = log_returns.rolling(vol_lb).mean()
        sigma = log_returns.rolling(vol_lb).std()
        sigma = sigma.clip(lower=1e-8)

        # Rolling higher moments for Padé (use longer lookback)
        roll_skew = log_returns.rolling(pade_lb).skew()
        roll_kurt = log_returns.rolling(pade_lb).apply(
            lambda x: stats.kurtosis(x, fisher=False) if len(x) > 10 else 3.0,
            raw=True,
        )

        # Suppress fragmentation warnings — building 165 columns one-by-one
        # is intentional; refactoring to pd.concat would add complexity for
        # marginal speed gain on already-fast vectorized computation.
        import warnings as _w

        _w.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        feat_df = pd.DataFrame(index=df.index)

        # ── GBM analytical features per horizon ──
        for h in self.mc.horizons:
            prefix = f"mc_{h}d"

            # Log-normal terminal distribution:
            # log(S_T/S_0) ~ N(drift*T, var*T) where drift = mu - sigma^2/2
            drift = (mu - 0.5 * sigma**2) * h
            diffusion = sigma * np.sqrt(h)

            # Mean return: E[S_T/S_0 - 1] = exp(mu*T) - 1 (exact GBM)
            feat_df[f"{prefix}_mean_return"] = np.exp(mu * h) - 1

            # Median return: exp(drift*T) - 1
            feat_df[f"{prefix}_median_return"] = np.exp(drift) - 1

            # P(up) = P(log_return > 0) = Phi(drift/diffusion)
            z_up = drift / (diffusion + 1e-10)
            feat_df[f"{prefix}_prob_up"] = stats.norm.cdf(z_up)

            # P(return > x) for various thresholds
            for pct, pct_str in [(0.01, "1pct"), (0.05, "5pct")]:
                log_thresh = np.log(1 + pct)
                z = (drift - log_thresh) / (diffusion + 1e-10)
                feat_df[f"{prefix}_prob_up_{pct_str}"] = stats.norm.cdf(z)

                log_thresh_neg = np.log(1 - pct)
                z_neg = (drift - log_thresh_neg) / (diffusion + 1e-10)
                feat_df[f"{prefix}_prob_down_{pct_str}"] = 1 - stats.norm.cdf(z_neg)

            # Std of terminal returns (via log-normal variance)
            feat_df[f"{prefix}_std"] = _safe_clip(
                np.sqrt(np.exp(2 * mu * h + sigma**2 * h) * (np.exp(sigma**2 * h) - 1)),
                0,
                10,
            )

            # VaR 5% (5th percentile of return)
            z_05 = stats.norm.ppf(0.05)
            feat_df[f"{prefix}_var_5"] = np.exp(drift + diffusion * z_05) - 1

            # CVaR 5% (expected shortfall below VaR)
            # For log-normal: ES_alpha = E[X|X<VaR] = exp(mu_T + sig_T^2/2) * Phi(z_05 - sig_T) / alpha
            feat_df[f"{prefix}_cvar_5"] = (
                np.exp(drift + 0.5 * diffusion**2)
                * stats.norm.cdf(z_05 - diffusion)
                / 0.05
            ) - 1

            # Upside 95th percentile
            z_95 = stats.norm.ppf(0.95)
            feat_df[f"{prefix}_upside_95"] = np.exp(drift + diffusion * z_95) - 1

            # Skew of log-normal: (exp(sig^2*T) + 2) * sqrt(exp(sig^2*T) - 1)
            exp_v = np.exp(sigma**2 * h)
            feat_df[f"{prefix}_skew"] = _safe_clip(
                (exp_v + 2) * np.sqrt(exp_v - 1), -50, 50
            )

            # Excess kurtosis of log-normal
            feat_df[f"{prefix}_kurt"] = _safe_clip(
                exp_v**4 + 2 * exp_v**3 + 3 * exp_v**2 - 6, -100, 100
            )

            # Gain/loss ratio
            up_prob = _safe_clip(feat_df[f"{prefix}_prob_up"], 0.01, 0.99)
            exp_gain = (
                np.exp(drift + 0.5 * diffusion**2)
                * stats.norm.cdf(z_up + diffusion)
                / up_prob
                - 1
            )
            exp_loss = (
                np.exp(drift + 0.5 * diffusion**2)
                * stats.norm.cdf(-z_up - diffusion)
                / (1 - up_prob)
            ) - 1
            feat_df[f"{prefix}_gain_loss_ratio"] = _safe_clip(
                np.abs(exp_gain) / (np.abs(exp_loss) + 1e-8), 0, 20
            )

            # Avg max drawdown approximation: -sigma * sqrt(T) * E[max of BM]
            feat_df[f"{prefix}_avg_max_dd"] = _safe_clip(
                -(sigma * np.sqrt(h) * 0.7979), -1, 0
            )

            # Avg max runup (symmetric)
            feat_df[f"{prefix}_avg_max_runup"] = _safe_clip(
                sigma * np.sqrt(h) * 0.7979 + _safe_clip(drift, 0, None), 0, 5
            )

            # Barrier touch probability (via reflection principle)
            for bpct, bstr in [(0.03, "3pct")]:
                b = np.log(1 + bpct)
                z_bar = (b - drift) / (diffusion + 1e-10)
                feat_df[f"{prefix}_prob_touch_up_{bstr}"] = _safe_clip(
                    2 * stats.norm.cdf(-_safe_clip(z_bar, -10, None)), 0, 1
                )

                b_neg = np.log(1 - bpct)
                z_bar_neg = (b_neg - drift) / (diffusion + 1e-10)
                feat_df[f"{prefix}_prob_touch_down_{bstr}"] = _safe_clip(
                    2 * stats.norm.cdf(_safe_clip(z_bar_neg, None, 10)), 0, 1
                )

            # ── Jump-diffusion features (horizons >= 5) ──
            if h >= 5:
                jd_prefix = f"jd_{h}d"
                lam = self.mc.jump_intensity
                jm = self.mc.jump_mean
                js = self.mc.jump_std

                # JD mean: adds expected jumps
                k_jump = np.exp(jm + 0.5 * js**2) - 1
                jd_drift = (mu - lam * k_jump - 0.5 * sigma**2) * h
                # Total variance: diffusion + jump contribution
                jd_var = sigma**2 * h + lam * h * (js**2 + jm**2)
                jd_diff = _safe_clip(np.sqrt(jd_var), 1e-8, None)

                feat_df[f"{jd_prefix}_mean_return"] = (
                    np.exp(jd_drift + 0.5 * jd_var) - 1
                )
                feat_df[f"{jd_prefix}_median_return"] = np.exp(jd_drift) - 1

                z_jd_up = jd_drift / (jd_diff + 1e-10)
                feat_df[f"{jd_prefix}_prob_up"] = stats.norm.cdf(z_jd_up)

                for pct, pct_str in [(0.01, "1pct"), (0.05, "5pct")]:
                    lt = np.log(1 + pct)
                    z = (jd_drift - lt) / (jd_diff + 1e-10)
                    feat_df[f"{jd_prefix}_prob_up_{pct_str}"] = stats.norm.cdf(z)
                    lt_neg = np.log(1 - pct)
                    z_neg = (jd_drift - lt_neg) / (jd_diff + 1e-10)
                    feat_df[f"{jd_prefix}_prob_down_{pct_str}"] = 1 - stats.norm.cdf(
                        z_neg
                    )

                feat_df[f"{jd_prefix}_std"] = _safe_clip(
                    np.sqrt(np.exp(2 * jd_drift + jd_var) * (np.exp(jd_var) - 1)),
                    None,
                    10,
                )
                feat_df[f"{jd_prefix}_var_5"] = np.exp(jd_drift + jd_diff * z_05) - 1
                feat_df[f"{jd_prefix}_cvar_5"] = (
                    np.exp(jd_drift + 0.5 * jd_var)
                    * stats.norm.cdf(z_05 - jd_diff)
                    / 0.05
                ) - 1
                feat_df[f"{jd_prefix}_upside_95"] = (
                    np.exp(jd_drift + jd_diff * z_95) - 1
                )

                # JD has higher kurtosis due to jumps
                jd_skew_extra = lam * h * jm * js**2 / (jd_var**1.5 + 1e-10)
                feat_df[f"{jd_prefix}_skew"] = _safe_clip(
                    feat_df[f"{prefix}_skew"] + jd_skew_extra, -50, 50
                )
                jd_kurt_extra = 3 * lam * h * js**4 / (jd_var**2 + 1e-10)
                feat_df[f"{jd_prefix}_kurt"] = _safe_clip(
                    feat_df[f"{prefix}_kurt"] + jd_kurt_extra, -100, 100
                )

                feat_df[f"{jd_prefix}_gain_loss_ratio"] = feat_df[
                    f"{prefix}_gain_loss_ratio"
                ]
                feat_df[f"{jd_prefix}_avg_max_dd"] = feat_df[f"{prefix}_avg_max_dd"] * (
                    jd_diff / (diffusion + 1e-10)
                )
                feat_df[f"{jd_prefix}_avg_max_runup"] = feat_df[
                    f"{prefix}_avg_max_runup"
                ] * (jd_diff / (diffusion + 1e-10))

                for bpct, bstr in [(0.03, "3pct")]:
                    b = np.log(1 + bpct)
                    z_bar = (b - jd_drift) / (jd_diff + 1e-10)
                    feat_df[f"{jd_prefix}_prob_touch_up_{bstr}"] = _safe_clip(
                        2 * stats.norm.cdf(-_safe_clip(z_bar, -10, None)), 0, 1
                    )
                    b_neg = np.log(1 - bpct)
                    z_bar_neg = (b_neg - jd_drift) / (jd_diff + 1e-10)
                    feat_df[f"{jd_prefix}_prob_touch_down_{bstr}"] = _safe_clip(
                        2 * stats.norm.cdf(_safe_clip(z_bar_neg, None, 10)), 0, 1
                    )

            # ── Regime-switching features (horizons >= 21) ──
            if h >= 21:
                rs_prefix = f"rs_{h}d"
                vol_ratio = 2.0
                p_high = 0.3  # stationary probability of high-vol regime

                # Mixture: weighted average of low-vol and high-vol GBM
                rs_sigma_eff = np.sqrt(
                    (1 - p_high) * sigma**2 + p_high * (sigma * vol_ratio) ** 2
                )
                rs_drift = (mu - 0.5 * rs_sigma_eff**2) * h
                rs_diff = rs_sigma_eff * np.sqrt(h)

                feat_df[f"{rs_prefix}_mean_return"] = np.exp(mu * h) - 1
                feat_df[f"{rs_prefix}_median_return"] = np.exp(rs_drift) - 1

                z_rs_up = rs_drift / (rs_diff + 1e-10)
                feat_df[f"{rs_prefix}_prob_up"] = stats.norm.cdf(z_rs_up)

                for pct, pct_str in [(0.01, "1pct"), (0.05, "5pct")]:
                    lt = np.log(1 + pct)
                    z = (rs_drift - lt) / (rs_diff + 1e-10)
                    feat_df[f"{rs_prefix}_prob_up_{pct_str}"] = stats.norm.cdf(z)
                    lt_neg = np.log(1 - pct)
                    z_neg = (rs_drift - lt_neg) / (rs_diff + 1e-10)
                    feat_df[f"{rs_prefix}_prob_down_{pct_str}"] = 1 - stats.norm.cdf(
                        z_neg
                    )

                feat_df[f"{rs_prefix}_std"] = _safe_clip(
                    np.sqrt(
                        np.exp(2 * mu * h + rs_sigma_eff**2 * h)
                        * (np.exp(rs_sigma_eff**2 * h) - 1)
                    ),
                    None,
                    10,
                )
                feat_df[f"{rs_prefix}_var_5"] = np.exp(rs_drift + rs_diff * z_05) - 1
                feat_df[f"{rs_prefix}_cvar_5"] = (
                    np.exp(rs_drift + 0.5 * rs_diff**2)
                    * stats.norm.cdf(z_05 - rs_diff)
                    / 0.05
                ) - 1
                feat_df[f"{rs_prefix}_upside_95"] = (
                    np.exp(rs_drift + rs_diff * z_95) - 1
                )

                exp_v_rs = np.exp(rs_sigma_eff**2 * h)
                feat_df[f"{rs_prefix}_skew"] = _safe_clip(
                    (exp_v_rs + 2) * np.sqrt(exp_v_rs - 1), -50, 50
                )
                feat_df[f"{rs_prefix}_kurt"] = _safe_clip(
                    exp_v_rs**4 + 2 * exp_v_rs**3 + 3 * exp_v_rs**2 - 6, -100, 100
                )

                feat_df[f"{rs_prefix}_gain_loss_ratio"] = feat_df[
                    f"{prefix}_gain_loss_ratio"
                ]
                feat_df[f"{rs_prefix}_avg_max_dd"] = _safe_clip(
                    -(rs_sigma_eff * np.sqrt(h) * 0.7979), -1, 0
                )
                feat_df[f"{rs_prefix}_avg_max_runup"] = _safe_clip(
                    rs_sigma_eff * np.sqrt(h) * 0.7979 + _safe_clip(rs_drift, 0, None),
                    0,
                    5,
                )

                for bpct, bstr in [(0.03, "3pct")]:
                    b = np.log(1 + bpct)
                    z_bar = (b - rs_drift) / (rs_diff + 1e-10)
                    feat_df[f"{rs_prefix}_prob_touch_up_{bstr}"] = _safe_clip(
                        2 * stats.norm.cdf(-_safe_clip(z_bar, -10, None)), 0, 1
                    )
                    b_neg = np.log(1 - bpct)
                    z_bar_neg = (b_neg - rs_drift) / (rs_diff + 1e-10)
                    feat_df[f"{rs_prefix}_prob_touch_down_{bstr}"] = _safe_clip(
                        2 * stats.norm.cdf(_safe_clip(z_bar_neg, None, 10)), 0, 1
                    )

        # ── MC estimated parameters ──
        feat_df["mc_est_mu"] = mu
        feat_df["mc_est_sigma"] = sigma
        feat_df["mc_est_skew"] = roll_skew
        feat_df["mc_est_kurt"] = roll_kurt

        # ── Padé features (vectorized) ──
        roll_mean = log_returns.rolling(pade_lb).mean()
        roll_var = log_returns.rolling(pade_lb).var(ddof=1)
        roll_std = np.sqrt(_safe_clip(roll_var, 1e-12, None))

        # Padé tail probabilities via Cornish-Fisher expansion
        # Much faster than full Padé coefficient computation
        thresholds = [
            (-0.05, "down_5pct"),
            (-0.03, "down_3pct"),
            (-0.02, "down_2pct"),
            (-0.01, "down_1pct"),
            (0.01, "up_1pct"),
            (0.02, "up_2pct"),
            (0.03, "up_3pct"),
            (0.05, "up_5pct"),
        ]

        s = roll_skew.fillna(0)
        k = roll_kurt.fillna(3) - 3  # excess kurtosis

        for thresh, label in thresholds:
            # Cornish-Fisher adjusted z-score
            z_normal = (thresh - roll_mean) / (roll_std + 1e-10)
            z_cf = z_normal - (z_normal**2 - 1) * s / 6
            z_cf = z_cf - (z_normal**3 - 3 * z_normal) * k / 24
            z_cf = z_cf + (2 * z_normal**3 - 5 * z_normal) * s**2 / 36

            if thresh < 0:
                feat_df[f"pade_prob_{label}"] = stats.norm.cdf(z_cf)
            else:
                feat_df[f"pade_prob_{label}"] = 1 - stats.norm.cdf(z_cf)

        feat_df["pade_mean"] = roll_mean
        feat_df["pade_var"] = roll_var
        feat_df["pade_skew"] = roll_skew
        feat_df["pade_kurt"] = roll_kurt
        feat_df["pade_m5"] = (
            log_returns.rolling(pade_lb)
            .apply(
                lambda x: stats.moment(x, moment=5) / (np.std(x, ddof=1) ** 5 + 1e-15)
                if len(x) > 10
                else 0.0,
                raw=True,
            )
            .clip(-50, 50)
        )
        feat_df["pade_m6"] = (
            log_returns.rolling(pade_lb)
            .apply(
                lambda x: stats.moment(x, moment=6) / (np.std(x, ddof=1) ** 6 + 1e-15)
                if len(x) > 10
                else 0.0,
                raw=True,
            )
            .clip(0, 200)
        )

        # Tail ratio
        right_tail = feat_df.get("pade_prob_up_2pct", pd.Series(0.5, index=df.index))
        left_tail = feat_df.get("pade_prob_down_2pct", pd.Series(0.5, index=df.index))
        feat_df["pade_tail_ratio"] = right_tail / (left_tail + 1e-8)

        # Expected shortfall (Cornish-Fisher)
        z_alpha = stats.norm.ppf(0.05)
        cf_z = z_alpha + (z_alpha**2 - 1) * s / 6
        cf_z = cf_z + (z_alpha**3 - 3 * z_alpha) * k / 24
        cf_z = cf_z - (2 * z_alpha**3 - 5 * z_alpha) * s**2 / 36
        feat_df["pade_es_5pct"] = roll_mean + roll_std * cf_z

        # Padé convergence quality (approximated via moment ratio)
        feat_df["pade_convergence"] = (
            roll_var / (roll_var.rolling(21).mean() + 1e-10)
        ).clip(0, 10)

        # Zero out warm-up period
        feat_df.iloc[:min_lb] = 0.0

        # Clean
        feat_df = feat_df.fillna(0.0)
        feat_df = feat_df.replace([np.inf, -np.inf], 0.0)

        # Merge
        result = pd.concat([df, feat_df], axis=1)
        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all MC/Padé feature names."""
        # Generate a dummy to get column names
        names = []

        # MC features per horizon
        for h in self.mc.horizons:
            prefix = f"mc_{h}d"
            names.extend(
                [
                    f"{prefix}_mean_return",
                    f"{prefix}_median_return",
                    f"{prefix}_prob_up",
                    f"{prefix}_prob_up_1pct",
                    f"{prefix}_prob_down_1pct",
                    f"{prefix}_prob_up_5pct",
                    f"{prefix}_prob_down_5pct",
                    f"{prefix}_std",
                    f"{prefix}_var_5",
                    f"{prefix}_cvar_5",
                    f"{prefix}_upside_95",
                    f"{prefix}_skew",
                    f"{prefix}_kurt",
                    f"{prefix}_gain_loss_ratio",
                    f"{prefix}_avg_max_dd",
                    f"{prefix}_avg_max_runup",
                    f"{prefix}_prob_touch_up_3pct",
                    f"{prefix}_prob_touch_down_3pct",
                ]
            )

        # JD features (5d, 10d, 21d)
        for h in [h for h in self.mc.horizons if h >= 5]:
            prefix = f"jd_{h}d"
            names.extend(
                [
                    f"{prefix}_mean_return",
                    f"{prefix}_median_return",
                    f"{prefix}_prob_up",
                    f"{prefix}_prob_up_1pct",
                    f"{prefix}_prob_down_1pct",
                    f"{prefix}_prob_up_5pct",
                    f"{prefix}_prob_down_5pct",
                    f"{prefix}_std",
                    f"{prefix}_var_5",
                    f"{prefix}_cvar_5",
                    f"{prefix}_upside_95",
                    f"{prefix}_skew",
                    f"{prefix}_kurt",
                    f"{prefix}_gain_loss_ratio",
                    f"{prefix}_avg_max_dd",
                    f"{prefix}_avg_max_runup",
                    f"{prefix}_prob_touch_up_3pct",
                    f"{prefix}_prob_touch_down_3pct",
                ]
            )

        # RS features (21d only)
        for h in [h for h in self.mc.horizons if h >= 21]:
            prefix = f"rs_{h}d"
            names.extend(
                [
                    f"{prefix}_mean_return",
                    f"{prefix}_median_return",
                    f"{prefix}_prob_up",
                    f"{prefix}_prob_up_1pct",
                    f"{prefix}_prob_down_1pct",
                    f"{prefix}_prob_up_5pct",
                    f"{prefix}_prob_down_5pct",
                    f"{prefix}_std",
                    f"{prefix}_var_5",
                    f"{prefix}_cvar_5",
                    f"{prefix}_upside_95",
                    f"{prefix}_skew",
                    f"{prefix}_kurt",
                    f"{prefix}_gain_loss_ratio",
                    f"{prefix}_avg_max_dd",
                    f"{prefix}_avg_max_runup",
                    f"{prefix}_prob_touch_up_3pct",
                    f"{prefix}_prob_touch_down_3pct",
                ]
            )

        # MC estimated params
        names.extend(["mc_est_mu", "mc_est_sigma", "mc_est_skew", "mc_est_kurt"])

        # Padé features
        names.extend(
            [
                "pade_prob_down_5pct",
                "pade_prob_down_3pct",
                "pade_prob_down_2pct",
                "pade_prob_down_1pct",
                "pade_prob_up_1pct",
                "pade_prob_up_2pct",
                "pade_prob_up_3pct",
                "pade_prob_up_5pct",
                "pade_mean",
                "pade_var",
                "pade_skew",
                "pade_kurt",
                "pade_m5",
                "pade_m6",
                "pade_tail_ratio",
                "pade_es_5pct",
                "pade_convergence",
            ]
        )

        return names


# =====================================================================
# Convenience function
# =====================================================================


def compute_mc_pade_features(
    df: pd.DataFrame,
    n_paths: int = 500,
    horizons: List[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convenience function to compute MC + Padé features for a DataFrame.

    Args:
        df: DataFrame with 'close' column
        n_paths: Number of MC simulation paths
        horizons: Forward horizons in days
        seed: Random seed

    Returns:
        DataFrame with features added
    """
    gen = MCPadeFeatureGenerator(
        n_paths=n_paths,
        horizons=horizons or [1, 5, 10, 21],
        seed=seed,
    )
    return gen.generate_features(df)


if __name__ == "__main__":
    """Quick test."""
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "volume": np.random.randint(int(1e6), int(1e7), n),
        },
        index=dates,
    )

    gen = MCPadeFeatureGenerator(n_paths=200, horizons=[1, 5, 21])
    result = gen.generate_features(df)

    mc_cols = [
        c for c in result.columns if c.startswith(("mc_", "jd_", "rs_", "pade_"))
    ]
    print(f"Generated {len(mc_cols)} MC/Padé features")
    print(f"Sample features (last row):")
    for col in mc_cols[:20]:
        print(f"  {col:45s} = {result[col].iloc[-1]:.6f}")

    print(f"\nFeature names ({len(mc_cols)} total):")
    for col in mc_cols:
        print(f"  {col}")
