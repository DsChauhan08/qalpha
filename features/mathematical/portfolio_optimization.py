"""
Advanced Portfolio Optimization

Implements:
1. Mean-Variance Optimization (Markowitz) + Efficient Frontier
2. Black-Litterman Model (Bayesian)
3. Risk Parity
4. Maximum Diversification
5. Hierarchical Risk Parity (Lopez de Prado)

All optimizers return weights and portfolio metrics.
No print statements - uses logging per project standards.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """
    Markowitz mean-variance portfolio optimization.

    min_w  w^T Sigma w - lambda * w^T mu
    s.t.   sum(w_i) = 1, w_i in [lb, ub]
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        allow_short: bool = False,
        max_position: float = 0.3,
    ):
        """
        Args:
            risk_aversion: Risk aversion parameter (lambda)
            allow_short: Allow short positions
            max_position: Maximum absolute position size per asset
        """
        self.lambda_risk = risk_aversion
        self.allow_short = allow_short
        self.max_position = max_position

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[List] = None,
    ) -> Dict:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            constraints: Additional scipy constraints

        Returns:
            dict: weights, expected_return, volatility, sharpe_ratio, success
        """
        n = len(expected_returns)

        def objective(w: np.ndarray) -> float:
            portfolio_return = np.dot(w, expected_returns)
            portfolio_var = np.dot(w, np.dot(cov_matrix, w))
            return float(-portfolio_return + self.lambda_risk * portfolio_var)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if constraints:
            cons.extend(constraints)

        if self.allow_short:
            bounds = [(-self.max_position, self.max_position)] * n
        else:
            bounds = [(0, self.max_position)] * n

        w0 = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=cons
        )

        if not result.success:
            logger.warning("MVO optimization warning: %s", result.message)

        weights = result.x

        portfolio_return = float(np.dot(weights, expected_returns))
        portfolio_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
        portfolio_vol = np.sqrt(max(portfolio_var, 0))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

        return {
            "weights": weights,
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
            "success": result.success,
        }

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Generate efficient frontier.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            n_points: Number of points on frontier

        Returns:
            DataFrame: return, volatility, sharpe for each frontier point
        """
        # Save and restore lambda
        original_lambda = self.lambda_risk

        # Find minimum variance portfolio
        self.lambda_risk = 1e6
        min_var = self.optimize(expected_returns, cov_matrix)

        max_return = float(np.max(expected_returns))

        target_returns = np.linspace(min_var["expected_return"], max_return, n_points)

        frontier = []
        for target in target_returns:
            self.lambda_risk = 1e6  # Minimize risk subject to return constraint
            return_constraint = {
                "type": "eq",
                "fun": lambda w, t=target: np.dot(w, expected_returns) - t,
            }

            result = self.optimize(expected_returns, cov_matrix, [return_constraint])

            frontier.append(
                {
                    "return": result["expected_return"],
                    "volatility": result["volatility"],
                    "sharpe": result["sharpe_ratio"],
                }
            )

        self.lambda_risk = original_lambda

        return pd.DataFrame(frontier)


class BlackLitterman:
    """
    Black-Litterman model for Bayesian portfolio optimization.

    Combines market equilibrium (CAPM implied returns) with investor views.
    """

    def __init__(self, tau: float = 0.05):
        """
        Args:
            tau: Uncertainty scaling parameter (typically 0.01-0.1)
        """
        self.tau = tau

    def optimize(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 2.5,
        views: Optional[Dict] = None,
    ) -> Dict:
        """
        Run Black-Litterman optimization.

        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (delta)
            views: dict with keys:
                'P': View portfolio matrix (K x N)
                'Q': View expected returns (K,)
                'omega': View uncertainty matrix (K x K)

        Returns:
            dict: posterior_return, optimal_weights, etc.
        """
        n = len(market_weights)

        # Implied equilibrium returns (reverse optimization)
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        if views is None:
            posterior_return = pi
            posterior_cov = cov_matrix
        else:
            P = np.asarray(views["P"])
            Q = np.asarray(views["Q"])
            omega = np.asarray(views["omega"])

            inv_tau_sigma = np.linalg.inv(self.tau * cov_matrix)
            inv_omega = np.linalg.inv(omega)

            posterior_cov_inv = inv_tau_sigma + P.T @ inv_omega @ P
            posterior_cov = np.linalg.inv(posterior_cov_inv)

            posterior_return = posterior_cov @ (
                inv_tau_sigma @ pi + P.T @ inv_omega @ Q
            )

        # Optimize with posterior estimates
        mvo = MeanVarianceOptimizer(risk_aversion=risk_aversion)
        opt_result = mvo.optimize(posterior_return, posterior_cov)

        return {
            "posterior_return": posterior_return,
            "posterior_cov": posterior_cov,
            "equilibrium_return": pi,
            "optimal_weights": opt_result["weights"],
            "expected_return": opt_result["expected_return"],
            "volatility": opt_result["volatility"],
            "sharpe_ratio": opt_result["sharpe_ratio"],
        }


class RiskParityOptimizer:
    """
    Risk parity portfolio optimization.

    Allocates such that each asset contributes equally to total portfolio risk.
    """

    def __init__(self, target_risk: Optional[float] = None):
        """
        Args:
            target_risk: Target portfolio volatility (None = fully invested)
        """
        self.target_risk = target_risk

    def optimize(
        self,
        cov_matrix: np.ndarray,
        budget_constraint: bool = True,
    ) -> Dict:
        """
        Optimize risk parity portfolio.

        Args:
            cov_matrix: Covariance matrix
            budget_constraint: Enforce sum of weights = 1

        Returns:
            dict: weights, risk_contributions, portfolio_volatility
        """
        n = cov_matrix.shape[0]

        def objective(w: np.ndarray) -> float:
            portfolio_var = np.dot(w, np.dot(cov_matrix, w))
            mrc = np.dot(cov_matrix, w)
            rc = w * mrc
            target_rc = portfolio_var / n
            return float(np.sum((rc - target_rc) ** 2))

        constraints = []
        if budget_constraint:
            constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})

        if self.target_risk is not None:
            target = self.target_risk
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w: np.sqrt(np.dot(w, np.dot(cov_matrix, w))) - target,
                }
            )

        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        weights = result.x

        portfolio_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
        mrc = np.dot(cov_matrix, weights)
        rc = weights * mrc
        rc_pct = rc / portfolio_var if portfolio_var > 0 else np.zeros(n)

        return {
            "weights": weights,
            "risk_contributions": rc,
            "risk_contribution_pct": rc_pct,
            "portfolio_volatility": np.sqrt(max(portfolio_var, 0)),
            "success": result.success,
        }


class MaximumDiversification:
    """
    Maximum diversification portfolio.

    Maximizes the diversification ratio:
    DR = (sum w_i * sigma_i) / sigma_p
    """

    def optimize(self, cov_matrix: np.ndarray) -> Dict:
        """
        Optimize maximum diversification portfolio.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            dict: weights, diversification_ratio, portfolio_volatility
        """
        n = cov_matrix.shape[0]
        vols = np.sqrt(np.diag(cov_matrix))

        def neg_diversification_ratio(w: np.ndarray) -> float:
            weighted_vols = np.dot(w, vols)
            portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            if portfolio_vol < 1e-12:
                return 0.0
            return float(-weighted_vols / portfolio_vol)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        result = minimize(
            neg_diversification_ratio,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x

        portfolio_vol = np.sqrt(float(np.dot(weights, np.dot(cov_matrix, weights))))
        weighted_vols = float(np.dot(weights, vols))
        diversification_ratio = (
            weighted_vols / portfolio_vol if portfolio_vol > 0 else 1.0
        )

        return {
            "weights": weights,
            "diversification_ratio": diversification_ratio,
            "portfolio_volatility": portfolio_vol,
            "success": result.success,
        }


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) by Marcos Lopez de Prado.

    Steps:
    1. Compute distance matrix from correlation
    2. Hierarchical clustering
    3. Quasi-diagonalization
    4. Recursive bisection for weight allocation
    """

    def __init__(self, linkage_method: str = "single"):
        """
        Args:
            linkage_method: Clustering linkage method
                ('single', 'complete', 'average', 'ward')
        """
        self.linkage_method = linkage_method

    def optimize(
        self,
        cov_matrix: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run HRP optimization.

        Args:
            cov_matrix: Covariance matrix
            returns: Return series (optional, for distance calculation)

        Returns:
            dict: weights, sorted_indices, linkage_matrix
        """
        # Step 1: Distance matrix from correlation
        corr = self._cov_to_corr(cov_matrix)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)  # Ensure diagonal is exactly zero

        # Step 2: Hierarchical clustering
        condensed = squareform(dist)
        linkage_matrix = linkage(condensed, method=self.linkage_method)

        # Step 3: Quasi-diagonalization
        sorted_idx = leaves_list(linkage_matrix)

        # Step 4: Recursive bisection for weights
        weights = self._recursive_bisection(cov_matrix, sorted_idx)

        return {
            "weights": weights,
            "sorted_indices": sorted_idx,
            "linkage_matrix": linkage_matrix,
        }

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation matrix."""
        vols = np.sqrt(np.diag(cov))
        vols = np.where(vols > 0, vols, 1e-10)
        corr = cov / np.outer(vols, vols)
        # Clamp for numerical stability
        return np.clip(corr, -1, 1)

    def _recursive_bisection(
        self, cov: np.ndarray, sorted_idx: np.ndarray
    ) -> np.ndarray:
        """
        Allocate weights using recursive bisection.

        Args:
            cov: Covariance matrix
            sorted_idx: Hierarchically sorted indices

        Returns:
            np.ndarray: Portfolio weights (in original order)
        """
        n = len(sorted_idx)
        weights = np.ones(n)

        # Recursive allocation
        clusters: List[np.ndarray] = [sorted_idx.copy()]

        while clusters:
            new_clusters: List[np.ndarray] = []

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                split = len(cluster) // 2
                left = cluster[:split]
                right = cluster[split:]

                left_var = self._get_cluster_var(cov, left)
                right_var = self._get_cluster_var(cov, right)

                total_var = left_var + right_var
                alpha = 1 - left_var / total_var if total_var > 0 else 0.5

                for idx in left:
                    pos = np.where(sorted_idx == idx)[0][0]
                    weights[pos] *= alpha

                for idx in right:
                    pos = np.where(sorted_idx == idx)[0][0]
                    weights[pos] *= 1 - alpha

                new_clusters.extend([left, right])

            clusters = new_clusters

        # Normalize
        weights = weights / np.sum(weights)

        # Map back to original order
        final_weights = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            final_weights[idx] = weights[i]

        return final_weights

    @staticmethod
    def _get_cluster_var(cov: np.ndarray, cluster: np.ndarray) -> float:
        """Calculate variance of cluster with inverse-variance weights."""
        cluster_cov = cov[np.ix_(cluster, cluster)]
        diag = np.diag(cluster_cov)
        diag = np.where(diag > 0, diag, 1e-10)
        iv_weights = 1 / diag
        iv_weights = iv_weights / np.sum(iv_weights)

        return float(np.dot(iv_weights, np.dot(cluster_cov, iv_weights)))
