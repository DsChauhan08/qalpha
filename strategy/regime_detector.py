"""
Regime Detector

Hidden Markov Model-based market regime detection.
Integrates with TDA signals from Phase 1 for enhanced regime classification.

Regimes:
- Bull: Low volatility, positive momentum, expansion
- Bear: High volatility, negative momentum, contraction
- Sideways: Low volatility, no trend, consolidation
- Crisis: Extreme volatility, correlation breakdown
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with probabilities."""

    regime: MarketRegime
    probability: float
    regime_probs: Dict[str, float]
    duration: int  # Days in current regime
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Confidence in current regime classification."""
        return self.probability

    @property
    def is_transition(self) -> bool:
        """Whether we're likely transitioning regimes."""
        return self.probability < 0.6


class GaussianHMM:
    """
    Simple Gaussian Hidden Markov Model implementation.

    Pure numpy/scipy implementation with no external HMM library dependency.
    Uses Baum-Welch algorithm for training.
    """

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        """
        Initialize HMM.

        Args:
            n_states: Number of hidden states
            n_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Model parameters (initialized during fit)
        self.start_prob: Optional[np.ndarray] = None  # Initial state distribution
        self.trans_prob: Optional[np.ndarray] = None  # Transition matrix
        self.means: Optional[np.ndarray] = None  # Emission means
        self.covars: Optional[np.ndarray] = None  # Emission covariances

        self._fitted = False

    def _initialize_params(self, X: np.ndarray) -> None:
        """Initialize parameters using K-means."""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize start probabilities (uniform)
        self.start_prob = np.ones(self.n_states) / self.n_states

        # Initialize transition matrix (slightly sticky)
        self.trans_prob = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.trans_prob, 0.7)
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)

        # Initialize means using k-means-like approach
        indices = np.random.choice(n_samples, self.n_states, replace=False)
        self.means = X[indices].copy()

        # Initialize covariances as diagonal
        self.covars = np.zeros((self.n_states, n_features, n_features))
        global_var = np.var(X, axis=0)
        for i in range(self.n_states):
            self.covars[i] = np.diag(global_var + 1e-6)

    def _emission_prob(self, X: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities P(X|state)."""
        n_samples = X.shape[0]
        emission = np.zeros((n_samples, self.n_states))

        for i in range(self.n_states):
            try:
                # Multivariate normal probability
                emission[:, i] = stats.multivariate_normal.pdf(
                    X, mean=self.means[i], cov=self.covars[i], allow_singular=True
                )
            except Exception:
                # Fallback to diagonal
                emission[:, i] = np.exp(
                    -0.5
                    * np.sum(
                        ((X - self.means[i]) ** 2) / (np.diag(self.covars[i]) + 1e-6),
                        axis=1,
                    )
                )

        # Add small constant to avoid numerical issues
        emission = np.maximum(emission, 1e-300)

        return emission

    def _forward(self, emission: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm (alpha pass)."""
        n_samples = emission.shape[0]
        alpha = np.zeros((n_samples, self.n_states))
        scaling = np.zeros(n_samples)

        # Initialize
        alpha[0] = self.start_prob * emission[0]
        scaling[0] = alpha[0].sum()
        if scaling[0] > 0:
            alpha[0] /= scaling[0]

        # Forward pass
        for t in range(1, n_samples):
            alpha[t] = (alpha[t - 1] @ self.trans_prob) * emission[t]
            scaling[t] = alpha[t].sum()
            if scaling[t] > 0:
                alpha[t] /= scaling[t]

        return alpha, scaling

    def _backward(self, emission: np.ndarray, scaling: np.ndarray) -> np.ndarray:
        """Backward algorithm (beta pass)."""
        n_samples = emission.shape[0]
        beta = np.zeros((n_samples, self.n_states))

        # Initialize
        beta[-1] = 1.0

        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            beta[t] = self.trans_prob @ (emission[t + 1] * beta[t + 1])
            if scaling[t + 1] > 0:
                beta[t] /= scaling[t + 1]

        return beta

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step: compute responsibilities."""
        emission = self._emission_prob(X)
        alpha, scaling = self._forward(emission)
        beta = self._backward(emission, scaling)

        # State responsibilities (gamma)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # Transition responsibilities (xi)
        n_samples = X.shape[0]
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        for t in range(n_samples - 1):
            xi_t = (
                alpha[t, :, np.newaxis]
                * self.trans_prob
                * emission[t + 1, np.newaxis, :]
                * beta[t + 1, np.newaxis, :]
            )
            xi[t] = xi_t / (xi_t.sum() + 1e-300)

        # Log likelihood
        log_likelihood = np.sum(np.log(scaling + 1e-300))

        return gamma, xi, log_likelihood

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step: update parameters."""
        # Update start probabilities
        self.start_prob = gamma[0] / (gamma[0].sum() + 1e-300)

        # Update transition probabilities
        xi_sum = xi.sum(axis=0)
        self.trans_prob = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

        # Update emission parameters
        gamma_sum = gamma.sum(axis=0)

        for i in range(self.n_states):
            # Update means
            self.means[i] = (gamma[:, i : i + 1] * X).sum(axis=0) / (
                gamma_sum[i] + 1e-300
            )

            # Update covariances
            diff = X - self.means[i]
            self.covars[i] = (
                (gamma[:, i : i + 1] * diff).T @ diff / (gamma_sum[i] + 1e-300)
            )
            # Regularize
            self.covars[i] += np.eye(X.shape[1]) * 1e-4

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit HMM using Baum-Welch algorithm.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Remove NaN rows
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        if len(X_clean) < self.n_states * 10:
            logger.warning("Insufficient data for HMM training")
            return self

        self._initialize_params(X_clean)

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            try:
                gamma, xi, log_likelihood = self._e_step(X_clean)
                self._m_step(X_clean, gamma, xi)

                # Check convergence
                if abs(log_likelihood - prev_ll) < self.tol:
                    logger.debug(f"HMM converged at iteration {iteration}")
                    break

                prev_ll = log_likelihood

            except Exception as e:
                logger.warning(f"HMM iteration failed: {e}")
                break

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence (Viterbi).

        Args:
            X: Observation sequence

        Returns:
            Most likely state sequence
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        emission = self._emission_prob(X)

        # Viterbi algorithm
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self.start_prob + 1e-300) + np.log(emission[0] + 1e-300)

        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                probs = viterbi[t - 1] + np.log(self.trans_prob[:, j] + 1e-300)
                backpointer[t, j] = np.argmax(probs)
                viterbi[t, j] = probs[backpointer[t, j]] + np.log(
                    emission[t, j] + 1e-300
                )

        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])

        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities.

        Args:
            X: Observation sequence

        Returns:
            State probabilities for each time step
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        emission = self._emission_prob(X)
        alpha, scaling = self._forward(emission)
        beta = self._backward(emission, scaling)

        # State posteriors
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        return gamma


class RegimeDetector:
    """
    Market regime detector using HMM and technical features.

    Combines:
    - Price momentum and volatility
    - Yield curve signals (from FRED)
    - TDA features (from Phase 1)
    - Credit spreads
    """

    REGIME_NAMES = {
        0: MarketRegime.BULL,
        1: MarketRegime.BEAR,
        2: MarketRegime.SIDEWAYS,
        3: MarketRegime.CRISIS,
    }

    def __init__(
        self,
        lookback_window: int = 252,
        volatility_window: int = 20,
        momentum_window: int = 60,
        n_regimes: int = 4,
        use_tda: bool = True,
    ):
        """
        Initialize regime detector.

        Args:
            lookback_window: Historical data for training
            volatility_window: Window for volatility calculation
            momentum_window: Window for momentum calculation
            n_regimes: Number of regimes (default 4)
            use_tda: Whether to use TDA features
        """
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.n_regimes = n_regimes
        self.use_tda = use_tda

        self.hmm = GaussianHMM(n_states=n_regimes)
        self._fitted = False
        self._last_regime = MarketRegime.UNKNOWN
        self._regime_duration = 0

        # Feature history
        self._feature_history: List[Dict[str, float]] = []

    def _calculate_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime detection features.

        Args:
            prices: OHLCV DataFrame

        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame(index=prices.index)

        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]

        # Returns
        returns = close.pct_change()

        # 1. Volatility features
        features["volatility"] = returns.rolling(
            self.volatility_window
        ).std() * np.sqrt(252)
        features["volatility_change"] = features["volatility"].pct_change(20)

        # Volatility regime (normalized)
        vol_mean = features["volatility"].rolling(252).mean()
        vol_std = features["volatility"].rolling(252).std()
        features["volatility_zscore"] = (features["volatility"] - vol_mean) / (
            vol_std + 1e-6
        )

        # 2. Momentum features
        features["momentum_20d"] = close.pct_change(20)
        features["momentum_60d"] = close.pct_change(60)

        # Trend strength
        features["trend_strength"] = features["momentum_60d"].abs() / (
            features["volatility"] + 1e-6
        )

        # 3. Drawdown
        rolling_max = close.rolling(252).max()
        features["drawdown"] = (close - rolling_max) / rolling_max

        # 4. Autocorrelation (mean-reversion indicator)
        features["autocorr"] = returns.rolling(60).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        # 5. Skewness and Kurtosis
        features["skewness"] = returns.rolling(60).skew()
        features["kurtosis"] = returns.rolling(60).kurt()

        # 6. Volume regime (if available)
        if "volume" in prices.columns:
            vol_ma = prices["volume"].rolling(20).mean()
            features["volume_ratio"] = prices["volume"] / (vol_ma + 1e-6)

        # 7. High-Low range
        if "high" in prices.columns and "low" in prices.columns:
            features["range"] = (prices["high"] - prices["low"]) / close
            features["range_zscore"] = (
                features["range"] - features["range"].rolling(60).mean()
            ) / (features["range"].rolling(60).std() + 1e-6)

        return features

    def _add_tda_features(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add TDA features if available.

        Args:
            features: Existing features DataFrame
            prices: Price DataFrame

        Returns:
            Enhanced features DataFrame
        """
        if not self.use_tda:
            return features

        try:
            from quantum_alpha.features.mathematical.persistent_homology import (
                PersistentHomologyCalculator,
            )
            from quantum_alpha.features.mathematical.hurst_exponent import (
                HurstExponentCalculator,
            )

            close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]

            # TDA features
            ph_calc = PersistentHomologyCalculator()
            hurst_calc = HurstExponentCalculator()

            # Calculate rolling TDA (expensive, so use larger windows)
            window = 60
            tda_features = []
            hurst_values = []

            for i in range(len(close)):
                if i < window:
                    tda_features.append({})
                    hurst_values.append(0.5)
                    continue

                window_data = close.iloc[i - window : i].values

                # TDA
                try:
                    tda_result = ph_calc.calculate(window_data)
                    tda_features.append(tda_result)
                except Exception:
                    tda_features.append({})

                # Hurst
                try:
                    hurst = hurst_calc.calculate(window_data)
                    hurst_values.append(hurst.get("hurst_exponent", 0.5))
                except Exception:
                    hurst_values.append(0.5)

            features["hurst"] = hurst_values

            # Extract key TDA features
            features["tda_persistence"] = [
                f.get("total_persistence", 0) for f in tda_features
            ]

            logger.info("Added TDA features for regime detection")

        except ImportError:
            logger.debug("TDA features not available")
        except Exception as e:
            logger.warning(f"Failed to add TDA features: {e}")

        return features

    def _add_macro_features(
        self, features: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add macroeconomic features.

        Args:
            features: Existing features
            macro_data: DataFrame with macro indicators

        Returns:
            Enhanced features
        """
        if macro_data is None or macro_data.empty:
            return features

        # Align dates
        macro_aligned = macro_data.reindex(features.index, method="ffill")

        # Add key macro features
        macro_cols = [
            "spread_10y_2y",
            "curve_inverted",
            "vix",
            "hy_spread",
            "recession_risk",
            "stress_indicator",
        ]

        for col in macro_cols:
            if col in macro_aligned.columns:
                features[f"macro_{col}"] = macro_aligned[col]

        return features

    def fit(
        self, prices: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None
    ) -> "RegimeDetector":
        """
        Fit regime detector on historical data.

        Args:
            prices: OHLCV DataFrame
            macro_data: Optional macro indicators

        Returns:
            self
        """
        # Calculate features
        features = self._calculate_features(prices)

        # Add TDA features
        features = self._add_tda_features(features, prices)

        # Add macro features
        features = self._add_macro_features(features, macro_data)

        # Select features for HMM
        feature_cols = [
            "volatility_zscore",
            "momentum_60d",
            "drawdown",
            "trend_strength",
            "autocorr",
        ]

        # Add available features
        for col in ["hurst", "macro_spread_10y_2y", "macro_vix", "macro_hy_spread"]:
            if col in features.columns:
                feature_cols.append(col)

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in features.columns]

        if len(available_cols) < 2:
            logger.error("Insufficient features for regime detection")
            return self

        # Prepare training data
        X = features[available_cols].dropna().values

        if len(X) < self.lookback_window:
            logger.warning("Insufficient data for regime training")
            return self

        # Use recent data for training
        X_train = X[-self.lookback_window :]

        # Fit HMM
        logger.info(
            f"Fitting HMM with {len(X_train)} samples, {len(available_cols)} features"
        )
        self.hmm.fit(X_train)

        self._fitted = True
        self._feature_cols = available_cols

        # Label regimes based on characteristics
        self._label_regimes(features[available_cols].dropna(), prices)

        return self

    def _label_regimes(self, features: pd.DataFrame, prices: pd.DataFrame) -> None:
        """
        Label HMM states with regime names based on characteristics.

        Maps hidden states to interpretable regimes.
        """
        X = features.values
        states = self.hmm.predict(X)

        # Calculate average characteristics per state
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        returns = close.pct_change().reindex(features.index).values

        state_stats = {}

        for state in range(self.n_regimes):
            mask = states == state
            if mask.sum() < 10:
                continue

            state_returns = returns[:-1][mask[1:]]  # Align returns
            if len(state_returns) == 0:
                continue

            state_stats[state] = {
                "mean_return": np.nanmean(state_returns),
                "volatility": np.nanstd(state_returns),
                "count": mask.sum(),
            }

        if not state_stats:
            logger.warning("Could not label regimes")
            return

        # Sort states by characteristics
        sorted_states = sorted(
            state_stats.items(),
            key=lambda x: (x[1]["mean_return"], -x[1]["volatility"]),
            reverse=True,
        )

        # Assign regime labels
        self._state_to_regime = {}

        if len(sorted_states) >= 4:
            self._state_to_regime[sorted_states[0][0]] = MarketRegime.BULL
            self._state_to_regime[sorted_states[-1][0]] = MarketRegime.CRISIS

            # Middle states - distinguish by volatility
            mid_states = sorted_states[1:-1]
            mid_states.sort(key=lambda x: x[1]["volatility"])

            if len(mid_states) >= 2:
                self._state_to_regime[mid_states[0][0]] = MarketRegime.SIDEWAYS
                self._state_to_regime[mid_states[1][0]] = MarketRegime.BEAR
            elif len(mid_states) == 1:
                self._state_to_regime[mid_states[0][0]] = MarketRegime.SIDEWAYS
        elif len(sorted_states) >= 2:
            self._state_to_regime[sorted_states[0][0]] = MarketRegime.BULL
            self._state_to_regime[sorted_states[-1][0]] = MarketRegime.BEAR

        logger.info(f"Regime labels: {self._state_to_regime}")

    def predict(
        self, prices: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None
    ) -> RegimeState:
        """
        Predict current market regime.

        Args:
            prices: Recent OHLCV data
            macro_data: Optional macro indicators

        Returns:
            Current regime state
        """
        if not self._fitted:
            logger.warning("Detector not fitted, using rule-based fallback")
            return self._rule_based_regime(prices)

        # Calculate features
        features = self._calculate_features(prices)
        features = self._add_tda_features(features, prices)
        features = self._add_macro_features(features, macro_data)

        # Get available features
        available_cols = [c for c in self._feature_cols if c in features.columns]

        if len(available_cols) < 2:
            return self._rule_based_regime(prices)

        # Prepare data
        X = features[available_cols].dropna().values

        if len(X) == 0:
            return self._rule_based_regime(prices)

        # Get state probabilities
        probs = self.hmm.predict_proba(X)
        current_probs = probs[-1]

        # Get most likely state
        state = np.argmax(current_probs)
        probability = current_probs[state]

        # Map to regime
        regime = self._state_to_regime.get(state, MarketRegime.UNKNOWN)

        # Update duration tracking
        if regime == self._last_regime:
            self._regime_duration += 1
        else:
            self._last_regime = regime
            self._regime_duration = 1

        # Build regime probabilities dict
        regime_probs = {}
        for s, r in self._state_to_regime.items():
            if s < len(current_probs):
                regime_probs[r.value] = float(current_probs[s])

        # Get current feature values
        current_features = {
            col: float(features[col].iloc[-1])
            for col in available_cols
            if not np.isnan(features[col].iloc[-1])
        }

        return RegimeState(
            regime=regime,
            probability=float(probability),
            regime_probs=regime_probs,
            duration=self._regime_duration,
            features=current_features,
        )

    def _rule_based_regime(self, prices: pd.DataFrame) -> RegimeState:
        """
        Fallback rule-based regime detection.

        Uses simple heuristics when HMM unavailable.
        """
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        returns = close.pct_change()

        # Calculate indicators
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        momentum = close.pct_change(60).iloc[-1]
        drawdown = (close.iloc[-1] - close.rolling(252).max().iloc[-1]) / close.rolling(
            252
        ).max().iloc[-1]

        # Simple classification
        if volatility > 0.35 and drawdown < -0.15:
            regime = MarketRegime.CRISIS
            prob = 0.7
        elif volatility > 0.25 or momentum < -0.1:
            regime = MarketRegime.BEAR
            prob = 0.6
        elif momentum > 0.1 and volatility < 0.2:
            regime = MarketRegime.BULL
            prob = 0.7
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.5

        return RegimeState(
            regime=regime,
            probability=prob,
            regime_probs={
                r.value: 0.25 for r in MarketRegime if r != MarketRegime.UNKNOWN
            },
            duration=1,
            features={
                "volatility": float(volatility) if not np.isnan(volatility) else 0.2,
                "momentum": float(momentum) if not np.isnan(momentum) else 0.0,
                "drawdown": float(drawdown) if not np.isnan(drawdown) else 0.0,
            },
        )

    def get_regime_history(
        self, prices: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get historical regime classifications.

        Args:
            prices: Historical OHLCV data
            macro_data: Optional macro indicators

        Returns:
            DataFrame with regime history
        """
        if not self._fitted:
            self.fit(prices, macro_data)

        features = self._calculate_features(prices)
        features = self._add_tda_features(features, prices)
        features = self._add_macro_features(features, macro_data)

        available_cols = [c for c in self._feature_cols if c in features.columns]
        X = features[available_cols].dropna()

        if len(X) == 0:
            return pd.DataFrame()

        # Get predictions
        states = self.hmm.predict(X.values)
        probs = self.hmm.predict_proba(X.values)

        # Build history DataFrame
        history = pd.DataFrame(index=X.index)
        history["state"] = states
        history["regime"] = [
            self._state_to_regime.get(s, MarketRegime.UNKNOWN).value for s in states
        ]
        history["probability"] = probs.max(axis=1)

        # Add regime probabilities
        for s, r in self._state_to_regime.items():
            if s < probs.shape[1]:
                history[f"prob_{r.value}"] = probs[:, s]

        return history

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame with transition probabilities
        """
        if not self._fitted or self.hmm.trans_prob is None:
            return pd.DataFrame()

        regime_names = [
            self._state_to_regime.get(i, MarketRegime.UNKNOWN).value
            for i in range(self.n_regimes)
        ]

        return pd.DataFrame(
            self.hmm.trans_prob, index=regime_names, columns=regime_names
        )

    def regime_adjusted_signal(self, signal: float, regime: RegimeState) -> float:
        """
        Adjust trading signal based on regime.

        Args:
            signal: Original signal (-1 to 1)
            regime: Current regime state

        Returns:
            Adjusted signal
        """
        # Regime-specific adjustments
        adjustments = {
            MarketRegime.BULL: 1.2,  # Amplify longs
            MarketRegime.BEAR: 0.5,  # Reduce exposure
            MarketRegime.SIDEWAYS: 0.8,  # Slight reduction
            MarketRegime.CRISIS: 0.2,  # Minimal exposure
            MarketRegime.UNKNOWN: 1.0,  # No adjustment
        }

        factor = adjustments.get(regime.regime, 1.0)

        # Weight by confidence
        adjusted_factor = 1.0 + (factor - 1.0) * regime.confidence

        return signal * adjusted_factor


def create_regime_detector(**kwargs) -> RegimeDetector:
    """
    Factory function to create regime detector.

    Args:
        **kwargs: Arguments for RegimeDetector

    Returns:
        RegimeDetector instance
    """
    return RegimeDetector(**kwargs)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    np.random.seed(42)
    n_days = 500

    # Simulate different regimes
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Bull regime (first 150 days)
    bull_returns = np.random.normal(0.001, 0.01, 150)
    # Bear regime (next 100 days)
    bear_returns = np.random.normal(-0.002, 0.02, 100)
    # Sideways (next 150 days)
    sideways_returns = np.random.normal(0, 0.008, 150)
    # Crisis (last 100 days)
    crisis_returns = np.random.normal(-0.003, 0.035, 100)

    all_returns = np.concatenate(
        [bull_returns, bear_returns, sideways_returns, crisis_returns]
    )
    prices = 100 * np.exp(np.cumsum(all_returns))

    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            "volume": np.random.randint(1000000, 10000000, n_days),
        },
        index=dates,
    )

    # Test regime detector
    detector = RegimeDetector(use_tda=False)  # Disable TDA for test
    detector.fit(df)

    # Get current regime
    current = detector.predict(df)
    print(f"\nCurrent Regime: {current.regime.value}")
    print(f"Probability: {current.probability:.2%}")
    print(f"Duration: {current.duration} days")

    # Get history
    history = detector.get_regime_history(df)
    if not history.empty:
        print(f"\nRegime Distribution:")
        print(history["regime"].value_counts(normalize=True))

    # Get transition matrix
    trans = detector.get_transition_matrix()
    if not trans.empty:
        print(f"\nTransition Matrix:")
        print(trans.round(3))
