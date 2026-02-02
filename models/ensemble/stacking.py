"""
Ensemble Stacking Module

Implements stacking ensemble methods for combining multiple model predictions.
Uses out-of-fold predictions to train meta-learner, avoiding overfitting.

Methods:
- StackingEnsemble: Train meta-model on base model predictions
- VotingEnsemble: Weighted voting across models
- CrossValidatedStacking: K-fold stacking for robust meta-learning
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model prediction."""

    name: str
    predictions: np.ndarray  # Shape: (n_samples,) or (n_samples, n_outputs)
    probabilities: Optional[np.ndarray] = None  # For classification
    confidence: Optional[np.ndarray] = None  # Model confidence per prediction
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods."""

    @abstractmethod
    def fit(self, predictions: List[ModelPrediction], y: np.ndarray) -> "BaseEnsemble":
        """Fit ensemble on predictions."""
        pass

    @abstractmethod
    def predict(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """Generate ensemble predictions."""
        pass

    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """Get model weights in ensemble."""
        pass


class SimpleMetaLearner:
    """
    Simple meta-learner for stacking.

    Uses ridge regression for continuous targets,
    logistic regression for binary targets.

    No sklearn dependency - pure numpy implementation.
    """

    def __init__(
        self, alpha: float = 1.0, fit_intercept: bool = True, task: str = "regression"
    ):
        """
        Initialize meta-learner.

        Args:
            alpha: Regularization strength
            fit_intercept: Whether to fit intercept
            task: 'regression' or 'classification'
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.task = task

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleMetaLearner":
        """
        Fit meta-learner using ridge regression.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Add intercept column if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1

        # Ridge regression closed form: (X'X + αI)^(-1) X'y
        XtX = X.T @ X
        XtX += self.alpha * np.eye(n_features)
        Xty = X.T @ y

        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            coef = np.linalg.pinv(XtX) @ Xty

        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = X @ self.coef_ + self.intercept_

        if self.task == "classification":
            # Sigmoid for binary classification
            y_pred = 1 / (1 + np.exp(-y_pred))

        return y_pred


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble with meta-learner.

    Uses out-of-fold predictions from base models
    to train a meta-learner that combines predictions.
    """

    def __init__(
        self,
        meta_learner: Optional[SimpleMetaLearner] = None,
        use_probabilities: bool = False,
        include_original_features: bool = False,
        n_folds: int = 5,
    ):
        """
        Initialize stacking ensemble.

        Args:
            meta_learner: Meta-learner instance (default: SimpleMetaLearner)
            use_probabilities: Use probability outputs instead of predictions
            include_original_features: Include original features in meta-learner
            n_folds: Number of CV folds for out-of-fold predictions
        """
        self.meta_learner = meta_learner or SimpleMetaLearner()
        self.use_probabilities = use_probabilities
        self.include_original_features = include_original_features
        self.n_folds = n_folds

        self._model_names: List[str] = []
        self._fitted = False

    def _prepare_meta_features(
        self,
        predictions: List[ModelPrediction],
        original_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Prepare features for meta-learner.

        Args:
            predictions: List of model predictions
            original_features: Optional original feature matrix

        Returns:
            Meta-feature matrix
        """
        meta_features = []

        for pred in predictions:
            if self.use_probabilities and pred.probabilities is not None:
                meta_features.append(pred.probabilities)
            else:
                p = pred.predictions
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                meta_features.append(p)

            # Add confidence if available
            if pred.confidence is not None:
                conf = pred.confidence.reshape(-1, 1)
                meta_features.append(conf)

        X_meta = np.hstack(meta_features)

        if self.include_original_features and original_features is not None:
            X_meta = np.hstack([X_meta, original_features])

        return X_meta

    def fit(
        self,
        predictions: List[ModelPrediction],
        y: np.ndarray,
        original_features: Optional[np.ndarray] = None,
    ) -> "StackingEnsemble":
        """
        Fit stacking ensemble.

        Args:
            predictions: List of base model predictions
            y: True target values
            original_features: Optional original features

        Returns:
            self
        """
        if len(predictions) == 0:
            raise ValueError("No predictions provided")

        self._model_names = [p.name for p in predictions]

        # Prepare meta-features
        X_meta = self._prepare_meta_features(predictions, original_features)

        # Fit meta-learner
        self.meta_learner.fit(X_meta, y)

        self._fitted = True
        logger.info(f"Fitted stacking ensemble with {len(predictions)} base models")

        return self

    def fit_with_cv(
        self,
        base_models: List[Callable],
        X: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> "StackingEnsemble":
        """
        Fit with cross-validation for out-of-fold predictions.

        This prevents overfitting by using OOF predictions for meta-learner.

        Args:
            base_models: List of model fit/predict callables
            X: Training features
            y: Training targets
            model_names: Optional names for models

        Returns:
            self
        """
        n_samples = len(y)
        n_models = len(base_models)

        if model_names is None:
            model_names = [f"model_{i}" for i in range(n_models)]

        self._model_names = model_names

        # Initialize OOF predictions
        oof_predictions = np.zeros((n_samples, n_models))

        # Generate fold indices
        fold_size = n_samples // self.n_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for fold in range(self.n_folds):
            # Create fold masks
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples

            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            # Train each base model and get OOF predictions
            for i, model in enumerate(base_models):
                try:
                    # Fit model (assumes sklearn-like interface)
                    model.fit(X_train, y_train)
                    oof_predictions[val_idx, i] = model.predict(X_val).flatten()
                except Exception as e:
                    logger.warning(f"Model {model_names[i]} failed on fold {fold}: {e}")
                    oof_predictions[val_idx, i] = 0

        # Create predictions list
        predictions = [
            ModelPrediction(name=name, predictions=oof_predictions[:, i])
            for i, name in enumerate(model_names)
        ]

        # Fit meta-learner on OOF predictions
        return self.fit(predictions, y)

    def predict(
        self,
        predictions: List[ModelPrediction],
        original_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate stacked predictions.

        Args:
            predictions: Base model predictions
            original_features: Optional original features

        Returns:
            Ensemble predictions
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted")

        X_meta = self._prepare_meta_features(predictions, original_features)
        return self.meta_learner.predict(X_meta)

    def get_weights(self) -> Dict[str, float]:
        """
        Get effective model weights.

        Approximated from meta-learner coefficients.
        """
        if not self._fitted or self.meta_learner.coef_ is None:
            return {}

        coef = self.meta_learner.coef_

        # Normalize to get relative importance
        n_models = len(self._model_names)
        if len(coef) < n_models:
            return {}

        model_coef = coef[:n_models]
        abs_coef = np.abs(model_coef)
        weights = abs_coef / (abs_coef.sum() + 1e-10)

        return {name: float(w) for name, w in zip(self._model_names, weights)}


class VotingEnsemble(BaseEnsemble):
    """
    Weighted voting ensemble.

    Combines predictions using learned or fixed weights.
    """

    def __init__(
        self,
        voting: str = "soft",
        weights: Optional[Dict[str, float]] = None,
        optimize_weights: bool = True,
    ):
        """
        Initialize voting ensemble.

        Args:
            voting: 'soft' (weighted average) or 'hard' (majority vote)
            weights: Fixed weights per model (if not optimizing)
            optimize_weights: Whether to learn optimal weights
        """
        self.voting = voting
        self.fixed_weights = weights
        self.optimize_weights = optimize_weights

        self._weights: Dict[str, float] = {}
        self._model_names: List[str] = []
        self._fitted = False

    def _optimize_weights_grid(
        self, predictions: np.ndarray, y: np.ndarray, n_points: int = 21
    ) -> np.ndarray:
        """
        Optimize weights using grid search.

        Minimizes MSE for regression.
        """
        n_models = predictions.shape[1]

        if n_models == 1:
            return np.array([1.0])

        if n_models == 2:
            # Simple grid search for 2 models
            best_weights = None
            best_score = float("inf")

            for w1 in np.linspace(0, 1, n_points):
                w2 = 1 - w1
                weights = np.array([w1, w2])

                y_pred = predictions @ weights
                score = np.mean((y - y_pred) ** 2)

                if score < best_score:
                    best_score = score
                    best_weights = weights

            return best_weights

        # For 3+ models, use constrained optimization
        from scipy.optimize import minimize

        def objective(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            y_pred = predictions @ weights
            return np.mean((y - y_pred) ** 2)

        # Initialize with equal weights
        x0 = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0, 1)] * n_models,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        )

        weights = np.abs(result.x)
        weights = weights / weights.sum()

        return weights

    def fit(
        self, predictions: List[ModelPrediction], y: np.ndarray
    ) -> "VotingEnsemble":
        """
        Fit voting ensemble.

        Args:
            predictions: List of base model predictions
            y: True target values

        Returns:
            self
        """
        if len(predictions) == 0:
            raise ValueError("No predictions provided")

        self._model_names = [p.name for p in predictions]

        # Stack predictions
        pred_matrix = np.column_stack([p.predictions for p in predictions])

        if self.fixed_weights is not None:
            # Use fixed weights
            weights = np.array(
                [
                    self.fixed_weights.get(name, 1.0 / len(predictions))
                    for name in self._model_names
                ]
            )
            weights = weights / weights.sum()
        elif self.optimize_weights:
            # Optimize weights
            weights = self._optimize_weights_grid(pred_matrix, y)
        else:
            # Equal weights
            weights = np.ones(len(predictions)) / len(predictions)

        self._weights = {name: float(w) for name, w in zip(self._model_names, weights)}

        self._fitted = True
        logger.info(f"Fitted voting ensemble with weights: {self._weights}")

        return self

    def predict(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """
        Generate voting predictions.

        Args:
            predictions: Base model predictions

        Returns:
            Ensemble predictions
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted")

        if self.voting == "soft":
            # Weighted average
            result = np.zeros(len(predictions[0].predictions))

            for pred in predictions:
                weight = self._weights.get(pred.name, 0)
                result += weight * pred.predictions

            return result

        else:  # hard voting
            # Stack predictions and take weighted mode
            pred_matrix = np.column_stack([p.predictions for p in predictions])
            weights = np.array([self._weights.get(p.name, 0) for p in predictions])

            # Round to classes and take weighted mode
            pred_classes = np.round(pred_matrix)

            result = np.zeros(pred_matrix.shape[0])
            for i in range(pred_matrix.shape[0]):
                classes, counts = np.unique(pred_classes[i], return_counts=True)
                # Weight counts
                weighted_counts = np.zeros_like(counts, dtype=float)
                for j, c in enumerate(classes):
                    mask = pred_classes[i] == c
                    weighted_counts[j] = weights[mask].sum()
                result[i] = classes[np.argmax(weighted_counts)]

            return result

    def get_weights(self) -> Dict[str, float]:
        """Get model weights."""
        return self._weights.copy()


class DiversityWeightedEnsemble(BaseEnsemble):
    """
    Ensemble that weights models by both accuracy and diversity.

    Models that are accurate AND different from others get higher weight.
    """

    def __init__(
        self,
        accuracy_weight: float = 0.6,
        diversity_weight: float = 0.4,
        min_weight: float = 0.05,
    ):
        """
        Initialize diversity-weighted ensemble.

        Args:
            accuracy_weight: Weight for accuracy in final weight
            diversity_weight: Weight for diversity in final weight
            min_weight: Minimum weight for any model
        """
        self.accuracy_weight = accuracy_weight
        self.diversity_weight = diversity_weight
        self.min_weight = min_weight

        self._weights: Dict[str, float] = {}
        self._model_names: List[str] = []
        self._fitted = False

    def _calculate_accuracy_scores(
        self, predictions: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Calculate accuracy score for each model."""
        n_models = predictions.shape[1]
        scores = np.zeros(n_models)

        for i in range(n_models):
            # Use negative MSE as score (higher is better)
            mse = np.mean((predictions[:, i] - y) ** 2)
            scores[i] = 1 / (1 + mse)  # Transform to 0-1 range

        return scores

    def _calculate_diversity_scores(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate diversity score for each model.

        Higher score = more different from other models.
        """
        n_models = predictions.shape[1]

        if n_models == 1:
            return np.array([1.0])

        # Calculate pairwise correlations
        corr_matrix = np.corrcoef(predictions.T)

        # Diversity = 1 - average correlation with others
        diversity_scores = np.zeros(n_models)
        for i in range(n_models):
            other_corrs = np.abs(corr_matrix[i, :])
            other_corrs[i] = 0  # Exclude self
            diversity_scores[i] = 1 - other_corrs.mean()

        return diversity_scores

    def fit(
        self, predictions: List[ModelPrediction], y: np.ndarray
    ) -> "DiversityWeightedEnsemble":
        """
        Fit diversity-weighted ensemble.

        Args:
            predictions: List of base model predictions
            y: True target values

        Returns:
            self
        """
        if len(predictions) == 0:
            raise ValueError("No predictions provided")

        self._model_names = [p.name for p in predictions]

        # Stack predictions
        pred_matrix = np.column_stack([p.predictions for p in predictions])

        # Calculate scores
        accuracy_scores = self._calculate_accuracy_scores(pred_matrix, y)
        diversity_scores = self._calculate_diversity_scores(pred_matrix)

        # Combine scores
        combined_scores = (
            self.accuracy_weight * accuracy_scores
            + self.diversity_weight * diversity_scores
        )

        # Normalize to weights
        weights = combined_scores / combined_scores.sum()

        # Apply minimum weight
        weights = np.maximum(weights, self.min_weight)
        weights = weights / weights.sum()

        self._weights = {name: float(w) for name, w in zip(self._model_names, weights)}

        self._fitted = True
        logger.info(
            f"Fitted diversity ensemble. Accuracy: {accuracy_scores}, Diversity: {diversity_scores}"
        )

        return self

    def predict(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """
        Generate weighted predictions.

        Args:
            predictions: Base model predictions

        Returns:
            Ensemble predictions
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted")

        result = np.zeros(len(predictions[0].predictions))

        for pred in predictions:
            weight = self._weights.get(pred.name, 0)
            result += weight * pred.predictions

        return result

    def get_weights(self) -> Dict[str, float]:
        """Get model weights."""
        return self._weights.copy()


class AdaptiveEnsemble(BaseEnsemble):
    """
    Ensemble with regime-adaptive weights.

    Adjusts weights based on recent performance in different regimes.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.05,
    ):
        """
        Initialize adaptive ensemble.

        Args:
            lookback_window: Window for recent performance calculation
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum weight for any model
        """
        self.lookback_window = lookback_window
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight

        self._weights: Dict[str, float] = {}
        self._model_names: List[str] = []
        self._performance_history: Dict[str, List[float]] = {}
        self._fitted = False

    def fit(
        self, predictions: List[ModelPrediction], y: np.ndarray
    ) -> "AdaptiveEnsemble":
        """
        Initial fit of adaptive ensemble.

        Args:
            predictions: List of base model predictions
            y: True target values

        Returns:
            self
        """
        if len(predictions) == 0:
            raise ValueError("No predictions provided")

        self._model_names = [p.name for p in predictions]

        # Initialize with equal weights
        n_models = len(predictions)
        self._weights = {name: 1.0 / n_models for name in self._model_names}

        # Initialize performance history
        self._performance_history = {name: [] for name in self._model_names}

        # Calculate initial performance
        for pred in predictions:
            errors = np.abs(pred.predictions - y)
            self._performance_history[pred.name] = errors.tolist()

        # Update weights based on recent performance
        self._update_weights()

        self._fitted = True
        return self

    def _update_weights(self) -> None:
        """Update weights based on recent performance."""
        recent_scores = {}

        for name in self._model_names:
            history = self._performance_history.get(name, [])
            if len(history) == 0:
                recent_scores[name] = 1.0
                continue

            # Use recent errors
            recent = history[-self.lookback_window :]
            # Convert error to score (lower error = higher score)
            mean_error = np.mean(recent)
            recent_scores[name] = 1 / (1 + mean_error)

        # Normalize scores to weights
        total_score = sum(recent_scores.values())

        for name in self._model_names:
            new_weight = recent_scores[name] / total_score

            # Smooth adaptation
            old_weight = self._weights.get(name, 1.0 / len(self._model_names))
            self._weights[name] = (
                1 - self.adaptation_rate
            ) * old_weight + self.adaptation_rate * new_weight

        # Apply minimum and renormalize
        for name in self._weights:
            self._weights[name] = max(self._weights[name], self.min_weight)

        total = sum(self._weights.values())
        self._weights = {n: w / total for n, w in self._weights.items()}

    def update(self, predictions: List[ModelPrediction], y_true: float) -> None:
        """
        Update ensemble with new observation.

        Args:
            predictions: Model predictions for current timestep
            y_true: True value
        """
        for pred in predictions:
            if pred.name in self._performance_history:
                error = abs(
                    pred.predictions[-1]
                    if len(pred.predictions) > 1
                    else pred.predictions[0] - y_true
                )
                self._performance_history[pred.name].append(error)

        self._update_weights()

    def predict(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """
        Generate adaptive weighted predictions.

        Args:
            predictions: Base model predictions

        Returns:
            Ensemble predictions
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted")

        result = np.zeros(len(predictions[0].predictions))

        for pred in predictions:
            weight = self._weights.get(pred.name, 0)
            result += weight * pred.predictions

        return result

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self._weights.copy()


def create_stacking_ensemble(**kwargs) -> StackingEnsemble:
    """Factory for stacking ensemble."""
    return StackingEnsemble(**kwargs)


def create_voting_ensemble(**kwargs) -> VotingEnsemble:
    """Factory for voting ensemble."""
    return VotingEnsemble(**kwargs)


def create_diversity_ensemble(**kwargs) -> DiversityWeightedEnsemble:
    """Factory for diversity-weighted ensemble."""
    return DiversityWeightedEnsemble(**kwargs)


def create_adaptive_ensemble(**kwargs) -> AdaptiveEnsemble:
    """Factory for adaptive ensemble."""
    return AdaptiveEnsemble(**kwargs)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample predictions
    np.random.seed(42)
    n_samples = 100

    # True values
    y_true = np.sin(np.linspace(0, 4 * np.pi, n_samples)) + np.random.normal(
        0, 0.1, n_samples
    )

    # Model predictions (with different biases/noise)
    predictions = [
        ModelPrediction(
            name="model_1", predictions=y_true + np.random.normal(0, 0.2, n_samples)
        ),
        ModelPrediction(
            name="model_2",
            predictions=y_true * 0.9 + np.random.normal(0, 0.15, n_samples),
        ),
        ModelPrediction(
            name="model_3",
            predictions=y_true + 0.1 + np.random.normal(0, 0.25, n_samples),
        ),
    ]

    # Test stacking ensemble
    print("Testing Stacking Ensemble...")
    stacking = StackingEnsemble()
    stacking.fit(predictions, y_true)
    stacking_pred = stacking.predict(predictions)
    stacking_mse = np.mean((stacking_pred - y_true) ** 2)
    print(f"Stacking MSE: {stacking_mse:.4f}")
    print(f"Stacking weights: {stacking.get_weights()}")

    # Test voting ensemble
    print("\nTesting Voting Ensemble...")
    voting = VotingEnsemble(optimize_weights=True)
    voting.fit(predictions, y_true)
    voting_pred = voting.predict(predictions)
    voting_mse = np.mean((voting_pred - y_true) ** 2)
    print(f"Voting MSE: {voting_mse:.4f}")
    print(f"Voting weights: {voting.get_weights()}")

    # Test diversity ensemble
    print("\nTesting Diversity Ensemble...")
    diversity = DiversityWeightedEnsemble()
    diversity.fit(predictions, y_true)
    diversity_pred = diversity.predict(predictions)
    diversity_mse = np.mean((diversity_pred - y_true) ** 2)
    print(f"Diversity MSE: {diversity_mse:.4f}")
    print(f"Diversity weights: {diversity.get_weights()}")

    # Compare with individual models
    print("\nIndividual Model MSEs:")
    for pred in predictions:
        mse = np.mean((pred.predictions - y_true) ** 2)
        print(f"  {pred.name}: {mse:.4f}")
