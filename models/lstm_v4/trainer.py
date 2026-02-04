"""
LSTM Trainer Module.

Handles training workflow for Multi-Horizon LSTM:
- Data preparation and normalization
- Training with validation
- Model evaluation and metrics
- Checkpointing and logging
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import os
from datetime import datetime

from .architecture import MultiHorizonLSTM, JumpingWindowGenerator, HorizonConfig


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    validation_split: float = 0.2
    sequence_length: int = 90
    window_step_size: Optional[int] = None  # None = no overlap


class FeatureScaler:
    """
    Feature scaler for time series data.

    Supports:
    - Z-score normalization (standard scaler)
    - Min-max normalization
    - Robust scaling (median/IQR)
    """

    def __init__(self, method: str = "zscore"):
        """
        Initialize scaler.

        Args:
            method: 'zscore', 'minmax', or 'robust'
        """
        self.method = method
        self.params = {}
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Fit scaler to data.

        Args:
            X: Data of shape (n_samples, seq_len, n_features) or (n_samples, n_features)

        Returns:
            self
        """
        # Reshape to 2D if needed
        original_shape = X.shape
        if len(original_shape) == 3:
            X_flat = X.reshape(-1, original_shape[-1])
        else:
            X_flat = X

        if self.method == "zscore":
            self.params["mean"] = np.mean(X_flat, axis=0)
            self.params["std"] = np.std(X_flat, axis=0)
            self.params["std"] = np.where(
                self.params["std"] < 1e-8, 1.0, self.params["std"]
            )

        elif self.method == "minmax":
            self.params["min"] = np.min(X_flat, axis=0)
            self.params["max"] = np.max(X_flat, axis=0)
            range_vals = self.params["max"] - self.params["min"]
            self.params["range"] = np.where(range_vals < 1e-8, 1.0, range_vals)

        elif self.method == "robust":
            self.params["median"] = np.median(X_flat, axis=0)
            q75, q25 = np.percentile(X_flat, [75, 25], axis=0)
            iqr = q75 - q25
            self.params["iqr"] = np.where(iqr < 1e-8, 1.0, iqr)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        original_shape = X.shape
        if len(original_shape) == 3:
            X_flat = X.reshape(-1, original_shape[-1])
        else:
            X_flat = X.copy()

        if self.method == "zscore":
            X_scaled = (X_flat - self.params["mean"]) / self.params["std"]

        elif self.method == "minmax":
            X_scaled = (X_flat - self.params["min"]) / self.params["range"]

        elif self.method == "robust":
            X_scaled = (X_flat - self.params["median"]) / self.params["iqr"]

        if len(original_shape) == 3:
            X_scaled = X_scaled.reshape(original_shape)

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")

        original_shape = X.shape
        if len(original_shape) == 3:
            X_flat = X.reshape(-1, original_shape[-1])
        else:
            X_flat = X.copy()

        if self.method == "zscore":
            X_orig = X_flat * self.params["std"] + self.params["mean"]

        elif self.method == "minmax":
            X_orig = X_flat * self.params["range"] + self.params["min"]

        elif self.method == "robust":
            X_orig = X_flat * self.params["iqr"] + self.params["median"]

        if len(original_shape) == 3:
            X_orig = X_orig.reshape(original_shape)

        return X_orig

    def save(self, path: str):
        """Save scaler parameters."""
        with open(path, "w") as f:
            json.dump(
                {
                    "method": self.method,
                    "params": {k: v.tolist() for k, v in self.params.items()},
                    "fitted": self._fitted,
                },
                f,
            )

    def load(self, path: str):
        """Load scaler parameters."""
        with open(path, "r") as f:
            data = json.load(f)
        self.method = data["method"]
        self.params = {k: np.array(v) for k, v in data["params"].items()}
        self._fitted = data["fitted"]


class LSTMTrainer:
    """
    Training workflow manager for Multi-Horizon LSTM.

    Handles:
    - Data preparation with jumping windows
    - Feature scaling
    - Model training with callbacks
    - Evaluation and metrics
    - Checkpointing
    """

    def __init__(
        self,
        model: Optional[MultiHorizonLSTM] = None,
        config: Optional[TrainingConfig] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: MultiHorizonLSTM instance (or will create one)
            config: Training configuration
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.checkpoint_dir = checkpoint_dir

        self.scaler = FeatureScaler(method="zscore")
        self.window_generator = None
        self.history = {}
        self.metrics = {}

    def prepare_data(
        self,
        data: np.ndarray,
        feature_cols: Optional[List[int]] = None,
        close_col: int = 3,
        horizons: Optional[List[HorizonConfig]] = None,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare data for training.

        Args:
            data: Raw data array (n_samples, n_features)
            feature_cols: Indices of feature columns
            close_col: Index of close price for returns calculation
            horizons: Prediction horizons

        Returns:
            (X_train, y_train, X_val, y_val) with scaled features
        """
        self.window_generator = JumpingWindowGenerator(
            window_size=self.config.sequence_length,
            prediction_horizon=21,  # Default, will use max of horizons
            step_size=self.config.window_step_size,
            test_size=self.config.validation_split,
        )

        # Generate multi-horizon windows
        X_train, y_train, X_val, y_val = (
            self.window_generator.generate_multi_horizon_windows(
                data=data,
                horizons=horizons,
                feature_cols=feature_cols,
                close_col=close_col,
            )
        )

        # Scale features
        if fit_scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            if not self.scaler._fitted:
                raise RuntimeError("Scaler must be fitted before transform")
            X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        return X_train_scaled, y_train, X_val_scaled, y_val

    def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        verbose: int = 1,
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets (dict by horizon)
            X_val: Validation features
            y_val: Validation targets
            verbose: Verbosity level

        Returns:
            Training history
        """
        # Initialize model if needed
        if self.model is None:
            input_dim = X_train.shape[-1]
            self.model = MultiHorizonLSTM(
                input_dim=input_dim, sequence_length=self.config.sequence_length
            )

        # Build and compile (do not rebuild if already built)
        if self.model.model is None:
            self.model.build_model()
        self.model.compile_model(learning_rate=self.config.learning_rate)

        if verbose:
            print(f"Model built with input shape: {X_train.shape}")
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")

        # Train
        self.history = self.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            early_stopping_patience=self.config.early_stopping_patience,
            reduce_lr_patience=self.config.reduce_lr_patience,
            verbose=verbose,
        )

        return self.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray],
        use_uncertainty: bool = True,
        n_mc_iterations: int = 100,
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test targets
            use_uncertainty: Whether to use MC dropout
            n_mc_iterations: Number of MC iterations

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first")

        # Make predictions
        if use_uncertainty:
            predictions = self.model.predict_with_uncertainty(X_test, n_mc_iterations)
        else:
            predictions = self.model.predict(X_test)

        # Calculate metrics for each horizon
        metrics = {}

        for horizon_name, y_true in y_test.items():
            y_pred = predictions[horizon_name]["mean"]
            y_std = predictions[horizon_name]["std"]

            # Basic metrics
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(mse)

            # Directional accuracy
            direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

            # Calibration (% within 1 std)
            within_1std = np.mean(np.abs(y_true - y_pred) <= y_std)
            within_2std = np.mean(np.abs(y_true - y_pred) <= 2 * y_std)

            # Information coefficient (correlation)
            if np.std(y_true) > 0 and np.std(y_pred) > 0:
                ic = np.corrcoef(y_true, y_pred)[0, 1]
            else:
                ic = 0.0

            metrics[horizon_name] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "direction_accuracy": float(direction_accuracy),
                "information_coefficient": float(ic),
                "within_1std": float(within_1std),
                "within_2std": float(within_2std),
                "mean_uncertainty": float(np.mean(y_std)),
            }

        self.metrics = metrics
        return metrics

    def cross_validate(
        self,
        data: np.ndarray,
        n_folds: int = 5,
        feature_cols: Optional[List[int]] = None,
        close_col: int = 3,
        verbose: int = 1,
    ) -> Dict:
        """
        Time series cross-validation using expanding window.

        Args:
            data: Full dataset
            n_folds: Number of folds
            feature_cols: Feature column indices
            close_col: Close price column index
            verbose: Verbosity level

        Returns:
            Cross-validation results
        """
        n_samples = len(data)
        min_train_size = n_samples // (n_folds + 1)

        fold_metrics = []

        for fold in range(n_folds):
            if verbose:
                print(f"\n=== Fold {fold + 1}/{n_folds} ===")

            # Expanding window: train on all data up to fold point
            train_end = min_train_size * (fold + 2)
            test_end = min(train_end + min_train_size, n_samples)

            train_data = data[:train_end]
            test_data = data[train_end:test_end]

            if len(test_data) < self.config.sequence_length + 126:  # Min required
                if verbose:
                    print(f"Skipping fold {fold + 1}: insufficient test data")
                continue

            # Prepare data for this fold
            X_train, y_train, X_val, y_val = self.prepare_data(
                train_data, feature_cols=feature_cols, close_col=close_col
            )

            # Create fresh model for this fold
            self.model = None

            # Train
            self.train(X_train, y_train, X_val, y_val, verbose=max(0, verbose - 1))

            # Evaluate on test data
            X_test, y_test, _, _ = JumpingWindowGenerator(
                window_size=self.config.sequence_length,
                test_size=0.99,  # Use almost all for testing
            ).generate_multi_horizon_windows(
                test_data, feature_cols=feature_cols, close_col=close_col
            )

            X_test_scaled = self.scaler.transform(X_test)
            fold_metric = self.evaluate(X_test_scaled, y_test, use_uncertainty=True)
            fold_metrics.append(fold_metric)

            if verbose:
                for horizon, metrics in fold_metric.items():
                    print(
                        f"  {horizon}: IC={metrics['information_coefficient']:.4f}, "
                        f"Dir={metrics['direction_accuracy']:.2%}"
                    )

        # Aggregate metrics across folds
        aggregated = {}
        if fold_metrics:
            for horizon in fold_metrics[0].keys():
                aggregated[horizon] = {}
                for metric_name in fold_metrics[0][horizon].keys():
                    values = [fm[horizon][metric_name] for fm in fold_metrics]
                    aggregated[horizon][f"{metric_name}_mean"] = float(np.mean(values))
                    aggregated[horizon][f"{metric_name}_std"] = float(np.std(values))

        return {
            "fold_metrics": fold_metrics,
            "aggregated": aggregated,
            "n_folds": len(fold_metrics),
        }

    def save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        name = name or f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save model
        model_path = os.path.join(self.checkpoint_dir, f"{name}_model.keras")
        self.model.save(model_path)

        # Save scaler
        scaler_path = os.path.join(self.checkpoint_dir, f"{name}_scaler.json")
        self.scaler.save(scaler_path)

        # Save metrics and config
        meta_path = os.path.join(self.checkpoint_dir, f"{name}_meta.json")
        with open(meta_path, "w") as f:
            model_info = {}
            if self.model is not None:
                model_info = {
                    "input_dim": self.model.input_dim,
                    "sequence_length": self.model.seq_len,
                    "horizons": [
                        {"name": h.name, "days": h.days, "weight": h.weight}
                        for h in self.model.horizons
                    ],
                }
            json.dump(
                {
                    "config": self.config.__dict__,
                    "metrics": self.metrics,
                    "model_info": model_info,
                    "history": {
                        k: [float(v) for v in vals] for k, vals in self.history.items()
                    }
                    if self.history
                    else {},
                },
                f,
                indent=2,
            )

        return name

    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set")

        meta_path = os.path.join(self.checkpoint_dir, f"{name}_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        model_info = meta.get("model_info") or {}
        horizons_data = model_info.get("horizons") or None
        horizons = None
        if horizons_data:
            horizons = [HorizonConfig(**h) for h in horizons_data]
        input_dim = model_info.get("input_dim") or 1
        seq_len = model_info.get("sequence_length") or meta.get("config", {}).get(
            "sequence_length", 90
        )

        # Load model
        model_path = os.path.join(self.checkpoint_dir, f"{name}_model.keras")
        self.model = MultiHorizonLSTM(
            input_dim=input_dim,
            sequence_length=seq_len,
            horizons=horizons,
        )
        self.model.load(model_path)

        # Load scaler
        scaler_path = os.path.join(self.checkpoint_dir, f"{name}_scaler.json")
        self.scaler.load(scaler_path)

        self.config = TrainingConfig(**meta["config"])
        self.metrics = meta["metrics"]
        self.history = meta.get("history", {})
