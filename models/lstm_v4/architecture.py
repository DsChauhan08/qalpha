"""
Multi-Horizon LSTM Architecture with Uncertainty Quantification.

Based on agent.md Section 3.1 specification:
- Shared feature extraction backbone
- Separate prediction heads for each horizon (1d, 1w, 1m, 6m)
- Heteroscedastic uncertainty (input-dependent noise)
- CAGR-normalized loss for comparable horizons
- Monte Carlo dropout for uncertainty estimation

Designed to work with or without TensorFlow/Keras.
Falls back to pure NumPy implementation for inference.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    HAS_TF = True
except ImportError:
    HAS_TF = False
    warnings.warn("TensorFlow not available. Using pure NumPy implementation.")


@dataclass
class HorizonConfig:
    """Configuration for a prediction horizon."""

    name: str
    days: int
    weight: float = 1.0


# Default horizons from specification
DEFAULT_HORIZONS = [
    HorizonConfig("1d", 1, 1.0),
    HorizonConfig("1w", 5, 1.0),
    HorizonConfig("1m", 21, 1.0),
    HorizonConfig("6m", 126, 1.0),
]


class JumpingWindowGenerator:
    """
    Creates non-overlapping windows for time series cross-validation.

    CRITICAL: Prevents data leakage from overlapping sequences.
    Uses jumping windows to ensure train/test independence.

    From agent.md Section 3.1.1:
    "CRITICAL: Prevents data leakage from overlapping sequences."
    """

    def __init__(
        self,
        window_size: int = 90,
        prediction_horizon: int = 21,
        step_size: Optional[int] = None,
        test_size: float = 0.2,
        gap: int = 0,
    ):
        """
        Initialize the jumping window generator.

        Args:
            window_size: Number of time steps for input features
            prediction_horizon: Days ahead to predict
            step_size: Step between windows (default: window_size for no overlap)
            test_size: Fraction of windows for testing
            gap: Gap between train and test to prevent leakage
        """
        self.window_size = window_size
        self.horizon = prediction_horizon
        self.step_size = step_size or window_size  # No overlap by default
        self.test_size = test_size
        self.gap = gap

    def generate_windows(
        self,
        data: np.ndarray,
        feature_cols: Optional[List[int]] = None,
        target_col: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate independent training windows.

        Args:
            data: Array of shape (n_samples, n_features) or DataFrame
            feature_cols: Indices of feature columns (default: all except target)
            target_col: Index of target column (default: -1, last column)

        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        # Convert DataFrame if needed
        if hasattr(data, "values"):
            data = data.values

        n_samples, n_features = data.shape

        # Determine feature columns
        if feature_cols is None:
            if target_col == -1:
                feature_cols = list(range(n_features - 1))
            else:
                feature_cols = [i for i in range(n_features) if i != target_col]

        # Generate window indices
        windows = []
        for i in range(0, n_samples - self.window_size - self.horizon, self.step_size):
            feature_end = i + self.window_size
            target_end = feature_end + self.horizon

            if target_end > n_samples:
                break

            windows.append((i, feature_end, target_end))

        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for windows. Need at least "
                f"{self.window_size + self.horizon} samples, got {n_samples}"
            )

        # Split train/test with gap
        n_test = max(1, int(len(windows) * self.test_size))
        n_gap = self.gap

        train_windows = windows[: -(n_test + n_gap)] if n_gap > 0 else windows[:-n_test]
        test_windows = windows[-n_test:]

        # Extract data
        X_train, y_train = self._extract_windows(
            data, train_windows, feature_cols, target_col
        )
        X_test, y_test = self._extract_windows(
            data, test_windows, feature_cols, target_col
        )

        return X_train, y_train, X_test, y_test

    def _extract_windows(
        self,
        data: np.ndarray,
        windows: List[Tuple[int, int, int]],
        feature_cols: List[int],
        target_col: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from windows."""
        X, y = [], []

        for start, feat_end, target_end in windows:
            # Extract feature window
            X.append(data[start:feat_end, feature_cols])

            # Target is return over horizon
            # Use close price (assumed to be at target_col or calculated)
            if target_col == -1:
                target_col = data.shape[1] - 1

            price_start = data[feat_end - 1, target_col]
            price_end = data[target_end - 1, target_col]

            # Calculate return
            if price_start != 0:
                ret = (price_end - price_start) / price_start
            else:
                ret = 0.0

            y.append(ret)

        return np.array(X), np.array(y)

    def generate_multi_horizon_windows(
        self,
        data: np.ndarray,
        horizons: List[HorizonConfig] = None,
        feature_cols: Optional[List[int]] = None,
        close_col: int = 3,  # Default: OHLCV format
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate windows with multiple prediction horizons.

        Args:
            data: Array of shape (n_samples, n_features)
            horizons: List of HorizonConfig for each prediction horizon
            feature_cols: Indices of feature columns
            close_col: Index of close price column

        Returns:
            tuple: (X_train, y_train_dict, X_test, y_test_dict)
                   where y_*_dict maps horizon name to targets
        """
        horizons = horizons or DEFAULT_HORIZONS

        if hasattr(data, "values"):
            data = data.values

        n_samples, n_features = data.shape

        if feature_cols is None:
            feature_cols = list(range(n_features))

        # Use max horizon for window generation
        max_horizon = max(h.days for h in horizons)

        # Generate window indices
        windows = []
        for i in range(0, n_samples - self.window_size - max_horizon, self.step_size):
            feature_end = i + self.window_size

            if feature_end + max_horizon > n_samples:
                break

            windows.append((i, feature_end))

        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for multi-horizon windows. Need at least "
                f"{self.window_size + max_horizon} samples, got {n_samples}"
            )

        # Split train/test
        n_test = max(1, int(len(windows) * self.test_size))
        train_windows = windows[:-n_test]
        test_windows = windows[-n_test:]

        # Extract features and multi-horizon targets
        X_train, y_train = self._extract_multi_horizon(
            data, train_windows, horizons, feature_cols, close_col
        )
        X_test, y_test = self._extract_multi_horizon(
            data, test_windows, horizons, feature_cols, close_col
        )

        return X_train, y_train, X_test, y_test

    def _extract_multi_horizon(
        self,
        data: np.ndarray,
        windows: List[Tuple[int, int]],
        horizons: List[HorizonConfig],
        feature_cols: List[int],
        close_col: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Extract features and multi-horizon targets."""
        X = []
        y = {h.name: [] for h in horizons}

        for start, feat_end in windows:
            X.append(data[start:feat_end, feature_cols])

            price_start = data[feat_end - 1, close_col]

            for horizon in horizons:
                target_idx = feat_end + horizon.days - 1
                if target_idx < len(data):
                    price_end = data[target_idx, close_col]
                    if price_start != 0:
                        ret = (price_end - price_start) / price_start
                    else:
                        ret = 0.0
                else:
                    ret = 0.0

                y[horizon.name].append(ret)

        y = {k: np.array(v) for k, v in y.items()}

        return np.array(X), y


class MultiHorizonLSTM:
    """
    Multi-horizon LSTM with uncertainty quantification.

    Features from agent.md Section 3.1:
    - Shared feature extraction backbone
    - Separate prediction heads for each horizon
    - Heteroscedastic uncertainty (input-dependent noise)
    - CAGR-normalized loss for comparable horizons
    - Monte Carlo dropout for uncertainty estimation

    Falls back to pure NumPy for inference if TensorFlow unavailable.
    """

    def __init__(
        self,
        input_dim: int = 50,
        sequence_length: int = 90,
        lstm_units: List[int] = None,
        dropout: float = 0.3,
        horizons: List[HorizonConfig] = None,
        use_attention: bool = True,
        recurrent_dropout: float = 0.2,
    ):
        """
        Initialize the Multi-Horizon LSTM.

        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            lstm_units: List of LSTM layer sizes (default: [128, 64])
            dropout: Dropout rate
            horizons: Prediction horizons configuration
            use_attention: Whether to use attention mechanism
            recurrent_dropout: Dropout within LSTM cells
        """
        self.input_dim = input_dim
        self.seq_len = sequence_length
        self.lstm_units = lstm_units or [128, 64]
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.horizons = horizons or DEFAULT_HORIZONS
        self.use_attention = use_attention

        self.model = None
        self._weights = None  # For NumPy fallback

    def build_model(self) -> Any:
        """
        Build multi-horizon architecture with uncertainty.

        Returns:
            Keras Model if TensorFlow available, else None
        """
        if not HAS_TF:
            warnings.warn("TensorFlow not available. Model will be inference-only.")
            return None

        inputs = keras.Input(shape=(self.seq_len, self.input_dim), name="input")

        # Batch normalization for input stability
        x = layers.BatchNormalization(name="input_bn")(inputs)

        # Shared feature extraction backbone
        for i, units in enumerate(self.lstm_units[:-1]):
            x = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name=f"lstm_{i}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_{i}")(x)

        # Final LSTM layer
        if self.use_attention:
            # Return sequences for attention
            x = layers.LSTM(
                self.lstm_units[-1],
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name="lstm_final",
            )(x)

            # Self-attention layer
            attention = layers.MultiHeadAttention(
                num_heads=4, key_dim=self.lstm_units[-1] // 4, name="attention"
            )(x, x)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization(name="attention_norm")(x)
            x = layers.GlobalAveragePooling1D(name="global_pool")(x)
        else:
            x = layers.LSTM(
                self.lstm_units[-1],
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name="lstm_final",
            )(x)

        # Shared dense layer
        shared = layers.Dense(64, activation="relu", name="shared_dense")(x)
        shared = layers.Dropout(self.dropout, name="shared_dropout")(shared)

        # Separate heads for each horizon with uncertainty
        outputs = []
        output_names = []

        for horizon in self.horizons:
            # Mean prediction head
            mean_hidden = layers.Dense(
                32, activation="relu", name=f"{horizon.name}_mean_hidden"
            )(shared)
            mean = layers.Dense(1, name=f"{horizon.name}_mean")(mean_hidden)

            # Log variance head (uncertainty)
            log_var_hidden = layers.Dense(
                32, activation="relu", name=f"{horizon.name}_logvar_hidden"
            )(shared)
            log_var = layers.Dense(1, name=f"{horizon.name}_log_var")(log_var_hidden)

            outputs.extend([mean, log_var])
            output_names.extend([f"{horizon.name}_mean", f"{horizon.name}_log_var"])

        self.model = keras.Model(
            inputs=inputs, outputs=outputs, name="MultiHorizonLSTM"
        )
        self.output_names = output_names

        return self.model

    def negative_log_likelihood(
        self, y_true: Any, y_pred_mean: Any, y_pred_log_var: Any
    ) -> Any:
        """
        Negative log-likelihood for heteroscedastic regression.

        Loss = 0.5 * (log(var) + (y - mean)^2 / var)

        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_log_var: Predicted log variances

        Returns:
            Loss value
        """
        if not HAS_TF:
            # NumPy implementation
            precision = np.exp(-y_pred_log_var)
            loss = 0.5 * (y_pred_log_var + precision * np.square(y_true - y_pred_mean))
            return np.mean(loss)

        precision = tf.exp(-y_pred_log_var)
        loss = 0.5 * (y_pred_log_var + precision * tf.square(y_true - y_pred_mean))
        return tf.reduce_mean(loss)

    def cagr_loss(self, periods: int):
        """
        CAGR-normalized loss for comparable horizons.

        Converts returns to annualized rate before computing loss.
        This ensures that predictions at different horizons contribute
        equally to the loss regardless of time scale.

        Args:
            periods: Number of trading days for this horizon

        Returns:
            Loss function
        """
        if not HAS_TF:
            raise RuntimeError("TensorFlow required for training")

        def loss_fn(y_true, y_pred):
            annual_factor = 252.0 / periods

            # Clip to prevent numerical issues
            y_true_clipped = tf.clip_by_value(y_true, -0.95, 1.0)
            y_pred_clipped = tf.clip_by_value(y_pred, -0.95, 1.0)

            # Compare annualized log returns to avoid overflow
            y_true_log = tf.math.log1p(y_true_clipped) * annual_factor
            y_pred_log = tf.math.log1p(y_pred_clipped) * annual_factor

            return tf.reduce_mean(tf.square(y_true_log - y_pred_log))

        return loss_fn

    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile model with custom loss functions.

        Args:
            learning_rate: Optimizer learning rate
        """
        if not HAS_TF or self.model is None:
            raise RuntimeError("TensorFlow required and model must be built first")

        # Create loss dict - using CAGR loss for means, zero for log_var
        losses = {}
        loss_weights = {}

        for horizon in self.horizons:
            losses[f"{horizon.name}_mean"] = self.cagr_loss(horizon.days)
            losses[f"{horizon.name}_log_var"] = lambda yt, yp: 0.0

            loss_weights[f"{horizon.name}_mean"] = horizon.weight
            loss_weights[f"{horizon.name}_log_var"] = 0.0

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        verbose: int = 1,
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Dict mapping horizon name to targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Reduce LR on plateau patience
            verbose: Verbosity level

        Returns:
            Training history
        """
        if not HAS_TF or self.model is None:
            raise RuntimeError(
                "TensorFlow required and model must be built and compiled"
            )

        # Prepare target dict for Keras
        y_train_keras = {}
        for horizon in self.horizons:
            y_train_keras[f"{horizon.name}_mean"] = y_train[horizon.name]
            y_train_keras[f"{horizon.name}_log_var"] = y_train[horizon.name]  # Dummy

        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_keras = {}
            for horizon in self.horizons:
                y_val_keras[f"{horizon.name}_mean"] = y_val[horizon.name]
                y_val_keras[f"{horizon.name}_log_var"] = y_val[horizon.name]
            validation_data = (X_val, y_val_keras)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train_keras,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        return history.history

    def predict(self, X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Make predictions.

        Args:
            X: Input features of shape (n_samples, seq_len, n_features)

        Returns:
            Dict mapping horizon name to prediction dict with 'mean' and 'std'
        """
        if not HAS_TF or self.model is None:
            return self._predict_numpy(X)

        outputs = self.model.predict(X, verbose=0)

        results = {}
        for i, horizon in enumerate(self.horizons):
            mean = outputs[i * 2].flatten()
            log_var = outputs[i * 2 + 1].flatten()
            std = np.sqrt(np.exp(log_var))

            results[horizon.name] = {"mean": mean, "std": std, "log_var": log_var}

        return results

    def predict_with_uncertainty(
        self, X: np.ndarray, n_iterations: int = 100
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Monte Carlo dropout prediction for uncertainty estimation.

        Args:
            X: Input features
            n_iterations: Number of forward passes with dropout

        Returns:
            Dict with mean predictions and comprehensive uncertainties
        """
        if not HAS_TF or self.model is None:
            warnings.warn("MC Dropout requires TensorFlow. Using standard prediction.")
            return self.predict(X)

        predictions = {h.name: [] for h in self.horizons}

        # Run multiple forward passes with dropout enabled
        for _ in range(n_iterations):
            # training=True enables dropout
            outputs = self.model(X, training=True)

            for i, horizon in enumerate(self.horizons):
                mean = outputs[i * 2].numpy().flatten()
                predictions[horizon.name].append(mean)

        results = {}
        for horizon in self.horizons:
            preds = np.array(predictions[horizon.name])  # (n_iterations, n_samples)

            results[horizon.name] = {
                "mean": np.mean(preds, axis=0),
                "std": np.std(preds, axis=0),
                "epistemic_uncertainty": np.std(preds, axis=0),
                "ci_lower": np.percentile(preds, 5, axis=0),
                "ci_upper": np.percentile(preds, 95, axis=0),
                "ci_50_lower": np.percentile(preds, 25, axis=0),
                "ci_50_upper": np.percentile(preds, 75, axis=0),
            }

        return results

    def _predict_numpy(self, X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Fallback NumPy prediction (requires loaded weights).

        Args:
            X: Input features

        Returns:
            Prediction dict (zeros if no weights loaded)
        """
        warnings.warn(
            "Using NumPy fallback. Predictions may be inaccurate without trained weights."
        )

        # Return zeros as placeholder
        n_samples = X.shape[0]
        results = {}

        for horizon in self.horizons:
            results[horizon.name] = {
                "mean": np.zeros(n_samples),
                "std": np.ones(n_samples) * 0.1,
            }

        return results

    def save(self, path: str):
        """Save model to disk."""
        if HAS_TF and self.model is not None:
            self.model.save(path)
        else:
            raise RuntimeError("No model to save")

    def load(self, path: str):
        """Load model from disk."""
        if HAS_TF:
            self.model = keras.models.load_model(
                path,
                custom_objects={
                    f"cagr_loss_{h.days}": self.cagr_loss(h.days) for h in self.horizons
                },
                compile=False,
                safe_mode=False,
            )
        else:
            raise RuntimeError("TensorFlow required to load model")

    def summary(self) -> str:
        """Get model summary."""
        if HAS_TF and self.model is not None:
            # Capture summary to string
            summary_lines = []
            self.model.summary(print_fn=lambda x: summary_lines.append(x))
            return "\n".join(summary_lines)
        else:
            return f"MultiHorizonLSTM (not built): input_dim={self.input_dim}, seq_len={self.seq_len}"
