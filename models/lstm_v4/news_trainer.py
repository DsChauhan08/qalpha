"""
News-Driven LSTM Trainer (PyTorch).

Handles the full pipeline:
1. Fetch price data
2. Build sentiment proxy features
3. Create labelled training samples (buy/hold/sell)
4. Train the NewsDrivenLSTM
5. Evaluate and save checkpoint
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from quantum_alpha.data.collectors.news_collector import (
    NewsCollector,
    SENTIMENT_FEATURE_COLS,
)
from quantum_alpha.models.lstm_v4.news_lstm import NewsDrivenLSTM, NewsLSTMConfig


class NewsLSTMTrainer:
    """
    End-to-end trainer for the News-Driven LSTM.

    Pipeline:
    1. Build sentiment features from price data
    2. Label samples: buy/hold/sell based on forward returns
    3. Create windowed sequences
    4. Scale features
    5. Train model
    6. Evaluate on held-out period
    """

    def __init__(
        self,
        config: NewsLSTMConfig = None,
        checkpoint_dir: str = None,
        buy_threshold: float = 0.005,
        sell_threshold: float = -0.005,
    ):
        self.config = config or NewsLSTMConfig()
        self.checkpoint_dir = checkpoint_dir or str(
            Path(__file__).parent.parent / "checkpoints" / "news_lstm"
        )
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.model: Optional[NewsDrivenLSTM] = None
        self.scaler_params: Dict = {}
        self.history: Dict = {}
        self.metrics: Dict = {}
        self.device = NewsDrivenLSTM._select_device()

    def prepare_data(
        self,
        price_df: pd.DataFrame,
        symbol: str = "SPY",
        forward_period: int = 1,
        val_split: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Returns:
            X_train, y_signal_train, y_conf_train,
            X_val, y_signal_val, y_conf_val
        """
        collector = NewsCollector()
        features_df = collector.build_training_features(price_df, symbol)

        # Create labels from forward returns
        features_df["forward_return"] = (
            features_df["close"].pct_change(forward_period).shift(-forward_period)
        )
        features_df = features_df.dropna()

        n_classes = self.config.n_classes

        if n_classes == 2:
            # Binary: 0=down, 1=up
            labels = (features_df["forward_return"].values > 0).astype(np.int64)
        else:
            # Ternary: 0=sell, 1=hold, 2=buy
            labels = np.ones(len(features_df), dtype=np.int64)
            labels[features_df["forward_return"].values > self.buy_threshold] = 2
            labels[features_df["forward_return"].values < self.sell_threshold] = 0

        # Confidence target: abs(forward_return) clipped to [0, 1]
        confidence = np.clip(np.abs(features_df["forward_return"].values) * 20, 0, 1)

        # Extract feature matrix
        feature_cols = SENTIMENT_FEATURE_COLS
        available = [c for c in feature_cols if c in features_df.columns]
        if len(available) < 10:
            raise ValueError(
                f"Only {len(available)} features available, need at least 10"
            )

        X_raw = features_df[available].values

        # Z-score normalize
        self.scaler_params = {
            "mean": np.nanmean(X_raw, axis=0),
            "std": np.nanstd(X_raw, axis=0),
        }
        self.scaler_params["std"] = np.where(
            self.scaler_params["std"] < 1e-8, 1.0, self.scaler_params["std"]
        )
        X_scaled = (X_raw - self.scaler_params["mean"]) / self.scaler_params["std"]
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Create windowed sequences
        seq_len = self.config.sequence_length
        n_samples = len(X_scaled) - seq_len

        X = np.zeros((n_samples, seq_len, X_scaled.shape[1]), dtype=np.float32)
        y_signal = np.zeros(n_samples, dtype=np.int64)
        y_conf = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X[i] = X_scaled[i : i + seq_len]
            y_signal[i] = labels[i + seq_len]
            y_conf[i] = confidence[i + seq_len]

        # Update config with actual feature count
        # The feature matrix has sentiment cols first, then price context cols.
        # Count how many of each are actually present.
        from quantum_alpha.data.collectors.news_collector import NewsCollector as _NC

        _sentiment_cols = [
            c
            for c in available
            if c
            in [
                "overnight_gap",
                "overnight_gap_zscore",
                "range_surprise",
                "volume_surprise",
                "returns_accel",
                "sentiment_proxy",
                "sentiment_ma3",
                "sentiment_ma7",
                "sentiment_momentum",
                "news_intensity",
                "fear_greed",
                "return_zscore",
                "vol_regime",
                "trend_strength",
                "vol_price_div",
                "gap_fill_rate",
            ]
        ]
        n_sent = len(_sentiment_cols)
        self.config.n_sentiment_features = n_sent
        self.config.n_price_features = X.shape[2] - n_sent

        # Train/val split (chronological)
        n_val = max(1, int(len(X) * val_split))
        X_train = X[:-n_val]
        y_signal_train = y_signal[:-n_val]
        y_conf_train = y_conf[:-n_val]
        X_val = X[-n_val:]
        y_signal_val = y_signal[-n_val:]
        y_conf_val = y_conf[-n_val:]

        # Print class distribution
        n_classes = self.config.n_classes
        for name, y in [("Train", y_signal_train), ("Val", y_signal_val)]:
            total = len(y)
            if n_classes == 2:
                downs = (y == 0).sum()
                ups = (y == 1).sum()
                print(
                    f"{name}: {total} samples | "
                    f"down={downs} ({100 * downs / total:.1f}%) | "
                    f"up={ups} ({100 * ups / total:.1f}%)"
                )
            else:
                sells = (y == 0).sum()
                holds = (y == 1).sum()
                buys = (y == 2).sum()
                print(
                    f"{name}: {total} samples | "
                    f"sell={sells} ({100 * sells / total:.1f}%) | "
                    f"hold={holds} ({100 * holds / total:.1f}%) | "
                    f"buy={buys} ({100 * buys / total:.1f}%)"
                )

        return X_train, y_signal_train, y_conf_train, X_val, y_signal_val, y_conf_val

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """
        Focal loss with label smoothing.

        Focal: down-weights easy examples, focuses on hard ones.
        Label smoothing: prevents overconfident wrong predictions.
        """
        ce = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        pt = torch.exp(-ce)  # probability of correct class
        focal = ((1 - pt) ** gamma) * ce
        return focal.mean()

    def train(
        self,
        X_train: np.ndarray,
        y_signal_train: np.ndarray,
        y_conf_train: np.ndarray,
        X_val: np.ndarray = None,
        y_signal_val: np.ndarray = None,
        y_conf_val: np.ndarray = None,
        verbose: int = 1,
    ) -> Dict:
        """Train the model."""
        cfg = self.config

        # Class weights for imbalance
        unique, counts = np.unique(y_signal_train, return_counts=True)
        total = len(y_signal_train)
        n_classes = cfg.n_classes
        weight_arr = np.ones(n_classes, dtype=np.float32)
        for cls, cnt in zip(unique, counts):
            if int(cls) < n_classes:
                weight_arr[int(cls)] = total / (len(unique) * cnt)
        class_weights = torch.tensor(weight_arr, device=self.device)

        # Build model
        self.model = NewsDrivenLSTM(cfg)
        self.model.build_model()
        model = self.model.model

        if verbose:
            print(self.model.summary())
            print(f"\nClass weights: {weight_arr.tolist()}")
            print(f"Device: {self.device}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        # Build DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_signal_train, dtype=torch.long),
            torch.tensor(y_conf_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_signal_val, dtype=torch.long),
                torch.tensor(y_conf_val, dtype=torch.float32),
            )
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        # Training loop
        history = {"loss": [], "val_loss": [], "signal_acc": [], "val_signal_acc": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.epochs):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for X_b, y_sig_b, y_conf_b in train_loader:
                X_b = X_b.to(self.device)
                y_sig_b = y_sig_b.to(self.device)
                y_conf_b = y_conf_b.to(self.device)

                optimizer.zero_grad()
                logits, conf_pred = model(X_b)

                loss_signal = self._focal_loss(logits, y_sig_b, class_weights)
                loss_conf = F.mse_loss(conf_pred.squeeze(-1), y_conf_b)
                loss = loss_signal + 0.2 * loss_conf

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item() * X_b.size(0)
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y_sig_b).sum().item()
                epoch_total += X_b.size(0)

            train_loss = epoch_loss / epoch_total
            train_acc = epoch_correct / epoch_total
            history["loss"].append(train_loss)
            history["signal_acc"].append(train_acc)

            # --- Validate ---
            val_loss = train_loss
            val_acc = train_acc
            if val_loader is not None:
                model.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for X_b, y_sig_b, y_conf_b in val_loader:
                        X_b = X_b.to(self.device)
                        y_sig_b = y_sig_b.to(self.device)
                        y_conf_b = y_conf_b.to(self.device)

                        logits, conf_pred = model(X_b)
                        l_sig = self._focal_loss(logits, y_sig_b, class_weights)
                        l_conf = F.mse_loss(conf_pred.squeeze(-1), y_conf_b)
                        v_loss += (l_sig + 0.2 * l_conf).item() * X_b.size(0)
                        v_correct += (logits.argmax(dim=1) == y_sig_b).sum().item()
                        v_total += X_b.size(0)

                val_loss = v_loss / v_total
                val_acc = v_correct / v_total

            history["val_loss"].append(val_loss)
            history["val_signal_acc"].append(val_acc)

            scheduler.step(epoch)

            if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                    f"lr={lr:.2e}"
                )

            # Log prediction distribution every 20 epochs
            if verbose >= 1 and (epoch % 20 == 0 or epoch == cfg.epochs - 1):
                model.eval()
                with torch.no_grad():
                    sample_X = torch.tensor(
                        X_val[:100] if X_val is not None else X_train[:100],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    sample_logits, sample_conf = model(sample_X)
                    sample_preds = sample_logits.argmax(dim=1).cpu().numpy()
                    sample_probs = F.softmax(sample_logits, dim=1).cpu().numpy()
                    conf_mean = sample_conf.mean().item()
                    max_prob_mean = sample_probs.max(axis=1).mean()
                    if n_classes == 2:
                        down_pct = (sample_preds == 0).mean() * 100
                        up_pct = (sample_preds == 1).mean() * 100
                        print(
                            f"  -> Pred dist: down={down_pct:.0f}% up={up_pct:.0f}% "
                            f"| avg_conf={conf_mean:.3f} "
                            f"avg_max_prob={max_prob_mean:.3f}"
                        )
                    else:
                        sell_pct = (sample_preds == 0).mean() * 100
                        hold_pct = (sample_preds == 1).mean() * 100
                        buy_pct = (sample_preds == 2).mean() * 100
                        print(
                            f"  -> Pred dist: sell={sell_pct:.0f}% hold={hold_pct:.0f}% "
                            f"buy={buy_pct:.0f}% | avg_conf={conf_mean:.3f} "
                            f"avg_max_prob={max_prob_mean:.3f}"
                        )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        model.eval()
        self.history = history
        return history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_signal_test: np.ndarray,
        y_conf_test: np.ndarray,
    ) -> Dict:
        """Evaluate model on test data."""
        if self.model is None:
            raise RuntimeError("Train model first")

        preds = self.model.predict(X_test)
        pred_signals = preds["signal"]
        trade_actions = preds["trade_action"]

        accuracy = float(np.mean(pred_signals == y_signal_test))

        n_classes = self.config.n_classes
        class_acc = {}
        if n_classes == 2:
            for cls, name in [(0, "down"), (1, "up")]:
                mask = y_signal_test == cls
                if mask.sum() > 0:
                    class_acc[name] = float(np.mean(pred_signals[mask] == cls))
                else:
                    class_acc[name] = 0.0
        else:
            for cls, name in [(0, "sell"), (1, "hold"), (2, "buy")]:
                mask = y_signal_test == cls
                if mask.sum() > 0:
                    class_acc[name] = float(np.mean(pred_signals[mask] == cls))
                else:
                    class_acc[name] = 0.0

        trade_mask = trade_actions != 0
        n_trades = int(trade_mask.sum())

        if n_trades > 0:
            if n_classes == 2:
                # Binary: 0=down(sell), 1=up(buy). trade_action: -1=sell, 1=buy
                correct_direction = (
                    (trade_actions[trade_mask] > 0) & (y_signal_test[trade_mask] == 1)
                ) | ((trade_actions[trade_mask] < 0) & (y_signal_test[trade_mask] == 0))
            else:
                correct_direction = (
                    (trade_actions[trade_mask] > 0) & (y_signal_test[trade_mask] == 2)
                ) | ((trade_actions[trade_mask] < 0) & (y_signal_test[trade_mask] == 0))
            trade_accuracy = float(correct_direction.mean())

            if n_classes == 3:
                hold_mask = y_signal_test == 1
                hold_accuracy = (
                    float((trade_actions[hold_mask] == 0).mean())
                    if hold_mask.sum() > 0
                    else 0.0
                )
            else:
                hold_accuracy = 0.0  # no hold class in binary
        else:
            trade_accuracy = 0.0
            hold_accuracy = 1.0

        selectivity = float(n_trades / len(X_test)) if len(X_test) > 0 else 0.0

        self.metrics = {
            "accuracy": accuracy,
            "class_accuracy": class_acc,
            "trade_accuracy": trade_accuracy,
            "hold_accuracy": hold_accuracy,
            "n_trades": n_trades,
            "n_samples": int(len(X_test)),
            "selectivity": selectivity,
        }

        return self.metrics

    def save_checkpoint(self, name: str = None) -> str:
        """Save model, scaler, and metadata."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        name = name or f"news_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        self.model.save(model_path)

        scaler_path = os.path.join(self.checkpoint_dir, f"{name}_scaler.json")
        with open(scaler_path, "w") as f:
            json.dump(
                {k: v.tolist() for k, v in self.scaler_params.items()},
                f,
            )

        meta_path = os.path.join(self.checkpoint_dir, f"{name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "config": self.config.__dict__,
                    "metrics": self.metrics,
                    "buy_threshold": self.buy_threshold,
                    "sell_threshold": self.sell_threshold,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

        print(f"Saved checkpoint: {name}")
        return name

    def load_checkpoint(self, name: str):
        """Load a saved checkpoint."""
        model_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        scaler_path = os.path.join(self.checkpoint_dir, f"{name}_scaler.json")
        meta_path = os.path.join(self.checkpoint_dir, f"{name}_meta.json")

        with open(meta_path) as f:
            meta = json.load(f)

        self.config = NewsLSTMConfig(**meta["config"])
        self.metrics = meta.get("metrics", {})
        self.buy_threshold = meta.get("buy_threshold", 0.005)
        self.sell_threshold = meta.get("sell_threshold", -0.005)

        self.model = NewsDrivenLSTM(self.config)
        self.model.load(model_path)

        with open(scaler_path) as f:
            data = json.load(f)
        self.scaler_params = {k: np.array(v) for k, v in data.items()}
