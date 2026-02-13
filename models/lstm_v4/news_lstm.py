"""
News-Driven LSTM Architecture (PyTorch).

A specialized LSTM that uses news sentiment as the PRIMARY signal
for trade decisions.  Unlike the general MultiHorizonLSTM which
predicts returns from technical indicators, this model:

1. Takes sentiment features as primary input (16 features)
2. Uses minimal price context (5 features) as secondary
3. Outputs a TRADE SIGNAL (buy/hold/sell) not a return forecast
4. Only trades when sentiment gives a strong signal

Design rationale:
- Technical indicators have zero edge (everyone uses them)
- News sentiment gives information advantage (speed of reaction)
- LSTM learns temporal patterns in sentiment flow
- Only trading on news avoids the noise of day-to-day price action
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NewsLSTMConfig:
    """Configuration for the News-Driven LSTM."""

    # Input dimensions
    n_sentiment_features: int = 16
    n_price_features: int = 5
    sequence_length: int = 30

    # Architecture
    lstm_units: list = field(default_factory=lambda: [32])
    dropout: float = 0.10
    use_attention: bool = False
    n_classes: int = 3  # 2=binary (down/up), 3=ternary (sell/hold/buy)

    # Training
    learning_rate: float = 0.002
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 25

    # Signal thresholds
    signal_threshold: float = 0.35
    min_sentiment_strength: float = 0.2

    @property
    def total_features(self) -> int:
        return self.n_sentiment_features + self.n_price_features


class SentimentBranch(nn.Module):
    """LSTM branch for sentiment features (primary signal)."""

    def __init__(self, cfg: NewsLSTMConfig):
        super().__init__()
        self.cfg = cfg

        # Stacked LSTM (no BatchNorm — it hurts small sequential data)
        input_size = cfg.n_sentiment_features
        self.lstm_layers = nn.ModuleList()
        for units in cfg.lstm_units:
            self.lstm_layers.append(
                nn.LSTM(input_size, units, batch_first=True, dropout=0.0)
            )
            input_size = units

        self.drop = nn.Dropout(cfg.dropout)

        # Optional self-attention
        if cfg.use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=cfg.lstm_units[-1],
                num_heads=2,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(cfg.lstm_units[-1])

        self.fc = nn.Linear(cfg.lstm_units[-1], 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_sentiment_features)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.drop(x)

        if self.cfg.use_attention:
            attn_out, _ = self.attn(x, x, x)
            x = self.attn_norm(x + attn_out)
            x = x.mean(dim=1)  # global average pooling
        else:
            x = x[:, -1, :]  # last hidden state

        x = F.relu(self.fc(x))
        x = self.drop(x)
        return x


class PriceBranch(nn.Module):
    """Lightweight LSTM branch for price context (secondary)."""

    def __init__(self, cfg: NewsLSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(cfg.n_price_features, 16, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc(x))
        x = self.drop(x)
        return x


class NewsDrivenLSTMModel(nn.Module):
    """
    PyTorch dual-branch LSTM for news-driven trading.

    Outputs:
        signal_logits: (batch, 3)  — raw logits for [sell, hold, buy]
        confidence:    (batch, 1)  — sigmoid confidence score
    """

    def __init__(self, cfg: NewsLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.sentiment_branch = SentimentBranch(cfg)
        self.price_branch = PriceBranch(cfg)

        # Fusion
        self.fusion1 = nn.Linear(32 + 16, 32)
        self.fusion_drop = nn.Dropout(cfg.dropout / 2)
        self.fusion2 = nn.Linear(32, 16)

        # Heads
        self.signal_head = nn.Linear(16, cfg.n_classes)
        self.confidence_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor):
        # Split input into sentiment and price branches
        s_in = x[:, :, : self.cfg.n_sentiment_features]
        p_in = x[:, :, self.cfg.n_sentiment_features :]

        s_out = self.sentiment_branch(s_in)
        p_out = self.price_branch(p_in)

        merged = torch.cat([s_out, p_out], dim=1)
        h = F.relu(self.fusion1(merged))
        h = self.fusion_drop(h)
        h = F.relu(self.fusion2(h))

        signal_logits = self.signal_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))

        return signal_logits, confidence


class NewsDrivenLSTM:
    """
    Wrapper that manages the PyTorch model, training, and inference.

    Provides the same interface as before: build, predict, save, load.
    """

    def __init__(self, config: NewsLSTMConfig = None):
        self.config = config or NewsLSTMConfig()
        self.model: Optional[NewsDrivenLSTMModel] = None
        self.device = self._select_device()

    @staticmethod
    def _select_device() -> torch.device:
        """Select the best available device, with GPU validation."""
        if torch.cuda.is_available():
            try:
                # Verify GPU actually works (ROCm can report available but fail)
                t = torch.randn(2, 2, device="cuda")
                _ = t + t
                return torch.device("cuda")
            except RuntimeError:
                pass
        return torch.device("cpu")

    def build_model(self) -> NewsDrivenLSTMModel:
        self.model = NewsDrivenLSTMModel(self.config).to(self.device)
        return self.model

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict trade signals.

        Args:
            X: (n_samples, seq_len, total_features) numpy array

        Returns:
            dict with signal_probs, signal, confidence, trade_action
        """
        if self.model is None:
            n = X.shape[0]
            return {
                "signal_probs": np.tile([0, 1, 0], (n, 1)).astype(float),
                "signal": np.ones(n, dtype=int),
                "confidence": np.zeros(n),
                "trade_action": np.zeros(n, dtype=int),
            }

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            signal_logits, conf = self.model(X_t)
            signal_probs = F.softmax(signal_logits, dim=1).cpu().numpy()
            confidence = conf.squeeze(-1).cpu().numpy()

        signal = np.argmax(signal_probs, axis=1)
        max_prob = np.max(signal_probs, axis=1)

        trade_action = np.zeros(len(X), dtype=int)
        n_classes = signal_probs.shape[1]
        for i in range(len(X)):
            # Only gate on max_prob exceeding threshold — confidence is informational
            if max_prob[i] >= self.config.signal_threshold:
                if n_classes == 2:
                    # Binary: 0=down(sell), 1=up(buy)
                    trade_action[i] = -1 if signal[i] == 0 else 1
                else:
                    # Ternary: 0=sell, 1=hold, 2=buy
                    if signal[i] == 0:
                        trade_action[i] = -1
                    elif signal[i] == 2:
                        trade_action[i] = 1

        return {
            "signal_probs": signal_probs,
            "signal": signal,
            "confidence": confidence,
            "trade_action": trade_action,
        }

    def predict_with_uncertainty(
        self, X: np.ndarray, n_iterations: int = 50
    ) -> Dict[str, np.ndarray]:
        """MC Dropout prediction for uncertainty estimation."""
        if self.model is None:
            return self.predict(X)

        self.model.train()  # enable dropout
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        all_probs = []
        all_conf = []

        with torch.no_grad():
            for _ in range(n_iterations):
                logits, conf = self.model(X_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_conf.append(conf.squeeze(-1).cpu().numpy())

        self.model.eval()

        probs_arr = np.array(all_probs)
        conf_arr = np.array(all_conf)

        mean_probs = probs_arr.mean(axis=0)
        mean_conf = conf_arr.mean(axis=0)
        epistemic_unc = probs_arr.std(axis=0).mean(axis=-1)

        signal = np.argmax(mean_probs, axis=1)
        max_prob = np.max(mean_probs, axis=1)

        trade_action = np.zeros(len(X), dtype=int)
        n_classes = mean_probs.shape[1]
        for i in range(len(X)):
            effective_conf = mean_conf[i] * (1 - epistemic_unc[i])
            if max_prob[i] >= self.config.signal_threshold and effective_conf >= 0.4:
                if n_classes == 2:
                    trade_action[i] = -1 if signal[i] == 0 else 1
                else:
                    if signal[i] == 0:
                        trade_action[i] = -1
                    elif signal[i] == 2:
                        trade_action[i] = 1

        return {
            "signal_probs": mean_probs,
            "signal": signal,
            "confidence": mean_conf,
            "epistemic_uncertainty": epistemic_unc,
            "trade_action": trade_action,
        }

    def save(self, path: str):
        """Save model weights and config."""
        if self.model is None:
            raise RuntimeError("No model to save")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

        config_path = path.replace(".pt", "_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

    def load(self, path: str):
        """Load model weights and config."""
        config_path = path.replace(".pt", "_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                data = json.load(f)
            self.config = NewsLSTMConfig(**data)

        self.build_model()
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def summary(self) -> str:
        if self.model is not None:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            return (
                f"NewsDrivenLSTM: {total:,} params ({trainable:,} trainable)\n"
                f"  Sentiment branch: LSTM {self.config.lstm_units} + attention={self.config.use_attention}\n"
                f"  Price branch: LSTM [16]\n"
                f"  Fusion -> 3-class signal + confidence\n"
                f"  Device: {self.device}"
            )
        return f"NewsDrivenLSTM (not built): {self.config.total_features} features"
