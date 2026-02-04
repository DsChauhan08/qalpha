"""
Online reward adapter for simple reinforcement-style signal tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class OnlineRewardAdapter:
    n_features: int
    learning_rate: float = 0.05
    l2: float = 0.001
    clip: float = 3.0
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=float)

    def predict(self, features: np.ndarray) -> float:
        if features.shape[0] != self.n_features:
            raise ValueError("Feature vector length mismatch")
        score = float(np.dot(self.weights, features))
        score = float(np.clip(score, -self.clip, self.clip))
        return float(np.tanh(score))

    def update(self, features: np.ndarray, reward: float) -> None:
        if features.shape[0] != self.n_features:
            return
        if not np.isfinite(reward):
            return
        grad = reward * features - self.l2 * self.weights
        self.weights += self.learning_rate * grad
