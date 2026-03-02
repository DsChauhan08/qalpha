"""Dual-run stack selector for allocation mode switching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np


@dataclass
class DualRunSelector:
    window_cycles: int = 20
    switch_margin: float = 0.02
    min_hold_cycles: int = 5
    current_stack: str = "static"
    hold_counter: int = 0
    history_static: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    history_regime: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        self.window_cycles = max(3, int(self.window_cycles))
        self.switch_margin = float(max(self.switch_margin, 0.0))
        self.min_hold_cycles = max(1, int(self.min_hold_cycles))
        self.history_static = deque(maxlen=self.window_cycles)
        self.history_regime = deque(maxlen=self.window_cycles)

    def update_and_select(self, static_score: float, regime_score: float) -> Dict[str, object]:
        self.history_static.append(float(static_score))
        self.history_regime.append(float(regime_score))

        avg_static = float(np.mean(self.history_static)) if self.history_static else 0.0
        avg_regime = float(np.mean(self.history_regime)) if self.history_regime else 0.0

        preferred = "static" if avg_static >= avg_regime else "regime"
        improvement = abs(avg_regime - avg_static)

        switched = False
        if preferred != self.current_stack:
            if self.hold_counter >= self.min_hold_cycles and improvement > self.switch_margin:
                self.current_stack = preferred
                self.hold_counter = 0
                switched = True
            else:
                self.hold_counter += 1
        else:
            self.hold_counter += 1

        return {
            "chosen_stack": self.current_stack,
            "preferred_stack": preferred,
            "avg_static": avg_static,
            "avg_regime": avg_regime,
            "improvement": improvement,
            "switched": switched,
            "hold_counter": self.hold_counter,
        }


__all__ = ["DualRunSelector"]
