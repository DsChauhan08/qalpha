"""Contracts for portfolio allocation engine inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


@dataclass
class AllocatorInput:
    prices: pd.DataFrame
    returns: pd.DataFrame
    features: Optional[pd.DataFrame]
    constraints: Dict[str, float | bool]
    benchmark: Optional[pd.Series]
    timestamp: datetime


@dataclass
class RiskSnapshot:
    var: float
    cvar: float
    drawdown: float
    ulcer_index: float
    vol_estimates: Dict[str, float] = field(default_factory=dict)
    net_exposure: float = 0.0
    gross_exposure: float = 0.0


@dataclass
class AllocatorOutput:
    weights: Dict[str, float]
    risk_snapshot: RiskSnapshot
    diagnostics: Dict[str, object] = field(default_factory=dict)
    chosen_stack: str = "static"


__all__ = ["AllocatorInput", "AllocatorOutput", "RiskSnapshot"]
