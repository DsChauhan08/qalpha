"""
Visualization Module.

Provides the Streamlit-based trading dashboard for Quantum Alpha.
"""

from __future__ import annotations

from quantum_alpha.visualization.dashboard import (
    TradingDashboard,
    start_dashboard,
)

__all__ = [
    "TradingDashboard",
    "start_dashboard",
]
