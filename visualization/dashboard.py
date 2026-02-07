"""
Quantum Alpha Trading Dashboard.

Real-time / post-hoc trading dashboard built with Streamlit + Plotly.
Supports both live monitoring and backtest result visualization.

Usage:
    streamlit run quantum_alpha/visualization/dashboard.py

Or programmatically:
    from quantum_alpha.visualization.dashboard import start_dashboard
    start_dashboard()
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports -- only load heavy deps when dashboard actually runs
_st = None
_go = None
_make_subplots = None


def _lazy_import():
    """Import streamlit and plotly only when needed."""
    global _st, _go, _make_subplots
    if _st is None:
        import streamlit as st
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        _st = st
        _go = go
        _make_subplots = make_subplots


class TradingDashboard:
    """
    Real-time trading dashboard using Streamlit.

    Modes:
    - **backtest**: Visualize completed backtest results from a
      results dict or saved report.
    - **live**: Connect to a running paper/live trading system
      via SQLite database or API.

    Args:
        db_connection: Database connection for live mode.
            Can be None for backtest-only mode.
        backtest_results: Optional dict of backtest results to display.
    """

    def __init__(
        self,
        db_connection: Any = None,
        backtest_results: Optional[Dict] = None,
    ) -> None:
        self.db = db_connection
        self.results = backtest_results or {}

    def run(self) -> None:
        """Run the Streamlit dashboard."""
        _lazy_import()
        st = _st

        st.set_page_config(
            page_title="Quantum Alpha Dashboard",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("Quantum Alpha Trading Dashboard")

        # Sidebar
        st.sidebar.header("Controls")
        mode = st.sidebar.selectbox("Mode", ["Backtest", "Live"])
        refresh_rate = st.sidebar.slider("Refresh (seconds)", 5, 60, 30)

        if mode == "Backtest":
            self._run_backtest_view()
        else:
            self._run_live_view(refresh_rate)

    # ------------------------------------------------------------------
    # Backtest view
    # ------------------------------------------------------------------

    def _run_backtest_view(self) -> None:
        """Display backtest results."""
        st = _st

        # Metrics row
        self._display_metrics()

        # Charts row
        col1, col2 = st.columns(2)
        with col1:
            self._display_equity_curve()
        with col2:
            self._display_drawdown()

        # Detail rows
        col1, col2 = st.columns(2)
        with col1:
            self._display_positions()
        with col2:
            self._display_signals()

        # Model performance
        self._display_model_performance()

        # Gate results
        self._display_gate_results()

    # ------------------------------------------------------------------
    # Live view
    # ------------------------------------------------------------------

    def _run_live_view(self, refresh_rate: int) -> None:
        """Display live trading status."""
        st = _st
        import time

        placeholder = st.empty()

        while True:
            with placeholder.container():
                self._display_metrics()

                col1, col2 = st.columns(2)
                with col1:
                    self._display_equity_curve()
                with col2:
                    self._display_drawdown()

                col1, col2 = st.columns(2)
                with col1:
                    self._display_positions()
                with col2:
                    self._display_signals()

            time.sleep(refresh_rate)

    # ------------------------------------------------------------------
    # Display components
    # ------------------------------------------------------------------

    def _display_metrics(self) -> None:
        """Display key performance metrics in a 6-column row."""
        st = _st
        metrics = self._fetch_latest_metrics()

        cols = st.columns(6)

        with cols[0]:
            st.metric(
                "Portfolio Value",
                f"${metrics['portfolio_value']:,.2f}",
                f"{metrics['daily_return']:.2%}",
            )
        with cols[1]:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
        with cols[2]:
            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        with cols[3]:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        with cols[4]:
            st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        with cols[5]:
            st.metric("Active Positions", f"{metrics['n_positions']}")

    def _display_equity_curve(self) -> None:
        """Plot equity curve vs benchmark."""
        st = _st
        go = _go

        equity = self._fetch_equity_history()
        if equity.empty:
            st.info("No equity data available")
            return

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity["portfolio"],
                name="Quantum Alpha",
                line=dict(color="green", width=2),
            )
        )

        if "spy" in equity.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity["spy"],
                    name="S&P 500",
                    line=dict(color="blue", width=1),
                )
            )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_drawdown(self) -> None:
        """Plot drawdown over time."""
        st = _st
        go = _go

        equity = self._fetch_equity_history()
        if equity.empty or "portfolio" not in equity.columns:
            st.info("No drawdown data available")
            return

        peak = equity["portfolio"].cummax()
        drawdown = (equity["portfolio"] - peak) / peak

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_positions(self) -> None:
        """Display current / final positions."""
        st = _st
        st.subheader("Current Positions")

        positions = self._fetch_positions()
        if positions.empty:
            st.info("No active positions")
            return

        display_df = positions.copy()
        if "pnl_pct" in display_df.columns:
            display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:.2%}")
        if "value" in display_df.columns:
            display_df["value"] = display_df["value"].apply(lambda x: f"${x:,.2f}")

        st.dataframe(display_df, use_container_width=True)

    def _display_signals(self) -> None:
        """Display recent trading signals."""
        st = _st
        st.subheader("Recent Signals")

        signals = self._fetch_recent_signals()
        if signals.empty:
            st.info("No recent signals")
            return

        st.dataframe(signals, use_container_width=True)

    def _display_model_performance(self) -> None:
        """Display individual model performance."""
        st = _st
        go = _go
        make_subplots = _make_subplots

        st.subheader("Model Performance")

        model_perf = self._fetch_model_performance()
        if not model_perf:
            st.info("No model performance data available")
            return

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Prediction Accuracy", "Contribution to Returns"),
            specs=[[{"type": "bar"}, {"type": "pie"}]],
        )

        fig.add_trace(
            go.Bar(
                x=list(model_perf.keys()),
                y=[p["accuracy"] for p in model_perf.values()],
                name="Accuracy",
                marker_color="steelblue",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Pie(
                labels=list(model_perf.keys()),
                values=[p["contribution"] for p in model_perf.values()],
            ),
            row=1,
            col=2,
        )

        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    def _display_gate_results(self) -> None:
        """Display performance gate pass/fail breakdown."""
        st = _st

        gate = self.results.get("gate_detail")
        if gate is None:
            return

        st.subheader("Performance Gate Results")

        if isinstance(gate, pd.DataFrame):
            st.dataframe(gate, use_container_width=True)
        elif isinstance(gate, dict):
            df = pd.DataFrame(gate).T
            st.dataframe(df, use_container_width=True)

    # ------------------------------------------------------------------
    # Data fetching (pluggable)
    # ------------------------------------------------------------------

    def _fetch_latest_metrics(self) -> Dict[str, float]:
        """Fetch latest performance metrics."""
        if self.results:
            return {
                "portfolio_value": self.results.get("final_equity", 100_000),
                "daily_return": self.results.get("daily_return", 0.0),
                "total_return": self.results.get("total_return", 0.0),
                "sharpe": self.results.get("sharpe_ratio", 0.0),
                "max_drawdown": self.results.get("max_drawdown", 0.0),
                "win_rate": self.results.get("win_rate", 0.0),
                "n_positions": self.results.get("n_positions", 0),
            }

        # Placeholder for live mode
        return {
            "portfolio_value": 100_000.0,
            "daily_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_positions": 0,
        }

    def _fetch_equity_history(self) -> pd.DataFrame:
        """Fetch equity curve data."""
        if "equity_curve" in self.results:
            return self.results["equity_curve"]

        # Placeholder
        return pd.DataFrame()

    def _fetch_positions(self) -> pd.DataFrame:
        """Fetch current positions."""
        if "positions" in self.results:
            return self.results["positions"]

        return pd.DataFrame()

    def _fetch_recent_signals(self) -> pd.DataFrame:
        """Fetch recent signals."""
        if "signals" in self.results:
            return self.results["signals"]

        return pd.DataFrame()

    def _fetch_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Fetch model performance data."""
        if "model_performance" in self.results:
            return self.results["model_performance"]

        # Default models
        return {
            "LSTM": {"accuracy": 0.62, "contribution": 0.30},
            "Mathematical": {"accuracy": 0.58, "contribution": 0.25},
            "Sentiment": {"accuracy": 0.55, "contribution": 0.20},
            "Technical": {"accuracy": 0.52, "contribution": 0.15},
            "TDA": {"accuracy": 0.60, "contribution": 0.10},
        }

    # ------------------------------------------------------------------
    # Report loading
    # ------------------------------------------------------------------

    @classmethod
    def from_report(cls, report_path: str) -> "TradingDashboard":
        """
        Create dashboard from a saved backtest report JSON.

        Args:
            report_path: Path to backtest_summary.json.

        Returns:
            TradingDashboard instance.
        """
        import json

        path = Path(report_path)
        if not path.exists():
            logger.warning("Report not found: %s", report_path)
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls(backtest_results=data)


def start_dashboard(
    report_path: Optional[str] = None,
    backtest_results: Optional[Dict] = None,
) -> None:
    """
    Entry point for the dashboard.

    Can be called directly or via:
        streamlit run quantum_alpha/visualization/dashboard.py
    """
    if report_path:
        dashboard = TradingDashboard.from_report(report_path)
    elif backtest_results:
        dashboard = TradingDashboard(backtest_results=backtest_results)
    else:
        dashboard = TradingDashboard()

    dashboard.run()


if __name__ == "__main__":
    start_dashboard()
