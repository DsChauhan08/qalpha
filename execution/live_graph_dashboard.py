#!/usr/bin/env python3
"""Live graph dashboard for realtime paper sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _latest_session(root: Path) -> Path | None:
    candidates = [p for p in root.glob("realtime_paper_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def load_status(session_dir: Path) -> dict:
    path = session_dir / "live_status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_equity_curve(session_dir: Path) -> pd.DataFrame:
    path = session_dir / "equity_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def load_trades(session_dir: Path) -> pd.DataFrame:
    path = session_dir / "live_trades.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp_dt"] = pd.NaT

    for col in ("notional", "exec_price", "qty", "commission", "cash_after"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.sort_values("timestamp_dt")


def build_trade_flow_by_second(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "timestamp_dt" not in trades.columns:
        return pd.DataFrame()
    df = trades.dropna(subset=["timestamp_dt"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["ts_second"] = df["timestamp_dt"].dt.floor("1s")
    df["signed_notional"] = df["notional"]
    if "side" in df.columns:
        sells = df["side"].astype(str).str.upper() == "SELL"
        df.loc[sells, "signed_notional"] = -df.loc[sells, "signed_notional"].abs()

    out = (
        df.groupby("ts_second", as_index=False)["signed_notional"]
        .sum()
        .sort_values("ts_second")
    )
    return out


def _fmt_money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "n/a"


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "n/a"


def _equity_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        fig.update_layout(title="Equity Curve (no data yet)", template="plotly_white")
        return fig

    y_col = "equity" if "equity" in equity.columns else "cash"
    fig.add_trace(
        go.Scatter(
            x=equity["timestamp"],
            y=equity[y_col],
            mode="lines",
            name="Equity",
            line=dict(color="#0f766e", width=2),
        )
    )
    fig.update_layout(
        title="Live Equity Curve",
        xaxis_title="Time",
        yaxis_title="USD",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _trade_price_chart(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(title="Trade Price Tape (no trades yet)", template="plotly_white")
        return fig

    t = trades.dropna(subset=["timestamp_dt"]).copy()
    if t.empty:
        fig.update_layout(title="Trade Price Tape (no timestamps)", template="plotly_white")
        return fig

    t["side_norm"] = t.get("side", "").astype(str).str.upper()
    buys = t[t["side_norm"] == "BUY"]
    sells = t[t["side_norm"] == "SELL"]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["timestamp_dt"],
                y=buys["exec_price"],
                mode="markers",
                name="BUY",
                marker=dict(color="#15803d", size=7, symbol="triangle-up"),
                customdata=buys[["symbol", "qty", "notional"]].values,
                hovertemplate=(
                    "BUY %{customdata[0]}<br>px=%{y:.4f}<br>qty=%{customdata[1]:.4f}"
                    "<br>notional=$%{customdata[2]:,.2f}<extra></extra>"
                ),
            )
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["timestamp_dt"],
                y=sells["exec_price"],
                mode="markers",
                name="SELL",
                marker=dict(color="#b91c1c", size=7, symbol="triangle-down"),
                customdata=sells[["symbol", "qty", "notional"]].values,
                hovertemplate=(
                    "SELL %{customdata[0]}<br>px=%{y:.4f}<br>qty=%{customdata[1]:.4f}"
                    "<br>notional=$%{customdata[2]:,.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Trade Price Tape (1s-detail timestamps)",
        xaxis_title="Time",
        yaxis_title="Execution Price",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _flow_chart(flow: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if flow.empty:
        fig.update_layout(title="Signed Notional per Second (no trades yet)", template="plotly_white")
        return fig

    colors = ["#15803d" if v >= 0 else "#b91c1c" for v in flow["signed_notional"]]
    fig.add_trace(
        go.Bar(
            x=flow["ts_second"],
            y=flow["signed_notional"],
            marker_color=colors,
            name="Signed Notional",
        )
    )
    fig.update_layout(
        title="Signed Notional by Second",
        xaxis_title="Time",
        yaxis_title="USD",
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _render_header(status: dict, session_dir: Path) -> None:
    st.title("Quantum Alpha Live Graph Runner")
    st.caption(f"session: {session_dir}")
    st.caption(f"last status update: {status.get('timestamp_utc', 'n/a')}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Equity", _fmt_money(status.get("equity")), _fmt_money(status.get("equity_change_dollars")))
    c2.metric("Cash", _fmt_money(status.get("cash")))
    c3.metric("P/L", _fmt_money(status.get("profit_dollars")), _fmt_pct(status.get("return_pct")))
    c4.metric("Trades", str(status.get("trades", 0)))
    c5.metric("Positions", str(status.get("positions", 0)))
    c6.metric(
        "LLM Cycle",
        "ON" if status.get("llm_active_this_cycle") else "OFF",
        f"every {status.get('llm_decision_interval_cycles', 1)} cycles",
    )


def _render_risk_panel(status: dict) -> None:
    risk = status.get("risk_snapshot") or {}
    if not isinstance(risk, dict):
        st.info("Risk snapshot unavailable")
        return
    st.subheader("Live Risk Panel")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("VaR", _fmt_pct(risk.get("var", 0.0) * 100.0))
    c2.metric("CVaR", _fmt_pct(risk.get("cvar", 0.0) * 100.0))
    c3.metric("Drawdown", _fmt_pct(risk.get("drawdown", 0.0) * 100.0))
    c4.metric("Ulcer", f"{float(risk.get('ulcer_index', 0.0)):.4f}")
    c5.metric("Net Exp", f"{float(risk.get('net_exposure', 0.0)):.3f}")
    c6.metric("Gross Exp", f"{float(risk.get('gross_exposure', 0.0)):.3f}")


def _render_allocator_panel(status: dict) -> None:
    st.subheader("Allocator Decision Panel")
    chosen = status.get("allocator_chosen_stack", "n/a")
    scores = status.get("allocator_stack_scores") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Chosen Stack", str(chosen))
    c2.metric("Static Score", f"{float(scores.get('static', 0.0)):.4f}" if isinstance(scores, dict) else "n/a")
    c3.metric("Regime Score", f"{float(scores.get('regime', 0.0)):.4f}" if isinstance(scores, dict) else "n/a")


def _render_provider_panel(status: dict) -> None:
    st.subheader("Provider Health / Fallback Panel")
    provider_status = status.get("provider_status") or {}
    openbb_api = status.get("openbb_api") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Provider Degraded", str(bool(status.get("provider_degraded", False))))
    c2.metric("OpenBB Enabled", str(bool(openbb_api.get("enabled", False))))
    c3.metric("OpenBB Healthy", str(bool(openbb_api.get("healthy", False))))
    if provider_status:
        st.dataframe(pd.DataFrame(provider_status).T, use_container_width=True, height=220)
    else:
        st.info("No provider status yet")


def _render_pipeline_panel(status: dict) -> None:
    st.subheader("Pipeline / Model Health")
    latency = status.get("latency_ms") or {}
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pipeline", str(status.get("pipeline_health", "n/a")))
    c2.metric("Engine", str(status.get("decision_engine", "n/a")))
    c3.metric("A/B Group", str(status.get("ab_group", "n/a")))
    c4.metric("Anchor Fresh (min)", f"{float(status.get('anchor_cache_freshness_minutes', 0.0)):.1f}" if status.get("anchor_cache_freshness_minutes") is not None else "n/a")
    c5.metric("Base Model", str(bool(status.get("model_health_base", False))))
    c6.metric("MC Model", str(bool(status.get("model_health_mc", False))))

    c7, c8, c9, c10 = st.columns(4)
    c7.metric(
        "Missing Base",
        f"{float(status.get('feature_missing_ratio_base', 0.0)):.3f}",
    )
    c8.metric(
        "Missing MC",
        f"{float(status.get('feature_missing_ratio_mc', 0.0)):.3f}",
    )
    c9.metric(
        "Cycle ms",
        f"{float(latency.get('total_cycle_ms', 0.0)):.0f}",
        f"budget={float(latency.get('budget_ms', 0.0)):.0f}",
    )
    c10.metric("Latency Pass", str(bool(latency.get("budget_pass", True))))


def _auto_refresh(refresh_seconds: float) -> None:
    sec = max(1, int(round(float(refresh_seconds))))
    st.markdown(
        f"<meta http-equiv='refresh' content='{sec}'>",
        unsafe_allow_html=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live graph dashboard for realtime paper session")
    parser.add_argument("--session-dir", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    args = parser.parse_args()

    st.set_page_config(page_title="QA Live Graph", layout="wide")

    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = _latest_session(Path(args.artifacts_dir))
        if session_dir is None:
            st.error(f"No realtime_paper_* directory found under {args.artifacts_dir}")
            return

    _auto_refresh(args.refresh_seconds)

    status = load_status(session_dir)
    equity = load_equity_curve(session_dir)
    trades = load_trades(session_dir)
    flow = build_trade_flow_by_second(trades)

    _render_header(status, session_dir)
    _render_risk_panel(status)
    _render_allocator_panel(status)
    _render_provider_panel(status)
    _render_pipeline_panel(status)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_equity_chart(equity), use_container_width=True)
    with col2:
        st.plotly_chart(_trade_price_chart(trades.tail(4000)), use_container_width=True)

    st.plotly_chart(_flow_chart(flow.tail(3600)), use_container_width=True)

    st.subheader("Latest Model Decisions")
    decisions = pd.DataFrame(status.get("latest_decisions") or [])
    if decisions.empty:
        st.info("No decisions yet")
    else:
        st.dataframe(decisions, use_container_width=True, height=240)

    st.subheader("Trade Tape")
    if trades.empty:
        st.info("No trades yet")
    else:
        trade_cols = [
            c
            for c in [
                "timestamp",
                "side",
                "symbol",
                "qty",
                "exec_price",
                "notional",
                "commission",
                "cash_after",
            ]
            if c in trades.columns
        ]
        display = trades.sort_values("timestamp_dt", ascending=False)
        st.dataframe(display[trade_cols], use_container_width=True, height=360)

    st.subheader("Positions Snapshot")
    positions = pd.DataFrame(status.get("positions_snapshot") or [])
    if positions.empty:
        st.info("No open positions")
    else:
        st.dataframe(positions, use_container_width=True, height=260)

    st.subheader("Runner Notes")
    st.write(
        "This dashboard refreshes every second, shows full trade tape with execution prices, "
        "and displays model decisions from the latest cycle."
    )


if __name__ == "__main__":
    main()
