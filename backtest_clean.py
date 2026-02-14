#!/usr/bin/env python3
"""
Clean Meta-Ensemble Backtester
================================
Operates directly on walk-forward out-of-sample predictions to give
an honest assessment of the model's actual edge.

Bypasses the complex main.py backtesting framework (PositionSizer,
DrawdownController, momentum filters, risk-parity overlays) that
were shown to destroy the model's modest edge through:
  - Commission drag (0.1% per trade with daily rebalancing)
  - Kelly criterion cold-start (zero sizing for first 10 trades)
  - Momentum filter overriding meta-ensemble signals
  - Signal lag + weekly rebalance timing mismatch

Instead, this implements a simple, transparent simulation:
  1. Load walk-forward OOS predictions
  2. Apply configurable trading rules (thresholds, hold periods, etc.)
  3. Simulate portfolio with realistic transaction costs
  4. Report clean performance metrics

Usage:
  python backtest_clean.py                          # Defaults
  python backtest_clean.py --confidence 0.15        # High-confidence only
  python backtest_clean.py --commission 0.0         # Zero commission
  python backtest_clean.py --hold-days 5            # Min 5-day hold
  python backtest_clean.py --top-k 10               # Trade top-10 signals
  python backtest_clean.py --long-only              # No shorting
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/regulus/Trade")


def load_predictions(checkpoint_dir: str | Path) -> pd.DataFrame:
    """Load walk-forward out-of-sample predictions."""
    pred_path = Path(checkpoint_dir) / "walk_forward_predictions.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions found at {pred_path}")

    with open(pred_path, "rb") as f:
        df = pickle.load(f)

    print(f"Loaded {len(df):,} predictions from {pred_path.name}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def deduplicate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward folds overlap (6-month step, 1-year window).
    For duplicate (date, symbol) pairs, keep the prediction from the
    MOST RECENT training fold (most data seen = best model).
    """
    n_before = len(df)

    # If there's a 'fold' column, sort by it and keep last
    if "fold" in df.columns:
        df = df.sort_values("fold").drop_duplicates(
            subset=["date", "symbol"], keep="last"
        )
    else:
        # Keep last occurrence (assumes folds are appended in order)
        df = df.drop_duplicates(subset=["date", "symbol"], keep="last")

    n_after = len(df)
    if n_before != n_after:
        print(
            f"  Deduplicated: {n_before:,} -> {n_after:,} predictions "
            f"({n_before - n_after:,} overlapping duplicates removed)"
        )

    return df


def compute_signals(
    df: pd.DataFrame,
    signal_threshold: float = 0.52,
    confidence_threshold: float = 0.0,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Convert model probabilities to trading signals.

    Args:
        df: DataFrame with 'y_proba' column (P(up))
        signal_threshold: Min probability to trigger long (1 - threshold for short)
        confidence_threshold: Min |P - 0.5| to trade at all
        long_only: If True, only take long positions
    """
    df = df.copy()

    # Confidence = distance from 0.5
    df["confidence"] = (df["y_proba"] - 0.5).abs() * 2.0  # [0, 1]

    # Raw signal: +1 (long), -1 (short), 0 (flat)
    df["raw_signal"] = 0.0
    long_mask = df["y_proba"] >= signal_threshold
    short_mask = df["y_proba"] <= (1.0 - signal_threshold)
    conf_mask = df["confidence"] >= confidence_threshold

    df.loc[long_mask & conf_mask, "raw_signal"] = 1.0
    if not long_only:
        df.loc[short_mask & conf_mask, "raw_signal"] = -1.0

    n_long = (df["raw_signal"] == 1.0).sum()
    n_short = (df["raw_signal"] == -1.0).sum()
    n_flat = (df["raw_signal"] == 0.0).sum()
    pct_active = (n_long + n_short) / len(df) * 100

    print(
        f"  Signals: {n_long:,} long, {n_short:,} short, {n_flat:,} flat "
        f"({pct_active:.1f}% active)"
    )

    return df


def backtest_equal_weight(
    df: pd.DataFrame,
    max_positions: int = 20,
    commission_bps: float = 0.0,
    hold_days: int = 1,
    top_k: int | None = None,
    initial_capital: float = 100_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Simple equal-weight portfolio backtest (vectorized for speed).

    On each trading day:
      1. Rank active signals by confidence
      2. Select top-K (or all, up to max_positions)
      3. Equal-weight allocation
      4. Compute returns after transaction costs

    Args:
        df: DataFrame with 'date', 'symbol', 'raw_signal', 'confidence',
            'forward_return', 'y_proba', 'y_true'
        max_positions: Maximum concurrent positions
        commission_bps: Commission in basis points (each way). 0 = free.
        hold_days: Minimum holding period in days
        top_k: If set, only trade top-K most confident signals per day
        initial_capital: Starting capital for equity curve
        start_date: Filter predictions from this date
        end_date: Filter predictions to this date
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if len(df) == 0:
        return {"error": "No predictions in date range"}

    # Drop NaN forward returns
    df = df.dropna(subset=["forward_return"])

    # --- Vectorized approach: rank within each day, filter top-K ---

    # Only keep active signals
    active = df[df["raw_signal"] != 0.0].copy()

    if len(active) == 0:
        return {"error": "No active signals after filtering"}

    # Rank by confidence within each date (1 = highest confidence)
    active["conf_rank"] = active.groupby("date")["confidence"].rank(
        ascending=False, method="first"
    )

    # Apply top-K and max_positions cap
    cap = max_positions
    if top_k is not None:
        cap = min(cap, top_k)
    active = active[active["conf_rank"] <= cap]

    # Count positions per day
    pos_counts = active.groupby("date").size()

    # Equal weight per day: 1/N
    active = active.merge(
        pos_counts.rename("n_pos").reset_index(), on="date", how="left"
    )
    active["weight"] = 1.0 / active["n_pos"]

    # Weighted return per position
    active["weighted_return"] = (
        active["raw_signal"] * active["forward_return"] * active["weight"]
    )

    # Correctness
    active["correct"] = (active["raw_signal"] > 0) == (active["y_true"] == 1)

    # --- Aggregate to daily level ---
    daily = (
        active.groupby("date")
        .agg(
            gross_return=("weighted_return", "sum"),
            n_positions=("symbol", "count"),
            n_correct=("correct", "sum"),
            n_total=("correct", "count"),
        )
        .sort_index()
    )

    # Fill missing dates (days with no active signals = 0 return)
    all_dates = sorted(df["date"].unique())
    date_idx = pd.DatetimeIndex(all_dates)
    daily = daily.reindex(date_idx, fill_value=0)
    daily.index.name = "date"

    # --- Transaction costs ---
    # Approximate turnover: for simplicity, compare active symbol sets
    # between consecutive days. This is fast enough.
    commission_rate = commission_bps / 10_000.0

    if commission_rate > 0 and hold_days <= 1:
        # Build per-day symbol sets for turnover calculation
        day_symbols = active.groupby("date").apply(
            lambda g: set(zip(g["symbol"], g["raw_signal"])), include_groups=False
        )
        day_symbols = day_symbols.reindex(date_idx)

        daily_costs = np.zeros(len(date_idx))
        prev_set = set()
        for i, date in enumerate(date_idx):
            curr_set = day_symbols.get(date, set()) or set()
            if not isinstance(curr_set, set):
                curr_set = set()
            # Turnover = positions that changed
            changed = len(curr_set.symmetric_difference(prev_set))
            n_pos = daily.iloc[i]["n_positions"]
            w = 1.0 / max(n_pos, 1)
            daily_costs[i] = changed * w * commission_rate * 2
            prev_set = curr_set
    else:
        daily_costs = np.zeros(len(date_idx))

    daily_net_returns = daily["gross_return"].values - daily_costs

    # --- Build trade log ---
    trade_df = active[
        [
            "date",
            "symbol",
            "raw_signal",
            "forward_return",
            "weight",
            "weighted_return",
            "correct",
        ]
    ].copy()
    trade_df.columns = [
        "date",
        "symbol",
        "signal",
        "forward_return",
        "weight",
        "pnl",
        "correct",
    ]

    # --- Compute equity curve ---
    equity = initial_capital * np.cumprod(1 + daily_net_returns)

    n_dates = len(date_idx)
    daily_turnover_arr = np.zeros(n_dates)  # simplified
    daily_n_positions = daily["n_positions"].values

    results = compute_metrics(
        daily_net_returns,
        daily_costs,
        np.array(date_idx),
        equity,
        trade_df,
        daily_n_positions,
        daily_turnover_arr,
        initial_capital,
        n_dates,
    )

    return results


def compute_metrics(
    daily_returns,
    daily_costs,
    daily_dates,
    equity,
    trade_df,
    daily_n_positions,
    daily_turnover,
    initial_capital,
    n_dates,
) -> dict:
    """Compute comprehensive backtest metrics."""

    total_return = equity[-1] / initial_capital - 1 if len(equity) > 0 else 0
    n_years = n_dates / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Risk metrics
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = np.inf
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(equity)
    drawdowns = equity / cummax - 1
    max_drawdown = drawdowns.min()

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Trade accuracy
    if len(trade_df) > 0:
        win_rate = trade_df["correct"].mean()
        winning_trades = trade_df[trade_df["pnl"] > 0]["pnl"]
        losing_trades = trade_df[trade_df["pnl"] < 0]["pnl"]
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = (
            winning_trades.sum() / abs(losing_trades.sum())
            if len(losing_trades) > 0 and losing_trades.sum() != 0
            else np.inf
        )
        total_trades = len(trade_df)
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        total_trades = 0

    # Position stats
    avg_positions = np.mean(daily_n_positions)
    avg_turnover = np.mean(daily_turnover)
    total_cost = np.sum(daily_costs)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "avg_positions_per_day": avg_positions,
        "avg_daily_turnover": avg_turnover,
        "total_commission_cost": total_cost,
        "n_trading_days": len(daily_returns),
        "n_years": n_dates / 252.0,
        "final_equity": equity[-1] if len(equity) > 0 else initial_capital,
        "equity_curve": equity,
        "daily_returns": daily_returns,
        "daily_dates": daily_dates,
        "trade_df": trade_df,
    }


def print_results(results: dict, label: str = ""):
    """Pretty-print backtest results."""
    if "error" in results:
        print(f"\n  ERROR: {results['error']}")
        return

    print(f"\n{'=' * 65}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 65}")

    rows = [
        ("Total Return", f"{results['total_return']:+.2%}"),
        ("Annual Return", f"{results['annual_return']:+.2%}"),
        ("Sharpe Ratio", f"{results['sharpe']:.3f}"),
        ("Sortino Ratio", f"{results['sortino']:.3f}"),
        ("Max Drawdown", f"{results['max_drawdown']:.2%}"),
        ("Calmar Ratio", f"{results['calmar']:.3f}"),
        ("", ""),
        ("Win Rate (directional)", f"{results['win_rate']:.1%}"),
        ("Avg Win", f"{results['avg_win']:.4%}"),
        ("Avg Loss", f"{results['avg_loss']:.4%}"),
        ("Profit Factor", f"{results['profit_factor']:.3f}"),
        ("Total Trade-Days", f"{results['total_trades']:,}"),
        ("", ""),
        ("Avg Positions/Day", f"{results['avg_positions_per_day']:.1f}"),
        ("Avg Daily Turnover", f"{results['avg_daily_turnover']:.1f}"),
        ("Total Commission Cost", f"{results['total_commission_cost']:.4%} of capital"),
        ("", ""),
        ("Trading Days", f"{results['n_trading_days']:,}"),
        ("Period (years)", f"{results['n_years']:.1f}"),
        ("Final Equity", f"${results['final_equity']:,.0f}"),
    ]

    for label_str, val in rows:
        if label_str == "":
            continue
        print(f"  {label_str:.<35} {val}")

    print(f"{'=' * 65}")


def run_confidence_sweep(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    long_only: bool = False,
    commission_bps: float = 0.0,
    hold_days: int = 1,
    max_positions: int = 20,
) -> pd.DataFrame:
    """
    Sweep across confidence thresholds to validate calibration.
    """
    if thresholds is None:
        thresholds = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    rows = []
    for ct in thresholds:
        sig_df = compute_signals(
            df,
            signal_threshold=0.50,
            confidence_threshold=ct,
            long_only=long_only,
        )
        n_active = (sig_df["raw_signal"] != 0).sum()
        pct_active = n_active / len(sig_df) * 100

        if n_active == 0:
            rows.append(
                {
                    "confidence_threshold": ct,
                    "pct_active": 0.0,
                    "total_return": 0.0,
                    "annual_return": 0.0,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "n_trades": 0,
                }
            )
            continue

        res = backtest_equal_weight(
            sig_df,
            max_positions=max_positions,
            commission_bps=commission_bps,
            hold_days=hold_days,
        )

        rows.append(
            {
                "confidence_threshold": ct,
                "pct_active": pct_active,
                "total_return": res["total_return"],
                "annual_return": res["annual_return"],
                "sharpe": res["sharpe"],
                "max_drawdown": res["max_drawdown"],
                "win_rate": res["win_rate"],
                "profit_factor": res["profit_factor"],
                "n_trades": res["total_trades"],
                "avg_positions": res["avg_positions_per_day"],
            }
        )

    return pd.DataFrame(rows)


def run_year_by_year(
    df: pd.DataFrame,
    signal_threshold: float = 0.52,
    confidence_threshold: float = 0.0,
    long_only: bool = False,
    commission_bps: float = 0.0,
    hold_days: int = 1,
    max_positions: int = 20,
) -> pd.DataFrame:
    """Break down performance by calendar year."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    sig_df = compute_signals(
        df,
        signal_threshold=signal_threshold,
        confidence_threshold=confidence_threshold,
        long_only=long_only,
    )

    years = sorted(sig_df["year"].unique())
    rows = []

    for year in years:
        year_df = sig_df[sig_df["year"] == year]
        n_active = (year_df["raw_signal"] != 0).sum()

        if n_active == 0:
            rows.append(
                {
                    "year": year,
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "n_trades": 0,
                }
            )
            continue

        res = backtest_equal_weight(
            year_df,
            max_positions=max_positions,
            commission_bps=commission_bps,
            hold_days=hold_days,
        )

        rows.append(
            {
                "year": year,
                "total_return": res["total_return"],
                "sharpe": res["sharpe"],
                "max_drawdown": res["max_drawdown"],
                "win_rate": res["win_rate"],
                "n_trades": res["total_trades"],
                "avg_positions": res["avg_positions_per_day"],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Clean Meta-Ensemble Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_clean.py                           # Default: all signals, 0 commission
  python backtest_clean.py --confidence 0.15         # High-confidence only
  python backtest_clean.py --commission 10            # 10 bps (0.1%) commission each way
  python backtest_clean.py --hold-days 5             # Min 5-day holding period
  python backtest_clean.py --top-k 10 --long-only    # Top-10 long only
  python backtest_clean.py --sweep                   # Confidence threshold sweep
  python backtest_clean.py --yearly                  # Year-by-year breakdown
        """,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/meta_ensemble",
        help="Directory containing walk_forward_predictions.pkl",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.52,
        help="Min probability for long/short (default: 0.52)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Min confidence to trade (default: 0.0 = all)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission in basis points each way (default: 0 = free)",
    )
    parser.add_argument(
        "--hold-days",
        type=int,
        default=1,
        help="Minimum holding period in days (default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only trade top-K most confident signals per day",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Maximum concurrent positions (default: 20)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Only take long positions",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run confidence threshold sweep",
    )
    parser.add_argument(
        "--yearly",
        action="store_true",
        help="Show year-by-year breakdown",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Filter to specific symbols",
    )

    args = parser.parse_args()

    # Resolve checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).parent / ckpt_dir

    # Load predictions
    print(f"\n{'=' * 65}")
    print("  CLEAN META-ENSEMBLE BACKTESTER")
    print(f"{'=' * 65}")
    print(f"  Using walk-forward OOS predictions (no lookahead)")
    print(f"  Bypasses PositionSizer / DrawdownController / momentum filter")
    print(f"{'=' * 65}\n")

    df = load_predictions(ckpt_dir)
    df = deduplicate_predictions(df)

    # Filter symbols
    if args.symbols:
        df = df[df["symbol"].isin(args.symbols)]
        print(f"  Filtered to symbols: {args.symbols}")
        print(f"  Remaining predictions: {len(df):,}")

    # Print config
    print(f"\n  Config:")
    print(f"    Signal threshold:   {args.signal_threshold:.2f}")
    print(f"    Confidence min:     {args.confidence:.2f}")
    print(f"    Commission:         {args.commission:.1f} bps each way")
    print(f"    Hold days:          {args.hold_days}")
    print(f"    Top-K:              {args.top_k or 'all'}")
    print(f"    Max positions:      {args.max_positions}")
    print(f"    Long only:          {args.long_only}")
    print(f"    Capital:            ${args.capital:,.0f}")

    if args.start_date:
        print(f"    Start date:         {args.start_date}")
    if args.end_date:
        print(f"    End date:           {args.end_date}")

    # --- Confidence sweep mode ---
    if args.sweep:
        print(f"\n\n{'=' * 90}")
        print("  CONFIDENCE THRESHOLD SWEEP")
        print(f"{'=' * 90}")

        sweep_df = run_confidence_sweep(
            df,
            long_only=args.long_only,
            commission_bps=args.commission,
            hold_days=args.hold_days,
            max_positions=args.max_positions,
        )

        print(
            f"\n  {'Conf':>6} | {'Active%':>8} | {'Return':>10} | "
            f"{'Ann.Ret':>10} | {'Sharpe':>8} | {'MaxDD':>8} | "
            f"{'WinRate':>8} | {'PF':>6} | {'Trades':>10} | {'Avg Pos':>8}"
        )
        print(
            f"  {'-' * 6}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-"
            f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 8}"
        )

        for _, row in sweep_df.iterrows():
            print(
                f"  {row['confidence_threshold']:>5.0%} | "
                f"{row['pct_active']:>7.1f}% | "
                f"{row.get('total_return', 0):>+9.2%} | "
                f"{row.get('annual_return', 0):>+9.2%} | "
                f"{row.get('sharpe', 0):>7.3f} | "
                f"{row.get('max_drawdown', 0):>7.2%} | "
                f"{row.get('win_rate', 0):>7.1%} | "
                f"{row.get('profit_factor', 0):>5.2f} | "
                f"{row.get('n_trades', 0):>10,} | "
                f"{row.get('avg_positions', 0):>7.1f}"
            )

        print(f"\n{'=' * 90}\n")
        return

    # --- Year-by-year mode ---
    if args.yearly:
        print(f"\n\n{'=' * 80}")
        print("  YEAR-BY-YEAR BREAKDOWN")
        print(f"{'=' * 80}")

        yearly_df = run_year_by_year(
            df,
            signal_threshold=args.signal_threshold,
            confidence_threshold=args.confidence,
            long_only=args.long_only,
            commission_bps=args.commission,
            hold_days=args.hold_days,
            max_positions=args.max_positions,
        )

        print(
            f"\n  {'Year':>6} | {'Return':>10} | {'Sharpe':>8} | "
            f"{'MaxDD':>8} | {'WinRate':>8} | {'Trades':>10} | {'Avg Pos':>8}"
        )
        print(
            f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-"
            f"{'-' * 8}-+-{'-' * 10}-+-{'-' * 8}"
        )

        for _, row in yearly_df.iterrows():
            print(
                f"  {int(row['year']):>6} | "
                f"{row['total_return']:>+9.2%} | "
                f"{row.get('sharpe', 0):>7.3f} | "
                f"{row.get('max_drawdown', 0):>7.2%} | "
                f"{row.get('win_rate', 0):>7.1%} | "
                f"{row.get('n_trades', 0):>10,} | "
                f"{row.get('avg_positions', 0):>7.1f}"
            )

        # Summary row
        print(
            f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-"
            f"{'-' * 8}-+-{'-' * 10}-+-{'-' * 8}"
        )
        total_ret = (1 + yearly_df["total_return"]).prod() - 1
        print(f"  {'TOTAL':>6} | {total_ret:>+9.2%}")

        print(f"\n{'=' * 80}\n")
        return

    # --- Standard single-run mode ---
    sig_df = compute_signals(
        df,
        signal_threshold=args.signal_threshold,
        confidence_threshold=args.confidence,
        long_only=args.long_only,
    )

    results = backtest_equal_weight(
        sig_df,
        max_positions=args.max_positions,
        commission_bps=args.commission,
        hold_days=args.hold_days,
        top_k=args.top_k,
        initial_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    config_label = (
        f"Signal >= {args.signal_threshold}, Confidence >= {args.confidence:.0%}, "
        f"Commission={args.commission:.0f}bps, Hold={args.hold_days}d, "
        f"{'Long Only' if args.long_only else 'Long+Short'}"
    )
    print_results(results, config_label)

    # Quick comparison: also show with various commission levels
    if args.commission == 0.0:
        print(f"\n  --- Commission sensitivity ---")
        for c_bps in [5, 10, 20]:
            res_c = backtest_equal_weight(
                sig_df,
                max_positions=args.max_positions,
                commission_bps=c_bps,
                hold_days=args.hold_days,
                top_k=args.top_k,
                initial_capital=args.capital,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print(
                f"  @{c_bps:>2}bps commission: Return={res_c['total_return']:+.2%}, "
                f"Sharpe={res_c['sharpe']:.3f}, "
                f"Cost={res_c['total_commission_cost']:.4%}"
            )
        print()


if __name__ == "__main__":
    main()
