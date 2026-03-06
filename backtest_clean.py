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
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/regulus/Trade")


def _load_earnings_helpers():
    """
    Resolve earnings helper functions regardless of execution context.

    Supports running as:
    - module: python -m quantum_alpha.backtest_clean
    - script: python quantum_alpha/backtest_clean.py
    """
    module_candidates = [
        "quantum_alpha.data.collectors.earnings_calendar",
        "data.collectors.earnings_calendar",
    ]
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return (
                module.load_earnings_calendar,
                module.get_earnings_mask,
            )
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        "Could not import earnings calendar helpers. "
        "Expected module: quantum_alpha.data.collectors.earnings_calendar"
    )


SEMICONDUCTOR_SYMBOLS = {
    "NVDA",
    "AMD",
    "AVGO",
    "QCOM",
    "INTC",
    "AMAT",
    "LRCX",
    "MU",
    "TXN",
    "ADI",
    "MRVL",
    "MCHP",
    "NXPI",
    "ON",
    "KLAC",
    "WDC",
    "STX",
}


def _load_ai_regime_features() -> pd.DataFrame:
    module_candidates = [
        "quantum_alpha.data.collectors.ai_regime",
        "data.collectors.ai_regime",
    ]
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            regime = module.load_ai_regime_features()
            if isinstance(regime, pd.DataFrame):
                return regime
        except ModuleNotFoundError:
            continue
        except Exception:
            break
    return pd.DataFrame()


def _apply_semiconductor_short_gate(
    df: pd.DataFrame,
    gate_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Allow semiconductor shorts only when AI regime is weak.
    """
    out = df.copy()
    regime = _load_ai_regime_features()
    if regime.empty or "ai_regime_strength" not in regime.columns:
        print("  Semiconductor short gate: regime features unavailable, skipping")
        return out

    r = regime.copy()
    r.index = pd.to_datetime(r.index)
    if isinstance(r.index, pd.DatetimeIndex) and r.index.tz is not None:
        r.index = r.index.tz_localize(None)
    r.index = r.index.normalize()
    strength = r["ai_regime_strength"]

    trade_dates = pd.to_datetime(out["date"]).dt.normalize()
    gate_strength = trade_dates.map(strength)
    weak_regime = gate_strength <= gate_threshold
    mask = (
        out["symbol"].isin(SEMICONDUCTOR_SYMBOLS)
        & (out["raw_signal"] < 0)
        & (~weak_regime.fillna(False))
    )
    removed = int(mask.sum())
    out.loc[mask, "raw_signal"] = 0.0
    if removed > 0:
        print(
            f"  Semiconductor short gate: removed {removed:,} short signals "
            f"(threshold={gate_threshold:.2f})"
        )
    return out


def load_predictions(
    checkpoint_dir: str | Path,
    prediction_file: str = "walk_forward_predictions.pkl",
) -> pd.DataFrame:
    """Load walk-forward out-of-sample predictions."""
    pred_path = Path(checkpoint_dir) / str(prediction_file)
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions found at {pred_path}")

    with open(pred_path, "rb") as f:
        df = pickle.load(f)

    print(f"Loaded {len(df):,} predictions from {pred_path.name}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def blend_prediction_probabilities(
    primary_df: pd.DataFrame,
    checkpoint_dirs: list[str | Path],
    blend_weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Blend y_proba from the primary checkpoint with additional checkpoints.

    Args:
        primary_df: DataFrame from primary checkpoint (already deduplicated)
        checkpoint_dirs: Additional checkpoint dirs to blend
        blend_weights: Optional weights. Supports:
            - len == n_models (primary + extras)
            - len == n_models - 1 (extras only; primary defaults to 1.0)
    """
    if not checkpoint_dirs:
        return primary_df

    merged = primary_df[
        ["date", "symbol", "y_true", "forward_return", "y_proba"]
    ].copy()
    merged = merged.rename(columns={"y_proba": "p0"})
    prob_cols = ["p0"]

    for i, ckpt in enumerate(checkpoint_dirs, start=1):
        extra = deduplicate_predictions(load_predictions(ckpt))
        extra = extra[["date", "symbol", "y_proba"]].copy()
        col = f"p{i}"
        extra = extra.rename(columns={"y_proba": col})
        merged = merged.merge(extra, on=["date", "symbol"], how="inner")
        prob_cols.append(col)

    if len(merged) == 0:
        raise RuntimeError("No overlapping predictions across blended checkpoints")

    n_models = len(prob_cols)
    if blend_weights is None:
        weights = np.ones(n_models, dtype=float) / n_models
    else:
        w = np.asarray(blend_weights, dtype=float)
        if len(w) == n_models - 1:
            w = np.concatenate(([1.0], w))
        if len(w) != n_models:
            raise ValueError(
                f"blend_weights length mismatch: got {len(w)}, expected {n_models} "
                f"(or {n_models - 1} for extras-only weights)"
            )
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("blend_weights must sum to a positive number")
        weights = w / w_sum

    merged["y_proba_blend"] = 0.0
    for wi, col in zip(weights, prob_cols):
        merged["y_proba_blend"] += float(wi) * pd.to_numeric(
            merged[col], errors="coerce"
        ).fillna(0.5)

    out = primary_df.merge(
        merged[["date", "symbol", "y_proba_blend"]],
        on=["date", "symbol"],
        how="inner",
    )
    out["y_proba"] = out["y_proba_blend"]
    out = out.drop(columns=["y_proba_blend"])
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    weight_str = ", ".join(f"{w:.3f}" for w in weights)
    print(
        f"  Blended checkpoints: primary + {len(checkpoint_dirs)} extras "
        f"(weights: {weight_str})"
    )
    print(f"  Blended predictions rows: {len(out):,}")
    return out


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
    signal_threshold: float = 0.53,
    short_threshold: float | None = None,
    confidence_threshold: float = 0.0,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Convert model probabilities to trading signals.

    Args:
        df: DataFrame with 'y_proba' column (P(up))
        signal_threshold: Min probability to trigger long (1 - threshold for short)
        short_threshold: Max probability to trigger short. If None, use (1 - signal_threshold)
        confidence_threshold: Min |P - 0.5| to trade at all
        long_only: If True, only take long positions
    """
    df = df.copy()

    # Confidence = distance from 0.5
    df["confidence"] = (df["y_proba"] - 0.5).abs() * 2.0  # [0, 1]

    # Raw signal: +1 (long), -1 (short), 0 (flat)
    df["raw_signal"] = 0.0
    long_mask = df["y_proba"] >= signal_threshold
    if short_threshold is None:
        short_threshold = 1.0 - signal_threshold
    short_mask = df["y_proba"] <= short_threshold
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


def _apply_pead_boost(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    pead_window_days: int = 60,
    min_surprise_pct: float = 5.0,
    positive_boost: float = 1.5,
    negative_penalty: float = 0.5,
) -> pd.DataFrame:
    """
    Apply Post-Earnings Announcement Drift (PEAD) confidence adjustment.

    After a significant earnings surprise, stocks tend to drift in the
    direction of the surprise for 20-60 trading days. This function boosts
    confidence for positive surprises and reduces it for negative ones.

    Args:
        df: DataFrame with 'date', 'symbol', 'confidence' columns
        calendar: Earnings calendar with surprise data
        pead_window_days: Days after earnings to apply PEAD boost
        min_surprise_pct: Minimum |surprise_pct| to trigger PEAD
        positive_boost: Multiply confidence by this for positive surprise
        negative_penalty: Multiply confidence by this for negative surprise

    Returns:
        DataFrame with adjusted 'confidence' column
    """
    df = df.copy()

    # Only use historical earnings with surprise data
    hist = calendar.dropna(subset=["surprise_pct"]).copy()
    if hist.empty:
        return df

    # Filter to significant surprises
    hist = hist[hist["surprise_pct"].abs() >= min_surprise_pct].copy()
    if hist.empty:
        return df

    trade_dates = pd.to_datetime(df["date"])
    if isinstance(trade_dates, pd.DatetimeIndex):
        trade_dates = pd.Series(trade_dates, index=df.index)
    n_boosted = 0
    n_penalized = 0

    for _, erow in hist.iterrows():
        sym = erow["symbol"]
        edate = pd.to_datetime(erow["earnings_date"])
        surprise = erow["surprise_pct"]

        sym_mask = df["symbol"] == sym
        if not sym_mask.any():
            continue

        sym_dates = trade_dates.loc[sym_mask]

        # PEAD window: [earnings_date + 1, earnings_date + pead_window_days]
        days_since = (sym_dates - edate).dt.days
        in_pead = (days_since >= 1) & (days_since <= pead_window_days)

        if not bool(in_pead.any()):
            continue

        pead_indices = sym_dates.index[in_pead]

        if surprise > 0:
            df.loc[pead_indices, "confidence"] *= positive_boost
            n_boosted += len(pead_indices)
        else:
            df.loc[pead_indices, "confidence"] *= negative_penalty
            n_penalized += len(pead_indices)

    # Clip confidence to [0, 1]
    df["confidence"] = df["confidence"].clip(0.0, 1.0)

    if n_boosted > 0 or n_penalized > 0:
        print(
            f"  PEAD boost: {n_boosted:,} boosted, {n_penalized:,} penalized "
            f"(window={pead_window_days}d, min_surprise={min_surprise_pct}%)"
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
    confidence_weight: bool = False,
    earnings_filter: bool = False,
    pead_boost: bool = False,
    semiconductor_short_gate: bool = False,
    gate_threshold: float = 0.0,
) -> dict:
    """
    Simple equal-weight portfolio backtest (or confidence-weighted if enabled).

    When hold_days=1: Vectorized daily rebalance (fast).
    When hold_days>1: Day-by-day simulation that holds positions for N days
                      and only rebalances when positions expire.

    On each rebalance day:
      1. Rank active signals by confidence
      2. Select top-K (or all, up to max_positions)
      3. Equal-weight or confidence-weighted allocation
      4. Hold for hold_days trading days
      5. Compute returns after transaction costs

    Args:
        df: DataFrame with 'date', 'symbol', 'raw_signal', 'confidence',
            'forward_return', 'y_proba', 'y_true'
        max_positions: Maximum concurrent positions
        commission_bps: Commission in basis points (each way). 0 = free.
        hold_days: Minimum holding period in days (rebalance every N days)
        top_k: If set, only trade top-K most confident signals per day
        initial_capital: Starting capital for equity curve
        start_date: Filter predictions from this date
        end_date: Filter predictions to this date
        confidence_weight: If True, weight positions by confidence instead of equal weight
        earnings_filter: If True, skip trades where earnings fall within hold period
        pead_boost: If True, boost confidence for stocks with recent positive earnings surprise
        semiconductor_short_gate: If True, disable semi shorts unless AI regime is weak
        gate_threshold: AI regime threshold below which semi shorts are allowed
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

    # --- Long-horizon semiconductor short gate ---
    if semiconductor_short_gate:
        df = _apply_semiconductor_short_gate(df, gate_threshold=gate_threshold)

    calendar: pd.DataFrame | None = None

    # --- Earnings avoidance filter ---
    if earnings_filter:
        load_earnings_calendar, get_earnings_mask = _load_earnings_helpers()
        calendar = load_earnings_calendar()
        if not calendar.empty:
            earn_mask = get_earnings_mask(
                df, hold_days=max(hold_days, 1), calendar=calendar
            )
            n_filtered = earn_mask.sum()
            n_active_before = (df["raw_signal"] != 0).sum()
            df.loc[earn_mask, "raw_signal"] = 0.0
            n_active_after = (df["raw_signal"] != 0).sum()
            if n_filtered > 0:
                print(
                    f"  Earnings filter: {n_filtered:,} signals removed "
                    f"({n_active_before:,} -> {n_active_after:,} active)"
                )

    # --- PEAD confidence boost ---
    if pead_boost:
        if calendar is None:
            load_earnings_calendar, _ = _load_earnings_helpers()
            calendar = load_earnings_calendar()
        cal = calendar
        if cal is not None and not cal.empty:
            df = _apply_pead_boost(df, cal)

    if hold_days <= 1:
        return _backtest_daily_rebalance(
            df,
            max_positions,
            commission_bps,
            top_k,
            initial_capital,
            confidence_weight=confidence_weight,
        )
    else:
        return _backtest_hold_period(
            df,
            max_positions,
            commission_bps,
            hold_days,
            top_k,
            initial_capital,
            confidence_weight=confidence_weight,
        )


def _backtest_daily_rebalance(
    df: pd.DataFrame,
    max_positions: int,
    commission_bps: float,
    top_k: int | None,
    initial_capital: float,
    confidence_weight: bool = False,
) -> dict:
    """Original vectorized daily-rebalance backtest (hold_days=1)."""

    # --- Vectorized approach: rank within each day, filter top-K ---
    active = df[df["raw_signal"] != 0.0].copy()

    if len(active) == 0:
        return {"error": "No active signals after filtering"}

    # Rank by confidence within each date (1 = highest confidence)
    active["conf_rank"] = active.groupby("date")["confidence"].rank(
        ascending=False, method="first"
    )

    cap = max_positions
    if top_k is not None:
        cap = min(cap, top_k)
    active = active[active["conf_rank"] <= cap]

    # Count positions per day
    pos_counts = active.groupby("date").size()

    # Position weighting: confidence-weighted or equal-weight
    active = active.merge(
        pos_counts.rename("n_pos").reset_index(), on="date", how="left"
    )
    if confidence_weight:
        # Weight by confidence, normalized within each day to sum to 1.0
        active["weight"] = active.groupby("date")["confidence"].transform(
            lambda c: c / c.sum() if c.sum() > 0 else 1.0 / len(c)
        )
    else:
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

    # Fill missing dates
    all_dates = sorted(df["date"].unique())
    date_idx = pd.DatetimeIndex(all_dates)
    daily = daily.reindex(date_idx, fill_value=0)
    daily.index.name = "date"

    # --- Transaction costs ---
    commission_rate = commission_bps / 10_000.0

    if commission_rate > 0:
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
    daily_turnover_arr = np.zeros(n_dates)
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


def _backtest_hold_period(
    df: pd.DataFrame,
    max_positions: int,
    commission_bps: float,
    hold_days: int,
    top_k: int | None,
    initial_capital: float,
    confidence_weight: bool = False,
) -> dict:
    """
    Hold-period simulation for multi-day forward returns.

    Rebalances only every hold_days trading days:
    - On rebalance days: select new positions from available signals
    - Realize the stored forward_return once on the entry/rebalance date
    - On non-rebalance days: hold positions but do not re-apply the same
      multi-day forward return again
    - Track turnover properly for commission calculation

    Optimized: uses vectorized dict-building and groupby instead of iterrows().
    """
    commission_rate = commission_bps / 10_000.0
    cap = max_positions
    if top_k is not None:
        cap = min(cap, top_k)

    # Build lookup: date -> list of (symbol, signal, confidence, forward_return, y_true)
    active = df[df["raw_signal"] != 0.0].copy()
    if len(active) == 0:
        return {"error": "No active signals after filtering"}

    all_dates = sorted(df["date"].unique())
    n_dates = len(all_dates)

    # Build return lookup: (date, symbol) -> forward_return  [VECTORIZED]
    keys = list(zip(df["date"].values, df["symbol"].values))
    return_lookup = dict(zip(keys, df["forward_return"].values))
    ytrue_lookup = dict(zip(keys, df["y_true"].values))

    # Build signal candidates by date [VECTORIZED via groupby + rank]
    active["conf_rank"] = active.groupby("date")["confidence"].rank(
        ascending=False, method="first"
    )
    top_active = active[active["conf_rank"] <= cap].copy()

    date_candidates = {}
    if len(top_active) > 0:
        # Group once, extract arrays per group
        for date, grp in top_active.groupby("date"):
            # Sort by rank (already ranked, just sort)
            grp_sorted = grp.sort_values("conf_rank")
            date_candidates[date] = list(
                zip(
                    grp_sorted["symbol"].values,
                    grp_sorted["raw_signal"].values,
                    grp_sorted["confidence"].values,
                )
            )

    # Day-by-day simulation
    daily_returns = np.zeros(n_dates)
    daily_costs = np.zeros(n_dates)
    daily_n_positions = np.zeros(n_dates)
    daily_turnover = np.zeros(n_dates)

    # current_positions: dict symbol -> (signal, confidence)
    current_positions = {}
    # Pre-allocate trade record arrays for speed
    tr_dates = []
    tr_symbols = []
    tr_signals = []
    tr_fwd_rets = []
    tr_weights = []
    tr_pnls = []
    tr_corrects = []

    for i, date in enumerate(all_dates):
        # Check if we need to rebalance
        rebalance = (i == 0) or (i % hold_days == 0) or (not current_positions)

        if rebalance:
            # Select new positions
            new_cands = date_candidates.get(date, [])
            old_symbols = set(current_positions.keys())
            new_positions = {}
            for sym, sig, conf in new_cands:
                new_positions[sym] = (sig, conf)

            new_symbols = set(new_positions.keys())
            # Turnover: positions that changed
            changed = len(old_symbols.symmetric_difference(new_symbols))
            # Also count signal direction changes for same symbol
            for sym in old_symbols & new_symbols:
                if current_positions[sym][0] != new_positions[sym][0]:
                    changed += 1  # Direction flip counts as 2 trades (exit + re-enter)

            current_positions = new_positions
            n_pos = len(current_positions)

            if n_pos > 0 and commission_rate > 0:
                w = 1.0 / n_pos
                daily_costs[i] = changed * w * commission_rate * 2
                daily_turnover[i] = changed / max(n_pos, 1)

        # For multi-day prediction panels, realize the stored forward return once
        # on rebalance/entry dates only. Re-applying it on every held day would
        # massively overstate both gains and losses.
        n_pos = len(current_positions)
        daily_n_positions[i] = n_pos

        if rebalance and n_pos > 0:
            # Compute weights: confidence-weighted or equal-weight
            if confidence_weight:
                conf_sum = sum(conf for (_, conf) in current_positions.values())
                if conf_sum <= 0:
                    conf_sum = 1.0  # Fallback to avoid division by zero
            else:
                conf_sum = 0.0  # Not used in equal-weight mode

            day_return = 0.0
            for sym, (sig, conf) in current_positions.items():
                if confidence_weight:
                    w = conf / conf_sum
                else:
                    w = 1.0 / n_pos
                fwd_ret = return_lookup.get((date, sym), 0.0)
                pnl = sig * fwd_ret * w
                day_return += pnl
                # Track trades (append to lists, not dicts)
                tr_dates.append(date)
                tr_symbols.append(sym)
                tr_signals.append(sig)
                tr_fwd_rets.append(fwd_ret)
                tr_weights.append(w)
                tr_pnls.append(pnl)
                yt = ytrue_lookup.get((date, sym), 0)
                tr_corrects.append((sig > 0) == (yt == 1))
            daily_returns[i] = day_return

    daily_net_returns = daily_returns - daily_costs

    # Build trade log from arrays (much faster than list-of-dicts)
    if tr_dates:
        trade_df = pd.DataFrame(
            {
                "date": tr_dates,
                "symbol": tr_symbols,
                "signal": tr_signals,
                "forward_return": tr_fwd_rets,
                "weight": tr_weights,
                "pnl": tr_pnls,
                "correct": tr_corrects,
            }
        )
    else:
        trade_df = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "signal",
                "forward_return",
                "weight",
                "pnl",
                "correct",
            ]
        )

    # Equity curve
    equity = initial_capital * np.cumprod(1 + daily_net_returns)
    date_idx = pd.DatetimeIndex(all_dates)

    results = compute_metrics(
        daily_net_returns,
        daily_costs,
        np.array(date_idx),
        equity,
        trade_df,
        daily_n_positions,
        daily_turnover,
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
    signal_threshold: float = 0.53,
    short_threshold: float | None = None,
    confidence_threshold: float = 0.0,
    long_only: bool = False,
    commission_bps: float = 0.0,
    hold_days: int = 1,
    max_positions: int = 20,
    confidence_weight: bool = False,
    top_k: int | None = None,
    earnings_filter: bool = False,
    pead_boost: bool = False,
    semiconductor_short_gate: bool = False,
    gate_threshold: float = 0.0,
) -> pd.DataFrame:
    """Break down performance by calendar year."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    sig_df = compute_signals(
        df,
        signal_threshold=signal_threshold,
        short_threshold=short_threshold,
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
            top_k=top_k,
            confidence_weight=confidence_weight,
            earnings_filter=earnings_filter,
            pead_boost=pead_boost,
            semiconductor_short_gate=semiconductor_short_gate,
            gate_threshold=gate_threshold,
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
        "--prediction-file",
        type=str,
        default="walk_forward_predictions.pkl",
        help="Prediction filename inside checkpoint dir",
    )
    parser.add_argument(
        "--blend-checkpoint-dirs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated extra checkpoint dirs. "
            "Their y_proba values are blended with primary predictions."
        ),
    )
    parser.add_argument(
        "--blend-weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated blend weights. "
            "Provide N weights for primary+extras, or N-1 for extras only."
        ),
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.53,
        help="Min probability for long/short (default: 0.53)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Max probability for short entries (default: 1 - signal-threshold)",
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
        default=10,
        help="Only trade top-K most confident signals per day (default: 10)",
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
    parser.add_argument(
        "--conf-weight",
        action="store_true",
        help="Weight positions by confidence instead of equal weight",
    )
    parser.add_argument(
        "--earnings-filter",
        action="store_true",
        help="Skip trades where earnings fall within hold period (requires earnings data)",
    )
    parser.add_argument(
        "--pead-boost",
        action="store_true",
        help=(
            "Boost confidence after significant earnings surprise "
            "(post-earnings drift signal)"
        ),
    )
    parser.add_argument(
        "--semiconductor-short-gate",
        action="store_true",
        help="Only allow semiconductor shorts when AI regime is weak",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.0,
        help="AI regime threshold for semiconductor short gate (default: 0.0)",
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

    df = load_predictions(ckpt_dir, prediction_file=args.prediction_file)
    df = deduplicate_predictions(df)

    # Optional cross-horizon probability blending
    if args.blend_checkpoint_dirs:
        blend_dirs_raw = [
            x.strip() for x in args.blend_checkpoint_dirs.split(",") if x.strip()
        ]
        blend_dirs = []
        for p in blend_dirs_raw:
            pp = Path(p)
            if not pp.is_absolute():
                pp = Path(__file__).parent / pp
            blend_dirs.append(pp)

        blend_weights = None
        if args.blend_weights:
            blend_weights = [
                float(x) for x in args.blend_weights.split(",") if x.strip()
            ]
        df = blend_prediction_probabilities(
            primary_df=df,
            checkpoint_dirs=blend_dirs,
            blend_weights=blend_weights,
        )

    # Filter symbols
    if args.symbols:
        df = df[df["symbol"].isin(args.symbols)]
        print(f"  Filtered to symbols: {args.symbols}")
        print(f"  Remaining predictions: {len(df):,}")

    # Print config
    print(f"\n  Config:")
    print(f"    Signal threshold:   {args.signal_threshold:.2f}")
    if args.short_threshold is None:
        print(f"    Short threshold:    {1.0 - args.signal_threshold:.2f} (auto)")
    else:
        print(f"    Short threshold:    {args.short_threshold:.2f}")
    print(f"    Confidence min:     {args.confidence:.2f}")
    print(f"    Commission:         {args.commission:.1f} bps each way")
    print(f"    Hold days:          {args.hold_days}")
    print(f"    Top-K:              {args.top_k or 'all'}")
    print(f"    Max positions:      {args.max_positions}")
    print(f"    Long only:          {args.long_only}")
    print(f"    Conf weight:        {args.conf_weight}")
    print(f"    Earnings filter:    {args.earnings_filter}")
    print(f"    PEAD boost:         {args.pead_boost}")
    print(f"    Semi short gate:    {args.semiconductor_short_gate}")
    print(f"    Gate threshold:     {args.gate_threshold:.2f}")
    print(
        f"    Blend checkpoints:  "
        f"{args.blend_checkpoint_dirs if args.blend_checkpoint_dirs else 'none'}"
    )
    print(f"    Blend weights:      {args.blend_weights if args.blend_weights else 'auto'}")
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
            short_threshold=args.short_threshold,
            confidence_threshold=args.confidence,
            long_only=args.long_only,
            commission_bps=args.commission,
            hold_days=args.hold_days,
            max_positions=args.max_positions,
            confidence_weight=args.conf_weight,
            top_k=args.top_k,
            earnings_filter=args.earnings_filter,
            pead_boost=args.pead_boost,
            semiconductor_short_gate=args.semiconductor_short_gate,
            gate_threshold=args.gate_threshold,
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
        short_threshold=args.short_threshold,
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
        confidence_weight=args.conf_weight,
        earnings_filter=args.earnings_filter,
        pead_boost=args.pead_boost,
        semiconductor_short_gate=args.semiconductor_short_gate,
        gate_threshold=args.gate_threshold,
    )

    config_label = (
        f"Signal >= {args.signal_threshold}, Confidence >= {args.confidence:.0%}, "
        f"Commission={args.commission:.0f}bps, Hold={args.hold_days}d, "
        f"{'Long Only' if args.long_only else 'Long+Short'}"
        f"{', ConfWeight' if args.conf_weight else ''}"
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
                confidence_weight=args.conf_weight,
                earnings_filter=args.earnings_filter,
                pead_boost=args.pead_boost,
                semiconductor_short_gate=args.semiconductor_short_gate,
                gate_threshold=args.gate_threshold,
            )
            print(
                f"  @{c_bps:>2}bps commission: Return={res_c['total_return']:+.2%}, "
                f"Sharpe={res_c['sharpe']:.3f}, "
                f"Cost={res_c['total_commission_cost']:.4%}"
            )
        print()


if __name__ == "__main__":
    main()
