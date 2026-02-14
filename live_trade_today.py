#!/usr/bin/env python3
"""
Live Meta-Ensemble Paper Trader
================================
Runs the meta-ensemble model on real market data to generate
predictions for today's trading session. Tracks actual P&L.

This is a PAPER TRADE — no real money. Uses yfinance for data.

Usage:
  # Generate predictions for next trading day (run after market close)
  python live_trade_today.py --predict

  # Check results at end of day
  python live_trade_today.py --results

  # Full cycle: predict + monitor through the day
  python live_trade_today.py --predict --monitor

  # Validate against a past date (backtest single day)
  python live_trade_today.py --validate-date 2026-02-13
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/home/regulus/Trade")

# Paths
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "meta_ensemble"
TRADES_DIR = PROJECT_ROOT / "data_store" / "live_trades"
TRADES_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load the trained meta-ensemble model."""
    model_path = CHECKPOINT_DIR / "meta_ensemble_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}")

    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]

    print(f"  Model loaded: {len(feature_cols)} features")
    print(f"  Trained at: {checkpoint.get('trained_at', 'unknown')}")
    print(f"  Training samples: {checkpoint.get('n_samples', 'unknown'):,}")

    return model, scaler, feature_cols


def fetch_ohlcv(symbol: str, days: int = 600) -> pd.DataFrame | None:
    """Fetch OHLCV data from yfinance. Need ~600 days for SMA200 warmup."""
    try:
        ticker = yf.Ticker(symbol)
        end = datetime.now()
        start = end - timedelta(days=days)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if df is None or len(df) < 300:
            return None

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        # Keep only what we need
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                return None

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["returns"] = df["close"].pct_change()
        df = df.dropna(subset=["returns"])

        return df
    except Exception as e:
        print(f"    WARN: Failed to fetch {symbol}: {e}")
        return None


def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Compute all 65 features for a single symbol."""
    try:
        from quantum_alpha.meta_ensemble import compute_features_single_symbol

        featured = compute_features_single_symbol(symbol, df)
        if featured is None or len(featured) == 0:
            return None
        return featured
    except Exception as e:
        print(f"    WARN: Feature computation failed for {symbol}: {e}")
        return None


def predict_single_symbol(
    df_features: pd.DataFrame, model, scaler, feature_cols: list[str]
) -> dict | None:
    """Generate prediction for the most recent date."""
    # Get the last row (most recent date)
    if len(df_features) == 0:
        return None

    last_row = df_features.iloc[[-1]]
    date = last_row.index[0]

    # Align features
    X = pd.DataFrame(index=last_row.index)
    for col in feature_cols:
        if col in last_row.columns:
            X[col] = last_row[col].values
        else:
            X[col] = 0.0  # Missing feature

    # Scale
    X_vals = X.values.astype(np.float64)
    X_vals = np.nan_to_num(X_vals, nan=0.0, posinf=3.0, neginf=-3.0)
    X_scaled = scaler.transform(X_vals)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    # Predict
    proba = model.predict_proba(X_scaled)
    up_prob = float(proba[0, 1])
    confidence = abs(up_prob - 0.5) * 2.0

    return {
        "date": str(date.date()) if hasattr(date, "date") else str(date)[:10],
        "up_probability": up_prob,
        "confidence": confidence,
        "signal": "LONG" if up_prob > 0.5 else "SHORT",
    }


def get_liquid_sp500(n: int = 50) -> list[str]:
    """Get top-N most liquid S&P 500 symbols."""
    # Hardcoded top-50 most liquid S&P 500 stocks by volume
    # (to avoid slow Wikipedia scraping)
    top50 = [
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "GOOGL",
        "META",
        "TSLA",
        "BRK-B",
        "UNH",
        "JNJ",
        "JPM",
        "V",
        "XOM",
        "PG",
        "MA",
        "HD",
        "CVX",
        "MRK",
        "ABBV",
        "LLY",
        "PEP",
        "KO",
        "COST",
        "AVGO",
        "WMT",
        "MCD",
        "CSCO",
        "ACN",
        "TMO",
        "ABT",
        "CRM",
        "DHR",
        "LIN",
        "NEE",
        "TXN",
        "PM",
        "AMGN",
        "UNP",
        "ORCL",
        "AMD",
        "INTC",
        "HON",
        "QCOM",
        "CMCSA",
        "LOW",
        "BA",
        "CAT",
        "GS",
        "BLK",
        "SPY",
    ]
    return top50[:n]


def run_predictions(
    symbols: list[str] | None = None,
    n_symbols: int = 30,
    min_confidence: float = 0.05,
) -> pd.DataFrame:
    """Run model predictions for all symbols."""
    print("\n" + "=" * 65)
    print("  GENERATING PREDICTIONS")
    print("=" * 65)

    model, scaler, feature_cols = load_model()

    if symbols is None:
        symbols = get_liquid_sp500(n_symbols)

    print(f"  Scanning {len(symbols)} symbols...")
    print()

    results = []
    for i, sym in enumerate(symbols):
        print(f"  [{i + 1}/{len(symbols)}] {sym}...", end=" ", flush=True)

        df = fetch_ohlcv(sym, days=700)
        if df is None:
            print("SKIP (insufficient data)")
            continue

        features = compute_features(df, sym)
        if features is None:
            print("SKIP (feature error)")
            continue

        pred = predict_single_symbol(features, model, scaler, feature_cols)
        if pred is None:
            print("SKIP (prediction error)")
            continue

        last_close = df["close"].iloc[-1]
        pred["symbol"] = sym
        pred["last_close"] = float(last_close)
        results.append(pred)

        print(
            f"P(up)={pred['up_probability']:.3f} "
            f"conf={pred['confidence']:.3f} "
            f"-> {pred['signal']}"
        )

    if not results:
        print("\n  ERROR: No successful predictions!")
        return pd.DataFrame()

    pred_df = pd.DataFrame(results)
    pred_df = pred_df.sort_values("confidence", ascending=False)

    return pred_df


def select_trades(
    pred_df: pd.DataFrame,
    max_positions: int = 10,
    min_confidence: float = 0.05,
    long_only: bool = True,
) -> pd.DataFrame:
    """Select the best trades for today."""
    # Filter by confidence
    trades = pred_df[pred_df["confidence"] >= min_confidence].copy()

    if long_only:
        trades = trades[trades["up_probability"] > 0.5]

    # Take top-N by confidence
    trades = trades.head(max_positions)

    # Equal weight
    n = len(trades)
    if n > 0:
        trades["weight"] = 1.0 / n
    else:
        trades["weight"] = 0.0

    return trades


def print_trade_plan(trades: pd.DataFrame, capital: float = 100_000.0):
    """Print the trade plan."""
    print("\n" + "=" * 65)
    print("  TRADE PLAN FOR TODAY")
    print("=" * 65)

    if len(trades) == 0:
        print("  No trades meet confidence threshold.")
        return

    print(
        f"  {'Symbol':>8} | {'Signal':>6} | {'P(up)':>7} | {'Conf':>7} | "
        f"{'Weight':>7} | {'$ Alloc':>10} | {'Shares':>7}"
    )
    print(
        f"  {'-' * 8}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}-+-"
        f"{'-' * 7}-+-{'-' * 10}-+-{'-' * 7}"
    )

    total_allocated = 0
    for _, row in trades.iterrows():
        alloc = capital * row["weight"]
        shares = int(alloc / row["last_close"])
        actual_alloc = shares * row["last_close"]
        total_allocated += actual_alloc

        print(
            f"  {row['symbol']:>8} | {row['signal']:>6} | "
            f"{row['up_probability']:>6.3f} | {row['confidence']:>6.3f} | "
            f"{row['weight']:>6.1%} | ${alloc:>9,.0f} | {shares:>7,}"
        )

    print(f"\n  Total positions: {len(trades)}")
    print(f"  Capital allocated: ${total_allocated:,.0f} / ${capital:,.0f}")
    print("=" * 65)


def save_trade_plan(trades: pd.DataFrame, prediction_date: str):
    """Save trade plan to disk for later verification."""
    filepath = TRADES_DIR / f"trades_{prediction_date}.json"

    records = trades.to_dict(orient="records")
    plan = {
        "prediction_date": prediction_date,
        "generated_at": datetime.now().isoformat(),
        "n_positions": len(trades),
        "trades": records,
    }

    with open(filepath, "w") as f:
        json.dump(plan, f, indent=2, default=str)

    print(f"\n  Trade plan saved to: {filepath}")
    return filepath


def check_results(trade_date: str | None = None):
    """Check results for a previously saved trade plan."""
    if trade_date is None:
        # Find most recent trade plan
        plans = sorted(TRADES_DIR.glob("trades_*.json"))
        if not plans:
            print("No trade plans found.")
            return
        filepath = plans[-1]
    else:
        filepath = TRADES_DIR / f"trades_{trade_date}.json"

    if not filepath.exists():
        print(f"No trade plan found at {filepath}")
        return

    with open(filepath) as f:
        plan = json.load(f)

    print(f"\n{'=' * 65}")
    print(f"  RESULTS FOR {plan['prediction_date']}")
    print(f"{'=' * 65}")
    print(f"  Plan generated: {plan['generated_at']}")
    print(f"  Positions: {plan['n_positions']}")
    print()

    trades = plan["trades"]
    if not trades:
        print("  No trades in plan.")
        return

    # Fetch today's actual prices
    symbols = [t["symbol"] for t in trades]
    total_pnl_pct = 0.0
    n_correct = 0
    n_total = 0

    print(
        f"  {'Symbol':>8} | {'Predicted':>9} | {'P(up)':>7} | "
        f"{'Open':>9} | {'Close':>9} | {'Return':>8} | {'Correct':>8}"
    )
    print(
        f"  {'-' * 8}-+-{'-' * 9}-+-{'-' * 7}-+-"
        f"{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 8}"
    )

    for trade in trades:
        sym = trade["symbol"]
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
            if hist is None or len(hist) == 0:
                print(f"  {sym:>8} | {'N/A':>9}")
                continue

            # Get the most recent day's data
            latest = hist.iloc[-1]
            open_price = latest["Open"]
            close_price = latest["Close"]
            day_return = (close_price - open_price) / open_price

            predicted_up = trade["signal"] == "LONG"
            actual_up = day_return > 0
            correct = predicted_up == actual_up

            weighted_return = day_return * trade["weight"]
            if trade["signal"] == "SHORT":
                weighted_return = -weighted_return

            total_pnl_pct += weighted_return
            n_total += 1
            if correct:
                n_correct += 1

            print(
                f"  {sym:>8} | {trade['signal']:>9} | "
                f"{trade['up_probability']:>6.3f} | "
                f"${open_price:>8.2f} | ${close_price:>8.2f} | "
                f"{day_return:>+7.2%} | "
                f"{'YES' if correct else 'NO':>8}"
            )

        except Exception as e:
            print(f"  {sym:>8} | ERROR: {e}")

    print(f"\n  Portfolio return: {total_pnl_pct:+.4%}")
    print(f"  Accuracy: {n_correct}/{n_total} = {n_correct / max(n_total, 1):.0%}")
    pnl_dollars = total_pnl_pct * 100_000
    print(f"  On $100,000: ${pnl_dollars:+,.0f}")
    print("=" * 65)


def validate_past_date(target_date: str, n_symbols: int = 30):
    """
    Validate the model against a specific past date.
    Uses data up to the day BEFORE target_date for prediction,
    then checks against actual return on target_date.
    """
    target_dt = pd.Timestamp(target_date)
    print(f"\n{'=' * 65}")
    print(f"  VALIDATING MODEL ON {target_date}")
    print(f"  (Using data available BEFORE this date)")
    print(f"{'=' * 65}")

    model, scaler, feature_cols = load_model()
    symbols = get_liquid_sp500(n_symbols)

    results = []
    for i, sym in enumerate(symbols):
        print(f"  [{i + 1}/{len(symbols)}] {sym}...", end=" ", flush=True)

        # Fetch data up to and including target_date
        end_dt = target_dt + timedelta(days=5)
        start_dt = target_dt - timedelta(days=700)

        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
            )
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        if df is None or len(df) < 300:
            print("SKIP (insufficient data)")
            continue

        df.columns = [c.lower() for c in df.columns]
        if "open" not in df.columns or "close" not in df.columns:
            print("SKIP (missing columns)")
            continue

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["returns"] = df["close"].pct_change()
        df = df.dropna(subset=["returns"])

        # Split: use data BEFORE target_date for features
        # Check actual return on target_date
        df.index = df.index.tz_localize(None) if df.index.tz else df.index

        pre_target = df[df.index < target_dt]
        on_target = df[
            (df.index >= target_dt) & (df.index < target_dt + timedelta(days=3))
        ]

        if len(pre_target) < 300 or len(on_target) == 0:
            print("SKIP (date not found)")
            continue

        # Compute features using ONLY pre-target data
        features = compute_features(pre_target, sym)
        if features is None or len(features) == 0:
            print("SKIP (feature error)")
            continue

        pred = predict_single_symbol(features, model, scaler, feature_cols)
        if pred is None:
            print("SKIP")
            continue

        # Actual return on target date (open-to-close)
        actual_open = on_target.iloc[0]["open"]
        actual_close = on_target.iloc[0]["close"]
        actual_return = (actual_close - actual_open) / actual_open

        # Close-to-close return (what the model actually predicts)
        prev_close = pre_target.iloc[-1]["close"]
        actual_cc_return = (actual_close - prev_close) / prev_close

        predicted_up = pred["up_probability"] > 0.5
        actual_up = actual_cc_return > 0

        results.append(
            {
                "symbol": sym,
                "up_prob": pred["up_probability"],
                "confidence": pred["confidence"],
                "signal": pred["signal"],
                "actual_oc_return": actual_return,
                "actual_cc_return": actual_cc_return,
                "correct": predicted_up == actual_up,
                "open": actual_open,
                "close": actual_close,
            }
        )

        mark = "OK" if predicted_up == actual_up else "WRONG"
        print(
            f"P(up)={pred['up_probability']:.3f} actual={actual_cc_return:+.2%} [{mark}]"
        )

    if not results:
        print("\n  No results!")
        return

    res_df = pd.DataFrame(results)

    # Overall accuracy
    accuracy = res_df["correct"].mean()
    n_correct = res_df["correct"].sum()
    n_total = len(res_df)

    print(f"\n{'=' * 65}")
    print(f"  VALIDATION RESULTS FOR {target_date}")
    print(f"{'=' * 65}")
    print(f"  Overall accuracy: {n_correct}/{n_total} = {accuracy:.1%}")

    # Portfolio simulation: long top-10 by confidence
    top10 = res_df.sort_values("confidence", ascending=False).head(10)
    top10_longs = top10[top10["up_prob"] > 0.5]

    if len(top10_longs) > 0:
        port_return = top10_longs["actual_cc_return"].mean()
        port_accuracy = top10_longs["correct"].mean()
        print(f"\n  Top-{len(top10_longs)} confident LONG trades:")
        print(f"    Accuracy: {port_accuracy:.0%}")
        print(f"    Avg return: {port_return:+.2%}")
        pnl_dollar = port_return * 100_000
        print(f"    On $100K: {'$' if pnl_dollar >= 0 else '-$'}{abs(pnl_dollar):,.0f}")

        for _, r in top10_longs.iterrows():
            mark = "OK" if r["correct"] else "WRONG"
            print(
                f"    {r['symbol']:>6} P(up)={r['up_prob']:.3f} "
                f"ret={r['actual_cc_return']:+.2%} [{mark}]"
            )

    # By confidence bucket
    print(f"\n  By confidence level:")
    for conf_min in [0.0, 0.05, 0.10, 0.15, 0.20]:
        bucket = res_df[res_df["confidence"] >= conf_min]
        longs = bucket[bucket["up_prob"] > 0.5]
        if len(longs) > 0:
            acc = longs["correct"].mean()
            ret = longs["actual_cc_return"].mean()
            print(
                f"    conf>={conf_min:.0%}: {len(longs)} trades, "
                f"acc={acc:.0%}, ret={ret:+.2%}"
            )

    # Market benchmark (SPY return on that day)
    spy_row = res_df[res_df["symbol"] == "SPY"]
    if len(spy_row) > 0:
        spy_ret = spy_row.iloc[0]["actual_cc_return"]
        print(f"\n  SPY return: {spy_ret:+.2%}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Live Meta-Ensemble Paper Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Generate predictions for next trading day",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Check results for most recent trade plan",
    )
    parser.add_argument(
        "--results-date", type=str, default=None, help="Check results for specific date"
    )
    parser.add_argument(
        "--validate-date",
        type=str,
        default=None,
        help="Validate model on a past date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=30,
        help="Number of symbols to scan (default: 30)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Max positions to take (default: 10)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.05,
        help="Min confidence to trade (default: 0.05)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Paper trading capital (default: $100,000)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=True,
        help="Only take long positions (default: True)",
    )

    args = parser.parse_args()

    if args.validate_date:
        validate_past_date(args.validate_date, n_symbols=args.n_symbols)
        return

    if args.results or args.results_date:
        check_results(args.results_date)
        return

    if args.predict:
        pred_df = run_predictions(n_symbols=args.n_symbols)
        if len(pred_df) == 0:
            return

        trades = select_trades(
            pred_df,
            max_positions=args.max_positions,
            min_confidence=args.min_confidence,
            long_only=args.long_only,
        )

        print_trade_plan(trades, capital=args.capital)

        # Save for later verification
        pred_date = trades.iloc[0]["date"] if len(trades) > 0 else "unknown"
        save_trade_plan(trades, pred_date)

        # Also print ALL predictions sorted
        print(f"\n{'=' * 65}")
        print("  ALL PREDICTIONS (sorted by confidence)")
        print(f"{'=' * 65}")
        for _, row in pred_df.iterrows():
            print(
                f"  {row['symbol']:>8} | P(up)={row['up_probability']:.3f} | "
                f"Conf={row['confidence']:.3f} | {row['signal']}"
            )
        print(f"{'=' * 65}")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
