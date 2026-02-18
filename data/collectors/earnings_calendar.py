"""
Earnings Calendar Fetcher
==========================
Fetches earnings announcement dates and surprise data from yfinance
for all symbols in the meta-ensemble universe.

Coverage: ~6 years back (yfinance provides ~25 quarters)
Source: yfinance .earnings_dates property
No API key required.

Output: Pickle file at data_store/earnings/earnings_calendar.pkl
        DataFrame with columns: symbol, earnings_date, eps_estimate,
        reported_eps, surprise_pct

Usage:
    python -m quantum_alpha.data.collectors.earnings_calendar
    python -m quantum_alpha.data.collectors.earnings_calendar --symbols AAPL MSFT TSLA
    python -m quantum_alpha.data.collectors.earnings_calendar --refresh
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data_store" / "earnings"
OUTPUT_FILE = OUTPUT_DIR / "earnings_calendar.pkl"
RAW_CACHE_DIR = OUTPUT_DIR / "raw_cache"

REQUEST_DELAY = 1.0  # seconds between symbols (yfinance rate limit)

# Active symbols in the meta-ensemble (those with OHLCV data)
OHLCV_CACHE_DIR = (
    Path(__file__).parent.parent.parent / "data_store" / "meta_ensemble" / "ohlcv_cache"
)


def get_active_symbols() -> list[str]:
    """Get list of symbols that have cached OHLCV data."""
    if not OHLCV_CACHE_DIR.exists():
        logger.warning("OHLCV cache dir not found: %s", OHLCV_CACHE_DIR)
        return []
    symbols = sorted(p.stem for p in OHLCV_CACHE_DIR.glob("*.pkl"))
    return symbols


def fetch_earnings_for_symbol(symbol: str) -> pd.DataFrame | None:
    """
    Fetch earnings dates for a single symbol from yfinance.

    Returns DataFrame with columns:
        symbol, earnings_date, eps_estimate, reported_eps, surprise_pct

    Returns None if no data available (e.g. ETFs).
    """
    try:
        ticker = yf.Ticker(symbol)
        ed = ticker.earnings_dates

        if ed is None or len(ed) == 0:
            logger.info("%s: No earnings dates available", symbol)
            return None

        # Normalize timezone-aware index to date-only
        df = ed.reset_index()
        df.columns = [
            "earnings_datetime",
            "eps_estimate",
            "reported_eps",
            "surprise_pct",
        ]

        # Convert to date (strip time + timezone)
        earnings_ts = pd.to_datetime(df["earnings_datetime"], errors="coerce")
        df["earnings_date"] = earnings_ts.dt.normalize()
        df["symbol"] = symbol

        # Select and order columns
        df = df[
            ["symbol", "earnings_date", "eps_estimate", "reported_eps", "surprise_pct"]
        ].copy()

        # Sort by date
        df = df.sort_values("earnings_date").reset_index(drop=True)

        return df

    except Exception as e:
        logger.warning("%s: Error fetching earnings dates: %s", symbol, e)
        return None


def fetch_all_earnings(
    symbols: list[str] | None = None,
    refresh: bool = False,
    delay: float = REQUEST_DELAY,
) -> pd.DataFrame:
    """
    Fetch earnings dates for all symbols.

    Args:
        symbols: Specific symbols to fetch. If None, use all active symbols.
        refresh: If True, re-fetch all. If False, skip symbols with cached data.
        delay: Seconds to wait between API calls.

    Returns:
        Combined DataFrame with all earnings data.
    """
    if symbols is None:
        symbols = get_active_symbols()

    if not symbols:
        logger.error("No symbols to fetch")
        return pd.DataFrame()

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    skipped = 0
    fetched = 0
    errors = 0

    for i, symbol in enumerate(symbols):
        cache_file = RAW_CACHE_DIR / f"{symbol}.pkl"

        # Skip if cached and not refreshing
        if not refresh and cache_file.exists():
            try:
                cached_df = pd.read_pickle(cache_file)
                if cached_df is not None and len(cached_df) > 0:
                    all_dfs.append(cached_df)
                skipped += 1
                continue
            except Exception:
                pass  # Re-fetch if cache is corrupted

        # Fetch from yfinance
        if i > 0 and fetched > 0:
            time.sleep(delay)

        print(
            f"  [{i + 1}/{len(symbols)}] Fetching {symbol}...",
            end="",
            flush=True,
        )

        df = fetch_earnings_for_symbol(symbol)

        if df is not None and len(df) > 0:
            # Cache individual symbol
            df.to_pickle(cache_file)
            all_dfs.append(df)
            print(
                f" {len(df)} earnings dates "
                f"({df['earnings_date'].min().date()} to {df['earnings_date'].max().date()})"
            )
            fetched += 1
        else:
            # Cache empty result to avoid re-fetching
            pd.DataFrame().to_pickle(cache_file)
            print(" no data (ETF?)")
            errors += 1

    print(f"\n  Summary: {fetched} fetched, {skipped} cached, {errors} no data")

    if not all_dfs:
        logger.warning("No earnings data collected")
        return pd.DataFrame()

    # Combine all symbols
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)

    # Save combined file
    combined.to_pickle(OUTPUT_FILE)
    print(f"  Saved {len(combined):,} records to {OUTPUT_FILE}")
    print(
        f"  Date range: {combined['earnings_date'].min().date()} "
        f"to {combined['earnings_date'].max().date()}"
    )
    print(f"  Symbols with data: {combined['symbol'].nunique()}")

    return combined


def load_earnings_calendar() -> pd.DataFrame:
    """Load the cached earnings calendar. Returns empty DataFrame if not found."""
    if OUTPUT_FILE.exists():
        df = pd.read_pickle(OUTPUT_FILE)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        return df
    logger.warning("Earnings calendar not found at %s", OUTPUT_FILE)
    return pd.DataFrame()


def has_earnings_in_window(
    symbol: str,
    entry_date: str | pd.Timestamp,
    hold_days: int,
    calendar: pd.DataFrame | None = None,
) -> bool:
    """
    Check if a symbol has an earnings announcement within the hold window.

    Args:
        symbol: Stock ticker
        entry_date: Trade entry date
        hold_days: Number of days to hold
        calendar: Pre-loaded earnings calendar (loaded if None)

    Returns:
        True if earnings fall within [entry_date, entry_date + hold_days]
    """
    if calendar is None:
        calendar = load_earnings_calendar()

    if calendar.empty:
        return False

    entry_dt = pd.to_datetime(entry_date)
    exit_dt = entry_dt + pd.Timedelta(days=hold_days)

    sym_earnings = calendar[calendar["symbol"] == symbol]
    if sym_earnings.empty:
        return False

    # Check if any earnings date falls in the hold window
    mask = (sym_earnings["earnings_date"] >= entry_dt) & (
        sym_earnings["earnings_date"] <= exit_dt
    )
    return mask.any()


def get_earnings_mask(
    df: pd.DataFrame,
    hold_days: int = 12,
    calendar: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Create a boolean mask for rows that have earnings in the hold window.

    Vectorized for performance — avoids per-row lookups.

    Args:
        df: DataFrame with 'symbol' and 'date' columns
        hold_days: Hold period in trading days
        calendar: Pre-loaded earnings calendar

    Returns:
        Boolean Series (True = earnings in window, should be filtered out)
    """
    if calendar is None:
        calendar = load_earnings_calendar()

    if calendar.empty:
        return pd.Series(False, index=df.index)

    trade_dates = pd.to_datetime(df["date"])
    if isinstance(trade_dates, pd.DatetimeIndex):
        trade_dates = pd.Series(trade_dates, index=df.index)
    exit_dates = trade_dates + pd.Timedelta(days=hold_days)

    # Build a set of (symbol, earnings_date) for fast lookup
    earnings_set: dict[str, list[pd.Timestamp]] = {}
    for _, row in calendar.iterrows():
        sym = row["symbol"]
        edate = pd.to_datetime(row["earnings_date"])
        if sym not in earnings_set:
            earnings_set[sym] = []
        earnings_set[sym].append(edate)

    # Sort each symbol's dates for binary search
    for sym in earnings_set:
        earnings_set[sym].sort()

    mask = pd.Series(False, index=df.index)

    for sym, sym_dates in earnings_set.items():
        sym_mask = df["symbol"] == sym
        if not sym_mask.any():
            continue

        sym_indices = df.index[sym_mask]
        sym_entry = trade_dates.loc[sym_mask]
        sym_exit = exit_dates.loc[sym_mask]

        for edate in sym_dates:
            # Vectorized: check if earnings_date falls in [entry, exit]
            in_window = (sym_entry <= edate) & (edate <= sym_exit)
            if bool(in_window.any()):
                mask.loc[sym_indices[in_window.values]] = True

    return mask


def get_pead_signals(
    df: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
    lookback_days: int = 5,
    min_surprise_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Generate Post-Earnings Announcement Drift (PEAD) signals.

    After a stock reports earnings with a significant surprise, the stock
    tends to drift in the direction of the surprise for 20-60 trading days.

    Args:
        df: DataFrame with 'symbol' and 'date' columns
        calendar: Pre-loaded earnings calendar with surprise data
        lookback_days: How many days after earnings to look for PEAD trades
        min_surprise_pct: Minimum absolute surprise % to trigger PEAD signal

    Returns:
        DataFrame with added columns:
            pead_signal: +1 (positive surprise), -1 (negative), 0 (no signal)
            pead_surprise_pct: The surprise % that triggered the signal
            pead_days_since: Days since the earnings announcement
    """
    if calendar is None:
        calendar = load_earnings_calendar()

    result = df.copy()
    result["pead_signal"] = 0
    result["pead_surprise_pct"] = 0.0
    result["pead_days_since"] = np.nan

    if calendar.empty:
        return result

    # Only use historical earnings (with actual surprise data)
    hist = calendar.dropna(subset=["surprise_pct"]).copy()
    if hist.empty:
        return result

    # Filter to significant surprises
    hist = hist[hist["surprise_pct"].abs() >= min_surprise_pct].copy()
    if hist.empty:
        return result

    trade_dates = pd.to_datetime(result["date"])
    if isinstance(trade_dates, pd.DatetimeIndex):
        trade_dates = pd.Series(trade_dates, index=result.index)

    for _, erow in hist.iterrows():
        sym = erow["symbol"]
        edate = pd.to_datetime(erow["earnings_date"])
        surprise = erow["surprise_pct"]

        sym_mask = result["symbol"] == sym
        if not sym_mask.any():
            continue

        sym_dates = trade_dates.loc[sym_mask]

        # PEAD window: [earnings_date + 1, earnings_date + lookback_days]
        # (day after earnings through lookback window)
        days_since = (sym_dates - edate).dt.days
        in_pead_window = (days_since >= 1) & (days_since <= lookback_days)

        if not bool(in_pead_window.any()):
            continue

        pead_indices = sym_dates.index[in_pead_window]

        # Set PEAD signal based on surprise direction
        signal = 1 if surprise > 0 else -1
        result.loc[pead_indices, "pead_signal"] = signal
        result.loc[pead_indices, "pead_surprise_pct"] = surprise
        result.loc[pead_indices, "pead_days_since"] = days_since[in_pead_window].values

    n_pead = (result["pead_signal"] != 0).sum()
    if n_pead > 0:
        n_pos = (result["pead_signal"] > 0).sum()
        n_neg = (result["pead_signal"] < 0).sum()
        logger.info(
            "PEAD signals: %d total (%d positive, %d negative)",
            n_pead,
            n_pos,
            n_neg,
        )

    return result


def get_upcoming_earnings(
    symbols: list[str] | None = None,
    days_ahead: int = 14,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Get symbols with upcoming earnings within the specified window.

    Useful for live trading to identify which positions to avoid.

    Args:
        symbols: Filter to these symbols (None = all)
        days_ahead: Look this many days ahead
        calendar: Pre-loaded earnings calendar

    Returns:
        DataFrame with upcoming earnings sorted by date
    """
    if calendar is None:
        calendar = load_earnings_calendar()

    if calendar.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)

    upcoming = calendar[
        (calendar["earnings_date"] >= today) & (calendar["earnings_date"] <= cutoff)
    ].copy()

    if symbols is not None:
        upcoming = upcoming[upcoming["symbol"].isin(symbols)]

    upcoming = upcoming.sort_values("earnings_date").reset_index(drop=True)
    return upcoming


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fetch earnings calendar from yfinance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to fetch (default: all active)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch all symbols (ignore cache)",
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Show upcoming earnings (next 14 days)",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=14,
        help="Days ahead for --upcoming (default: 14)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.upcoming:
        print("\n" + "=" * 60)
        print("  UPCOMING EARNINGS")
        print("=" * 60)

        upcoming = get_upcoming_earnings(
            symbols=args.symbols,
            days_ahead=args.days_ahead,
        )

        if upcoming.empty:
            print(
                "  No upcoming earnings found. Run without --upcoming first to fetch data."
            )
        else:
            print(f"\n  {'Symbol':>8} | {'Date':>12} | {'EPS Est':>8}")
            print(f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 8}")
            for _, row in upcoming.iterrows():
                est = (
                    f"{row['eps_estimate']:.2f}"
                    if pd.notna(row["eps_estimate"])
                    else "N/A"
                )
                print(
                    f"  {row['symbol']:>8} | "
                    f"{row['earnings_date'].strftime('%Y-%m-%d'):>12} | "
                    f"{est:>8}"
                )
            print(f"\n  Total: {len(upcoming)} earnings in next {args.days_ahead} days")

        print("=" * 60 + "\n")
        return

    print("\n" + "=" * 60)
    print("  EARNINGS CALENDAR FETCHER")
    print("=" * 60)

    df = fetch_all_earnings(
        symbols=args.symbols,
        refresh=args.refresh,
    )

    if not df.empty:
        # Show summary stats
        hist = df.dropna(subset=["surprise_pct"])
        if not hist.empty:
            print(f"\n  Historical earnings with surprise data: {len(hist):,}")
            print(f"  Average surprise: {hist['surprise_pct'].mean():+.2f}%")
            print(f"  Beat rate: {(hist['surprise_pct'] > 0).mean():.1%}")
            print(f"  Median surprise: {hist['surprise_pct'].median():+.2f}%")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
