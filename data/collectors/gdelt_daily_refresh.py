"""
Daily GDELT Tone Refresh
========================
Incrementally refreshes recent GDELT tone data and merges it into the
existing historical dataset.

Usage:
    python -m quantum_alpha.data.collectors.gdelt_daily_refresh
    python -m quantum_alpha.data.collectors.gdelt_daily_refresh --days-back 30
    python -m quantum_alpha.data.collectors.gdelt_daily_refresh --symbols AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from quantum_alpha.data.collectors.gdelt_historical import (
    OUTPUT_FILE,
    _fetch_timeline_tone,
    _get_query,
    compute_tone_features,
)

logger = logging.getLogger(__name__)


def _default_symbols(existing: pd.DataFrame | None = None) -> List[str]:
    if existing is not None and len(existing) > 0 and "symbol" in existing.columns:
        syms = sorted(existing["symbol"].dropna().unique().tolist())
        if syms:
            return syms
    try:
        from quantum_alpha.universe import get_sp500

        return sorted(get_sp500())
    except Exception:
        return [
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "JPM",
            "XOM",
        ]


def _fetch_symbol_window(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    query = _get_query(symbol)
    points = _fetch_timeline_tone(
        query=query,
        start_dt=start_dt.strftime("%Y%m%d%H%M%S"),
        end_dt=end_dt.strftime("%Y%m%d%H%M%S"),
    )
    if not points:
        return pd.DataFrame(columns=["date", "symbol", "tone"])

    rows = []
    for pt in points:
        try:
            dt = datetime.strptime(pt["date"], "%Y%m%dT%H%M%SZ")
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "tone": float(pt["value"]),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "tone"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset=["date", "symbol"], keep="last").sort_values(
        "date"
    )


def refresh_gdelt(
    symbols: Iterable[str] | None = None,
    days_back: int = 45,
) -> pd.DataFrame:
    existing = pd.DataFrame()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "rb") as f:
            existing = pickle.load(f)
        if len(existing) > 0:
            existing = existing.copy()
            existing["date"] = pd.to_datetime(existing["date"])
            logger.info(
                "Loaded existing GDELT dataset: %d rows, %d symbols",
                len(existing),
                existing["symbol"].nunique(),
            )

    sym_list = list(symbols) if symbols else _default_symbols(existing)
    if not sym_list:
        raise RuntimeError("No symbols available for refresh")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=max(7, days_back))

    logger.info(
        "Refreshing GDELT for %d symbols (%s -> %s)",
        len(sym_list),
        start_dt.date(),
        end_dt.date(),
    )

    new_frames = []
    for i, symbol in enumerate(sym_list, start=1):
        logger.info("[%d/%d] %s", i, len(sym_list), symbol)
        try:
            sym_df = _fetch_symbol_window(symbol, start_dt=start_dt, end_dt=end_dt)
        except Exception as e:
            logger.warning("  %s failed: %s", symbol, e)
            continue
        if len(sym_df) == 0:
            logger.info("  %s: no new records", symbol)
            continue
        logger.info("  %s: %d rows", symbol, len(sym_df))
        new_frames.append(sym_df)

    if new_frames:
        new_data = pd.concat(new_frames, ignore_index=True)
    else:
        new_data = pd.DataFrame(columns=["date", "symbol", "tone"])

    combined = (
        pd.concat([existing[["date", "symbol", "tone"]], new_data], ignore_index=True)
        if len(existing) > 0
        else new_data.copy()
    )

    if len(combined) == 0:
        logger.warning("No GDELT data available after refresh")
        return combined

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    combined = compute_tone_features(combined)

    output_dir = Path(OUTPUT_FILE).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(combined, f)

    logger.info(
        "Saved refreshed GDELT dataset: %d rows, %d symbols, through %s",
        len(combined),
        combined["symbol"].nunique(),
        combined["date"].max().date(),
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily incremental GDELT refresh")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to refresh (default: existing dataset symbols)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=45,
        help="How many recent calendar days to refresh (default: 45)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    refresh_gdelt(symbols=args.symbols, days_back=args.days_back)


if __name__ == "__main__":
    main()
