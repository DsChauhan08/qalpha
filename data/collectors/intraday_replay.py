"""
Replay-first storage for trade, quote, depth, and intraday bar data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

DOMAINS = ("trades", "quotes", "depth", "bars_1s", "bars_5s", "bars_1m")


def _to_utc_timestamp(value: str | pd.Timestamp | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported replay file format: {path}")


def _write_table(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(path, index=False)
    return path


@dataclass
class ReplayQuality:
    symbols: int
    dates: int
    total_files: int
    completeness: float
    depth_completeness: float
    negative_spread_rate: float
    crossed_market_rate: float
    median_quote_staleness_ms: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "symbols": int(self.symbols),
            "dates": int(self.dates),
            "total_files": int(self.total_files),
            "completeness": float(self.completeness),
            "depth_completeness": float(self.depth_completeness),
            "negative_spread_rate": float(self.negative_spread_rate),
            "crossed_market_rate": float(self.crossed_market_rate),
            "median_quote_staleness_ms": float(self.median_quote_staleness_ms),
        }


class IntradayReplayStore:
    """Partitioned replay store rooted at ``date/symbol/domain``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def domain_path(
        self,
        date: str | pd.Timestamp | datetime,
        symbol: str,
        domain: str,
        prefer_parquet: bool = True,
    ) -> Path:
        if domain not in DOMAINS:
            raise ValueError(f"Unsupported replay domain: {domain}")
        day = _to_utc_timestamp(date).strftime("%Y-%m-%d")
        base = self.root / day / symbol.upper()
        return base / f"{domain}{'.parquet' if prefer_parquet else '.csv'}"

    def save_domain(
        self,
        date: str | pd.Timestamp | datetime,
        symbol: str,
        domain: str,
        df: pd.DataFrame,
        prefer_parquet: bool = True,
    ) -> Path:
        out = self.domain_path(date, symbol, domain, prefer_parquet=prefer_parquet)
        clean = df.copy()
        if "timestamp" in clean.columns:
            clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
        return _write_table(out, clean)

    def load_domain(
        self,
        date: str | pd.Timestamp | datetime,
        symbol: str,
        domain: str,
    ) -> pd.DataFrame:
        parquet = self.domain_path(date, symbol, domain, prefer_parquet=True)
        csv = self.domain_path(date, symbol, domain, prefer_parquet=False)
        path = parquet if parquet.exists() else csv
        if not path.exists():
            raise FileNotFoundError(f"Missing replay domain {domain} for {symbol} on {date}")
        df = _read_table(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def available_dates(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def available_symbols(self, date: str | pd.Timestamp | datetime) -> List[str]:
        day = _to_utc_timestamp(date).strftime("%Y-%m-%d")
        base = self.root / day
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def load_symbol_bundle(
        self,
        date: str | pd.Timestamp | datetime,
        symbol: str,
        domains: Iterable[str] = DOMAINS,
    ) -> Dict[str, pd.DataFrame]:
        return {domain: self.load_domain(date, symbol, domain) for domain in domains}

    def summarize_quality(
        self,
        dates: Optional[Iterable[str | pd.Timestamp | datetime]] = None,
        symbols: Optional[Iterable[str]] = None,
    ) -> ReplayQuality:
        selected_dates = [
            _to_utc_timestamp(d).strftime("%Y-%m-%d") for d in (dates or self.available_dates())
        ]
        selected_symbols = {str(s).upper() for s in symbols} if symbols else None

        total_expected = 0
        total_present = 0
        depth_expected = 0
        depth_present = 0
        negative = 0
        crossed = 0
        spread_rows = 0
        staleness_ms: List[float] = []

        for day in selected_dates:
            day_symbols = self.available_symbols(day)
            if selected_symbols is not None:
                day_symbols = [s for s in day_symbols if s in selected_symbols]
            for symbol in day_symbols:
                for domain in DOMAINS:
                    total_expected += 1
                    try:
                        df = self.load_domain(day, symbol, domain)
                    except FileNotFoundError:
                        continue
                    total_present += 1
                    if domain == "depth":
                        depth_expected += 1
                        if not df.empty:
                            depth_present += 1
                    if domain == "quotes" and {"bid", "ask"}.issubset(df.columns):
                        spreads = pd.to_numeric(df["ask"], errors="coerce") - pd.to_numeric(
                            df["bid"], errors="coerce"
                        )
                        spread_rows += int(spreads.notna().sum())
                        negative += int((spreads < 0).sum())
                        crossed += int((spreads <= 0).sum())
                    if domain == "trades" and "timestamp" in df.columns:
                        quotes_path = None
                        try:
                            quotes_path = self.load_domain(day, symbol, "quotes")
                        except FileNotFoundError:
                            quotes_path = None
                        if quotes_path is not None and not quotes_path.empty:
                            tr = pd.to_datetime(df["timestamp"], utc=True).sort_values()
                            qt = pd.to_datetime(quotes_path["timestamp"], utc=True).sort_values()
                            if not tr.empty and not qt.empty:
                                aligned = pd.merge_asof(
                                    tr.to_frame(name="trade_ts"),
                                    qt.to_frame(name="quote_ts"),
                                    left_on="trade_ts",
                                    right_on="quote_ts",
                                    direction="backward",
                                )
                                delta = (
                                    aligned["trade_ts"] - aligned["quote_ts"]
                                ).dt.total_seconds() * 1000.0
                                staleness_ms.extend(delta.dropna().tolist())

        completeness = total_present / total_expected if total_expected else 0.0
        depth_completeness = depth_present / depth_expected if depth_expected else 0.0
        negative_rate = negative / spread_rows if spread_rows else 0.0
        crossed_rate = crossed / spread_rows if spread_rows else 0.0
        median_staleness = float(np.median(staleness_ms)) if staleness_ms else 0.0
        return ReplayQuality(
            symbols=len(selected_symbols) if selected_symbols else 0,
            dates=len(selected_dates),
            total_files=total_present,
            completeness=completeness,
            depth_completeness=depth_completeness,
            negative_spread_rate=negative_rate,
            crossed_market_rate=crossed_rate,
            median_quote_staleness_ms=median_staleness,
        )


def build_synthetic_intraday_replay(
    root: str | Path,
    *,
    date: str = "2025-01-03",
    symbols: Optional[Iterable[str]] = None,
    minutes: int = 240,
    seed: int = 42,
) -> Path:
    """Create a deterministic replay dataset for tests and smoke runs."""

    store = IntradayReplayStore(root)
    rng = np.random.default_rng(seed)
    chosen = [s.upper() for s in (symbols or ("SPY", "XLK", "AAPL", "MSFT"))]
    start = pd.Timestamp(f"{date} 14:30:00+00:00")
    minute_index = pd.date_range(start, periods=minutes, freq="1min")
    second_index = pd.date_range(start, periods=minutes * 60, freq="1s")
    five_second_index = pd.date_range(start, periods=minutes * 12, freq="5s")
    base_paths: Dict[str, np.ndarray] = {}

    market_noise = rng.normal(0.0, 0.00035, size=minutes)
    market_path = 100.0 * np.exp(np.cumsum(market_noise))
    base_paths["SPY"] = market_path
    base_paths["XLK"] = 80.0 * np.exp(np.cumsum(market_noise * 1.15 + rng.normal(0, 0.0002, size=minutes)))
    base_paths["AAPL"] = 150.0 + (base_paths["XLK"] - base_paths["XLK"].mean()) * 1.8 + rng.normal(0, 0.35, size=minutes)
    base_paths["MSFT"] = 240.0 + (base_paths["XLK"] - base_paths["XLK"].mean()) * 2.1 + rng.normal(0, 0.30, size=minutes)

    for symbol in chosen:
        close = pd.Series(base_paths.get(symbol, 50.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, size=minutes)))), index=minute_index)
        open_ = close.shift(1).fillna(close.iloc[0] * (1 - 0.0005))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0008, 0.0002, size=minutes)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0008, 0.0002, size=minutes)))
        volume = rng.integers(40000, 120000, size=minutes)
        bars_1m = pd.DataFrame(
            {
                "timestamp": minute_index,
                "open": open_.values,
                "high": high,
                "low": low,
                "close": close.values,
                "volume": volume,
            }
        )
        store.save_domain(date, symbol, "bars_1m", bars_1m, prefer_parquet=False)

        second_close = close.reindex(second_index, method="ffill").values * (
            1.0 + rng.normal(0, 0.00008, size=len(second_index))
        )
        second_volume = rng.integers(100, 1800, size=len(second_index))
        bars_1s = pd.DataFrame(
            {
                "timestamp": second_index,
                "open": second_close,
                "high": second_close * (1.0 + np.abs(rng.normal(0.0001, 0.00004, size=len(second_index)))),
                "low": second_close * (1.0 - np.abs(rng.normal(0.0001, 0.00004, size=len(second_index)))),
                "close": second_close,
                "volume": second_volume,
            }
        )
        store.save_domain(date, symbol, "bars_1s", bars_1s, prefer_parquet=False)

        bars_5s = (
            bars_1s.set_index("timestamp")
            .resample("5s")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        store.save_domain(date, symbol, "bars_5s", bars_5s, prefer_parquet=False)

        quote_mid = close.reindex(five_second_index, method="ffill").values * (
            1.0 + rng.normal(0, 0.00005, size=len(five_second_index))
        )
        rel_spread = np.clip(0.0004 + np.abs(rng.normal(0.00015, 0.00005, size=len(five_second_index))), 0.0002, 0.0020)
        bid = quote_mid * (1.0 - rel_spread / 2.0)
        ask = quote_mid * (1.0 + rel_spread / 2.0)
        bid_size = rng.integers(200, 1600, size=len(five_second_index))
        ask_size = rng.integers(200, 1600, size=len(five_second_index))
        quotes = pd.DataFrame(
            {
                "timestamp": five_second_index,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        )
        store.save_domain(date, symbol, "quotes", quotes, prefer_parquet=False)

        side = rng.choice([-1, 1], size=len(second_index))
        trade_mid = close.reindex(second_index, method="ffill").values
        signed_spread = rel_spread.repeat(12)[: len(second_index)] / 2.0
        trade_price = trade_mid * (1.0 + side * signed_spread + rng.normal(0, 0.00003, size=len(second_index)))
        trades = pd.DataFrame(
            {
                "timestamp": second_index,
                "price": trade_price,
                "volume": second_volume,
                "side": side,
            }
        )
        store.save_domain(date, symbol, "trades", trades, prefer_parquet=False)

        depth_rows: List[Dict[str, object]] = []
        for ts, mid in zip(five_second_index, quote_mid):
            base_step = max(mid * 0.0002, 0.01)
            for level in range(1, 11):
                depth_rows.append(
                    {
                        "timestamp": ts,
                        "level": level,
                        "bid_price": mid - (base_step * level),
                        "bid_size": int(rng.integers(100, 1000) * (12 - level)),
                        "ask_price": mid + (base_step * level),
                        "ask_size": int(rng.integers(100, 1000) * (12 - level)),
                    }
                )
        depth = pd.DataFrame(depth_rows)
        store.save_domain(date, symbol, "depth", depth, prefer_parquet=False)

    return Path(root)


__all__ = [
    "DOMAINS",
    "ReplayQuality",
    "IntradayReplayStore",
    "build_synthetic_intraday_replay",
]
