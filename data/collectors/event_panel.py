"""Unified daily event-panel builder for event-driven research sleeves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.data.collectors.alternative.minimal_loaders import (
    load_congress_trades,
    load_insider_trades,
    load_options_sentiment,
)
from quantum_alpha.data.collectors.earnings_calendar import load_earnings_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data_store"
OHLCV_CACHE_DIR = DATA_ROOT / "meta_ensemble" / "ohlcv_cache"
GDELT_FILE = DATA_ROOT / "gdelt_tone" / "gdelt_daily_tone.pkl"


@dataclass
class EventPanelBundle:
    panel: pd.DataFrame
    quality: Dict[str, object]


def _normalize_date(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce")
    return pd.DatetimeIndex(pd.Series(idx).dt.tz_localize(None)).normalize()


def _load_symbol_ohlcv(symbol: str) -> pd.DataFrame:
    path = OHLCV_CACHE_DIR / f"{symbol.upper()}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_pickle(path).copy()
    df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    if "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = _normalize_date(df["date"])
    df.columns = [str(c).lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns for {symbol}: {sorted(missing)}")
    df["symbol"] = symbol.upper()
    df["returns"] = pd.to_numeric(df.get("returns"), errors="coerce")
    if "returns" not in df.columns or df["returns"].isna().all():
        df["returns"] = pd.to_numeric(df["close"], errors="coerce").pct_change(fill_method=None)
    df["dollar_volume"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
    return df.loc[:, ["date", "symbol", "open", "high", "low", "close", "volume", "returns", "dollar_volume"]]


def _load_gdelt_cache() -> pd.DataFrame:
    if not GDELT_FILE.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "tone",
                "tone_ma5",
                "tone_ma20",
                "tone_zscore",
                "tone_momentum",
                "tone_reversal",
                "news_tone_regime",
                "tone_volatility",
                "tone_acceleration",
            ]
        )
    gdelt = pd.read_pickle(GDELT_FILE).copy()
    gdelt["date"] = _normalize_date(gdelt["date"])
    gdelt["symbol"] = gdelt["symbol"].astype(str).str.upper()
    return gdelt


def _top_liquid_symbols(universe_size: int = 800) -> List[str]:
    if not OHLCV_CACHE_DIR.exists():
        return ["SPY"]

    earnings = load_earnings_calendar()
    earnings_symbols = set(earnings["symbol"].astype(str).str.upper()) if not earnings.empty else set()
    gdelt_symbols = set(_load_gdelt_cache()["symbol"].astype(str).str.upper())

    rows: List[Dict[str, object]] = []
    for path in sorted(OHLCV_CACHE_DIR.glob("*.pkl")):
        symbol = path.stem.upper()
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue
        if len(df) < 63 or "close" not in df.columns or "volume" not in df.columns:
            continue
        tail = df.tail(63)
        adv = float((pd.to_numeric(tail["close"], errors="coerce") * pd.to_numeric(tail["volume"], errors="coerce")).mean())
        coverage_score = int(symbol in earnings_symbols) + int(symbol in gdelt_symbols)
        rows.append({"symbol": symbol, "adv63": adv, "coverage_score": coverage_score})

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ["SPY"]
    ranked = ranked.sort_values(["coverage_score", "adv63"], ascending=[False, False])
    symbols = ranked["symbol"].head(max(1, int(universe_size))).tolist()
    if "SPY" not in symbols:
        symbols.insert(0, "SPY")
    return symbols


def _prepare_earnings_features(
    symbol: str,
    calendar_dates: pd.DatetimeIndex,
    earnings: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    cols = [
        "ev_last_earnings_surprise_pct",
        "ev_pead_days_since",
        "ev_pead_signal_raw",
        "ev_next_earnings_days_ahead",
        "ev_earnings_within_5d_raw",
        "ev_revision_delta_raw",
        "ev_guidance_delta_raw",
        "ev_earnings_event_flag",
    ]
    base = pd.DataFrame(index=calendar_dates, columns=cols, dtype=float)
    if earnings.empty:
        base = base.fillna(0.0)
        return base.reset_index(names="date"), {"earnings": 0.0}

    sym = earnings.loc[earnings["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if sym.empty:
        base = base.fillna(0.0)
        return base.reset_index(names="date"), {"earnings": 0.0}

    sym["earnings_date"] = _normalize_date(sym["earnings_date"])
    sym = sym.sort_values("earnings_date")
    sym["ev_revision_delta_raw"] = pd.to_numeric(sym["eps_estimate"], errors="coerce").pct_change(fill_method=None)
    sym["ev_guidance_delta_raw"] = np.nan
    sym["ev_earnings_event_flag"] = 1.0

    event_rows = sym.loc[
        :,
        [
            "earnings_date",
            "surprise_pct",
            "ev_revision_delta_raw",
            "ev_guidance_delta_raw",
            "ev_earnings_event_flag",
        ],
    ].rename(columns={"earnings_date": "date", "surprise_pct": "ev_last_earnings_surprise_pct"})
    event_rows = event_rows.sort_values("date")

    out = pd.DataFrame({"date": calendar_dates}).sort_values("date")
    out = pd.merge_asof(
        out,
        event_rows[["date", "ev_last_earnings_surprise_pct", "ev_revision_delta_raw", "ev_guidance_delta_raw"]],
        on="date",
        direction="backward",
    )
    out["ev_pead_days_since"] = (
        out["date"]
        - pd.merge_asof(
            pd.DataFrame({"date": calendar_dates}).sort_values("date"),
            event_rows[["date"]].rename(columns={"date": "last_earnings_date"}),
            left_on="date",
            right_on="last_earnings_date",
            direction="backward",
        )["last_earnings_date"]
    ).dt.days.astype(float)
    out["ev_pead_signal_raw"] = 0.0
    pead_mask = (
        out["ev_pead_days_since"].between(1, 60, inclusive="both")
        & out["ev_last_earnings_surprise_pct"].abs().ge(5.0)
    )
    out.loc[pead_mask, "ev_pead_signal_raw"] = np.sign(out.loc[pead_mask, "ev_last_earnings_surprise_pct"])

    next_dates = event_rows["date"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(next_dates, calendar_dates.to_numpy(dtype="datetime64[ns]"), side="left")
    next_vals = np.full(len(calendar_dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = idx < len(next_dates)
    next_vals[valid] = next_dates[idx[valid]]
    next_ts = pd.to_datetime(next_vals)
    out["ev_next_earnings_days_ahead"] = (next_ts - out["date"]).dt.days.astype(float)
    out["ev_earnings_within_5d_raw"] = (
        out["ev_next_earnings_days_ahead"].between(0, 5, inclusive="both")
    ).astype(float)

    event_flags = event_rows.set_index("date")["ev_earnings_event_flag"].reindex(calendar_dates).fillna(0.0)
    out["ev_earnings_event_flag"] = event_flags.values
    out["ev_revision_delta_raw"] = pd.to_numeric(out["ev_revision_delta_raw"], errors="coerce").fillna(0.0)
    out["ev_guidance_delta_raw"] = pd.to_numeric(out["ev_guidance_delta_raw"], errors="coerce").fillna(0.0)
    out["ev_last_earnings_surprise_pct"] = pd.to_numeric(out["ev_last_earnings_surprise_pct"], errors="coerce").fillna(0.0)
    out["ev_pead_days_since"] = pd.to_numeric(out["ev_pead_days_since"], errors="coerce").fillna(999.0)
    coverage = float(event_flags.mean() > 0 or out["ev_last_earnings_surprise_pct"].abs().sum() > 0)
    return out, {"earnings": coverage}


def _prepare_options_features(symbol: str, calendar_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, Dict[str, float]]:
    cols = [
        "options_put_volume_raw",
        "options_call_volume_raw",
        "options_total_volume_raw",
        "options_pc_ratio_raw",
        "options_iv_skew_raw",
        "options_volume_zscore_raw",
        "options_signal_raw",
    ]
    raw = load_options_sentiment(symbol, use_live=False)
    if raw.empty:
        out = pd.DataFrame({"date": calendar_dates})
        for col in cols:
            out[col] = 0.0
        return out, {"options": 0.0}

    df = raw.copy()
    df["date"] = _normalize_date(df.get("timestamp", df.get("date")))
    grouped = (
        df.groupby("date", as_index=False)
        .agg(
            {
                "put_volume": "sum",
                "call_volume": "sum",
                "total_volume": "sum",
                "otm_put_iv": "mean",
                "atm_call_iv": "mean",
            }
        )
        .sort_values("date")
    )
    grouped["options_pc_ratio_raw"] = grouped["put_volume"] / grouped["call_volume"].replace(0, np.nan)
    grouped["options_iv_skew_raw"] = grouped["otm_put_iv"] - grouped["atm_call_iv"]
    vol_ma = grouped["total_volume"].rolling(20, min_periods=1).mean()
    vol_std = grouped["total_volume"].rolling(20, min_periods=1).std()
    grouped["options_volume_zscore_raw"] = np.where(
        vol_std > 0,
        (grouped["total_volume"] - vol_ma) / vol_std,
        0.0,
    )
    grouped["options_signal_raw"] = np.sign(
        grouped["options_volume_zscore_raw"].fillna(0.0)
        + (grouped["options_iv_skew_raw"].fillna(0.0) > grouped["options_iv_skew_raw"].fillna(0.0).rolling(20, min_periods=1).mean()).astype(float)
        - (grouped["options_pc_ratio_raw"].fillna(1.0) > grouped["options_pc_ratio_raw"].fillna(1.0).rolling(20, min_periods=1).mean() * 1.3).astype(float)
    )
    grouped = grouped.rename(
        columns={
            "put_volume": "options_put_volume_raw",
            "call_volume": "options_call_volume_raw",
            "total_volume": "options_total_volume_raw",
        }
    )
    out = pd.DataFrame({"date": calendar_dates}).merge(grouped.loc[:, ["date", *cols]], on="date", how="left")
    out.loc[:, cols] = out.loc[:, cols].ffill(limit=5).fillna(0.0)
    coverage = float((out["options_total_volume_raw"] > 0).mean())
    return out, {"options": coverage}


def _prepare_insider_features(symbol: str, calendar_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, Dict[str, float]]:
    cols = [
        "insider_n_buys_raw",
        "insider_n_sells_raw",
        "insider_value_sentiment_raw",
        "insider_cluster_score_raw",
    ]
    raw = load_insider_trades(symbol, use_live=False)
    if raw.empty:
        out = pd.DataFrame({"date": calendar_dates})
        for col in cols:
            out[col] = 0.0
        return out, {"insider": 0.0}

    df = raw.copy()
    df["transaction_date"] = _normalize_date(df["transaction_date"])
    ttype = df.get("transaction_type", pd.Series("", index=df.index)).astype(str)
    df["is_buy"] = ttype.str.contains("purchase|buy|p-purchase", case=False, na=False)
    df["is_sell"] = ttype.str.contains("sale|sell|s-sale", case=False, na=False)
    if "transaction_value" not in df.columns:
        df["transaction_value"] = pd.to_numeric(df.get("shares", 0), errors="coerce").abs() * pd.to_numeric(df.get("price", 0), errors="coerce")
    df["buy_value"] = np.where(df["is_buy"], pd.to_numeric(df["transaction_value"], errors="coerce").fillna(0.0), 0.0)
    df["sell_value"] = np.where(df["is_sell"], pd.to_numeric(df["transaction_value"], errors="coerce").fillna(0.0), 0.0)
    daily = (
        df.groupby("transaction_date", as_index=False)
        .agg(
            insider_n_buys_raw=("is_buy", "sum"),
            insider_n_sells_raw=("is_sell", "sum"),
            buy_value=("buy_value", "sum"),
            sell_value=("sell_value", "sum"),
        )
        .rename(columns={"transaction_date": "date"})
        .sort_values("date")
    )
    daily = daily.set_index("date").reindex(calendar_dates).fillna(0.0)
    buy_roll = daily["buy_value"].rolling(90, min_periods=1).sum()
    sell_roll = daily["sell_value"].rolling(90, min_periods=1).sum()
    count_roll = daily["insider_n_buys_raw"].rolling(14, min_periods=1).sum()
    daily["insider_value_sentiment_raw"] = (buy_roll - sell_roll) / (buy_roll + sell_roll + 1e-6)
    daily["insider_cluster_score_raw"] = np.log1p(count_roll) * np.log1p((buy_roll + 1.0) / 100000.0)
    out = daily.reset_index(names="date").loc[:, ["date", *cols]]
    coverage = float(((daily["insider_n_buys_raw"] + daily["insider_n_sells_raw"]) > 0).mean())
    return out, {"insider": coverage}


def _prepare_congress_features(symbol: str, calendar_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, Dict[str, float]]:
    cols = [
        "congress_sentiment_raw",
        "congress_confidence_raw",
        "congress_n_trades_raw",
        "congress_disclosure_lag_days_raw",
    ]
    raw = load_congress_trades(symbol, use_live=False)
    if raw.empty:
        out = pd.DataFrame({"date": calendar_dates})
        for col in cols:
            out[col] = 0.0
        return out, {"congress": 0.0, "median_congress_disclosure_lag_days": 0.0}

    df = raw.copy()
    df["transaction_date"] = _normalize_date(df.get("transaction_date"))
    has_disclosure = "disclosure_date" in df.columns
    if has_disclosure:
        df["disclosure_date"] = _normalize_date(df["disclosure_date"])
    else:
        df["disclosure_date"] = df["transaction_date"]
    df["actionable_date"] = df["disclosure_date"] + pd.Timedelta(days=30)
    if "amount_mid" not in df.columns:
        if "amount" in df.columns and df["amount"].dtype == object:
            amount_clean = df["amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
            parts = amount_clean.str.split(r"\s*-\s*", expand=True)
            df["amount_mid"] = (
                pd.to_numeric(parts[0], errors="coerce").fillna(0.0)
                + pd.to_numeric(parts[1] if 1 in parts.columns else parts[0], errors="coerce").fillna(0.0)
            ) / 2.0
        else:
            df["amount_mid"] = 0.0
    ttype = df.get("type", pd.Series("", index=df.index)).astype(str)
    df["signal"] = np.where(ttype.str.contains("purchase", case=False, na=False), 1.0, np.where(ttype.str.contains("sale", case=False, na=False), -1.0, 0.0))
    df["buy_value"] = np.where(df["signal"] > 0, pd.to_numeric(df["amount_mid"], errors="coerce").fillna(0.0), 0.0)
    df["sell_value"] = np.where(df["signal"] < 0, pd.to_numeric(df["amount_mid"], errors="coerce").fillna(0.0), 0.0)
    daily = (
        df.groupby("actionable_date", as_index=False)
        .agg(
            congress_n_trades_raw=("signal", lambda x: float((x != 0).sum())),
            buy_value=("buy_value", "sum"),
            sell_value=("sell_value", "sum"),
        )
        .rename(columns={"actionable_date": "date"})
        .sort_values("date")
    )
    daily = daily.set_index("date").reindex(calendar_dates).fillna(0.0)
    buy_roll = daily["buy_value"].rolling(90, min_periods=1).sum()
    sell_roll = daily["sell_value"].rolling(90, min_periods=1).sum()
    trade_roll = daily["congress_n_trades_raw"].rolling(90, min_periods=1).sum()
    daily["congress_sentiment_raw"] = (buy_roll - sell_roll) / (buy_roll + sell_roll + 1e-6)
    daily["congress_confidence_raw"] = np.clip(trade_roll / 5.0, 0.0, 1.0)
    disclosure_lag = (df["disclosure_date"] - df["transaction_date"]).dt.days
    if not has_disclosure:
        disclosure_lag = pd.Series(30.0, index=df.index)
    daily["congress_disclosure_lag_days_raw"] = float(pd.to_numeric(disclosure_lag, errors="coerce").median() or 0.0)
    out = daily.reset_index(names="date").loc[:, ["date", *cols]]
    coverage = float((daily["congress_n_trades_raw"] > 0).mean())
    return out, {
        "congress": coverage,
        "median_congress_disclosure_lag_days": float(pd.to_numeric(disclosure_lag, errors="coerce").median() or 0.0),
    }


def build_synthetic_event_panel(
    symbols: Sequence[str],
    *,
    days: int = 252,
    seed: int = 42,
) -> EventPanelBundle:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=max(40, int(days)), freq="B")
    symbols = [str(s).upper() for s in symbols]
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]

    market_returns = rng.normal(0.0004, 0.01, size=len(dates))
    event_points = np.arange(10, len(dates), 21)
    market_returns[event_points] += rng.normal(0.0, 0.015, size=len(event_points))

    frames: List[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols):
        beta = 1.0 if symbol == "SPY" else rng.uniform(0.7, 1.3)
        idio = rng.normal(0.00015, 0.012 + idx * 0.0005, size=len(dates))
        surprise_signal = np.zeros(len(dates))
        surprise_signal[event_points] = rng.choice([-1.0, 1.0], size=len(event_points)) * rng.uniform(0.03, 0.15, size=len(event_points))
        pead = pd.Series(surprise_signal, index=dates).replace(0.0, np.nan).ffill(limit=20).fillna(0.0).to_numpy()
        ret = beta * market_returns + idio + 0.15 * pead
        close = 100.0 * np.exp(np.cumsum(ret))
        high = close * (1.0 + rng.uniform(0.002, 0.03, size=len(dates)))
        low = close * (1.0 - rng.uniform(0.002, 0.03, size=len(dates)))
        open_ = close / (1.0 + rng.normal(0.0, 0.005, size=len(dates)))
        volume = rng.integers(1_500_000, 12_000_000, size=len(dates))
        tone = np.clip(pead * 8.0 + rng.normal(0.0, 0.7, size=len(dates)), -5, 5)
        pc_ratio = np.clip(1.0 - pead * 4.0 + rng.normal(0.0, 0.2, size=len(dates)), 0.2, 3.0)
        iv_skew = np.clip(0.04 + np.abs(pead) * 0.2 + rng.normal(0.0, 0.02, size=len(dates)), -0.1, 0.8)
        opt_vol = rng.lognormal(mean=11.0, sigma=0.25, size=len(dates))
        insider_signal = pd.Series(surprise_signal, index=dates).rolling(10, min_periods=1).mean().fillna(0.0).to_numpy()
        congress_signal = pd.Series(surprise_signal[::-1], index=dates).rolling(15, min_periods=1).mean().fillna(0.0).to_numpy()
        next_days = np.full(len(dates), 999.0)
        for ep in event_points:
            next_days = np.minimum(next_days, np.arange(len(dates)) - ep)
        last_event_idx = np.full(len(dates), -999.0)
        last_seen = -999.0
        for i in range(len(dates)):
            if i in set(event_points):
                last_seen = i
            last_event_idx[i] = i - last_seen
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "returns": pd.Series(close).pct_change(fill_method=None).fillna(0.0).to_numpy(),
                "dollar_volume": close * volume,
                "tone": tone,
                "tone_ma5": pd.Series(tone).rolling(5, min_periods=1).mean().to_numpy(),
                "tone_ma20": pd.Series(tone).rolling(20, min_periods=1).mean().to_numpy(),
                "tone_zscore": ((pd.Series(tone) - pd.Series(tone).rolling(20, min_periods=5).mean()) / pd.Series(tone).rolling(20, min_periods=5).std()).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(),
                "tone_momentum": pd.Series(tone).rolling(5, min_periods=1).mean().diff(5).fillna(0.0).to_numpy(),
                "tone_reversal": (pd.Series(tone) - pd.Series(tone).rolling(20, min_periods=1).mean()).fillna(0.0).to_numpy(),
                "news_tone_regime": np.sign(pd.Series(tone).rolling(20, min_periods=1).mean()).fillna(0.0).to_numpy(),
                "tone_volatility": pd.Series(tone).rolling(20, min_periods=1).std().fillna(0.0).to_numpy(),
                "tone_acceleration": pd.Series(tone).diff().diff().fillna(0.0).to_numpy(),
                "ev_last_earnings_surprise_pct": pead * 100.0,
                "ev_pead_days_since": np.where(last_event_idx >= 0, last_event_idx, 999.0),
                "ev_pead_signal_raw": np.sign(pead),
                "ev_next_earnings_days_ahead": np.where(next_days >= 0, next_days, 999.0),
                "ev_earnings_within_5d_raw": (np.where(next_days >= 0, next_days, 999.0) <= 5).astype(float),
                "ev_revision_delta_raw": pd.Series(surprise_signal).replace(0.0, np.nan).ffill().pct_change(fill_method=None).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(),
                "ev_guidance_delta_raw": (pd.Series(surprise_signal).rolling(2, min_periods=1).mean().fillna(0.0) * 0.5).to_numpy(),
                "ev_earnings_event_flag": np.isin(np.arange(len(dates)), event_points).astype(float),
                "options_put_volume_raw": opt_vol * pc_ratio,
                "options_call_volume_raw": opt_vol,
                "options_total_volume_raw": opt_vol * (1.0 + pc_ratio),
                "options_pc_ratio_raw": pc_ratio,
                "options_iv_skew_raw": iv_skew,
                "options_volume_zscore_raw": ((pd.Series(opt_vol) - pd.Series(opt_vol).rolling(20, min_periods=5).mean()) / pd.Series(opt_vol).rolling(20, min_periods=5).std()).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(),
                "options_signal_raw": np.sign(pead + rng.normal(0.0, 0.1, size=len(dates))),
                "insider_n_buys_raw": np.clip(np.round((insider_signal > 0).astype(float) + rng.uniform(0, 2, size=len(dates))), 0, None),
                "insider_n_sells_raw": np.clip(np.round((insider_signal < 0).astype(float) + rng.uniform(0, 2, size=len(dates))), 0, None),
                "insider_value_sentiment_raw": np.tanh(insider_signal * 4.0),
                "insider_cluster_score_raw": np.abs(insider_signal) * 3.0,
                "congress_sentiment_raw": np.tanh(congress_signal * 3.0),
                "congress_confidence_raw": np.clip(np.abs(congress_signal) * 2.0, 0.0, 1.0),
                "congress_n_trades_raw": np.clip(np.round(np.abs(congress_signal) * 4.0 + rng.uniform(0, 1.5, size=len(dates))), 0, None),
                "congress_disclosure_lag_days_raw": 32.0,
            }
        )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    quality = {
        "source_mode": "fixture",
        "symbols": int(panel["symbol"].nunique()),
        "rows": int(len(panel)),
        "coverage_by_domain": {
            "earnings": 1.0,
            "gdelt": 1.0,
            "options": 1.0,
            "insider": 1.0,
            "congress": 1.0,
        },
        "latest_dates": {
            "earnings": str(panel["date"].max().date()),
            "gdelt": str(panel["date"].max().date()),
            "options": str(panel["date"].max().date()),
            "insider": str(panel["date"].max().date()),
            "congress": str(panel["date"].max().date()),
        },
        "staleness_days": {
            "earnings": 0.0,
            "gdelt": 0.0,
            "options": 0.0,
            "insider": 0.0,
            "congress": 0.0,
        },
        "event_lag_ok": True,
        "paid_data_eligible": False,
        "median_congress_disclosure_lag_days": 32.0,
    }
    return EventPanelBundle(panel=panel, quality=quality)


def build_event_panel(
    *,
    symbols: Sequence[str] | None = None,
    universe_size: int = 800,
    start_date: str | None = None,
    end_date: str | None = None,
    use_fixture: bool = False,
    fixture_days: int = 252,
    seed: int = 42,
) -> EventPanelBundle:
    if use_fixture:
        requested = [str(s).upper() for s in (symbols or ["SPY", "AAPL", "MSFT", "NVDA", "XOM", "JPM"])]
        return build_synthetic_event_panel(requested, days=fixture_days, seed=seed)

    selected = [str(s).upper() for s in symbols] if symbols else _top_liquid_symbols(universe_size=universe_size)
    if "SPY" not in selected:
        selected = ["SPY", *selected]

    gdelt = _load_gdelt_cache()
    earnings = load_earnings_calendar()
    if not earnings.empty:
        earnings = earnings.copy()
        earnings["symbol"] = earnings["symbol"].astype(str).str.upper()
        earnings["earnings_date"] = _normalize_date(earnings["earnings_date"])

    start_ts = pd.Timestamp(start_date).normalize() if start_date else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date else None

    frames: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, float]] = []
    latest_dates: Dict[str, pd.Timestamp | None] = {
        "earnings": None,
        "gdelt": None,
        "options": None,
        "insider": None,
        "congress": None,
    }
    median_congress_lags: List[float] = []

    for symbol in selected:
        try:
            base = _load_symbol_ohlcv(symbol)
        except Exception:
            continue
        if start_ts is not None:
            base = base.loc[base["date"] >= start_ts]
        if end_ts is not None:
            base = base.loc[base["date"] <= end_ts]
        if base.empty:
            continue
        calendar_dates = pd.DatetimeIndex(base["date"])

        sym_gdelt = gdelt.loc[gdelt["symbol"] == symbol].copy()
        sym_gdelt = sym_gdelt.rename(
            columns={
                "tone": "tone",
                "tone_ma5": "tone_ma5",
                "tone_ma20": "tone_ma20",
                "tone_zscore": "tone_zscore",
                "tone_momentum": "tone_momentum",
                "tone_reversal": "tone_reversal",
                "news_tone_regime": "news_tone_regime",
                "tone_volatility": "tone_volatility",
                "tone_acceleration": "tone_acceleration",
            }
        )
        merged = base.merge(
            sym_gdelt.loc[:, ["date", "tone", "tone_ma5", "tone_ma20", "tone_zscore", "tone_momentum", "tone_reversal", "news_tone_regime", "tone_volatility", "tone_acceleration"]],
            on="date",
            how="left",
        )
        earnings_df, earnings_cov = _prepare_earnings_features(symbol, calendar_dates, earnings)
        options_df, options_cov = _prepare_options_features(symbol, calendar_dates)
        insider_df, insider_cov = _prepare_insider_features(symbol, calendar_dates)
        congress_df, congress_cov = _prepare_congress_features(symbol, calendar_dates)

        merged = merged.merge(earnings_df, on="date", how="left")
        merged = merged.merge(options_df, on="date", how="left")
        merged = merged.merge(insider_df, on="date", how="left")
        merged = merged.merge(congress_df, on="date", how="left")
        fill_cols = [c for c in merged.columns if c not in {"date", "symbol"}]
        merged.loc[:, fill_cols] = merged.loc[:, fill_cols].replace([np.inf, -np.inf], np.nan)
        merged.loc[:, fill_cols] = merged.loc[:, fill_cols].fillna(0.0)
        frames.append(merged)

        coverage = {"gdelt": float(merged["tone"].abs().sum() > 0)}
        coverage.update(earnings_cov)
        coverage.update(options_cov)
        coverage.update(insider_cov)
        coverage.update({k: v for k, v in congress_cov.items() if k == "congress"})
        coverage_rows.append(coverage)

        if not sym_gdelt.empty:
            latest_dates["gdelt"] = max(filter(None, [latest_dates["gdelt"], pd.Timestamp(sym_gdelt["date"].max())]))
        if not earnings.empty:
            sym_e = earnings.loc[earnings["symbol"] == symbol]
            if not sym_e.empty:
                latest_dates["earnings"] = max(filter(None, [latest_dates["earnings"], pd.Timestamp(sym_e["earnings_date"].max())]))
        if float(options_cov.get("options", 0.0)) > 0:
            latest_dates["options"] = calendar_dates.max()
        if float(insider_cov.get("insider", 0.0)) > 0:
            latest_dates["insider"] = calendar_dates.max()
        if float(congress_cov.get("congress", 0.0)) > 0:
            latest_dates["congress"] = calendar_dates.max()
            median_congress_lags.append(float(congress_cov.get("median_congress_disclosure_lag_days", 0.0)))

    if not frames:
        raise ValueError("No event panel rows could be built from local data")

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    cov_df = pd.DataFrame(coverage_rows) if coverage_rows else pd.DataFrame()
    coverage_by_domain = {col: float(cov_df[col].mean()) for col in cov_df.columns} if not cov_df.empty else {}

    end_ref = panel["date"].max()
    staleness = {}
    for key, ts in latest_dates.items():
        staleness[key] = max(0.0, float((end_ref - ts).days)) if ts is not None else 9999.0

    quality = {
        "source_mode": "free_cache",
        "symbols": int(panel["symbol"].nunique()),
        "rows": int(len(panel)),
        "coverage_by_domain": coverage_by_domain,
        "latest_dates": {k: (str(v.date()) if v is not None else None) for k, v in latest_dates.items()},
        "staleness_days": staleness,
        "event_lag_ok": True if not median_congress_lags else float(np.nanmedian(median_congress_lags)) >= 30.0,
        "paid_data_eligible": False,
        "median_congress_disclosure_lag_days": float(np.nanmedian(median_congress_lags) if median_congress_lags else 0.0),
    }
    return EventPanelBundle(panel=panel, quality=quality)


__all__ = [
    "EventPanelBundle",
    "build_event_panel",
    "build_synthetic_event_panel",
]
