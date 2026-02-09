"""
Minimal local-cache loaders for alternative data feeds.

These loaders prefer local cached CSV/Parquet files and return empty
frames with expected schemas when data is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging

import pandas as pd

logger = logging.getLogger(__name__)

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_frame(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_social_sentiment(
    symbol: str,
    cache_dir: str = ".cache/social_sentiment",
    use_live: bool = False,
) -> pd.DataFrame:
    base = Path(cache_dir)
    candidates = [
        base / f"{symbol}.parquet",
        base / f"{symbol}.csv",
        base / "social_sentiment.parquet",
        base / "social_sentiment.csv",
    ]
    df = _load_frame(_first_existing(candidates))
    if df.empty and use_live:
        try:
            from quantum_alpha.data.collectors.alternative.reddit_sentiment import (
                RedditCollector,
            )
            collector = RedditCollector()
            posts = collector.fetch_all_subreddits()
            agg = collector.aggregate_ticker_sentiment(posts)
            if not agg.empty:
                agg = agg.rename(
                    columns={"ticker": "symbol", "avg_sentiment": "sentiment_score", "mentions": "volume"}
                )
                agg["timestamp"] = datetime.utcnow()
                df = agg[["timestamp", "symbol", "sentiment_score", "volume"]]
        except Exception as exc:
            logger.warning("Live Reddit sentiment fetch failed: %s", exc)

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "volume"])
    df = df.copy()
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = df["date"]
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    return df


def load_options_sentiment(
    symbol: str,
    cache_dir: str = ".cache/options_sentiment",
    use_live: bool = False,
) -> pd.DataFrame:
    base = Path(cache_dir)
    candidates = [
        base / f"{symbol}.parquet",
        base / f"{symbol}.csv",
        base / "options_sentiment.parquet",
        base / "options_sentiment.csv",
    ]
    df = _load_frame(_first_existing(candidates))
    if df.empty and use_live:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if expirations:
                chain = ticker.option_chain(expirations[0])
                calls = chain.calls
                puts = chain.puts
                call_vol = float(calls["volume"].fillna(0).sum()) if "volume" in calls else 0.0
                put_vol = float(puts["volume"].fillna(0).sum()) if "volume" in puts else 0.0
                total_vol = call_vol + put_vol
                # Rough IV skew using closest ATM and 10% OTM put
                spot = float(ticker.history(period="5d")["Close"].iloc[-1])
                calls["abs_strike"] = (calls["strike"] - spot).abs()
                puts["abs_strike"] = (puts["strike"] - spot).abs()
                atm_call_iv = float(calls.sort_values("abs_strike").iloc[0]["impliedVolatility"])
                otm_put = puts[puts["strike"] < spot]
                otm_put_iv = float(otm_put.sort_values("strike", ascending=False).iloc[0]["impliedVolatility"]) if not otm_put.empty else atm_call_iv
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.utcnow(),
                            "symbol": symbol,
                            "put_volume": put_vol,
                            "call_volume": call_vol,
                            "total_volume": total_vol,
                            "otm_put_iv": otm_put_iv,
                            "atm_call_iv": atm_call_iv,
                        }
                    ]
                )
        except Exception as exc:
            logger.warning("Live options sentiment fetch failed: %s", exc)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "put_volume",
                "call_volume",
                "total_volume",
                "otm_put_iv",
                "atm_call_iv",
            ]
        )
    df = df.copy()
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = df["date"]
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    return df


def load_insider_trades(
    symbol: str,
    cache_dir: str = ".cache/insider",
    use_live: bool = False,
) -> pd.DataFrame:
    base = Path(cache_dir)
    candidates = [
        base / f"{symbol}.parquet",
        base / f"{symbol}.csv",
        base / "insider.parquet",
        base / "insider.csv",
    ]
    df = _load_frame(_first_existing(candidates))
    if df.empty and use_live:
        try:
            from quantum_alpha.data.collectors.alternative.sec_edgar import SECEdgarCollector

            collector = SECEdgarCollector()
            df = collector.get_insider_trades(symbol)
            if not df.empty:
                df = df.rename(
                    columns={
                        "date": "transaction_date",
                        "transaction_type": "transaction_type",
                        "value": "transaction_value",
                    }
                )
                df["symbol"] = symbol
        except Exception as exc:
            logger.warning("Live insider trades fetch failed: %s", exc)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "transaction_date",
                "symbol",
                "transaction_type",
                "shares",
                "price",
                "transaction_value",
            ]
        )
    df = df.copy()
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    return df


def load_congress_trades(
    symbol: str,
    cache_dir: str = ".cache/congress",
    use_live: bool = False,
) -> pd.DataFrame:
    base = Path(cache_dir)
    candidates = [
        base / f"{symbol}.parquet",
        base / f"{symbol}.csv",
        base / "congress.parquet",
        base / "congress.csv",
    ]
    df = _load_frame(_first_existing(candidates))
    if df.empty and use_live:
        try:
            from quantum_alpha.data.collectors.congress_trades import CongressTradesCollector

            collector = CongressTradesCollector()
            df = collector.fetch_trades()
            if not df.empty:
                base.mkdir(parents=True, exist_ok=True)
                try:
                    df.to_parquet(base / "congress.parquet", index=False)
                except Exception:
                    df.to_csv(base / "congress.csv", index=False)
        except Exception as exc:
            logger.warning("Live congress trades fetch failed: %s", exc)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "transaction_date",
                "symbol",
                "ticker",
                "type",
                "amount",
            ]
        )
    df = df.copy()
    if "symbol" not in df.columns and "ticker" in df.columns:
        df["symbol"] = df["ticker"]
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "ticker" not in df.columns:
        df["ticker"] = df["symbol"]
    if symbol:
        df = df[df["symbol"] == symbol]
    return df
