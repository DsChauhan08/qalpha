from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CongressTradeSource:
    name: str
    url: str
    chamber: str


class CongressTradesCollector:
    DEFAULT_SENATE_URL = (
        "https://raw.githubusercontent.com/"
        "timothycarambat/senate-stock-watcher-data/"
        "master/aggregate/all_transactions.json"
    )
    DEFAULT_HOUSE_URL = (
        "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/"
        "data/all_transactions.json"
    )

    def __init__(
        self,
        senate_url: Optional[str] = None,
        house_url: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        env_senate = os.getenv("CONGRESS_SENATE_URL")
        env_house = os.getenv("CONGRESS_HOUSE_URL")
        self.senate_url = senate_url or env_senate or self.DEFAULT_SENATE_URL
        self.house_url = house_url or env_house or self.DEFAULT_HOUSE_URL
        self.timeout = timeout

    def fetch_trades(self, symbol: Optional[str] = None) -> pd.DataFrame:
        sources: List[CongressTradeSource] = []
        if self.senate_url:
            sources.append(
                CongressTradeSource("senate_stock_watcher", self.senate_url, "senate")
            )
        if self.house_url:
            sources.append(
                CongressTradeSource("house_stock_watcher", self.house_url, "house")
            )

        frames: List[pd.DataFrame] = []
        for source in sources:
            try:
                response = requests.get(
                    source.url,
                    timeout=self.timeout,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                response.raise_for_status()
                records = response.json()
                if isinstance(records, dict):
                    for key in ("data", "results", "transactions"):
                        if key in records:
                            records = records[key]
                            break
                    else:
                        records = []
                df = self._normalize(records, source.chamber)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.warning("Congress trades fetch failed for %s: %s", source.name, exc)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "transaction_date",
                    "disclosure_date",
                    "ticker",
                    "symbol",
                    "name",
                    "type",
                    "amount",
                    "chamber",
                ]
            )

        df = pd.concat(frames, ignore_index=True)
        if symbol:
            sym = symbol.upper()
            if "ticker" in df.columns:
                df = df[df["ticker"] == sym]
            elif "symbol" in df.columns:
                df = df[df["symbol"] == sym]
        return df

    @staticmethod
    def _normalize(records: list, chamber: str) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df.columns = [str(c).lower() for c in df.columns]

        rename_map = {}
        if "ticker" not in df.columns:
            if "symbol" in df.columns:
                rename_map["symbol"] = "ticker"
            elif "stock" in df.columns:
                rename_map["stock"] = "ticker"
        if "name" not in df.columns:
            if "representative" in df.columns:
                rename_map["representative"] = "name"
            elif "senator" in df.columns:
                rename_map["senator"] = "name"
        if "type" not in df.columns and "transaction_type" in df.columns:
            rename_map["transaction_type"] = "type"
        if "transaction_date" not in df.columns and "date" in df.columns:
            rename_map["date"] = "transaction_date"
        if "disclosure_date" not in df.columns:
            for candidate in ("disclosure_date", "report_date", "date_received", "date_recieved"):
                if candidate in df.columns:
                    rename_map[candidate] = "disclosure_date"
                    break

        if rename_map:
            df = df.rename(columns=rename_map)

        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df = df[df["ticker"].str.len() > 0]
            df = df[df["ticker"] != "NAN"]
            df["symbol"] = df["ticker"]
        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(
                df["transaction_date"], errors="coerce"
            )
        if "disclosure_date" in df.columns:
            df["disclosure_date"] = pd.to_datetime(
                df["disclosure_date"], errors="coerce"
            )
        elif "transaction_date" in df.columns:
            df["disclosure_date"] = df["transaction_date"]
        if "name" not in df.columns:
            df["name"] = ""
        df["chamber"] = chamber
        return df


__all__ = ["CongressTradesCollector"]
