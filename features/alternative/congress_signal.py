"""
Congressional Trading Signal Generator.

Generates trading signals from congressional stock disclosure data.
Politicians often have access to privileged information through
committee assignments, briefings, and legislative influence.

Data sources:
- Senate Stock Watcher (senatestockwatcher.com)
- House Stock Watcher
- Capitol Trades (capitoltrades.com)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CongressSignalGenerator:
    """
    Generate trading signals from congressional stock disclosures.

    Tracks politician buy/sell activity, weights by transaction size
    and politician track record, and produces a sentiment score per symbol.

    Args:
        lookback_days: Days of historical trades to consider.
        min_delay_days: Minimum delay after disclosure before acting
            (accounts for disclosure lag and avoids front-running).
        min_amount: Minimum transaction midpoint value to consider.
        min_trades_for_track_record: Minimum trades to evaluate a
            politician's historical accuracy.
        track_record_threshold: Minimum average return for a politician
            to be considered a "good trader".
    """

    def __init__(
        self,
        lookback_days: int = 90,
        min_delay_days: int = 30,
        min_amount: float = 1_000.0,
        min_trades_for_track_record: int = 5,
        track_record_threshold: float = 0.05,
    ) -> None:
        self.lookback_days = lookback_days
        self.min_delay_days = min_delay_days
        self.min_amount = min_amount
        self.min_trades_for_record = min_trades_for_track_record
        self.track_record_threshold = track_record_threshold

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def parse_trades(self, raw_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalise raw congressional trade data.

        Expected columns:
            transaction_date, disclosure_date, ticker, name, type, amount

        Returns:
            DataFrame with parsed dates, midpoint amounts, and signal direction.
        """
        df = raw_trades.copy()

        for col in ("transaction_date", "disclosure_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Parse dollar-range amounts like "$1,001 - $15,000"
        if "amount" in df.columns and df["amount"].dtype == object:
            amount_clean = (
                df["amount"]
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            parts = amount_clean.str.split(r"\s*-\s*", expand=True)
            df["amount_min"] = pd.to_numeric(parts[0], errors="coerce")
            df["amount_max"] = pd.to_numeric(
                parts[1] if 1 in parts.columns else parts[0], errors="coerce"
            )
            df["amount_mid"] = (df["amount_min"] + df["amount_max"]) / 2.0
        elif "amount_mid" not in df.columns:
            df["amount_mid"] = 0.0

        # Signal direction
        if "type" in df.columns:
            df["signal"] = df["type"].apply(
                lambda x: 1 if "Purchase" in str(x) else (-1 if "Sale" in str(x) else 0)
            )
        else:
            df["signal"] = 0

        return df

    def filter_recent(
        self,
        trades: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Filter trades to the lookback window and minimum amount."""
        if reference_date is None:
            reference_date = datetime.now()

        cutoff = reference_date - timedelta(days=self.lookback_days)
        mask = (trades["transaction_date"] >= cutoff) & (
            trades["amount_mid"] >= self.min_amount
        )
        return trades.loc[mask].copy()

    def compute_politician_performance(
        self,
        trades: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        holding_period: int = 90,
    ) -> pd.DataFrame:
        """
        Evaluate historical return of each politician's trades.

        Args:
            trades: Parsed congressional trades.
            price_data: ``{symbol: DataFrame}`` with DatetimeIndex and 'close' column.
            holding_period: Calendar days to hold after entry.

        Returns:
            DataFrame with columns [symbol, politician, type, entry_date, return].
        """
        results: List[Dict] = []

        for _, trade in trades.iterrows():
            symbol = trade.get("ticker", "")
            if symbol not in price_data:
                continue

            prices = price_data[symbol]
            disclosure = pd.Timestamp(trade["disclosure_date"])
            entry_date = disclosure + timedelta(days=self.min_delay_days)

            # Find nearest trading day on or after entry_date
            valid_dates = prices.index[prices.index >= entry_date]
            if len(valid_dates) == 0:
                continue
            entry_dt = valid_dates[0]
            entry_price = float(prices.loc[entry_dt, "close"])

            exit_target = entry_dt + timedelta(days=holding_period)
            exit_candidates = prices.index[prices.index <= exit_target]
            if len(exit_candidates) == 0:
                continue
            exit_dt = exit_candidates[-1]
            exit_price = float(prices.loc[exit_dt, "close"])

            if trade["signal"] == 1:
                ret = (exit_price - entry_price) / entry_price
            elif trade["signal"] == -1:
                ret = (entry_price - exit_price) / entry_price
            else:
                continue

            results.append(
                {
                    "symbol": symbol,
                    "politician": trade.get("name", "unknown"),
                    "type": trade.get("type", ""),
                    "entry_date": entry_dt,
                    "return": ret,
                }
            )

        return (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                columns=["symbol", "politician", "type", "entry_date", "return"]
            )
        )

    def get_good_traders(self, performance: pd.DataFrame) -> pd.Index:
        """Identify politicians whose historical trades have positive alpha."""
        if performance.empty:
            return pd.Index([])

        track = performance.groupby("politician")["return"].agg(["mean", "count"])
        track = track[track["count"] >= self.min_trades_for_record]
        return track[track["mean"] > self.track_record_threshold].index

    def compute_symbol_sentiment(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-symbol sentiment from congressional trades.

        Sentiment = (buy_value - sell_value) / (buy_value + sell_value)
        Confidence = min(n_trades / 5, 1.0)

        Returns:
            DataFrame with [symbol, sentiment, confidence, n_trades, total_buy_value,
            total_sell_value].
        """
        if trades.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "sentiment",
                    "confidence",
                    "n_trades",
                    "total_buy_value",
                    "total_sell_value",
                ]
            )

        results: List[Dict] = []
        for symbol, grp in trades.groupby("ticker"):
            buys = grp.loc[grp["signal"] == 1, "amount_mid"].sum()
            sells = grp.loc[grp["signal"] == -1, "amount_mid"].sum()
            total = buys + sells
            sentiment = (buys - sells) / total if total > 0 else 0.0
            confidence = min(len(grp) / 5.0, 1.0)
            results.append(
                {
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "n_trades": len(grp),
                    "total_buy_value": buys,
                    "total_sell_value": sells,
                }
            )

        return pd.DataFrame(results)

    def generate_signals(
        self,
        trades: pd.DataFrame,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        performance: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from congressional disclosures.

        If *performance* is supplied, only follow politicians with a
        proven track record. Otherwise, use all trades.

        Args:
            trades: Recent parsed congressional trades.
            price_data: Price data keyed by symbol (for track-record eval).
            performance: Pre-computed politician performance DataFrame.

        Returns:
            DataFrame with [symbol, signal, n_buys, n_sells, sentiment, confidence].
        """
        df = self.parse_trades(trades)
        df = self.filter_recent(df)

        # Optionally filter to good traders
        if performance is not None:
            good = self.get_good_traders(performance)
            if len(good) > 0:
                df = df[df["name"].isin(good)]

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "signal",
                    "n_buys",
                    "n_sells",
                    "sentiment",
                    "confidence",
                ]
            )

        results: List[Dict] = []
        for symbol, grp in df.groupby("ticker"):
            buys = int((grp["signal"] == 1).sum())
            sells = int((grp["signal"] == -1).sum())
            total = buys + sells
            sentiment = (buys - sells) / total if total > 0 else 0.0
            signal = 1 if buys > sells else (-1 if sells > buys else 0)
            confidence = min(total / 5.0, 1.0)

            results.append(
                {
                    "symbol": symbol,
                    "signal": signal,
                    "n_buys": buys,
                    "n_sells": sells,
                    "sentiment": sentiment,
                    "confidence": confidence,
                }
            )

        return pd.DataFrame(results)
