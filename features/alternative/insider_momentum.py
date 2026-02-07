"""
Insider Trading Momentum Generator.

Generates trading signals from SEC Form 4 insider transaction filings.
Corporate insiders (officers, directors, 10%+ shareholders) must disclose
trades within two business days. Academic research documents a significant
post-disclosure drift, especially for open-market purchases.

Data sources:
- SEC EDGAR Form 4 filings
- OpenInsider (openinsider.com)
- Finviz insider transactions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InsiderMomentumGenerator:
    """
    Generate momentum signals from insider (Form 4) trading activity.

    Follows the "smart money" -- insiders often possess superior
    information about their own companies.

    Args:
        lookback_days: Window of insider trades to consider.
        min_transaction_value: Minimum USD value to filter noise.
        buy_sentiment_threshold: Net-buy sentiment above which we
            generate a long signal.
        sell_sentiment_threshold: Net-sell sentiment below which we
            generate a short signal.
    """

    def __init__(
        self,
        lookback_days: int = 90,
        min_transaction_value: float = 10_000.0,
        buy_sentiment_threshold: float = 0.5,
        sell_sentiment_threshold: float = -0.5,
    ) -> None:
        self.lookback_days = lookback_days
        self.min_value = min_transaction_value
        self.buy_threshold = buy_sentiment_threshold
        self.sell_threshold = sell_sentiment_threshold

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def parse_filings(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise raw insider filing data.

        Expected columns:
            transaction_date, symbol, insider_name, insider_title,
            transaction_type, shares, price, transaction_value

        Returns:
            Cleaned DataFrame with is_buy / is_sell flags.
        """
        df = raw.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

        if "transaction_value" not in df.columns:
            if "shares" in df.columns and "price" in df.columns:
                df["transaction_value"] = pd.to_numeric(
                    df["shares"], errors="coerce"
                ).abs() * pd.to_numeric(df["price"], errors="coerce")
            else:
                df["transaction_value"] = 0.0
        else:
            df["transaction_value"] = pd.to_numeric(
                df["transaction_value"], errors="coerce"
            ).abs()

        ttype = df.get("transaction_type", pd.Series(dtype=str)).astype(str)
        df["is_buy"] = ttype.str.contains(
            "Purchase|Buy|P-Purchase", case=False, na=False
        )
        df["is_sell"] = ttype.str.contains("Sale|Sell|S-Sale", case=False, na=False)

        return df

    def filter_recent(
        self,
        filings: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Filter to lookback window and minimum value."""
        if reference_date is None:
            reference_date = datetime.now()

        cutoff = reference_date - timedelta(days=self.lookback_days)
        mask = (filings["transaction_date"] >= cutoff) & (
            filings["transaction_value"] >= self.min_value
        )
        return filings.loc[mask].copy()

    # ------------------------------------------------------------------
    # Sentiment computation
    # ------------------------------------------------------------------

    def calculate_insider_sentiment(
        self,
        filings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute insider sentiment per symbol.

        Sentiment score:
            score = (n_buys - n_sells) / (n_buys + n_sells + 1)

        Also reports aggregate buy/sell value and a value-weighted
        sentiment:
            value_sentiment = (buy_value - sell_value) / (buy_value + sell_value)

        Returns:
            DataFrame indexed by symbol.
        """
        if filings.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "n_buys",
                    "n_sells",
                    "buy_value",
                    "sell_value",
                    "net_buys",
                    "sentiment_score",
                    "value_sentiment",
                ]
            )

        results: List[Dict] = []
        for symbol, grp in filings.groupby("symbol"):
            n_buys = int(grp["is_buy"].sum())
            n_sells = int(grp["is_sell"].sum())
            buy_val = float(grp.loc[grp["is_buy"], "transaction_value"].sum())
            sell_val = float(grp.loc[grp["is_sell"], "transaction_value"].sum())
            net = n_buys - n_sells
            score = net / (n_buys + n_sells + 1)
            val_total = buy_val + sell_val
            val_sent = (buy_val - sell_val) / val_total if val_total > 0 else 0.0

            results.append(
                {
                    "symbol": symbol,
                    "n_buys": n_buys,
                    "n_sells": n_sells,
                    "buy_value": buy_val,
                    "sell_value": sell_val,
                    "net_buys": net,
                    "sentiment_score": score,
                    "value_sentiment": val_sent,
                }
            )

        return pd.DataFrame(results)

    def calculate_cluster_score(
        self,
        filings: pd.DataFrame,
        cluster_window_days: int = 14,
    ) -> pd.DataFrame:
        """
        Detect insider buying *clusters* -- multiple insiders buying
        within a short window is a stronger signal than a single buy.

        Args:
            filings: Parsed insider filings.
            cluster_window_days: Days within which transactions count
                as clustered.

        Returns:
            DataFrame with [symbol, cluster_count, cluster_value,
            cluster_score].
        """
        buys = filings[filings["is_buy"]].copy()
        if buys.empty:
            return pd.DataFrame(
                columns=["symbol", "cluster_count", "cluster_value", "cluster_score"]
            )

        results: List[Dict] = []
        for symbol, grp in buys.groupby("symbol"):
            grp = grp.sort_values("transaction_date")
            # Unique insiders buying within the window
            latest = grp["transaction_date"].max()
            window_start = latest - timedelta(days=cluster_window_days)
            cluster = grp[grp["transaction_date"] >= window_start]

            n_unique = (
                cluster["insider_name"].nunique()
                if "insider_name" in cluster.columns
                else len(cluster)
            )
            total_val = float(cluster["transaction_value"].sum())
            # Score: log-scaled unique insiders * log-scaled value
            cluster_score = np.log1p(n_unique) * np.log1p(total_val / 100_000)

            results.append(
                {
                    "symbol": symbol,
                    "cluster_count": n_unique,
                    "cluster_value": total_val,
                    "cluster_score": cluster_score,
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        filings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate trading signals from insider activity.

        Signal logic:
            sentiment_score > buy_threshold  => +1 (long)
            sentiment_score < sell_threshold  => -1 (short)
            otherwise                         =>  0 (neutral)

        Returns:
            DataFrame with [symbol, signal, sentiment_score, value_sentiment,
            n_buys, n_sells].
        """
        parsed = self.parse_filings(filings)
        filtered = self.filter_recent(parsed)
        sentiment = self.calculate_insider_sentiment(filtered)

        if sentiment.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "signal",
                    "sentiment_score",
                    "value_sentiment",
                    "n_buys",
                    "n_sells",
                ]
            )

        sentiment["signal"] = 0
        sentiment.loc[sentiment["sentiment_score"] > self.buy_threshold, "signal"] = 1
        sentiment.loc[sentiment["sentiment_score"] < self.sell_threshold, "signal"] = -1

        return sentiment[
            [
                "symbol",
                "signal",
                "sentiment_score",
                "value_sentiment",
                "n_buys",
                "n_sells",
            ]
        ]
