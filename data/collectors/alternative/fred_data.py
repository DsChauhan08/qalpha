"""
FRED (Federal Reserve Economic Data) Collector

Free economic data from the St. Louis Fed.
API Key: Free at https://fred.stlouisfed.org/docs/api/api_key.html

Key Series for Trading:
- GDP, Unemployment, Inflation (CPI)
- Interest Rates (Fed Funds, Treasury Yields)
- Credit Spreads, Volatility (VIX)
- Leading Economic Indicators
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FREDSeries:
    """FRED series metadata."""

    series_id: str
    name: str
    category: str
    frequency: str  # D, W, M, Q, A
    units: str
    seasonal_adj: bool = True
    transform: Optional[str] = None  # pct_change, diff, log, etc.


# Key economic series for trading signals
TRADING_SERIES: Dict[str, FREDSeries] = {
    # Interest Rates
    "DFF": FREDSeries("DFF", "Fed Funds Rate", "rates", "D", "Percent"),
    "DGS2": FREDSeries("DGS2", "2-Year Treasury", "rates", "D", "Percent"),
    "DGS10": FREDSeries("DGS10", "10-Year Treasury", "rates", "D", "Percent"),
    "DGS30": FREDSeries("DGS30", "30-Year Treasury", "rates", "D", "Percent"),
    "T10Y2Y": FREDSeries("T10Y2Y", "10Y-2Y Spread", "rates", "D", "Percent"),
    "T10Y3M": FREDSeries("T10Y3M", "10Y-3M Spread", "rates", "D", "Percent"),
    # Credit Spreads
    "BAMLH0A0HYM2": FREDSeries(
        "BAMLH0A0HYM2", "HY OAS Spread", "credit", "D", "Percent"
    ),
    "BAMLC0A0CM": FREDSeries("BAMLC0A0CM", "IG Corp Spread", "credit", "D", "Percent"),
    # Volatility & Risk
    "VIXCLS": FREDSeries("VIXCLS", "VIX Index", "volatility", "D", "Index"),
    # Inflation
    "CPIAUCSL": FREDSeries(
        "CPIAUCSL", "CPI All Urban", "inflation", "M", "Index", transform="pct_change"
    ),
    "CPILFESL": FREDSeries(
        "CPILFESL", "Core CPI", "inflation", "M", "Index", transform="pct_change"
    ),
    "T5YIE": FREDSeries("T5YIE", "5Y Breakeven Inflation", "inflation", "D", "Percent"),
    "T10YIE": FREDSeries(
        "T10YIE", "10Y Breakeven Inflation", "inflation", "D", "Percent"
    ),
    # Employment
    "UNRATE": FREDSeries("UNRATE", "Unemployment Rate", "employment", "M", "Percent"),
    "PAYEMS": FREDSeries(
        "PAYEMS", "Nonfarm Payrolls", "employment", "M", "Thousands", transform="diff"
    ),
    "ICSA": FREDSeries("ICSA", "Initial Claims", "employment", "W", "Thousands"),
    "CCSA": FREDSeries("CCSA", "Continuing Claims", "employment", "W", "Thousands"),
    # GDP & Output
    "GDP": FREDSeries(
        "GDP", "Real GDP", "output", "Q", "Billions", transform="pct_change"
    ),
    "INDPRO": FREDSeries(
        "INDPRO",
        "Industrial Production",
        "output",
        "M",
        "Index",
        transform="pct_change",
    ),
    # Consumer
    "UMCSENT": FREDSeries("UMCSENT", "Consumer Sentiment", "consumer", "M", "Index"),
    "RSXFS": FREDSeries(
        "RSXFS", "Retail Sales", "consumer", "M", "Millions", transform="pct_change"
    ),
    # Housing
    "HOUST": FREDSeries("HOUST", "Housing Starts", "housing", "M", "Thousands"),
    "PERMIT": FREDSeries("PERMIT", "Building Permits", "housing", "M", "Thousands"),
    # Money Supply & Liquidity
    "M2SL": FREDSeries(
        "M2SL", "M2 Money Supply", "money", "M", "Billions", transform="pct_change"
    ),
    "WALCL": FREDSeries(
        "WALCL", "Fed Balance Sheet", "money", "W", "Millions", transform="pct_change"
    ),
    # Leading Indicators
    "USSLIND": FREDSeries("USSLIND", "Leading Index", "leading", "M", "Percent"),
}


class FREDCollector:
    """
    Collector for FRED economic data.

    Free API with generous limits (120 requests/minute).
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rate_limit: float = 0.5,  # seconds between requests
    ):
        """
        Initialize FRED collector.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
            cache_dir: Directory for caching data
            rate_limit: Minimum seconds between API calls
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data_cache", "fred"
        )
        self.rate_limit = rate_limit
        self._last_request = 0.0

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        if not self.api_key:
            logger.warning(
                "No FRED API key provided. Get free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make API request to FRED.

        Args:
            endpoint: API endpoint (e.g., 'series/observations')
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        if not self.api_key:
            logger.error("FRED API key required")
            return None

        try:
            import requests
        except ImportError:
            logger.error("requests library required: pip install requests")
            return None

        self._rate_limit_wait()

        params["api_key"] = self.api_key
        params["file_type"] = "json"

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            return None

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        cache_days: int = 1,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a FRED series.

        Args:
            series_id: FRED series ID (e.g., 'DGS10')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            cache_days: Days before cache expires

        Returns:
            DataFrame with date index and value column
        """
        # Default date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )

        # Check cache
        cache_file = os.path.join(self.cache_dir, f"{series_id}.parquet")
        if use_cache and os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(cache_file)
            )
            if cache_age.days < cache_days:
                try:
                    df = pd.read_parquet(cache_file)
                    # Filter to requested date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    logger.debug(f"Loaded {series_id} from cache")
                    return df
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}")

        # Fetch from API
        data = self._make_request(
            "series/observations",
            {
                "series_id": series_id,
                "observation_start": start_date,
                "observation_end": end_date,
            },
        )

        if not data or "observations" not in data:
            logger.error(f"Failed to fetch series: {series_id}")
            return None

        # Parse observations
        records = []
        for obs in data["observations"]:
            try:
                value = float(obs["value"]) if obs["value"] != "." else np.nan
                records.append({"date": obs["date"], "value": value})
            except (ValueError, KeyError):
                continue

        if not records:
            logger.warning(f"No valid observations for: {series_id}")
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.columns = [series_id]

        # Cache the data
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return df

    def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        align: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series and combine.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date
            end_date: End date
            align: Forward-fill to align different frequencies

        Returns:
            DataFrame with all series as columns
        """
        dfs = []

        for series_id in series_ids:
            df = self.get_series(series_id, start_date, end_date)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Combine all series
        combined = pd.concat(dfs, axis=1)

        if align:
            # Forward-fill for different frequencies
            combined = combined.ffill()

        return combined

    def get_trading_signals(
        self,
        categories: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get key economic series for trading signals.

        Args:
            categories: Filter to specific categories
                       ['rates', 'credit', 'volatility', 'inflation',
                        'employment', 'output', 'consumer', 'housing',
                        'money', 'leading']
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with economic indicators
        """
        series_ids = []

        for sid, meta in TRADING_SERIES.items():
            if categories is None or meta.category in categories:
                series_ids.append(sid)

        df = self.get_multiple_series(series_ids, start_date, end_date)

        # Apply transforms
        for sid, meta in TRADING_SERIES.items():
            if sid in df.columns and meta.transform:
                if meta.transform == "pct_change":
                    df[f"{sid}_pct"] = df[sid].pct_change() * 100
                elif meta.transform == "diff":
                    df[f"{sid}_diff"] = df[sid].diff()
                elif meta.transform == "log":
                    df[f"{sid}_log"] = np.log(df[sid])

        return df

    def calculate_yield_curve_features(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate yield curve features for regime detection.

        Returns:
            DataFrame with yield curve metrics
        """
        # Get treasury yields
        yields = self.get_multiple_series(
            ["DGS2", "DGS10", "DGS30", "T10Y2Y", "T10Y3M"], start_date, end_date
        )

        if yields.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=yields.index)

        # Yield levels
        features["yield_2y"] = yields.get("DGS2", np.nan)
        features["yield_10y"] = yields.get("DGS10", np.nan)
        features["yield_30y"] = yields.get("DGS30", np.nan)

        # Spreads (already provided by FRED)
        features["spread_10y_2y"] = yields.get("T10Y2Y", np.nan)
        features["spread_10y_3m"] = yields.get("T10Y3M", np.nan)

        # Curve slope (long vs short)
        if "DGS30" in yields.columns and "DGS2" in yields.columns:
            features["curve_slope"] = yields["DGS30"] - yields["DGS2"]

        # Curve curvature (belly)
        if all(col in yields.columns for col in ["DGS2", "DGS10", "DGS30"]):
            features["curve_curvature"] = (
                2 * yields["DGS10"] - yields["DGS2"] - yields["DGS30"]
            )

        # Yield changes
        features["yield_10y_chg_5d"] = features["yield_10y"].diff(5)
        features["yield_10y_chg_20d"] = features["yield_10y"].diff(20)

        # Inversion signal
        features["curve_inverted"] = (features["spread_10y_2y"] < 0).astype(int)

        return features

    def calculate_credit_features(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate credit market features.

        Returns:
            DataFrame with credit metrics
        """
        credit = self.get_multiple_series(
            ["BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS"], start_date, end_date
        )

        if credit.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=credit.index)

        # Spread levels
        features["hy_spread"] = credit.get("BAMLH0A0HYM2", np.nan)
        features["ig_spread"] = credit.get("BAMLC0A0CM", np.nan)
        features["vix"] = credit.get("VIXCLS", np.nan)

        # Spread changes
        for col in ["hy_spread", "ig_spread", "vix"]:
            if col in features.columns:
                features[f"{col}_chg_5d"] = features[col].diff(5)
                features[f"{col}_chg_20d"] = features[col].diff(20)
                features[f"{col}_zscore_60d"] = (
                    features[col] - features[col].rolling(60).mean()
                ) / features[col].rolling(60).std()

        # Credit-equity ratio (HY spread / VIX)
        if "hy_spread" in features.columns and "vix" in features.columns:
            features["credit_equity_ratio"] = features["hy_spread"] / features["vix"]

        # Stress indicator
        features["stress_indicator"] = (
            features.get("hy_spread_zscore_60d", 0) + features.get("vix_zscore_60d", 0)
        ) / 2

        return features

    def calculate_economic_momentum(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate economic momentum indicators.

        Returns:
            DataFrame with momentum metrics
        """
        # Get key economic series
        econ = self.get_trading_signals(
            categories=["employment", "output", "consumer", "leading"],
            start_date=start_date,
            end_date=end_date,
        )

        if econ.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=econ.index)

        # Employment momentum
        if "ICSA" in econ.columns:
            features["claims_4wma"] = econ["ICSA"].rolling(4).mean()
            features["claims_momentum"] = (
                features["claims_4wma"].pct_change(4) * -100
            )  # Negative claims = positive

        if "UNRATE" in econ.columns:
            features["unemployment"] = econ["UNRATE"]
            features["unemployment_trend"] = (
                econ["UNRATE"].diff(3) * -1
            )  # Falling unemployment = positive

        # Output momentum
        if "INDPRO_pct" in econ.columns:
            features["indpro_momentum"] = econ["INDPRO_pct"].rolling(3).mean()

        # Consumer momentum
        if "UMCSENT" in econ.columns:
            features["sentiment"] = econ["UMCSENT"]
            features["sentiment_momentum"] = econ["UMCSENT"].pct_change(3) * 100

        # Leading indicator
        if "USSLIND" in econ.columns:
            features["leading_index"] = econ["USSLIND"]
            features["leading_momentum"] = econ["USSLIND"].rolling(3).mean()

        # Composite economic momentum
        momentum_cols = [
            col for col in features.columns if "momentum" in col or "trend" in col
        ]
        if momentum_cols:
            # Normalize each momentum indicator
            normalized = features[momentum_cols].apply(
                lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std()
            )
            features["composite_momentum"] = normalized.mean(axis=1)

        return features

    def get_regime_indicators(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get comprehensive indicators for regime detection.

        Combines yield curve, credit, and economic momentum.

        Returns:
            DataFrame with regime indicators
        """
        # Get all features
        yield_features = self.calculate_yield_curve_features(start_date, end_date)
        credit_features = self.calculate_credit_features(start_date, end_date)
        econ_features = self.calculate_economic_momentum(start_date, end_date)

        # Combine
        dfs = [
            df
            for df in [yield_features, credit_features, econ_features]
            if not df.empty
        ]

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, axis=1)
        combined = combined.ffill()  # Align frequencies

        # Add regime classification hints
        combined["recession_risk"] = 0.0

        # Yield curve inversion = recession warning
        if "curve_inverted" in combined.columns:
            combined["recession_risk"] += combined["curve_inverted"] * 0.3

        # High credit spreads = stress
        if "hy_spread_zscore_60d" in combined.columns:
            combined["recession_risk"] += (
                combined["hy_spread_zscore_60d"].clip(0, 3) / 3 * 0.3
            )

        # Falling economic momentum
        if "composite_momentum" in combined.columns:
            combined["recession_risk"] += (
                (-combined["composite_momentum"]).clip(0, 3) / 3 * 0.4
            )

        combined["recession_risk"] = combined["recession_risk"].clip(0, 1)

        return combined


class FREDFallback:
    """
    Fallback when FRED API unavailable.

    Uses static averages for economic indicators.
    """

    # Historical averages for key indicators
    HISTORICAL_AVERAGES = {
        "DGS10": 4.0,  # 10Y Treasury
        "T10Y2Y": 1.0,  # Yield spread
        "VIXCLS": 18.0,  # VIX
        "BAMLH0A0HYM2": 4.5,  # HY spread
        "UNRATE": 5.0,  # Unemployment
    }

    def get_default_features(self) -> Dict[str, float]:
        """Get default feature values."""
        return {
            "yield_10y": self.HISTORICAL_AVERAGES["DGS10"],
            "spread_10y_2y": self.HISTORICAL_AVERAGES["T10Y2Y"],
            "curve_inverted": 0,
            "vix": self.HISTORICAL_AVERAGES["VIXCLS"],
            "hy_spread": self.HISTORICAL_AVERAGES["BAMLH0A0HYM2"],
            "unemployment": self.HISTORICAL_AVERAGES["UNRATE"],
            "recession_risk": 0.2,  # Baseline
            "stress_indicator": 0.0,
            "composite_momentum": 0.0,
        }


def create_fred_collector(api_key: Optional[str] = None, **kwargs) -> FREDCollector:
    """
    Factory function to create FRED collector.

    Args:
        api_key: FRED API key
        **kwargs: Additional arguments for FREDCollector

    Returns:
        FREDCollector instance
    """
    return FREDCollector(api_key=api_key, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize collector
    collector = FREDCollector()

    if collector.api_key:
        # Test fetching a single series
        print("Fetching 10-Year Treasury...")
        dgs10 = collector.get_series("DGS10")
        if dgs10 is not None:
            print(f"Got {len(dgs10)} observations")
            print(dgs10.tail())

        # Test regime indicators
        print("\nFetching regime indicators...")
        regime = collector.get_regime_indicators()
        if not regime.empty:
            print(f"Got {len(regime)} observations, {len(regime.columns)} features")
            print(regime.tail())
    else:
        print("No API key - using fallback")
        fallback = FREDFallback()
        print(fallback.get_default_features())
