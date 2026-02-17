"""
GDELT Historical Tone Fetcher
==============================
Fetches daily news sentiment (tone) from the GDELT DOC 2.0 TimelineTone API
for all symbols in the meta-ensemble universe.

Coverage: Jan 2017 to present (daily granularity)
Source: https://api.gdeltproject.org/api/v2/doc/doc
No API key required.

Output: Pickle file at data_store/gdelt_tone/gdelt_daily_tone.pkl
        DataFrame with columns: date, symbol, tone, tone_ma5, tone_ma20,
        tone_zscore, tone_momentum, tone_reversal, news_tone_regime

Usage:
    python -m quantum_alpha.data.collectors.gdelt_historical
    python -m quantum_alpha.data.collectors.gdelt_historical --symbols AAPL MSFT TSLA
    python -m quantum_alpha.data.collectors.gdelt_historical --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data_store" / "gdelt_tone"
OUTPUT_FILE = OUTPUT_DIR / "gdelt_daily_tone.pkl"
RAW_CACHE_DIR = OUTPUT_DIR / "raw_cache"

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_DELAY = 5.0  # seconds between API calls (be nice to GDELT)
MAX_RETRIES = 3
RETRY_DELAY = 30.0  # seconds to wait after 429

# Symbol -> GDELT search query (first term is most important)
# For symbols not listed, fallback is: "{COMPANY_NAME} stock" or "{SYMBOL} stock"
SYMBOL_QUERIES = {
    "SPY": '"S&P 500" stock market',
    "QQQ": '"Nasdaq" stock market',
    "IWM": '"Russell 2000" stock',
    "DIA": '"Dow Jones" stock market',
    "AAPL": '"Apple stock"',
    "MSFT": '"Microsoft stock"',
    "AMZN": '"Amazon stock"',
    "GOOGL": '"Google stock"',
    "TSLA": '"Tesla stock"',
    "META": '"Meta Platforms" OR "Facebook stock"',
    "NVDA": '"Nvidia stock"',
    "BRK-B": '"Berkshire Hathaway"',
    "JPM": '"JPMorgan" stock',
    "V": '"Visa" stock',
    "UNH": '"UnitedHealth" stock',
    "XOM": '"Exxon" stock',
    "LLY": '"Eli Lilly" stock',
    "AVGO": '"Broadcom" stock',
    "MA": '"Mastercard" stock',
    "HD": '"Home Depot" stock',
    "PG": '"Procter Gamble" stock',
    "COST": '"Costco" stock',
    "ABBV": '"AbbVie" stock',
    "WMT": '"Walmart" stock',
    "CRM": '"Salesforce" stock',
    "BAC": '"Bank of America" stock',
    "AMD": '"AMD" stock',
    "NFLX": '"Netflix" stock',
    "DIS": '"Disney" stock',
    "KO": '"Coca-Cola" stock',
    "PEP": '"PepsiCo" stock',
    "BA": '"Boeing" stock',
    "GS": '"Goldman Sachs" stock',
    "GM": '"General Motors" stock',
    "F": '"Ford Motor" stock',
    "ABT": '"Abbott Laboratories" stock',
    "ACN": '"Accenture" stock',
    "ADBE": '"Adobe" stock',
    "ADP": '"ADP" stock payroll',
    "AMAT": '"Applied Materials" stock',
    "AMT": '"American Tower" stock',
    "APD": '"Air Products" stock',
    "AXP": '"American Express" stock',
    "BKNG": '"Booking Holdings" stock',
    "BLK": '"BlackRock" stock',
    "C": '"Citigroup" stock',
    "CAT": '"Caterpillar" stock',
    "CB": '"Chubb" stock insurance',
    "CCI": '"Crown Castle" stock',
    "CME": '"CME Group" stock',
    "COP": '"ConocoPhillips" stock',
    "CSCO": '"Cisco" stock',
    "CVS": '"CVS Health" stock',
    "CVX": '"Chevron" stock',
    "DE": '"Deere" stock',
    "DHR": '"Danaher" stock',
    "DUK": '"Duke Energy" stock',
    "EOG": '"EOG Resources" stock',
    "FDX": '"FedEx" stock',
    "GE": '"General Electric" stock',
    "GILD": '"Gilead Sciences" stock',
    "HON": '"Honeywell" stock',
    "IBM": '"IBM" stock',
    "INTC": '"Intel" stock',
    "INTU": '"Intuit" stock',
    "ISRG": '"Intuitive Surgical" stock',
    "LIN": '"Linde" stock gas',
    "LMT": '"Lockheed Martin" stock',
    "LOW": '"Lowe" stock',
    "LRCX": '"Lam Research" stock',
    "MCD": '"McDonald" stock',
    "MDT": '"Medtronic" stock',
    "MMM": '"3M" stock',
    "MO": '"Altria" stock',
    "MRK": '"Merck" stock',
    "MS": '"Morgan Stanley" stock',
    "NEE": '"NextEra Energy" stock',
    "NKE": '"Nike" stock',
    "NOW": '"ServiceNow" stock',
    "ORCL": '"Oracle" stock',
    "PFE": '"Pfizer" stock',
    "PLD": '"Prologis" stock',
    "PNC": '"PNC Financial" stock',
    "PYPL": '"PayPal" stock',
    "QCOM": '"Qualcomm" stock',
    "RTX": '"Raytheon" stock',
    "SBUX": '"Starbucks" stock',
    "SCHW": '"Charles Schwab" stock',
    "SLB": '"Schlumberger" stock',
    "SNPS": '"Synopsys" stock',
    "SO": '"Southern Company" stock',
    "SPGI": '"S&P Global" stock',
    "T": '"AT&T" stock',
    "TGT": '"Target" stock',
    "TMO": '"Thermo Fisher" stock',
    "TXN": '"Texas Instruments" stock',
    "UNP": '"Union Pacific" stock',
    "UPS": '"UPS" stock',
    "USB": '"US Bancorp" stock',
    "VRTX": '"Vertex Pharmaceuticals" stock',
    "VZ": '"Verizon" stock',
    "ZTS": '"Zoetis" stock',
}

# For any META-era query that uses OR, parentheses are needed
# GDELT requires OR terms to be wrapped in parens
_PAREN_NEEDED = {"META"}


def _get_query(symbol: str) -> str:
    """Return the GDELT search query for a symbol."""
    q = SYMBOL_QUERIES.get(symbol, f'"{symbol} stock"')
    # GDELT requires parentheses around OR queries
    if " OR " in q:
        q = f"({q})"
    return q


def _fetch_timeline_tone(
    query: str,
    start_dt: str,
    end_dt: str,
    retries: int = MAX_RETRIES,
) -> list[dict] | None:
    """
    Fetch daily tone timeline from GDELT DOC API.

    Args:
        query: Search query string
        start_dt: Start datetime in YYYYMMDDHHMMSS format
        end_dt: End datetime in YYYYMMDDHHMMSS format

    Returns:
        List of {date, value} dicts, or None on failure.
    """
    params = {
        "query": query,
        "mode": "timelinetone",
        "format": "json",
        "STARTDATETIME": start_dt,
        "ENDDATETIME": end_dt,
    }
    url = f"{GDELT_BASE_URL}?{urllib.parse.urlencode(params)}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
                if not data.strip():
                    return []
                j = json.loads(data)
                tl = j.get("timeline", [])
                if tl and isinstance(tl, list) and len(tl) > 0:
                    return tl[0].get("data", [])
                return []
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"Rate limited (429). Waiting {wait:.0f}s before retry "
                    f"{attempt + 1}/{retries}..."
                )
                time.sleep(wait)
            else:
                logger.error(f"HTTP {e.code} for query '{query[:50]}': {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {e}")
            if attempt < retries - 1:
                time.sleep(10)
            else:
                return None
    return None


def fetch_symbol_tone(
    symbol: str,
    start_year: int = 2017,
    end_year: int = 2026,
    chunk_years: int = 4,
) -> pd.DataFrame:
    """
    Fetch daily tone for a single symbol across the full date range.

    Splits into multi-year chunks to stay within GDELT API limits.

    Returns:
        DataFrame with columns: date, symbol, tone
    """
    query = _get_query(symbol)
    all_records = []

    # Split into chunks
    year = start_year
    while year < end_year:
        chunk_end = min(year + chunk_years, end_year)
        start_dt = f"{year}0101000000"
        end_dt = f"{chunk_end}0101000000"
        if chunk_end == end_year:
            # Use today's date for the last chunk
            end_dt = datetime.now().strftime("%Y%m%d") + "000000"

        logger.info(f"  Fetching {symbol} {year}-{chunk_end}...")
        data_pts = _fetch_timeline_tone(query, start_dt, end_dt)

        if data_pts is None:
            logger.warning(f"  Failed to fetch {symbol} {year}-{chunk_end}")
        elif data_pts:
            for pt in data_pts:
                try:
                    dt = datetime.strptime(pt["date"], "%Y%m%dT%H%M%SZ")
                    all_records.append(
                        {
                            "date": dt.strftime("%Y-%m-%d"),
                            "symbol": symbol,
                            "tone": pt["value"],
                        }
                    )
                except (ValueError, KeyError):
                    continue

        time.sleep(REQUEST_DELAY)
        year = chunk_end

    if not all_records:
        return pd.DataFrame(columns=["date", "symbol", "tone"])

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    # Deduplicate (chunks may overlap at boundaries)
    df = df.drop_duplicates(subset=["date", "symbol"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_tone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived sentiment features from raw daily tone.

    Input: DataFrame with date, symbol, tone
    Output: DataFrame with additional columns:
        - tone_ma5: 5-day moving average of tone
        - tone_ma20: 20-day moving average of tone
        - tone_zscore: Z-score of tone over 60-day lookback
        - tone_momentum: 5-day change in tone_ma5
        - tone_reversal: tone - tone_ma20 (mean reversion signal)
        - news_tone_regime: 1 if tone_ma20 > 0, -1 if < 0, 0 if near zero
        - tone_volatility: 20-day rolling std of tone
        - tone_acceleration: change in tone_momentum (2nd derivative)
    """
    if len(df) == 0:
        return df

    result = df.copy()
    result = result.sort_values(["symbol", "date"])

    features = []
    for symbol, grp in result.groupby("symbol"):
        grp = grp.sort_values("date").copy()
        tone = grp["tone"]

        # Moving averages
        grp["tone_ma5"] = tone.rolling(5, min_periods=1).mean()
        grp["tone_ma20"] = tone.rolling(20, min_periods=5).mean()

        # Z-score (60-day lookback)
        roll_mean = tone.rolling(60, min_periods=20).mean()
        roll_std = tone.rolling(60, min_periods=20).std()
        grp["tone_zscore"] = np.where(
            roll_std > 0.01, (tone - roll_mean) / roll_std, 0.0
        )

        # Momentum (5-day change in MA5)
        grp["tone_momentum"] = grp["tone_ma5"].diff(5)

        # Mean reversion signal
        grp["tone_reversal"] = tone - grp["tone_ma20"]

        # Regime (bullish/bearish news)
        grp["news_tone_regime"] = np.where(
            grp["tone_ma20"] > 0.3, 1.0, np.where(grp["tone_ma20"] < -0.3, -1.0, 0.0)
        )

        # Volatility of tone
        grp["tone_volatility"] = tone.rolling(20, min_periods=5).std()

        # Acceleration (2nd derivative)
        grp["tone_acceleration"] = grp["tone_momentum"].diff(5)

        features.append(grp)

    return pd.concat(features, ignore_index=True)


def fetch_all_symbols(
    symbols: list[str],
    start_year: int = 2017,
    end_year: int = 2026,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch tone data for all symbols with resume support.

    Saves raw data per symbol to RAW_CACHE_DIR for resume capability.
    """
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    n_total = len(symbols)

    for i, symbol in enumerate(symbols):
        cache_file = RAW_CACHE_DIR / f"{symbol}.pkl"

        if resume and cache_file.exists():
            logger.info(
                f"[{i + 1}/{n_total}] {symbol}: loaded from cache "
                f"({cache_file.stat().st_size / 1024:.0f} KB)"
            )
            with open(cache_file, "rb") as f:
                sym_df = pickle.load(f)
            all_dfs.append(sym_df)
            continue

        logger.info(f"[{i + 1}/{n_total}] Fetching {symbol}...")
        sym_df = fetch_symbol_tone(symbol, start_year, end_year)

        if len(sym_df) > 0:
            with open(cache_file, "wb") as f:
                pickle.dump(sym_df, f)
            logger.info(
                f"  -> {len(sym_df)} daily tone records "
                f"({sym_df['date'].min().strftime('%Y-%m-%d')} to "
                f"{sym_df['date'].max().strftime('%Y-%m-%d')})"
            )
        else:
            logger.warning(f"  -> No data for {symbol}")
            # Save empty DataFrame to avoid re-fetching
            with open(cache_file, "wb") as f:
                pickle.dump(sym_df, f)

        all_dfs.append(sym_df)

    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if len(combined) > 0:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Compute derived features
        logger.info("Computing derived tone features...")
        combined = compute_tone_features(combined)

        # Save
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(combined, f)
        logger.info(
            f"Saved {len(combined)} records to {OUTPUT_FILE} "
            f"({OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB)"
        )

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fetch GDELT historical tone data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to fetch (default: all 99 meta-ensemble symbols)",
    )
    parser.add_argument(
        "--start-year", type=int, default=2017, help="Start year (default: 2017)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2026, help="End year (default: 2026)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from cached data (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force re-fetch all symbols",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.symbols:
        symbols = args.symbols
    else:
        # Default: all 99 meta-ensemble symbols
        symbols = [
            "AAPL",
            "ABBV",
            "ABT",
            "ACN",
            "ADBE",
            "ADP",
            "AMAT",
            "AMD",
            "AMT",
            "AMZN",
            "APD",
            "AVGO",
            "AXP",
            "BA",
            "BAC",
            "BKNG",
            "BLK",
            "BRK-B",
            "C",
            "CAT",
            "CB",
            "CCI",
            "CME",
            "COP",
            "COST",
            "CRM",
            "CSCO",
            "CVS",
            "CVX",
            "DE",
            "DHR",
            "DIS",
            "DUK",
            "EOG",
            "F",
            "FDX",
            "GE",
            "GILD",
            "GM",
            "GOOGL",
            "GS",
            "HD",
            "HON",
            "IBM",
            "INTC",
            "INTU",
            "ISRG",
            "JPM",
            "KO",
            "LIN",
            "LLY",
            "LMT",
            "LOW",
            "LRCX",
            "MA",
            "MCD",
            "MDT",
            "META",
            "MMM",
            "MO",
            "MRK",
            "MS",
            "MSFT",
            "NEE",
            "NFLX",
            "NKE",
            "NOW",
            "NVDA",
            "ORCL",
            "PEP",
            "PFE",
            "PG",
            "PLD",
            "PNC",
            "PYPL",
            "QCOM",
            "RTX",
            "SBUX",
            "SCHW",
            "SLB",
            "SNPS",
            "SO",
            "SPGI",
            "SPY",
            "T",
            "TGT",
            "TMO",
            "TSLA",
            "TXN",
            "UNH",
            "UNP",
            "UPS",
            "USB",
            "V",
            "VRTX",
            "VZ",
            "WMT",
            "XOM",
            "ZTS",
        ]

    resume = args.resume and not args.no_resume

    logger.info(
        f"Fetching GDELT tone for {len(symbols)} symbols ({args.start_year}-{args.end_year})"
    )
    logger.info(f"Resume: {resume}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Estimated time: ~{len(symbols) * 3 * REQUEST_DELAY / 60:.0f} minutes")

    t0 = time.time()
    result = fetch_all_symbols(symbols, args.start_year, args.end_year, resume)
    elapsed = time.time() - t0

    if len(result) > 0:
        logger.info(f"\n=== Summary ===")
        logger.info(f"Total records: {len(result):,}")
        logger.info(f"Symbols: {result['symbol'].nunique()}")
        logger.info(
            f"Date range: {result['date'].min().strftime('%Y-%m-%d')} to "
            f"{result['date'].max().strftime('%Y-%m-%d')}"
        )
        logger.info(f"Avg tone: {result['tone'].mean():.3f}")
        logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    else:
        logger.warning("No data collected!")


if __name__ == "__main__":
    main()
