"""
SEC EDGAR Data Collector.

FREE access to SEC filings:
- 10-K (Annual Reports)
- 10-Q (Quarterly Reports)
- 8-K (Current Reports)
- Form 4 (Insider Trading)

Uses the official SEC EDGAR API (no API key required).
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import re
import xml.etree.ElementTree as ET
import warnings


@dataclass
class SECFiling:
    """Container for SEC filing data."""

    cik: str
    company_name: str
    form_type: str
    filing_date: datetime
    accession_number: str
    file_url: str
    description: str = ""


class SECEdgarCollector:
    """
    Collector for SEC EDGAR filings.

    FREE data source for:
    - Company financial filings (10-K, 10-Q, 8-K)
    - Insider trading (Form 4)
    - Institutional holdings (13F)

    Rate limit: 10 requests per second (SEC guideline)
    """

    BASE_URL = "https://data.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"

    # User agent required by SEC
    HEADERS = {
        "User-Agent": "QuantumAlpha Trading Research (contact@example.com)",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

    # CIK lookup for common tickers
    TICKER_CIK_CACHE = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
        "TSLA": "0001318605",
        "META": "0001326801",
        "NVDA": "0001045810",
        "JPM": "0000019617",
        "V": "0001403161",
        "JNJ": "0000200406",
    }

    def __init__(
        self,
        user_agent: Optional[str] = None,
        rate_limit: float = 0.1,  # Seconds between requests
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize SEC EDGAR collector.

        Args:
            user_agent: User agent string (required by SEC)
            rate_limit: Seconds between requests
            cache_dir: Directory for caching filings
        """
        if user_agent:
            self.HEADERS["User-Agent"] = user_agent

        self.rate_limit = rate_limit
        self.cache_dir = cache_dir
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make rate-limited request."""
        self._rate_limit_wait()

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            warnings.warn(f"SEC EDGAR request failed: {e}")
            return None

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK number (padded to 10 digits)
        """
        ticker = ticker.upper()

        # Check cache first
        if ticker in self.TICKER_CIK_CACHE:
            return self.TICKER_CIK_CACHE[ticker]

        # Query SEC ticker lookup
        url = f"{self.BASE_URL}/cik-lookup?company={ticker}"

        # Alternative: use company tickers JSON
        tickers_url = f"{self.BASE_URL}/files/company_tickers.json"
        response = self._make_request(tickers_url)

        if response:
            try:
                data = response.json()
                for key, company in data.items():
                    if company.get("ticker", "").upper() == ticker:
                        cik = str(company.get("cik_str", "")).zfill(10)
                        self.TICKER_CIK_CACHE[ticker] = cik
                        return cik
            except Exception as e:
                warnings.warn(f"CIK lookup failed: {e}")

        return None

    def get_company_filings(
        self,
        ticker: str,
        form_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SECFiling]:
        """
        Get company filings from SEC EDGAR.

        Args:
            ticker: Stock ticker symbol
            form_types: List of form types (e.g., ['10-K', '10-Q', '8-K'])
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of filings

        Returns:
            List of SECFiling objects
        """
        cik = self.get_cik(ticker)
        if not cik:
            warnings.warn(f"Could not find CIK for {ticker}")
            return []

        # Get company submissions
        url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
        response = self._make_request(url)

        if not response:
            return []

        try:
            data = response.json()
        except Exception as e:
            warnings.warn(f"Failed to parse response: {e}")
            return []

        company_name = data.get("name", ticker)
        filings_data = data.get("filings", {}).get("recent", {})

        if not filings_data:
            return []

        # Extract filing lists
        forms = filings_data.get("form", [])
        dates = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        descriptions = filings_data.get("primaryDocument", [])

        filings = []
        form_types = form_types or ["10-K", "10-Q", "8-K", "4"]
        form_types = [f.upper() for f in form_types]

        for i in range(min(len(forms), limit * 5)):  # Over-fetch to filter
            form_type = forms[i] if i < len(forms) else ""

            # Filter by form type
            if form_type.upper() not in form_types:
                continue

            filing_date_str = dates[i] if i < len(dates) else ""
            try:
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
            except:
                continue

            # Filter by date
            if start_date and filing_date < start_date:
                continue
            if end_date and filing_date > end_date:
                continue

            accession = accessions[i] if i < len(accessions) else ""
            description = descriptions[i] if i < len(descriptions) else ""

            # Build file URL
            accession_no_dashes = accession.replace("-", "")
            file_url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession_no_dashes}/{description}"

            filing = SECFiling(
                cik=cik,
                company_name=company_name,
                form_type=form_type,
                filing_date=filing_date,
                accession_number=accession,
                file_url=file_url,
                description=description,
            )

            filings.append(filing)

            if len(filings) >= limit:
                break

        return filings

    def get_insider_trades(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """
        Get insider trading data (Form 4 filings).

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            DataFrame with insider trades
        """
        start_date = datetime.now() - timedelta(days=days)

        filings = self.get_company_filings(
            ticker, form_types=["4"], start_date=start_date, limit=100
        )

        trades = []

        for filing in filings:
            # Parse Form 4 XML (simplified)
            try:
                response = self._make_request(filing.file_url)
                if response and filing.file_url.endswith(".xml"):
                    trade_info = self._parse_form4(response.text, filing)
                    if trade_info:
                        trades.extend(trade_info)
            except Exception as e:
                continue

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        return df

    def _parse_form4(self, xml_content: str, filing: SECFiling) -> List[Dict]:
        """Parse Form 4 XML for insider trade details."""
        trades = []

        try:
            root = ET.fromstring(xml_content)

            # Get reporting owner
            owner_elem = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
            owner_name = owner_elem.text if owner_elem is not None else "Unknown"

            # Get transactions
            for trans in root.findall(".//nonDerivativeTransaction"):
                code_elem = trans.find(".//transactionCoding/transactionCode")
                shares_elem = trans.find(
                    ".//transactionAmounts/transactionShares/value"
                )
                price_elem = trans.find(
                    ".//transactionAmounts/transactionPricePerShare/value"
                )

                trans_code = code_elem.text if code_elem is not None else ""
                shares = float(shares_elem.text) if shares_elem is not None else 0
                price = float(price_elem.text) if price_elem is not None else 0

                # P = Purchase, S = Sale
                is_buy = trans_code in ["P", "A", "M"]

                trades.append(
                    {
                        "date": filing.filing_date,
                        "owner": owner_name,
                        "transaction_type": "buy" if is_buy else "sell",
                        "shares": shares,
                        "price": price,
                        "value": shares * price,
                        "form_type": "4",
                    }
                )

        except ET.ParseError:
            pass
        except Exception as e:
            pass

        return trades

    def get_financial_data(
        self, ticker: str, metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get company financials from SEC filings.

        Args:
            ticker: Stock ticker symbol
            metrics: List of metrics to extract (e.g., ['Revenue', 'NetIncome'])

        Returns:
            DataFrame with financial metrics over time
        """
        cik = self.get_cik(ticker)
        if not cik:
            return pd.DataFrame()

        # Use SEC company facts API
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        response = self._make_request(url)

        if not response:
            return pd.DataFrame()

        try:
            data = response.json()
        except:
            return pd.DataFrame()

        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        # Default metrics
        if metrics is None:
            metrics = [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "NetIncomeLoss",
                "EarningsPerShareBasic",
                "Assets",
                "Liabilities",
                "StockholdersEquity",
                "OperatingIncomeLoss",
                "GrossProfit",
            ]

        records = []

        for metric in metrics:
            if metric not in us_gaap:
                continue

            units = us_gaap[metric].get("units", {})

            # Get USD values
            for unit_type, values in units.items():
                if unit_type in ["USD", "USD/shares"]:
                    for v in values:
                        records.append(
                            {
                                "metric": metric,
                                "value": v.get("val"),
                                "end_date": v.get("end"),
                                "form": v.get("form"),
                                "fiscal_year": v.get("fy"),
                                "fiscal_period": v.get("fp"),
                            }
                        )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["end_date"] = pd.to_datetime(df["end_date"])

        # Pivot to wide format
        pivot = df.pivot_table(
            values="value", index="end_date", columns="metric", aggfunc="last"
        )

        return pivot.sort_index()

    def get_filing_text(self, filing: SECFiling, section: Optional[str] = None) -> str:
        """
        Get text content of a filing.

        Args:
            filing: SECFiling object
            section: Optional section to extract (e.g., 'item1a' for Risk Factors)

        Returns:
            Filing text content
        """
        response = self._make_request(filing.file_url)

        if not response:
            return ""

        text = response.text

        # Basic HTML cleanup
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Extract specific section if requested
        if section:
            section_patterns = {
                "item1a": r"Item\s*1A[.\s]*Risk\s*Factors(.*?)Item\s*1B",
                "item7": r"Item\s*7[.\s]*Management.*?Discussion(.*?)Item\s*7A",
                "item8": r"Item\s*8[.\s]*Financial\s*Statements(.*?)Item\s*9",
            }

            pattern = section_patterns.get(section.lower())
            if pattern:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    text = match.group(1)

        return text.strip()
