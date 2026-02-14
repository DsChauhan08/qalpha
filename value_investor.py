#!/usr/bin/env python3
"""
Value Investing Screener & Portfolio Manager
==============================================
Finds undervalued S&P 500 stocks using fundamental analysis,
confirms with news sentiment, and manages a conviction portfolio.

Rules:
  1. Buy undervalued stocks (below fair value estimate)
  2. Hold until they reach fair value — could be weeks or months
  3. NEVER sell at a loss for S&P 500 stocks (they recover)
  4. Only sell at a loss if: devastating company news, or proven market collapse
  5. Rinse and repeat

Valuation Methods:
  - Analyst consensus target price (free from yfinance)
  - Relative valuation (P/E vs sector, P/B vs sector)
  - DCF-lite (FCF yield + growth rate)
  - Graham number (classic value formula)
  - Earnings yield vs 10-year Treasury

Usage:
  python value_investor.py --screen           # Screen all S&P 500
  python value_investor.py --deep AAPL MSFT   # Deep analysis on specific stocks
  python value_investor.py --portfolio        # Show current portfolio recommendations
  python value_investor.py --backtest         # Backtest value strategy historically
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/home/regulus/Trade")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_store" / "value_investing"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# S&P 500 symbols
# --------------------------------------------------------------------------

SP500_CORE = [
    # Tech
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "ORCL",
    "CRM",
    "AMD",
    "INTC",
    "CSCO",
    "ACN",
    "ADBE",
    "TXN",
    "QCOM",
    "IBM",
    "NOW",
    "AMAT",
    "MU",
    "LRCX",
    "SNPS",
    "CDNS",
    "KLAC",
    # Finance
    "JPM",
    "V",
    "MA",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "BLK",
    "SCHW",
    "AXP",
    "C",
    "USB",
    "PNC",
    "TFC",
    "COF",
    "BK",
    "CME",
    # Healthcare
    "UNH",
    "JNJ",
    "LLY",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "PFE",
    "DHR",
    "BMY",
    "AMGN",
    "MDT",
    "GILD",
    "ISRG",
    "SYK",
    "CI",
    "ELV",
    # Consumer
    "PG",
    "KO",
    "PEP",
    "COST",
    "WMT",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "HD",
    "LOW",
    "TJX",
    "DG",
    "DLTR",
    "CL",
    "EL",
    # Industrial
    "CAT",
    "BA",
    "HON",
    "UNP",
    "GE",
    "RTX",
    "DE",
    "LMT",
    "MMM",
    "FDX",
    "UPS",
    "WM",
    "ETN",
    "ITW",
    "EMR",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "MPC",
    "PSX",
    "VLO",
    "OXY",
    # Utilities / Real Estate / Telecom
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    "T",
    "VZ",
    "TMUS",
    # Materials
    "LIN",
    "APD",
    "SHW",
    "DD",
    "NEM",
    "FCX",
    # Other
    "BRK-B",
    "PM",
    "DIS",
]


def get_sp500_symbols() -> list[str]:
    """Return S&P 500 symbol list."""
    return SP500_CORE.copy()


# --------------------------------------------------------------------------
# Fundamental data fetching
# --------------------------------------------------------------------------


def fetch_fundamentals(symbol: str) -> dict | None:
    """Fetch all fundamental data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or "currentPrice" not in info:
            return None

        # Core price data
        data = {
            "symbol": symbol,
            "price": info.get("currentPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }

        # Valuation ratios
        data["trailing_pe"] = info.get("trailingPE")
        data["forward_pe"] = info.get("forwardPE")
        data["price_to_book"] = info.get("priceToBook")
        data["price_to_sales"] = info.get("priceToSalesTrailing12Months")
        data["ev_to_ebitda"] = info.get("enterpriseToEbitda")
        data["ev_to_revenue"] = info.get("enterpriseToRevenue")
        data["peg_ratio"] = info.get("pegRatio")

        # Per-share data
        data["trailing_eps"] = info.get("trailingEps")
        data["forward_eps"] = info.get("forwardEps")
        data["book_value"] = info.get("bookValue")
        data["revenue_per_share"] = info.get("revenuePerShare")

        # Cash flow
        data["free_cashflow"] = info.get("freeCashflow")
        data["operating_cashflow"] = info.get("operatingCashflow")
        data["total_revenue"] = info.get("totalRevenue")
        data["net_income"] = info.get("netIncomeToCommon")
        data["ebitda"] = info.get("ebitda")

        # Balance sheet
        data["total_debt"] = info.get("totalDebt", 0)
        data["total_cash"] = info.get("totalCash", 0)
        data["enterprise_value"] = info.get("enterpriseValue")

        # Margins & returns
        data["gross_margin"] = info.get("grossMargins")
        data["operating_margin"] = info.get("operatingMargins")
        data["profit_margin"] = info.get("profitMargins")
        data["roe"] = info.get("returnOnEquity")
        data["roa"] = info.get("returnOnAssets")

        # Dividend
        data["dividend_yield"] = info.get("dividendYield")
        data["payout_ratio"] = info.get("payoutRatio")

        # Growth
        data["earnings_growth"] = info.get("earningsGrowth")
        data["revenue_growth"] = info.get("revenueGrowth")

        # Risk
        data["beta"] = info.get("beta")
        data["debt_to_equity"] = info.get("debtToEquity")
        data["current_ratio"] = info.get("currentRatio")

        # Analyst targets
        data["target_mean"] = info.get("targetMeanPrice")
        data["target_median"] = info.get("targetMedianPrice")
        data["target_high"] = info.get("targetHighPrice")
        data["target_low"] = info.get("targetLowPrice")
        data["n_analysts"] = info.get("numberOfAnalystOpinions")
        data["recommendation"] = info.get("recommendationKey")
        data["recommendation_score"] = info.get("recommendationMean")

        # Technical context
        data["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
        data["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
        data["fifty_day_avg"] = info.get("fiftyDayAverage")
        data["two_hundred_day_avg"] = info.get("twoHundredDayAverage")

        # Compute derived
        if data["price"] and data["fifty_two_week_high"]:
            data["pct_from_52w_high"] = data["price"] / data["fifty_two_week_high"] - 1
        if data["price"] and data["fifty_two_week_low"]:
            data["pct_from_52w_low"] = data["price"] / data["fifty_two_week_low"] - 1

        return data

    except Exception as e:
        return None


# --------------------------------------------------------------------------
# Valuation models
# --------------------------------------------------------------------------


def compute_valuations(data: dict) -> dict:
    """Compute multiple fair value estimates for a stock."""
    price = data.get("price", 0)
    if not price or price <= 0:
        return {}

    valuations = {}

    # 1. Analyst consensus target
    target_mean = data.get("target_mean")
    if target_mean and target_mean > 0:
        valuations["analyst_target"] = target_mean
        valuations["analyst_upside"] = (target_mean - price) / price

    # 2. Graham Number: sqrt(22.5 * EPS * Book Value)
    eps = data.get("trailing_eps")
    bv = data.get("book_value")
    if eps and bv and eps > 0 and bv > 0:
        graham = np.sqrt(22.5 * eps * bv)
        valuations["graham_number"] = graham
        valuations["graham_upside"] = (graham - price) / price

    # 3. Earnings Yield vs Treasury
    #    If earnings yield > 2x the 10yr Treasury, stock is cheap
    if eps and eps > 0:
        earnings_yield = eps / price
        valuations["earnings_yield"] = earnings_yield
        # Approximate 10yr Treasury at ~4.2% (Feb 2026)
        treasury_10y = 0.042
        valuations["earnings_yield_spread"] = earnings_yield - treasury_10y

    # 4. FCF Yield
    fcf = data.get("free_cashflow")
    mcap = data.get("market_cap")
    if fcf and mcap and mcap > 0:
        fcf_yield = fcf / mcap
        valuations["fcf_yield"] = fcf_yield

    # 5. DCF-lite: 2-stage model
    #    Stage 1 (5 yrs): grow FCF at estimated rate
    #    Stage 2 (terminal): grow at 3% forever
    #    Discount everything at 10%
    if fcf and fcf > 0 and mcap and mcap > 0:
        shares = mcap / price if price > 0 else 1
        fcf_per_share = fcf / shares

        # Estimate near-term growth from earnings/revenue growth
        eg = data.get("earnings_growth")
        rg = data.get("revenue_growth")
        growth = 0.05  # default 5%
        if eg is not None and rg is not None:
            growth = min(max((eg + rg) / 2, 0.0), 0.20)  # cap at 20%
        elif eg is not None:
            growth = min(max(eg, 0.0), 0.20)
        elif rg is not None:
            growth = min(max(rg, 0.0), 0.20)

        discount_rate = 0.10  # 10% required return
        terminal_growth = 0.03  # long-term GDP growth
        projection_years = 5

        # Stage 1: project FCF for 5 years
        dcf_value = 0.0
        projected_fcf = fcf_per_share
        for yr in range(1, projection_years + 1):
            projected_fcf *= 1 + growth
            dcf_value += projected_fcf / (1 + discount_rate) ** yr

        # Stage 2: terminal value using Gordon Growth at conservative rate
        terminal_fcf = projected_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        dcf_value += terminal_value / (1 + discount_rate) ** projection_years

        # Apply margin of safety
        dcf_value_safe = dcf_value * 0.75  # 25% margin of safety

        # Sanity check: cap DCF upside at 200% (3x current price)
        dcf_upside = (dcf_value - price) / price
        if dcf_upside > 2.0:
            # DCF is unrealistically high — likely bad inputs or extreme growth
            # Still record but flag it
            valuations["dcf_fair_value"] = dcf_value
            valuations["dcf_safe_value"] = dcf_value_safe
            valuations["dcf_upside"] = dcf_upside
            valuations["dcf_growth_used"] = growth
            valuations["dcf_unreliable"] = True
        else:
            valuations["dcf_fair_value"] = dcf_value
            valuations["dcf_safe_value"] = dcf_value_safe
            valuations["dcf_upside"] = dcf_upside
            valuations["dcf_growth_used"] = growth
            valuations["dcf_unreliable"] = False

    # 6. Relative P/E valuation
    fwd_pe = data.get("forward_pe")
    fwd_eps = data.get("forward_eps")
    if fwd_pe and fwd_eps and fwd_eps > 0:
        # Fair PE by industry group (more granular than sector)
        industry = data.get("industry", "")
        sector = data.get("sector", "Unknown")

        # Industry-specific fair PEs
        industry_pe = {
            # Tech - Software (high margins, recurring revenue)
            "Software - Application": 25,
            "Software - Infrastructure": 28,
            "Information Technology Services": 20,
            # Tech - Hardware/Semi
            "Semiconductors": 20,
            "Semiconductor Equipment & Materials": 18,
            "Consumer Electronics": 22,
            "Computer Hardware": 18,
            "Electronic Components": 16,
            "Scientific & Technical Instruments": 22,
            # Healthcare
            "Drug Manufacturers - General": 18,
            "Drug Manufacturers - Specialty & Generic": 12,
            "Medical Devices": 22,
            "Diagnostics & Research": 22,
            "Biotechnology": 15,
            "Healthcare Plans": 12,  # Low-margin insurers
            "Medical Care Facilities": 15,
            "Medical Distribution": 14,
            # Finance
            "Banks - Diversified": 11,
            "Banks - Regional": 10,
            "Capital Markets": 14,
            "Credit Services": 12,
            "Insurance - Diversified": 11,
            "Insurance - Life": 10,
            "Insurance - Property & Casualty": 12,
            "Asset Management": 16,
            "Financial Data & Stock Exchanges": 22,
            "Insurance Brokers": 18,
            # Consumer
            "Discount Stores": 22,
            "Home Improvement Retail": 20,
            "Specialty Retail": 18,
            "Restaurants": 22,
            "Apparel Retail": 18,
            "Internet Retail": 25,
            "Household & Personal Products": 22,
            "Beverages - Non-Alcoholic": 22,
            "Packaged Foods": 18,
            "Tobacco": 10,
            # Industrial
            "Aerospace & Defense": 18,
            "Farm & Heavy Construction Machinery": 16,
            "Diversified Industrials": 18,
            "Railroads": 18,
            "Integrated Freight & Logistics": 16,
            "Waste Management": 22,
            "Specialty Industrial Machinery": 18,
            "Electrical Equipment & Parts": 20,
            # Energy
            "Oil & Gas Integrated": 11,
            "Oil & Gas E&P": 10,
            "Oil & Gas Refining & Marketing": 8,
            "Oil & Gas Equipment & Services": 12,
            # Utilities
            "Utilities - Regulated Electric": 17,
            "Utilities - Diversified": 16,
            "Utilities - Renewable": 20,
            # Telecom
            "Telecom Services": 10,
            # Materials
            "Specialty Chemicals": 18,
            "Gold": 15,
            "Copper": 12,
            "Industrial Metals & Mining": 10,
            "Chemicals": 15,
            # Other
            "Conglomerates": 15,
            "Entertainment": 18,
            "Insurance - Reinsurance": 10,
        }

        # Use industry PE if available, otherwise fall back to sector PE
        sector_pe = {
            "Technology": 22,
            "Healthcare": 17,
            "Financial Services": 12,
            "Consumer Cyclical": 18,
            "Consumer Defensive": 20,
            "Industrials": 17,
            "Energy": 10,
            "Utilities": 16,
            "Basic Materials": 14,
            "Communication Services": 16,
            "Real Estate": 18,
        }
        fair_pe = industry_pe.get(industry, sector_pe.get(sector, 16))

        relative_fair_value = fwd_eps * fair_pe
        relative_upside = (relative_fair_value - price) / price
        # Cap relative PE upside at 100% — beyond that the PE assumption is probably wrong
        relative_upside = min(relative_upside, 1.0)
        valuations["relative_pe_fair_value"] = relative_fair_value
        valuations["relative_pe_upside"] = relative_upside
        valuations["sector_avg_pe"] = fair_pe
        valuations["industry_used"] = industry

    # 7. Composite fair value (weighted average of all methods)
    fair_values = []
    weights = []

    if "analyst_target" in valuations:
        fair_values.append(valuations["analyst_target"])
        weights.append(3.0)  # Analysts have information we don't

    if "dcf_fair_value" in valuations and not valuations.get("dcf_unreliable", False):
        fair_values.append(valuations["dcf_fair_value"])
        weights.append(2.0)

    if "relative_pe_fair_value" in valuations:
        fair_values.append(valuations["relative_pe_fair_value"])
        weights.append(1.5)

    if "graham_number" in valuations:
        fair_values.append(valuations["graham_number"])
        weights.append(1.0)

    if fair_values:
        composite = np.average(fair_values, weights=weights)
        raw_upside = (composite - price) / price
        # Cap composite upside at 100% — anything beyond that is likely noise
        capped_upside = min(raw_upside, 1.0)
        valuations["composite_fair_value"] = composite
        valuations["composite_upside_raw"] = raw_upside
        valuations["composite_upside"] = capped_upside

    return valuations


# --------------------------------------------------------------------------
# Quality scoring
# --------------------------------------------------------------------------


def compute_quality_score(data: dict) -> dict:
    """Score stock quality (0-100). Higher = better quality company."""
    scores = {}
    total = 0
    max_score = 0

    # 1. Profitability (max 25 points)
    max_score += 25
    roe = data.get("roe")
    if roe is not None:
        if roe > 0.20:
            scores["roe"] = 10
        elif roe > 0.15:
            scores["roe"] = 8
        elif roe > 0.10:
            scores["roe"] = 5
        elif roe > 0:
            scores["roe"] = 2
        else:
            scores["roe"] = 0
        total += scores["roe"]

    pm = data.get("profit_margin")
    if pm is not None:
        if pm > 0.20:
            scores["profit_margin"] = 8
        elif pm > 0.10:
            scores["profit_margin"] = 6
        elif pm > 0.05:
            scores["profit_margin"] = 3
        elif pm > 0:
            scores["profit_margin"] = 1
        else:
            scores["profit_margin"] = 0
        total += scores["profit_margin"]

    om = data.get("operating_margin")
    if om is not None:
        if om > 0.25:
            scores["operating_margin"] = 7
        elif om > 0.15:
            scores["operating_margin"] = 5
        elif om > 0.05:
            scores["operating_margin"] = 3
        else:
            scores["operating_margin"] = 0
        total += scores["operating_margin"]

    # 2. Growth (max 20 points)
    max_score += 20
    eg = data.get("earnings_growth")
    if eg is not None:
        if eg > 0.20:
            scores["earnings_growth"] = 10
        elif eg > 0.10:
            scores["earnings_growth"] = 7
        elif eg > 0.05:
            scores["earnings_growth"] = 4
        elif eg > 0:
            scores["earnings_growth"] = 2
        else:
            scores["earnings_growth"] = 0
        total += scores["earnings_growth"]

    rg = data.get("revenue_growth")
    if rg is not None:
        if rg > 0.15:
            scores["revenue_growth"] = 10
        elif rg > 0.08:
            scores["revenue_growth"] = 7
        elif rg > 0.03:
            scores["revenue_growth"] = 4
        elif rg > 0:
            scores["revenue_growth"] = 2
        else:
            scores["revenue_growth"] = 0
        total += scores["revenue_growth"]

    # 3. Financial health (max 20 points)
    max_score += 20
    cr = data.get("current_ratio")
    if cr is not None:
        if cr > 2.0:
            scores["current_ratio"] = 7
        elif cr > 1.5:
            scores["current_ratio"] = 5
        elif cr > 1.0:
            scores["current_ratio"] = 3
        else:
            scores["current_ratio"] = 0
        total += scores["current_ratio"]

    dte = data.get("debt_to_equity")
    if dte is not None:
        if dte < 30:
            scores["debt_to_equity"] = 7
        elif dte < 60:
            scores["debt_to_equity"] = 5
        elif dte < 100:
            scores["debt_to_equity"] = 3
        elif dte < 200:
            scores["debt_to_equity"] = 1
        else:
            scores["debt_to_equity"] = 0
        total += scores["debt_to_equity"]

    fcf = data.get("free_cashflow")
    if fcf is not None:
        if fcf > 0:
            scores["positive_fcf"] = 6
        else:
            scores["positive_fcf"] = 0
        total += scores["positive_fcf"]

    # 4. Dividend (max 10 points)
    max_score += 10
    dy = data.get("dividend_yield")
    pr = data.get("payout_ratio")
    if dy is not None and dy > 0:
        if dy > 0.03:
            scores["dividend"] = 5
        elif dy > 0.015:
            scores["dividend"] = 3
        else:
            scores["dividend"] = 1
        # Sustainable payout?
        if pr is not None and 0 < pr < 0.7:
            scores["dividend"] += 3
        elif pr is not None and pr < 0.9:
            scores["dividend"] += 1
        total += scores["dividend"]

    # 5. Analyst consensus (max 10 points)
    max_score += 10
    rec = data.get("recommendation_score")
    n_an = data.get("n_analysts", 0)
    if rec is not None and n_an and n_an >= 5:
        if rec <= 1.5:
            scores["analyst"] = 10  # Strong buy
        elif rec <= 2.0:
            scores["analyst"] = 8  # Buy
        elif rec <= 2.5:
            scores["analyst"] = 5  # Moderate buy
        elif rec <= 3.0:
            scores["analyst"] = 3  # Hold
        else:
            scores["analyst"] = 0  # Sell
        total += scores["analyst"]

    # 6. Momentum context (max 15 points) — not for value, but for timing
    max_score += 15
    pct_high = data.get("pct_from_52w_high")
    if pct_high is not None:
        # More points for being far from 52-week high (potential bargain)
        if pct_high < -0.30:
            scores["52w_position"] = 10  # 30%+ below high — deep value territory
        elif pct_high < -0.20:
            scores["52w_position"] = 8
        elif pct_high < -0.10:
            scores["52w_position"] = 5
        elif pct_high < -0.05:
            scores["52w_position"] = 2
        else:
            scores["52w_position"] = 0  # Near highs — not a bargain
        total += scores["52w_position"]

    sma200 = data.get("two_hundred_day_avg")
    price = data.get("price")
    if sma200 and price and sma200 > 0:
        ratio = price / sma200
        if ratio < 0.85:
            scores["vs_sma200"] = 5  # Well below 200-day — oversold
        elif ratio < 0.95:
            scores["vs_sma200"] = 3
        elif ratio > 1.10:
            scores["vs_sma200"] = 0  # Extended above 200-day
        else:
            scores["vs_sma200"] = 1
        total += scores["vs_sma200"]

    quality_pct = total / max_score * 100 if max_score > 0 else 0

    return {
        "quality_score": quality_pct,
        "quality_points": total,
        "quality_max": max_score,
        "quality_details": scores,
    }


# --------------------------------------------------------------------------
# Value Score: combines valuation + quality
# --------------------------------------------------------------------------


def compute_value_score(data: dict, valuations: dict, quality: dict) -> dict:
    """
    Compute a composite VALUE SCORE (0-100).
    Combines: undervaluation + company quality + safety.
    Higher = better buy opportunity.
    """
    score = 0.0

    # Undervaluation component (0-40 points)
    composite_upside = valuations.get("composite_upside")
    if composite_upside is not None:
        # More upside = higher score
        if composite_upside > 0.50:
            score += 40
        elif composite_upside > 0.30:
            score += 35
        elif composite_upside > 0.20:
            score += 30
        elif composite_upside > 0.10:
            score += 22
        elif composite_upside > 0.05:
            score += 15
        elif composite_upside > 0.0:
            score += 8
        else:
            score += 0  # Overvalued

    # Quality component (0-30 points)
    qscore = quality.get("quality_score", 0)
    score += qscore * 0.30  # 0-30 points

    # Safety component (0-20 points)
    # S&P 500 stocks are inherently safer
    beta = data.get("beta", 1.0)
    if beta is not None:
        if beta < 0.8:
            score += 10  # Low beta = defensive
        elif beta < 1.0:
            score += 7
        elif beta < 1.2:
            score += 4
        else:
            score += 1  # High beta

    dte = data.get("debt_to_equity")
    if dte is not None:
        if dte < 50:
            score += 10
        elif dte < 100:
            score += 6
        elif dte < 200:
            score += 3
        else:
            score += 0

    # FCF yield bonus (0-10 points)
    fcf_yield = valuations.get("fcf_yield")
    if fcf_yield is not None:
        if fcf_yield > 0.08:
            score += 10
        elif fcf_yield > 0.05:
            score += 7
        elif fcf_yield > 0.03:
            score += 4
        elif fcf_yield > 0.01:
            score += 2

    return {
        "value_score": min(score, 100),
        "composite_upside": composite_upside,
    }


# --------------------------------------------------------------------------
# News sentiment check
# --------------------------------------------------------------------------


def check_news_sentiment(symbol: str) -> dict:
    """Quick news sentiment check using yfinance news."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return {"news_count": 0, "sentiment": "neutral", "devastating": False}

        # Check for devastating keywords
        devastating_keywords = [
            "fraud",
            "bankrupt",
            "sec charges",
            "criminal",
            "lawsuit",
            "recall",
            "scandal",
            "investigation",
            "collapse",
            "default",
            "delisted",
            "class action",
            "accounting irregularity",
        ]

        n_articles = len(news)
        n_negative = 0
        devastating = False

        for article in news[:10]:  # Check last 10 articles
            title = article.get("title", "").lower()
            for kw in devastating_keywords:
                if kw in title:
                    devastating = True
                    break

        return {
            "news_count": n_articles,
            "devastating": devastating,
        }
    except Exception:
        return {"news_count": 0, "devastating": False}


# --------------------------------------------------------------------------
# Full screening
# --------------------------------------------------------------------------


def screen_sp500(
    min_value_score: float = 40.0,
    min_upside: float = 0.05,
    max_results: int = 20,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Screen S&P 500 for undervalued stocks."""
    print("\n" + "=" * 80)
    print("  VALUE INVESTING SCREENER — S&P 500")
    print("  Finding undervalued stocks with strong fundamentals")
    print("=" * 80)

    if symbols is None:
        symbols = get_sp500_symbols()

    print(f"  Scanning {len(symbols)} symbols...\n")

    results = []
    for i, sym in enumerate(symbols):
        pct = (i + 1) / len(symbols) * 100
        print(f"\r  [{i + 1}/{len(symbols)}] {sym:>6} ({pct:.0f}%)", end="", flush=True)

        data = fetch_fundamentals(sym)
        if data is None:
            continue

        valuations = compute_valuations(data)
        quality = compute_quality_score(data)
        value = compute_value_score(data, valuations, quality)

        composite_upside = value.get("composite_upside")
        if composite_upside is None:
            continue

        results.append(
            {
                "symbol": sym,
                "price": data["price"],
                "sector": data["sector"],
                "value_score": value["value_score"],
                "composite_upside": composite_upside,
                "analyst_upside": valuations.get("analyst_upside"),
                "analyst_target": valuations.get("analyst_target"),
                "dcf_upside": valuations.get("dcf_upside"),
                "graham_upside": valuations.get("graham_upside"),
                "quality_score": quality["quality_score"],
                "trailing_pe": data.get("trailing_pe"),
                "forward_pe": data.get("forward_pe"),
                "fcf_yield": valuations.get("fcf_yield"),
                "earnings_yield": valuations.get("earnings_yield"),
                "roe": data.get("roe"),
                "profit_margin": data.get("profit_margin"),
                "debt_to_equity": data.get("debt_to_equity"),
                "dividend_yield": data.get("dividend_yield"),
                "pct_from_52w_high": data.get("pct_from_52w_high"),
                "beta": data.get("beta"),
                "n_analysts": data.get("n_analysts"),
                "recommendation": data.get("recommendation"),
                "market_cap": data.get("market_cap"),
                # Store raw data for deep analysis
                "_data": data,
                "_valuations": valuations,
                "_quality": quality,
            }
        )

        # Rate limit
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    print(f"\n\n  Analyzed {len(results)} stocks successfully")

    if not results:
        print("  No results!")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Filter
    df_filtered = df[
        (df["value_score"] >= min_value_score) & (df["composite_upside"] >= min_upside)
    ].copy()

    df_filtered = df_filtered.sort_values("value_score", ascending=False)
    df_filtered = df_filtered.head(max_results)

    return df, df_filtered


def print_screen_results(df_all: pd.DataFrame, df_top: pd.DataFrame):
    """Print screening results."""
    print(f"\n{'=' * 120}")
    print(f"  TOP VALUE PICKS (sorted by Value Score)")
    print(f"{'=' * 120}")

    if len(df_top) == 0:
        print("  No stocks meet the criteria.")
        return

    print(
        f"  {'#':>3} {'Symbol':>7} {'Price':>9} {'Sector':>15} "
        f"{'ValScore':>9} {'Upside':>8} {'Analyst':>8} {'DCF':>8} "
        f"{'FwdPE':>7} {'FCF%':>6} {'ROE':>6} {'Quality':>8} {'Rec':>6}"
    )
    print(
        f"  {'—' * 3} {'—' * 7} {'—' * 9} {'—' * 15} "
        f"{'—' * 9} {'—' * 8} {'—' * 8} {'—' * 8} "
        f"{'—' * 7} {'—' * 6} {'—' * 6} {'—' * 8} {'—' * 6}"
    )

    for rank, (_, row) in enumerate(df_top.iterrows(), 1):
        sect = (row["sector"] or "")[:15]
        fpe = f"{row['forward_pe']:.1f}" if pd.notna(row.get("forward_pe")) else "N/A"
        fcf = f"{row['fcf_yield']:.1%}" if pd.notna(row.get("fcf_yield")) else "N/A"
        roe = f"{row['roe']:.0%}" if pd.notna(row.get("roe")) else "N/A"
        analyst = (
            f"{row['analyst_upside']:+.0%}"
            if pd.notna(row.get("analyst_upside"))
            else "N/A"
        )
        dcf = f"{row['dcf_upside']:+.0%}" if pd.notna(row.get("dcf_upside")) else "N/A"
        rec = row.get("recommendation", "N/A") or "N/A"
        rec = rec[:6]

        print(
            f"  {rank:>3} {row['symbol']:>7} ${row['price']:>7.2f} {sect:>15} "
            f"{row['value_score']:>8.1f} {row['composite_upside']:>+7.1%} "
            f"{analyst:>8} {dcf:>8} "
            f"{fpe:>7} {fcf:>6} {roe:>6} {row['quality_score']:>7.0f}% {rec:>6}"
        )

    print(f"{'=' * 120}")

    # Summary stats
    avg_upside = df_top["composite_upside"].mean()
    avg_quality = df_top["quality_score"].mean()
    print(f"\n  Avg composite upside: {avg_upside:+.1%}")
    print(f"  Avg quality score: {avg_quality:.0f}%")
    print(
        f"  Stocks with analyst 'buy': "
        f"{(df_top['recommendation'].isin(['buy', 'strong_buy'])).sum()}/{len(df_top)}"
    )


def deep_analysis(symbols: list[str]):
    """Deep analysis of specific stocks."""
    for sym in symbols:
        print(f"\n{'=' * 70}")
        print(f"  DEEP ANALYSIS: {sym}")
        print(f"{'=' * 70}")

        data = fetch_fundamentals(sym)
        if data is None:
            print(f"  Could not fetch data for {sym}")
            continue

        valuations = compute_valuations(data)
        quality = compute_quality_score(data)
        value = compute_value_score(data, valuations, quality)
        news = check_news_sentiment(sym)

        # Basic info
        print(f"\n  Price: ${data['price']:.2f}")
        print(f"  Sector: {data['sector']} | Industry: {data['industry']}")
        print(f"  Market Cap: ${data['market_cap'] / 1e9:.1f}B")
        print(f"  Beta: {data.get('beta', 'N/A')}")

        # Valuation
        print(f"\n  --- VALUATION ---")
        print(f"  Trailing P/E: {data.get('trailing_pe', 'N/A')}")
        print(f"  Forward P/E:  {data.get('forward_pe', 'N/A')}")
        print(f"  P/B:          {data.get('price_to_book', 'N/A')}")
        print(f"  EV/EBITDA:    {data.get('ev_to_ebitda', 'N/A')}")

        if "analyst_target" in valuations:
            print(
                f"\n  Analyst Target: ${valuations['analyst_target']:.2f} "
                f"(upside: {valuations['analyst_upside']:+.1%}, "
                f"{data.get('n_analysts', '?')} analysts)"
            )
        if "graham_number" in valuations:
            print(
                f"  Graham Number: ${valuations['graham_number']:.2f} "
                f"(upside: {valuations['graham_upside']:+.1%})"
            )
        if "dcf_fair_value" in valuations:
            print(
                f"  DCF Fair Value: ${valuations['dcf_fair_value']:.2f} "
                f"(upside: {valuations['dcf_upside']:+.1%}, "
                f"growth={valuations.get('dcf_growth_used', 0):.1%})"
            )
        if "relative_pe_fair_value" in valuations:
            print(
                f"  Relative P/E Value: ${valuations['relative_pe_fair_value']:.2f} "
                f"(upside: {valuations['relative_pe_upside']:+.1%})"
            )
        if "composite_fair_value" in valuations:
            print(
                f"  >>> COMPOSITE FAIR VALUE: ${valuations['composite_fair_value']:.2f} "
                f"(upside: {valuations['composite_upside']:+.1%}) <<<"
            )

        # Quality
        print(f"\n  --- QUALITY (Score: {quality['quality_score']:.0f}%) ---")
        print(f"  ROE:             {data.get('roe', 'N/A')}")
        print(f"  Profit Margin:   {data.get('profit_margin', 'N/A')}")
        print(f"  Revenue Growth:  {data.get('revenue_growth', 'N/A')}")
        print(f"  Earnings Growth: {data.get('earnings_growth', 'N/A')}")
        print(f"  FCF Yield:       {valuations.get('fcf_yield', 'N/A')}")
        print(f"  Debt/Equity:     {data.get('debt_to_equity', 'N/A')}")
        print(f"  Dividend Yield:  {data.get('dividend_yield', 'N/A')}")

        # Position
        print(f"\n  --- TECHNICAL POSITION ---")
        print(f"  vs 52W High:  {data.get('pct_from_52w_high', 0):+.1%}")
        print(f"  vs 52W Low:   {data.get('pct_from_52w_low', 0):+.1%}")
        sma200 = data.get("two_hundred_day_avg")
        if sma200 and sma200 > 0:
            print(f"  vs 200-day:   {(data['price'] / sma200 - 1):+.1%}")

        # News
        print(f"\n  --- NEWS ---")
        print(f"  Recent articles: {news.get('news_count', 0)}")
        if news.get("devastating"):
            print(f"  WARNING: DEVASTATING NEWS DETECTED!")
        else:
            print(f"  No devastating news found")

        # Verdict
        print(f"\n  {'=' * 50}")
        vs = value["value_score"]
        cu = value.get("composite_upside", 0)
        print(f"  VALUE SCORE: {vs:.1f}/100")
        if vs >= 60 and cu and cu > 0.15:
            print(f"  VERDICT: STRONG BUY — significantly undervalued")
        elif vs >= 45 and cu and cu > 0.05:
            print(f"  VERDICT: BUY — moderately undervalued")
        elif cu and cu > 0:
            print(f"  VERDICT: HOLD — near fair value")
        else:
            print(f"  VERDICT: AVOID — overvalued or poor quality")
        print(f"  {'=' * 50}")


def generate_portfolio(df_top: pd.DataFrame, capital: float = 100_000) -> pd.DataFrame:
    """Generate a conviction portfolio from top picks."""
    print(f"\n{'=' * 80}")
    print(f"  CONVICTION PORTFOLIO (${capital:,.0f})")
    print(f"  Rule: Buy and hold until fair value. NEVER sell at a loss.")
    print(f"{'=' * 80}")

    if len(df_top) == 0:
        print("  No positions to take.")
        return pd.DataFrame()

    # Take top 5-10 by value score
    picks = df_top.head(10).copy()

    # Weight by value score (higher score = larger position)
    picks["raw_weight"] = picks["value_score"] / picks["value_score"].sum()
    # But cap any single position at 20%
    picks["weight"] = picks["raw_weight"].clip(upper=0.20)
    picks["weight"] = picks["weight"] / picks["weight"].sum()  # Re-normalize

    picks["allocation"] = picks["weight"] * capital
    picks["shares"] = (picks["allocation"] / picks["price"]).astype(int)
    picks["actual_allocation"] = picks["shares"] * picks["price"]

    # Target exit prices
    picks["target_price"] = picks.apply(
        lambda r: r["price"] * (1 + r["composite_upside"])
        if r["composite_upside"]
        else r["price"],
        axis=1,
    )

    print(
        f"\n  {'#':>3} {'Symbol':>7} {'Price':>9} {'Weight':>8} {'Shares':>7} "
        f"{'Alloc':>10} {'Target':>9} {'Upside':>8} {'Sector':>15}"
    )
    print(
        f"  {'—' * 3} {'—' * 7} {'—' * 9} {'—' * 8} {'—' * 7} "
        f"{'—' * 10} {'—' * 9} {'—' * 8} {'—' * 15}"
    )

    total_alloc = 0
    for rank, (_, row) in enumerate(picks.iterrows(), 1):
        sect = (row["sector"] or "")[:15]
        total_alloc += row["actual_allocation"]
        print(
            f"  {rank:>3} {row['symbol']:>7} ${row['price']:>7.2f} "
            f"{row['weight']:>7.1%} {row['shares']:>7,} "
            f"${row['actual_allocation']:>9,.0f} "
            f"${row['target_price']:>8.2f} "
            f"{row['composite_upside']:>+7.1%} {sect:>15}"
        )

    print(f"\n  Total allocated: ${total_alloc:,.0f} / ${capital:,.0f}")
    cash = capital - total_alloc
    print(f"  Cash reserve:    ${cash:,.0f} ({cash / capital:.1%})")

    # Expected return if all stocks hit fair value
    expected_return = (picks["actual_allocation"] * picks["composite_upside"]).sum()
    print(
        f"\n  Expected profit if all hit fair value: ${expected_return:,.0f} "
        f"({expected_return / capital:.1%})"
    )
    print(f"\n  RULES:")
    print(f"  - Hold each position until it reaches target price")
    print(f"  - NEVER sell at a loss (S&P 500 stocks recover)")
    print(f"  - Only sell at loss if: bankruptcy risk, fraud, or SEC action")
    print(f"  - Re-screen monthly to find new opportunities")
    print(f"{'=' * 80}")

    # Save portfolio
    save_data = picks[
        [
            "symbol",
            "price",
            "weight",
            "shares",
            "actual_allocation",
            "target_price",
            "composite_upside",
            "value_score",
            "sector",
        ]
    ].to_dict("records")
    portfolio = {
        "generated_at": datetime.now().isoformat(),
        "capital": capital,
        "positions": save_data,
        "rules": {
            "sell_at_loss": "NEVER for S&P 500 (unless bankruptcy/fraud/SEC)",
            "sell_target": "When stock reaches composite fair value",
            "rescreen_frequency": "Monthly",
        },
    }
    filepath = DATA_DIR / f"portfolio_{datetime.now().strftime('%Y%m%d')}.json"
    with open(filepath, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)
    print(f"\n  Portfolio saved to: {filepath}")

    return picks


def backtest_value_strategy():
    """
    Backtest a value strategy using price-based proxies.

    Since yfinance doesn't provide historical fundamentals, we use:
    - Trailing P/E proxy: (price / trailing EPS from current data)
    - Distance from rolling 52-week high (buy beaten-down stocks)
    - Distance from 200-day SMA (buy below SMA)

    Monthly rebalance, hold top 10 by composite value proxy signal.
    Compare vs equal-weight benchmark of all stocks.
    """
    print("\n" + "=" * 80)
    print("  VALUE STRATEGY BACKTEST")
    print("  Using price-based value proxies (P/E ratio, distance from highs)")
    print("=" * 80)

    symbols = get_sp500_symbols()

    # Fetch current fundamentals to get trailing EPS (assume stable over 2 years)
    print("\n  Phase 1: Fetching current fundamentals for EPS data...")
    eps_data = {}
    for i, sym in enumerate(symbols):
        print(f"\r  [{i + 1}/{len(symbols)}] {sym:>6}", end="", flush=True)
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            eps = info.get("trailingEps")
            if eps and eps > 0:
                eps_data[sym] = eps
        except Exception:
            pass
        if (i + 1) % 15 == 0:
            time.sleep(0.3)

    print(f"\n  Got EPS data for {len(eps_data)} / {len(symbols)} stocks")

    # Fetch 3 years of price data for all symbols
    print("\n  Phase 2: Fetching 3 years of price history...")
    valid_symbols = list(eps_data.keys())
    # Download in batches to avoid yfinance throttling
    all_prices = {}
    batch_size = 30
    for batch_start in range(0, len(valid_symbols), batch_size):
        batch = valid_symbols[batch_start : batch_start + batch_size]
        batch_str = " ".join(batch)
        print(
            f"\r  Downloading batch {batch_start // batch_size + 1}...",
            end="",
            flush=True,
        )
        try:
            df = yf.download(batch_str, period="3y", interval="1d", progress=False)
            if "Close" in df.columns:
                close = df["Close"]
                if isinstance(close, pd.Series):
                    # Single stock
                    all_prices[batch[0]] = close
                else:
                    for sym in batch:
                        if sym in close.columns:
                            series = close[sym].dropna()
                            if len(series) > 200:
                                all_prices[sym] = series
        except Exception as e:
            print(f"\n  Warning: batch download failed: {e}")
        time.sleep(0.5)

    print(f"\n  Got price history for {len(all_prices)} stocks")

    if len(all_prices) < 20:
        print("  ERROR: Not enough price data to backtest")
        return

    # Build a common date index
    common_dates = None
    for sym, prices in all_prices.items():
        if common_dates is None:
            common_dates = set(prices.index)
        else:
            common_dates &= set(prices.index)

    common_dates = sorted(common_dates)
    print(
        f"  Common trading dates: {len(common_dates)} ({common_dates[0].date()} to {common_dates[-1].date()})"
    )

    # Build price matrix
    price_matrix = pd.DataFrame(index=common_dates)
    for sym in all_prices:
        price_matrix[sym] = all_prices[sym].reindex(common_dates)

    price_matrix = price_matrix.dropna(axis=1, how="any")
    syms = list(price_matrix.columns)
    print(f"  Stocks with full history: {len(syms)}")

    # Monthly rebalance dates (first trading day of each month)
    price_matrix.index = pd.DatetimeIndex(price_matrix.index)
    months = price_matrix.resample("MS").first().index
    # Use actual trading dates closest to month start
    rebalance_dates = []
    for m in months:
        future = price_matrix.index[price_matrix.index >= m]
        if len(future) > 0:
            rebalance_dates.append(future[0])

    # Need at least 252 trading days of lookback for 52-week high
    # and at least 21 days of forward returns
    min_lookback = 252
    rebalance_dates = [
        d
        for d in rebalance_dates
        if (
            price_matrix.index.get_loc(d) >= min_lookback
            and price_matrix.index.get_loc(d) < len(price_matrix) - 21
        )
    ]

    print(f"  Rebalance dates: {len(rebalance_dates)} months")
    print(f"  ({rebalance_dates[0].date()} to {rebalance_dates[-1].date()})")

    # Compute value signals at each rebalance date and measure forward returns
    results = []

    for reb_date in rebalance_dates:
        idx = price_matrix.index.get_loc(reb_date)

        value_scores = {}
        for sym in syms:
            price = price_matrix.iloc[idx][sym]
            if pd.isna(price) or price <= 0:
                continue

            # 1. Trailing P/E (lower = cheaper)
            eps = eps_data.get(sym)
            pe = price / eps if eps and eps > 0 else None

            # 2. Distance from 52-week high (more negative = more beaten down)
            lookback = price_matrix[sym].iloc[max(0, idx - 252) : idx + 1]
            high_52w = lookback.max()
            pct_from_high = (price / high_52w - 1) if high_52w > 0 else 0

            # 3. Distance from 200-day SMA
            sma200 = price_matrix[sym].iloc[max(0, idx - 200) : idx + 1].mean()
            pct_from_sma200 = (price / sma200 - 1) if sma200 > 0 else 0

            # Composite value signal (lower = more undervalued)
            # Normalize PE: lower PE → higher score
            score = 0
            if pe is not None and pe > 0 and pe < 100:
                # Score PE: 0-10 PE → 40pts, 10-15 → 30pts, 15-20 → 20pts, 20-30 → 10pts, >30 → 0
                if pe < 10:
                    score += 40
                elif pe < 15:
                    score += 30
                elif pe < 20:
                    score += 20
                elif pe < 30:
                    score += 10

            # Score distance from high: more beaten down → higher score
            if pct_from_high < -0.30:
                score += 35
            elif pct_from_high < -0.20:
                score += 28
            elif pct_from_high < -0.15:
                score += 20
            elif pct_from_high < -0.10:
                score += 12
            elif pct_from_high < -0.05:
                score += 5

            # Score distance from SMA200: below → higher score
            if pct_from_sma200 < -0.15:
                score += 25
            elif pct_from_sma200 < -0.05:
                score += 15
            elif pct_from_sma200 < 0:
                score += 8

            value_scores[sym] = {
                "score": score,
                "pe": pe,
                "pct_from_high": pct_from_high,
                "pct_from_sma200": pct_from_sma200,
                "price": price,
            }

        if len(value_scores) < 20:
            continue

        # Rank by value score
        ranked = sorted(value_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top10 = [s for s, _ in ranked[:10]]
        bottom10 = [s for s, _ in ranked[-10:]]
        all_syms = [s for s, _ in ranked]

        # Forward returns at 1M, 3M, 6M horizons
        for horizon_name, horizon_days in [("1M", 21), ("3M", 63), ("6M", 126)]:
            fwd_idx = min(idx + horizon_days, len(price_matrix) - 1)
            if fwd_idx <= idx:
                continue

            # Top 10 (value picks) returns
            top_returns = []
            for sym in top10:
                entry = price_matrix.iloc[idx][sym]
                exit_p = price_matrix.iloc[fwd_idx][sym]
                if pd.notna(entry) and pd.notna(exit_p) and entry > 0:
                    top_returns.append(exit_p / entry - 1)

            # Bottom 10 (expensive stocks) returns
            bottom_returns = []
            for sym in bottom10:
                entry = price_matrix.iloc[idx][sym]
                exit_p = price_matrix.iloc[fwd_idx][sym]
                if pd.notna(entry) and pd.notna(exit_p) and entry > 0:
                    bottom_returns.append(exit_p / entry - 1)

            # Equal-weight benchmark
            bench_returns = []
            for sym in all_syms:
                entry = price_matrix.iloc[idx][sym]
                exit_p = price_matrix.iloc[fwd_idx][sym]
                if pd.notna(entry) and pd.notna(exit_p) and entry > 0:
                    bench_returns.append(exit_p / entry - 1)

            if top_returns and bottom_returns and bench_returns:
                results.append(
                    {
                        "date": reb_date,
                        "horizon": horizon_name,
                        "top10_mean": np.mean(top_returns),
                        "bottom10_mean": np.mean(bottom_returns),
                        "benchmark_mean": np.mean(bench_returns),
                        "top10_median": np.median(top_returns),
                        "spread": np.mean(top_returns) - np.mean(bottom_returns),
                        "top10_win_rate": np.mean([r > 0 for r in top_returns]),
                        "n_top": len(top_returns),
                    }
                )

    if not results:
        print("  ERROR: No backtest results generated")
        return

    df_results = pd.DataFrame(results)

    # Print summary
    print(f"\n{'=' * 90}")
    print(f"  BACKTEST RESULTS: Value (Top 10) vs Expensive (Bottom 10) vs Benchmark")
    print(f"{'=' * 90}")

    for horizon in ["1M", "3M", "6M"]:
        subset = df_results[df_results["horizon"] == horizon]
        if len(subset) == 0:
            continue
        print(
            f"\n  --- {horizon} Forward Returns (across {len(subset)} monthly rebalances) ---"
        )
        print(
            f"  {'Metric':<25} {'Value Top-10':>14} {'Expensive Bot-10':>18} {'Benchmark':>12} {'Spread':>10}"
        )
        print(f"  {'—' * 25} {'—' * 14} {'—' * 18} {'—' * 12} {'—' * 10}")

        print(
            f"  {'Mean return':<25} {subset['top10_mean'].mean():>+13.2%} "
            f"{subset['bottom10_mean'].mean():>+17.2%} "
            f"{subset['benchmark_mean'].mean():>+11.2%} "
            f"{subset['spread'].mean():>+9.2%}"
        )

        print(
            f"  {'Median return':<25} {subset['top10_median'].mean():>+13.2%} "
            f"{'':>18} {'':>12} {'':>10}"
        )

        # Win rate: how often top10 beats benchmark
        beats_bench = (subset["top10_mean"] > subset["benchmark_mean"]).mean()
        beats_expensive = (subset["spread"] > 0).mean()
        print(f"  {'Beats benchmark':<25} {beats_bench:>13.0%}")
        print(f"  {'Beats expensive':<25} {beats_expensive:>13.0%}")

        # Average win rate of top-10 stocks
        print(f"  {'Avg stock win rate':<25} {subset['top10_win_rate'].mean():>13.0%}")

    # Year-by-year for 6M horizon
    subset_6m = df_results[df_results["horizon"] == "6M"].copy()
    if len(subset_6m) > 0:
        subset_6m["year"] = subset_6m["date"].dt.year
        print(f"\n  --- Year-by-Year: 6M Forward Returns ---")
        print(
            f"  {'Year':<8} {'Value':>10} {'Expensive':>12} {'Bench':>10} {'Spread':>10} {'Months':>8}"
        )
        print(f"  {'—' * 8} {'—' * 10} {'—' * 12} {'—' * 10} {'—' * 10} {'—' * 8}")
        for year, grp in subset_6m.groupby("year"):
            print(
                f"  {year:<8} {grp['top10_mean'].mean():>+9.2%} "
                f"{grp['bottom10_mean'].mean():>+11.2%} "
                f"{grp['benchmark_mean'].mean():>+9.2%} "
                f"{grp['spread'].mean():>+9.2%} "
                f"{len(grp):>8}"
            )

    # Iron-hands analysis: what if we never sell at a loss?
    print(f"\n  --- IRON HANDS ANALYSIS (Never sell at a loss) ---")
    # For each month's top-10 picks, check if they ever go positive within 12 months
    iron_results = []
    for reb_date in rebalance_dates:
        idx = price_matrix.index.get_loc(reb_date)
        # Get value scores again
        value_scores_local = {}
        for sym in syms:
            price = price_matrix.iloc[idx][sym]
            eps = eps_data.get(sym)
            pe = price / eps if eps and eps > 0 else None
            lookback = price_matrix[sym].iloc[max(0, idx - 252) : idx + 1]
            high_52w = lookback.max()
            pct_from_high = (price / high_52w - 1) if high_52w > 0 else 0
            sma200 = price_matrix[sym].iloc[max(0, idx - 200) : idx + 1].mean()
            pct_from_sma200 = (price / sma200 - 1) if sma200 > 0 else 0
            score = 0
            if pe is not None and pe > 0 and pe < 100:
                if pe < 10:
                    score += 40
                elif pe < 15:
                    score += 30
                elif pe < 20:
                    score += 20
                elif pe < 30:
                    score += 10
            if pct_from_high < -0.30:
                score += 35
            elif pct_from_high < -0.20:
                score += 28
            elif pct_from_high < -0.15:
                score += 20
            elif pct_from_high < -0.10:
                score += 12
            elif pct_from_high < -0.05:
                score += 5
            if pct_from_sma200 < -0.15:
                score += 25
            elif pct_from_sma200 < -0.05:
                score += 15
            elif pct_from_sma200 < 0:
                score += 8
            value_scores_local[sym] = score

        ranked_local = sorted(
            value_scores_local.items(), key=lambda x: x[1], reverse=True
        )
        top10_local = [s for s, _ in ranked_local[:10]]

        for sym in top10_local:
            entry = price_matrix.iloc[idx][sym]
            # Look forward up to 12 months (252 days)
            max_fwd = min(idx + 252, len(price_matrix))
            if max_fwd <= idx + 1:
                continue
            future_prices = price_matrix[sym].iloc[idx + 1 : max_fwd]
            max_price = future_prices.max()
            min_price = future_prices.min()
            final_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry

            # Iron hands: never sell at a loss, sell at max gain in 12 months
            max_return = (max_price / entry - 1) if entry > 0 else 0
            min_drawdown = (min_price / entry - 1) if entry > 0 else 0
            final_return = (final_price / entry - 1) if entry > 0 else 0

            # Did it ever go positive?
            ever_positive = max_return > 0.01  # at least 1% gain
            # How long to first profit?
            days_to_profit = None
            for j, p in enumerate(future_prices):
                if p > entry * 1.01:
                    days_to_profit = j + 1
                    break

            iron_results.append(
                {
                    "date": reb_date,
                    "symbol": sym,
                    "entry": entry,
                    "max_return": max_return,
                    "min_drawdown": min_drawdown,
                    "final_return": final_return,
                    "ever_positive": ever_positive,
                    "days_to_profit": days_to_profit,
                }
            )

    if iron_results:
        df_iron = pd.DataFrame(iron_results)
        pct_ever_positive = df_iron["ever_positive"].mean()
        avg_max_return = df_iron["max_return"].mean()
        avg_min_dd = df_iron["min_drawdown"].mean()
        avg_final = df_iron["final_return"].mean()
        avg_days_profit = df_iron["days_to_profit"].dropna().mean()
        median_days_profit = df_iron["days_to_profit"].dropna().median()

        print(f"  Positions analyzed: {len(df_iron)}")
        print(f"  Ever went positive (>1%):  {pct_ever_positive:.1%}")
        print(f"  Avg max return (12M):      {avg_max_return:+.1%}")
        print(f"  Avg worst drawdown (12M):  {avg_min_dd:+.1%}")
        print(f"  Avg final return (12M):    {avg_final:+.1%}")
        print(f"  Avg days to first profit:  {avg_days_profit:.0f}")
        print(f"  Median days to first profit: {median_days_profit:.0f}")

        # By year
        df_iron["year"] = df_iron["date"].dt.year
        print(
            f"\n  {'Year':<8} {'Positions':>10} {'Ever +':>8} {'Avg Max':>10} {'Avg Final':>10} {'AvgDaysProfit':>14}"
        )
        print(f"  {'—' * 8} {'—' * 10} {'—' * 8} {'—' * 10} {'—' * 10} {'—' * 14}")
        for year, grp in df_iron.groupby("year"):
            avg_dtp = grp["days_to_profit"].dropna().mean()
            dtp_str = f"{avg_dtp:.0f}" if pd.notna(avg_dtp) else "N/A"
            print(
                f"  {year:<8} {len(grp):>10} {grp['ever_positive'].mean():>7.0%} "
                f"{grp['max_return'].mean():>+9.1%} "
                f"{grp['final_return'].mean():>+9.1%} "
                f"{dtp_str:>14}"
            )

    print(f"\n{'=' * 90}")
    print(f"  Backtest complete.")
    print(f"{'=' * 90}")


# --------------------------------------------------------------------------
# Combined Strategy: Value + Meta-Ensemble
# --------------------------------------------------------------------------


def combined_strategy(capital: float = 100_000):
    """
    Combined strategy:
    - 70% capital → Value portfolio (long-term holds, monthly rebalance)
    - 30% capital → Meta-ensemble tactical overlay (daily/weekly signals)

    The value portfolio holds undervalued stocks until they reach fair value.
    The meta-ensemble overlay adds short-term alpha on top.
    """
    import pickle

    print("\n" + "=" * 90)
    print("  COMBINED STRATEGY: Value Investing + Meta-Ensemble")
    print("=" * 90)

    value_alloc = 0.70
    meta_alloc = 0.30
    value_capital = capital * value_alloc
    meta_capital = capital * meta_alloc

    print(f"\n  Total capital: ${capital:,.0f}")
    print(f"  Value allocation (70%): ${value_capital:,.0f}")
    print(f"  Meta-ensemble allocation (30%): ${meta_capital:,.0f}")

    # --- Value Portfolio ---
    print(f"\n  {'=' * 70}")
    print(f"  PART 1: VALUE PORTFOLIO (long-term conviction holds)")
    print(f"  {'=' * 70}")

    portfolio_file = DATA_DIR / "portfolio_20260214.json"
    if portfolio_file.exists():
        with open(portfolio_file) as f:
            saved_portfolio = json.load(f)
        value_positions = saved_portfolio.get("positions", [])
        print(f"\n  Loading saved portfolio from {portfolio_file.name}")
    else:
        print(f"\n  No saved portfolio found. Running screener...")
        df_all, df_top = screen_sp500(
            min_value_score=35.0, min_upside=0.05, max_results=10
        )
        print_screen_results(df_all, df_top)
        picks = generate_portfolio(df_top, capital=value_capital)
        value_positions = picks.to_dict("records") if len(picks) > 0 else []

    # Re-scale value positions to value_capital
    if value_positions:
        total_orig = sum(p.get("actual_allocation", 0) for p in value_positions)
        scale = value_capital / total_orig if total_orig > 0 else 1

        print(
            f"\n  {'#':>3} {'Symbol':>7} {'Price':>9} {'Shares':>7} {'Alloc':>10} {'Target':>9} {'Upside':>8}"
        )
        print(
            f"  {'—' * 3} {'—' * 7} {'—' * 9} {'—' * 7} {'—' * 10} {'—' * 9} {'—' * 8}"
        )
        total_value_alloc = 0
        for i, pos in enumerate(value_positions, 1):
            alloc = pos.get("actual_allocation", 0) * scale
            price = pos.get("price", 0)
            shares = int(alloc / price) if price > 0 else 0
            actual = shares * price
            target = pos.get("target_price", price)
            upside = pos.get("composite_upside", 0)
            total_value_alloc += actual
            print(
                f"  {i:>3} {pos['symbol']:>7} ${price:>7.2f} {shares:>7,} ${actual:>9,.0f} ${target:>8.2f} {upside:>+7.1%}"
            )
        print(
            f"\n  Value portfolio deployed: ${total_value_alloc:,.0f} / ${value_capital:,.0f}"
        )

    # --- Meta-Ensemble Overlay ---
    print(f"\n  {'=' * 70}")
    print(f"  PART 2: META-ENSEMBLE TACTICAL OVERLAY (short-term signals)")
    print(f"  {'=' * 70}")

    trades_file = PROJECT_ROOT / "data_store" / "live_trades" / "trades_2026-02-13.json"
    if trades_file.exists():
        with open(trades_file) as f:
            trades_data = json.load(f)
        predictions = trades_data.get("predictions", trades_data.get("trades", []))
        print(f"\n  Loading meta-ensemble predictions from {trades_file.name}")
        print(f"  Prediction date: {trades_data.get('prediction_date', 'unknown')}")
    else:
        print(f"\n  No meta-ensemble predictions found. Run live_trade_today.py first.")
        predictions = []

    # Filter to high-confidence predictions (>10% confidence)
    high_conf = [
        p for p in predictions if isinstance(p, dict) and p.get("confidence", 0) > 0.10
    ]

    if high_conf:
        # Allocate meta capital equally across high-confidence picks
        per_stock = meta_capital / len(high_conf) if high_conf else 0

        print(f"\n  High-confidence predictions ({len(high_conf)} stocks):")
        print(
            f"  {'#':>3} {'Symbol':>7} {'Direction':>10} {'Confidence':>11} {'Allocation':>11}"
        )
        print(f"  {'—' * 3} {'—' * 7} {'—' * 10} {'—' * 11} {'—' * 11}")

        total_meta_alloc = 0
        for i, pred in enumerate(high_conf, 1):
            sym = pred.get("symbol", "?")
            direction = pred.get("signal", pred.get("direction", "?"))
            conf = pred.get("confidence", 0)
            alloc = per_stock * min(conf * 5, 1.0)  # Scale by confidence
            total_meta_alloc += alloc
            print(f"  {i:>3} {sym:>7} {direction:>10} {conf:>10.1%} ${alloc:>10,.0f}")

        print(
            f"\n  Meta overlay deployed: ${total_meta_alloc:,.0f} / ${meta_capital:,.0f}"
        )
    else:
        print(f"\n  No high-confidence predictions available.")
        total_meta_alloc = 0

    # --- Cross-reference ---
    print(f"\n  {'=' * 70}")
    print(f"  CROSS-REFERENCE: Value picks × Meta-ensemble signals")
    print(f"  {'=' * 70}")

    value_syms = {p["symbol"] for p in value_positions} if value_positions else set()
    meta_syms = {p.get("symbol") for p in predictions if isinstance(p, dict)}

    overlap = value_syms & meta_syms
    if overlap:
        print(f"\n  OVERLAP (both value & meta recommend): {sorted(overlap)}")
        print(
            f"  → These are HIGHEST CONVICTION — both undervalued AND short-term bullish"
        )
    else:
        print(f"\n  No overlap between value and meta picks.")
        print(f"  → Strategies are COMPLEMENTARY (different stock selection)")

    # Check if meta-ensemble is bearish on any value picks
    meta_bearish = set()
    for pred in predictions:
        if isinstance(pred, dict):
            sym = pred.get("symbol", "?")
            conf = pred.get("confidence", 0)
            signal = pred.get("signal", pred.get("direction", ""))
            if sym in value_syms and signal == "SHORT":
                meta_bearish.add(sym)

    if meta_bearish:
        print(
            f"\n  WARNING: Meta-ensemble is BEARISH on value picks: {sorted(meta_bearish)}"
        )
        print(
            f"  → Consider delaying entry on these until meta-signal turns neutral/bullish"
        )
    else:
        print(f"  Meta-ensemble has no SHORT signals on value picks (safe to enter)")

    # --- Summary ---
    print(f"\n  {'=' * 70}")
    print(f"  COMBINED STRATEGY SUMMARY")
    print(f"  {'=' * 70}")
    print(f"\n  Total capital:        ${capital:,.0f}")
    value_deployed = total_value_alloc if value_positions else 0
    total_deployed = value_deployed + total_meta_alloc
    print(
        f"  Value deployed:       ${value_deployed:,.0f} ({value_deployed / capital:.0%})"
    )
    print(
        f"  Meta overlay deployed: ${total_meta_alloc:,.0f} ({total_meta_alloc / capital:.0%})"
    )
    print(
        f"  Total deployed:       ${total_deployed:,.0f} ({total_deployed / capital:.0%})"
    )
    print(
        f"  Cash reserve:         ${capital - total_deployed:,.0f} ({(capital - total_deployed) / capital:.0%})"
    )
    print(f"\n  RULES:")
    print(f"  1. Value positions: HOLD until target price. NEVER sell at a loss.")
    print(f"  2. Meta positions: Hold 1-5 days. Cut at -2% stop loss.")
    print(f"  3. If meta confirms a value pick (overlap): DOUBLE position size.")
    print(f"  4. If meta is SHORT on a value pick: Delay entry, don't sell existing.")
    print(f"  5. Rebalance value monthly, meta daily/weekly.")
    print(f"  {'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Value Investing Screener & Portfolio Manager",
    )
    parser.add_argument(
        "--screen", action="store_true", help="Screen S&P 500 for undervalued stocks"
    )
    parser.add_argument(
        "--deep", nargs="+", default=None, help="Deep analysis on specific symbols"
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Generate conviction portfolio from screening",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=35.0,
        help="Minimum value score (default: 35)",
    )
    parser.add_argument(
        "--min-upside",
        type=float,
        default=0.05,
        help="Minimum composite upside (default: 5%%)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Portfolio capital (default: $100,000)",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Show top N results (default: 20)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None, help="Custom symbol list to screen"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Backtest value strategy using price-based proxies",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Run combined value + meta-ensemble strategy",
    )

    args = parser.parse_args()

    if args.deep:
        deep_analysis(args.deep)
        return

    if args.backtest:
        backtest_value_strategy()
        return

    if args.combined:
        combined_strategy(capital=args.capital)
        return

    if args.screen or args.portfolio:
        df_all, df_top = screen_sp500(
            min_value_score=args.min_score,
            min_upside=args.min_upside,
            max_results=args.top,
            symbols=args.symbols,
        )

        print_screen_results(df_all, df_top)

        if args.portfolio and len(df_top) > 0:
            # News check on top picks
            print(f"\n  Checking news sentiment on top picks...")
            bad_news = []
            for _, row in df_top.iterrows():
                news = check_news_sentiment(row["symbol"])
                if news.get("devastating"):
                    bad_news.append(row["symbol"])
                    print(
                        f"    WARNING: {row['symbol']} has devastating news — EXCLUDING"
                    )

            if bad_news:
                df_top = df_top[~df_top["symbol"].isin(bad_news)]

            generate_portfolio(df_top, capital=args.capital)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
