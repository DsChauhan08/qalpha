#!/usr/bin/env python3
"""
Meta-Ensemble Trading System
=============================
Builds a stacking model that combines:
  - 30+ technical indicators (RSI, MACD, Bollinger, ATR, Stochastic, ADX, OBV, momentum)
  - Strategy signals (Momentum, MeanReversion, TrendFollowing, Breakout, TSMomentum)
  - Regime detection features (HMM-based Bull/Bear/Sideways/Crisis)
  - Cross-sectional features (rank, relative strength)
  - LSTM sentiment signal (where available)

into a unified HistGradientBoosting classifier for next-day direction prediction.

Data pipeline:
  100+ S&P 500 symbols × 20 years daily data → 500K+ training samples

Training:
  Walk-forward (expanding window) validation — never train on future data.

Usage:
  python meta_ensemble.py --phase fetch      # Step 1: Download data
  python meta_ensemble.py --phase features   # Step 2: Compute features
  python meta_ensemble.py --phase train      # Step 3: Train meta-ensemble
  python meta_ensemble.py --phase all        # All steps
  python meta_ensemble.py --phase train --quick  # Quick test with 10 symbols
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure PYTHONPATH
sys.path.insert(0, "/home/regulus/Trade")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_store" / "meta_ensemble"
CACHE_DIR = DATA_DIR / "ohlcv_cache"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = PROJECT_DIR / "models" / "checkpoints" / "meta_ensemble"

for d in [DATA_DIR, CACHE_DIR, FEATURE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────
START_DATE = datetime(2005, 1, 1)
END_DATE = datetime(2026, 2, 14)

# Forward periods for target
FORWARD_PERIOD = 1  # Next-day direction (binary: up/down)

# Walk-forward settings
WF_TRAIN_MIN_DAYS = 1260  # 5 years minimum training
WF_TEST_DAYS = 252  # 1 year test window
WF_STEP_DAYS = 126  # 6 month step

# ── Feature column names ─────────────────────────────────────────────
TECHNICAL_FEATURES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_position",
    "atr",
    "atr_pct",
    "stoch_k",
    "stoch_d",
    "adx",
    "obv",
    "obv_sma",
    "vwap",
    "mom_12m",
    "mom_3m",
    "rsi_signal",
    "bb_signal",
    "macd_signal_norm",
]

STRATEGY_FEATURES = [
    "sig_momentum",
    "sig_momentum_conf",
    "sig_meanrev",
    "sig_meanrev_conf",
    "sig_trend",
    "sig_trend_conf",
    "sig_breakout",
    "sig_breakout_conf",
    "sig_tsmom",
    "sig_tsmom_conf",
]

REGIME_FEATURES = [
    "regime_volatility",
    "regime_volatility_zscore",
    "regime_momentum_20d",
    "regime_momentum_60d",
    "regime_drawdown",
    "regime_trend_strength",
    "regime_autocorr",
    "regime_skewness",
    "regime_kurtosis",
]

CROSS_SECTIONAL_FEATURES = [
    "xs_return_rank_21d",
    "xs_return_rank_63d",
    "xs_return_rank_252d",
    "xs_vol_rank_21d",
    "xs_relative_strength_63d",
]

PRICE_DERIVED_FEATURES = [
    "returns_1d",
    "returns_5d",
    "returns_21d",
    "log_returns_1d",
    "volatility_10d",
    "volatility_21d",
    "volatility_63d",
    "vol_ratio_10_63",
    "price_vs_sma10",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma_cross_10_50",
    "sma_cross_50_200",
    "high_low_range",
    "close_position_in_range",
    "gap",
    "volume_ratio_20d",
    "up_days_ratio_10d",
    "up_days_ratio_21d",
]

ALL_FEATURE_COLS = (
    TECHNICAL_FEATURES
    + STRATEGY_FEATURES
    + REGIME_FEATURES
    + CROSS_SECTIONAL_FEATURES
    + PRICE_DERIVED_FEATURES
)


# =====================================================================
# Phase 1: Data Fetching
# =====================================================================


def get_symbol_universe(n_symbols: int = 100) -> List[str]:
    """Get S&P 500 symbols, limited to n_symbols."""
    try:
        from quantum_alpha.data.collectors.market_data import DataCollector

        dc = DataCollector()
        symbols = dc.get_sp500_symbols()
        logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 list: {e}, using fallback")
        symbols = [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "NVDA",
            "TSLA",
            "BRK-B",
            "JPM",
            "V",
            "UNH",
            "XOM",
            "PG",
            "MA",
            "HD",
            "AVGO",
            "LLY",
            "COST",
            "ABBV",
            "WMT",
            "DIS",
            "KO",
            "PEP",
            "BAC",
            "ADBE",
            "CRM",
            "NFLX",
            "ORCL",
            "CSCO",
            "INTC",
            "QCOM",
            "TXN",
            "TMO",
            "LIN",
            "ACN",
            "MCD",
            "NKE",
            "UPS",
            "HON",
            "UNP",
            "CAT",
            "GE",
            "IBM",
            "AMD",
            "AMAT",
            "GS",
            "MS",
            "C",
            "BA",
            "RTX",
            "LMT",
            "GM",
            "F",
            "CVX",
            "COP",
            "SLB",
            "SPGI",
            "BKNG",
            "SBUX",
            "T",
            "VZ",
            "PFE",
            "MRK",
            "ABT",
            "CVS",
            "DHR",
            "LOW",
            "ISRG",
            "GILD",
            "MDT",
            "INTU",
            "NOW",
            "PYPL",
            "SNPS",
            "VRTX",
            "ADP",
            "BLK",
            "DE",
            "MO",
            "SO",
            "DUK",
            "NEE",
            "PLD",
            "AMT",
            "CCI",
            "SCHW",
            "USB",
            "AXP",
            "TGT",
            "CME",
            "CB",
            "ZTS",
            "FDX",
            "PNC",
            "APD",
            "EOG",
            "LRCX",
            "KLAC",
            "REGN",
            "ETN",
        ]

    # Also include SPY as the market benchmark
    if "SPY" not in symbols:
        symbols = ["SPY"] + symbols

    return symbols[:n_symbols]


def fetch_all_ohlcv(symbols: List[str], force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all symbols, with per-symbol pickle caching.

    Returns dict of symbol -> DataFrame.
    """
    from quantum_alpha.data.collectors.market_data import DataCollector

    dc = DataCollector(cache_dir=str(CACHE_DIR / ".yf_cache"))

    results = {}
    failed = []

    for i, symbol in enumerate(symbols):
        cache_file = CACHE_DIR / f"{symbol}.pkl"

        # Check cache
        if not force and cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                if len(df) > 100:
                    results[symbol] = df
                    if (i + 1) % 20 == 0:
                        logger.info(
                            f"  [{i + 1}/{len(symbols)}] {symbol}: {len(df)} bars (cached)"
                        )
                    continue
            except Exception:
                pass

        # Fetch from yfinance
        try:
            df = dc.fetch_ohlcv(
                symbol, START_DATE, END_DATE, interval="1d", use_cache=False
            )
            if len(df) < 252:  # Need at least 1 year
                logger.warning(f"  {symbol}: only {len(df)} bars, skipping")
                failed.append(symbol)
                continue

            # Normalize timezone
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Save to cache
            df.to_pickle(cache_file)
            results[symbol] = df

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{i + 1}/{len(symbols)}] {symbol}: {len(df)} bars (fetched)"
                )

            # Small delay to avoid rate limiting
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"  {symbol}: FAILED - {e}")
            failed.append(symbol)
            time.sleep(1.0)

    logger.info(f"Fetched {len(results)} symbols, {len(failed)} failed")
    if failed:
        logger.info(f"Failed: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    return results


# =====================================================================
# Phase 2: Feature Computation
# =====================================================================


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators using TechnicalFeatureGenerator."""
    from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator

    gen = TechnicalFeatureGenerator()
    result = gen.generate(df)
    return result


def compute_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signals from all single-asset strategies.
    Returns DataFrame with strategy signal columns added.
    """
    from quantum_alpha.strategy.signals import (
        MomentumStrategy,
        MeanReversionStrategy,
        TrendFollowingStrategy,
        BreakoutTrendStrategy,
        TimeSeriesMomentumStrategy,
    )

    result = df.copy()

    # Each strategy needs technical indicators already computed
    strategies = {
        "momentum": MomentumStrategy(),
        "meanrev": MeanReversionStrategy(),
        "trend": TrendFollowingStrategy(),
        "breakout": BreakoutTrendStrategy(),
        "tsmom": TimeSeriesMomentumStrategy(),
    }

    for name, strategy in strategies.items():
        try:
            sig_df = strategy.generate_signals(df)
            result[f"sig_{name}"] = sig_df["signal"].values
            result[f"sig_{name}_conf"] = sig_df.get(
                "signal_confidence", pd.Series(0.5, index=df.index)
            ).values
        except Exception as e:
            logger.debug(f"Strategy {name} failed: {e}")
            result[f"sig_{name}"] = 0.0
            result[f"sig_{name}_conf"] = 0.0

    return result


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime detection features (raw features, not HMM — HMM is too slow for 100+ symbols)."""
    result = df.copy()
    close = df["close"]
    returns = close.pct_change()

    # Volatility features
    result["regime_volatility"] = returns.rolling(20).std() * np.sqrt(252)
    vol_mean = result["regime_volatility"].rolling(252).mean()
    vol_std = result["regime_volatility"].rolling(252).std()
    result["regime_volatility_zscore"] = (result["regime_volatility"] - vol_mean) / (
        vol_std + 1e-6
    )

    # Momentum features
    result["regime_momentum_20d"] = close.pct_change(20)
    result["regime_momentum_60d"] = close.pct_change(60)

    # Drawdown
    rolling_max = close.rolling(252).max()
    result["regime_drawdown"] = (close - rolling_max) / (rolling_max + 1e-8)

    # Trend strength
    result["regime_trend_strength"] = result["regime_momentum_60d"].abs() / (
        result["regime_volatility"] + 1e-6
    )

    # Autocorrelation
    result["regime_autocorr"] = returns.rolling(60).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )

    # Distribution shape
    result["regime_skewness"] = returns.rolling(60).skew()
    result["regime_kurtosis"] = returns.rolling(60).kurt()

    return result


def compute_price_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional price-derived features beyond what TechnicalFeatureGenerator provides."""
    result = df.copy()
    close = df["close"]

    # Multi-horizon returns
    result["returns_1d"] = close.pct_change(1)
    result["returns_5d"] = close.pct_change(5)
    result["returns_21d"] = close.pct_change(21)
    result["log_returns_1d"] = np.log(close / close.shift(1))

    # Multi-horizon volatility
    returns = close.pct_change()
    result["volatility_10d"] = returns.rolling(10).std() * np.sqrt(252)
    result["volatility_21d"] = returns.rolling(21).std() * np.sqrt(252)
    result["volatility_63d"] = returns.rolling(63).std() * np.sqrt(252)
    result["vol_ratio_10_63"] = result["volatility_10d"] / (
        result["volatility_63d"] + 1e-8
    )

    # Price relative to SMAs
    for period in [10, 20, 50, 200]:
        sma = close.rolling(period).mean()
        result[f"price_vs_sma{period}"] = (close - sma) / (sma + 1e-8)

    # SMA crossovers
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    result["sma_cross_10_50"] = (sma10 - sma50) / (sma50 + 1e-8)
    result["sma_cross_50_200"] = (sma50 - sma200) / (sma200 + 1e-8)

    # Range features
    if "high" in df.columns and "low" in df.columns:
        result["high_low_range"] = (df["high"] - df["low"]) / (close + 1e-8)
        day_high = df["high"].rolling(20).max()
        day_low = df["low"].rolling(20).min()
        result["close_position_in_range"] = (close - day_low) / (
            day_high - day_low + 1e-8
        )
    else:
        result["high_low_range"] = 0.0
        result["close_position_in_range"] = 0.5

    # Gap (open vs previous close)
    if "open" in df.columns:
        result["gap"] = (df["open"] - close.shift(1)) / (close.shift(1) + 1e-8)
    else:
        result["gap"] = 0.0

    # Volume features
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20).mean()
        result["volume_ratio_20d"] = df["volume"] / (vol_ma + 1e-8)
    else:
        result["volume_ratio_20d"] = 1.0

    # Up-days ratio
    up = (returns > 0).astype(float)
    result["up_days_ratio_10d"] = up.rolling(10).mean()
    result["up_days_ratio_21d"] = up.rolling(21).mean()

    return result


def compute_features_single_symbol(
    symbol: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute ALL features for a single symbol.
    Returns a DataFrame with all feature columns + target.
    """
    # Need enough data for indicators
    if len(df) < 300:
        return pd.DataFrame()

    try:
        # Step 1: Technical indicators
        featured = compute_technical_features(df)

        # Step 2: Strategy signals (needs technical indicators)
        featured = compute_strategy_signals(featured)

        # Step 3: Regime features
        featured = compute_regime_features(featured)

        # Step 4: Price-derived features
        featured = compute_price_derived_features(featured)

        # Step 5: Target — next-day direction (1 = up, 0 = down)
        future_return = (
            featured["close"].pct_change(FORWARD_PERIOD).shift(-FORWARD_PERIOD)
        )
        featured["target"] = (future_return > 0).astype(int)

        # Also store the actual forward return for backtesting
        featured["forward_return"] = future_return

        # Add symbol identifier
        featured["symbol"] = symbol

        # Drop warmup period (first 252 days have NaN indicators)
        featured = featured.iloc[252:].copy()

        # Drop rows where target is NaN (last FORWARD_PERIOD rows)
        featured = featured.dropna(subset=["target"])

        # Select only the feature columns we want + target + metadata
        available_features = [c for c in ALL_FEATURE_COLS if c in featured.columns]
        keep_cols = available_features + ["target", "forward_return", "symbol"]

        result = featured[keep_cols].copy()

        # Replace inf with NaN, then fill NaN with 0
        result = result.replace([np.inf, -np.inf], np.nan)
        # Don't fill target NaN
        feature_cols = [
            c for c in result.columns if c not in ["target", "forward_return", "symbol"]
        ]
        result[feature_cols] = result[feature_cols].fillna(0.0)

        return result

    except Exception as e:
        logger.warning(f"Feature computation failed for {symbol}: {e}")
        return pd.DataFrame()


def compute_cross_sectional_features(
    all_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Compute cross-sectional (relative) features.
    These require data from ALL symbols at each point in time.

    We compute: return rank, volatility rank, relative strength.
    """
    # Build aligned return matrices
    close_dict = {}
    for sym, df in all_data.items():
        if "close" in df.columns:
            close_dict[sym] = df["close"]

    if len(close_dict) < 5:
        logger.warning("Not enough symbols for cross-sectional features")
        return all_data

    closes = pd.DataFrame(close_dict).dropna(how="all")

    # Returns at various horizons
    ret_21d = closes.pct_change(21)
    ret_63d = closes.pct_change(63)
    ret_252d = closes.pct_change(252)

    # Volatility
    daily_ret = closes.pct_change()
    vol_21d = daily_ret.rolling(21).std()

    # Rankings (0 to 1, higher = better return or lower vol)
    rank_21d = ret_21d.rank(axis=1, pct=True)
    rank_63d = ret_63d.rank(axis=1, pct=True)
    rank_252d = ret_252d.rank(axis=1, pct=True)
    vol_rank_21d = vol_21d.rank(
        axis=1, pct=True, ascending=False
    )  # Lower vol = higher rank

    # Relative strength: symbol return / mean return
    mean_ret_63d = ret_63d.mean(axis=1)
    rel_strength_63d = ret_63d.subtract(mean_ret_63d, axis=0)

    # Assign back to each symbol's DataFrame
    result = {}
    for sym, df in all_data.items():
        enhanced = df.copy()

        if sym in rank_21d.columns:
            aligned_idx = enhanced.index.intersection(rank_21d.index)
            enhanced.loc[aligned_idx, "xs_return_rank_21d"] = rank_21d.loc[
                aligned_idx, sym
            ]
            enhanced.loc[aligned_idx, "xs_return_rank_63d"] = rank_63d.loc[
                aligned_idx, sym
            ]
            enhanced.loc[aligned_idx, "xs_return_rank_252d"] = rank_252d.loc[
                aligned_idx, sym
            ]
            enhanced.loc[aligned_idx, "xs_vol_rank_21d"] = vol_rank_21d.loc[
                aligned_idx, sym
            ]
            enhanced.loc[aligned_idx, "xs_relative_strength_63d"] = (
                rel_strength_63d.loc[aligned_idx, sym]
            )

        # Fill NaN cross-sectional features with 0.5 (neutral rank)
        for col in CROSS_SECTIONAL_FEATURES:
            if col in enhanced.columns:
                enhanced[col] = enhanced[col].fillna(0.5)
            else:
                enhanced[col] = 0.5

        result[sym] = enhanced

    return result


def build_feature_matrix(
    ohlcv_data: Dict[str, pd.DataFrame],
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix from all symbols.

    Steps:
    1. Compute per-symbol features (technical, strategy, regime, price-derived)
    2. Compute cross-sectional features (requires all symbols)
    3. Concatenate into single DataFrame

    Returns:
        DataFrame with all features + target, indexed by date, with 'symbol' column.
    """
    logger.info(f"Computing features for {len(ohlcv_data)} symbols...")

    # Step 1: Per-symbol features + cross-sectional prep
    # First compute cross-sectional features on raw data
    logger.info("Computing cross-sectional features...")
    enhanced_data = compute_cross_sectional_features(ohlcv_data)

    # Step 2: Compute all features per symbol
    all_frames = []
    for i, (symbol, df) in enumerate(enhanced_data.items()):
        features = compute_features_single_symbol(symbol, df)
        if len(features) > 0:
            all_frames.append(features)

        if (i + 1) % 20 == 0:
            logger.info(
                f"  [{i + 1}/{len(enhanced_data)}] Features computed, {len(features)} samples from {symbol}"
            )

    if not all_frames:
        logger.error("No features computed!")
        return pd.DataFrame()

    # Step 3: Concatenate
    combined = pd.concat(all_frames, axis=0)
    combined = combined.sort_index()

    logger.info(
        f"Feature matrix: {len(combined)} samples, {len(combined.columns)} columns"
    )
    logger.info(f"  Date range: {combined.index.min()} to {combined.index.max()}")
    logger.info(f"  Symbols: {combined['symbol'].nunique()}")
    logger.info(
        f"  Target distribution: up={combined['target'].mean():.1%}, down={(1 - combined['target'].mean()):.1%}"
    )

    # Available features
    feature_cols = [c for c in ALL_FEATURE_COLS if c in combined.columns]
    logger.info(f"  Available features: {len(feature_cols)} / {len(ALL_FEATURE_COLS)}")

    if save:
        out_path = FEATURE_DIR / "feature_matrix.pkl"
        combined.to_pickle(out_path)
        logger.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return combined


# =====================================================================
# Phase 3: Meta-Ensemble Training
# =====================================================================


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns available in the DataFrame."""
    return [c for c in ALL_FEATURE_COLS if c in df.columns]


def walk_forward_train(
    df: pd.DataFrame,
    n_folds: int = None,
) -> Dict:
    """
    Walk-forward (expanding window) training and evaluation.

    Uses chronological splits:
    - Train on data up to time T
    - Test on data from T to T + test_window
    - Expand training window, repeat

    Returns dict with fold results and final model.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feature_cols = get_feature_columns(df)
    logger.info(f"Training with {len(feature_cols)} features on {len(df)} samples")

    # Sort by date
    df = df.sort_index()

    # Get unique dates
    dates = df.index.unique().sort_values()
    n_dates = len(dates)

    logger.info(f"Date range: {dates[0]} to {dates[-1]} ({n_dates} unique dates)")

    # Calculate fold boundaries
    min_train_idx = WF_TRAIN_MIN_DAYS
    if min_train_idx >= n_dates:
        logger.warning(
            f"Not enough dates ({n_dates}) for walk-forward with {WF_TRAIN_MIN_DAYS} min train days"
        )
        min_train_idx = n_dates // 2

    folds = []
    idx = min_train_idx
    while idx + WF_TEST_DAYS <= n_dates:
        train_end = dates[idx]
        test_start = dates[idx]
        test_end_idx = min(idx + WF_TEST_DAYS, n_dates - 1)
        test_end = dates[test_end_idx]
        folds.append((train_end, test_start, test_end))
        idx += WF_STEP_DAYS

    if not folds:
        # Fallback: single split at 80%
        split_idx = int(n_dates * 0.8)
        train_end = dates[split_idx]
        test_end = dates[-1]
        folds = [(train_end, train_end, test_end)]

    if n_folds is not None:
        folds = folds[:n_folds]

    logger.info(f"Walk-forward: {len(folds)} folds")

    # Training loop
    all_results = []
    all_predictions = []
    best_model = None
    best_auc = 0
    best_scaler = None

    for fold_i, (train_end, test_start, test_end) in enumerate(folds):
        logger.info(f"\n--- Fold {fold_i + 1}/{len(folds)} ---")
        logger.info(f"  Train: ... to {train_end.date()}")
        logger.info(f"  Test:  {test_start.date()} to {test_end.date()}")

        # Split
        train_mask = df.index < train_end
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "target"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "target"].values

        if len(X_train) < 100 or len(X_test) < 50:
            logger.warning(f"  Skipping fold: train={len(X_train)}, test={len(X_test)}")
            continue

        logger.info(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(
            f"  Train target: up={y_train.mean():.1%}, Test target: up={y_test.mean():.1%}"
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Replace any remaining NaN/inf after scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

        # Train HistGradientBoosting
        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            l2_regularization=1.0,
            max_bins=255,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            class_weight="balanced",
        )

        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        majority = max(y_test.mean(), 1 - y_test.mean())
        edge = acc - majority

        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.5

        try:
            ll = log_loss(y_test, y_proba)
        except Exception:
            ll = np.nan

        # Simulated PnL
        # Signal: if predicted up, go long (+1), else short (-1)
        # Scaled by model confidence
        test_returns = df.loc[test_mask, "forward_return"].values
        signal = 2 * y_proba - 1  # Map [0,1] -> [-1,1]
        strategy_returns = signal * test_returns
        strategy_returns = np.nan_to_num(strategy_returns, nan=0.0)

        total_return = np.sum(strategy_returns)
        sharpe = (
            np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        )

        logger.info(
            f"  Accuracy: {acc:.1%} (majority: {majority:.1%}, edge: {edge:+.1%})"
        )
        logger.info(f"  AUC: {auc:.4f}, LogLoss: {ll:.4f}")
        logger.info(f"  Return: {total_return:+.2%}, Sharpe: {sharpe:.2f}")
        logger.info(f"  Train time: {train_time:.1f}s, Iterations: {model.n_iter_}")

        fold_result = {
            "fold": fold_i + 1,
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": float(acc),
            "majority_baseline": float(majority),
            "edge": float(edge),
            "auc": float(auc),
            "log_loss": float(ll),
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "train_time": float(train_time),
            "n_iterations": int(model.n_iter_),
        }
        all_results.append(fold_result)

        # Store predictions for analysis
        test_dates = df.loc[test_mask].index
        test_symbols = df.loc[test_mask, "symbol"].values
        pred_df = pd.DataFrame(
            {
                "date": test_dates,
                "symbol": test_symbols,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "forward_return": test_returns,
                "strategy_return": strategy_returns,
            }
        )
        all_predictions.append(pred_df)

        # Track best model
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_scaler = scaler

    # ── Summary ───────────────────────────────────────────────────────
    if not all_results:
        logger.error("No folds completed!")
        return {"error": "No folds completed"}

    avg_acc = np.mean([r["accuracy"] for r in all_results])
    avg_edge = np.mean([r["edge"] for r in all_results])
    avg_auc = np.mean([r["auc"] for r in all_results])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    avg_return = np.mean([r["total_return"] for r in all_results])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"WALK-FORWARD RESULTS ({len(all_results)} folds)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Avg Accuracy:  {avg_acc:.1%}")
    logger.info(f"  Avg Edge:      {avg_edge:+.1%}")
    logger.info(f"  Avg AUC:       {avg_auc:.4f}")
    logger.info(f"  Avg Sharpe:    {avg_sharpe:.2f}")
    logger.info(f"  Avg Return:    {avg_return:+.2%}")

    # Feature importance (from best model)
    if best_model is not None and hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info(f"\nTop 20 Features:")
        for fname, imp in feat_imp[:20]:
            logger.info(f"  {fname:40s} {imp:.4f}")

    # Combine predictions
    if all_predictions:
        all_preds = pd.concat(all_predictions, axis=0)
    else:
        all_preds = pd.DataFrame()

    # ── Train final model on ALL data (for deployment) ────────────────
    logger.info(f"\nTraining final model on ALL {len(df)} samples...")

    X_all = df[feature_cols].values
    y_all = df["target"].values

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    final_model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        class_weight="balanced",
    )
    final_model.fit(X_all_scaled, y_all)

    # ── Save everything ───────────────────────────────────────────────
    # Save model
    model_path = MODEL_DIR / "meta_ensemble_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "scaler": final_scaler,
                "feature_cols": feature_cols,
                "trained_at": datetime.now().isoformat(),
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "walk_forward_results": all_results,
            },
            f,
        )
    logger.info(f"Final model saved to {model_path}")

    # Save walk-forward best model
    best_path = MODEL_DIR / "meta_ensemble_best_wf.pkl"
    with open(best_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "scaler": best_scaler,
                "feature_cols": feature_cols,
                "best_auc": best_auc,
            },
            f,
        )

    # Save results
    results_path = MODEL_DIR / "walk_forward_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "n_folds": len(all_results),
                    "avg_accuracy": float(avg_acc),
                    "avg_edge": float(avg_edge),
                    "avg_auc": float(avg_auc),
                    "avg_sharpe": float(avg_sharpe),
                    "avg_return": float(avg_return),
                    "n_features": len(feature_cols),
                    "n_total_samples": len(df),
                    "feature_cols": feature_cols,
                },
                "folds": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {results_path}")

    # Save predictions
    if not all_preds.empty:
        pred_path = MODEL_DIR / "walk_forward_predictions.pkl"
        all_preds.to_pickle(pred_path)
        logger.info(f"Predictions saved to {pred_path}")

    return {
        "summary": {
            "avg_accuracy": avg_acc,
            "avg_edge": avg_edge,
            "avg_auc": avg_auc,
            "avg_sharpe": avg_sharpe,
            "avg_return": avg_return,
        },
        "folds": all_results,
        "model": final_model,
        "scaler": final_scaler,
        "feature_cols": feature_cols,
        "predictions": all_preds,
    }


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Meta-Ensemble Trading System")
    parser.add_argument(
        "--phase",
        choices=["fetch", "features", "train", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=100,
        help="Number of symbols to use (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: 10 symbols, fewer folds",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Re-download all OHLCV data",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Limit number of walk-forward folds",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_symbols = 10
        if args.n_folds is None:
            args.n_folds = 3

    logger.info(
        f"Meta-Ensemble Pipeline: phase={args.phase}, n_symbols={args.n_symbols}"
    )

    # ── Phase 1: Fetch ────────────────────────────────────────────────
    if args.phase in ("fetch", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Fetching OHLCV Data")
        logger.info("=" * 60)

        symbols = get_symbol_universe(args.n_symbols)
        logger.info(f"Universe: {len(symbols)} symbols")

        ohlcv_data = fetch_all_ohlcv(symbols, force=args.force_fetch)

        total_bars = sum(len(df) for df in ohlcv_data.values())
        logger.info(f"Total: {len(ohlcv_data)} symbols, {total_bars:,} bars")

        if args.phase == "fetch":
            return

    # ── Phase 2: Features ─────────────────────────────────────────────
    if args.phase in ("features", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Computing Features")
        logger.info("=" * 60)

        # Load OHLCV data from cache
        if args.phase == "features" or "ohlcv_data" not in dir():
            symbols = get_symbol_universe(args.n_symbols)
            ohlcv_data = {}
            for sym in symbols:
                cache_file = CACHE_DIR / f"{sym}.pkl"
                if cache_file.exists():
                    try:
                        ohlcv_data[sym] = pd.read_pickle(cache_file)
                    except Exception:
                        pass
            logger.info(f"Loaded {len(ohlcv_data)} symbols from cache")

        feature_matrix = build_feature_matrix(ohlcv_data)

        if args.phase == "features":
            return

    # ── Phase 3: Train ────────────────────────────────────────────────
    if args.phase in ("train", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Training Meta-Ensemble")
        logger.info("=" * 60)

        # Load feature matrix
        if args.phase == "train" or "feature_matrix" not in dir():
            fm_path = FEATURE_DIR / "feature_matrix.pkl"
            if not fm_path.exists():
                logger.error(
                    f"Feature matrix not found at {fm_path}. Run --phase features first."
                )
                return
            feature_matrix = pd.read_pickle(fm_path)
            logger.info(f"Loaded feature matrix: {len(feature_matrix)} samples")

        results = walk_forward_train(feature_matrix, n_folds=args.n_folds)

        if "error" not in results:
            logger.info(f"\nFinal Results:")
            logger.info(f"  Accuracy: {results['summary']['avg_accuracy']:.1%}")
            logger.info(f"  Edge:     {results['summary']['avg_edge']:+.1%}")
            logger.info(f"  AUC:      {results['summary']['avg_auc']:.4f}")
            logger.info(f"  Sharpe:   {results['summary']['avg_sharpe']:.2f}")


if __name__ == "__main__":
    main()
