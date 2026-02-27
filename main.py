"""
Quantum Alpha V1 - Main Entry Point
Single command deployment for backtesting.
"""

import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.strategy.signals import (
    MomentumStrategy,
    CompositeStrategy,
    AdaptiveCompositeStrategy,
    EnhancedCompositeStrategy,
)
from quantum_alpha.backtesting.engine import Backtester, OrderSide, OrderType
from quantum_alpha.backtesting.validation import MCPT, BootstrapAnalysis
from quantum_alpha.backtesting.performance_metrics import (
    compute_metrics,
    compute_metrics_from_returns,
)
from quantum_alpha.backtesting.performance_gate import (
    evaluate_gate,
    aggregate_fundamentals,
)
from quantum_alpha.backtesting.benchmark_profiles import (
    evaluate_quant_firm_benchmarks,
    benchmark_rows,
)
from quantum_alpha.risk.position_sizing import PositionSizer, VaRCalculator
from quantum_alpha.risk.drawdown_control import DrawdownController, DrawdownState
from quantum_alpha.execution.paper_trader import PaperTrader
from quantum_alpha.config.validator import (
    validate_settings,
    validate_strategies,
    validate_risk_limits,
    validate_data_sources,
)
from quantum_alpha.monitoring.logging import configure_logging
from quantum_alpha.monitoring.alert_system import AlertManager, build_default_rules
from quantum_alpha.plugins import load_plugins


def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "quantum_alpha" / "config" / "settings.yaml"

    config_path = Path(config_path)
    if config_path.is_dir():
        config_path = config_path / "settings.yaml"

    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    issues = validate_settings(settings)
    config_dir = config_path.parent

    strategies_path = config_dir / "strategies.yaml"
    if strategies_path.exists():
        with open(strategies_path, "r") as f:
            strategies_cfg = yaml.safe_load(f)
        issues.extend(validate_strategies(strategies_cfg))

    risk_limits_path = config_dir / "risk_limits.yaml"
    if risk_limits_path.exists():
        with open(str(risk_limits_path), "r") as f:
            risk_cfg = yaml.safe_load(f)
        issues.extend(validate_risk_limits(risk_cfg))

    data_sources_path = config_dir / "data_sources.yaml"
    if data_sources_path.exists():
        with open(str(data_sources_path), "r") as f:
            data_cfg = yaml.safe_load(f)
        issues.extend(validate_data_sources(data_cfg))

    if issues:
        raise ValueError(f"Config validation failed: {', '.join(issues)}")

    return settings


def _resolve_config_dir(config_path: Optional[str]) -> Path:
    if config_path is None:
        return PROJECT_ROOT / "quantum_alpha" / "config"

    config_path = Path(config_path)
    return config_path if config_path.is_dir() else config_path.parent


def _load_optional_yaml(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_signal_lag(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Lag signals and sizing features to avoid lookahead."""
    for col in (
        "signal",
        "signal_confidence",
        "position_signal",
        "atr_pct",
        "mom_12m",
        "mom_3m",
    ):
        if col in df.columns:
            df[col] = df[col].shift(lag)

    if "signal" in df.columns:
        df["signal"] = df["signal"].fillna(0.0)
    if "signal_confidence" in df.columns:
        df["signal_confidence"] = df["signal_confidence"].fillna(0.0)
    if "position_signal" in df.columns:
        df["position_signal"] = df["position_signal"].fillna(0.0)
    if "atr_pct" in df.columns:
        df["atr_pct"] = df["atr_pct"].fillna(0.02)
    if "mom_12m" in df.columns:
        df["mom_12m"] = df["mom_12m"].fillna(0.0)
    if "mom_3m" in df.columns:
        df["mom_3m"] = df["mom_3m"].fillna(0.0)

    return df


def _realized_vol_from_equity(equity_curve, window: int = 63) -> Optional[float]:
    """Compute annualized realized vol from equity curve entries."""
    if not equity_curve or len(equity_curve) < 2:
        return None
    eq = np.array([row["equity"] for row in equity_curve], dtype=float)
    if len(eq) < 2:
        return None
    window = min(window, len(eq) - 1)
    if window < 2:
        return None
    eq = eq[-(window + 1) :]
    rets = np.diff(eq) / eq[:-1]
    if rets.std() == 0:
        return None
    return float(rets.std() * np.sqrt(252))


def _vol_of_vol_from_equity(
    equity_curve, short_window: int = 21, long_window: int = 63
) -> Optional[float]:
    """Estimate volatility of volatility from equity curve."""
    if not equity_curve or len(equity_curve) < (long_window + 2):
        return None
    eq = np.array([row["equity"] for row in equity_curve], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    if len(rets) < long_window:
        return None
    vol_series = pd.Series(rets).rolling(short_window).std().dropna()
    if len(vol_series) < long_window // 2:
        return None
    return float(vol_series.tail(long_window).std())


def _align_signal_frame(
    sig_df: pd.DataFrame, index: pd.Index, limit: int = 10
) -> pd.DataFrame:
    """Align sparse signal frames to price index with a short forward fill."""
    if sig_df.empty:
        return pd.DataFrame(
            {"signal": np.zeros(len(index)), "signal_confidence": np.zeros(len(index))},
            index=index,
        )
    frame = sig_df.copy()
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    signal = frame["signal"].reindex(index, method="ffill", limit=limit).fillna(0.0)
    confidence = (
        frame.get("signal_confidence", pd.Series(0.5, index=frame.index))
        .reindex(index, method="ffill", limit=limit)
        .fillna(0.0)
    )
    return pd.DataFrame(
        {"signal": signal, "signal_confidence": confidence}, index=index
    )


def _resolve_symbols(
    symbols: Optional[list],
    collector: DataCollector,
    settings: Optional[Dict],
) -> list:
    data_cfg = settings.get("data", {}) if settings else {}
    universe_limit = int(data_cfg.get("universe_limit", 0))

    def _limit(values: list) -> list:
        if universe_limit and universe_limit > 0:
            return list(values)[:universe_limit]
        return list(values)

    # Try to load from centralized universe module
    try:
        from quantum_alpha import universe as _u

        _has_universe = True
    except ImportError:
        _has_universe = False

    if symbols:
        if len(symbols) == 1:
            token = symbols[0].upper()
            if token in {"SP500", "S&P500", "SPX"}:
                if _has_universe:
                    return _limit(_u.get_sp500())
                return _limit(collector.get_sp500_symbols())
            if token in {"SP400", "MIDCAP", "MID"}:
                if _has_universe:
                    return _limit(_u.get_sp400())
                logger.warning(
                    "universe.py not available; SP400 token unsupported, falling back to SP500"
                )
                return _limit(collector.get_sp500_symbols())
            if token in {"FULL", "ALL", "UNIVERSE"}:
                if _has_universe:
                    return _limit(_u.get_stocks_only())
                logger.warning(
                    "universe.py not available; FULL token unsupported, falling back to SP500"
                )
                return _limit(collector.get_sp500_symbols())
            if token in {"LIQUID", "LIQUID50"}:
                if _has_universe:
                    return _limit(_u.get_liquid_largecap())
                return _limit(collector.get_sp500_symbols()[:50])
            if token in {"AUTO", "DEFAULT"}:
                return _limit(data_cfg.get("default_universe", ["SPY"]))
        return symbols

    return _limit(data_cfg.get("default_universe", ["SPY"]))


def _format_symbols(symbols: list) -> str:
    if len(symbols) <= 15:
        return str(symbols)
    head = ", ".join(symbols[:10])
    return f"{len(symbols)} symbols (e.g., {head}, ...)"


def _select_liquid_subset(
    frames: Dict[str, pd.DataFrame],
    subset_size: int,
    adv_window: int = 20,
    min_history: int = 120,
) -> list:
    if subset_size <= 0 or not frames:
        return list(frames.keys())

    scores = []
    for symbol, df in frames.items():
        if df is None or len(df) < min_history:
            continue
        close = pd.to_numeric(df.get("close"), errors="coerce")
        volume = pd.to_numeric(df.get("volume"), errors="coerce")
        if close is None or volume is None:
            continue
        dollar_volume = (close * volume).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dollar_volume) < max(5, adv_window):
            continue
        adv = float(dollar_volume.tail(adv_window).mean())
        if np.isfinite(adv) and adv > 0:
            scores.append((symbol, adv))

    if not scores:
        return list(frames.keys())

    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [sym for sym, _ in scores[:subset_size]]
    return selected


def _total_return_from_returns(returns: pd.Series) -> float:
    if returns is None:
        return 0.0
    s = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 1:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def _rolling_oos_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window_days: int = 126,
    min_windows: int = 3,
    min_beat_ratio: float = 0.75,
) -> Dict[str, object]:
    strat = (
        pd.Series(strategy_returns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    bench = (
        pd.Series(benchmark_returns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    aligned = strat.align(bench, join="inner")
    if aligned[0].empty or window_days <= 1:
        return {
            "available": False,
            "reason": "insufficient_overlap",
            "n_windows": 0,
            "beats": 0,
            "passed": False,
            "windows": [],
        }

    strat_aligned, bench_aligned = aligned
    n = len(strat_aligned)
    n_windows = n // window_days
    if n_windows < 1:
        return {
            "available": False,
            "reason": "insufficient_length",
            "n_windows": 0,
            "beats": 0,
            "passed": False,
            "windows": [],
        }

    start_idx = n - (n_windows * window_days)
    windows = []
    beats = 0
    for w in range(n_windows):
        lo = start_idx + w * window_days
        hi = lo + window_days
        s_win = strat_aligned.iloc[lo:hi]
        b_win = bench_aligned.iloc[lo:hi]
        if s_win.empty or b_win.empty:
            continue
        s_ret = _total_return_from_returns(s_win)
        b_ret = _total_return_from_returns(b_win)
        beat = s_ret > b_ret
        if beat:
            beats += 1
        windows.append(
            {
                "window_id": w + 1,
                "start": str(s_win.index[0].date()),
                "end": str(s_win.index[-1].date()),
                "strategy_total_return": s_ret,
                "benchmark_total_return": b_ret,
                "beat": bool(beat),
            }
        )

    ratio = float(min_beat_ratio)
    if not np.isfinite(ratio):
        ratio = 0.75
    ratio = float(np.clip(ratio, 0.0, 1.0))
    ratio_required = int(np.ceil(ratio * len(windows))) if windows else 0
    required = max(int(min_windows), int(ratio_required))
    passed = len(windows) >= min_windows and beats >= required
    return {
        "available": True,
        "n_windows": int(len(windows)),
        "beats": int(beats),
        "required_beats": int(required),
        "passed": bool(passed),
        "window_days": int(window_days),
        "min_beat_ratio": ratio,
        "windows": windows,
    }


def _should_rebalance(
    ts: datetime, last_ts: Optional[datetime], frequency: str
) -> bool:
    if last_ts is None:
        return True

    freq = (frequency or "daily").lower()
    if freq == "daily":
        return ts.date() != last_ts.date()
    if freq == "weekly":
        ts_iso = ts.isocalendar()
        last_iso = last_ts.isocalendar()
        return (ts_iso.year, ts_iso.week) != (last_iso.year, last_iso.week)
    if freq == "biweekly":
        # Rebalance every 2 weeks (10 trading days)
        delta = (ts - last_ts).days
        return delta >= 14
    if freq == "monthly":
        return (ts.year, ts.month) != (last_ts.year, last_ts.month)
    return True


def run_backtest(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "enhanced",
    validate: bool = False,
    verbose: bool = True,
    config_path: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    strategy_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Run a complete backtest.

    Args:
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        strategy_type: Trading strategy key (default 'enhanced')
        validate: Whether to run MCPT validation
        verbose: Print progress
        config_path: Path to config directory or settings.yaml
        checkpoint_name: Model checkpoint name (for news_lstm strategy)
        strategy_kwargs: Extra keyword arguments passed to the strategy constructor

    Returns:
        Dict with backtest results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("QUANTUM ALPHA V1 - BACKTEST")
        print(f"{'=' * 60}")
        print(f"Symbols: {_format_symbols(symbols)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Capital: ${initial_capital:,.0f}")
        print(f"Strategy: {strategy_type}")
        print(f"{'=' * 60}\n")

    settings = load_config(config_path)
    risk_cfg = settings.get("risk", {}) if settings else {}
    strategy_cfg = settings.get("strategy", {}) if settings else {}
    validation_cfg = settings.get("validation", {}) if settings else {}
    config_dir = _resolve_config_dir(config_path)
    strategies_cfg = _load_optional_yaml(config_dir / "strategies.yaml")
    sentiment_cfg = {}
    if strategies_cfg and "strategies" in strategies_cfg:
        sentiment_cfg = strategies_cfg["strategies"].get("sentiment", {})
    max_position = float(risk_cfg.get("max_position_size", 0.25))
    max_leverage = float(risk_cfg.get("max_portfolio_leverage", 1.0))
    max_drawdown = float(risk_cfg.get("max_drawdown", 0.10))
    kelly_fraction = float(risk_cfg.get("kelly_fraction", 0.5))
    rebalance_frequency = str(strategy_cfg.get("rebalance_frequency", "daily"))
    paper_rebalance = strategy_cfg.get("paper_rebalance_frequency")
    if paper_rebalance:
        rebalance_frequency = str(paper_rebalance)
    momentum_top_pct = float(strategy_cfg.get("momentum_top_pct", 65))
    momentum_bottom_pct = float(strategy_cfg.get("momentum_bottom_pct", 35))
    signal_threshold = float(strategy_cfg.get("signal_threshold", 0.3))
    signal_scale = float(strategy_cfg.get("signal_scale", 1.0))
    min_long_signal = float(strategy_cfg.get("min_long_signal", 0.0))
    market_off_scale = float(strategy_cfg.get("market_off_scale", 1.0))
    vol_of_vol_threshold = float(risk_cfg.get("vol_of_vol_threshold", 0.03))
    vol_of_vol_scale = float(risk_cfg.get("vol_of_vol_scale", 0.7))
    rs_top_n = int(strategy_cfg.get("relative_strength_top_n", 0))
    rs_min_mom = float(strategy_cfg.get("relative_strength_min_mom", 0.0))
    use_relative_strength = bool(strategy_cfg.get("use_relative_strength", False))
    liquid_subset_size = int(strategy_cfg.get("liquid_subset_size", 0))
    liquid_adv_window = int(strategy_cfg.get("liquid_adv_window", 20))
    liquid_min_history = int(strategy_cfg.get("liquid_min_history", 120))
    liquid_for_full_only = bool(strategy_cfg.get("liquid_for_full_only", True))
    long_only = bool(strategy_cfg.get("long_only", False))
    risk_off_cash = bool(strategy_cfg.get("risk_off_cash", False))
    if max_leverage <= 0:
        max_leverage = 1.0
    # Volatility target for scaling (annualized) - nudged up to restore exposure
    target_vol = float(risk_cfg.get("target_volatility", 0.15))
    dd_leverage_cap_warning = float(risk_cfg.get("dd_leverage_cap_warning", 1.0))
    dd_leverage_cap_critical = float(risk_cfg.get("dd_leverage_cap_critical", 1.0))
    market_off_leverage_cap = float(risk_cfg.get("market_off_leverage_cap", 1.0))

    # Initialize components
    collector = DataCollector()
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    full_token_requested = bool(
        symbols
        and len(symbols) == 1
        and str(symbols[0]).upper() in {"FULL", "ALL", "UNIVERSE"}
    )
    symbols = _resolve_symbols(symbols, collector, settings)

    if strategy_type == "momentum":
        strategy = MomentumStrategy()
    elif strategy_type == "composite":
        strategy = CompositeStrategy()
    elif strategy_type in ("adaptive", "enhanced"):
        strategy = EnhancedCompositeStrategy()
    elif strategy_type == "sentiment":
        from quantum_alpha.strategy.sentiment_strategies import SocialSentimentStrategy

        strategy = SocialSentimentStrategy()
    elif strategy_type == "ml":
        from quantum_alpha.strategy.ml_strategies import MLTradingStrategy

        strategy = MLTradingStrategy(**(strategy_kwargs or {}))
    elif strategy_type == "news_lstm":
        from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy

        kwargs = dict(strategy_kwargs or {})
        if checkpoint_name is not None:
            kwargs.setdefault("checkpoint_name", checkpoint_name)
        strategy = NewsLSTMStrategy(**kwargs)
    elif strategy_type == "meta_ensemble":
        from quantum_alpha.strategy.meta_ensemble_strategy import MetaEnsembleStrategy

        strategy = MetaEnsembleStrategy(**(strategy_kwargs or {}))
    else:
        strategy = MomentumStrategy()

    use_enhanced = isinstance(strategy, EnhancedCompositeStrategy)
    use_sentiment = strategy_type == "sentiment"

    if use_sentiment:
        from quantum_alpha.data.collectors.alternative import (
            load_social_sentiment,
            load_options_sentiment,
            load_insider_trades,
            load_congress_trades,
        )
        from quantum_alpha.strategy.sentiment_strategies import (
            SocialSentimentStrategy,
            OptionsSentimentStrategy,
            InsiderTradingStrategy,
            CongressTradingStrategy,
        )

        social_strategy = SocialSentimentStrategy(**sentiment_cfg.get("social", {}))
        options_strategy = OptionsSentimentStrategy(**sentiment_cfg.get("options", {}))
        insider_strategy = InsiderTradingStrategy(**sentiment_cfg.get("insider", {}))
        congress_strategy = CongressTradingStrategy(**sentiment_cfg.get("congress", {}))
        sentiment_weights = sentiment_cfg.get(
            "combined_weights",
            {"social": 0.4, "options": 0.2, "insider": 0.2, "congress": 0.2},
        )
    use_sentiment = strategy_type == "sentiment"

    if use_sentiment:
        from quantum_alpha.data.collectors.alternative import (
            load_social_sentiment,
            load_options_sentiment,
            load_insider_trades,
            load_congress_trades,
        )
        from quantum_alpha.strategy.sentiment_strategies import (
            SocialSentimentStrategy,
            OptionsSentimentStrategy,
            InsiderTradingStrategy,
            CongressTradingStrategy,
        )

        social_strategy = SocialSentimentStrategy(**sentiment_cfg.get("social", {}))
        options_strategy = OptionsSentimentStrategy(**sentiment_cfg.get("options", {}))
        insider_strategy = InsiderTradingStrategy(**sentiment_cfg.get("insider", {}))
        congress_strategy = CongressTradingStrategy(**sentiment_cfg.get("congress", {}))
        sentiment_weights = sentiment_cfg.get(
            "combined_weights",
            {"social": 0.4, "options": 0.2, "insider": 0.2, "congress": 0.2},
        )

    if use_sentiment:
        signal_threshold = float(
            sentiment_cfg.get("signal_threshold", signal_threshold)
        )

    position_sizer = PositionSizer(
        max_position=max_position,
        kelly_fraction=kelly_fraction,
        max_drawdown=max_drawdown,
    )
    backtester = Backtester(initial_capital=initial_capital)

    bench_cfg = settings.get("benchmarks", {})
    market_benchmark = bench_cfg.get("market", "SPY")
    quant_benchmark = bench_cfg.get("quant_composite", ["QQQ", "IWM"])
    market_df = None
    market_trend = None
    try:
        market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        m_close = market_df["close"]
        m_ma = m_close.rolling(200).mean()
        market_trend = m_close.shift(1) > m_ma.shift(1)
    except Exception:
        market_df = None
        market_trend = None

    # Collect data
    if verbose:
        print("Collecting price data...")

    data = {}
    raw_featured = {}  # For enhanced strategy: feature-generated but pre-signal
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = cleaner.clean(df)
            df = imputer.impute(df)
            df = feature_gen.generate(df)
            raw_featured[symbol] = df
            if use_sentiment:
                social_df = load_social_sentiment(symbol, use_live=False)
                options_df = load_options_sentiment(symbol, use_live=False)
                insider_df = load_insider_trades(symbol, use_live=True)
                congress_df = load_congress_trades(symbol, use_live=True)

                if "symbol" in social_df.columns:
                    social_df = social_df[social_df["symbol"] == symbol]
                if "symbol" in options_df.columns:
                    options_df = options_df[options_df["symbol"] == symbol]
                if "symbol" in insider_df.columns:
                    insider_df = insider_df[insider_df["symbol"] == symbol]
                if "symbol" in congress_df.columns:
                    congress_df = congress_df[congress_df["symbol"] == symbol]

                frames = {}
                if not social_df.empty:
                    frames["social"] = _align_signal_frame(
                        social_strategy.generate_signals(social_df), df.index
                    )
                if not options_df.empty:
                    if "timestamp" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["timestamp"] = pd.to_datetime(
                            options_df["timestamp"]
                        )
                        options_df = options_df.set_index("timestamp")
                    elif "date" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["date"] = pd.to_datetime(options_df["date"])
                        options_df = options_df.set_index("date")
                    options_sig = options_strategy.generate_signals(options_df)
                    frames["options"] = _align_signal_frame(options_sig, df.index)
                if not insider_df.empty:
                    frames["insider"] = _align_signal_frame(
                        insider_strategy.generate_signals(insider_df),
                        df.index,
                        limit=20,
                    )
                if not congress_df.empty:
                    frames["congress"] = _align_signal_frame(
                        congress_strategy.generate_signals(congress_df),
                        df.index,
                        limit=20,
                    )

                combined_signal = pd.Series(0.0, index=df.index)
                combined_conf = pd.Series(0.0, index=df.index)
                total_w = 0.0
                for key, sig_frame in frames.items():
                    weight = float(sentiment_weights.get(key, 0.0))
                    if weight <= 0:
                        continue
                    total_w += weight
                    combined_signal += weight * sig_frame["signal"]
                    combined_conf += weight * sig_frame["signal_confidence"]

                if total_w > 0:
                    combined_signal = combined_signal / total_w
                    combined_conf = combined_conf / total_w
                else:
                    combined_signal = combined_signal * 0.0
                    combined_conf = combined_conf * 0.0

                df["signal"] = combined_signal
                df["signal_confidence"] = combined_conf
                df["position_signal"] = np.where(
                    np.abs(df["signal"]) >= signal_threshold,
                    np.sign(df["signal"]),
                    0.0,
                )
                df = _apply_signal_lag(df)
                data[symbol] = df
            elif not use_enhanced:
                # Pass symbol kwarg for strategies that need it (e.g. news_lstm)
                import inspect

                sig = inspect.signature(strategy.generate_signals)
                if "symbol" in sig.parameters:
                    df = strategy.generate_signals(df, symbol=symbol)
                else:
                    df = strategy.generate_signals(df)
                df = _apply_signal_lag(df)
                data[symbol] = df
            if verbose:
                print(f"  {symbol}: {len(df)} bars")
        except Exception as e:
            if verbose:
                print(f"  {symbol}: FAILED - {e}")

    apply_liquid_filter = (
        liquid_subset_size > 0
        and len(raw_featured) > liquid_subset_size
        and (full_token_requested or not liquid_for_full_only)
    )
    if apply_liquid_filter:
        selected = _select_liquid_subset(
            raw_featured,
            subset_size=liquid_subset_size,
            adv_window=liquid_adv_window,
            min_history=liquid_min_history,
        )
        selected_set = set(selected)
        raw_featured = {
            sym: df for sym, df in raw_featured.items() if sym in selected_set
        }
        data = {sym: df for sym, df in data.items() if sym in selected_set}
        if verbose:
            print(
                "  Liquidity filter applied: "
                f"trading top {len(selected)} symbols by {liquid_adv_window}D ADV"
            )

    # Enhanced strategy: fit cross-asset signals, then generate per-symbol
    if use_enhanced and raw_featured:
        if verbose:
            print("  Computing cross-asset signals...")
        strategy.fit_cross_asset(raw_featured)
        for symbol, df in raw_featured.items():
            df = strategy.generate_signals(df, symbol=symbol)
            df = _apply_signal_lag(df)
            data[symbol] = df

    if not data:
        return {"error": "No data collected"}

    if verbose:
        print(f"\nRunning backtest...")

    # Track state for strategy
    state = {
        "positions": {},
        "trade_history": [],
        "current_drawdown": 0,
        "peak_equity": initial_capital,
        "last_rebalance": None,
    }

    # Dynamic drawdown control — soft scaling only, no circuit breaker.
    # Thresholds are wide enough for multi-decade equity backtests where
    # 20-30% drawdowns are normal (e.g. 2008, 2020, 2022).
    dd_warning = float(risk_cfg.get("dd_warning", 0.15))
    dd_critical = float(risk_cfg.get("dd_critical", 0.25))
    dd_breaker = float(risk_cfg.get("dd_circuit_breaker", 0.40))
    dd_limit = float(risk_cfg.get("dd_max_limit", 0.50))
    dd_min_exposure = float(risk_cfg.get("dd_min_exposure", 0.25))
    dd_controller = DrawdownController(
        warning_threshold=dd_warning,
        critical_threshold=dd_critical,
        circuit_breaker_threshold=dd_breaker,
        max_drawdown_limit=dd_limit,
        scaling_method="linear",
        min_exposure=dd_min_exposure,
        cooldown_days=10,
    )
    dd_controller.reset(initial_capital)

    def trading_strategy(timestamp, bars, bt):
        """Strategy execution function."""
        equity = bt._total_equity()

        # Update drawdown controller every bar
        dd_metrics = dd_controller.update(equity, timestamp)

        # Soft exposure scaling only — no hard circuit breaker that closes
        # all positions.  The exposure_multiplier already approaches zero
        # as drawdown deepens, which achieves the same effect without the
        # all-or-nothing slam that destroys multi-decade backtests.
        exposure_mult = dd_metrics.exposure_multiplier

        if not _should_rebalance(
            timestamp, state["last_rebalance"], rebalance_frequency
        ):
            state["peak_equity"] = max(state["peak_equity"], equity)
            state["current_drawdown"] = (equity - state["peak_equity"]) / state[
                "peak_equity"
            ]
            return

        state["last_rebalance"] = timestamp
        target_positions = {}
        volatilities = {}
        trade_history = (
            np.array([t["pnl"] for t in bt.trades]) if bt.trades else np.array([0])
        )
        realized_vol = _realized_vol_from_equity(bt.equity_curve)
        vol_of_vol = _vol_of_vol_from_equity(bt.equity_curve)
        vol_scale = 1.0
        if realized_vol and realized_vol > 0:
            vol_scale = float(np.clip(target_vol / realized_vol, 0.5, 1.5))
        if (
            vol_of_vol is not None
            and vol_of_vol > vol_of_vol_threshold
            and dd_metrics.state
            in {
                DrawdownState.WARNING,
                DrawdownState.CRITICAL,
                DrawdownState.RECOVERY,
                DrawdownState.CIRCUIT_BREAKER,
            }
        ):
            vol_scale *= vol_of_vol_scale
        market_risk_on = True
        if market_trend is not None and timestamp in market_trend.index:
            market_risk_on = bool(market_trend.loc[timestamp])
        allow_shorts = not long_only
        if not market_risk_on:
            allow_shorts = False

        mom_scores = {}
        for sym, df in data.items():
            if timestamp in df.index:
                mom_val = df.loc[timestamp].get("mom_12m", 0.0)
                if pd.notna(mom_val):
                    mom_scores[sym] = float(mom_val)
        top_cut = None
        bottom_cut = None
        top_syms = None
        if len(mom_scores) >= 3:
            vals = np.array(list(mom_scores.values()), dtype=float)
            top_cut = np.nanpercentile(vals, momentum_top_pct)
            bottom_cut = np.nanpercentile(vals, momentum_bottom_pct)
            if use_relative_strength and rs_top_n > 0:
                ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                top_syms = {sym for sym, val in ranked[:rs_top_n] if val >= rs_min_mom}

        use_rp_allocation = top_syms is not None

        for symbol, bar in bars.items():
            if symbol not in data:
                continue

            df = data[symbol]
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            signal = (
                row.get("position_signal")
                if "position_signal" in row
                else row.get("signal", 0)
            )
            confidence = float(row.get("signal_confidence", 0.5))
            if top_syms is not None:
                if use_enhanced:
                    # Enhanced strategy: use top_syms as universe filter,
                    # but preserve strategy's signal magnitude & direction
                    if symbol not in top_syms:
                        signal = 0.0
                    else:
                        # Ensure long-only compliance; keep magnitude
                        signal = max(signal, 0.0) if long_only else signal
                        # Floor at a small positive so selected symbols trade
                        if signal == 0.0:
                            signal = 0.5
                else:
                    signal = 1.0 if symbol in top_syms else 0.0
            else:
                if signal < 0 and not allow_shorts:
                    signal = 0.0
                if top_cut is not None and bottom_cut is not None:
                    mom_val = mom_scores.get(symbol)
                    if signal > 0 and (mom_val is None or mom_val < top_cut):
                        signal = 0.0
                    if signal < 0 and (
                        mom_val is None or mom_val > bottom_cut or long_only
                    ):
                        signal = 0.0
            if signal_scale != 1.0 and signal != 0.0:
                signal = float(np.clip(signal * signal_scale, -1.0, 1.0))
            if long_only and signal > 0 and min_long_signal > 0:
                signal = max(signal, min_long_signal)
            volatility = row.get("atr_pct", 0.02)
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.02
            volatility = volatility * np.sqrt(252)
            volatilities[symbol] = max(volatility, 1e-6)
            if use_rp_allocation:
                continue

            confidence = row.get("signal_confidence", 0.5)

            # Get current position
            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0

            # Calculate position size
            sizing = position_sizer.calculate(
                trade_history=trade_history,
                current_volatility=max(volatility, 0.01),
                current_drawdown=state["current_drawdown"],
                signal_strength=signal,
                signal_confidence=confidence,
            )

            if sizing["halt_trading"]:
                return

            target_positions[symbol] = sizing["position_size"]

        if use_rp_allocation:
            # Only allocate to symbols selected by relative strength
            selected_vols = {
                s: v
                for s, v in volatilities.items()
                if top_syms is None or s in top_syms
            }
            if selected_vols:
                inv = {s: 1 / v for s, v in selected_vols.items()}
                total_inv = sum(inv.values())
                if total_inv > 0:
                    target_positions = {
                        s: (inv[s] / total_inv) * max_leverage for s in inv
                    }
            if not target_positions:
                return
        if risk_off_cash and not market_risk_on:
            target_positions = {s: 0.0 for s in bars.keys()}
        elif not target_positions:
            return
        if not market_risk_on and market_off_scale < 1.0:
            target_positions = {
                s: w * market_off_scale for s, w in target_positions.items()
            }

        if volatilities and not use_rp_allocation:
            inv = {s: 1 / v for s, v in volatilities.items()}
            total_inv = sum(inv.values())
            if total_inv > 0:
                rp_weights = {s: inv[s] / total_inv for s in inv}
                weight_scale = len(rp_weights)
                target_positions = {
                    s: target_positions[s] * rp_weights.get(s, 0) * weight_scale
                    for s in target_positions
                }

        total_abs = sum(abs(w) for w in target_positions.values())
        cap = max_leverage
        if dd_metrics.state == DrawdownState.WARNING:
            cap = min(cap, dd_leverage_cap_warning)
        elif dd_metrics.state in {DrawdownState.CRITICAL, DrawdownState.RECOVERY}:
            cap = min(cap, dd_leverage_cap_critical)
        if not market_risk_on:
            cap = min(cap, market_off_leverage_cap)
        if total_abs > cap and total_abs > 0:
            scale = cap / total_abs
            target_positions = {s: w * scale for s, w in target_positions.items()}

        # Apply drawdown-based exposure scaling
        if exposure_mult < 1.0:
            target_positions = {
                s: w * exposure_mult for s, w in target_positions.items()
            }

        # Volatility targeting overlay
        if vol_scale != 1.0:
            target_positions = {s: w * vol_scale for s, w in target_positions.items()}

        equity = bt._total_equity()
        for symbol, target_position in target_positions.items():
            bar = bars.get(symbol)
            if bar is None:
                continue
            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0
            price = bar.get("open", bar.get("close", 0))
            target_value = equity * target_position
            target_qty = target_value / price if price > 0 else 0

            qty_diff = target_qty - current_qty

            if price > 0 and abs(qty_diff) > 0.01 * equity / price:
                if qty_diff > 0:
                    bt.submit_order(
                        symbol, OrderSide.BUY, abs(qty_diff), OrderType.MARKET
                    )
                else:
                    bt.submit_order(
                        symbol, OrderSide.SELL, abs(qty_diff), OrderType.MARKET
                    )

        # Update drawdown tracking
        equity = bt._total_equity()
        state["peak_equity"] = max(state["peak_equity"], equity)
        state["current_drawdown"] = (equity - state["peak_equity"]) / state[
            "peak_equity"
        ]

    # Run backtest
    backtester.run(data, trading_strategy)

    # Get results
    metrics = backtester.get_metrics()

    # Extended metrics and gating
    def _returns(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change().dropna()

    market_returns = None
    quant_returns = None

    try:
        if market_df is None:
            market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        market_returns = _returns(market_df)
    except Exception:
        market_returns = None

    if isinstance(quant_benchmark, list) and quant_benchmark:
        returns_list = []
        for sym in quant_benchmark:
            try:
                qdf = collector.fetch_ohlcv(sym, start_date, end_date)
                returns_list.append(_returns(qdf))
            except Exception:
                continue
        if returns_list:
            quant_returns = pd.concat(returns_list, axis=1).mean(axis=1).dropna()

    extended_metrics = compute_metrics(
        backtester.equity_curve,
        trades=backtester.trades,
        benchmark_returns=market_returns,
    )
    metrics.update(extended_metrics)

    fundamentals = []
    for symbol in symbols:
        try:
            fundamentals.append(collector.fetch_fundamentals(symbol))
        except Exception:
            continue
    metrics.update(aggregate_fundamentals(fundamentals))

    market_metrics = (
        compute_metrics_from_returns(market_returns, benchmark_returns=market_returns)
        if market_returns is not None
        else {}
    )
    quant_metrics = (
        compute_metrics_from_returns(quant_returns, benchmark_returns=market_returns)
        if quant_returns is not None
        else {}
    )

    if quant_returns is not None and not quant_returns.empty:
        strategy_returns = backtester.get_equity_series().pct_change().dropna()
        strat_aligned, quant_aligned = strategy_returns.align(quant_returns, join="inner")
        if not strat_aligned.empty:
            rel_metrics = compute_metrics_from_returns(
                strat_aligned, benchmark_returns=quant_aligned
            )
            quant_total = _total_return_from_returns(quant_aligned)
            strat_total = _total_return_from_returns(strat_aligned)
            metrics["excess_total_return_vs_quant"] = float(strat_total - quant_total)
            metrics["quant_beta"] = float(rel_metrics.get("beta", 0.0))
            metrics["quant_information_ratio"] = float(
                rel_metrics.get("information_ratio", 0.0)
            )
            md_limit = float(validation_cfg.get("constraint_max_drawdown", 0.25))
            beta_min = float(validation_cfg.get("constraint_beta_min", 0.8))
            beta_max = float(validation_cfg.get("constraint_beta_max", 1.2))
            min_info = float(validation_cfg.get("constraint_min_information_ratio", 0.0))
            constraints_pass = (
                abs(float(metrics.get("max_drawdown", 0.0))) <= md_limit
                and beta_min <= float(rel_metrics.get("beta", 0.0)) <= beta_max
                and float(rel_metrics.get("information_ratio", 0.0)) > min_info
            )
            metrics["benchmark_constraints_passed"] = bool(constraints_pass)

    try:
        market_fund = collector.fetch_fundamentals(market_benchmark)
        market_metrics.update(aggregate_fundamentals([market_fund]))
    except Exception:
        pass

    if isinstance(quant_benchmark, list) and quant_benchmark:
        quant_funds = []
        for sym in quant_benchmark:
            try:
                quant_funds.append(collector.fetch_fundamentals(sym))
            except Exception:
                continue
        if quant_funds:
            quant_metrics.update(aggregate_fundamentals(quant_funds))

    gate_details = None
    if market_returns is not None and quant_returns is not None:
        gate = evaluate_gate(metrics, market_metrics, quant_metrics)
        metrics["gate_passed"] = gate.passed
        metrics["gate_ratio_good"] = gate.ratio_good
        metrics["gate_coverage"] = gate.coverage
        metrics["gate_good_count"] = gate.good_count
        metrics["gate_available"] = gate.available
        metrics["gate_required"] = gate.required
        metrics["gate_relaxed"] = gate.relaxed
        gate_details = gate.details

    quant_firm_benchmarks = {}
    quant_firm_rows = []
    quant_bench_cfg = bench_cfg.get("quant_firm_proxies", {})
    if isinstance(quant_bench_cfg, dict) and quant_bench_cfg.get("enabled", False):
        profile_names = quant_bench_cfg.get("profiles")
        if isinstance(profile_names, str):
            profile_names = [p.strip() for p in profile_names.split(",") if p.strip()]
        strategy_returns = backtester.get_equity_series().pct_change().dropna()
        try:
            quant_firm_benchmarks = evaluate_quant_firm_benchmarks(
                strategy_returns=strategy_returns,
                collector=collector,
                start_date=start_date,
                end_date=end_date,
                profile_names=profile_names,
                interval=str(quant_bench_cfg.get("interval", "1d")),
                min_assets=int(quant_bench_cfg.get("min_assets", 2)),
            )
            quant_firm_rows = benchmark_rows(quant_firm_benchmarks)
            metrics["quant_firm_profiles_evaluated"] = len(quant_firm_rows)
            metrics["quant_firm_profiles_outperformed"] = int(
                sum(1 for r in quant_firm_rows if r["excess_total_return"] > 0)
            )
            if quant_firm_rows:
                excess_vals = np.array(
                    [float(r["excess_total_return"]) for r in quant_firm_rows],
                    dtype=float,
                )
                metrics["quant_firm_best_excess_total_return"] = float(
                    excess_vals.max()
                )
                metrics["quant_firm_median_excess_total_return"] = float(
                    np.median(excess_vals)
                )
        except Exception as exc:
            quant_firm_benchmarks = {"error": str(exc)}

    if verbose:
        print(f"\n{'=' * 60}")
        print("BACKTEST RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Return:    {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annual Return:   {metrics['annual_return'] * 100:>10.2f}%")
        print(f"Volatility:      {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:   {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:    {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Calmar Ratio:    {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:        {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Profit Factor:   {metrics['profit_factor']:>10.2f}")
        print(f"Total Trades:    {metrics['n_trades']:>10d}")
        print(f"Final Equity:    ${metrics['final_equity']:>10,.2f}")
        if "excess_total_return_vs_quant" in metrics:
            print(
                "Excess vs QQQ/IWM:"
                f" {metrics['excess_total_return_vs_quant'] * 100:>9.2f}%"
            )
            print(f"Quant Beta:      {metrics.get('quant_beta', 0.0):>10.2f}")
            print(
                "Quant InfoRatio: "
                f"{metrics.get('quant_information_ratio', 0.0):>10.2f}"
            )
            if "benchmark_constraints_passed" in metrics:
                print(
                    "Bench Constraints: "
                    f"{str(metrics['benchmark_constraints_passed']).upper():>7}"
                )
        if "gate_passed" in metrics:
            print(f"Gate Passed:     {str(metrics['gate_passed']).upper():>10}")
            print(f"Gate Coverage:   {metrics['gate_coverage']:>10d}")
            if "gate_available" in metrics and "gate_required" in metrics:
                print(f"Gate Available:  {metrics['gate_available']:>10d}")
                print(f"Gate Required:   {metrics['gate_required']:>10d}")
                if metrics.get("gate_relaxed"):
                    print(f"Gate Relaxed:    {'TRUE':>10}")
            print(f"Gate Good:       {metrics['gate_good_count']:>10d}")
            print(f"Gate Ratio:      {metrics['gate_ratio_good'] * 100:>9.2f}%")
        print(f"{'=' * 60}\n")

        # Print gate detail breakdown
        if gate_details:
            print(f"{'GATE DETAIL BREAKDOWN':^60}")
            print(f"{'Metric':<28} {'Value':>8} {'Market':>8} {'Quant':>8} {'Pass':>6}")
            print("-" * 60)
            for m, d in sorted(gate_details.items()):
                if d.get("reason") == "missing":
                    continue
                v = d.get("value")
                mk = d.get("market")
                qt = d.get("quant")
                ok = d.get("good", False)
                fmt = (
                    lambda x: f"{x:>8.4f}"
                    if x is not None and abs(x) < 100
                    else f"{x:>8.1f}"
                    if x is not None
                    else f"{'N/A':>8}"
                )
                print(
                    f"{m:<28} {fmt(v)} {fmt(mk)} {fmt(qt)} {'  YES' if ok else '   NO':>6}"
                )
            print()

        if quant_firm_rows:
            print(f"{'QUANT-FIRM PROXY SCOREBOARD':^60}")
            print(
                f"{'Profile':<30} {'Excess':>9} {'IR':>7} {'DownCap':>9} {'HitRate':>9}"
            )
            print("-" * 60)
            for row in quant_firm_rows:
                print(
                    f"{row['profile_label']:<30} "
                    f"{row['excess_total_return'] * 100:>8.2f}% "
                    f"{row['information_ratio']:>7.2f} "
                    f"{row['downside_capture']:>9.2f} "
                    f"{row['hit_rate_vs_profile'] * 100:>8.2f}%"
                )
            print()

    results = {
        "metrics": metrics,
        "gate_details": gate_details,
        "quant_firm_benchmarks": quant_firm_benchmarks,
        "quant_firm_rows": quant_firm_rows,
        "equity_curve": backtester.equity_curve,
        "trades": backtester.trades,
        "fills": backtester.fills,
    }

    # Run validation if requested
    if validate and len(data) > 0:
        if verbose:
            print("Running MCPT validation...")

        equity_series = backtester.get_equity_series()
        returns = equity_series.pct_change().dropna().values
        mcpt_p10 = float(validation_cfg.get("mcpt_threshold_stage1", 0.10))
        mcpt_p05 = float(validation_cfg.get("mcpt_threshold_stage2", 0.05))
        oos_window_days = int(validation_cfg.get("rolling_oos_window_days", 126))
        oos_min_windows = int(validation_cfg.get("rolling_oos_min_windows", 3))
        oos_min_beat_ratio = float(
            validation_cfg.get("rolling_oos_min_beat_ratio", 0.75)
        )
        promo_req_windows = int(
            validation_cfg.get("promotion_oos_required_windows", 4)
        )
        promo_req_beats = int(validation_cfg.get("promotion_oos_required_beats", 3))

        mcpt = MCPT(n_permutations=500, test_statistic="sharpe")
        mcpt_results = mcpt.run_on_returns(
            returns, show_progress=verbose, block_size=5, method="sign_flip"
        )
        p_value = float(mcpt_results.get("p_value", 1.0))
        mcpt_results["passes_stage1_0_10"] = p_value < mcpt_p10
        mcpt_results["passes_stage2_0_05"] = p_value < mcpt_p05

        results["mcpt"] = mcpt_results
        metrics["mcpt_p_value"] = p_value
        metrics["mcpt_pass_stage1_0_10"] = bool(mcpt_results["passes_stage1_0_10"])
        metrics["mcpt_pass_stage2_0_05"] = bool(mcpt_results["passes_stage2_0_05"])

        if quant_returns is not None and not quant_returns.empty:
            strategy_returns = equity_series.pct_change().dropna()
            rolling_oos = _rolling_oos_vs_benchmark(
                strategy_returns=strategy_returns,
                benchmark_returns=quant_returns,
                window_days=oos_window_days,
                min_windows=oos_min_windows,
                min_beat_ratio=oos_min_beat_ratio,
            )
            results["rolling_oos_vs_quant"] = rolling_oos
            metrics["rolling_oos_n_windows"] = int(rolling_oos.get("n_windows", 0))
            metrics["rolling_oos_beats"] = int(rolling_oos.get("beats", 0))
            metrics["rolling_oos_pass"] = bool(rolling_oos.get("passed", False))
            metrics["rolling_oos_required_beats"] = int(
                rolling_oos.get("required_beats", 0)
            )
            metrics["rolling_oos_min_beat_ratio"] = float(
                rolling_oos.get("min_beat_ratio", oos_min_beat_ratio)
            )
            metrics["promotion_oos_required_windows"] = int(promo_req_windows)
            metrics["promotion_oos_required_beats"] = int(promo_req_beats)

            metrics["promotion_ready"] = bool(
                metrics.get("mcpt_pass_stage2_0_05", False)
                and metrics.get("benchmark_constraints_passed", False)
                and metrics["rolling_oos_n_windows"] >= promo_req_windows
                and metrics["rolling_oos_beats"] >= promo_req_beats
            )
        else:
            metrics["promotion_ready"] = False

        if verbose:
            print(f"\nMCPT Results:")
            print(f"  P-Value: {mcpt_results['p_value']:.4f}")
            print(f"  Significant: {'YES' if mcpt_results['is_significant'] else 'NO'}")
            print(f"  Percentile: {mcpt_results['percentile']:.1f}%")
            print(
                f"  Stage1 (p<{mcpt_p10:.2f}): "
                f"{'PASS' if mcpt_results['passes_stage1_0_10'] else 'FAIL'}"
            )
            print(
                f"  Stage2 (p<{mcpt_p05:.2f}): "
                f"{'PASS' if mcpt_results['passes_stage2_0_05'] else 'FAIL'}"
            )
            if "rolling_oos_vs_quant" in results:
                oos = results["rolling_oos_vs_quant"]
                print("\nRolling OOS vs QQQ/IWM:")
                print(f"  Windows: {oos.get('n_windows', 0)}")
                print(f"  Beats:   {oos.get('beats', 0)}")
                print(f"  Needed:  {oos.get('required_beats', 0)}")
                print(f"  Passed:  {'YES' if oos.get('passed', False) else 'NO'}")
                print(
                    "  Promotion (>=3/4 + constraints + MCPT stage2): "
                    f"{'YES' if metrics.get('promotion_ready', False) else 'NO'}"
                )

    return results


def run_paper(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "enhanced",
    paper_bars: int = 120,
    strategy_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run a paper trading simulation on the most recent bars.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("QUANTUM ALPHA V1 - PAPER TRADING")
        print(f"{'=' * 60}")
        print(f"Symbols: {_format_symbols(symbols)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Capital: ${initial_capital:,.0f}")
        print(f"Strategy: {strategy_type}")
        print(f"Paper Bars: {paper_bars}")
        print(f"{'=' * 60}\n")

    settings = load_config(None)
    risk_cfg = settings.get("risk", {}) if settings else {}
    strategy_cfg = settings.get("strategy", {}) if settings else {}
    validation_cfg = settings.get("validation", {}) if settings else {}
    config_dir = _resolve_config_dir(None)
    strategies_cfg = _load_optional_yaml(config_dir / "strategies.yaml")
    sentiment_cfg = {}
    if strategies_cfg and "strategies" in strategies_cfg:
        sentiment_cfg = strategies_cfg["strategies"].get("sentiment", {})
    max_position = float(risk_cfg.get("max_position_size", 0.25))
    max_leverage = float(risk_cfg.get("max_portfolio_leverage", 1.0))
    max_drawdown = float(risk_cfg.get("max_drawdown", 0.10))
    kelly_fraction = float(risk_cfg.get("kelly_fraction", 0.5))
    rebalance_frequency = str(strategy_cfg.get("rebalance_frequency", "daily"))
    momentum_top_pct = float(strategy_cfg.get("momentum_top_pct", 80))
    momentum_bottom_pct = float(strategy_cfg.get("momentum_bottom_pct", 35))
    signal_threshold = float(strategy_cfg.get("signal_threshold", 0.3))
    signal_scale = float(strategy_cfg.get("signal_scale", 1.0))
    min_long_signal = float(strategy_cfg.get("min_long_signal", 0.0))
    market_off_scale = float(strategy_cfg.get("market_off_scale", 1.0))
    rs_top_n = int(strategy_cfg.get("relative_strength_top_n", 0))
    rs_min_mom = float(strategy_cfg.get("relative_strength_min_mom", 0.0))
    use_relative_strength = bool(strategy_cfg.get("use_relative_strength", False))
    liquid_subset_size = int(strategy_cfg.get("liquid_subset_size", 0))
    liquid_adv_window = int(strategy_cfg.get("liquid_adv_window", 20))
    liquid_min_history = int(strategy_cfg.get("liquid_min_history", 120))
    liquid_for_full_only = bool(strategy_cfg.get("liquid_for_full_only", True))
    long_only = bool(strategy_cfg.get("long_only", False))
    risk_off_cash = bool(strategy_cfg.get("risk_off_cash", False))
    target_vol = float(risk_cfg.get("target_volatility", 0.15))
    dd_warning = float(risk_cfg.get("dd_warning", 0.15))
    dd_critical = float(risk_cfg.get("dd_critical", 0.25))
    dd_breaker = float(risk_cfg.get("dd_circuit_breaker", 0.40))
    dd_limit = float(risk_cfg.get("dd_max_limit", 0.50))
    dd_min_exposure = float(risk_cfg.get("dd_min_exposure", 0.25))
    dd_leverage_cap_warning = float(risk_cfg.get("dd_leverage_cap_warning", 0.85))
    dd_leverage_cap_critical = float(risk_cfg.get("dd_leverage_cap_critical", 0.8))
    market_off_leverage_cap = float(risk_cfg.get("market_off_leverage_cap", 0.85))
    if max_leverage <= 0:
        max_leverage = 1.0
    min_paper_bars = int(validation_cfg.get("paper_min_bars", 120))
    if paper_bars < min_paper_bars:
        if verbose:
            print(
                f"paper-bars {paper_bars} is below validation minimum "
                f"{min_paper_bars}; using {min_paper_bars}"
            )
        paper_bars = min_paper_bars

    collector = DataCollector()
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    full_token_requested = bool(
        symbols
        and len(symbols) == 1
        and str(symbols[0]).upper() in {"FULL", "ALL", "UNIVERSE"}
    )
    symbols = _resolve_symbols(symbols, collector, settings)

    if strategy_type == "momentum":
        strategy = MomentumStrategy()
    elif strategy_type == "composite":
        strategy = CompositeStrategy()
    elif strategy_type in ("adaptive", "enhanced"):
        strategy = EnhancedCompositeStrategy()
    elif strategy_type == "sentiment":
        from quantum_alpha.strategy.sentiment_strategies import SocialSentimentStrategy

        strategy = SocialSentimentStrategy()
    elif strategy_type == "ml":
        from quantum_alpha.strategy.ml_strategies import MLTradingStrategy

        strategy = MLTradingStrategy(**(strategy_kwargs or {}))
    elif strategy_type == "news_lstm":
        from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy

        strategy = NewsLSTMStrategy(**(strategy_kwargs or {}))
    elif strategy_type == "meta_ensemble":
        from quantum_alpha.strategy.meta_ensemble_strategy import MetaEnsembleStrategy

        strategy = MetaEnsembleStrategy(**(strategy_kwargs or {}))
    else:
        strategy = MomentumStrategy()

    use_enhanced = isinstance(strategy, EnhancedCompositeStrategy)
    use_sentiment = strategy_type == "sentiment"

    if use_sentiment:
        from quantum_alpha.data.collectors.alternative import (
            load_social_sentiment,
            load_options_sentiment,
            load_insider_trades,
            load_congress_trades,
        )
        from quantum_alpha.strategy.sentiment_strategies import (
            SocialSentimentStrategy,
            OptionsSentimentStrategy,
            InsiderTradingStrategy,
            CongressTradingStrategy,
        )

        social_strategy = SocialSentimentStrategy(**sentiment_cfg.get("social", {}))
        options_strategy = OptionsSentimentStrategy(**sentiment_cfg.get("options", {}))
        insider_strategy = InsiderTradingStrategy(**sentiment_cfg.get("insider", {}))
        congress_strategy = CongressTradingStrategy(**sentiment_cfg.get("congress", {}))
        sentiment_weights = sentiment_cfg.get(
            "combined_weights",
            {"social": 0.4, "options": 0.2, "insider": 0.2, "congress": 0.2},
        )
        signal_threshold = float(
            sentiment_cfg.get("signal_threshold", signal_threshold)
        )

    position_sizer = PositionSizer(
        max_position=max_position,
        kelly_fraction=kelly_fraction,
        max_drawdown=max_drawdown,
    )
    paper_trader = PaperTrader(initial_capital=initial_capital, paper_bars=paper_bars)

    bench_cfg = settings.get("benchmarks", {})
    market_benchmark = bench_cfg.get("market", "SPY")
    market_df = None
    market_trend = None
    try:
        market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        m_close = market_df["close"]
        m_ma = m_close.rolling(200).mean()
        market_trend = m_close.shift(1) > m_ma.shift(1)
    except Exception:
        market_df = None
        market_trend = None

    if verbose:
        print("Collecting price data...")

    data = {}
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = cleaner.clean(df)
            df = imputer.impute(df)
            df = feature_gen.generate(df)
            if use_sentiment:
                social_df = load_social_sentiment(symbol, use_live=True)
                options_df = load_options_sentiment(symbol, use_live=True)
                insider_df = load_insider_trades(symbol, use_live=True)
                congress_df = load_congress_trades(symbol, use_live=True)

                if "symbol" in social_df.columns:
                    social_df = social_df[social_df["symbol"] == symbol]
                if "symbol" in options_df.columns:
                    options_df = options_df[options_df["symbol"] == symbol]
                if "symbol" in insider_df.columns:
                    insider_df = insider_df[insider_df["symbol"] == symbol]
                if "symbol" in congress_df.columns:
                    congress_df = congress_df[congress_df["symbol"] == symbol]

                frames = {}
                if not social_df.empty:
                    frames["social"] = _align_signal_frame(
                        social_strategy.generate_signals(social_df), df.index
                    )
                if not options_df.empty:
                    if "timestamp" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["timestamp"] = pd.to_datetime(
                            options_df["timestamp"]
                        )
                        options_df = options_df.set_index("timestamp")
                    elif "date" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["date"] = pd.to_datetime(options_df["date"])
                        options_df = options_df.set_index("date")
                    options_sig = options_strategy.generate_signals(options_df)
                    frames["options"] = _align_signal_frame(options_sig, df.index)
                if not insider_df.empty:
                    frames["insider"] = _align_signal_frame(
                        insider_strategy.generate_signals(insider_df),
                        df.index,
                        limit=20,
                    )
                if not congress_df.empty:
                    frames["congress"] = _align_signal_frame(
                        congress_strategy.generate_signals(congress_df),
                        df.index,
                        limit=20,
                    )

                combined_signal = pd.Series(0.0, index=df.index)
                combined_conf = pd.Series(0.0, index=df.index)
                total_w = 0.0
                for key, sig_frame in frames.items():
                    weight = float(sentiment_weights.get(key, 0.0))
                    if weight <= 0:
                        continue
                    total_w += weight
                    combined_signal += weight * sig_frame["signal"]
                    combined_conf += weight * sig_frame["signal_confidence"]

                if total_w > 0:
                    combined_signal = combined_signal / total_w
                    combined_conf = combined_conf / total_w
                else:
                    combined_signal = combined_signal * 0.0
                    combined_conf = combined_conf * 0.0

                df["signal"] = combined_signal
                df["signal_confidence"] = combined_conf
                df["position_signal"] = np.where(
                    np.abs(df["signal"]) >= signal_threshold,
                    np.sign(df["signal"]),
                    0.0,
                )
                df = _apply_signal_lag(df)
                data[symbol] = df
            else:
                df = strategy.generate_signals(df)
                df = _apply_signal_lag(df)
                data[symbol] = df
            if verbose:
                print(f"  {symbol}: {len(df)} bars")
        except Exception as e:
            if verbose:
                print(f"  {symbol}: FAILED - {e}")

    if liquid_subset_size > 0 and len(data) > liquid_subset_size:
        if full_token_requested or not liquid_for_full_only:
            selected = _select_liquid_subset(
                data,
                subset_size=liquid_subset_size,
                adv_window=liquid_adv_window,
                min_history=liquid_min_history,
            )
            selected_set = set(selected)
            data = {sym: df for sym, df in data.items() if sym in selected_set}
            if verbose:
                print(
                    "  Liquidity filter applied: "
                    f"trading top {len(selected)} symbols by {liquid_adv_window}D ADV"
                )

    if not data:
        return {"error": "No data collected"}

    if verbose:
        print(f"\nRunning paper trading simulation...")

    state = {
        "positions": {},
        "trade_history": [],
        "current_drawdown": 0,
        "peak_equity": initial_capital,
        "last_rebalance": None,
    }
    dd_controller = DrawdownController(
        warning_threshold=dd_warning,
        critical_threshold=dd_critical,
        circuit_breaker_threshold=dd_breaker,
        max_drawdown_limit=dd_limit,
        scaling_method="linear",
        min_exposure=dd_min_exposure,
        cooldown_days=10,
    )
    dd_controller.reset(initial_capital)

    def trading_strategy(timestamp, bars, bt):
        equity = bt._total_equity()
        dd_metrics = dd_controller.update(equity, timestamp)
        exposure_mult = dd_metrics.exposure_multiplier

        if not _should_rebalance(
            timestamp, state["last_rebalance"], rebalance_frequency
        ):
            state["peak_equity"] = max(state["peak_equity"], equity)
            state["current_drawdown"] = (equity - state["peak_equity"]) / state[
                "peak_equity"
            ]
            return

        state["last_rebalance"] = timestamp
        target_positions = {}
        volatilities = {}
        trade_history = (
            np.array([t["pnl"] for t in bt.trades]) if bt.trades else np.array([0])
        )
        realized_vol = _realized_vol_from_equity(bt.equity_curve)
        vol_scale = 1.0
        if realized_vol and realized_vol > 0:
            vol_scale = float(np.clip(target_vol / realized_vol, 0.5, 1.5))
        market_risk_on = True
        if market_trend is not None and timestamp in market_trend.index:
            market_risk_on = bool(market_trend.loc[timestamp])
        allow_shorts = not long_only
        if not market_risk_on:
            allow_shorts = False

        mom_scores = {}
        for sym, df in data.items():
            if timestamp in df.index:
                mom_val = df.loc[timestamp].get("mom_12m", 0.0)
                if pd.notna(mom_val):
                    mom_scores[sym] = float(mom_val)
        top_cut = None
        bottom_cut = None
        top_syms = None
        if len(mom_scores) >= 3:
            vals = np.array(list(mom_scores.values()), dtype=float)
            top_cut = np.nanpercentile(vals, momentum_top_pct)
            bottom_cut = np.nanpercentile(vals, momentum_bottom_pct)
            if use_relative_strength and rs_top_n > 0:
                ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                top_syms = {sym for sym, val in ranked[:rs_top_n] if val >= rs_min_mom}
        use_rp_allocation = top_syms is not None
        for symbol, bar in bars.items():
            if symbol not in data:
                continue

            df = data[symbol]
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            signal = (
                row.get("position_signal")
                if "position_signal" in row
                else row.get("signal", 0)
            )
            confidence = float(row.get("signal_confidence", 0.5))
            if top_syms is not None:
                signal = 1.0 if symbol in top_syms else 0.0
            else:
                if signal < 0 and not allow_shorts:
                    signal = 0.0
                if top_cut is not None and bottom_cut is not None:
                    mom_val = mom_scores.get(symbol)
                    if signal > 0 and (mom_val is None or mom_val < top_cut):
                        signal = 0.0
                    if signal < 0 and (
                        mom_val is None or mom_val > bottom_cut or long_only
                    ):
                        signal = 0.0
            if signal_scale != 1.0 and signal != 0.0:
                signal = float(np.clip(signal * signal_scale, -1.0, 1.0))
            if long_only and signal > 0 and min_long_signal > 0:
                signal = max(signal, min_long_signal)
            volatility = row.get("atr_pct", 0.02)
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.02
            volatility = volatility * np.sqrt(252)
            volatilities[symbol] = max(volatility, 1e-6)
            if use_rp_allocation:
                continue

            confidence = row.get("signal_confidence", 0.5)

            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0

            sizing = position_sizer.calculate(
                trade_history=trade_history,
                current_volatility=max(volatility, 0.01),
                current_drawdown=state["current_drawdown"],
                signal_strength=signal,
                signal_confidence=confidence,
            )

            if sizing["halt_trading"]:
                return

            target_positions[symbol] = sizing["position_size"]

        if use_rp_allocation:
            # Only allocate to symbols selected by relative strength
            selected_vols = {
                s: v
                for s, v in volatilities.items()
                if top_syms is None or s in top_syms
            }
            if selected_vols:
                inv = {s: 1 / v for s, v in selected_vols.items()}
                total_inv = sum(inv.values())
                if total_inv > 0:
                    target_positions = {
                        s: (inv[s] / total_inv) * max_leverage for s in inv
                    }
            if not target_positions:
                return
        if risk_off_cash and not market_risk_on:
            target_positions = {s: 0.0 for s in bars.keys()}
        elif not target_positions:
            return
        if not market_risk_on and market_off_scale < 1.0:
            target_positions = {
                s: w * market_off_scale for s, w in target_positions.items()
            }

        if volatilities and not use_rp_allocation:
            inv = {s: 1 / v for s, v in volatilities.items()}
            total_inv = sum(inv.values())
            if total_inv > 0:
                rp_weights = {s: inv[s] / total_inv for s in inv}
                weight_scale = len(rp_weights)
                target_positions = {
                    s: target_positions[s] * rp_weights.get(s, 0) * weight_scale
                    for s in target_positions
                }

        if vol_scale != 1.0:
            target_positions = {s: w * vol_scale for s, w in target_positions.items()}

        if exposure_mult < 1.0:
            target_positions = {
                s: w * exposure_mult for s, w in target_positions.items()
            }

        total_abs = sum(abs(w) for w in target_positions.values())
        cap = max_leverage
        if dd_metrics.state == DrawdownState.WARNING:
            cap = min(cap, dd_leverage_cap_warning)
        elif dd_metrics.state in {DrawdownState.CRITICAL, DrawdownState.RECOVERY}:
            cap = min(cap, dd_leverage_cap_critical)
        if not market_risk_on:
            cap = min(cap, market_off_leverage_cap)
        if total_abs > cap and total_abs > 0:
            scale = cap / total_abs
            target_positions = {s: w * scale for s, w in target_positions.items()}

        equity = bt._total_equity()
        for symbol, target_position in target_positions.items():
            bar = bars.get(symbol)
            if bar is None:
                continue
            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0
            price = bar.get("open", bar.get("close", 0))
            target_value = equity * target_position
            target_qty = target_value / price if price > 0 else 0

            qty_diff = target_qty - current_qty

            if price > 0 and abs(qty_diff) > 0.01 * equity / price:
                if qty_diff > 0:
                    bt.submit_order(
                        symbol, OrderSide.BUY, abs(qty_diff), OrderType.MARKET
                    )
                else:
                    bt.submit_order(
                        symbol, OrderSide.SELL, abs(qty_diff), OrderType.MARKET
                    )

        equity = bt._total_equity()
        state["peak_equity"] = max(state["peak_equity"], equity)
        state["current_drawdown"] = (equity - state["peak_equity"]) / state[
            "peak_equity"
        ]

    metrics, paper_start = paper_trader.run(data, trading_strategy)

    if verbose and "error" not in metrics:
        print(f"\n{'=' * 60}")
        print("PAPER RESULTS")
        print(f"{'=' * 60}")
        print(f"Paper Start:    {paper_start.date()}")
        print(f"Total Return:   {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annual Return:  {metrics['annual_return'] * 100:>10.2f}%")
        print(f"Volatility:     {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:  {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:   {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Calmar Ratio:   {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:       {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Profit Factor:  {metrics['profit_factor']:>10.2f}")
        print(f"Total Trades:   {metrics['n_trades']:>10d}")
        print(f"Final Equity:   ${metrics['final_equity']:>10,.2f}")
        print(f"{'=' * 60}\n")

    return {
        "metrics": metrics,
        "paper_start": paper_start,
        "equity_curve": paper_trader.backtester.equity_curve,
        "trades": paper_trader.backtester.trades,
        "fills": paper_trader.backtester.fills,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantum Alpha V1 - Algorithmic Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Operating mode",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["SPY"], help="Symbols to trade"
    )
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument(
        "--start", type=str, default=None, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        choices=[
            "momentum",
            "mean_reversion",
            "composite",
            "adaptive",
            "enhanced",
            "sentiment",
            "ml",
            "news_lstm",
            "meta_ensemble",
        ],
        default="enhanced",
        help="Strategy type",
    )
    parser.add_argument(
        "--firm-mode",
        action="store_true",
        help="Enable firm-grade execution safeguards",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch local dashboard (if available)",
    )
    parser.add_argument(
        "--paper-bars",
        type=int,
        default=120,
        help="Number of most recent bars to simulate in paper mode",
    )
    parser.add_argument("--validate", action="store_true", help="Run MCPT validation")
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    try:
        settings = load_config(args.config)
    except Exception as e:
        print(f"Config error: {e}")
        return None

    load_plugins()

    config_dir = _resolve_config_dir(args.config)
    log_cfg = settings.get("logging", {}) if settings else {}
    configure_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "quantum_alpha.log"),
    )
    thresholds = {}
    risk_cfg = _load_optional_yaml(config_dir / "risk_limits.yaml")
    if risk_cfg and "limits" in risk_cfg:
        limits = risk_cfg["limits"]
        if "max_drawdown" in limits:
            thresholds["max_drawdown"] = limits["max_drawdown"]
        if "min_sharpe" in limits:
            thresholds["min_sharpe"] = limits["min_sharpe"]
        if "min_win_rate" in limits:
            thresholds["min_win_rate"] = limits["min_win_rate"]

    alert_manager = AlertManager()
    for rule in build_default_rules(thresholds):
        alert_manager.add_rule(rule)

    if args.firm_mode and args.mode != "live":
        print(
            "Firm mode requested, but live execution is not enabled. Firm mode disabled."
        )
        args.firm_mode = False

    if args.firm_mode and args.mode == "live":
        print(
            "Firm mode requested, but live execution is not implemented in this phase."
        )
        return None

    if args.dashboard:
        print("Dashboard flag enabled. Local dashboard is not implemented in V1.")

    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=365 * 2)  # 2 years default

    if args.mode == "backtest":
        results = run_backtest(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            strategy_type=args.strategy,
            validate=args.validate,
            verbose=True,
            config_path=args.config,
        )
        if isinstance(results, dict) and "metrics" in results:
            alert_manager.evaluate(results["metrics"])
        return results

    elif args.mode == "paper":
        results = run_paper(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            strategy_type=args.strategy,
            paper_bars=args.paper_bars,
            verbose=True,
        )
        if isinstance(results, dict) and "metrics" in results:
            alert_manager.evaluate(results["metrics"])
        return results

    elif args.mode == "live":
        print("Live trading requires additional safety checks")
        print("Coming in Phase 2...")
        return None


if __name__ == "__main__":
    main()
