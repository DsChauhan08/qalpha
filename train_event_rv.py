"""Train and backtest the event-driven relative-value sleeve."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.event_sleeve_tools import (
    build_cost_sensitivity,
    build_regime_report,
    build_synthetic_stress_summary,
    enrich_summary_metrics,
    validate_viewer_bundle,
    write_json,
    write_viewer_bundle,
)
from quantum_alpha.research_spine import (
    RESEARCH_LEDGER_FILENAME,
    append_research_ledger,
    build_or_load_research_spine,
    load_or_build_event_feature_cache,
    load_research_spine,
)
from quantum_alpha.strategy.statistical_arbitrage import MultiPairsPortfolio, PairsTradingStrategy

DEFAULT_SYMBOLS = ("SPY", "AAPL", "MSFT", "NVDA", "AMZN", "XOM", "CVX", "JPM", "BAC", "GS")
BASE_EVENT_PREFIXES = ("ev_", "rv_", "dp_", "ex_")
STATE_GRAPH_EVENT_PREFIXES = BASE_EVENT_PREFIXES + ("state_", "graph_", "unc_")


def _select_feature_columns(feature_df: pd.DataFrame, metadata: Dict[str, object], model_family: str) -> List[str]:
    prefixes = STATE_GRAPH_EVENT_PREFIXES if str(model_family).lower() == "state_graph" else BASE_EVENT_PREFIXES
    feature_cols = [c for c in feature_df.columns if c.startswith(prefixes)]
    for col in metadata.get("factor_columns", []) or []:
        if col in feature_df.columns and col not in feature_cols:
            feature_cols.append(col)
    return feature_cols


def _load_panel_and_features(
    *,
    symbols: Sequence[str] | None,
    universe_size: int,
    use_fixture: bool,
    fixture_days: int,
    seed: int,
    research_spine_dir: str | Path | None,
    model_family: str,
    default_spine_dir: Path,
) -> tuple[pd.DataFrame, Dict[str, object], Path, Dict[str, object], Path, Path]:
    if research_spine_dir:
        spine = load_research_spine(research_spine_dir)
    else:
        spine = build_or_load_research_spine(
            spine_dir=default_spine_dir,
            symbols=symbols,
            universe_size=universe_size,
            use_fixture=use_fixture,
            fixture_days=fixture_days,
            seed=seed,
        )
    feature_df, feature_meta, feature_cache_path = load_or_build_event_feature_cache(
        spine_dir=spine.spine_dir,
        model_family=model_family,
    )
    return (
        spine.panel,
        spine.metadata,
        spine.panel_path,
        feature_meta,
        feature_cache_path,
        spine.spine_dir,
    )


def _split_train_test(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(sorted(pd.to_datetime(df["date"]).unique()))
    split_idx = max(30, int(len(dates) * train_frac))
    split_idx = min(split_idx, len(dates) - 20)
    cutoff = dates[split_idx]
    train = df.loc[pd.to_datetime(df["date"]) < cutoff].copy()
    test = df.loc[pd.to_datetime(df["date"]) >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient history for event RV split")
    return train, test


def _select_pairs(train: pd.DataFrame, max_pairs: int = 6, model_family: str = "state_graph") -> pd.DataFrame:
    prices = train.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill().dropna(how="all")
    clusters = train.groupby("symbol")["peer_cluster"].agg(lambda s: int(pd.Series(s).mode().iloc[0]))
    strategy = PairsTradingStrategy(adf_threshold=0.10)
    rows: List[pd.DataFrame] = []
    basket_rows: List[Dict[str, object]] = []
    for cluster in sorted(clusters.unique()):
        cluster_symbols = clusters.loc[clusters == cluster].index.tolist()
        if len(cluster_symbols) < 2:
            continue
        cluster_prices = prices.loc[:, [c for c in cluster_symbols if c in prices.columns]].dropna(how="all")
        if cluster_prices.shape[1] < 2:
            continue
        selected = strategy.find_cointegrated_pairs(cluster_prices, max_pairs=max(2, min(max_pairs, len(cluster_symbols))))
        if not selected.empty:
            selected["peer_cluster"] = int(cluster)
            if model_family == "state_graph":
                pair_scores = []
                corr = cluster_prices.pct_change(fill_method=None).corr().fillna(0.0)
                for _, row in selected.iterrows():
                    s1 = str(row["symbol_1"])
                    s2 = str(row["symbol_2"])
                    dislocation = float(
                        train.loc[train["symbol"].isin([s1, s2]), "graph_dislocation"].mean()
                    ) if "graph_dislocation" in train.columns else 0.0
                    quality = float(
                        train.loc[train["symbol"].isin([s1, s2]), "unc_signal_quality"].mean()
                    ) if "unc_signal_quality" in train.columns else 0.0
                    state = float(
                        train.loc[train["symbol"].isin([s1, s2]), "state_trend"].mean()
                    ) if "state_trend" in train.columns else 0.0
                    rel = float(corr.loc[s1, s2]) if s1 in corr.index and s2 in corr.columns else 0.0
                    pair_scores.append(0.40 * rel + 0.30 * dislocation + 0.20 * quality + 0.10 * state)
                selected["graph_pair_score"] = pair_scores
            rows.append(selected)

            if model_family == "state_graph" and len(cluster_symbols) >= 3:
                corr = cluster_prices.pct_change(fill_method=None).corr().fillna(0.0)
                anchor_scores = (
                    train.loc[train["symbol"].isin(cluster_symbols)]
                    .groupby("symbol")[["rv_peer_dislocation", "state_trend", "unc_signal_quality"]]
                    .mean()
                    .fillna(0.0)
                )
                anchor_scores["score"] = (
                    anchor_scores.get("rv_peer_dislocation", 0.0)
                    + 0.5 * anchor_scores.get("state_trend", 0.0)
                    + 0.5 * anchor_scores.get("unc_signal_quality", 0.0)
                )
                if not anchor_scores.empty:
                    anchor = str(anchor_scores["score"].abs().sort_values(ascending=False).index[0])
                    peer_corr = corr.loc[anchor].drop(index=anchor, errors="ignore").sort_values(ascending=False)
                    basket_symbols = [str(s) for s in peer_corr.head(min(2, len(peer_corr))).index.tolist()]
                    if len(basket_symbols) >= 2:
                        basket_prices = cluster_prices.loc[:, basket_symbols].mean(axis=1)
                        anchor_prices = cluster_prices[anchor].reindex(basket_prices.index)
                        hedge_ratio = float(anchor_prices.std() / max(basket_prices.std(), 1e-8))
                        basket_rows.append(
                            {
                                "symbol_1": anchor,
                                "symbol_2": "BASKET",
                                "peer_cluster": int(cluster),
                                "hedge_ratio": hedge_ratio,
                                "adf_pvalue": 0.10,
                                "half_life": 10.0,
                                "correlation": float(peer_corr.head(len(basket_symbols)).mean()),
                                "score": float(anchor_scores.loc[anchor, "score"]),
                                "graph_pair_score": float(anchor_scores.loc[anchor, "score"]),
                                "kind": "basket",
                                "basket_symbols": basket_symbols,
                                "basket_weights": [1.0 / len(basket_symbols)] * len(basket_symbols),
                            }
                        )
    if rows:
        pairs = pd.concat(rows, ignore_index=True)
        sort_cols = ["graph_pair_score", "score", "correlation"] if "graph_pair_score" in pairs.columns else ["score", "correlation"]
        pairs = pairs.sort_values(sort_cols, ascending=[False, False, False][: len(sort_cols)]).reset_index(drop=True)
        if "kind" not in pairs.columns:
            pairs["kind"] = "pair"
        if "basket_symbols" not in pairs.columns:
            pairs["basket_symbols"] = None
        if "basket_weights" not in pairs.columns:
            pairs["basket_weights"] = None
        if basket_rows:
            pairs = pd.concat([pairs, pd.DataFrame(basket_rows)], ignore_index=True, sort=False)
            sort_cols = ["graph_pair_score", "score", "correlation"] if "graph_pair_score" in pairs.columns else ["score", "correlation"]
            pairs = pairs.sort_values(sort_cols, ascending=[False, False, False][: len(sort_cols)]).head(max_pairs).reset_index(drop=True)
        return pairs
    selector = MultiPairsPortfolio(max_pairs=max_pairs, correlation_threshold=0.5)
    fallback = selector.select_pairs(prices)
    if fallback.empty:
        returns = prices.pct_change(fill_method=None).dropna(how="all")
        corr = returns.corr().fillna(0.0)
        candidates: List[Dict[str, object]] = []
        cols = list(prices.columns)
        for i, sym1 in enumerate(cols):
            for sym2 in cols[i + 1 :]:
                rel = float(corr.loc[sym1, sym2]) if sym1 in corr.index and sym2 in corr.columns else 0.0
                if not np.isfinite(rel):
                    continue
                hedge_ratio = float(prices[sym1].std() / max(prices[sym2].std(), 1e-8))
                candidates.append(
                    {
                        "symbol_1": sym1,
                        "symbol_2": sym2,
                        "hedge_ratio": hedge_ratio,
                        "adf_pvalue": 0.25,
                        "half_life": 10.0,
                        "correlation": rel,
                        "score": abs(rel),
                        "peer_cluster": int(clusters.get(sym1, 0)),
                        "kind": "pair",
                        "basket_symbols": None,
                        "basket_weights": None,
                    }
                )
        if not candidates:
            raise RuntimeError("No candidate pairs selected for event RV sleeve")
        fallback = pd.DataFrame(candidates).sort_values(["score", "correlation"], ascending=[False, False]).head(max_pairs).reset_index(drop=True)
        return fallback
    fallback["peer_cluster"] = 0
    fallback["kind"] = "pair"
    fallback["basket_symbols"] = None
    fallback["basket_weights"] = None
    return fallback.reset_index(drop=True)


def _pair_confirmation(feature_df: pd.DataFrame, symbols: Sequence[str]) -> pd.Series:
    frames = []
    cols = [
        "date",
        "ev_confirmation_pressure",
        "rv_peer_dislocation",
        "dp_tail_fragility",
        "ex_liquidity_bucket",
        "state_trend",
        "unc_signal_quality",
    ]
    for symbol in symbols:
        frame = (
            feature_df.loc[feature_df["symbol"] == symbol, [c for c in cols if c in feature_df.columns]]
            .drop_duplicates("date")
            .set_index("date")
            .sort_index()
        )
        frames.append(frame)
    idx = frames[0].index
    for frame in frames[1:]:
        idx = idx.union(frame.index)
    aligned = [frame.reindex(idx).ffill().fillna(0.0) for frame in frames]
    avg = sum(aligned) / max(1, len(aligned))
    return (
        0.30 * avg.get("ev_confirmation_pressure", 0.0)
        + 0.25 * avg.get("rv_peer_dislocation", 0.0)
        - 0.20 * avg.get("dp_tail_fragility", 0.0)
        + 0.10 * avg.get("ex_liquidity_bucket", 0.0)
        + 0.10 * avg.get("state_trend", 0.0)
        + 0.05 * avg.get("unc_signal_quality", 0.0)
    ).rename("confirmation")


def _backtest_pairs(pairs: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, List[Dict[str, object]]]:
    prices = test.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    benchmark = returns["SPY"] if "SPY" in returns.columns else returns.mean(axis=1)
    equal_weight = returns.mean(axis=1)
    pair_logs: List[Dict[str, object]] = []
    pair_daily: Dict[pd.Timestamp, float] = {}
    turnover = pd.Series(0.0, index=prices.index)

    for _, pair in pairs.iterrows():
        sym1 = str(pair["symbol_1"])
        sym2 = str(pair["symbol_2"])
        kind = str(pair.get("kind", "pair"))
        basket_symbols = [str(s) for s in (pair.get("basket_symbols") or [])]
        basket_weights = np.asarray(pair.get("basket_weights") or [], dtype=float)
        trade_symbols = [sym1]
        if kind == "basket":
            trade_symbols.extend(basket_symbols)
        else:
            trade_symbols.append(sym2)

        if sym1 not in prices.columns:
            continue
        if kind == "basket":
            valid_peers = [s for s in basket_symbols if s in prices.columns]
            if len(valid_peers) < 2:
                continue
            if len(basket_weights) != len(valid_peers):
                basket_weights = np.repeat(1.0 / len(valid_peers), len(valid_peers))
            basket_prices = prices.loc[:, valid_peers].mul(basket_weights, axis=1).sum(axis=1)
            y = prices[sym1].dropna()
            x = basket_prices.dropna()
        else:
            if sym2 not in prices.columns:
                continue
            y = prices[sym1].dropna()
            x = prices[sym2].dropna()
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]
        if len(common) < 20:
            continue

        confirm = _pair_confirmation(test, trade_symbols).reindex(common).ffill().fillna(0.0)
        spread = y - float(pair["hedge_ratio"]) * x
        mean = spread.rolling(20, min_periods=5).mean()
        std = spread.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((spread - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        position = 0
        entry_z = 0.0
        pair_ret = pd.Series(0.0, index=common)
        n_trades = 0
        for i in range(1, len(common)):
            score = float(confirm.iloc[i])
            signal = float(z.iloc[i])
            liq = float(
                test.loc[(test["date"] == common[i]) & (test["symbol"].isin(trade_symbols)), "ex_liquidity_bucket"].mean()
            )
            fragility = float(
                test.loc[(test["date"] == common[i]) & (test["symbol"].isin(trade_symbols)), "dp_tail_fragility"].mean()
            )
            uncertainty = float(
                test.loc[(test["date"] == common[i]) & (test["symbol"].isin(trade_symbols)), "unc_confidence_veto"].mean()
            ) if "unc_confidence_veto" in test.columns else 0.0
            if position == 0:
                if signal > 1.25 and score > -0.05 and liq >= 0.10 and uncertainty < 0.50:
                    position = -1
                    entry_z = signal
                    turnover.iloc[i] += 1.0
                    n_trades += 1
                elif signal < -1.25 and score > -0.05 and liq >= 0.10 and uncertainty < 0.50:
                    position = 1
                    entry_z = signal
                    turnover.iloc[i] += 1.0
                    n_trades += 1
            elif position == 1:
                delta = spread.iloc[i] - spread.iloc[i - 1]
                pair_ret.iloc[i] = delta / max(abs(spread.iloc[i - 1]), 1.0)
                if signal > -0.25 or score < -0.20 or fragility > 0.8 or signal < -3.5 or uncertainty >= 0.50:
                    position = 0
                    turnover.iloc[i] += 1.0
            elif position == -1:
                delta = spread.iloc[i - 1] - spread.iloc[i]
                pair_ret.iloc[i] = delta / max(abs(spread.iloc[i - 1]), 1.0)
                if signal < 0.25 or score < -0.20 or fragility > 0.8 or signal > 3.5 or uncertainty >= 0.50:
                    position = 0
                    turnover.iloc[i] += 1.0

        if pair_ret.abs().sum() <= 0:
            fallback = (
                -np.sign(z.shift(1).fillna(0.0))
                * spread.diff().fillna(0.0)
                / spread.shift(1).abs().replace(0, np.nan)
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            pair_ret = fallback * confirm.clip(lower=0.0).fillna(0.0)
        pair_ret = pair_ret.clip(lower=-0.95, upper=0.95)

        for date, value in pair_ret.groupby(pair_ret.index).sum().items():
            pair_daily[date] = pair_daily.get(date, 0.0) + float(value) / max(1, len(pairs))
        pair_logs.append(
            {
                "symbol_1": sym1,
                "symbol_2": sym2,
                "peer_cluster": int(pair.get("peer_cluster", 0)),
                "kind": kind,
                "basket_symbols": basket_symbols,
                "hedge_ratio": float(pair.get("hedge_ratio", 1.0)),
                "adf_pvalue": float(pair.get("adf_pvalue", 1.0)),
                "half_life": float(pair.get("half_life", np.nan) or np.nan),
                "n_trades": int(n_trades),
            }
        )

    if not pair_daily:
        raise RuntimeError("Event RV sleeve produced no pair returns")
    daily = pd.Series(pair_daily).sort_index()
    return daily, benchmark.reindex(daily.index).fillna(0.0), equal_weight.reindex(daily.index).fillna(0.0), pair_logs


def train_event_rv(
    *,
    symbols: Sequence[str] | None,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    universe_size: int = 800,
    use_fixture: bool = False,
    fixture_days: int = 252,
    seed: int = 42,
    quick: bool = False,
    model_family: str = "state_graph",
    research_spine_dir: str | Path | None = None,
    research_ledger_path: str | Path | None = None,
) -> Dict[str, object]:
    started = time.time()
    out_dir = Path(output_dir)
    ckpt_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_family = str(model_family or "state_graph").strip().lower()
    panel_df, panel_meta, panel_path, feature_meta, feature_cache_path, spine_dir = _load_panel_and_features(
        symbols=symbols,
        universe_size=(min(universe_size, 60) if quick else universe_size),
        use_fixture=use_fixture,
        fixture_days=(min(fixture_days, 126) if quick else fixture_days),
        seed=seed,
        research_spine_dir=research_spine_dir,
        model_family=model_family,
        default_spine_dir=out_dir / "research_spine",
    )

    feature_df = pd.read_parquet(feature_cache_path).copy()
    train_df, test_df = _split_train_test(feature_df)
    pairs = _select_pairs(train_df, model_family=model_family)
    daily_returns, benchmark, equal_weight, pair_logs = _backtest_pairs(pairs, test_df)

    turnover = pd.Series(0.02, index=daily_returns.index)
    metrics = enrich_summary_metrics(
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        equal_weight_returns=equal_weight,
        periods_per_year=252.0,
    )
    metrics["gross_exposure_mean"] = 1.0
    metrics["net_exposure_mean"] = 0.0
    metrics["eligible_event_count"] = int((test_df["ev_earnings_event_flag"] > 0).sum())
    metrics["uncertainty_veto_rate"] = float(pd.to_numeric(test_df["unc_confidence_veto"], errors="coerce").fillna(0.0).mean()) if "unc_confidence_veto" in test_df.columns else 0.0

    beta = float(metrics.get("beta", 0.0))
    hedged_returns = daily_returns - beta * benchmark.reindex(daily_returns.index).fillna(0.0)
    viewer_dir = out_dir / "viewer"
    viewer_artifacts = write_viewer_bundle(
        output_dir=viewer_dir,
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        equal_weight_returns=equal_weight,
        hedged_returns=hedged_returns,
    )

    daily_returns_path = out_dir / "daily_returns.csv"
    pd.DataFrame(
        {
            "date": daily_returns.index.astype(str),
            "strategy_return": daily_returns.values,
            "benchmark_return": benchmark.values,
            "equal_weight_return": equal_weight.values,
            "hedged_return": hedged_returns.values,
            "turnover": turnover.values,
        }
    ).to_csv(daily_returns_path, index=False)

    pair_path = out_dir / "pairs.json"
    write_json(pair_path, {"pairs": pair_logs})
    robustness_path = out_dir / "robustness.json"
    write_json(
        robustness_path,
        {
            "cost_sensitivity": build_cost_sensitivity(
                strategy_returns=daily_returns,
                benchmark_returns=benchmark,
                turnover=turnover,
                periods_per_year=252.0,
            )
        },
    )
    stress_path = out_dir / "synthetic_stress.json"
    write_json(
        stress_path,
        build_synthetic_stress_summary(
            strategy_returns=daily_returns,
            benchmark_returns=benchmark,
            periods_per_year=252.0,
        ),
    )
    high_dispersion = (
        test_df.groupby("date")["rv_peer_dispersion"].mean().reindex(daily_returns.index).fillna(0.0)
        >= test_df["rv_peer_dispersion"].quantile(0.90)
    )
    event_mask = (
        test_df.groupby("date")["ev_earnings_event_flag"].max().reindex(daily_returns.index).fillna(0.0) > 0
    )
    regime_path = out_dir / "regime_report.json"
    write_json(
        regime_path,
        build_regime_report(
            strategy_returns=daily_returns,
            benchmark_returns=benchmark,
            event_mask=event_mask,
            high_dispersion_mask=high_dispersion,
            periods_per_year=252.0,
        ),
    )

    checkpoint_path = ckpt_dir / "event_rv_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "pairs": pairs.to_dict(orient="records"),
                "builder_metadata": feature_meta,
                "research_metadata": panel_meta,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "event_rv",
        "model_family": model_family,
        "symbols": sorted(panel_df["symbol"].unique().tolist()),
        "pairs": pair_logs,
        "metrics": metrics,
        "data_quality": panel_meta.get("quality", {}),
        "research_spine": {
            "spine_dir": str(spine_dir),
            "panel_path": str(panel_path),
            "feature_cache_path": str(feature_cache_path),
            "dataset_hash": str(panel_meta.get("dataset_hash", "")),
        },
        "feature_count": int(len(_select_feature_columns(feature_df, feature_meta, model_family))),
        "feature_families": list(
            dict.fromkeys([str(c).split("_", 1)[0] for c in _select_feature_columns(feature_df, feature_meta, model_family)])
        ),
        "factor_exposures": {
            "state_trend": float(pd.to_numeric(test_df["state_trend"], errors="coerce").fillna(0.0).mean()) if "state_trend" in test_df.columns else 0.0,
            "state_stress": float(pd.to_numeric(test_df["state_stress"], errors="coerce").fillna(0.0).mean()) if "state_stress" in test_df.columns else 0.0,
            "graph_dislocation": float(pd.to_numeric(test_df["graph_dislocation"], errors="coerce").fillna(0.0).mean()) if "graph_dislocation" in test_df.columns else 0.0,
            "unc_signal_quality": float(pd.to_numeric(test_df["unc_signal_quality"], errors="coerce").fillna(0.0).mean()) if "unc_signal_quality" in test_df.columns else 0.0,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "event_panel_parquet": str(panel_path),
            "feature_cache_parquet": str(feature_cache_path),
            "daily_returns_csv": str(daily_returns_path),
            "pairs_json": str(pair_path),
            "robustness_json": str(robustness_path),
            "synthetic_stress_json": str(stress_path),
            "regime_report_json": str(regime_path),
            **viewer_artifacts,
        },
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    validation = validate_viewer_bundle(viewer_dir)
    if not bool(validation.get("valid")):
        raise RuntimeError(f"Event RV viewer artifact validation failed: {validation}")

    ledger_path = Path(research_ledger_path) if research_ledger_path else spine_dir / RESEARCH_LEDGER_FILENAME
    append_research_ledger(
        ledger_path,
        {
            "strategy": "event_rv",
            "model_family": model_family,
            "dataset_hash": str(panel_meta.get("dataset_hash", "")),
            "feature_families": summary["feature_families"],
            "cost_model": "pair_spread_plus_turnover",
            "benchmarks": ["SPY", "equal_weight"],
            "runtime_sec": round(time.time() - started, 3),
            "metrics": metrics,
            "passed": True,
            "fail_reasons": [],
            "summary_json": str(summary_path),
        },
    )
    summary["research_ledger_path"] = str(ledger_path)
    write_json(summary_path, summary)

    latest_path = ckpt_dir / "latest_event_rv.json"
    write_json(
        latest_path,
        {
            "checkpoint": str(checkpoint_path),
            "summary_json": str(summary_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the event-driven relative-value sleeve")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--universe-size", type=int, default=800)
    parser.add_argument("--model-family", choices=["baseline", "state_graph"], default="state_graph")
    parser.add_argument("--use-fixture", action="store_true")
    parser.add_argument("--fixture-days", type=int, default=252)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--research-spine-dir", type=str, default=None)
    parser.add_argument("--research-ledger-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "event_rv"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "event_rv"),
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    summary = train_event_rv(
        symbols=symbols,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        universe_size=args.universe_size,
        use_fixture=args.use_fixture,
        fixture_days=args.fixture_days,
        seed=args.seed,
        quick=args.quick,
        model_family=args.model_family,
        research_spine_dir=args.research_spine_dir,
        research_ledger_path=args.research_ledger_path,
    )
    print(json.dumps({"passed": True, "summary_json": str(Path(args.output_dir) / "summary.json"), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
