"""Train and backtest the event-driven cross-sectional sleeve."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from quantum_alpha.backtesting.event_sleeve_tools import (
    build_cost_sensitivity,
    build_regime_report,
    build_synthetic_stress_summary,
    enrich_summary_metrics,
    validate_viewer_bundle,
    write_json,
    write_viewer_bundle,
)
from quantum_alpha.features.event_feature_builder import UnifiedEventFeatureBuilder
from quantum_alpha.research_spine import (
    RESEARCH_LEDGER_FILENAME,
    append_research_ledger,
    build_or_load_research_spine,
    load_or_build_event_feature_cache,
    load_research_spine,
)

DEFAULT_SYMBOLS = ("SPY", "AAPL", "MSFT", "NVDA", "AMZN", "XOM", "JPM", "LLY")
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


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["forward_return_5d"] = out.groupby("symbol")["close"].shift(-5) / out["close"] - 1.0
    out["forward_return_20d"] = out.groupby("symbol")["close"].shift(-20) / out["close"] - 1.0

    market = (
        out.loc[out["symbol"] == "SPY", ["date", "forward_return_5d", "forward_return_20d", "returns"]]
        .drop_duplicates("date")
        .rename(
            columns={
                "forward_return_5d": "market_forward_5d",
                "forward_return_20d": "market_forward_20d",
                "returns": "market_daily_return",
            }
        )
    )
    if market.empty:
        market = (
            out.groupby("date")[["forward_return_5d", "forward_return_20d", "returns"]]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "forward_return_5d": "market_forward_5d",
                    "forward_return_20d": "market_forward_20d",
                    "returns": "market_daily_return",
                }
            )
        )
    out = out.merge(market, on="date", how="left")

    out["peer_forward_5d"] = 0.0
    out["peer_forward_20d"] = 0.0
    out["equal_weight_daily_return"] = out.groupby("date")["returns"].transform("mean")
    for _, grp in out.groupby(["date", "peer_cluster"]):
        total5 = grp["forward_return_5d"].sum()
        total20 = grp["forward_return_20d"].sum()
        n = len(grp)
        if n <= 1:
            out.loc[grp.index, "peer_forward_5d"] = grp["market_forward_5d"].fillna(0.0)
            out.loc[grp.index, "peer_forward_20d"] = grp["market_forward_20d"].fillna(0.0)
        else:
            out.loc[grp.index, "peer_forward_5d"] = (total5 - grp["forward_return_5d"]) / (n - 1)
            out.loc[grp.index, "peer_forward_20d"] = (total20 - grp["forward_return_20d"]) / (n - 1)

    out["target_residual_5d"] = (
        out["forward_return_5d"]
        - 0.5 * out["market_forward_5d"].fillna(0.0)
        - 0.5 * out["peer_forward_5d"].fillna(0.0)
    )
    out["target_residual_20d"] = (
        out["forward_return_20d"]
        - 0.5 * out["market_forward_20d"].fillna(0.0)
        - 0.5 * out["peer_forward_20d"].fillna(0.0)
    )
    out["target_failure_risk"] = (
        out["target_residual_5d"].clip(upper=0).abs() + 0.5 * out["target_residual_20d"].clip(upper=0).abs()
    )
    out["target_cost"] = (
        pd.to_numeric(out["ex_expected_spread_cost"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["ex_turnover_cost"], errors="coerce").fillna(0.0)
    )
    return out


def _split_train_test(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(sorted(pd.to_datetime(df["date"]).unique()))
    split_idx = max(40, int(len(dates) * train_frac))
    split_idx = min(split_idx, len(dates) - 20)
    cutoff = dates[split_idx]
    train = df.loc[pd.to_datetime(df["date"]) < cutoff].copy()
    test = df.loc[pd.to_datetime(df["date"]) >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient history for event cross-sectional split")
    return train, test


def _fit_models(train: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, object]:
    params = dict(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=50,
        random_state=42,
    )
    X = train.loc[:, feature_cols].to_numpy(dtype=float)
    models = {
        "edge_5d": HistGradientBoostingRegressor(**params),
        "edge_20d": HistGradientBoostingRegressor(**params),
        "failure": HistGradientBoostingRegressor(**params),
        "cost": HistGradientBoostingRegressor(**params),
    }
    models["edge_5d"].fit(X, train["target_residual_5d"].to_numpy(dtype=float))
    models["edge_20d"].fit(X, train["target_residual_20d"].to_numpy(dtype=float))
    models["failure"].fit(X, train["target_failure_risk"].to_numpy(dtype=float))
    models["cost"].fit(X, train["target_cost"].to_numpy(dtype=float))
    models["feature_cols"] = list(feature_cols)
    return models


def _score_predictions(bundle: Dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    X = out.loc[:, bundle["feature_cols"]].to_numpy(dtype=float)
    out["predicted_edge_5d"] = bundle["edge_5d"].predict(X)
    out["predicted_edge_20d"] = bundle["edge_20d"].predict(X)
    out["predicted_failure_risk"] = np.clip(bundle["failure"].predict(X), 0.0, None)
    out["predicted_cost"] = np.clip(bundle["cost"].predict(X), 0.0, 0.05)
    out["expected_residual_edge"] = 0.6 * out["predicted_edge_5d"] + 0.4 * out["predicted_edge_20d"]
    out["trade_score"] = out["expected_residual_edge"] - out["predicted_cost"]
    failure_cap = float(out["predicted_failure_risk"].quantile(0.80))
    fragile = out["dp_transport_distance"] > out["dp_transport_distance"].quantile(0.90)
    top_decile = out.groupby("date")["trade_score"].transform(lambda s: s >= s.quantile(0.90))
    out["vetoed"] = (
        (out["predicted_failure_risk"] > failure_cap)
        | (out["ex_liquidity_bucket"] < 0.20)
        | (out["ex_event_blackout"] > 0.0)
        | (fragile & ~top_decile)
    ).astype(int)
    out.loc[out["vetoed"] == 1, "trade_score"] = -1e9
    return out


def _build_entry_portfolio(
    chunk: pd.DataFrame,
    *,
    gross_target: float,
    per_name_cap: float,
    per_cluster_cap: float,
) -> pd.DataFrame:
    active = chunk.loc[chunk["vetoed"] == 0].copy()
    if active.empty:
        return pd.DataFrame(columns=["symbol", "weight", "side", "peer_cluster", "realized_return", "predicted_cost"])

    longs = active.loc[active["trade_score"] > 0].copy()
    shorts = active.loc[active["trade_score"] < 0].copy()
    if longs.empty or shorts.empty:
        return pd.DataFrame(columns=["symbol", "weight", "side", "peer_cluster", "realized_return", "predicted_cost"])

    rows: List[Dict[str, object]] = []
    long_budget = gross_target / 2.0
    short_budget = gross_target / 2.0

    long_clusters = sorted(longs["peer_cluster"].unique().tolist())
    short_clusters = sorted(shorts["peer_cluster"].unique().tolist())
    long_cluster_budget = min(per_cluster_cap, long_budget / max(1, len(long_clusters)))
    short_cluster_budget = min(per_cluster_cap, short_budget / max(1, len(short_clusters)))

    for cluster in long_clusters:
        cluster_chunk = longs.loc[longs["peer_cluster"] == cluster].sort_values("trade_score", ascending=False)
        if cluster_chunk.empty:
            continue
        n = max(1, int(np.floor(long_cluster_budget / max(per_name_cap, 1e-8))))
        selected = cluster_chunk.head(n)
        weight = min(per_name_cap, long_cluster_budget / max(1, len(selected)))
        for _, row in selected.iterrows():
            rows.append(
                {
                    "symbol": row["symbol"],
                    "weight": float(weight),
                    "side": "long",
                    "peer_cluster": int(cluster),
                    "realized_return": float(row["target_residual_5d"]),
                    "predicted_cost": float(row["predicted_cost"]),
                }
            )

    for cluster in short_clusters:
        cluster_chunk = shorts.loc[shorts["peer_cluster"] == cluster].sort_values("trade_score", ascending=True)
        if cluster_chunk.empty:
            continue
        n = max(1, int(np.floor(short_cluster_budget / max(per_name_cap, 1e-8))))
        selected = cluster_chunk.head(n)
        weight = min(per_name_cap, short_cluster_budget / max(1, len(selected)))
        for _, row in selected.iterrows():
            rows.append(
                {
                    "symbol": row["symbol"],
                    "weight": float(-weight),
                    "side": "short",
                    "peer_cluster": int(cluster),
                    "realized_return": float(row["target_residual_5d"]),
                    "predicted_cost": float(row["predicted_cost"]),
                }
            )

    return pd.DataFrame(rows)


def _portfolio_backtest(
    pred: pd.DataFrame,
    *,
    hold_days: int = 5,
    gross_target: float = 2.0,
    per_name_cap: float = 0.02,
    per_cluster_cap: float = 0.15,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    unique_dates = pd.Index(sorted(pred["date"].unique()))

    daily_returns = pd.Series(0.0, index=unique_dates, dtype=float)
    turnover = pd.Series(0.0, index=unique_dates, dtype=float)
    gross_exposure = pd.Series(0.0, index=unique_dates, dtype=float)
    net_exposure = pd.Series(0.0, index=unique_dates, dtype=float)
    positions: List[Dict[str, object]] = []
    prev_weights: Dict[str, float] = {}

    for entry_idx in range(0, len(unique_dates), max(1, int(hold_days))):
        entry_date = unique_dates[entry_idx]
        exit_slice = unique_dates[entry_idx : min(len(unique_dates), entry_idx + hold_days)]
        chunk = pred.loc[pred["date"] == entry_date].copy()
        book = _build_entry_portfolio(
            chunk,
            gross_target=gross_target,
            per_name_cap=per_name_cap,
            per_cluster_cap=per_cluster_cap,
        )
        if book.empty:
            continue

        curr_weights = {str(row["symbol"]): float(row["weight"]) for _, row in book.iterrows()}
        all_symbols = sorted(set(prev_weights) | set(curr_weights))
        trade_turnover = 0.5 * sum(abs(curr_weights.get(sym, 0.0) - prev_weights.get(sym, 0.0)) for sym in all_symbols)
        turnover.loc[entry_date] = float(trade_turnover)
        prev_weights = curr_weights

        period_return = float((book["weight"] * np.sign(book["weight"]) * book["realized_return"]).sum() - (book["weight"].abs() * book["predicted_cost"]).sum())
        gross = float(book["weight"].abs().sum())
        net = float(book["weight"].sum())
        spread = period_return / max(1, len(exit_slice))
        for date in exit_slice:
            daily_returns.loc[date] += spread
            gross_exposure.loc[date] += gross
            net_exposure.loc[date] += net
            for _, row in book.iterrows():
                positions.append(
                    {
                        "entry_date": str(entry_date.date()),
                        "date": str(date.date()),
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "weight": float(row["weight"]),
                        "peer_cluster": int(row["peer_cluster"]),
                    }
                )

    benchmark = (
        pred.groupby("date")["market_daily_return"].mean().reindex(unique_dates).fillna(0.0)
    )
    equal_weight = pred.groupby("date")["equal_weight_daily_return"].mean().reindex(unique_dates).fillna(0.0)
    return daily_returns, benchmark, equal_weight, pd.DataFrame(positions), turnover, gross_exposure, net_exposure


def train_event_cross_sectional(
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

    feature_df = _add_targets(pd.read_parquet(feature_cache_path))
    feature_cols = _select_feature_columns(feature_df, feature_meta, model_family)
    feature_df = feature_df.dropna(subset=["target_residual_5d", "target_residual_20d"]).copy()
    train_df, test_df = _split_train_test(feature_df)
    model_bundle = _fit_models(train_df, feature_cols)
    scored = _score_predictions(model_bundle, test_df)
    if "unc_confidence_veto" in scored.columns:
        scored.loc[pd.to_numeric(scored["unc_confidence_veto"], errors="coerce").fillna(0.0) > 0.0, "vetoed"] = 1
        scored.loc[scored["vetoed"] == 1, "trade_score"] = -1e9

    daily_returns, benchmark, equal_weight, positions, turnover, gross_series, net_exposure = _portfolio_backtest(scored)
    if daily_returns.empty:
        raise RuntimeError("Event cross-sectional backtest produced no returns")

    metrics = enrich_summary_metrics(
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        equal_weight_returns=equal_weight,
        periods_per_year=252.0,
    )
    metrics["gross_exposure_mean"] = float(gross_series.mean())
    metrics["net_exposure_mean"] = float(net_exposure.mean())
    metrics["eligible_event_count"] = int((scored["ev_earnings_event_flag"] > 0).sum())
    unc_veto = pd.Series(scored["unc_confidence_veto"], index=scored.index) if "unc_confidence_veto" in scored.columns else pd.Series(0.0, index=scored.index)
    metrics["uncertainty_veto_rate"] = float(pd.to_numeric(unc_veto, errors="coerce").fillna(0.0).mean())
    metrics["turnover_mean"] = float(turnover.reindex(daily_returns.index).fillna(0.0).mean())

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
            "benchmark_return": benchmark.reindex(daily_returns.index).fillna(0.0).values,
            "equal_weight_return": equal_weight.reindex(daily_returns.index).fillna(0.0).values,
            "hedged_return": hedged_returns.values,
            "turnover": turnover.reindex(daily_returns.index).fillna(0.0).values,
            "gross_exposure": gross_series.values,
            "net_exposure": net_exposure.values,
        }
    ).to_csv(daily_returns_path, index=False)
    positions_path = out_dir / "positions.csv"
    positions.to_csv(positions_path, index=False)

    robustness = {
        "cost_sensitivity": build_cost_sensitivity(
            strategy_returns=daily_returns,
            benchmark_returns=benchmark,
            turnover=turnover.reindex(daily_returns.index).fillna(0.0),
            periods_per_year=252.0,
        ),
    }
    robustness_path = out_dir / "robustness.json"
    write_json(robustness_path, robustness)

    stress = build_synthetic_stress_summary(
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        periods_per_year=252.0,
    )
    stress_path = out_dir / "synthetic_stress.json"
    write_json(stress_path, stress)

    high_dispersion_mask = (
        scored.groupby("date")["rv_peer_dispersion"].mean().reindex(daily_returns.index).fillna(0.0)
        >= scored["rv_peer_dispersion"].quantile(0.90)
    )
    event_mask = (
        scored.groupby("date")["ev_earnings_event_flag"].max().reindex(daily_returns.index).fillna(0.0) > 0
    )
    regime_report = build_regime_report(
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        event_mask=event_mask,
        high_dispersion_mask=high_dispersion_mask,
        periods_per_year=252.0,
    )
    regime_path = out_dir / "regime_report.json"
    write_json(regime_path, regime_report)

    checkpoint_path = ckpt_dir / "event_cross_sectional_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "models": model_bundle,
                "feature_columns": feature_cols,
                "builder_metadata": feature_meta,
                "research_metadata": panel_meta,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "event_cross_sectional",
        "model_family": model_family,
        "symbols": sorted(panel_df["symbol"].unique().tolist()),
        "universe_size": int(panel_df["symbol"].nunique()),
        "feature_count": int(len(feature_cols)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "data_quality": panel_meta.get("quality", {}),
        "research_spine": {
            "spine_dir": str(spine_dir),
            "panel_path": str(panel_path),
            "feature_cache_path": str(feature_cache_path),
            "dataset_hash": str(panel_meta.get("dataset_hash", "")),
        },
        "feature_families": list(dict.fromkeys([str(c).split("_", 1)[0] for c in feature_cols])),
        "factor_exposures": {
            "ev_information_gap": float(scored["ev_information_gap"].mean()),
            "ev_confirmation_pressure": float(scored["ev_confirmation_pressure"].mean()),
            "rv_peer_dislocation": float(scored["rv_peer_dislocation"].mean()),
            "dp_tail_fragility": float(scored["dp_tail_fragility"].mean()),
            "state_trend": float(pd.to_numeric(scored["state_trend"], errors="coerce").fillna(0.0).mean()) if "state_trend" in scored.columns else 0.0,
            "state_stress": float(pd.to_numeric(scored["state_stress"], errors="coerce").fillna(0.0).mean()) if "state_stress" in scored.columns else 0.0,
            "graph_dislocation": float(pd.to_numeric(scored["graph_dislocation"], errors="coerce").fillna(0.0).mean()) if "graph_dislocation" in scored.columns else 0.0,
            "unc_signal_quality": float(pd.to_numeric(scored["unc_signal_quality"], errors="coerce").fillna(0.0).mean()) if "unc_signal_quality" in scored.columns else 0.0,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "event_panel_parquet": str(panel_path),
            "feature_cache_parquet": str(feature_cache_path),
            "daily_returns_csv": str(daily_returns_path),
            "positions_csv": str(positions_path),
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
        raise RuntimeError(f"Event cross-sectional viewer artifact validation failed: {validation}")

    ledger_path = Path(research_ledger_path) if research_ledger_path else spine_dir / RESEARCH_LEDGER_FILENAME
    append_research_ledger(
        ledger_path,
        {
            "strategy": "event_cross_sectional",
            "model_family": model_family,
            "dataset_hash": str(panel_meta.get("dataset_hash", "")),
            "feature_families": summary["feature_families"],
            "cost_model": "predicted_spread_plus_turnover",
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

    latest_path = ckpt_dir / "latest_event_cross_sectional.json"
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
    parser = argparse.ArgumentParser(description="Train the event-driven cross-sectional sleeve")
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
        default=str(Path(__file__).parent / "artifacts" / "event_cross_sectional"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "event_cross_sectional"),
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    summary = train_event_cross_sectional(
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
