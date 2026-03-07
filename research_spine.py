"""Shared cached research spine for parallel alpha research workflows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from quantum_alpha.data.collectors.event_panel import EventPanelBundle, build_event_panel

RESEARCH_PANEL_FILENAME = "panel.parquet"
RESEARCH_METADATA_FILENAME = "metadata.json"
RESEARCH_LEDGER_FILENAME = "research_ledger.jsonl"
FEATURE_CACHE_PREFIX = "event_features"

MICROSTRUCTURE_COLUMNS = [
    "micro_mean_spread",
    "micro_mean_ofi",
    "micro_vol_sig_slope",
    "micro_has_intraday",
]


@dataclass
class ResearchSpineBundle:
    panel: pd.DataFrame
    metadata: Dict[str, object]
    spine_dir: Path
    panel_path: Path
    metadata_path: Path
    dataset_hash: str


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _stable_hash(payload: Dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cluster_symbols_by_returns(df: pd.DataFrame, cluster_count: int | None = None) -> Dict[str, int]:
    pivot = (
        df.pivot_table(index="date", columns="symbol", values="returns", aggfunc="mean")
        .sort_index()
        .fillna(0.0)
    )
    symbols = list(pivot.columns)
    if len(symbols) <= 3:
        return {symbol: 0 for symbol in symbols}

    X = pivot.T.to_numpy(dtype=float)
    n_components = max(2, min(5, X.shape[0], X.shape[1]))
    embed = PCA(n_components=n_components, random_state=42).fit_transform(X)
    n_clusters = cluster_count or min(max(2, int(np.ceil(np.sqrt(len(symbols))))), len(symbols))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(embed)
    return {symbol: int(label) for symbol, label in zip(symbols, labels)}


def _rolling_beta(symbol_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
    out = np.zeros(len(symbol_returns), dtype=float)
    r = pd.to_numeric(symbol_returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    b = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for i in range(len(out)):
        lo = max(0, i - window + 1)
        rs = r[lo : i + 1]
        bs = b[lo : i + 1]
        if len(rs) < max(10, window // 4):
            continue
        bvar = float(np.var(bs))
        out[i] = float(np.cov(rs, bs, ddof=0)[0, 1] / bvar) if bvar > 1e-12 else 0.0
    return pd.Series(out, index=symbol_returns.index)


def _derive_research_panel(bundle: EventPanelBundle) -> pd.DataFrame:
    df = bundle.panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume", "returns", "dollar_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "returns" not in df.columns:
        df["returns"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
    df["returns"] = df["returns"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["dollar_volume"] = pd.to_numeric(df.get("dollar_volume"), errors="coerce").fillna(
        pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
    )

    cluster_map = _cluster_symbols_by_returns(df)
    df["research_peer_group"] = df["symbol"].map(cluster_map).fillna(0).astype(int)

    market_series = (
        df.loc[df["symbol"] == "SPY", ["date", "returns"]]
        .drop_duplicates("date")
        .set_index("date")["returns"]
        .sort_index()
    )
    if market_series.empty:
        market_series = df.groupby("date")["returns"].mean().sort_index()
    df["research_market_return"] = df["date"].map(market_series).fillna(0.0)

    cluster_returns = (
        df.groupby(["date", "research_peer_group"])["returns"]
        .mean()
        .rename("research_peer_return")
        .reset_index()
    )
    df = df.merge(cluster_returns, on=["date", "research_peer_group"], how="left")
    df["research_peer_return"] = df["research_peer_return"].fillna(0.0)

    df["research_peer_return_ex_self"] = 0.0
    for _, grp in df.groupby("research_peer_group"):
        cluster_sum = grp.groupby("date")["returns"].transform("sum")
        cluster_n = grp.groupby("date")["returns"].transform("count")
        peer_ex_self = np.where(
            cluster_n > 1,
            (cluster_sum - grp["returns"]) / (cluster_n - 1),
            grp["research_peer_return"],
        )
        df.loc[grp.index, "research_peer_return_ex_self"] = peer_ex_self

    df["return_5d"] = df.groupby("symbol")["close"].pct_change(periods=5, fill_method=None).fillna(0.0)
    df["return_20d"] = df.groupby("symbol")["close"].pct_change(periods=20, fill_method=None).fillna(0.0)
    market_close = (
        df.loc[df["symbol"] == "SPY", ["date", "close"]]
        .drop_duplicates("date")
        .set_index("date")["close"]
        .sort_index()
    )
    if market_close.empty:
        market_close = df.groupby("date")["close"].mean().sort_index()
    market_ret_5 = market_close.pct_change(5, fill_method=None).fillna(0.0)
    market_ret_20 = market_close.pct_change(20, fill_method=None).fillna(0.0)
    df["research_market_return_5d"] = df["date"].map(market_ret_5).fillna(0.0)
    df["research_market_return_20d"] = df["date"].map(market_ret_20).fillna(0.0)

    peer_ret_5 = (
        df.groupby(["date", "research_peer_group"])["return_5d"]
        .mean()
        .rename("peer_return_5d")
        .reset_index()
    )
    peer_ret_20 = (
        df.groupby(["date", "research_peer_group"])["return_20d"]
        .mean()
        .rename("peer_return_20d")
        .reset_index()
    )
    df = df.merge(peer_ret_5, on=["date", "research_peer_group"], how="left")
    df = df.merge(peer_ret_20, on=["date", "research_peer_group"], how="left")
    df["peer_return_5d"] = df["peer_return_5d"].fillna(0.0)
    df["peer_return_20d"] = df["peer_return_20d"].fillna(0.0)

    df["research_residual_return_1d"] = (
        df["returns"]
        - 0.5 * df["research_market_return"]
        - 0.5 * df["research_peer_return_ex_self"]
    )
    df["research_residual_return_5d"] = (
        df["return_5d"]
        - 0.5 * df["research_market_return_5d"]
        - 0.5 * df["peer_return_5d"]
    )
    df["research_residual_return_20d"] = (
        df["return_20d"]
        - 0.5 * df["research_market_return_20d"]
        - 0.5 * df["peer_return_20d"]
    )

    df["research_event_flag"] = (
        pd.to_numeric(df.get("ev_earnings_event_flag", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("ev_earnings_within_5d_raw", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("options_signal_raw", 0.0), errors="coerce").abs().fillna(0.0)
    ).clip(upper=1.0)

    quality = bundle.quality or {}
    coverage = quality.get("coverage_by_domain", {}) or {}
    staleness = quality.get("staleness_days", {}) or {}
    coverage_score = float(np.mean(list(coverage.values()))) if coverage else 0.0
    staleness_penalty = float(np.mean([min(float(v), 365.0) / 365.0 for v in staleness.values()])) if staleness else 1.0
    event_quality_score = max(0.0, coverage_score * (1.0 - 0.5 * staleness_penalty))
    df["research_event_quality_score"] = float(event_quality_score)
    df["research_event_quality_flag"] = float(
        bool(quality.get("event_lag_ok", False)) and event_quality_score >= 0.20
    )

    for col in MICROSTRUCTURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df["micro_has_intraday"] = pd.to_numeric(df["micro_has_intraday"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    market_map = df[["date", "research_market_return"]].drop_duplicates("date").set_index("date")["research_market_return"]
    peer_map = (
        df.groupby(["date", "research_peer_group"])["research_peer_return_ex_self"]
        .mean()
        .rename("peer_benchmark")
        .reset_index()
    )
    df = df.merge(peer_map, on=["date", "research_peer_group"], how="left")
    df["peer_benchmark"] = df["peer_benchmark"].fillna(0.0)
    df["research_market_beta_63d"] = 0.0
    df["research_peer_beta_63d"] = 0.0
    for _, grp in df.groupby("symbol"):
        g = grp.sort_values("date")
        market_bench = g["date"].map(market_map).fillna(0.0)
        peer_bench = g["peer_benchmark"].fillna(0.0)
        df.loc[g.index, "research_market_beta_63d"] = _rolling_beta(g["returns"], market_bench).values
        df.loc[g.index, "research_peer_beta_63d"] = _rolling_beta(g["returns"], peer_bench).values

    numeric_cols = [c for c in df.columns if c not in {"date", "symbol"}]
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def build_or_load_research_spine(
    *,
    spine_dir: str | Path,
    symbols: Sequence[str] | None = None,
    universe_size: int = 800,
    start_date: str | None = None,
    end_date: str | None = None,
    use_fixture: bool = False,
    fixture_days: int = 252,
    seed: int = 42,
) -> ResearchSpineBundle:
    spine_dir = Path(spine_dir)
    spine_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "symbols": [str(s).upper() for s in symbols] if symbols is not None else None,
        "universe_size": int(universe_size),
        "start_date": start_date,
        "end_date": end_date,
        "use_fixture": bool(use_fixture),
        "fixture_days": int(fixture_days),
        "seed": int(seed),
    }
    dataset_hash = _stable_hash(config)
    scoped_dir = spine_dir / dataset_hash
    scoped_dir.mkdir(parents=True, exist_ok=True)
    panel_path = scoped_dir / RESEARCH_PANEL_FILENAME
    metadata_path = scoped_dir / RESEARCH_METADATA_FILENAME

    if panel_path.exists() and metadata_path.exists():
        panel = pd.read_parquet(panel_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return ResearchSpineBundle(
            panel=panel,
            metadata=metadata,
            spine_dir=scoped_dir,
            panel_path=panel_path,
            metadata_path=metadata_path,
            dataset_hash=dataset_hash,
        )

    started = time.time()
    raw_bundle = build_event_panel(
        symbols=symbols,
        universe_size=universe_size,
        start_date=start_date,
        end_date=end_date,
        use_fixture=use_fixture,
        fixture_days=fixture_days,
        seed=seed,
    )
    panel = _derive_research_panel(raw_bundle)
    runtime_sec = round(time.time() - started, 3)

    metadata = {
        "dataset_hash": dataset_hash,
        "config": config,
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "date_min": str(pd.to_datetime(panel["date"]).min().date()),
        "date_max": str(pd.to_datetime(panel["date"]).max().date()),
        "columns": list(panel.columns),
        "quality": raw_bundle.quality,
        "runtime_sec": runtime_sec,
    }
    panel.to_parquet(panel_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    return ResearchSpineBundle(
        panel=panel,
        metadata=metadata,
        spine_dir=scoped_dir,
        panel_path=panel_path,
        metadata_path=metadata_path,
        dataset_hash=dataset_hash,
    )


def load_research_spine(spine_dir: str | Path) -> ResearchSpineBundle:
    scoped_dir = Path(spine_dir)
    panel_path = scoped_dir / RESEARCH_PANEL_FILENAME
    metadata_path = scoped_dir / RESEARCH_METADATA_FILENAME
    if not panel_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing research spine files under {scoped_dir}")
    panel = pd.read_parquet(panel_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_hash = str(metadata.get("dataset_hash") or scoped_dir.name)
    return ResearchSpineBundle(
        panel=panel,
        metadata=metadata,
        spine_dir=scoped_dir,
        panel_path=panel_path,
        metadata_path=metadata_path,
        dataset_hash=dataset_hash,
    )


def load_or_build_event_feature_cache(
    *,
    spine_dir: str | Path,
    model_family: str,
) -> tuple[pd.DataFrame, Dict[str, object], Path]:
    scoped_dir = Path(spine_dir)
    feature_path = scoped_dir / f"{FEATURE_CACHE_PREFIX}_{model_family}.parquet"
    metadata_path = scoped_dir / f"{FEATURE_CACHE_PREFIX}_{model_family}_metadata.json"
    if feature_path.exists() and metadata_path.exists():
        feature_df = pd.read_parquet(feature_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return feature_df, metadata, feature_path

    spine = load_research_spine(scoped_dir)
    from quantum_alpha.features.event_feature_builder import UnifiedEventFeatureBuilder

    built = UnifiedEventFeatureBuilder().build(spine.panel, model_family=model_family)
    feature_df = built.features
    metadata = dict(built.metadata)
    metadata["source_panel_path"] = str(spine.panel_path)
    feature_df.to_parquet(feature_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    return feature_df, metadata, feature_path


def append_research_ledger(ledger_path: str | Path, entry: Dict[str, object]) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(entry)
    payload.setdefault("recorded_at_utc", pd.Timestamp.utcnow().isoformat())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default) + "\n")


def load_research_ledger(ledger_path: str | Path) -> list[Dict[str, object]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


__all__ = [
    "FEATURE_CACHE_PREFIX",
    "MICROSTRUCTURE_COLUMNS",
    "RESEARCH_LEDGER_FILENAME",
    "ResearchSpineBundle",
    "append_research_ledger",
    "build_or_load_research_spine",
    "load_or_build_event_feature_cache",
    "load_research_ledger",
    "load_research_spine",
]
