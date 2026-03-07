"""Unified event-driven feature builder for cross-sectional and RV sleeves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from quantum_alpha.features.mathematical.copula_models import CopulaAnalyzer
from quantum_alpha.features.mathematical.optimal_transport import OptimalTransportAnalyzer

EVENT_PREFIXES = ("ev_", "rv_", "dp_", "ex_")
EVENT_FACTOR_COLUMNS = [
    "ev_information_gap",
    "ev_confirmation_pressure",
    "rv_peer_dislocation",
    "dp_tail_fragility",
]


@dataclass
class EventFeatureBuildResult:
    features: pd.DataFrame
    metadata: Dict[str, object]


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std()
    out = (series - mean) / std.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _safe_percentile_rank(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True).fillna(0.5)


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - float(s.mean())) / std


def _numeric_copula_code(name: str) -> float:
    mapping = {"gaussian": 0.25, "clayton": 0.50, "frank": 0.75, "gumbel": 1.00}
    return float(mapping.get(name, 0.0))


class UnifiedEventFeatureBuilder:
    def __init__(
        self,
        *,
        cluster_count: int | None = None,
        dp_window: int = 63,
        dp_step: int = 21,
    ) -> None:
        self.cluster_count = cluster_count
        self.dp_window = int(dp_window)
        self.dp_step = int(dp_step)
        self.copula = CopulaAnalyzer()
        self.transport = OptimalTransportAnalyzer(n_bins=20, epsilon=0.05)

    def build(self, panel: pd.DataFrame) -> EventFeatureBuildResult:
        df = panel.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume", "returns", "dollar_volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "returns" not in df.columns:
            df["returns"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
        df["returns"] = df["returns"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["dollar_volume"] = pd.to_numeric(df["dollar_volume"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["dollar_volume"] = df["dollar_volume"].fillna(df["close"] * df["volume"])

        cluster_map = self._cluster_symbols(df)
        df["peer_cluster"] = df["symbol"].map(cluster_map).fillna(0).astype(int)

        market_series = (
            df.loc[df["symbol"] == "SPY", ["date", "returns"]]
            .drop_duplicates("date")
            .set_index("date")["returns"]
            .sort_index()
        )
        if market_series.empty:
            market_series = df.groupby("date")["returns"].mean().sort_index()
        df["market_return"] = df["date"].map(market_series).fillna(0.0)

        cluster_returns = (
            df.groupby(["date", "peer_cluster"])["returns"]
            .mean()
            .rename("cluster_return")
            .reset_index()
        )
        df = df.merge(cluster_returns, on=["date", "peer_cluster"], how="left")
        df["cluster_return"] = df["cluster_return"].fillna(0.0)
        df["peer_return_ex_self"] = 0.0
        for cluster, grp in df.groupby("peer_cluster"):
            cluster_sum = grp.groupby("date")["returns"].transform("sum")
            cluster_n = grp.groupby("date")["returns"].transform("count")
            peer_ex_self = np.where(cluster_n > 1, (cluster_sum - grp["returns"]) / (cluster_n - 1), grp["cluster_return"])
            df.loc[grp.index, "peer_return_ex_self"] = peer_ex_self

        df["return_5d"] = df.groupby("symbol")["close"].pct_change(periods=5, fill_method=None).fillna(0.0)
        df["return_20d"] = df.groupby("symbol")["close"].pct_change(periods=20, fill_method=None).fillna(0.0)
        df["vol_21d"] = (
            df.groupby("symbol")["returns"].transform(lambda s: s.rolling(21, min_periods=5).std()).fillna(0.0)
        )
        df["vol_63d"] = (
            df.groupby("symbol")["returns"].transform(lambda s: s.rolling(63, min_periods=10).std()).fillna(0.0)
        )
        df["range_pct"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["adv_21d"] = (
            df.groupby("symbol")["dollar_volume"].transform(lambda s: s.rolling(21, min_periods=5).mean()).fillna(0.0)
        )

        self._build_ev_features(df)
        self._build_rv_features(df)
        self._build_dp_features(df)
        self._build_ex_features(df)
        self._build_composite_factors(df)

        feature_cols = [c for c in df.columns if c.startswith(EVENT_PREFIXES)]
        df.loc[:, feature_cols] = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return EventFeatureBuildResult(
            features=df,
            metadata={
                "feature_columns": feature_cols,
                "factor_columns": list(EVENT_FACTOR_COLUMNS),
                "peer_clusters": {str(k): int(v) for k, v in cluster_map.items()},
            },
        )

    def _cluster_symbols(self, df: pd.DataFrame) -> Dict[str, int]:
        pivot = df.pivot_table(index="date", columns="symbol", values="returns", aggfunc="mean").fillna(0.0)
        symbols = list(pivot.columns)
        if len(symbols) <= 3:
            return {symbol: 0 for symbol in symbols}

        X = pivot.T.to_numpy(dtype=float)
        n_components = max(2, min(5, X.shape[0], X.shape[1]))
        embed = PCA(n_components=n_components).fit_transform(X)
        n_clusters = self.cluster_count or min(max(2, int(np.ceil(np.sqrt(len(symbols))))), len(symbols))
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(embed)
        return {symbol: int(label) for symbol, label in zip(symbols, labels)}

    def _build_ev_features(self, df: pd.DataFrame) -> None:
        df["ev_surprise_percentile"] = df.groupby("date")["ev_last_earnings_surprise_pct"].transform(_safe_percentile_rank)
        df["ev_revision_breadth"] = (
            df.groupby("symbol")["ev_revision_delta_raw"].transform(lambda s: np.sign(s).rolling(126, min_periods=5).mean()).fillna(0.0)
        )
        df["ev_guidance_delta"] = pd.to_numeric(df["ev_guidance_delta_raw"], errors="coerce").fillna(0.0)
        pead_decay = np.exp(-np.clip(df["ev_pead_days_since"], 0.0, 120.0) / 20.0)
        df["ev_pead_strength"] = np.sign(df["ev_last_earnings_surprise_pct"]) * (df["ev_last_earnings_surprise_pct"].abs() / 100.0) * pead_decay
        df["ev_pead_age"] = np.clip(df["ev_pead_days_since"], 0.0, 120.0) / 120.0
        df["ev_volatility_compression"] = (1.0 - (df["vol_21d"] / df["vol_63d"].replace(0, np.nan))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["ev_options_skew"] = pd.to_numeric(df["options_iv_skew_raw"], errors="coerce").fillna(0.0)
        df["ev_put_call_extreme"] = _rolling_zscore(pd.to_numeric(df["options_pc_ratio_raw"], errors="coerce").fillna(0.0), 63)
        df["ev_unusual_options_activity"] = pd.to_numeric(df["options_volume_zscore_raw"], errors="coerce").fillna(0.0)
        df["ev_insider_intensity"] = (
            pd.to_numeric(df["insider_value_sentiment_raw"], errors="coerce").fillna(0.0)
            * (1.0 + pd.to_numeric(df["insider_cluster_score_raw"], errors="coerce").fillna(0.0))
        )
        df["ev_congress_intensity"] = (
            pd.to_numeric(df["congress_sentiment_raw"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["congress_confidence_raw"], errors="coerce").fillna(0.0)
        )
        df["ev_event_news_surprise"] = (
            pd.to_numeric(df.get("tone_zscore", 0.0), errors="coerce").fillna(0.0)
            + 0.5 * pd.to_numeric(df.get("tone_acceleration", 0.0), errors="coerce").fillna(0.0)
        )

    def _build_rv_features(self, df: pd.DataFrame) -> None:
        peer_surprise = (
            df.groupby(["date", "peer_cluster"])["ev_last_earnings_surprise_pct"]
            .transform("mean")
            .fillna(0.0)
        )
        df["rv_peer_surprise_spread"] = (df["ev_last_earnings_surprise_pct"] - peer_surprise) / 100.0
        df["rv_event_relative_basis"] = df["return_5d"] - df["peer_return_ex_self"].fillna(0.0).groupby(df["symbol"]).transform(lambda s: s.rolling(5, min_periods=1).sum())
        df["rv_peer_dispersion"] = (
            df.groupby(["date", "peer_cluster"])["returns"].transform("std").fillna(0.0)
        )
        df["rv_market_residual"] = df["returns"] - df["market_return"]
        df["rv_peer_residual"] = df["returns"] - df["peer_return_ex_self"].fillna(0.0)

        beta_proxy = []
        stability = []
        for _, grp in df.groupby("symbol"):
            market = grp["market_return"].to_numpy(dtype=float)
            ret = grp["returns"].to_numpy(dtype=float)
            peer = grp["peer_return_ex_self"].to_numpy(dtype=float)
            sym_beta = np.zeros(len(grp), dtype=float)
            sym_stability = np.zeros(len(grp), dtype=float)
            for i in range(len(grp)):
                lo = max(0, i - 63)
                r = ret[lo : i + 1]
                m = market[lo : i + 1]
                p = peer[lo : i + 1]
                if len(r) < 10:
                    continue
                mvar = float(np.var(m))
                beta = float(np.cov(r, m, ddof=0)[0, 1] / mvar) if mvar > 0 else 0.0
                sym_beta[i] = beta
                resid = r - beta * m - p
                if len(resid) >= 10:
                    resid_lag = resid[:-1]
                    resid_now = resid[1:]
                    denom = float(np.dot(resid_lag, resid_lag))
                    ar1 = float(np.dot(resid_now, resid_lag) / denom) if denom > 0 else 0.0
                    sym_stability[i] = max(0.0, 1.0 - abs(ar1))
            beta_proxy.extend(sym_beta.tolist())
            stability.extend(sym_stability.tolist())
        df["rv_market_beta_proxy"] = beta_proxy
        df["rv_cointegration_stability"] = stability

    def _build_dp_features(self, df: pd.DataFrame) -> None:
        df["dp_best_copula_code"] = 0.0
        df["dp_lower_tail_dependence"] = 0.0
        df["dp_upper_tail_dependence"] = 0.0
        df["dp_transport_distance"] = 0.0
        df["dp_crowding_score"] = 0.0

        for _, grp in df.groupby("symbol"):
            g = grp.copy()
            returns = g["returns"].to_numpy(dtype=float)
            peer = g["peer_return_ex_self"].to_numpy(dtype=float)
            tone = pd.to_numeric(g.get("tone_zscore", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            out_best = np.zeros(len(g), dtype=float)
            out_lower = np.zeros(len(g), dtype=float)
            out_upper = np.zeros(len(g), dtype=float)
            out_transport = np.zeros(len(g), dtype=float)
            out_crowding = np.zeros(len(g), dtype=float)

            last = {
                "best": 0.0,
                "lower": 0.0,
                "upper": 0.0,
                "transport": 0.0,
                "crowding": 0.0,
            }
            for i in range(len(g)):
                if i < self.dp_window:
                    out_best[i] = last["best"]
                    out_lower[i] = last["lower"]
                    out_upper[i] = last["upper"]
                    out_transport[i] = last["transport"]
                    out_crowding[i] = last["crowding"]
                    continue
                active_event = abs(float(g["ev_pead_strength"].iloc[i])) > 0.01 or abs(float(g["ev_event_news_surprise"].iloc[i])) > 0.2
                if (i % self.dp_step != 0) and not active_event:
                    out_best[i] = last["best"]
                    out_lower[i] = last["lower"]
                    out_upper[i] = last["upper"]
                    out_transport[i] = last["transport"]
                    out_crowding[i] = last["crowding"]
                    continue
                rx = returns[i - self.dp_window : i]
                ry = peer[i - self.dp_window : i]
                if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
                    out_best[i] = last["best"]
                    out_lower[i] = last["lower"]
                    out_upper[i] = last["upper"]
                else:
                    try:
                        u = self.copula.to_pseudo_observations(rx)
                        v = self.copula.to_pseudo_observations(ry)
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            tail = self.copula.tail_dependence_analysis(u, v)
                        last["best"] = _numeric_copula_code(str(tail.get("best_model", "")))
                        last["lower"] = float(tail.get("lower_tail_dependence", 0.0))
                        last["upper"] = float(tail.get("upper_tail_dependence", 0.0))
                    except Exception:
                        pass
                    out_best[i] = last["best"]
                    out_lower[i] = last["lower"]
                    out_upper[i] = last["upper"]

                prev_window = np.column_stack([returns[i - self.dp_window : i - self.dp_window // 2], tone[i - self.dp_window : i - self.dp_window // 2]])
                curr_window = np.column_stack([returns[i - self.dp_window // 2 : i], tone[i - self.dp_window // 2 : i]])
                try:
                    prev_projection = prev_window[:, 0] + 0.2 * prev_window[:, 1]
                    curr_projection = curr_window[:, 0] + 0.2 * curr_window[:, 1]
                    last["transport"] = float(self.transport.wasserstein_distance_1d(prev_projection, curr_projection))
                except Exception:
                    pass
                crowd = (
                    abs(float(g["ev_unusual_options_activity"].iloc[i]))
                    + abs(float(g["ev_insider_intensity"].iloc[i]))
                    + abs(float(g["ev_congress_intensity"].iloc[i]))
                ) / 3.0
                last["crowding"] = float(crowd)
                out_transport[i] = last["transport"]
                out_crowding[i] = last["crowding"]

            df.loc[g.index, "dp_best_copula_code"] = out_best
            df.loc[g.index, "dp_lower_tail_dependence"] = out_lower
            df.loc[g.index, "dp_upper_tail_dependence"] = out_upper
            df.loc[g.index, "dp_transport_distance"] = out_transport
            df.loc[g.index, "dp_crowding_score"] = out_crowding

    def _build_ex_features(self, df: pd.DataFrame) -> None:
        df["ex_liquidity_bucket"] = df.groupby("date")["adv_21d"].transform(_safe_percentile_rank)
        df["ex_expected_spread_cost"] = (
            0.15 * df["range_pct"].clip(lower=0.0)
            + 0.00005 * (1.0 - df["ex_liquidity_bucket"])
            + 0.00002 * df["rv_peer_dispersion"].clip(lower=0.0)
        )
        df["ex_turnover_cost"] = (
            df.groupby("symbol")["rv_event_relative_basis"].diff().abs().fillna(0.0) * 0.05
        )
        stale_tone = pd.to_numeric(df.get("tone_volatility", 0.0), errors="coerce").fillna(0.0).eq(0.0)
        blackout = (
            (df["ev_next_earnings_days_ahead"] <= 1.0)
            | (df["ex_liquidity_bucket"] < 0.10)
            | stale_tone
        )
        df["ex_event_blackout"] = blackout.astype(float)

    def _build_composite_factors(self, df: pd.DataFrame) -> None:
        date_groups = df.groupby("date")
        z = pd.DataFrame(index=df.index)
        for col in [
            "ev_pead_strength",
            "ev_surprise_percentile",
            "ev_revision_breadth",
            "ev_guidance_delta",
            "ev_event_news_surprise",
            "ev_unusual_options_activity",
            "ev_insider_intensity",
            "ev_congress_intensity",
            "rv_peer_surprise_spread",
            "rv_event_relative_basis",
            "rv_peer_dispersion",
            "dp_lower_tail_dependence",
            "dp_upper_tail_dependence",
            "dp_transport_distance",
            "dp_crowding_score",
        ]:
            z[col] = date_groups[col].transform(_cross_sectional_zscore)

        df["ev_information_gap"] = (
            0.35 * z["ev_pead_strength"]
            + 0.20 * z["ev_surprise_percentile"]
            + 0.15 * z["ev_revision_breadth"]
            + 0.10 * z["ev_guidance_delta"]
            + 0.20 * z["ev_event_news_surprise"]
        ).fillna(0.0)
        df["ev_confirmation_pressure"] = (
            0.40 * z["ev_unusual_options_activity"]
            + 0.30 * z["ev_insider_intensity"]
            + 0.30 * z["ev_congress_intensity"]
        ).fillna(0.0)
        df["rv_peer_dislocation"] = (
            0.45 * z["rv_peer_surprise_spread"]
            + 0.35 * z["rv_event_relative_basis"]
            - 0.20 * z["rv_peer_dispersion"]
        ).fillna(0.0)
        df["dp_tail_fragility"] = (
            0.35 * z["dp_lower_tail_dependence"]
            + 0.20 * z["dp_upper_tail_dependence"]
            + 0.30 * z["dp_transport_distance"]
            + 0.15 * z["dp_crowding_score"]
        ).fillna(0.0)


__all__ = [
    "EVENT_FACTOR_COLUMNS",
    "EVENT_PREFIXES",
    "EventFeatureBuildResult",
    "UnifiedEventFeatureBuilder",
]
