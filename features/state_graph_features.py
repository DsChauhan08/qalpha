"""State, graph, and uncertainty features for parallel alpha research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

STATE_GRAPH_PREFIXES = ("state_", "graph_", "unc_")
STATE_GRAPH_FACTOR_COLUMNS = [
    "state_trend",
    "state_stress",
    "graph_dislocation",
    "unc_signal_quality",
]


def _sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -12.0, 12.0)))


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std()
    out = (series - mean) / std.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - float(s.mean())) / std


def _binary_entropy(probability: pd.Series) -> pd.Series:
    p = pd.to_numeric(probability, errors="coerce").fillna(0.5).clip(1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .rolling(window, min_periods=max(5, window // 4))
        .quantile(q)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )


@dataclass
class StateGraphFeatureResult:
    features: pd.DataFrame
    metadata: Dict[str, object]


class StateGraphFeatureBuilder:
    def __init__(
        self,
        *,
        state_window: int = 63,
        long_window: int = 126,
        uncertainty_window: int = 63,
    ) -> None:
        self.state_window = int(state_window)
        self.long_window = int(long_window)
        self.uncertainty_window = int(uncertainty_window)

    def build(
        self,
        panel: pd.DataFrame,
        *,
        include_graph: bool = True,
    ) -> StateGraphFeatureResult:
        df = panel.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        else:
            raise ValueError("StateGraphFeatureBuilder requires a date column or DatetimeIndex")

        if "symbol" not in df.columns:
            df["symbol"] = "SINGLE"
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        if "returns" not in df.columns:
            df["returns"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
        df["returns"] = pd.to_numeric(df["returns"], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

        if "research_market_return" in df.columns:
            df["market_return"] = pd.to_numeric(df["research_market_return"], errors="coerce").fillna(0.0)
        else:
            market_series = (
                df.loc[df["symbol"] == "SPY", ["date", "returns"]]
                .drop_duplicates("date")
                .set_index("date")["returns"]
                .sort_index()
            )
            if market_series.empty:
                market_series = df.groupby("date")["returns"].mean().sort_index()
            df["market_return"] = df["date"].map(market_series).fillna(0.0)

        if "peer_cluster" not in df.columns:
            if "research_peer_group" in df.columns:
                df["peer_cluster"] = pd.to_numeric(df["research_peer_group"], errors="coerce").fillna(0).astype(int)
            else:
                df["peer_cluster"] = 0

        self._build_state_features(df)
        if include_graph:
            self._build_graph_features(df)
        else:
            for col in (
                "graph_residual",
                "graph_peer_shock_propagation",
                "graph_laplacian_deviation",
                "graph_local_dispersion",
                "graph_crowding_concentration",
            ):
                df[col] = 0.0
        self._build_uncertainty_features(df)
        self._build_factors(df)

        feature_cols = [c for c in df.columns if c.startswith(STATE_GRAPH_PREFIXES)] + list(STATE_GRAPH_FACTOR_COLUMNS)
        feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))
        df.loc[:, feature_cols] = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return StateGraphFeatureResult(
            features=df,
            metadata={
                "feature_columns": feature_cols,
                "factor_columns": list(STATE_GRAPH_FACTOR_COLUMNS),
            },
        )

    def _build_state_features(self, df: pd.DataFrame) -> None:
        market = (
            df[["date", "market_return"]]
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")["market_return"]
        )
        market_drift_21 = market.rolling(21, min_periods=5).mean()
        market_drift_63 = market.rolling(63, min_periods=10).mean()
        market_vol_21 = market.rolling(21, min_periods=5).std().fillna(0.0)
        market_vol_63 = market.rolling(63, min_periods=10).std().fillna(0.0)
        market_curve = (1.0 + market.fillna(0.0)).cumprod()
        market_drawdown = market_curve / market_curve.cummax() - 1.0

        market_stress_prob = pd.Series(
            _sigmoid(
                1.20 * _safe_zscore(market_vol_21, self.state_window)
                + 1.00 * _safe_zscore(market_drawdown.abs(), self.state_window)
                - 0.70 * _safe_zscore(market_drift_21, self.state_window)
            ),
            index=market.index,
        )
        market_trend_prob = pd.Series(
            _sigmoid(
                1.15 * _safe_zscore(market_drift_63, self.state_window)
                - 0.55 * _safe_zscore(market_vol_21, self.state_window)
            ),
            index=market.index,
        )
        market_transition_entropy = _binary_entropy(market_stress_prob)
        market_persistence = (
            1.0
            - market_trend_prob.diff().abs().ewm(span=10, adjust=False).mean().clip(lower=0.0, upper=1.0)
        ).fillna(1.0)

        df["state_market_trend_prob"] = df["date"].map(market_trend_prob).fillna(0.5)
        df["state_market_stress_prob"] = df["date"].map(market_stress_prob).fillna(0.5)
        df["state_market_transition_entropy"] = df["date"].map(market_transition_entropy).fillna(float(np.log(2.0)))
        df["state_market_persistence"] = df["date"].map(market_persistence).fillna(1.0)

        state_trend = np.zeros(len(df), dtype=float)
        state_stress = np.zeros(len(df), dtype=float)
        state_entropy = np.zeros(len(df), dtype=float)
        state_cond_drift = np.zeros(len(df), dtype=float)
        state_cond_vol = np.zeros(len(df), dtype=float)
        state_persistence = np.zeros(len(df), dtype=float)
        state_flag = np.zeros(len(df), dtype=float)

        for _, grp in df.groupby("symbol"):
            g = grp.sort_values("date")
            ret = g["returns"].fillna(0.0)
            residual = ret - g["market_return"].fillna(0.0)
            drift_21 = residual.rolling(21, min_periods=5).mean()
            drift_63 = residual.rolling(63, min_periods=10).mean()
            vol_21 = residual.rolling(21, min_periods=5).std().fillna(0.0)
            vol_63 = residual.rolling(63, min_periods=10).std().fillna(0.0)
            curve = (1.0 + residual.fillna(0.0)).cumprod()
            drawdown = curve / curve.cummax() - 1.0

            raw_stress = (
                1.10 * _safe_zscore(vol_21, self.state_window)
                + 0.75 * _safe_zscore(drawdown.abs(), self.state_window)
                - 0.45 * _safe_zscore(drift_21, self.state_window)
                + 0.50 * pd.Series(g["state_market_stress_prob"].to_numpy(dtype=float), index=g.index)
            )
            raw_trend = (
                1.00 * _safe_zscore(drift_63, self.state_window)
                - 0.40 * _safe_zscore(vol_21, self.state_window)
                + 0.35 * pd.Series(g["state_market_trend_prob"].to_numpy(dtype=float), index=g.index)
            )
            stress_prob = pd.Series(_sigmoid(raw_stress), index=g.index).ewm(span=8, adjust=False).mean()
            trend_prob = pd.Series(_sigmoid(raw_trend), index=g.index).ewm(span=8, adjust=False).mean()

            entropy = _binary_entropy(stress_prob)
            persistence = (
                1.0
                - trend_prob.diff().abs().ewm(span=10, adjust=False).mean().clip(lower=0.0, upper=1.0)
            ).fillna(1.0)
            cond_drift = drift_21.fillna(0.0) * trend_prob - 0.25 * vol_21.fillna(0.0) * stress_prob
            cond_vol = vol_21.fillna(0.0) * (1.0 + stress_prob)

            state_trend[g.index] = trend_prob.to_numpy(dtype=float)
            state_stress[g.index] = stress_prob.to_numpy(dtype=float)
            state_entropy[g.index] = entropy.to_numpy(dtype=float)
            state_cond_drift[g.index] = cond_drift.to_numpy(dtype=float)
            state_cond_vol[g.index] = cond_vol.to_numpy(dtype=float)
            state_persistence[g.index] = persistence.to_numpy(dtype=float)
            state_flag[g.index] = (stress_prob > 0.65).astype(float).to_numpy(dtype=float)

        df["state_trend_prob"] = state_trend
        df["state_stress_prob"] = state_stress
        df["state_transition_entropy"] = state_entropy
        df["state_conditional_drift"] = state_cond_drift
        df["state_conditional_volatility"] = state_cond_vol
        df["state_persistence"] = state_persistence
        df["state_stress_flag"] = state_flag

    def _build_graph_features(self, df: pd.DataFrame) -> None:
        if "return_5d" not in df.columns:
            df["return_5d"] = df.groupby("symbol")["close"].pct_change(5, fill_method=None).fillna(0.0)
        if "adv_21d" not in df.columns:
            df["adv_21d"] = (
                df.groupby("symbol")["dollar_volume"]
                .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(21, min_periods=5).mean())
                .fillna(0.0)
            )

        peer_mean = df.groupby(["date", "peer_cluster"])["returns"].transform("mean").fillna(0.0)
        local_dispersion = df.groupby(["date", "peer_cluster"])["returns"].transform("std").fillna(0.0)
        peer_mean_5d = df.groupby(["date", "peer_cluster"])["return_5d"].transform("mean").fillna(0.0)
        cluster_adv_rank = df.groupby(["date", "peer_cluster"])["adv_21d"].transform(
            lambda s: s.rank(pct=True).fillna(0.5)
        )

        df["_graph_peer_mean"] = peer_mean
        df["graph_residual"] = df["returns"] - peer_mean
        df["graph_local_dispersion"] = local_dispersion
        df["graph_laplacian_deviation"] = df["return_5d"] - 0.5 * (peer_mean_5d + df["market_return"].fillna(0.0))
        df["graph_crowding_concentration"] = cluster_adv_rank.fillna(0.5)
        df["graph_peer_shock_propagation"] = (
            df.groupby("symbol")["graph_residual"].shift(1).fillna(0.0)
            + df.groupby("symbol")["_graph_peer_mean"].shift(1).fillna(0.0)
        )
        df.drop(columns=["_graph_peer_mean"], inplace=True)

    def _build_uncertainty_features(self, df: pd.DataFrame) -> None:
        disagreement = np.zeros(len(df), dtype=float)
        interval_width = np.zeros(len(df), dtype=float)
        tail_miss = np.zeros(len(df), dtype=float)
        veto = np.zeros(len(df), dtype=float)

        component_cols = [
            "ev_information_gap",
            "ev_confirmation_pressure",
            "rv_peer_dislocation",
            "state_trend_prob",
            "state_stress_prob",
            "graph_residual",
        ]
        usable_component_cols = [c for c in component_cols if c in df.columns]
        if not usable_component_cols:
            usable_component_cols = ["returns", "return_5d", "state_trend_prob", "state_stress_prob"]

        for _, grp in df.groupby("symbol"):
            g = grp.sort_values("date")
            comp = g[usable_component_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            row_disagreement = comp.std(axis=1, ddof=0).fillna(0.0)
            base_signal = pd.Series(0.0, index=g.index, dtype=float)
            weights = np.linspace(1.0, 0.5, comp.shape[1])
            weights = weights / max(weights.sum(), 1e-8)
            for col, weight in zip(comp.columns, weights):
                sign = -1.0 if col in {"state_stress_prob"} else 1.0
                base_signal = base_signal + sign * weight * comp[col]

            realized_proxy = pd.to_numeric(
                g["research_residual_return_5d"] if "research_residual_return_5d" in g.columns else g.get("return_5d", g["returns"]),
                errors="coerce",
            ).fillna(0.0)
            normalized_signal = np.tanh(base_signal.astype(float))
            nonconformity = (realized_proxy - normalized_signal).abs()
            width = _rolling_quantile(nonconformity, self.uncertainty_window, 0.90) * 2.0
            miss = (
                (nonconformity > width.shift(1).replace(0.0, np.nan).fillna(width.median() or 0.0) / 2.0)
                .astype(float)
                .rolling(self.uncertainty_window, min_periods=max(5, self.uncertainty_window // 4))
                .mean()
                .fillna(0.0)
            )
            wide = width > width.rolling(self.uncertainty_window, min_periods=10).quantile(0.80).fillna(width.median() or 0.0)
            unstable = row_disagreement > row_disagreement.rolling(self.uncertainty_window, min_periods=10).quantile(0.80).fillna(row_disagreement.median() or 0.0)
            veto_flag = (wide | unstable | (miss > 0.35)).astype(float)

            disagreement[g.index] = row_disagreement.to_numpy(dtype=float)
            interval_width[g.index] = width.to_numpy(dtype=float)
            tail_miss[g.index] = miss.to_numpy(dtype=float)
            veto[g.index] = veto_flag.to_numpy(dtype=float)

        df["unc_prediction_interval_width"] = interval_width
        df["unc_tail_miss_rate"] = tail_miss
        df["unc_model_disagreement"] = disagreement
        df["unc_confidence_veto"] = veto

    def _build_factors(self, df: pd.DataFrame) -> None:
        date_groups = df.groupby("date")
        z = pd.DataFrame(index=df.index)
        for col in [
            "state_trend_prob",
            "state_conditional_drift",
            "state_persistence",
            "state_stress_prob",
            "state_conditional_volatility",
            "state_transition_entropy",
            "graph_residual",
            "graph_peer_shock_propagation",
            "graph_laplacian_deviation",
            "graph_local_dispersion",
            "graph_crowding_concentration",
            "unc_prediction_interval_width",
            "unc_tail_miss_rate",
            "unc_model_disagreement",
        ]:
            if col in df.columns:
                z[col] = date_groups[col].transform(_cross_sectional_zscore)
            else:
                z[col] = 0.0

        df["state_trend"] = (
            0.45 * z["state_trend_prob"]
            + 0.30 * z["state_conditional_drift"]
            + 0.25 * z["state_persistence"]
        ).fillna(0.0)
        df["state_stress"] = (
            0.45 * z["state_stress_prob"]
            + 0.30 * z["state_conditional_volatility"]
            + 0.25 * z["state_transition_entropy"]
        ).fillna(0.0)
        df["graph_dislocation"] = (
            0.35 * z["graph_residual"]
            + 0.25 * z["graph_peer_shock_propagation"]
            + 0.25 * z["graph_laplacian_deviation"]
            - 0.15 * z["graph_local_dispersion"]
        ).fillna(0.0)
        df["unc_signal_quality"] = (
            -0.45 * z["unc_prediction_interval_width"]
            - 0.30 * z["unc_tail_miss_rate"]
            - 0.25 * z["unc_model_disagreement"]
        ).fillna(0.0)


def augment_single_symbol_state_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.reset_index().rename(columns={work.index.name or "index": "date"})
    if "returns" not in work.columns:
        work["returns"] = pd.to_numeric(work["close"], errors="coerce").pct_change(fill_method=None)
    if "dollar_volume" not in work.columns:
        work["dollar_volume"] = pd.to_numeric(work["close"], errors="coerce") * pd.to_numeric(
            work.get("volume", 0.0), errors="coerce"
        )
    built = StateGraphFeatureBuilder().build(work, include_graph=False)
    out = built.features.set_index("date")
    cols = [c for c in out.columns if c.startswith(("state_", "unc_")) or c in {"state_trend", "state_stress", "unc_signal_quality"}]
    return out.loc[:, cols]


__all__ = [
    "STATE_GRAPH_FACTOR_COLUMNS",
    "STATE_GRAPH_PREFIXES",
    "StateGraphFeatureBuilder",
    "StateGraphFeatureResult",
    "augment_single_symbol_state_uncertainty",
]
