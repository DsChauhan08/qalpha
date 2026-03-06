"""Unified intraday feature builder for microstructure and path-shape research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.features.microstructure import (
    BidAskSpreadAnalyzer,
    OrderFlowImbalanceAnalyzer,
    VolatilitySignatureAnalyzer,
)
from quantum_alpha.features.mathematical.persistent_homology import (
    PersistentHomologyAnalyzer,
)
from quantum_alpha.features.regime_path_shape import RegimePathShapeFeatureGenerator


def _safe_series(series: pd.Series, fill: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
    )


def _rolling_linear_slope(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(0.0, index=series.index)
    x = np.arange(window, dtype=float)

    def _fit(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return 0.0
        y = np.nan_to_num(values.astype(float), nan=0.0)
        return float(np.polyfit(x, y, 1)[0])

    return series.rolling(window, min_periods=max(3, window // 2)).apply(_fit, raw=True).fillna(0.0)


def _minmax(series: pd.Series, lower: float = -10.0, upper: float = 10.0) -> pd.Series:
    return _safe_series(series).clip(lower=lower, upper=upper)


@dataclass(frozen=True)
class IntradayBuildResult:
    features: pd.DataFrame
    quality: Dict[str, float]


class UnifiedIntradayFeatureBuilder:
    """Build minute-level features from trades, quotes, depth, and bars."""

    RP_WINDOWS: Sequence[int] = (5, 15, 30)
    TP_WINDOWS: Sequence[int] = (15, 30)

    def __init__(self) -> None:
        self._ofi = OrderFlowImbalanceAnalyzer(window=20, vpin_n_buckets=24)
        self._spread = BidAskSpreadAnalyzer(window=20)
        self._volsig = VolatilitySignatureAnalyzer()
        self._ph = PersistentHomologyAnalyzer(lookback_window=30, max_dim=2)
        self._ps = RegimePathShapeFeatureGenerator()

    @staticmethod
    def get_base_feature_names() -> List[str]:
        names = [
            "ms_microprice",
            "ms_queue_imbalance",
            "ms_ofi",
            "ms_ofi_ma",
            "ms_signed_volume",
            "ms_vpin",
            "ms_relative_spread",
            "ms_depth_total",
            "ms_depth_slope",
            "ms_refill_rate",
            "ms_cancel_rate",
            "ms_spread_elasticity",
            "ms_downside_semivariance_15",
            "ms_vol_signature_slope",
            "tp_h0_total_persistence",
            "tp_h1_total_persistence",
            "tp_h1_num_cycles",
            "tp_persistence_entropy",
            "tp_depth_entropy",
            "tp_trend_persistence",
            "tp_reversion_pressure",
            "tp_liquidity_fracture",
            "tp_event_burst",
        ]
        for window in UnifiedIntradayFeatureBuilder.RP_WINDOWS:
            names.extend(
                [
                    f"rp_sig_mid_return_{window}",
                    f"rp_sig_spread_{window}",
                    f"rp_sig_ofi_{window}",
                    f"rp_sig_imbalance_{window}",
                    f"rp_sig_depth_slope_{window}",
                    f"rp_sig_mid_spread_{window}",
                    f"rp_sig_mid_ofi_{window}",
                    f"rp_path_entropy_{window}",
                    f"rp_energy_{window}",
                ]
            )
        names.extend(RegimePathShapeFeatureGenerator.get_feature_names())
        return names

    def build(
        self,
        *,
        symbol: str,
        trades: pd.DataFrame,
        quotes: pd.DataFrame,
        depth: pd.DataFrame,
        bars_1m: pd.DataFrame,
    ) -> IntradayBuildResult:
        bars = bars_1m.copy()
        if "timestamp" in bars.columns:
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
            bars = bars.set_index("timestamp")
        bars = bars.sort_index()
        bars = bars[["open", "high", "low", "close", "volume"]].copy()

        quotes_df = quotes.copy()
        quotes_df["timestamp"] = pd.to_datetime(quotes_df["timestamp"], utc=True)
        quotes_df = quotes_df.sort_values("timestamp")
        quotes_df = quotes_df.set_index("timestamp")
        quotes_min = quotes_df.resample("1min").last().ffill()
        quote_features = self._spread.compute_from_quotes(
            quotes_min.reset_index().rename(columns={"index": "timestamp"})
        ).set_index("timestamp")

        trades_df = trades.copy()
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True)
        trades_df = trades_df.sort_values("timestamp")
        ofi = self._ofi.compute_ofi_from_trades(trades_df.copy()).set_index("timestamp")
        signed_volume = (
            trades_df.assign(signed_volume=trades_df["volume"] * trades_df["side"])
            .set_index("timestamp")["signed_volume"]
            .resample("1min")
            .sum()
            .fillna(0.0)
        )
        depth_df = depth.copy()
        depth_df["timestamp"] = pd.to_datetime(depth_df["timestamp"], utc=True)
        depth_df = depth_df.sort_values(["timestamp", "level"])
        depth_group = depth_df.groupby("timestamp")
        depth_summary = depth_group.apply(self._summarize_depth_snapshot).reset_index()
        depth_summary = depth_summary.set_index("timestamp").resample("1min").last().ffill()

        out = bars.copy()
        out["symbol"] = symbol.upper()
        out["ms_microprice"] = _safe_series(
            (
                quotes_min["ask"] * quotes_min["bid_size"]
                + quotes_min["bid"] * quotes_min["ask_size"]
            )
            / (quotes_min["bid_size"] + quotes_min["ask_size"] + 1e-8)
        ).reindex(out.index).ffill().bfill()
        out["ms_queue_imbalance"] = _safe_series(
            (quotes_min["bid_size"] - quotes_min["ask_size"])
            / (quotes_min["bid_size"] + quotes_min["ask_size"] + 1e-8)
        ).reindex(out.index).ffill().fillna(0.0)
        out["ms_ofi"] = _safe_series(ofi["ofi"]).reindex(out.index).fillna(0.0)
        out["ms_ofi_ma"] = _safe_series(ofi["ofi_ma"]).reindex(out.index).fillna(0.0)
        out["ms_signed_volume"] = _safe_series(signed_volume).reindex(out.index).fillna(0.0)
        out["ms_vpin"] = _safe_series(
            out["ms_ofi"].abs().rolling(20, min_periods=5).sum()
            / (out["volume"].rolling(20, min_periods=5).sum() + 1e-8)
        ).fillna(0.0)
        out["ms_relative_spread"] = _safe_series(quote_features["relative_spread"]).reindex(out.index).ffill().fillna(0.0)
        out["ms_depth_total"] = _safe_series(depth_summary["depth_total"]).reindex(out.index).ffill().fillna(0.0)
        out["ms_depth_slope"] = _safe_series(depth_summary["depth_slope"]).reindex(out.index).ffill().fillna(0.0)
        out["ms_refill_rate"] = _safe_series(depth_summary["refill_rate"]).reindex(out.index).ffill().fillna(0.0)
        out["ms_cancel_rate"] = _safe_series(depth_summary["cancel_rate"]).reindex(out.index).ffill().fillna(0.0)
        out["ms_spread_elasticity"] = _safe_series(
            out["ms_relative_spread"].diff() / (out["ms_depth_total"].pct_change(fill_method=None).replace(0, np.nan))
        ).fillna(0.0)
        minute_returns = out["close"].pct_change(fill_method=None).fillna(0.0)
        out["ms_downside_semivariance_15"] = minute_returns.clip(upper=0.0).pow(2).rolling(
            15, min_periods=5
        ).mean().fillna(0.0)
        out["ms_vol_signature_slope"] = self._compute_signature_slope(out["close"])

        tp_features = self._compute_topology_features(out)
        for col, series in tp_features.items():
            out[col] = series.reindex(out.index).fillna(0.0)

        rp_features = self._compute_rough_path_features(out)
        for col, series in rp_features.items():
            out[col] = series.reindex(out.index).fillna(0.0)

        ps_frame = self._ps.generate(out[["open", "high", "low", "close", "volume"]].copy())
        for col in self._ps.get_feature_names():
            out[col] = _safe_series(ps_frame[col]).reindex(out.index).fillna(0.0)

        out["tp_trend_persistence"] = _minmax(
            0.45 * out.get("ps_trend_persistence", 0.0)
            + 0.25 * out["tp_h1_total_persistence"]
            + 0.15 * out["rp_energy_30"]
            - 0.15 * out["ms_relative_spread"]
        )
        out["tp_reversion_pressure"] = _minmax(
            0.40 * out.get("ps_mean_reversion_pressure", 0.0)
            + 0.30 * (-out["ms_ofi_ma"].rolling(5, min_periods=2).mean())
            + 0.20 * (-out["tp_h1_num_cycles"])
            + 0.10 * (-out["ms_queue_imbalance"])
        )
        out["tp_liquidity_fracture"] = _minmax(
            0.35 * out["ms_relative_spread"]
            + 0.25 * (-out["ms_depth_total"].pct_change(fill_method=None).fillna(0.0))
            + 0.20 * out["tp_persistence_entropy"]
            + 0.20 * out["ms_cancel_rate"]
        )
        out["tp_event_burst"] = _minmax(
            0.35 * out["ms_signed_volume"].abs().rolling(5, min_periods=2).mean()
            + 0.25 * out["ms_vpin"]
            + 0.20 * out["ms_vol_signature_slope"].abs()
            + 0.20 * out["tp_h0_total_persistence"]
        )

        quality = {
            "symbol": symbol.upper(),
            "rows": int(len(out)),
            "negative_spread_rate": float((out["ms_relative_spread"] < 0).mean()),
            "crossed_market_rate": float((quotes_min["ask"] <= quotes_min["bid"]).mean())
            if len(quotes_min) > 0
            else 0.0,
            "depth_completeness": float(depth_summary["depth_total"].notna().mean()) if len(depth_summary) else 0.0,
            "median_quote_staleness_ms": self._estimate_quote_staleness(trades_df, quotes_df),
        }

        feature_cols = [c for c in out.columns if c.startswith(("ms_", "tp_", "rp_", "ps_"))]
        out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return IntradayBuildResult(features=out, quality=quality)

    @staticmethod
    def _estimate_quote_staleness(trades: pd.DataFrame, quotes: pd.DataFrame) -> float:
        if trades.empty or quotes.empty:
            return 0.0
        trade_df = trades.loc[:, ["timestamp"]].sort_values("timestamp").copy()
        quote_df = quotes.reset_index(drop=False)
        if "index" in quote_df.columns and "timestamp" not in quote_df.columns:
            quote_df = quote_df.rename(columns={"index": "timestamp"})
        quote_df = quote_df.loc[:, ["timestamp"]].sort_values("timestamp").rename(columns={"timestamp": "quote_ts"})
        aligned = pd.merge_asof(
            trade_df,
            quote_df,
            left_on="timestamp",
            right_on="quote_ts",
            direction="backward",
            tolerance=pd.Timedelta(seconds=5),
        )
        if "quote_ts" not in aligned:
            return 0.0
        delta = (aligned["timestamp"] - aligned["quote_ts"]).dt.total_seconds() * 1000.0
        delta = delta.dropna()
        if delta.empty:
            return 0.0
        return float(delta.median())

    @staticmethod
    def _summarize_depth_snapshot(snapshot: pd.DataFrame) -> pd.Series:
        df = snapshot.sort_values("level").copy()
        depth_total = float(df["bid_size"].sum() + df["ask_size"].sum())
        prices = np.arange(1, len(df) + 1, dtype=float)
        bid_coeff = np.polyfit(prices, df["bid_size"].to_numpy(dtype=float), 1)[0] if len(df) > 1 else 0.0
        ask_coeff = np.polyfit(prices, df["ask_size"].to_numpy(dtype=float), 1)[0] if len(df) > 1 else 0.0
        size_delta = df["bid_size"].sum() - df["ask_size"].sum()
        level1 = df.iloc[0]
        refill_rate = max(float(level1["bid_size"]) - float(level1["ask_size"]), 0.0) / (depth_total + 1e-8)
        cancel_rate = max(float(level1["ask_size"]) - float(level1["bid_size"]), 0.0) / (depth_total + 1e-8)
        return pd.Series(
            {
                "depth_total": depth_total,
                "depth_slope": float((bid_coeff - ask_coeff) / 2.0),
                "depth_imbalance": float(size_delta / (depth_total + 1e-8)),
                "refill_rate": refill_rate,
                "cancel_rate": cancel_rate,
            }
        )

    def _compute_signature_slope(self, close: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=close.index, dtype=float)
        for i in range(19, len(close), 5):
            start = max(0, i - 59)
            window = close.iloc[start : i + 1]
            if len(window) < 20:
                continue
            sig = self._volsig.compute_signature(window, frequencies=[1, 2, 5, 10, 15])
            if len(sig) < 2:
                continue
            out.iloc[i] = float(np.polyfit(sig["frequency"], sig["realised_vol"], 1)[0])
        return out.ffill().fillna(0.0)

    def _compute_topology_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        output = {
            "tp_h0_total_persistence": pd.Series(np.nan, index=bars.index, dtype=float),
            "tp_h1_total_persistence": pd.Series(np.nan, index=bars.index, dtype=float),
            "tp_h1_num_cycles": pd.Series(np.nan, index=bars.index, dtype=float),
            "tp_persistence_entropy": pd.Series(np.nan, index=bars.index, dtype=float),
            "tp_depth_entropy": pd.Series(np.nan, index=bars.index, dtype=float),
        }
        price_block = bars[["open", "high", "low", "close", "volume"]]
        for window in self.TP_WINDOWS:
            for i in range(window - 1, len(price_block), 5):
                section = price_block.iloc[i - window + 1 : i + 1]
                point_cloud = self._ph.create_point_cloud(section)
                if len(point_cloud) < 5:
                    continue
                stats = self._ph.compute_persistence(point_cloud)
                idx = section.index[-1]
                output["tp_h0_total_persistence"].loc[idx] = float(stats.get("h0_total_persistence", 0.0))
                output["tp_h1_total_persistence"].loc[idx] = float(stats.get("h1_total_persistence", 0.0))
                output["tp_h1_num_cycles"].loc[idx] = float(stats.get("h1_num_cycles", 0.0))
                lifetimes = np.array(
                    [
                        float(stats.get("h0_max_lifetime", 0.0)),
                        float(stats.get("h1_max_lifetime", 0.0)),
                        float(stats.get("h1_mean_lifetime", 0.0)),
                    ],
                    dtype=float,
                )
                lifetimes = lifetimes[lifetimes > 0]
                probs = lifetimes / lifetimes.sum() if lifetimes.sum() > 0 else np.array([1.0])
                output["tp_persistence_entropy"].loc[idx] = float(-(probs * np.log(probs + 1e-12)).sum())
        output["tp_depth_entropy"] = _safe_series(
            bars["volume"].rolling(15, min_periods=5).apply(
                lambda x: float(
                    -np.sum(
                        (np.abs(x) / (np.abs(x).sum() + 1e-8))
                        * np.log(np.abs(x) / (np.abs(x).sum() + 1e-8) + 1e-12)
                    )
                ),
                raw=True,
            )
        )
        for key, series in list(output.items()):
            output[key] = series.ffill().fillna(0.0)
        return output

    def _compute_rough_path_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        mid_return = bars["ms_microprice"].pct_change(fill_method=None).fillna(0.0)
        spread = bars["ms_relative_spread"].fillna(0.0)
        ofi = bars["ms_ofi"].fillna(0.0)
        imbalance = bars["ms_queue_imbalance"].fillna(0.0)
        depth_slope = bars["ms_depth_slope"].fillna(0.0)
        streams = {
            "mid_return": mid_return,
            "spread": spread,
            "ofi": ofi,
            "imbalance": imbalance,
            "depth_slope": depth_slope,
        }

        for window in self.RP_WINDOWS:
            for name, series in streams.items():
                out[f"rp_sig_{name}_{window}"] = _safe_series(
                    series.rolling(window, min_periods=max(2, window // 2)).sum()
                )
            out[f"rp_sig_mid_spread_{window}"] = _safe_series(
                (mid_return * spread.shift(1).fillna(0.0)).rolling(window, min_periods=max(2, window // 2)).sum()
            )
            out[f"rp_sig_mid_ofi_{window}"] = _safe_series(
                (mid_return * ofi.shift(1).fillna(0.0)).rolling(window, min_periods=max(2, window // 2)).sum()
            )
            out[f"rp_path_entropy_{window}"] = _safe_series(
                mid_return.rolling(window, min_periods=max(3, window // 2)).apply(
                    lambda x: float(
                        -np.sum(
                            (np.abs(x) / (np.abs(x).sum() + 1e-8))
                            * np.log(np.abs(x) / (np.abs(x).sum() + 1e-8) + 1e-12)
                        )
                    ),
                    raw=True,
                )
            )
            out[f"rp_energy_{window}"] = _safe_series(
                (
                    mid_return.pow(2)
                    + spread.pow(2)
                    + imbalance.pow(2)
                    + depth_slope.pow(2)
                ).rolling(window, min_periods=max(3, window // 2)).mean()
            )
        return out


__all__ = ["IntradayBuildResult", "UnifiedIntradayFeatureBuilder"]
