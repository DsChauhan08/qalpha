"""Portfolio construction engine using HRP/RP/CVaR and dual-run selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quantum_alpha.portfolio.contracts import AllocatorInput, AllocatorOutput, RiskSnapshot
from quantum_alpha.portfolio.optimizers import (
    apply_cvar_guard,
    apply_signal_tilt,
    enforce_constraints,
    hrp_weights,
    portfolio_returns,
    risk_parity_weights,
)
from quantum_alpha.portfolio.risk_metrics import risk_snapshot
from quantum_alpha.portfolio.selector import DualRunSelector
from quantum_alpha.portfolio.volatility import estimate_all_vols


@dataclass
class PortfolioAllocatorEngine:
    config: Dict[str, object]
    selector: DualRunSelector

    @classmethod
    def from_config(cls, config: Optional[Dict[str, object]] = None) -> "PortfolioAllocatorEngine":
        cfg = config or {}
        selector_cfg = cfg.get("selector", {}) if isinstance(cfg.get("selector", {}), dict) else {}
        selector = DualRunSelector(
            window_cycles=int(selector_cfg.get("window_cycles", 20)),
            switch_margin=float(selector_cfg.get("switch_margin", 0.02)),
            min_hold_cycles=int(selector_cfg.get("min_hold_cycles", 5)),
        )
        return cls(config=cfg, selector=selector)

    def _constraints(self, overrides: Optional[Dict[str, object]] = None) -> Dict[str, float | bool]:
        base = {
            "long_short_enabled": True,
            "net_min": -0.20,
            "net_max": 0.20,
            "gross_max": 1.0,
            "max_position_abs": 0.10,
        }
        cfg = self.config.get("constraints", {}) if isinstance(self.config.get("constraints"), dict) else {}
        base.update(cfg)
        if overrides:
            base.update(overrides)
        return base

    @staticmethod
    def _build_input_from_featured(
        featured: Dict[str, pd.DataFrame],
        constraints: Dict[str, float | bool],
        benchmark_symbol: str,
        timestamp: datetime,
    ) -> AllocatorInput:
        price_cols = {}
        returns_cols = {}
        for sym, df in featured.items():
            if df is None or df.empty:
                continue
            close = pd.to_numeric(df.get("close"), errors="coerce")
            if close is None or close.dropna().empty:
                continue
            price_cols[sym] = close
            if "returns" in df.columns:
                r = pd.to_numeric(df["returns"], errors="coerce")
            else:
                r = close.pct_change()
            returns_cols[sym] = r

        prices = pd.DataFrame(price_cols)
        returns = pd.DataFrame(returns_cols).replace([np.inf, -np.inf], np.nan).dropna(how="all")
        benchmark = returns.get(benchmark_symbol) if benchmark_symbol in returns.columns else None

        return AllocatorInput(
            prices=prices,
            returns=returns,
            features=None,
            constraints=constraints,
            benchmark=benchmark,
            timestamp=timestamp,
        )

    @staticmethod
    def _cov_from_returns(returns: pd.DataFrame, vols: Dict[str, float], min_var: float = 1e-8) -> pd.DataFrame:
        cols = list(returns.columns)
        if not cols:
            return pd.DataFrame()
        corr = returns[cols].corr().fillna(0.0)
        vol_vec = np.array([max(float(vols.get(c, 0.0)), 0.0) for c in cols], dtype=float)
        # Convert annualized vol to per-step vol before covariance synthesis.
        vol_vec = vol_vec / np.sqrt(252.0)
        cov = corr.values * np.outer(vol_vec, vol_vec)
        cov = np.where(np.isfinite(cov), cov, 0.0)
        cov[np.diag_indices_from(cov)] = np.maximum(np.diag(cov), min_var)
        return pd.DataFrame(cov, index=cols, columns=cols)

    @staticmethod
    def _score_stack(port_returns: pd.Series) -> float:
        snap = risk_snapshot(port_returns)
        sharpe = float(snap.get("sharpe", 0.0))
        dd_pen = abs(float(snap.get("drawdown", 0.0)))
        cvar_pen = abs(float(snap.get("cvar", 0.0)))
        return float(sharpe - 0.5 * dd_pen - 3.0 * cvar_pen)

    @staticmethod
    def _vol_estimates(featured: Dict[str, pd.DataFrame], primary: str) -> Dict[str, float]:
        per_symbol = {}
        for sym, df in featured.items():
            if df is None or df.empty:
                continue
            vols = estimate_all_vols(df)
            per_symbol[sym] = float(vols.get(primary, vols.get("yang_zhang", 0.0)))
        return per_symbol

    def _stack_weights(
        self,
        alloc_input: AllocatorInput,
        signal_scores: Dict[str, float],
        featured: Dict[str, pd.DataFrame],
        regime_mult: float = 1.0,
    ) -> pd.Series:
        returns = alloc_input.returns.dropna(how="all")
        if returns.empty:
            return pd.Series({k: 0.0 for k in signal_scores})

        primary = str((self.config.get("vol_estimator") or {}).get("primary", "yang_zhang"))
        vol_map = self._vol_estimates(featured, primary=primary)
        cov = self._cov_from_returns(returns, vol_map)

        w_hrp = hrp_weights(returns)
        w_rp = risk_parity_weights(cov)

        cols = sorted(set(w_hrp.index) | set(w_rp.index) | set(signal_scores.keys()))
        base = 0.5 * w_hrp.reindex(cols).fillna(0.0) + 0.5 * w_rp.reindex(cols).fillna(0.0)

        long_short = bool(alloc_input.constraints.get("long_short_enabled", False))
        tilted = apply_signal_tilt(base, signal_scores=signal_scores, long_short_enabled=long_short)

        guarded = apply_cvar_guard(
            tilted,
            returns=returns.reindex(columns=tilted.index).fillna(0.0),
            alpha=float(self.config.get("cvar_alpha", 0.95)),
            max_tail_loss=float(self.config.get("cvar_guard_max_tail_loss", 0.06)),
        )

        cons = dict(alloc_input.constraints)
        cons["gross_max"] = float(cons.get("gross_max", 1.0)) * float(regime_mult)
        constrained = enforce_constraints(guarded, constraints=cons)
        return constrained.reindex(cols).fillna(0.0)

    def allocate(
        self,
        signal_scores: Dict[str, float],
        featured: Dict[str, pd.DataFrame],
        timestamp: Optional[datetime] = None,
        benchmark_symbol: str = "SPY",
        constraint_overrides: Optional[Dict[str, object]] = None,
    ) -> AllocatorOutput:
        ts = timestamp or datetime.utcnow()
        constraints = self._constraints(overrides=constraint_overrides)
        alloc_input = self._build_input_from_featured(
            featured=featured,
            constraints=constraints,
            benchmark_symbol=benchmark_symbol,
            timestamp=ts,
        )

        static_w = self._stack_weights(
            alloc_input=alloc_input,
            signal_scores=signal_scores,
            featured=featured,
            regime_mult=1.0,
        )

        bench = alloc_input.benchmark if alloc_input.benchmark is not None else pd.Series(dtype=float)
        bench_stats = risk_snapshot(bench) if not bench.empty else {"drawdown": 0.0, "var": 0.0}
        high_stress = abs(float(bench_stats.get("drawdown", 0.0))) > 0.08 or abs(
            float(bench_stats.get("var", 0.0))
        ) > 0.03
        regime_mult = float(self.config.get("regime_gross_scale", 0.8 if high_stress else 1.0))
        regime_mult = min(max(regime_mult, 0.4), 1.0)

        regime_w = self._stack_weights(
            alloc_input=alloc_input,
            signal_scores=signal_scores,
            featured=featured,
            regime_mult=regime_mult,
        )

        port_static = portfolio_returns(alloc_input.returns, static_w)
        port_regime = portfolio_returns(alloc_input.returns, regime_w)
        score_static = self._score_stack(port_static)
        score_regime = self._score_stack(port_regime)

        sel = self.selector.update_and_select(static_score=score_static, regime_score=score_regime)
        chosen = str(sel["chosen_stack"])
        chosen_w = static_w if chosen == "static" else regime_w

        chosen_ret = portfolio_returns(alloc_input.returns, chosen_w)
        chosen_snap = risk_snapshot(chosen_ret)

        # Collect primary estimator values for dashboard status.
        primary = str((self.config.get("vol_estimator") or {}).get("primary", "yang_zhang"))
        sym_vols = self._vol_estimates(featured, primary=primary)
        vol_estimates = {
            "primary": primary,
            "portfolio": float(np.nanmean(list(sym_vols.values())) if sym_vols else 0.0),
            "symbols": {k: float(v) for k, v in sym_vols.items()},
        }

        risk = RiskSnapshot(
            var=float(chosen_snap.get("var", 0.0)),
            cvar=float(chosen_snap.get("cvar", 0.0)),
            drawdown=float(chosen_snap.get("drawdown", 0.0)),
            ulcer_index=float(chosen_snap.get("ulcer_index", 0.0)),
            vol_estimates=vol_estimates,
            net_exposure=float(chosen_w.sum()) if not chosen_w.empty else 0.0,
            gross_exposure=float(np.abs(chosen_w).sum()) if not chosen_w.empty else 0.0,
        )

        out_w = {sym: float(chosen_w.get(sym, 0.0)) for sym in sorted(set(featured.keys()) | set(signal_scores.keys()))}

        diagnostics = {
            "stack_scores": {
                "static": float(score_static),
                "regime": float(score_regime),
            },
            "selector": sel,
            "regime_mult": float(regime_mult),
            "static_weights": {k: float(v) for k, v in static_w.to_dict().items()},
            "regime_weights": {k: float(v) for k, v in regime_w.to_dict().items()},
            "benchmark_stats": {
                "drawdown": float(bench_stats.get("drawdown", 0.0)),
                "var": float(bench_stats.get("var", 0.0)),
            },
        }

        return AllocatorOutput(
            weights=out_w,
            risk_snapshot=risk,
            diagnostics=diagnostics,
            chosen_stack=chosen,
        )


__all__ = ["PortfolioAllocatorEngine"]
