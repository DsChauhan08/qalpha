"""Portfolio optimization primitives (HRP, risk parity, CVaR guard)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.portfolio.risk_metrics import historical_cvar


def _normalize_weights(weights: pd.Series) -> pd.Series:
    w = weights.fillna(0.0).astype(float)
    total = float(np.abs(w).sum())
    if total <= 1e-12:
        return pd.Series(0.0, index=w.index)
    return w / total


def risk_parity_weights(cov: pd.DataFrame, risk_budget: Sequence[float] | None = None) -> pd.Series:
    """Diagonal risk-parity approximation via inverse variance."""

    if cov.empty:
        return pd.Series(dtype=float)

    diag = np.diag(cov.values).astype(float)
    diag = np.where(diag <= 1e-12, np.nan, diag)
    inv_var = 1.0 / diag
    if np.all(np.isnan(inv_var)):
        return pd.Series(0.0, index=cov.index)

    inv_var = np.nan_to_num(inv_var, nan=0.0, posinf=0.0, neginf=0.0)
    raw = pd.Series(inv_var, index=cov.index)

    if risk_budget is not None and len(risk_budget) == len(raw):
        rb = np.asarray(risk_budget, dtype=float)
        rb = np.clip(rb, 0.0, None)
        if rb.sum() > 0:
            rb = rb / rb.sum()
            raw = raw * pd.Series(rb, index=raw.index)

    return _normalize_weights(raw)


def _get_quasi_diag(link) -> List[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, cluster_items: Iterable[str]) -> float:
    c = cov.loc[list(cluster_items), list(cluster_items)]
    ivp = 1.0 / np.diag(c.values)
    ivp[np.isinf(ivp)] = 0.0
    if ivp.sum() <= 0:
        return float(np.trace(c.values))
    ivp = ivp / ivp.sum()
    w = ivp.reshape(-1, 1)
    return float(w.T @ c.values @ w)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado style)."""

    if returns.empty or returns.shape[1] == 0:
        return pd.Series(dtype=float)
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    clean = clean.dropna(axis=1, how="all")
    if clean.shape[1] == 0:
        return pd.Series(dtype=float)
    if clean.shape[1] == 1:
        return pd.Series({clean.columns[0]: 1.0})

    corr = clean.corr().fillna(0.0)
    cov = clean.cov().fillna(0.0)

    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        dist = np.sqrt(((1.0 - corr).clip(lower=0.0)) / 2.0)
        link = linkage(squareform(dist.values, checks=False), method="single")
        sort_ix = _get_quasi_diag(link)
        ordered = corr.index[sort_ix].tolist()

        w = pd.Series(1.0, index=ordered)
        clusters = [ordered]
        while len(clusters) > 0:
            clusters = [
                c[start:end]
                for c in clusters
                for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                c0 = clusters[i]
                c1 = clusters[i + 1]
                var0 = _cluster_var(cov, c0)
                var1 = _cluster_var(cov, c1)
                alpha = 1.0 - var0 / (var0 + var1 + 1e-12)
                w[c0] *= alpha
                w[c1] *= 1.0 - alpha

        w = w.reindex(clean.columns).fillna(0.0)
        return _normalize_weights(w)
    except Exception:
        # Fallback to inverse-vol approximation when clustering libs fail.
        cov_diag = clean.cov().fillna(0.0)
        return risk_parity_weights(cov_diag)


def apply_signal_tilt(
    base_weights: pd.Series,
    signal_scores: Dict[str, float],
    long_short_enabled: bool,
) -> pd.Series:
    w = base_weights.copy().astype(float)
    if w.empty:
        return w

    scores = pd.Series({k: float(v) for k, v in signal_scores.items()})
    scores = scores.reindex(w.index).fillna(0.0)

    if long_short_enabled:
        signs = np.sign(scores.replace(0.0, 1.0))
        tilt_mag = scores.abs()
        if tilt_mag.max() > 0:
            tilt_mag = tilt_mag / tilt_mag.max()
        tilt = 0.5 + 0.5 * tilt_mag
        w = w * tilt * signs
    else:
        tilt = scores.clip(lower=0.0)
        if tilt.max() > 0:
            tilt = tilt / tilt.max()
        w = w * (0.5 + 0.5 * tilt)
        w = w.clip(lower=0.0)

    return _normalize_weights(w)


def apply_cvar_guard(
    weights: pd.Series,
    returns: pd.DataFrame,
    alpha: float = 0.95,
    max_tail_loss: float = 0.06,
) -> pd.Series:
    w = weights.copy().astype(float)
    if w.empty or returns.empty:
        return w

    penalties = {}
    for col in w.index:
        if col not in returns.columns:
            penalties[col] = 1.0
            continue
        cvar = historical_cvar(returns[col].dropna(), alpha=alpha)
        tail = abs(float(cvar))
        if tail <= max_tail_loss:
            penalties[col] = 1.0
        else:
            penalties[col] = float(max(max_tail_loss / (tail + 1e-12), 0.1))

    p = pd.Series(penalties).reindex(w.index).fillna(1.0)
    w = w * p
    return _normalize_weights(w)


def enforce_constraints(weights: pd.Series, constraints: Dict[str, float | bool]) -> pd.Series:
    w = weights.copy().astype(float).fillna(0.0)
    if w.empty:
        return w

    long_short = bool(constraints.get("long_short_enabled", False))
    max_pos = float(constraints.get("max_position_abs", 0.10))
    gross_max = float(constraints.get("gross_max", 1.0))
    net_min = float(constraints.get("net_min", 0.0 if not long_short else -0.2))
    net_max = float(constraints.get("net_max", 1.0 if not long_short else 0.2))

    if long_short:
        w = w.clip(lower=-max_pos, upper=max_pos)
    else:
        w = w.clip(lower=0.0, upper=max_pos)

    gross = float(np.abs(w).sum())
    if gross > gross_max and gross > 1e-12:
        w *= gross_max / gross

    net = float(w.sum())
    if net < net_min:
        shift = (net_min - net) / len(w)
        w += shift
    elif net > net_max:
        shift = (net - net_max) / len(w)
        w -= shift

    if not long_short:
        w = w.clip(lower=0.0)

    # Re-enforce hard bounds after net shift.
    if long_short:
        w = w.clip(lower=-max_pos, upper=max_pos)
    else:
        w = w.clip(lower=0.0, upper=max_pos)

    gross = float(np.abs(w).sum())
    if gross > gross_max and gross > 1e-12:
        w *= gross_max / gross

    return w


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if returns.empty or weights.empty:
        return pd.Series(dtype=float)
    cols = [c for c in weights.index if c in returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    aligned = returns[cols].copy()
    w = weights[cols].astype(float)
    return aligned.mul(w, axis=1).sum(axis=1)


__all__ = [
    "risk_parity_weights",
    "hrp_weights",
    "apply_signal_tilt",
    "apply_cvar_guard",
    "enforce_constraints",
    "portfolio_returns",
]
