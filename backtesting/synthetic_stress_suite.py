#!/usr/bin/env python3
"""
Synthetic stress suite for meta-ensemble.

Creates intentionally weird/unpredictable synthetic markets plus synthetic
news distortions, then evaluates the strategy against:
  - Cost-aware equal-weight benchmark
  - Market proxy (cross-sectional average return)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from quantum_alpha.backtest_clean import (
    backtest_equal_weight,
    blend_prediction_probabilities,
    compute_signals,
    deduplicate_predictions,
    load_predictions,
)


@dataclass(frozen=True)
class StressConfig:
    signal_threshold: float = 0.52
    short_threshold: float | None = None
    confidence: float = 0.0
    long_only: bool = True
    commission_bps: float = 5.0
    hold_days: int = 5
    top_k: int = 10
    max_positions: int = 20


@dataclass
class BaseArrays:
    df_template: pd.DataFrame
    base_return: np.ndarray
    base_prob: np.ndarray
    base_conf: np.ndarray
    date_codes: np.ndarray
    symbol_codes: np.ndarray
    dates: pd.DatetimeIndex
    symbols: List[str]


def _parse_csv_str(value: str | None) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_float(value: str | None) -> List[float]:
    if not value:
        return []
    return [float(x) for x in value.split(",") if x.strip()]


def _returns_to_metrics(returns: pd.Series) -> Dict[str, float]:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(r) == 0:
        return {
            "n_days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    eq = (1.0 + r).cumprod()
    tr = float(eq.iloc[-1] - 1.0)
    years = max(len(r) / 252.0, 1.0 / 252.0)
    ann = float((1.0 + tr) ** (1.0 / years) - 1.0)
    sd = float(r.std(ddof=0))
    sh = float((r.mean() / sd) * np.sqrt(252.0)) if sd > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {
        "n_days": int(len(r)),
        "total_return": tr,
        "annual_return": ann,
        "sharpe": sh,
        "max_drawdown": mdd,
    }


def _build_base_arrays(pred: pd.DataFrame) -> BaseArrays:
    df = pred[["date", "symbol", "y_true", "forward_return", "y_proba"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    symbols = sorted(df["symbol"].unique().tolist())
    date_to_code = {d: i for i, d in enumerate(dates)}
    sym_to_code = {s: i for i, s in enumerate(symbols)}

    date_codes = df["date"].map(date_to_code).to_numpy(dtype=np.int32)
    symbol_codes = df["symbol"].map(sym_to_code).to_numpy(dtype=np.int16)
    base_return = pd.to_numeric(df["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_prob = pd.to_numeric(df["y_proba"], errors="coerce").fillna(0.5).clip(0.01, 0.99).to_numpy(dtype=float)
    base_conf = np.abs(base_prob - 0.5) * 2.0

    return BaseArrays(
        df_template=df,
        base_return=base_return,
        base_prob=base_prob,
        base_conf=base_conf,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        dates=dates,
        symbols=symbols,
    )


def _ar1_with_jumps(
    n: int,
    rng: np.random.Generator,
    phi: float = 0.85,
    noise_scale: float = 1.0,
    jump_prob: float = 0.02,
    jump_scale: float = 3.0,
) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, noise_scale)
        if rng.random() < jump_prob:
            x[i] += rng.normal(0.0, jump_scale)
    return x


def _simulate_scenario(
    base: BaseArrays,
    scenario: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_rows = len(base.df_template)
    n_dates = len(base.dates)
    n_syms = len(base.symbols)

    date_noise = rng.standard_t(df=4, size=n_dates) * 0.004
    idio_noise = rng.standard_t(df=5, size=n_rows) * 0.003
    news_factor = _ar1_with_jumps(
        n_dates,
        rng=rng,
        phi=0.82,
        noise_scale=0.8,
        jump_prob=0.03,
        jump_scale=2.5,
    )
    exposure = rng.uniform(-1.0, 1.0, size=n_syms)
    news_row = news_factor[base.date_codes] * exposure[base.symbol_codes]

    ret = base.base_return.copy()
    prob = base.base_prob.copy()

    if scenario == "vol_regime":
        state = np.zeros(n_dates, dtype=np.int8)
        for i in range(1, n_dates):
            if state[i - 1] == 0:
                state[i] = 1 if rng.random() < 0.05 else 0
            else:
                state[i] = 0 if rng.random() < 0.12 else 1
        vol_scale = np.where(state == 1, 2.6, 1.0)
        ret = ret * vol_scale[base.date_codes] + date_noise[base.date_codes] + idio_noise
        prob = prob + rng.normal(0.0, 0.025, size=n_rows)

    elif scenario == "crash_rebound":
        shock = np.zeros(n_dates, dtype=float)
        n_events = int(max(3, n_dates // 900))
        crash_starts = rng.choice(np.arange(20, n_dates - 60), size=n_events, replace=False)
        for start in crash_starts:
            drop = float(rng.uniform(-0.14, -0.07))
            shock[start] += drop
            # Rebound arc
            for j in range(1, 25):
                idx = start + j
                if idx >= n_dates:
                    break
                shock[idx] += float(abs(drop) * np.exp(-j / 7.0) * rng.uniform(0.4, 0.9))
        ret = ret + shock[base.date_codes] + idio_noise * 1.3
        prob = prob + rng.normal(0.0, 0.03, size=n_rows)

    elif scenario == "whipsaw_chop":
        trend = np.zeros(n_dates, dtype=float)
        sign = 1.0
        i = 0
        while i < n_dates:
            seg_len = int(rng.integers(8, 30))
            end = min(n_dates, i + seg_len)
            trend[i:end] = sign * rng.uniform(0.002, 0.007)
            sign *= -1.0
            i = end
        ret = ret + trend[base.date_codes] + idio_noise * 1.4 + date_noise[base.date_codes]
        # Model gets repeatedly faked out by short-lived news narratives
        prob = prob + np.sign(news_row) * rng.normal(0.0, 0.035, size=n_rows)

    elif scenario == "news_chaos":
        rumor = np.zeros(n_dates, dtype=float)
        for i in range(1, n_dates):
            rumor[i] = 0.75 * rumor[i - 1] + rng.normal(0.0, 1.1)
            if rng.random() < 0.08:
                rumor[i] += rng.normal(0.0, 3.8)
        rumor_row = rumor[base.date_codes] * exposure[base.symbol_codes]
        # Weird news: frequently contradictory to realized returns.
        sign_flip = np.where(rng.random(n_rows) < 0.35, -1.0, 1.0)
        ret = ret + sign_flip * rumor_row * 0.006 + idio_noise * 1.2
        prob = prob + rumor_row * 0.045 + rng.normal(0.0, 0.04, size=n_rows)
        # Misinformation spikes invert high-confidence calls.
        invert_mask = (base.base_conf > 0.25) & (rng.random(n_rows) < 0.22)
        prob[invert_mask] = 1.0 - prob[invert_mask]

    elif scenario == "anti_signal":
        anti = -np.sign(base.base_prob - 0.5) * (0.003 + 0.018 * base.base_conf)
        ret = ret + anti + idio_noise
        prob = prob + rng.normal(0.0, 0.03, size=n_rows)

    elif scenario == "mixed_chaos":
        # Combine regime shifts + crashes + chaotic news + anti-signal pockets.
        vol_state = _ar1_with_jumps(
            n_dates,
            rng=rng,
            phi=0.92,
            noise_scale=0.25,
            jump_prob=0.02,
            jump_scale=0.9,
        )
        vol_scale = 1.0 + np.clip(np.abs(vol_state), 0.0, 2.5)
        crash = np.zeros(n_dates, dtype=float)
        for start in rng.choice(np.arange(20, n_dates - 40), size=max(4, n_dates // 800), replace=False):
            crash[start] += float(rng.uniform(-0.12, -0.05))
        anti = -np.sign(base.base_prob - 0.5) * (0.002 + 0.01 * base.base_conf)
        ret = (
            base.base_return * vol_scale[base.date_codes]
            + date_noise[base.date_codes]
            + crash[base.date_codes]
            + idio_noise * 1.6
            + news_row * 0.006
            + anti * (rng.random(n_rows) < 0.45).astype(float)
        )
        prob = (
            prob
            + news_row * 0.04
            + rng.normal(0.0, 0.05, size=n_rows)
            + np.sign(date_noise[base.date_codes]) * 0.01
        )
        invert_mask = (base.base_conf > 0.20) & (rng.random(n_rows) < 0.15)
        prob[invert_mask] = 1.0 - prob[invert_mask]

    elif scenario == "liquidity_vacuum":
        # Liquidity gaps: larger open/close jumps with confidence collapse.
        liq_stress = np.clip(
            np.abs(
                _ar1_with_jumps(
                    n_dates,
                    rng=rng,
                    phi=0.90,
                    noise_scale=0.32,
                    jump_prob=0.05,
                    jump_scale=1.2,
                )
            ),
            0.0,
            2.8,
        )
        gap = rng.normal(0.0, 0.005, size=n_dates) * (1.0 + 1.8 * liq_stress)
        panic = np.zeros(n_dates, dtype=float)
        panic_events = np.where(rng.random(n_dates) < (0.008 + 0.02 * (liq_stress > 1.0)))[0]
        for idx in panic_events:
            drop = float(rng.uniform(-0.08, -0.025))
            panic[idx] += drop
            if idx + 1 < n_dates:
                panic[idx + 1] += float(abs(drop) * rng.uniform(0.35, 0.75))

        ret = (
            base.base_return * (1.0 - 0.3 * liq_stress[base.date_codes])
            + gap[base.date_codes]
            + panic[base.date_codes] * (0.5 + np.abs(exposure[base.symbol_codes]))
            + idio_noise * 1.9
        )
        uncertainty = 1.0 - np.clip(0.55 * liq_stress[base.date_codes], 0.0, 0.8)
        prob = 0.5 + (prob - 0.5) * uncertainty + rng.normal(0.0, 0.05, size=n_rows)

    elif scenario == "policy_whipsaw":
        # Macro/policy pivots that reverse rapidly and punish stale signals.
        regime = np.zeros(n_dates, dtype=float)
        t = 0
        sign = 1.0
        while t < n_dates:
            seg = int(rng.integers(10, 50))
            regime[t : min(n_dates, t + seg)] = sign
            sign *= -1.0
            t += seg
        lag_regime = np.roll(regime, 1)
        lag_regime[0] = regime[0]

        beta = rng.normal(0.0, 1.0, size=n_syms)
        row_beta = beta[base.symbol_codes]
        ret = (
            base.base_return
            + regime[base.date_codes] * row_beta * 0.0075
            + date_noise[base.date_codes]
            + idio_noise * 1.2
        )
        prob = prob + lag_regime[base.date_codes] * row_beta * 0.05 + rng.normal(0.0, 0.03, size=n_rows)
        policy_flip = regime[base.date_codes] != lag_regime[base.date_codes]
        prob[policy_flip] = prob[policy_flip] - np.sign(prob[policy_flip] - 0.5) * 0.09

    elif scenario == "news_hallucination":
        # Highly viral but often wrong narratives that invert outcomes.
        hall = _ar1_with_jumps(
            n_dates,
            rng=rng,
            phi=0.74,
            noise_scale=1.3,
            jump_prob=0.09,
            jump_scale=4.4,
        )
        hall_row = hall[base.date_codes] * exposure[base.symbol_codes]
        misled = (rng.random(n_rows) < 0.65).astype(float)
        ret = (
            base.base_return
            - np.sign(hall_row) * np.abs(hall_row) * 0.0055 * misled
            + np.sign(hall_row) * np.abs(hall_row) * 0.002 * (1.0 - misled)
            + idio_noise * 1.5
            + date_noise[base.date_codes]
        )
        prob = prob + hall_row * 0.07 + rng.normal(0.0, 0.06, size=n_rows)
        high_hall = np.abs(hall_row) > np.quantile(np.abs(hall_row), 0.80)
        invert_mask = high_hall & (base.base_conf > 0.15) & (rng.random(n_rows) < 0.45)
        prob[invert_mask] = 1.0 - prob[invert_mask]

    elif scenario == "theme_bubble_burst":
        # Theme crowding boom -> abrupt bust (e.g., AI mania then de-risking).
        raw_theme = rng.normal(0.0, 1.0, size=n_syms)
        cutoff = np.quantile(raw_theme, 0.75)
        theme = np.where(raw_theme >= cutoff, 1.0, -0.25)

        bubble = np.zeros(n_dates, dtype=float)
        bubble[: max(2, int(0.62 * n_dates))] = np.linspace(0.0, 1.0, max(2, int(0.62 * n_dates)))
        bubble[max(2, int(0.62 * n_dates)) :] = np.linspace(
            1.0,
            -0.6,
            max(1, n_dates - max(2, int(0.62 * n_dates))),
        )
        bubble += rng.normal(0.0, 0.08, size=n_dates)
        lag_bubble = np.roll(bubble, 1)
        lag_bubble[0] = bubble[0]

        row_theme = theme[base.symbol_codes]
        ret = base.base_return + bubble[base.date_codes] * row_theme * 0.007 + idio_noise * 1.1
        prob = prob + lag_bubble[base.date_codes] * row_theme * 0.055 + rng.normal(0.0, 0.03, size=n_rows)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    ret = np.clip(ret, -0.30, 0.30)
    prob = np.clip(prob, 0.01, 0.99)
    return ret, prob


def _evaluate_synthetic_path(
    base: BaseArrays,
    synthetic_return: np.ndarray,
    synthetic_prob: np.ndarray,
    cfg: StressConfig,
) -> Dict[str, float]:
    df = base.df_template.copy()
    df["forward_return"] = synthetic_return
    df["y_proba"] = synthetic_prob
    df["y_true"] = (synthetic_return > 0).astype(int)

    with contextlib.redirect_stdout(io.StringIO()):
        sig = compute_signals(
            df,
            signal_threshold=cfg.signal_threshold,
            short_threshold=cfg.short_threshold,
            confidence_threshold=cfg.confidence,
            long_only=cfg.long_only,
        )
        model_bt = backtest_equal_weight(
            sig,
            max_positions=cfg.max_positions,
            commission_bps=cfg.commission_bps,
            hold_days=cfg.hold_days,
            top_k=cfg.top_k,
            initial_capital=100_000.0,
            confidence_weight=False,
            earnings_filter=False,
            pead_boost=False,
            semiconductor_short_gate=False,
            gate_threshold=0.0,
        )

    bench = df[["date", "symbol", "forward_return", "y_true"]].copy()
    bench["raw_signal"] = 1.0
    bench["confidence"] = 1.0
    bench["y_proba"] = 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        ew_bt = backtest_equal_weight(
            bench,
            max_positions=len(base.symbols),
            commission_bps=cfg.commission_bps,
            hold_days=cfg.hold_days,
            top_k=None,
            initial_capital=100_000.0,
            confidence_weight=False,
            earnings_filter=False,
            pead_boost=False,
            semiconductor_short_gate=False,
            gate_threshold=0.0,
        )

    model_ret = pd.Series(
        model_bt["daily_returns"],
        index=pd.to_datetime(model_bt["daily_dates"]).normalize(),
    ).sort_index()
    ew_ret = pd.Series(
        ew_bt["daily_returns"],
        index=pd.to_datetime(ew_bt["daily_dates"]).normalize(),
    ).sort_index()
    ew_ret = ew_ret.reindex(model_ret.index).fillna(0.0)

    market_proxy = (
        df.groupby("date")["forward_return"].mean().reindex(model_ret.index).fillna(0.0)
    )
    if len(market_proxy) > 1 and cfg.commission_bps > 0:
        c = cfg.commission_bps / 10_000.0
        market_proxy.iloc[0] -= c
        market_proxy.iloc[-1] -= c

    m = _returns_to_metrics(model_ret)
    e = _returns_to_metrics(ew_ret)
    p = _returns_to_metrics(market_proxy)
    return {
        "model_annual_return": m["annual_return"],
        "model_sharpe": m["sharpe"],
        "model_max_drawdown": m["max_drawdown"],
        "equal_weight_annual_return": e["annual_return"],
        "equal_weight_sharpe": e["sharpe"],
        "equal_weight_max_drawdown": e["max_drawdown"],
        "market_proxy_annual_return": p["annual_return"],
        "market_proxy_sharpe": p["sharpe"],
        "market_proxy_max_drawdown": p["max_drawdown"],
    }


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for scen, g in rows.groupby("scenario"):
        out_rows.append(
            {
                "scenario": scen,
                "paths": int(len(g)),
                "model_ann_mean": float(g["model_annual_return"].mean()),
                "model_ann_p10": float(g["model_annual_return"].quantile(0.10)),
                "model_ann_p50": float(g["model_annual_return"].median()),
                "model_ann_p90": float(g["model_annual_return"].quantile(0.90)),
                "model_sharpe_mean": float(g["model_sharpe"].mean()),
                "model_mdd_mean": float(g["model_max_drawdown"].mean()),
                "equal_weight_ann_mean": float(g["equal_weight_annual_return"].mean()),
                "equal_weight_sharpe_mean": float(g["equal_weight_sharpe"].mean()),
                "market_proxy_ann_mean": float(g["market_proxy_annual_return"].mean()),
                "market_proxy_sharpe_mean": float(g["market_proxy_sharpe"].mean()),
                "p_model_ann_gt_equal_weight": float(
                    (g["model_annual_return"] > g["equal_weight_annual_return"]).mean()
                ),
                "p_model_ann_gt_market_proxy": float(
                    (g["model_annual_return"] > g["market_proxy_annual_return"]).mean()
                ),
                "p_model_sharpe_gt_equal_weight": float(
                    (g["model_sharpe"] > g["equal_weight_sharpe"]).mean()
                ),
                "p_model_sharpe_gt_market_proxy": float(
                    (g["model_sharpe"] > g["market_proxy_sharpe"]).mean()
                ),
                "p_model_mdd_better_than_equal_weight": float(
                    (g["model_max_drawdown"] > g["equal_weight_max_drawdown"]).mean()
                ),
            }
        )
    return pd.DataFrame(out_rows).sort_values("scenario").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic market/news stress suite")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/meta_ensemble",
    )
    parser.add_argument(
        "--blend-checkpoint-dirs",
        type=str,
        default=None,
        help="Optional comma-separated extra checkpoints to blend",
    )
    parser.add_argument(
        "--blend-weights",
        type=str,
        default=None,
        help="Optional comma-separated weights for primary+extras or extras-only",
    )
    parser.add_argument("--signal-threshold", type=float, default=0.52)
    parser.add_argument("--short-threshold", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=(
            "vol_regime,crash_rebound,whipsaw_chop,news_chaos,anti_signal,mixed_chaos,"
            "liquidity_vacuum,policy_whipsaw,news_hallucination,theme_bubble_burst"
        ),
    )
    parser.add_argument("--n-paths", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    long_only = True if not args.allow_shorts else False
    cfg = StressConfig(
        signal_threshold=float(args.signal_threshold),
        short_threshold=args.short_threshold,
        confidence=float(args.confidence),
        long_only=long_only,
        commission_bps=float(args.commission_bps),
        hold_days=int(args.hold_days),
        top_k=int(args.top_k),
        max_positions=int(args.max_positions),
    )

    out_dir = Path(
        args.output_dir
        or f"artifacts/synthetic_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = deduplicate_predictions(load_predictions(args.checkpoint_dir))
    blend_dirs = _parse_csv_str(args.blend_checkpoint_dirs)
    blend_weights = _parse_csv_float(args.blend_weights)
    if blend_dirs:
        pred = blend_prediction_probabilities(
            primary_df=pred,
            checkpoint_dirs=blend_dirs,
            blend_weights=blend_weights or None,
        )

    base = _build_base_arrays(pred)
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]

    rows = []
    for scen_i, scenario in enumerate(scenarios):
        for path_i in range(int(args.n_paths)):
            seed = int(args.seed + scen_i * 100000 + path_i * 37)
            ret, prob = _simulate_scenario(base=base, scenario=scenario, seed=seed)
            metrics = _evaluate_synthetic_path(base, ret, prob, cfg)
            row = {"scenario": scenario, "path": path_i + 1, "seed": seed}
            row.update(metrics)
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    summary_df = _aggregate(detail_df)

    detail_df.to_csv(out_dir / "scenario_path_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "scenario_summary.csv", index=False)

    summary_payload = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "checkpoint_dir": args.checkpoint_dir,
        "blend_checkpoint_dirs": blend_dirs,
        "blend_weights": blend_weights,
        "scenarios": scenarios,
        "n_paths": int(args.n_paths),
        "output_dir": str(out_dir),
        "overall": {
            "p_model_ann_gt_equal_weight": float(
                (detail_df["model_annual_return"] > detail_df["equal_weight_annual_return"]).mean()
            ),
            "p_model_ann_gt_market_proxy": float(
                (detail_df["model_annual_return"] > detail_df["market_proxy_annual_return"]).mean()
            ),
            "p_model_sharpe_gt_equal_weight": float(
                (detail_df["model_sharpe"] > detail_df["equal_weight_sharpe"]).mean()
            ),
            "p_model_sharpe_gt_market_proxy": float(
                (detail_df["model_sharpe"] > detail_df["market_proxy_sharpe"]).mean()
            ),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nSynthetic stress suite complete")
    print(f"Output: {out_dir}")
    print(summary_df.to_string(index=False))
    print(
        "\nOverall P(model ann > equal_weight):",
        f"{summary_payload['overall']['p_model_ann_gt_equal_weight']:.1%}",
    )
    print(
        "Overall P(model ann > market_proxy):",
        f"{summary_payload['overall']['p_model_ann_gt_market_proxy']:.1%}",
    )


if __name__ == "__main__":
    main()
