"""
Parameter sweep for backtest optimization — optimized version.

Phase 1: Quick sweep of total return across all combos (no year-by-year)
Phase 2: Year-by-year alpha analysis for top-N candidates

Usage:
    PYTHONPATH=/home/regulus/Trade python3 -u -m quantum_alpha.sweep_params
"""

import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np

from quantum_alpha.backtest_clean import (
    load_predictions,
    deduplicate_predictions,
    compute_signals,
    backtest_equal_weight,
)

# SPY annual returns (for alpha calculation)
SPY_RETURNS = {
    2011: 0.009,
    2012: 0.142,
    2013: 0.290,
    2014: 0.146,
    2015: 0.013,
    2016: 0.136,
    2017: 0.208,
    2018: -0.052,
    2019: 0.311,
    2020: 0.172,
    2021: 0.305,
    2022: -0.186,
    2023: 0.267,
    2024: 0.256,
    2025: 0.180,
}


def run_sweep():
    ckpt_dir = Path(__file__).parent / "models" / "checkpoints" / "meta_ensemble"

    print("Loading predictions...")
    df = load_predictions(ckpt_dir)
    df = deduplicate_predictions(df)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Exclude 2026 (partial year)
    df = df[df["year"] <= 2025]

    # Parameter grid
    signal_thresholds = [
        0.50,
        0.505,
        0.51,
        0.515,
        0.52,
        0.525,
        0.53,
        0.54,
        0.55,
        0.56,
        0.58,
        0.60,
    ]
    max_positions_list = [5, 10, 15, 20, 25, 30]
    top_k_list = [None, 5, 10, 15, 20]  # None = all
    conf_weight_list = [False, True]

    # =====================================================================
    # PHASE 1: Quick sweep — total return only (1 backtest per combo)
    # =====================================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: Quick total-return sweep")
    print("=" * 80)

    results = []
    combo_idx = 0
    total_combos = 0

    for sig_thresh in signal_thresholds:
        # Compute signals once per threshold
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        sig_df = compute_signals(df, signal_threshold=sig_thresh, long_only=True)
        sys.stdout = old_stdout

        n_active = (sig_df["raw_signal"] != 0).sum()
        pct_active = n_active / len(sig_df) * 100

        for max_pos in max_positions_list:
            for top_k in top_k_list:
                for cw in conf_weight_list:
                    total_combos += 1

                    # Skip nonsensical: top_k > max_positions
                    if top_k is not None and top_k > max_pos:
                        continue

                    combo_idx += 1

                    try:
                        res = backtest_equal_weight(
                            sig_df,
                            max_positions=max_pos,
                            commission_bps=5.0,
                            hold_days=12,
                            top_k=top_k,
                            confidence_weight=cw,
                        )

                        if "error" in res:
                            continue

                        results.append(
                            {
                                "signal_thresh": sig_thresh,
                                "max_pos": max_pos,
                                "top_k": top_k if top_k else "all",
                                "conf_weight": cw,
                                "total_return": res["total_return"],
                                "sharpe": res["sharpe"],
                                "max_dd": res["max_drawdown"],
                                "win_rate": res["win_rate"],
                                "n_trades": res["total_trades"],
                                "avg_pos": res["avg_positions_per_day"],
                                "pct_active": pct_active,
                            }
                        )

                    except Exception as e:
                        pass

                    if combo_idx % 20 == 0:
                        print(
                            f"  Progress: {combo_idx} combos tested, {len(results)} valid",
                            flush=True,
                        )

    print(f"\n  Phase 1 complete: {len(results)} valid configurations\n")

    results_df = pd.DataFrame(results)

    # Show top 30 by total return
    print("=" * 120)
    print("  TOP 30 BY TOTAL RETURN (Phase 1)")
    print("=" * 120)
    top_total = results_df.sort_values("total_return", ascending=False).head(30)
    print(format_table_phase1(top_total))

    # Show top 30 by Sharpe
    print("\n" + "=" * 120)
    print("  TOP 30 BY SHARPE RATIO (Phase 1)")
    print("=" * 120)
    top_sharpe = results_df.sort_values("sharpe", ascending=False).head(30)
    print(format_table_phase1(top_sharpe))

    # =====================================================================
    # PHASE 2: Year-by-year alpha for top 30 candidates
    # =====================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2: Year-by-year alpha analysis for top 30 candidates")
    print("=" * 80)

    # Get unique top candidates (union of top-return and top-sharpe)
    top_indices = set(top_total.index.tolist() + top_sharpe.index.tolist())
    candidates = results_df.loc[list(top_indices)].copy()

    detailed_results = []

    for idx, row in candidates.iterrows():
        sig_thresh = row["signal_thresh"]
        max_pos = row["max_pos"]
        top_k = row["top_k"]
        cw = row["conf_weight"]

        # Recompute signals
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        sig_df = compute_signals(df, signal_threshold=sig_thresh, long_only=True)
        sys.stdout = old_stdout

        top_k_val = None if top_k == "all" else int(top_k)

        yearly_alphas = {}
        years_beating_spy = 0
        years = sorted(sig_df["year"].unique())

        for year in years:
            year_df = sig_df[sig_df["year"] == year]
            n_yr_active = (year_df["raw_signal"] != 0).sum()
            if n_yr_active == 0:
                yearly_alphas[year] = -SPY_RETURNS.get(year, 0)
            else:
                yr_res = backtest_equal_weight(
                    year_df,
                    max_positions=int(max_pos),
                    commission_bps=5.0,
                    hold_days=12,
                    top_k=top_k_val,
                    confidence_weight=cw,
                )
                yr_ret = yr_res.get("total_return", 0.0)
                spy_ret = SPY_RETURNS.get(year, 0)
                alpha = yr_ret - spy_ret
                yearly_alphas[year] = alpha
                if alpha > 0:
                    years_beating_spy += 1

        avg_alpha = np.mean(list(yearly_alphas.values()))
        median_alpha = np.median(list(yearly_alphas.values()))
        worst_alpha = min(yearly_alphas.values())

        detailed_results.append(
            {
                "signal_thresh": sig_thresh,
                "max_pos": max_pos,
                "top_k": top_k,
                "conf_weight": cw,
                "total_return": row["total_return"],
                "sharpe": row["sharpe"],
                "max_dd": row["max_dd"],
                "win_rate": row["win_rate"],
                "avg_pos": row["avg_pos"],
                "avg_alpha": avg_alpha,
                "median_alpha": median_alpha,
                "worst_alpha": worst_alpha,
                "years_beat_spy": years_beating_spy,
                **{f"alpha_{y}": yearly_alphas.get(y, 0) for y in range(2011, 2026)},
            }
        )

    det_df = pd.DataFrame(detailed_results)

    print("\n" + "=" * 140)
    print("  TOP CANDIDATES — SORTED BY AVERAGE ANNUAL ALPHA")
    print("=" * 140)
    det_sorted = det_df.sort_values("avg_alpha", ascending=False)
    print(format_table_phase2(det_sorted))

    print("\n" + "=" * 140)
    print("  TOP CANDIDATES — SORTED BY YEARS BEATING SPY (then avg alpha)")
    print("=" * 140)
    det_consistent = det_df.sort_values(
        ["years_beat_spy", "avg_alpha"], ascending=[False, False]
    )
    print(format_table_phase2(det_consistent))

    print("\n" + "=" * 140)
    print("  TOP CANDIDATES — SORTED BY WORST-YEAR ALPHA (best worst case)")
    print("=" * 140)
    det_worst = det_df.sort_values("worst_alpha", ascending=False)
    print(format_table_phase2(det_worst))

    # Print year-by-year alpha for the #1 candidate by avg alpha
    best = det_sorted.iloc[0]
    print(f"\n{'=' * 80}")
    print(f"  BEST CANDIDATE DETAIL:")
    print(
        f"  signal_thresh={best['signal_thresh']}, max_pos={best['max_pos']}, "
        f"top_k={best['top_k']}, conf_weight={best['conf_weight']}"
    )
    print(
        f"  Total Return: {best['total_return']:+.1%}, Sharpe: {best['sharpe']:.3f}, "
        f"Avg Alpha: {best['avg_alpha']:+.1%}"
    )
    print(f"{'=' * 80}")
    for y in range(2011, 2026):
        alpha = best.get(f"alpha_{y}", 0)
        spy = SPY_RETURNS.get(y, 0)
        strat_ret = alpha + spy
        marker = " <<< LOST" if alpha < 0 else ""
        print(
            f"  {y}: Strategy {strat_ret:+.1%} vs SPY {spy:+.1%} = Alpha {alpha:+.1%}{marker}"
        )

    # Save full results
    out_path = Path(__file__).parent / "sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    det_out = Path(__file__).parent / "sweep_detailed.csv"
    det_df.to_csv(det_out, index=False)
    print(f"\n  Phase 1 results saved to {out_path}")
    print(f"  Phase 2 detailed results saved to {det_out}")


def format_table_phase1(df: pd.DataFrame) -> str:
    header = (
        f"  {'#':>3} | {'SigTh':>6} | {'MaxPos':>6} | {'TopK':>5} | {'CW':>3} | "
        f"{'TotalRet':>10} | {'Sharpe':>7} | {'MaxDD':>8} | {'WinRate':>7} | "
        f"{'AvgPos':>6} | {'Active%':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        line = (
            f"  {rank:>3} | "
            f"{row['signal_thresh']:>6.3f} | "
            f"{int(row['max_pos']):>6} | "
            f"{str(row['top_k']):>5} | "
            f"{'Y' if row['conf_weight'] else 'N':>3} | "
            f"{row['total_return']:>+9.1%} | "
            f"{row['sharpe']:>6.3f} | "
            f"{row['max_dd']:>7.2%} | "
            f"{row['win_rate']:>6.1%} | "
            f"{row['avg_pos']:>5.1f} | "
            f"{row['pct_active']:>6.1f}%"
        )
        lines.append(line)

    return "\n".join(lines)


def format_table_phase2(df: pd.DataFrame) -> str:
    header = (
        f"  {'SigTh':>6} | {'MaxPos':>6} | {'TopK':>5} | {'CW':>3} | "
        f"{'TotalRet':>10} | {'Sharpe':>7} | {'AvgAlpha':>9} | "
        f"{'MedAlpha':>9} | {'WorstAlpha':>10} | {'BeatSPY':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]

    for _, row in df.iterrows():
        line = (
            f"  {row['signal_thresh']:>6.3f} | "
            f"{int(row['max_pos']):>6} | "
            f"{str(row['top_k']):>5} | "
            f"{'Y' if row['conf_weight'] else 'N':>3} | "
            f"{row['total_return']:>+9.1%} | "
            f"{row['sharpe']:>6.3f} | "
            f"{row['avg_alpha']:>+8.1%} | "
            f"{row['median_alpha']:>+8.1%} | "
            f"{row['worst_alpha']:>+9.1%} | "
            f"{row['years_beat_spy']:>5.0f}/15"
        )
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    run_sweep()
