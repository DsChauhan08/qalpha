"""
Run rolling intraday backtests for each trained intraday checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from quantum_alpha.report_intraday_backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intraday backtests for all checkpoints")
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(Path(__file__).parent / "models" / "intraday_checkpoints"),
    )
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--horizon", type=str, default="1h")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "reports" / "intraday_backtests"),
    )
    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if not root.exists():
        print(f"No checkpoint root found at {root}")
        return

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        print("No intraday checkpoint directories found.")
        return

    for subdir in subdirs:
        symbol = subdir.name.upper()
        output_path = output_dir / f"{symbol.lower()}_{timestamp}.png"
        csv_path = output_dir / f"{symbol.lower()}_{timestamp}.csv"
        try:
            stats = run_backtest(
                symbol=symbol,
                interval=args.interval,
                lookback_days=args.lookback_days,
                checkpoint_dir=str(subdir),
                checkpoint=None,
                checkpoint_file=str(subdir / "latest_intraday.json"),
                horizon=args.horizon,
                window=None,
                cost_bps=args.cost_bps,
                scale=args.scale,
                output=str(output_path),
                csv_path=str(csv_path),
            )
            print(
                f"{symbol}: strategy={stats['strategy_return'] * 100:.2f}% "
                f"bench={stats['benchmark_return'] * 100:.2f}% "
                f"chart={output_path}"
            )
        except Exception as exc:
            print(f"{symbol}: backtest failed ({exc})")


if __name__ == "__main__":
    main()
