"""
Launch concurrent intraday LSTM training jobs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM"]


def _build_command(args: argparse.Namespace, symbol: str, checkpoint_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_intraday_lstm",
        "--symbol",
        symbol,
        "--interval",
        args.interval,
        "--lookback-days",
        str(args.lookback_days),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--window-step",
        str(args.window_step),
        "--chunk-days",
        str(args.chunk_days),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.end:
        cmd.extend(["--end", args.end])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concurrent intraday LSTM training jobs")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--window-step", type=int, default=3)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=str(Path(__file__).parent / "models" / "intraday_checkpoints"))
    parser.add_argument("--log-dir", type=str, default=str(Path(__file__).parent / "reports" / "night_train_logs"))
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")

    processes = []
    start_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for symbol in args.symbols:
        checkpoint_dir = checkpoint_root / symbol.lower()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{symbol.lower()}_{start_ts}.log"
        cmd = _build_command(args, symbol, checkpoint_dir)
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        processes.append((symbol, proc, str(log_path)))

    print("Started intraday training jobs:")
    for symbol, proc, log_path in processes:
        print(f"  {symbol}: PID={proc.pid} log={log_path}")

    if args.wait:
        exit_codes = {}
        for symbol, proc, _ in processes:
            exit_codes[symbol] = proc.wait()
        print("\nExit codes:")
        for symbol, code in exit_codes.items():
            print(f"  {symbol}: {code}")


if __name__ == "__main__":
    main()
