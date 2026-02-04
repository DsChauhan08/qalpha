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
from typing import List, Dict, Any
import itertools
import json

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM"]


def _build_command(
    args: argparse.Namespace,
    symbol: str,
    checkpoint_dir: Path,
    config: Dict[str, Any],
) -> List[str]:
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
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--sequence-length",
        str(config["sequence_length"]),
        "--window-step",
        str(config["window_step"]),
        "--chunk-days",
        str(args.chunk_days),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.end:
        cmd.extend(["--end", args.end])
    return cmd


def _parse_int_list(raw, fallback: List[int]) -> List[int]:
    if raw is None:
        return fallback
    if isinstance(raw, list):
        values = []
        for item in raw:
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    values.append(int(part))
        return values or fallback
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return [int(p) for p in parts] or fallback


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"index": 0}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"index": 0}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concurrent intraday LSTM training jobs")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--window-step", type=int, default=3)
    parser.add_argument("--epochs-grid", nargs="*", default=None)
    parser.add_argument("--batch-grid", nargs="*", default=None)
    parser.add_argument("--sequence-grid", nargs="*", default=None)
    parser.add_argument("--window-grid", nargs="*", default=None)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=str(Path(__file__).parent / "models" / "intraday_checkpoints"))
    parser.add_argument("--log-dir", type=str, default=str(Path(__file__).parent / "reports" / "night_train_logs"))
    parser.add_argument("--jobs", type=int, default=3, help="Number of concurrent jobs to launch")
    parser.add_argument("--state-file", type=str, default=None)
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

    epoch_list = _parse_int_list(args.epochs_grid, [args.epochs])
    batch_list = _parse_int_list(args.batch_grid, [args.batch_size])
    seq_list = _parse_int_list(args.sequence_grid, [args.sequence_length])
    window_list = _parse_int_list(args.window_grid, [args.window_step])

    grid = list(itertools.product(epoch_list, batch_list, seq_list, window_list))
    if not grid:
        grid = [(args.epochs, args.batch_size, args.sequence_length, args.window_step)]

    job_space = []
    for epochs, batch_size, sequence_length, window_step in grid:
        for symbol in args.symbols:
            job_space.append(
                (
                    symbol,
                    {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "window_step": window_step,
                    },
                )
            )

    if not job_space:
        print("No jobs to run.")
        return

    state_path = Path(args.state_file) if args.state_file else (Path(args.log_dir) / "night_train_state.json")
    state = _load_state(state_path)
    start_idx = int(state.get("index", 0)) % len(job_space)
    total_jobs = min(max(1, args.jobs), len(job_space))
    selected = [job_space[(start_idx + i) % len(job_space)] for i in range(total_jobs)]
    state["index"] = (start_idx + total_jobs) % len(job_space)
    _save_state(state_path, state)

    processes = []
    start_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for symbol, config in selected:
        checkpoint_dir = checkpoint_root / symbol.lower()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            f"{symbol.lower()}_e{config['epochs']}_b{config['batch_size']}"
            f"_s{config['sequence_length']}_w{config['window_step']}_{start_ts}.log"
        )
        cmd = _build_command(args, symbol, checkpoint_dir, config)
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        processes.append((symbol, config, proc, str(log_path)))

    print("Started intraday training jobs:")
    for symbol, config, proc, log_path in processes:
        print(
            f"  {symbol}: PID={proc.pid} "
            f"epochs={config['epochs']} batch={config['batch_size']} "
            f"seq={config['sequence_length']} step={config['window_step']} "
            f"log={log_path}"
        )

    if args.wait:
        exit_codes = {}
        for symbol, _, proc, _ in processes:
            exit_codes[symbol] = proc.wait()
        print("\nExit codes:")
        for symbol, code in exit_codes.items():
            print(f"  {symbol}: {code}")


if __name__ == "__main__":
    main()
