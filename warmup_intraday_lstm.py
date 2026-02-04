"""
Warm up intraday LSTM checkpoints before market open.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from quantum_alpha.train_intraday_lstm import train_intraday

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM"]


def _next_warmup_time(now_et: datetime, run_at: dtime) -> datetime:
    candidate = now_et.replace(hour=run_at.hour, minute=run_at.minute, second=0, microsecond=0)
    if now_et.time() < run_at and now_et.weekday() < 5:
        return candidate
    days_ahead = 1
    while True:
        next_day = now_et + timedelta(days=days_ahead)
        if next_day.weekday() < 5:
            return next_day.replace(hour=run_at.hour, minute=run_at.minute, second=0, microsecond=0)
        days_ahead += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up intraday LSTM checkpoints before open")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--window-step", type=int, default=3)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(Path(__file__).parent / "models" / "intraday_checkpoints"),
    )
    parser.add_argument("--run-at", type=str, default="09:10")
    parser.add_argument("--wait", action="store_true", help="Wait until run-at time")
    parser.add_argument("--skip-if-past", action="store_true", help="Exit if run-at time has passed")
    args = parser.parse_args()

    try:
        hour, minute = [int(part) for part in args.run_at.split(":", 1)]
    except ValueError:
        raise ValueError("--run-at must be HH:MM in 24h format")

    run_at = dtime(hour=hour, minute=minute)
    tz = ZoneInfo("America/New_York")
    now_et = datetime.now(timezone.utc).astimezone(tz)

    if now_et.time() > run_at and args.skip_if_past:
        print("Run-at time already passed; skipping warm-up.")
        return

    if args.wait:
        target = _next_warmup_time(now_et, run_at)
        wait_seconds = max(0, (target - now_et).total_seconds())
        if wait_seconds > 0:
            print(f"Waiting until {target.strftime('%Y-%m-%d %H:%M')} ET for warm-up...")
            time.sleep(wait_seconds)

    checkpoint_root = Path(args.checkpoint_root)
    for symbol in args.symbols:
        checkpoint_dir = checkpoint_root / symbol.lower()
        if not checkpoint_dir.exists():
            print(f"No checkpoint directory for {symbol} at {checkpoint_dir}")
            continue
        print(f"Warm-up fine-tune: {symbol}")
        train_intraday(
            symbol=symbol,
            end_date=datetime.now(),
            lookback_days=args.lookback_days,
            interval=args.interval,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            window_step_size=args.window_step,
            checkpoint_dir=str(checkpoint_dir),
            chunk_days=args.chunk_days,
            resume_latest=True,
        )


if __name__ == "__main__":
    main()
