#!/usr/bin/env python3
"""Simple terminal monitor for realtime paper sessions."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def _latest_session(root: Path) -> Path | None:
    candidates = [p for p in root.glob("realtime_paper_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists() or n <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return lines[-n:]


def _fmt_money(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "n/a"


def _fmt_pct(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "n/a"


def render(session_dir: Path, trades_to_show: int) -> str:
    status_path = session_dir / "live_status.json"
    trades_path = session_dir / "live_trades.jsonl"

    if not status_path.exists():
        return f"Waiting for status file: {status_path}"

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Failed to parse status: {exc}"

    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Quantum Alpha Live Monitor   now={now}")
    lines.append(f"session={session_dir}")
    lines.append("=" * 88)

    lines.append(
        " ".join(
            [
                f"status={status.get('status', 'n/a')}",
                f"strategy={status.get('strategy_type', 'n/a')}",
                f"latest_bar={status.get('latest_bar', 'n/a')}",
                f"cycle={status.get('cycle', 'n/a')}",
            ]
        )
    )
    lines.append(
        " ".join(
            [
                f"equity={_fmt_money(status.get('equity'))}",
                f"cash={_fmt_money(status.get('cash'))}",
                f"pnl={_fmt_money(status.get('profit_dollars'))}",
                f"ret={_fmt_pct(status.get('return_pct'))}",
                f"chg={_fmt_money(status.get('equity_change_dollars'))}",
            ]
        )
    )
    lines.append(
        " ".join(
            [
                f"trades={status.get('trades', 'n/a')}",
                f"positions={status.get('positions', 'n/a')}",
                f"symbols_total={status.get('symbols_total', 'n/a')}",
                f"symbols_trade={status.get('symbols_trade', 'n/a')}",
                f"news_refresh={status.get('news_last_refresh_utc', 'n/a')}",
            ]
        )
    )

    lines.append("-" * 88)
    lines.append("Recent Trades:")
    trade_lines = _tail_lines(trades_path, trades_to_show)
    if not trade_lines:
        lines.append("  (none yet)")
    else:
        for raw in trade_lines:
            try:
                t = json.loads(raw)
            except Exception:
                lines.append(f"  {raw}")
                continue
            lines.append(
                "  "
                + f"{t.get('timestamp', '')} {t.get('side', ''):>4} {t.get('symbol', ''):<8} "
                + f"qty={float(t.get('qty', 0.0)):.4f} "
                + f"px={float(t.get('exec_price', 0.0)):.4f} "
                + f"notional={_fmt_money(t.get('notional'))}"
            )

    lines.append("-" * 88)
    lines.append("Top Positions:")
    pos = status.get("positions_snapshot") or []
    if not pos:
        lines.append("  (none)")
    else:
        for row in pos[:10]:
            lines.append(
                "  "
                + f"{row.get('symbol', ''):<8} "
                + f"qty={float(row.get('qty', 0.0)):.4f} "
                + f"px={float(row.get('price', 0.0)):.4f} "
                + f"value={_fmt_money(row.get('value'))}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live terminal monitor for realtime paper trading.")
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Path to realtime_paper_<timestamp> directory. Defaults to latest in --artifacts-dir.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts root used when --session-dir is omitted.",
    )
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    parser.add_argument("--trades", type=int, default=15, help="How many latest trades to display")
    args = parser.parse_args()

    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = _latest_session(Path(args.artifacts_dir))
        if session_dir is None:
            print(f"No realtime_paper_* directory found under {args.artifacts_dir}")
            return

    while True:
        output = render(session_dir=session_dir, trades_to_show=max(1, int(args.trades)))
        print("\033[2J\033[H" + output, end="", flush=True)
        time.sleep(max(0.2, float(args.refresh_seconds)))


if __name__ == "__main__":
    main()
