"""App-managed lifecycle helper for a local OpenBB API process."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests


@dataclass
class OpenBBAPIManager:
    enabled: bool
    base_url: str = "http://127.0.0.1:6900"
    startup_timeout_s: float = 20.0
    start_cmd: Optional[str] = None
    proc: Optional[subprocess.Popen] = None

    def _health_paths(self):
        return ["health", "api/v1/health", "api/v1/system/health"]

    def health(self) -> bool:
        if not self.enabled:
            return False
        for path in self._health_paths():
            url = f"{self.base_url.rstrip('/')}/{path}"
            try:
                r = requests.get(url, timeout=0.75)
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def start(self) -> Dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "started": False, "healthy": False, "reason": "disabled"}

        if self.health():
            return {"enabled": True, "started": False, "healthy": True, "reused": True}

        cmd = self.start_cmd or os.getenv("OPENBB_API_START_CMD", "").strip()
        if not cmd:
            return {
                "enabled": True,
                "started": False,
                "healthy": False,
                "reason": "missing_openbb_api_start_cmd",
            }

        try:
            argv = shlex.split(cmd)
            self.proc = subprocess.Popen(  # noqa: S603
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            return {
                "enabled": True,
                "started": False,
                "healthy": False,
                "reason": f"spawn_failed:{exc}",
            }

        deadline = time.time() + max(float(self.startup_timeout_s), 1.0)
        while time.time() < deadline:
            if self.health():
                return {
                    "enabled": True,
                    "started": True,
                    "healthy": True,
                    "pid": int(self.proc.pid) if self.proc else None,
                }
            time.sleep(0.35)

        return {
            "enabled": True,
            "started": True,
            "healthy": False,
            "pid": int(self.proc.pid) if self.proc else None,
            "reason": "startup_timeout",
        }

    def stop(self) -> Dict[str, object]:
        if self.proc is None:
            return {"stopped": False, "reason": "not_managed"}

        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
            code = self.proc.returncode
            self.proc = None
            return {"stopped": True, "returncode": code}
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
            return {"stopped": True, "returncode": None, "forced": True}


__all__ = ["OpenBBAPIManager"]
