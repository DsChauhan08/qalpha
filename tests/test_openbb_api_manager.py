from types import SimpleNamespace

from quantum_alpha.execution.openbb_api_manager import OpenBBAPIManager


def test_openbb_api_manager_disabled_start():
    mgr = OpenBBAPIManager(enabled=False)
    out = mgr.start()
    assert out["enabled"] is False
    assert out["started"] is False


def test_openbb_api_manager_missing_cmd_when_enabled():
    mgr = OpenBBAPIManager(enabled=True, base_url="http://127.0.0.1:1", start_cmd="")
    out = mgr.start()
    assert out["enabled"] is True
    assert out["started"] is False
    assert "missing" in out.get("reason", "")


def test_openbb_api_manager_health_uses_known_paths(monkeypatch):
    calls = []

    def _fake_get(url, timeout=0.0):
        calls.append(url)
        if url.endswith("/api/v1/health"):
            return SimpleNamespace(status_code=200)
        raise RuntimeError("unreachable")

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)
    mgr = OpenBBAPIManager(enabled=True, base_url="http://local")
    assert mgr.health() is True
    assert any(u.endswith("/api/v1/health") for u in calls)


def test_openbb_api_manager_stop_without_managed_process():
    mgr = OpenBBAPIManager(enabled=True)
    out = mgr.stop()
    assert out["stopped"] is False
