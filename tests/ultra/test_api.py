import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.ultra.main import app, jobs, persist_job, ULTRA_API_PREFIX


client = TestClient(app)


def _url(path: str) -> str:
    prefix = ULTRA_API_PREFIX.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{prefix}{suffix}"


def test_health():
    r = client.get(_url("/health"))
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_terrain_sample_rejects_unknown_coverage():
    r = client.post(
        _url("/terrain/sample"),
        json={"lat": 40.4, "lon": -3.7, "radius_m": 50, "coverage_id": "nope"},
    )
    assert r.status_code == 422


def test_artifact_endpoint_whitelist():
    r = client.get(_url("/ultra/jobs/unknown-id/artifacts/coverage_signal"))
    assert r.status_code == 404
    r = client.get(_url("/ultra/jobs/unknown-id/artifacts/not_allowed"))
    assert r.status_code == 422


def test_create_ultra_job_returns_artifact_urls():
    r = client.post(
        _url("/ultra/jobs"),
        json={
            "lat": 40.4,
            "lon": -3.7,
            "radius_km": 0.05,
            "frequency_mhz": 869.525,
            "tx_height_m": 2,
            "rx_height_m": 1,
            "tx_power_w": 0.15,
            "tx_gain_dbi": 3,
            "rx_gain_dbi": 3,
            "rx_sensitivity_dbm": -130,
            "resolution_m": 2.5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    assert body["progress"]["phase"] == "terrain"


def test_cancel_ultra_job_marks_request():
    job_id = "cancel-me"
    job = {
        "status": "queued",
        "request": {},
        "estimate": {},
        "message": "Queued for tiled projected-grid ITM execution.",
        "cancel_requested": False,
        "progress": {"phase": "terrain", "fraction": 0.02},
    }
    jobs[job_id] = job
    persist_job(job_id, job)
    cancel = client.post(_url(f"/ultra/jobs/{job_id}/cancel"))
    assert cancel.status_code == 200
    body = cancel.json()
    assert body.get("cancel_requested") is True


def test_ultra_job_request_rejects_bad_radius():
    r = client.post(
        _url("/ultra/jobs"),
        json={
            "lat": 40.4,
            "lon": -3.7,
            "radius_km": 100,  # > 30 limit
            "frequency_mhz": 869.525,
            "tx_height_m": 2,
            "rx_height_m": 1,
            "tx_power_w": 0.15,
            "tx_gain_dbi": 3,
            "rx_gain_dbi": 3,
            "rx_sensitivity_dbm": -130,
            "resolution_m": 2.5,
        },
    )
    assert r.status_code == 422


def test_default_prefix_is_ultra_api():
    """The API must mount under /ultra-api by default so it can be reverse-
    proxied under the frontend's same origin."""
    assert ULTRA_API_PREFIX == "/ultra-api"


def test_prefix_can_be_overridden(monkeypatch: pytest.MonkeyPatch):
    """Setting ULTRA_API_PREFIX to empty mounts the routes at the root, which
    is useful for direct dev access on the backend's own port."""
    from importlib import reload
    import backend.ultra.main as main_mod

    monkeypatch.setenv("ULTRA_API_PREFIX", "")
    reload(main_mod)
    try:
        direct = TestClient(main_mod.app)
        assert direct.get("/health").status_code == 200
        assert direct.post(
            "/ultra/jobs",
            json={
                "lat": 40.4,
                "lon": -3.7,
                "radius_km": 0.05,
                "frequency_mhz": 869.525,
                "tx_height_m": 2,
                "rx_height_m": 1,
                "tx_power_w": 0.15,
                "tx_gain_dbi": 3,
                "rx_gain_dbi": 3,
                "rx_sensitivity_dbm": -130,
                "resolution_m": 2.5,
            },
        ).status_code == 200
    finally:
        monkeypatch.delenv("ULTRA_API_PREFIX", raising=False)
        reload(main_mod)
