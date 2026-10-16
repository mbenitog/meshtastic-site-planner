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


def test_nan_sanitization_in_artifact_json(tmp_path: Path):
    """Artifact endpoint must sanitize non-standard NaN and Infinity to standard null."""
    import json
    import backend.ultra.main as main_mod

    jid = "test-nan-job"
    main_mod.persist_job(jid, {"status": "coverage_ready", "request": {}})
    jdir = main_mod.job_dir(jid)
    try:
        meta_file = jdir / "surface_meta.json"
        meta_file.write_text('{\n  "width": 100,\n  "min_value": NaN,\n  "max_value": Infinity\n}\n', encoding="utf-8")

        r = client.get(_url(f"/ultra/jobs/{jid}/artifacts/surface_meta"))
        assert r.status_code == 200
        # Must be valid standard JSON (allow_nan=False should not fail)
        parsed = json.loads(r.text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid constant: {x}")))
        assert parsed["min_value"] is None
        assert parsed["max_value"] is None
        assert parsed["width"] == 100
    finally:
        main_mod.jobs.pop(jid, None)
        import shutil
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)


def test_write_coverage_surface_with_nans_produces_strict_json(tmp_path: Path):
    """write_surface_artifact must never write literal NaN to surface_meta.json."""
    import json
    import numpy as np
    from backend.ultra.surface import SurfaceGrid
    from backend.ultra.artifacts import write_surface_artifact

    grid = SurfaceGrid(
        width=3,
        height=3,
        min_x=700000.0,
        min_y=4300000.0,
        resolution_m=2.5,
        mode="dtm_only",
        values=np.array([[np.nan, 10.0, np.nan], [20.0, np.nan, 30.0], [np.nan, np.nan, np.nan]]),
    )
    art = write_surface_artifact(grid, tmp_path)
    meta_text = Path(art.meta_path).read_text(encoding="utf-8")
    assert "NaN" not in meta_text
    assert "Infinity" not in meta_text
    # Standard JSON parse without NaN extensions
    parsed = json.loads(meta_text)
    assert parsed["min_value"] == 10.0
    assert parsed["max_value"] == 30.0


def test_create_ultra_job_reuses_completed_job(tmp_path: Path):
    """If an identical request has already completed and artifacts exist, return cached result."""
    import backend.ultra.main as main_mod

    req = {
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
        "surface_mode": "dtm_only",
        "ground_dielectric": 15.0,
        "ground_conductivity": 0.005,
        "atmosphere_bending": 301.0,
        "radio_climate": 5,
        "polarization": 1,
        "confidence": 0.95,
        "reliability": 0.95,
    }
    jid = "cached-job-123"
    jdir = main_mod.job_dir(jid)
    jdir.mkdir(parents=True, exist_ok=True)
    try:
        (jdir / "coverage.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (jdir / "coverage.signal_i16le.bin").write_bytes(b"\x00" * 10)
        (jdir / "surface_meta.json").write_text('{"width": 10}', encoding="utf-8")
        job_data = {
            "status": "coverage_ready",
            "request": req,
            "message": "Cached run ready.",
            "progress": {"phase": "finalize", "fraction": 1.0},
        }
        main_mod.persist_job(jid, job_data)
        main_mod.jobs[jid] = job_data

        r = client.post(_url("/ultra/jobs"), json=req)
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == jid
        assert data["status"] == "coverage_ready"
    finally:
        main_mod.jobs.pop(jid, None)
        import shutil
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)


