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
    assert body["status"] in ("queued", "coverage_ready")
    assert "job_id" in body
    if body["status"] == "queued":
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
            "radius_km": -1.0,  # <= 0 is invalid
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

    r0 = client.post(
        _url("/ultra/jobs"),
        json={
            "lat": 40.4,
            "lon": -3.7,
            "radius_km": 0.0,  # <= 0 is invalid
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
    assert r0.status_code == 422


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


def test_coverage_probe_large_radius_never_fails_or_downgrades():
    """Probing coverage for a 30km radius must not crash with WCS bbox limits and must recommend combined surface mode."""
    r = client.post(
        _url("/coverage/probe"),
        json={"lat": 39.435, "lon": -0.465, "radius_m": 30000.0},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["recommended_surface_mode"] == "lod_dtm_plus_buildings"
    assert "error" not in data.get("availability", {}).get("mdsn_e025", {})


def test_surface_builder_mixed_coverage_preserves_streets_and_rooftops():
    """Verify that SurfaceBuilder layers 2.5m buildings and streets sharply without being blurred by 5m DSM."""
    import numpy as np
    from backend.ultra.surface import SurfaceBuilder, ArcGrid
    from backend.ultra.coverage_json import CoverageGrid

    class MockIgn:
        pass

    class MockCoverage:
        pass

    def mock_fetch_dtm(min_x, min_y, max_x, max_y, *args, **kwargs):
        return CoverageGrid(
            width=10,
            height=10,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            values=[50.0] * 100,
        )

    def mock_fetch_arcgrid(cov, min_x, min_y, max_x, max_y, *args, **kwargs):
        if cov == "mdsn_e025":
            b_vals = np.full((10, 10), np.nan, dtype=np.float32)
            b_vals[1, 0] = 15.0  # Rooftop
            b_vals[1, 1] = 0.0   # Street at (0, 1)
            return ArcGrid(
                ncols=10,
                nrows=10,
                xllcorner=min_x,
                yllcorner=min_y,
                cellsize=2.5,
                values=b_vals.ravel(),
                nodata=-9999.0,
            )
        return ArcGrid(
            ncols=5,
            nrows=5,
            xllcorner=min_x,
            yllcorner=min_y,
            cellsize=5.0,
            values=[58.0] * 25,
            nodata=-9999.0,
        )

    builder = SurfaceBuilder(dsm_client=MockIgn(), coverage_client=MockCoverage())
    builder._try_fetch_dtm = mock_fetch_dtm
    builder._try_fetch_arcgrid = mock_fetch_arcgrid

    res = builder.build(lat=40.0, lon=-3.0, radius_m=11.25, resolution_m=2.5, mode="lod_dtm_plus_buildings")
    vals = res.values

    # Cell (0, 0) is a 2.5m rooftop: ground (50) + building (15) = 65m
    assert vals[0, 0] == 65.0
    assert res.sources[0, 0] == 1  # 2.5m LiDAR

    # Cell (0, 1) is a 2.5m street: ground (50) + street (0) = 50m (NOT blurred by 5m DSM which is 58m!)
    assert vals[0, 1] == 50.0
    assert res.sources[0, 1] == 1  # 2.5m LiDAR

    # Cell (5, 5) is outside 2.5m coverage: must use 5m DSM (58.0m)
    assert vals[5, 5] == 58.0
    assert res.sources[5, 5] == 3  # 5m DSM


