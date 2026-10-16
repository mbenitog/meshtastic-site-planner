from pathlib import Path
from unittest.mock import patch
import backend.ultra.main as ultra_main
from fastapi.testclient import TestClient


def test_create_ultra_job_adopts_surface_from_prior_job(tmp_path: Path):
    import sys
    mod = sys.modules.get("backend.ultra.main", ultra_main)
    client = TestClient(mod.app)
    req = {
        "lat": 39.434044,
        "lon": -0.397694,
        "radius_km": 30.0,
        "resolution_m": 2.5,
        "tx_height_m": 80.0,
        "rx_height_m": 1.5,
        "tx_power_w": 1.0,
        "tx_gain_dbi": 5.0,
        "rx_gain_dbi": 2.15,
        "rx_sensitivity_dbm": -120.0,
        "frequency_mhz": 868.0,
    }

    # Simulate a prior job with completed surface and 2 tiles
    prior_id = "test-prior-job-1234"
    prior_dir = tmp_path / prior_id
    prior_dir.mkdir(parents=True, exist_ok=True)
    (prior_dir / "surface_done.flag").write_text("done")
    (prior_dir / "surface_i16le.bin").write_bytes(b"\x00\x00" * 100)
    (prior_dir / "surface_meta.json").write_text('{"width": 10, "height": 10, "path": "surface_i16le.bin"}')
    (prior_dir / "runner_input.json").write_text('{"surface_path": "surface_i16le.bin"}')
    (prior_dir / "coverage_x0_y0.signal_i16le.bin").write_bytes(b"\x00" * 200)

    # Use exact dict that payload.model_dump() produces
    validated_req = mod.UltraJobRequest(**req).model_dump()

    mod.jobs[prior_id] = {
        "status": "failed",
        "request": validated_req,
        "progress": {"phase": "compute", "fraction": 0.1},
    }

    prefix = mod.ULTRA_API_PREFIX.rstrip("/")
    jobs_url = f"{prefix}/ultra/jobs" if prefix else "/ultra/jobs"

    with patch.object(mod, "job_dir", side_effect=lambda jid: tmp_path / jid), \
         patch.object(mod, "JOB_ROOT", tmp_path), \
         patch.object(mod, "run_ultra_job"):
        resp = client.post(jobs_url, json=req)
        assert resp.status_code == 200, resp.text
        new_job_id = resp.json()["job_id"]
        assert new_job_id != prior_id

        # Verify new job directory adopted the cached surface and partial tiles!
        new_dir = tmp_path / new_job_id
        assert (new_dir / "surface_done.flag").exists()
        assert (new_dir / "surface_i16le.bin").exists()
        assert (new_dir / "surface_meta.json").exists()
        assert (new_dir / "coverage_x0_y0.signal_i16le.bin").exists()
