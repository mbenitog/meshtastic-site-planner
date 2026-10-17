from pathlib import Path
from unittest.mock import patch
import backend.ultra.main as ultra_main
from fastapi.testclient import TestClient

MOCK_PROBE_DICT = {
    "devices": [
        {
            "platform": 0,
            "device": 0,
            "name": "NVIDIA GeForce RTX 3060",
            "type": "GPU",
            "vram_mb": 12288,
            "compute_units": 28,
            "max_alloc_mb": 3072,
        },
        {
            "platform": 0,
            "device": 1,
            "name": "NVIDIA GeForce RTX 2070",
            "type": "GPU",
            "vram_mb": 7795,
            "compute_units": 36,
            "max_alloc_mb": 1948,
        },
    ]
}


def test_get_optimal_gpu_slots_scales_with_surface_footprint():
    binary = Path("/fake/ultra_cli_opencl")
    with patch("backend.ultra.main._probe_gpu", return_value=MOCK_PROBE_DICT):
        # 10km radius: small surface (~120 MB) -> 1 dedicated worker per GPU (2 total)
        slots_10km = ultra_main.get_optimal_gpu_slots(binary, "kernel_str", total_tiles=100, cpu_count=16, surf_bytes_mb=120)
        assert len(slots_10km) == 2

        # 30km radius: large surface (~1098 MB) -> 1 dedicated worker per GPU (2 total)
        slots_30km = ultra_main.get_optimal_gpu_slots(binary, "kernel_str", total_tiles=289, cpu_count=16, surf_bytes_mb=1098)
        assert len(slots_30km) == 2

        # Count workers per GPU for 30km
        rtx2070_workers = [s for s in slots_30km if "2070" in s["name"]]
        rtx3060_workers = [s for s in slots_30km if "3060" in s["name"]]
        assert len(rtx2070_workers) == 1
        assert len(rtx3060_workers) == 1

        # Verify that memory budget per worker is positive and safely bounded by 1920 MB (2GB OpenCL buffer limit)
        for s in slots_30km:
            assert 256 <= s["budget_mb"] <= 1920

    # Verify high-end GPU scaling (e.g. 24GB VRAM, 128 CUs): 1 dedicated worker with maximal safe single-buffer budget (1920 MB)
    high_end_probe = {
        "devices": [
            {
                "platform": 0,
                "device": 0,
                "name": "Generic High-End 24GB GPU",
                "type": "GPU",
                "vram_mb": 24576,
                "compute_units": 128,
                "max_alloc_mb": 6144,
            }
        ]
    }
    with patch("backend.ultra.main._probe_gpu", return_value=high_end_probe):
        slots_he = ultra_main.get_optimal_gpu_slots(binary, "kernel_str", total_tiles=289, cpu_count=32, surf_bytes_mb=1098)
        assert len(slots_he) == 1
        assert slots_he[0]["budget_mb"] == 1920

    # Verify explicit ULTRA_MAX_WORKERS_PER_GPU environment override allows multiple workers per GPU if requested
    with patch("backend.ultra.main._probe_gpu", return_value=high_end_probe), \
         patch.dict("os.environ", {"ULTRA_MAX_WORKERS_PER_GPU": "3"}):
        slots_multi = ultra_main.get_optimal_gpu_slots(binary, "kernel_str", total_tiles=289, cpu_count=32, surf_bytes_mb=1098)
        assert len(slots_multi) == 3


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


def test_resolve_native_binary_falls_back_to_cpu_when_no_opencl_platforms(tmp_path: Path):
    import sys
    mod = sys.modules.get("backend.ultra.main", ultra_main)

    cpu_bin = tmp_path / "ultra_cli"
    cpu_bin.write_text("#!/bin/sh\nexit 0\n")
    cpu_bin.chmod(0o755)

    opencl_bin = tmp_path / "ultra_cli_opencl"
    opencl_bin.write_text("#!/bin/sh\nexit 1\n")
    opencl_bin.chmod(0o755)

    kernel = tmp_path / "itm_kernel.cl"
    kernel.write_text("// kernel\n")

    with patch.dict("os.environ", {
        "ULTRA_CLI": str(opencl_bin),
        "ULTRA_OPENCL_KERNEL": str(kernel),
    }), patch.object(mod, "_probe_gpu", return_value=None):
        bin_path, resolved_kernel = mod._resolve_native_binary()
        # When OpenCL has no platforms (probe fails/returns None), it must fallback to CPU
        assert bin_path.name == "ultra_cli"
        assert resolved_kernel == ""


def test_resolve_native_binary_ultra_engine_cpu_forces_cpu(tmp_path: Path):
    import sys
    mod = sys.modules.get("backend.ultra.main", ultra_main)

    cpu_bin = tmp_path / "ultra_cli"
    cpu_bin.write_text("#!/bin/sh\nexit 0\n")
    cpu_bin.chmod(0o755)

    opencl_bin = tmp_path / "ultra_cli_opencl"
    opencl_bin.write_text("#!/bin/sh\nexit 0\n")
    opencl_bin.chmod(0o755)

    kernel = tmp_path / "itm_kernel.cl"
    kernel.write_text("// kernel\n")

    with patch.dict("os.environ", {
        "ULTRA_ENGINE": "cpu",
        "ULTRA_CLI": str(opencl_bin),
        "ULTRA_OPENCL_KERNEL": str(kernel),
    }), patch.object(mod, "_probe_gpu", return_value=MOCK_PROBE_DICT):
        bin_path, resolved_kernel = mod._resolve_native_binary()
        assert bin_path.name == "ultra_cli"
        assert resolved_kernel == ""


def test_resolve_native_binary_ultra_engine_auto_selects_gpu_when_present(tmp_path: Path):
    import sys
    mod = sys.modules.get("backend.ultra.main", ultra_main)

    cpu_bin = tmp_path / "ultra_cli"
    cpu_bin.write_text("#!/bin/sh\nexit 0\n")
    cpu_bin.chmod(0o755)

    opencl_bin = tmp_path / "ultra_cli_opencl"
    opencl_bin.write_text("#!/bin/sh\nexit 0\n")
    opencl_bin.chmod(0o755)

    kernel = tmp_path / "itm_kernel.cl"
    kernel.write_text("// kernel\n")

    with patch.dict("os.environ", {
        "ULTRA_ENGINE": "auto",
        "ULTRA_CLI": str(cpu_bin),
        "ULTRA_OPENCL_KERNEL": str(kernel),
    }), patch.object(mod, "_probe_gpu", return_value=MOCK_PROBE_DICT):
        bin_path, resolved_kernel = mod._resolve_native_binary()
        assert bin_path.name == "ultra_cli_opencl"
        assert resolved_kernel == str(kernel.resolve())


