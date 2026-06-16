from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .geo import meter_bbox_around
from .ign_dsm import IgnDsmClient
from .surface import SurfaceBuilder, SurfaceMode
from .artifacts import write_surface_artifact
from .runner import run_tiled_native_ultra, write_native_runner_input
from .render import write_coverage_png, write_png_world_file
from .tiler import plan_tiles


app = FastAPI(title="Meshtastic Site Planner Ultra DSM Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
ign = IgnDsmClient()
surface_builder = SurfaceBuilder(dsm_client=ign)
jobs: dict[str, dict] = {}
JOB_ROOT = Path(".cache/ultra-jobs")
TILE_MAX_CELLS = 250_000
ArtifactName = Literal[
    "job",
    "surface",
    "surface_meta",
    "runner_input",
    "coverage_signal",
    "coverage_mask",
    "coverage_meta",
    "coverage_png",
    "coverage_world",
]
ARTIFACT_FILES: dict[str, tuple[str, str]] = {
    "job": ("job.json", "application/json"),
    "surface": ("surface_i16le.bin", "application/octet-stream"),
    "surface_meta": ("surface_meta.json", "application/json"),
    "runner_input": ("runner_input.json", "application/json"),
    "coverage_signal": ("coverage.signal_i16le.bin", "application/octet-stream"),
    "coverage_mask": ("coverage.mask_u8.bin", "application/octet-stream"),
    "coverage_meta": ("coverage.meta.json", "application/json"),
    "coverage_png": ("coverage.png", "image/png"),
    "coverage_world": ("coverage.pgw", "text/plain"),
}


def job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def persist_job(job_id: str, job: dict) -> None:
    out = job_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


def load_job(job_id: str) -> dict | None:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_urls(job_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    base = job_dir(job_id)
    for name, (filename, _) in ARTIFACT_FILES.items():
        if (base / filename).exists():
            out[name] = f"/ultra/jobs/{job_id}/artifacts/{name}"
    return out


def public_job(job_id: str, job: dict) -> dict:
    return job | {"artifact_urls": artifact_urls(job_id)}


def set_progress(job: dict, phase: str, fraction: float) -> None:
    job["progress"] = {"phase": phase, "fraction": max(0.0, min(1.0, fraction))}


def run_ultra_job(job_id: str, request: UltraJobRequest) -> None:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        return
    job["status"] = "building_surface"
    job["message"] = "Building measured 2.5 m surface grid."
    set_progress(job, "terrain", 0.1)
    jobs[job_id] = job
    persist_job(job_id, job)
    try:
        grid = surface_builder.build(
            lat=request.lat,
            lon=request.lon,
            radius_m=request.radius_km * 1000,
            resolution_m=request.resolution_m,
            mode=request.surface_mode,
        )
        artifact = write_surface_artifact(grid, job_dir(job_id))
        runner_input = write_native_runner_input(artifact, job["request"], job_dir(job_id))
        job["surface"] = asdict(artifact)
        job["runner_input"] = asdict(runner_input)
        job["status"] = "running_native"
        job["message"] = "Running ITM/Longley-Rice over measured projected-grid terrain in tiles."
        set_progress(job, "compute", 0.4)
        tiles = plan_tiles(artifact.width, artifact.height, TILE_MAX_CELLS)
        job["tiles"] = {
            "count": len(tiles),
            "completed": 0,
        }
        persist_job(job_id, job)

        native_result = _run_tiles_with_progress(
            job_id=job_id,
            artifact=artifact,
            request=job["request"],
            tiles=tiles,
        )
        job["native_result"] = asdict(native_result)
        if native_result.status == "complete":
            png_path = write_coverage_png(
                signal_path=native_result.signal_path or "",
                mask_path=native_result.mask_path or "",
                out_path=job_dir(job_id) / "coverage.png",
                width=artifact.width,
                height=artifact.height,
                min_dbm=request.rx_sensitivity_dbm,
            )
            world_path = write_png_world_file(
                out_path=job_dir(job_id) / "coverage.pgw",
                width=artifact.width,
                height=artifact.height,
                bounds_wgs84=artifact.bounds_wgs84,
            )
            job["coverage_png"] = {"path": png_path, "world_path": world_path}
            job["status"] = "coverage_ready"
            set_progress(job, "finalize", 1.0)
            job["message"] = (
                "Native ultra coverage is ready. Model is ITM/Longley-Rice over the measured projected 2.5 m surface grid."
            )
        elif native_result.status == "missing_binary":
            job["status"] = "surface_ready"
            set_progress(job, "finalize", 0.75)
            job["message"] = native_result.message
        else:
            job["status"] = "failed"
            set_progress(job, "finalize", 1.0)
            job["error"] = f"native ultra runner failed: {native_result.message}"
    except Exception as exc:
        job["status"] = "failed"
        set_progress(job, "finalize", 1.0)
        job["error"] = f"surface build failed: {exc}"
    jobs[job_id] = job
    persist_job(job_id, job)


def _run_tiles_with_progress(job_id, artifact, request, tiles):
    """Run the tiled native runner, persisting tile-level progress into the
    in-memory job after each completed tile so the polling client sees a
    live fraction."""
    from .runner import run_tiled_native_ultra
    from .tiler import TileSpec
    # Inline the per-tile loop here so we can write per-tile progress; the
    # helper still owns the final aggregation step.
    import os
    import subprocess
    from pathlib import Path

    binary = Path(os.environ.get("ULTRA_CLI", "engine/build/ultra_cli")).resolve()
    if not binary.exists():
        from .runner import NativeRunResult
        return NativeRunResult(
            status="missing_binary",
            model="itm_projected_grid",
            message=f"native ultra runner not found at {binary}; run engine/build_native.sh",
        )

    out_dir = job_dir(job_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / "coverage"
    from .geo import latlon_to_utm30
    from .runner import _build_cmd

    tx_x, tx_y = latlon_to_utm30(request["lat"], request["lon"])
    total = len(tiles)
    for tile in tiles:
        sig_path = out_dir / f"coverage_x{tile.x0}_y{tile.y0}.signal_i16le.bin"
        if sig_path.exists():
            _advance_tile_progress(job_id, total)
            continue
        cmd = _build_cmd(binary, artifact, tile, request, tx_x, tx_y, out_prefix)
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=600
            )
        except Exception as exc:
            from .runner import NativeRunResult
            return NativeRunResult(
                status="failed",
                model="itm_projected_grid",
                message=f"tile ({tile.x0},{tile.y0}) failed: {exc}",
            )
        _advance_tile_progress(job_id, total)

    # Aggregation step is the same as the helper.
    return run_tiled_native_ultra(
        artifact=artifact,
        request=request,
        out_dir=out_dir,
        max_tile_cells=TILE_MAX_CELLS,
    )


def _advance_tile_progress(job_id: str, total: int) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    done = int(job.get("tiles", {}).get("completed", 0)) + 1
    job.setdefault("tiles", {})["completed"] = done
    fraction = 0.4 + 0.5 * (done / max(1, total))
    set_progress(job, "compute", fraction)
    persist_job(job_id, job)


def estimate_grid(radius_km: float, resolution_m: float) -> dict[str, float | int]:
    diameter_m = radius_km * 2000
    cells_per_side = int(diameter_m / resolution_m) + 1
    cells = cells_per_side * cells_per_side
    elevation_mb = cells * 2 / (1024 * 1024)
    signal_mask_mb = cells * 2 / (1024 * 1024)
    return {
        "cells_per_side": cells_per_side,
        "cells": cells,
        "elevation_mb": round(elevation_mb, 1),
        "signal_mask_mb": round(signal_mask_mb, 1),
        "minimum_grid_mb": round(elevation_mb + signal_mask_mb, 1),
    }


class TerrainSampleRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(25.0, gt=0, le=500)
    coverage_id: Literal["mds05", "mdsn_v025", "mdsn_e025"] = "mds05"


class UltraJobRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(..., gt=0, le=30)
    frequency_mhz: float = Field(..., gt=20, le=20000)
    tx_height_m: float = Field(..., ge=0)
    rx_height_m: float = Field(..., ge=0)
    tx_power_w: float = Field(..., gt=0)
    tx_gain_dbi: float = 0.0
    rx_gain_dbi: float = 0.0
    rx_sensitivity_dbm: float = -130.0
    ground_dielectric: float = 15.0
    ground_conductivity: float = 0.005
    atmosphere_bending: float = 301.0
    radio_climate: int = Field(5, ge=1, le=7)
    polarization: int = Field(1, ge=0, le=1)
    confidence: float = Field(0.95, gt=0, le=1)
    reliability: float = Field(0.95, gt=0, le=1)
    resolution_m: Literal[2.5] = 2.5
    surface_mode: SurfaceMode = "lod_dtm_plus_buildings"


class SurfaceSampleRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(25.0, gt=0, le=500)
    resolution_m: Literal[2.5] = 2.5
    mode: SurfaceMode = "dtm_plus_buildings_2_5m"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/terrain/sample")
def terrain_sample(request: TerrainSampleRequest) -> dict:
    min_x, min_y, max_x, max_y = meter_bbox_around(request.lat, request.lon, request.radius_m)
    try:
        grid = ign.fetch_arcgrid(
            coverage_id=request.coverage_id,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IGN DSM fetch failed: {exc}") from exc
    return {
        "coverage_id": request.coverage_id,
        "bbox_25830": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        "grid": asdict(grid) | {
            "values": None,
            "min_value": grid.min_value,
            "max_value": grid.max_value,
        },
        "note": "mdsn_* coverages are normalized heights above ground, not absolute RF terrain.",
    }


class CoverageProbeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(..., gt=0, le=10000)


@app.post("/coverage/probe")
def coverage_probe(request: CoverageProbeRequest) -> dict:
    """Return which IGN coverage layers are available for the bbox so the UI
    can pick a surface mode before submitting a job."""
    min_x, min_y, max_x, max_y = meter_bbox_around(request.lat, request.lon, request.radius_m)
    availability: dict[str, dict] = {}
    for coverage_id in ("mds05", "mdsn_v025", "mdsn_e025"):
        try:
            grid = ign.fetch_arcgrid(
                coverage_id=coverage_id,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            )
            availability[coverage_id] = {
                "available": True,
                "ncols": grid.ncols,
                "nrows": grid.nrows,
                "min_value": grid.min_value,
                "max_value": grid.max_value,
            }
        except Exception as exc:
            availability[coverage_id] = {
                "available": False,
                "error": str(exc),
            }
    mdsn_e025 = availability.get("mdsn_e025", {}).get("available", False)
    mdsn_v025 = availability.get("mdsn_v025", {}).get("available", False)
    mds05 = availability.get("mds05", {}).get("available", False)
    recommended = (
        "lod_dtm_plus_buildings"
        if mdsn_e025 or mds05
        else "dtm_only"
    )
    return {
        "bbox_25830": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        "availability": availability,
        "recommended_surface_mode": recommended,
    }


@app.post("/surface/sample")
def surface_sample(request: SurfaceSampleRequest) -> dict:
    try:
        grid = surface_builder.build(
            lat=request.lat,
            lon=request.lon,
            radius_m=request.radius_m,
            resolution_m=request.resolution_m,
            mode=request.mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IGN DSM surface build failed: {exc}") from exc
    return {
        "mode": grid.mode,
        "resolution_m": grid.resolution_m,
        "bbox_25830": {
            "min_x": grid.min_x,
            "min_y": grid.min_y,
            "max_x": grid.max_x,
            "max_y": grid.max_y,
        },
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "cells": grid.width * grid.height,
            "min_value": grid.min_value,
            "max_value": grid.max_value,
            "values": None,
        },
        "note": "2.5 m surface grid composes measured IGN DTM ground plus measured 2.5 m normalized DSM detail; no synthetic buildings are generated.",
    }


@app.post("/ultra/jobs")
def create_ultra_job(request: UltraJobRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid4())
    estimate = estimate_grid(request.radius_km, request.resolution_m)
    job = {
        "status": "queued",
        "request": request.model_dump(),
        "estimate": estimate,
        "message": "Queued for tiled projected-grid ITM execution.",
    }
    set_progress(job, "terrain", 0.02)
    jobs[job_id] = job
    persist_job(job_id, job)
    background_tasks.add_task(run_ultra_job, job_id, request)

    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
        "estimate": estimate,
        "artifact_urls": artifact_urls(job_id),
    }


@app.get("/ultra/jobs/{job_id}")
def get_ultra_job(job_id: str) -> dict:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    jobs[job_id] = job
    return public_job(job_id, job)


@app.get("/ultra/jobs/{job_id}/artifacts/{artifact}")
def get_ultra_artifact(job_id: str, artifact: ArtifactName) -> FileResponse:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    filename, media_type = ARTIFACT_FILES[artifact]
    path = job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=media_type, filename=filename)
