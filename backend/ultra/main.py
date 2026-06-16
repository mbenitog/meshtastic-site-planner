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
from .runner import run_native_ultra, write_native_runner_input
from .render import write_coverage_png, write_png_world_file


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
MAX_SYNC_SURFACE_CELLS = 100_000
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
        job["message"] = "Running ITM/Longley-Rice over measured projected-grid terrain."
        set_progress(job, "compute", 0.5)
        persist_job(job_id, job)

        native_result = run_native_ultra(artifact, job["request"], job_dir(job_id))
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
    surface_mode: SurfaceMode = "dtm_plus_buildings_2_5m"


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
        "message": "Native 2.5 m DSM RF runner is not wired yet.",
    }
    set_progress(job, "terrain", 0.0)
    jobs[job_id] = job
    persist_job(job_id, job)

    if estimate["cells"] <= MAX_SYNC_SURFACE_CELLS:
        job["message"] = "Queued for direct projected-grid ITM execution."
        set_progress(job, "terrain", 0.02)
        persist_job(job_id, job)
        background_tasks.add_task(run_ultra_job, job_id, request)
    else:
        job["status"] = "queued"
        set_progress(job, "terrain", 0.0)
        job["message"] = (
            "Surface grid is too large for synchronous direct ITM materialization; "
            "background tiled execution is required."
        )
        persist_job(job_id, job)

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
