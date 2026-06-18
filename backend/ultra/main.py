from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
import json
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
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


ULTRA_API_PREFIX = os.environ.get("ULTRA_API_PREFIX", "/ultra-api").rstrip("/") or ""
ULTRA_PUBLIC_BASE_URL = os.environ.get("ULTRA_PUBLIC_BASE_URL", "").rstrip("/")


def _public_base_url(request: Request) -> str:
    """Return the absolute base URL the browser should use to reach this API.

    Order of precedence:
      1. ``ULTRA_PUBLIC_BASE_URL`` env var (lets operators pin a public origin
         regardless of reverse-proxy headers).
      2. ``X-Forwarded-Proto`` + ``X-Forwarded-Host`` from the request, so a
         reverse proxy that preserves ``Host`` works without extra config.
      3. The request's own scheme + host as a final fallback for direct
         connections.
    """
    if ULTRA_PUBLIC_BASE_URL:
        return ULTRA_PUBLIC_BASE_URL
    fwd_proto = request.headers.get("x-forwarded-proto")
    fwd_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if fwd_proto and fwd_host:
        return f"{fwd_proto}://{fwd_host}".rstrip("/")
    return str(request.base_url).rstrip("/")


router = APIRouter()
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


def artifact_urls(job_id: str, request: Request) -> dict[str, str]:
    out: dict[str, str] = {}
    artifact_dir = job_dir(job_id)
    for name, (filename, _) in ARTIFACT_FILES.items():
        if (artifact_dir / filename).exists():
            out[name] = str(
                request.url_for("get_ultra_artifact", job_id=job_id, artifact=name)
            )
    return out


def public_job(job_id: str, job: dict, request: Request) -> dict:
    return job | {"artifact_urls": artifact_urls(job_id, request)}


def set_progress(job: dict, phase: str, fraction: float) -> None:
    job["progress"] = {"phase": phase, "fraction": max(0.0, min(1.0, fraction))}


def is_cancel_requested(job_id: str) -> bool:
    job = jobs.get(job_id) or load_job(job_id)
    return bool(job and job.get("cancel_requested"))


def mark_cancelled(job_id: str, *, message: str = "Ultra job cancelled.") -> None:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        return
    job["status"] = "cancelled"
    job["message"] = message
    set_progress(job, "finalize", 1.0)
    jobs[job_id] = job
    persist_job(job_id, job)


def run_ultra_job(job_id: str, request: UltraJobRequest) -> None:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        return
    if job.get("cancel_requested"):
        mark_cancelled(job_id, message="Ultra job cancelled before execution started.")
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
        if is_cancel_requested(job_id):
            mark_cancelled(job_id, message="Ultra job cancelled after surface build.")
            return
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
        if native_result.status == "cancelled":
            mark_cancelled(job_id)
            return
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
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
    workers = min(
        max(1, int(os.environ.get("ULTRA_TILE_WORKERS", max(1, (os.cpu_count() or 1) // 2)))),
        max(1, len(tiles)),
        4,
    )
    job = jobs.get(job_id)
    if job is not None:
        job.setdefault("tiles", {})["workers"] = workers
        persist_job(job_id, job)

    pending_tiles = []
    for tile in tiles:
        sig_path = out_dir / f"coverage_x{tile.x0}_y{tile.y0}.signal_i16le.bin"
        if sig_path.exists():
            _advance_tile_progress(job_id, total)
        else:
            pending_tiles.append(tile)

    for batch_start in range(0, len(pending_tiles), workers):
        if is_cancel_requested(job_id):
            from .runner import NativeRunResult
            return NativeRunResult(
                status="cancelled",
                model="itm_projected_grid",
                message="Ultra job cancelled during tiled native execution.",
            )
        batch = pending_tiles[batch_start : batch_start + workers]

        def run_tile(tile):
            cmd = _build_cmd(binary, artifact, tile, request, tx_x, tx_y, out_prefix)
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            return tile

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_tile, tile): tile for tile in batch}
            for future in as_completed(future_map):
                tile = future_map[future]
                try:
                    future.result()
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


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/terrain/sample")
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


@router.post("/coverage/probe")
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


@router.post("/surface/sample")
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


@router.post("/ultra/jobs")
def create_ultra_job(payload: UltraJobRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid4())
    estimate = estimate_grid(payload.radius_km, payload.resolution_m)
    job = {
        "status": "queued",
        "request": payload.model_dump(),
        "estimate": estimate,
        "message": "Queued for tiled projected-grid ITM execution.",
        "cancel_requested": False,
    }
    set_progress(job, "terrain", 0.02)
    jobs[job_id] = job
    persist_job(job_id, job)
    background_tasks.add_task(run_ultra_job, job_id, payload)

    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
        "estimate": estimate,
        "artifact_urls": {},
    }


@router.get("/ultra/jobs/{job_id}")
def get_ultra_job(job_id: str, request: Request) -> dict:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    jobs[job_id] = job
    return public_job(job_id, job, request)


@router.post("/ultra/jobs/{job_id}/cancel")
def cancel_ultra_job(job_id: str, request: Request) -> dict:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") in {"coverage_ready", "failed", "cancelled"}:
        jobs[job_id] = job
        return public_job(job_id, job, request)
    job["cancel_requested"] = True
    if job.get("status") == "queued":
        job["message"] = "Ultra job cancellation requested."
    jobs[job_id] = job
    persist_job(job_id, job)
    return public_job(job_id, job, request)


@router.get("/ultra/jobs/{job_id}/artifacts/{artifact}")
def get_ultra_artifact(job_id: str, artifact: ArtifactName) -> FileResponse:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    filename, media_type = ARTIFACT_FILES[artifact]
    path = job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=media_type, filename=filename)


app = FastAPI(title="Meshtastic Site Planner Ultra DSM Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=ULTRA_API_PREFIX)
