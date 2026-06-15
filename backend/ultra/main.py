from __future__ import annotations

from dataclasses import asdict
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .geo import meter_bbox_around
from .ign_dsm import IgnDsmClient


app = FastAPI(title="Meshtastic Site Planner Ultra DSM Backend")
ign = IgnDsmClient()
jobs: dict[str, dict] = {}


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
    resolution_m: Literal[2.5, 5.0, 10.0, 15.0] = 2.5


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


@app.post("/ultra/jobs")
def create_ultra_job(request: UltraJobRequest) -> dict:
    job_id = str(uuid4())
    estimate = estimate_grid(request.radius_km, request.resolution_m)
    jobs[job_id] = {
        "status": "queued",
        "request": request.model_dump(),
        "estimate": estimate,
        "message": "Native 2.5 m DSM RF runner is not wired yet; terrain fetch prototype is available at /terrain/sample.",
    }
    return {"job_id": job_id, "status": "queued", "estimate": estimate}


@app.get("/ultra/jobs/{job_id}")
def get_ultra_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
