from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Literal

from .coverage_json import CoverageApiClient, DTM_5M_25830
from .geo import meter_bbox_around
from .ign_dsm import ArcGrid, IgnDsmClient


SurfaceMode = Literal[
    "lod_dtm_plus_buildings",
    "dtm_plus_buildings_2_5m",
    "dtm_plus_surface_2_5m",
    "dtm_only",
]

SURFACE_PROCESS_MIN_CELLS = 200_000
SURFACE_ROW_BLOCK = 64
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class SurfaceGrid:
    width: int
    height: int
    min_x: float
    min_y: float
    resolution_m: float
    mode: SurfaceMode
    values: list[float]
    sources: list[int] = field(default_factory=list)
    """Per-cell source/origin. Same shape as ``values``. 0 = dtm fallback
    (DTM 5 m only, no 2.5 m DSM/vegetation sample available).
    1 = dtm + mdsn_e025 (2.5 m building DSM), 2 = dtm + mdsn_v025
    (vegetation), 3 = mds05 absolute 5 m surface."""

    @property
    def max_x(self) -> float:
        return self.min_x + (self.width - 1) * self.resolution_m

    @property
    def max_y(self) -> float:
        return self.min_y + (self.height - 1) * self.resolution_m

    @property
    def min_value(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max_value(self) -> float:
        return max(self.values) if self.values else 0.0


def _sample_arcgrid_nearest(grid: ArcGrid, x: float, y: float) -> float | None:
    """Return the ArcGrid value at the cell nearest (x, y), or ``None`` if
    the nearest cell is the no-data sentinel so callers can fall back
    instead of inventing a number."""
    if grid.ncols <= 0 or grid.nrows <= 0:
        return None
    ix = round((x - grid.xllcorner) / grid.cellsize)
    north = grid.yllcorner + (grid.nrows - 1) * grid.cellsize
    iy = round((north - y) / grid.cellsize)
    ix = min(max(int(ix), 0), grid.ncols - 1)
    iy = min(max(int(iy), 0), grid.nrows - 1)
    value = grid.values[iy * grid.ncols + ix]
    if grid.nodata is not None and value == grid.nodata:
        return None
    return float(value)


def _compose_surface_rows(
    *,
    row_start: int,
    row_end: int,
    width: int,
    min_x: float,
    max_y: float,
    resolution_m: float,
    mode: SurfaceMode,
    terrain,
    buildings: ArcGrid | None,
    vegetation: ArcGrid | None,
    surface_5m: ArcGrid | None,
) -> tuple[int, list[float], list[int]]:
    values: list[float] = []
    sources: list[int] = []
    for y in range(row_start, row_end):
        py = max_y - y * resolution_m
        for x in range(width):
            px = min_x + x * resolution_m
            terrain_m = terrain.sample_nearest(px, py)
            building_raw = _sample_arcgrid_nearest(buildings, px, py) if buildings is not None else None
            vegetation_raw = _sample_arcgrid_nearest(vegetation, px, py) if vegetation is not None else None
            surface_5m_raw = _sample_arcgrid_nearest(surface_5m, px, py) if surface_5m is not None else None

            building_m = max(0.0, building_raw or 0.0)
            vegetation_m = max(0.0, vegetation_raw or 0.0)
            surface_5m_m = max(terrain_m, surface_5m_raw) if surface_5m_raw is not None else None

            if mode == "dtm_only":
                values.append(terrain_m)
                sources.append(0)
                continue

            if building_m > 0.0:
                obstruction = max(building_m, vegetation_m)
                if vegetation_m >= building_m > 0.0:
                    source = 2
                else:
                    source = 1
            elif vegetation_m > 0.0:
                obstruction = vegetation_m
                source = 2
            elif surface_5m_m is not None and mode == "lod_dtm_plus_buildings":
                values.append(surface_5m_m)
                sources.append(3)
                continue
            else:
                obstruction = 0.0
                source = 0
            values.append(terrain_m + obstruction)
            sources.append(source)
    return row_start, values, sources


def _surface_process_count(height: int, cells: int) -> int:
    env_value = os.environ.get("ULTRA_SURFACE_WORKERS", "").strip()
    if env_value:
        try:
            workers = int(env_value)
        except ValueError:
            workers = 1
    else:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, height))
    if cells < SURFACE_PROCESS_MIN_CELLS:
        return 1
    return workers


def _surface_mp_context():
    if sys.platform == "win32":
        return mp.get_context("spawn")
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


class SurfaceBuilder:
    def __init__(
        self,
        *,
        dsm_client: IgnDsmClient | None = None,
        coverage_client: CoverageApiClient | None = None,
    ) -> None:
        self.dsm = dsm_client or IgnDsmClient()
        self.coverages = coverage_client or CoverageApiClient()
        self.last_stats: dict[str, float | int | str | None] = {}

    def build(
        self,
        *,
        lat: float,
        lon: float,
        radius_m: float,
        resolution_m: float = 2.5,
        mode: SurfaceMode = "lod_dtm_plus_buildings",
    ) -> SurfaceGrid:
        started = time.perf_counter()
        min_x, min_y, max_x, max_y = meter_bbox_around(lat, lon, radius_m)
        width = int((max_x - min_x) / resolution_m) + 1
        height = int((max_y - min_y) / resolution_m) + 1
        cells = width * height
        stats: dict[str, float | int | str | None] = {
            "cells": cells,
            "width": width,
            "height": height,
            "mode": mode,
        }

        # Always fetch the DTM 5 m ground reference. This is mandatory: every
        # surface cell is at minimum DTM elevation.
        t0 = time.perf_counter()
        terrain = self._try_fetch_dtm(min_x, min_y, max_x, max_y)
        stats["fetch_dtm_s"] = round(time.perf_counter() - t0, 3)
        if terrain is None:
            raise RuntimeError(
                "DTM 5 m coverage is required for ultra surface but is not "
                "available for this bbox. Move closer to Spanish mainland or "
                "choose dtm_only mode if your client supports it."
            )

        # Fetch the absolute 5 m surface as the measured coarse fallback.
        # In LOD mode this preserves real buildings/trees where 2.5 m products
        # are unavailable instead of dropping immediately to bare-ground DTM.
        t0 = time.perf_counter()
        surface_5m = self._try_fetch_arcgrid("mds05", min_x, min_y, max_x, max_y, tiled=True)
        stats["fetch_mds05_s"] = round(time.perf_counter() - t0, 3)

        # Fetch the 2.5 m building DSM and (optionally) the 2.5 m vegetation
        # layer. Each fetch is best-effort: if the bbox is outside IGN coverage
        # the corresponding layer is left as ``None`` and cells fall back to the
        # coarser measured surface. Large bboxes are split into WCS sub-requests by the client
        # because the IGN endpoint rejects oversized responses.
        t0 = time.perf_counter()
        buildings = self._try_fetch_arcgrid("mdsn_e025", min_x, min_y, max_x, max_y, tiled=True)
        stats["fetch_mdsn_e025_s"] = round(time.perf_counter() - t0, 3)
        want_vegetation = mode == "dtm_plus_surface_2_5m"
        vegetation = None
        if want_vegetation:
            t0 = time.perf_counter()
            vegetation = self._try_fetch_arcgrid("mdsn_v025", min_x, min_y, max_x, max_y, tiled=True)
            stats["fetch_mdsn_v025_s"] = round(time.perf_counter() - t0, 3)

        # In strict 2.5 m modes we require the 2.5 m building layer. If it is
        # not available the surface build fails because the user explicitly
        # asked for the highest resolution.
        require_buildings = mode in ("dtm_plus_buildings_2_5m", "dtm_plus_surface_2_5m")
        if require_buildings and buildings is None:
            raise RuntimeError(
                "Requested 2.5 m DSM coverage is not available for this bbox. "
                "Use lod_dtm_plus_buildings or dtm_only, or move closer to a "
                "city covered by mdsn_e025."
            )

        workers = _surface_process_count(height, cells)
        stats["surface_workers"] = workers
        t0 = time.perf_counter()
        row_ranges = [
            (row_start, min(height, row_start + SURFACE_ROW_BLOCK))
            for row_start in range(0, height, SURFACE_ROW_BLOCK)
        ]
        if workers <= 1 or len(row_ranges) <= 1:
            _, values, sources = _compose_surface_rows(
                row_start=0,
                row_end=height,
                width=width,
                min_x=min_x,
                max_y=max_y,
                resolution_m=resolution_m,
                mode=mode,
                terrain=terrain,
                buildings=buildings,
                vegetation=vegetation,
                surface_5m=surface_5m,
            )
        else:
            parts: list[tuple[int, list[float], list[int]]] = []
            with ProcessPoolExecutor(max_workers=workers, mp_context=_surface_mp_context()) as executor:
                futures = [
                    executor.submit(
                        _compose_surface_rows,
                        row_start=row_start,
                        row_end=row_end,
                        width=width,
                        min_x=min_x,
                        max_y=max_y,
                        resolution_m=resolution_m,
                        mode=mode,
                        terrain=terrain,
                        buildings=buildings,
                        vegetation=vegetation,
                        surface_5m=surface_5m,
                    )
                    for row_start, row_end in row_ranges
                ]
                for future in futures:
                    parts.append(future.result())
            parts.sort(key=lambda part: part[0])
            values = []
            sources = []
            for _, part_values, part_sources in parts:
                values.extend(part_values)
                sources.extend(part_sources)
        stats["compose_surface_s"] = round(time.perf_counter() - t0, 3)
        stats["total_surface_build_s"] = round(time.perf_counter() - started, 3)
        self.last_stats = stats
        logger.info(
            "surface.build cells=%s mode=%s workers=%s dtm=%.3fs mds05=%.3fs mdsn_e025=%.3fs mdsn_v025=%s compose=%.3fs total=%.3fs",
            cells,
            mode,
            workers,
            stats.get("fetch_dtm_s", 0.0),
            stats.get("fetch_mds05_s", 0.0),
            stats.get("fetch_mdsn_e025_s", 0.0),
            stats.get("fetch_mdsn_v025_s"),
            stats.get("compose_surface_s", 0.0),
            stats.get("total_surface_build_s", 0.0),
        )

        return SurfaceGrid(
            width=width,
            height=height,
            min_x=min_x,
            min_y=min_y,
            resolution_m=resolution_m,
            mode=mode,
            values=values,
            sources=sources,
        )

    def _try_fetch_dtm(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ):
        try:
            return self.coverages.fetch_grid_tiled(
                collection_id=DTM_5M_25830,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            )
        except Exception:
            return None

    def _try_fetch_arcgrid(
        self,
        coverage_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        *,
        tiled: bool = False,
    ) -> ArcGrid | None:
        try:
            if tiled:
                return self.dsm.fetch_arcgrid_tiled(
                    coverage_id=coverage_id,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                )
            return self.dsm.fetch_arcgrid(
                coverage_id=coverage_id,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            )
        except Exception:
            return None
