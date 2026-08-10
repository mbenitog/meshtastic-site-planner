from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .coverage_json import CoverageApiClient, CoverageGrid, DTM_5M_25830
from .geo import meter_bbox_around
from .ign_dsm import ArcGrid, IgnDsmClient


SurfaceMode = Literal[
    "lod_dtm_plus_buildings",
    "dtm_plus_buildings_2_5m",
    "dtm_plus_surface_2_5m",
    "dtm_only",
]


_MODES_USING_BUILDINGS = {
    "lod_dtm_plus_buildings",
    "dtm_plus_buildings_2_5m",
    "dtm_plus_surface_2_5m",
}
_MODES_USING_SURFACE_5M = {
    "lod_dtm_plus_buildings",
    "dtm_plus_surface_2_5m",
}
_MODES_REQUIRING_BUILDINGS = {
    "dtm_plus_buildings_2_5m",
    "dtm_plus_surface_2_5m",
}


@dataclass
class SurfaceGrid:
    width: int
    height: int
    min_x: float
    min_y: float
    resolution_m: float
    mode: SurfaceMode
    values: np.ndarray
    sources: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
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
        return float(self.values.min()) if self.values.size else 0.0

    @property
    def max_value(self) -> float:
        return float(self.values.max()) if self.values.size else 0.0


def _coverage_lookup_filled(
    grid: CoverageGrid, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Nearest-cell lookup of a CoverageGrid over a regular target grid.

    Returns a 2D float array of shape ``(len(ys), len(xs))``. Missing source
    cells are filled in by :class:`CoverageGrid`'s precomputed ``_filled``
    array so this lookup always returns a finite value.
    """
    if grid.width <= 0 or grid.height <= 0:
        return np.full((ys.shape[0], xs.shape[0]), np.nan, dtype=np.float64)
    cols = np.clip(
        np.round((xs[None, :] - grid.min_x) / grid.dx).astype(np.int64),
        0,
        grid.width - 1,
    )
    if grid.max_y >= grid.min_y:
        rows = np.clip(
            np.round((grid.max_y - ys[:, None]) / abs(grid.dy)).astype(np.int64),
            0,
            grid.height - 1,
        )
    else:
        rows = np.clip(
            np.round((ys[:, None] - grid.max_y) / abs(grid.dy)).astype(np.int64),
            0,
            grid.height - 1,
        )
    return grid._filled[rows, cols]


def _arcgrid_lookup_filled(
    grid: ArcGrid | None, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Nearest-cell lookup of an ArcGrid over a regular target grid.

    Returns a 2D float array of shape ``(len(ys), len(xs))`` with NaN where
    the source has no data (missing tile or nodata sentinel). ``grid=None``
    also yields all-NaN.
    """
    if grid is None or grid.ncols <= 0 or grid.nrows <= 0:
        return np.full((ys.shape[0], xs.shape[0]), np.nan, dtype=np.float64)
    src = np.array(grid.values, dtype=np.float64).reshape(grid.nrows, grid.ncols)
    if grid.nodata is not None:
        src = np.where(src == grid.nodata, np.nan, src)
    cols = np.clip(
        np.round((xs[None, :] - grid.xllcorner) / grid.cellsize).astype(np.int64),
        0,
        grid.ncols - 1,
    )
    north = grid.yllcorner + (grid.nrows - 1) * grid.cellsize
    rows = np.clip(
        np.round((north - ys[:, None]) / grid.cellsize).astype(np.int64),
        0,
        grid.nrows - 1,
    )
    return src[rows, cols]


class SurfaceBuilder:
    def __init__(
        self,
        *,
        dsm_client: IgnDsmClient | None = None,
        coverage_client: CoverageApiClient | None = None,
    ) -> None:
        self.dsm = dsm_client or IgnDsmClient()
        self.coverages = coverage_client or CoverageApiClient()

    def build(
        self,
        *,
        lat: float,
        lon: float,
        radius_m: float,
        resolution_m: float = 2.5,
        mode: SurfaceMode = "lod_dtm_plus_buildings",
    ) -> SurfaceGrid:
        min_x, min_y, max_x, max_y = meter_bbox_around(lat, lon, radius_m)
        width = int((max_x - min_x) / resolution_m) + 1
        height = int((max_y - min_y) / resolution_m) + 1
        cells = width * height

        # Decide which optional IGN coverages we actually consume. The
        # original code fetched surface_5m and buildings unconditionally,
        # which is wasted work for dtm_only / dtm_plus_buildings_2_5m. We
        # only fetch what the per-cell branching can use.
        needs_buildings = mode in _MODES_USING_BUILDINGS
        needs_surface_5m = mode in _MODES_USING_SURFACE_5M
        needs_vegetation = mode == "dtm_plus_surface_2_5m"

        # Fetch the DTM 5 m ground reference plus any optional coverages in
        # parallel. Each IGN endpoint is 1-15s on a cold cache, and they are
        # independent, so we issue them concurrently. DTM is mandatory; the
        # others are best-effort and tolerate fetch failures.
        terrain: CoverageGrid | None = None
        surface_5m: ArcGrid | None = None
        buildings: ArcGrid | None = None
        vegetation: ArcGrid | None = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures: dict[concurrent.futures.Future, str] = {}
            futures[
                pool.submit(self._try_fetch_dtm, min_x, min_y, max_x, max_y)
            ] = "dtm"
            if needs_surface_5m:
                futures[
                    pool.submit(
                        self._try_fetch_arcgrid,
                        "mds05",
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                        tiled=True,
                    )
                ] = "mds05"
            if needs_buildings:
                futures[
                    pool.submit(
                        self._try_fetch_arcgrid,
                        "mdsn_e025",
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                        tiled=True,
                    )
                ] = "buildings"
            if needs_vegetation:
                futures[
                    pool.submit(
                        self._try_fetch_arcgrid,
                        "mdsn_v025",
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                        tiled=True,
                    )
                ] = "vegetation"

            for future in concurrent.futures.as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                except Exception:
                    if label == "dtm":
                        raise RuntimeError(
                            "DTM 5 m coverage is required for ultra surface but "
                            "is not available for this bbox. Move closer to "
                            "Spanish mainland or choose dtm_only mode if your "
                            "client supports it."
                        )
                    continue
                if label == "dtm":
                    terrain = result
                elif label == "mds05":
                    surface_5m = result
                elif label == "buildings":
                    buildings = result
                elif label == "vegetation":
                    vegetation = result

        if terrain is None:
            raise RuntimeError(
                "DTM 5 m coverage is required for ultra surface but is not "
                "available for this bbox. Move closer to Spanish mainland or "
                "choose dtm_only mode if your client supports it."
            )

        # In strict 2.5 m modes we require the 2.5 m building layer. If it is
        # not available the surface build fails because the user explicitly
        # asked for the highest resolution.
        if mode in _MODES_REQUIRING_BUILDINGS and buildings is None:
            raise RuntimeError(
                "Requested 2.5 m DSM coverage is not available for this bbox. "
                "Use lod_dtm_plus_buildings or dtm_only, or move closer to a "
                "city covered by mdsn_e025."
            )

        # Vectorized surface build. The inner loop used to be a Python
        # double-for over every (x, y) cell calling sample_nearest four
        # times each; that was O(width*height*sources) and dominated the
        # 2.5 m build for 1 km and above. We replace it with a single
        # numpy meshgrid and four whole-array lookups, which is
        # O(width*height) numpy-side and completes in ~1 s for 1 km
        # instead of >60 s.
        xs = min_x + np.arange(width, dtype=np.float64) * resolution_m
        ys = max_y - np.arange(height, dtype=np.float64) * resolution_m

        terrain_arr = _coverage_lookup_filled(terrain, xs, ys)
        building_raw = _arcgrid_lookup_filled(buildings, xs, ys)
        vegetation_raw = _arcgrid_lookup_filled(vegetation, xs, ys)
        surface_5m_raw = _arcgrid_lookup_filled(surface_5m, xs, ys)

        building_m = np.where(
            np.isnan(building_raw), 0.0, np.maximum(0.0, building_raw)
        )
        vegetation_m = np.where(
            np.isnan(vegetation_raw), 0.0, np.maximum(0.0, vegetation_raw)
        )
        surface_5m_m = np.where(
            np.isnan(surface_5m_raw),
            np.nan,
            np.maximum(terrain_arr, surface_5m_raw),
        )

        if mode == "dtm_only":
            values_arr = terrain_arr
            sources_arr = np.zeros(terrain_arr.shape, dtype=np.uint8)
        else:
            has_building = building_m > 0.0
            has_vegetation = (vegetation_m > 0.0) & ~has_building

            values_arr = terrain_arr.copy()
            sources_arr = np.zeros(terrain_arr.shape, dtype=np.uint8)

            # building_m > 0 -> value = terrain + max(building, vegetation);
            # source = 2 if vegetation >= building else 1.
            building_obstruction = np.maximum(building_m, vegetation_m)
            building_source = np.where(
                vegetation_m >= building_m,
                np.uint8(2),
                np.uint8(1),
            ).astype(np.uint8)
            values_arr = np.where(
                has_building,
                terrain_arr + building_obstruction,
                values_arr,
            )
            sources_arr = np.where(has_building, building_source, sources_arr)

            # vegetation_m > 0 (and no building) -> value = terrain + vegetation;
            # source = 2.
            values_arr = np.where(
                has_vegetation,
                terrain_arr + vegetation_m,
                values_arr,
            )
            sources_arr = np.where(has_vegetation, np.uint8(2), sources_arr)

            # LOD fallback only for lod_dtm_plus_buildings: value = surface_5m_m,
            # source = 3, bypasses the terrain + obstruction addition.
            if mode == "lod_dtm_plus_buildings":
                has_lod = ~has_building & ~has_vegetation & ~np.isnan(surface_5m_m)
                values_arr = np.where(has_lod, surface_5m_m, values_arr)
                sources_arr = np.where(has_lod, np.uint8(3), sources_arr)

        return SurfaceGrid(
            width=width,
            height=height,
            min_x=min_x,
            min_y=min_y,
            resolution_m=resolution_m,
            mode=mode,
            values=values_arr,
            sources=sources_arr,
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
