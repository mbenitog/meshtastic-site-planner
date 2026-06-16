from __future__ import annotations

from dataclasses import dataclass, field
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
    (vegetation), 3 = dtm + mds05 (5 m absolute surface)."""

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

        # Always fetch the DTM 5 m ground reference. This is mandatory: every
        # surface cell is at minimum DTM elevation.
        terrain = self._try_fetch_dtm(min_x, min_y, max_x, max_y)
        if terrain is None:
            raise RuntimeError(
                "DTM 5 m coverage is required for ultra surface but is not "
                "available for this bbox. Move closer to Spanish mainland or "
                "choose dtm_only mode if your client supports it."
            )

        # Fetch the 2.5 m building DSM and (optionally) the 2.5 m vegetation
        # layer. Each fetch is best-effort: if the bbox is outside IGN coverage
        # the corresponding layer is left as ``None`` and cells fall back to DTM
        # only. Large bboxes are split into WCS sub-requests by the client
        # because the IGN endpoint rejects oversized responses.
        buildings = self._try_fetch_arcgrid("mdsn_e025", min_x, min_y, max_x, max_y, tiled=True)
        want_vegetation = mode == "dtm_plus_surface_2_5m"
        vegetation = None
        if want_vegetation:
            vegetation = self._try_fetch_arcgrid("mdsn_v025", min_x, min_y, max_x, max_y, tiled=True)

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

        values: list[float] = []
        sources: list[int] = []
        for y in range(height):
            py = max_y - y * resolution_m
            for x in range(width):
                px = min_x + x * resolution_m
                terrain_m = terrain.sample_nearest(px, py)
                building_m = (
                    max(0.0, _sample_arcgrid_nearest(buildings, px, py) or 0.0)
                    if buildings is not None
                    else 0.0
                )
                vegetation_m = (
                    max(0.0, _sample_arcgrid_nearest(vegetation, px, py) or 0.0)
                    if vegetation is not None
                    else 0.0
                )
                if building_m > 0.0:
                    obstruction = max(building_m, vegetation_m)
                    if vegetation_m >= building_m > 0.0:
                        source = 2
                    else:
                        source = 1
                elif vegetation_m > 0.0:
                    obstruction = vegetation_m
                    source = 2
                else:
                    obstruction = 0.0
                    source = 0
                values.append(terrain_m + obstruction)
                sources.append(source)

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
