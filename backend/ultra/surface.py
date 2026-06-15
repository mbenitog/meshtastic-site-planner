from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .coverage_json import CoverageApiClient, DTM_5M_25830
from .geo import meter_bbox_around
from .ign_dsm import ArcGrid, IgnDsmClient


SurfaceMode = Literal["dtm_plus_buildings_2_5m", "dtm_plus_surface_2_5m"]


@dataclass(frozen=True)
class SurfaceGrid:
    width: int
    height: int
    min_x: float
    min_y: float
    resolution_m: float
    mode: SurfaceMode
    values: list[float]

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


def _sample_arcgrid_nearest(grid: ArcGrid, x: float, y: float) -> float:
    ix = round((x - grid.xllcorner) / grid.cellsize)
    # ArcGrid rows are stored north-to-south while yllcorner is the south edge.
    north = grid.yllcorner + (grid.nrows - 1) * grid.cellsize
    iy = round((north - y) / grid.cellsize)
    ix = min(max(int(ix), 0), grid.ncols - 1)
    iy = min(max(int(iy), 0), grid.nrows - 1)
    return grid.values[iy * grid.ncols + ix]


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
        resolution_m: Literal[2.5] = 2.5,
        mode: SurfaceMode = "dtm_plus_buildings_2_5m",
    ) -> SurfaceGrid:
        min_x, min_y, max_x, max_y = meter_bbox_around(lat, lon, radius_m)
        width = int((max_x - min_x) / resolution_m) + 1
        height = int((max_y - min_y) / resolution_m) + 1

        terrain = self.coverages.fetch_grid(
            collection_id=DTM_5M_25830,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
        buildings = None
        vegetation = None
        buildings = self.dsm.fetch_arcgrid(
            coverage_id="mdsn_e025",
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
        if mode == "dtm_plus_surface_2_5m":
            vegetation = self.dsm.fetch_arcgrid(
                coverage_id="mdsn_v025",
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            )

        values = []
        for y in range(height):
            py = max_y - y * resolution_m
            for x in range(width):
                px = min_x + x * resolution_m
                terrain_m = terrain.sample_nearest(px, py)
                building_m = max(0.0, _sample_arcgrid_nearest(buildings, px, py)) if buildings else 0.0
                vegetation_m = max(0.0, _sample_arcgrid_nearest(vegetation, px, py)) if vegetation else 0.0
                values.append(terrain_m + max(building_m, vegetation_m))

        return SurfaceGrid(
            width=width,
            height=height,
            min_x=min_x,
            min_y=min_y,
            resolution_m=resolution_m,
            mode=mode,
            values=values,
        )
