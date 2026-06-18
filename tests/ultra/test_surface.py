from backend.ultra.coverage_json import CoverageGrid
from backend.ultra.ign_dsm import ArcGrid
from backend.ultra.surface import SurfaceBuilder
import backend.ultra.surface as surface_module


class FakeCoverageClient:
    def __init__(self, terrain: CoverageGrid):
        self.terrain = terrain

    def fetch_grid_tiled(self, **kwargs):
        return self.terrain


class FakeDsmClient:
    def __init__(self, grids: dict[str, ArcGrid]):
        self.grids = grids

    def fetch_arcgrid_tiled(self, *, coverage_id: str, **kwargs):
        grid = self.grids.get(coverage_id)
        if grid is None:
            raise RuntimeError(f"missing test coverage {coverage_id}")
        return grid


def _terrain_grid() -> CoverageGrid:
    return CoverageGrid(
        width=3,
        height=3,
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        values=[100.0] * 9,
    )


def _arcgrid(values: list[float]) -> ArcGrid:
    return ArcGrid(
        ncols=5,
        nrows=5,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=2.5,
        values=values,
        nodata=None,
    )


def test_parallel_surface_build_matches_serial(monkeypatch):
    monkeypatch.setattr(surface_module, "meter_bbox_around", lambda lat, lon, radius_m: (0.0, 0.0, 10.0, 10.0))
    monkeypatch.setattr(surface_module, "SURFACE_PROCESS_MIN_CELLS", 1)

    buildings = _arcgrid(
        [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 12, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ]
    )
    surface_5m = _arcgrid(
        [
            105, 105, 105, 105, 105,
            105, 105, 105, 105, 105,
            105, 105, 105, 105, 105,
            105, 105, 105, 105, 105,
            105, 105, 105, 105, 105,
        ]
    )

    builder = SurfaceBuilder(
        dsm_client=FakeDsmClient({"mds05": surface_5m, "mdsn_e025": buildings}),
        coverage_client=FakeCoverageClient(_terrain_grid()),
    )

    monkeypatch.setenv("ULTRA_SURFACE_WORKERS", "1")
    serial = builder.build(lat=40.0, lon=-3.0, radius_m=5.0, resolution_m=2.5, mode="lod_dtm_plus_buildings")

    monkeypatch.setenv("ULTRA_SURFACE_WORKERS", "2")
    parallel = builder.build(lat=40.0, lon=-3.0, radius_m=5.0, resolution_m=2.5, mode="lod_dtm_plus_buildings")

    assert parallel.width == serial.width == 5
    assert parallel.height == serial.height == 5
    assert parallel.values == serial.values
    assert parallel.sources == serial.sources
    assert max(parallel.values) == 112.0
    assert 1 in parallel.sources
    assert 3 in parallel.sources
