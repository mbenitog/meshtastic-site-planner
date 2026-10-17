import struct
from pathlib import Path

from backend.ultra.artifacts import write_surface_artifact
from backend.ultra.surface import SurfaceGrid
from backend.ultra.tiler import (
    TileSpec,
    aggregate_tiles,
    plan_tiles,
    tile_mask_path,
    tile_meta_path,
    tile_signal_path,
)


def _fake_grid(width: int, height: int) -> SurfaceGrid:
    return SurfaceGrid(
        width=width,
        height=height,
        min_x=0.0,
        min_y=0.0,
        resolution_m=2.5,
        mode="dtm_plus_buildings_2_5m",
        values=[100.0 + i for i in range(width * height)],
    )


def test_write_surface_artifact(tmp_path: Path):
    grid = _fake_grid(4, 3)
    artifact = write_surface_artifact(grid, tmp_path)
    assert (tmp_path / "surface_i16le.bin").exists()
    assert artifact.width == 4
    assert artifact.height == 3
    assert artifact.bounds_wgs84["north"] > artifact.bounds_wgs84["south"]
    with (tmp_path / "surface_i16le.bin").open("rb") as fp:
        values = struct.unpack("<12h", fp.read())
    assert values[0] == 100
    assert values[-1] == 100 + 11


def test_plan_tiles_respects_max_size():
    tiles = plan_tiles(8, 8, max_tile_cells=9)
    assert len(tiles) >= 4  # 8*8 = 64 cells, so 8 tiles of up to 9 cells.
    for tile in tiles:
        assert tile.width * tile.height <= 9


def test_plan_tiles_full_coverage():
    tiles = plan_tiles(6, 4, max_tile_cells=24)
    total_cells = sum(t.width * t.height for t in tiles)
    assert total_cells == 24


def test_aggregate_tiles_uses_per_tile_files(tmp_path: Path):
    grid = _fake_grid(4, 4)
    artifact = write_surface_artifact(grid, tmp_path)
    tiles = [
        TileSpec(index=0, col=0, row=0, x0=0, y0=0, width=2, height=2),
        TileSpec(index=1, col=0, row=0, x0=2, y0=0, width=2, height=2),
        TileSpec(index=2, col=0, row=0, x0=0, y0=2, width=2, height=2),
        TileSpec(index=3, col=0, row=0, x0=2, y0=2, width=2, height=2),
    ]
    for tile in tiles:
        (tmp_path / tile_signal_path(Path("coverage"), tile).name).write_bytes(
            b"\x00\x01" * (tile.width * tile.height)
        )
        (tmp_path / tile_mask_path(Path("coverage"), tile).name).write_bytes(
            b"\x01" * (tile.width * tile.height)
        )
        (tmp_path / tile_meta_path(Path("coverage"), tile).name).write_text(
            '{"covered_cells": 4, "itm_errnums": [0, 0, 0, 0, 4, 0]}', encoding="utf-8"
        )

    meta = aggregate_tiles(artifact=artifact, tiles=tiles, out_dir=tmp_path)
    assert (tmp_path / "coverage.signal_i16le.bin").exists()
    assert (tmp_path / "coverage.mask_u8.bin").exists()
    assert meta["covered_cells"] == 16
    assert meta["total_cells"] == 16
    assert meta["tiles"] == [
        {"x0": 0, "y0": 0, "width": 2, "height": 2},
        {"x0": 2, "y0": 0, "width": 2, "height": 2},
        {"x0": 0, "y0": 2, "width": 2, "height": 2},
        {"x0": 2, "y0": 2, "width": 2, "height": 2},
    ]


def test_write_coverage_tiles(tmp_path: Path):
    from backend.ultra.render import write_coverage_tiles
    import json

    w, h = 6, 4
    sig = tmp_path / "sig.bin"
    mask = tmp_path / "mask.bin"
    # Write -90.0 dBm (-900) everywhere
    sig.write_bytes(struct.pack("<24h", *([-900] * 24)))
    # Only top-left 3x2 has mask=1, rest mask=0
    mask_bytes = bytearray(24)
    for r in range(2):
        for c in range(3):
            mask_bytes[r * w + c] = 1
    mask.write_bytes(bytes(mask_bytes))

    tiles = write_coverage_tiles(
        signal_path=sig,
        mask_path=mask,
        out_dir=tmp_path,
        width=w,
        height=h,
        resolution_m=2.5,
        min_x=440000.0,
        max_y=4470000.0,
        min_dbm=-130.0,
        max_tile_dim=3,
    )

    # 6x4 with max_tile_dim=3 means 2 columns x 2 rows = 4 possible tiles.
    # Only tile (0, 0) has mask=1, so only 1 tile is written!
    assert len(tiles) == 1
    assert tiles[0]["c0"] == 0
    assert tiles[0]["r0"] == 0
    assert tiles[0]["width"] == 3
    assert tiles[0]["height"] == 3
    assert (tmp_path / "coverage_tiles.json").exists()
    assert (tmp_path / "tiles" / "tile_0_0.png").exists()

    data = json.loads((tmp_path / "coverage_tiles.json").read_text(encoding="utf-8"))
    assert len(data) == 1
    assert len(data[0]["coordinates"]) == 4
