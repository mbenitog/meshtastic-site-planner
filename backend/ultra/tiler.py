from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import struct

from .artifacts import SurfaceArtifact
from .geo import latlon_to_utm30


@dataclass(frozen=True)
class TileSpec:
    index: int
    col: int
    row: int
    x0: int
    y0: int
    width: int
    height: int


def plan_tiles(width: int, height: int, max_tile_cells: int = 250_000) -> list[TileSpec]:
    """Plan a non-overlapping tile grid that keeps each tile under
    ``max_tile_cells``. Tiles are column-major so adjacent tiles are
    spatially near, which is convenient for potential future incremental
    overlays."""
    side = int(max(1, max_tile_cells ** 0.5))
    side = min(side, width, height)
    tiles: list[TileSpec] = []
    index = 0
    for ry in range(0, height, side):
        for rx in range(0, width, side):
            tw = min(side, width - rx)
            th = min(side, height - ry)
            tiles.append(TileSpec(index=index, col=rx // side, row=ry // side,
                                  x0=rx, y0=ry, width=tw, height=th))
            index += 1
    return tiles


def _signal_suffix(tile: TileSpec) -> str:
    return f"_x{tile.x0}_y{tile.y0}.signal_i16le.bin"


def _mask_suffix(tile: TileSpec) -> str:
    return f"_x{tile.x0}_y{tile.y0}.mask_u8.bin"


def _meta_suffix(tile: TileSpec) -> str:
    return f"_x{tile.x0}_y{tile.y0}.meta.json"


def _build_native_command(
    binary: Path,
    artifact: SurfaceArtifact,
    tile: TileSpec,
    request: dict,
    tx_x: float,
    tx_y: float,
    out_prefix: Path,
) -> list[str]:
    return [
        str(binary),
        "--surface",
        artifact.path,
        "--out",
        str(out_prefix),
        "--width",
        str(artifact.width),
        "--height",
        str(artifact.height),
        "--tile-x0",
        str(tile.x0),
        "--tile-y0",
        str(tile.y0),
        "--tile-w",
        str(tile.width),
        "--tile-h",
        str(tile.height),
        "--min-x",
        f"{artifact.min_x:.6f}",
        "--max-y",
        f"{artifact.max_y:.6f}",
        "--resolution-m",
        f"{artifact.resolution_m:.6f}",
        "--tx-x",
        f"{tx_x:.6f}",
        "--tx-y",
        f"{tx_y:.6f}",
        "--tx-height-m",
        f"{request['tx_height_m']:.6f}",
        "--rx-height-m",
        f"{request['rx_height_m']:.6f}",
        "--freq-mhz",
        f"{request['frequency_mhz']:.6f}",
        "--tx-power-w",
        f"{request['tx_power_w']:.6f}",
        "--tx-gain-dbi",
        f"{request['tx_gain_dbi']:.6f}",
        "--rx-gain-dbi",
        f"{request['rx_gain_dbi']:.6f}",
        "--rx-sensitivity-dbm",
        f"{request['rx_sensitivity_dbm']:.6f}",
        "--dielect",
        f"{request['ground_dielectric']:.6f}",
        "--conductivity",
        f"{request['ground_conductivity']:.6f}",
        "--bend",
        f"{request['atmosphere_bending']:.6f}",
        "--climate",
        str(request['radio_climate']),
        "--pol",
        str(request['polarization']),
        "--conf",
        f"{request['confidence']:.6f}",
        "--rel",
        f"{request['reliability']:.6f}",
    ]


def aggregate_tiles(
    *,
    artifact: SurfaceArtifact,
    tiles: list[TileSpec],
    out_dir: Path,
) -> dict:
    """Stitch per-tile signal/mask into the final coverage output and
    write ``coverage.meta.json`` aggregating totals."""
    cells = artifact.width * artifact.height
    signal = bytearray(b"\x00") * (cells * 2)
    mask = bytearray(b"\x00") * cells
    covered = 0
    err_counts = [0] * 6

    for tile in tiles:
        sig_path = out_dir / f"coverage{_signal_suffix(tile)}"
        msk_path = out_dir / f"coverage{_mask_suffix(tile)}"
        meta_path = out_dir / f"coverage{_meta_suffix(tile)}"
        with sig_path.open("rb") as fp:
            tile_signal = fp.read()
        with msk_path.open("rb") as fp:
            tile_mask = fp.read()
        if len(tile_signal) != tile.width * tile.height * 2:
            raise ValueError(f"tile signal size mismatch: {sig_path}")
        if len(tile_mask) != tile.width * tile.height:
            raise ValueError(f"tile mask size mismatch: {msk_path}")
        # Copy tile data into the full grid row by row.
        for row in range(tile.height):
            src_y = row
            dst_y = tile.y0 + row
            src_offset = src_y * tile.width * 2
            dst_offset = (dst_y * artifact.width + tile.x0) * 2
            signal[dst_offset:dst_offset + tile.width * 2] = tile_signal[
                src_offset:src_offset + tile.width * 2
            ]
            src_m = src_y * tile.width
            dst_m = dst_y * artifact.width + tile.x0
            mask[dst_m:dst_m + tile.width] = tile_mask[src_m:src_m + tile.width]
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            covered += int(meta.get("covered_cells", 0))
            for i, c in enumerate(meta.get("itm_errnums", [])):
                if i < 6:
                    err_counts[i] += int(c)

    full_signal = out_dir / "coverage.signal_i16le.bin"
    full_mask = out_dir / "coverage.mask_u8.bin"
    full_signal.write_bytes(bytes(signal))
    full_mask.write_bytes(bytes(mask))

    meta = {
        "model": "itm_projected_grid",
        "width": artifact.width,
        "height": artifact.height,
        "resolution_m": artifact.resolution_m,
        "min_x": artifact.min_x,
        "max_y": artifact.max_y,
        "signal_scale": "dbm_x10_i16",
        "mask_value": "1 means dbm >= rx_sensitivity_dbm",
        "rx_sensitivity_dbm": None,  # filled by caller.
        "covered_cells": covered,
        "total_cells": cells,
        "itm_errnums": err_counts,
        "tiles": [
            {
                "x0": t.x0,
                "y0": t.y0,
                "width": t.width,
                "height": t.height,
            }
            for t in tiles
        ],
    }
    return meta


def build_native_command(
    *,
    binary: Path,
    artifact: SurfaceArtifact,
    request: dict,
    tile: TileSpec,
    out_prefix: Path,
) -> list[str]:
    tx_x, tx_y = latlon_to_utm30(request["lat"], request["lon"])
    return _build_native_command(binary, artifact, tile, request, tx_x, tx_y, out_prefix)


def tile_signal_path(out_prefix: Path, tile: TileSpec) -> Path:
    return out_prefix.with_name(out_prefix.name + _signal_suffix(tile))


def tile_mask_path(out_prefix: Path, tile: TileSpec) -> Path:
    return out_prefix.with_name(out_prefix.name + _mask_suffix(tile))


def tile_meta_path(out_prefix: Path, tile: TileSpec) -> Path:
    return out_prefix.with_name(out_prefix.name + _meta_suffix(tile))


def make_native_run_result(out_prefix: Path) -> dict:
    """Build the standard native-result payload without depending on the
    runner module (avoids a circular import)."""
    return {
        "status": "complete",
        "model": "itm_projected_grid",
        "signal_path": str(out_prefix) + ".signal_i16le.bin",
        "mask_path": str(out_prefix) + ".mask_u8.bin",
        "meta_path": str(out_prefix) + ".meta.json",
    }
