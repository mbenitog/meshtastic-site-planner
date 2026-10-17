from __future__ import annotations

import os
from pathlib import Path
import struct
import zlib
import numpy as np


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _build_color_lut(min_dbm: float, max_dbm: float) -> np.ndarray:
    """Precompute a 65,536-entry RGBA lookup table indexed by uint16(dbm_x10).
    Mapping 64M-256M grid cells through this 256 KB L2-cached LUT in NumPy
    takes ~1-2 seconds instead of minutes in a Python loop.
    """
    lut = np.zeros((65536, 4), dtype=np.uint8)
    raw_i16 = np.arange(-32768, 32768, dtype=np.int16)
    dbm = raw_i16.astype(np.float32) / 10.0
    valid = dbm >= min_dbm
    if not np.any(valid):
        return lut

    span = max(1.0, max_dbm - min_dbm)
    t = np.clip((dbm[valid] - min_dbm) / span, 0.0, 1.0)
    part1 = t < 0.5
    u1 = t[part1] * 2.0
    r1 = np.round(13 + 190 * u1).astype(np.uint8)
    g1 = np.round(8 + 55 * u1).astype(np.uint8)
    b1 = np.round(135 + 35 * u1).astype(np.uint8)

    part2 = ~part1
    u2 = (t[part2] - 0.5) * 2.0
    r2 = np.round(203 + 52 * u2).astype(np.uint8)
    g2 = np.round(63 + 186 * u2).astype(np.uint8)
    b2 = np.round(170 - 140 * u2).astype(np.uint8)

    colors = np.zeros((np.count_nonzero(valid), 4), dtype=np.uint8)
    colors[part1, 0] = r1
    colors[part1, 1] = g1
    colors[part1, 2] = b1
    colors[part1, 3] = 180

    colors[part2, 0] = r2
    colors[part2, 1] = g2
    colors[part2, 2] = b2
    colors[part2, 3] = 180

    lut[raw_i16[valid].view(np.uint16)] = colors
    return lut


def encode_png_bytes(rgba: np.ndarray, width: int, height: int, comp_level: int = 1) -> bytes:
    """Encode an RGBA NumPy array into valid PNG bytes."""
    raw = np.empty((height, width * 4 + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rgba.reshape((height, width * 4))
    compressed = zlib.compress(raw.tobytes(), level=comp_level)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")
    return bytes(png)


def write_coverage_png(
    *,
    signal_path: str | Path,
    mask_path: str | Path,
    out_path: str | Path,
    width: int,
    height: int,
    min_dbm: float,
    max_dbm: float = -80.0,
    max_dim: int = 8192,
) -> str:
    cells = width * height
    signal_arr = np.fromfile(str(signal_path), dtype="<i2")
    mask_arr = np.fromfile(str(mask_path), dtype=np.uint8)

    if len(signal_arr) != cells:
        raise ValueError(f"signal size mismatch: expected {cells}, got {len(signal_arr)}")
    if len(mask_arr) != cells:
        raise ValueError(f"mask size mismatch: expected {cells}, got {len(mask_arr)}")

    out_w, out_h = width, height
    step = 1
    if max_dim and (width > max_dim or height > max_dim):
        step = int(np.ceil(max(width, height) / max_dim))
        sig_2d = signal_arr.reshape((height, width))[::step, ::step]
        mask_2d = mask_arr.reshape((height, width))[::step, ::step]
        out_h, out_w = sig_2d.shape
        signal_arr = sig_2d.reshape(-1)
        mask_arr = mask_2d.reshape(-1)

    lut = _build_color_lut(min_dbm, max_dbm)
    rgba = lut[signal_arr.view(np.uint16)]
    rgba[mask_arr == 0] = 0

    comp_level = int(os.environ.get("ULTRA_PNG_COMPRESSION_LEVEL", "1"))
    png = encode_png_bytes(rgba, out_w, out_h, comp_level=comp_level)

    path = Path(out_path)
    path.write_bytes(png)
    return str(path)


def write_coverage_tiles(
    *,
    signal_path: str | Path,
    mask_path: str | Path,
    out_dir: str | Path,
    width: int,
    height: int,
    resolution_m: float,
    min_x: float,
    max_y: float,
    min_dbm: float,
    max_dbm: float = -80.0,
    max_tile_dim: int = 4096,
) -> list[dict]:
    """Slice coverage signal and mask arrays into tiles <= max_tile_dim and write
    per-tile PNGs to `out_dir / "tiles"`. Returns a list of tile specs with WGS84 corners.
    """
    import json
    from .geo import utm30_to_latlon

    out = Path(out_dir)
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    signal_arr = np.fromfile(str(signal_path), dtype="<i2").reshape((height, width))
    mask_arr = np.fromfile(str(mask_path), dtype=np.uint8).reshape((height, width))
    lut = _build_color_lut(min_dbm, max_dbm)
    comp_level = int(os.environ.get("ULTRA_PNG_COMPRESSION_LEVEL", "1"))
    half = resolution_m / 2.0

    tile_specs: list[dict] = []
    for r0 in range(0, height, max_tile_dim):
        r1 = min(height, r0 + max_tile_dim)
        th = r1 - r0
        for c0 in range(0, width, max_tile_dim):
            c1 = min(width, c0 + max_tile_dim)
            tw = c1 - c0

            sub_mask = mask_arr[r0:r1, c0:c1]
            if not np.any(sub_mask):
                continue  # Skip tiles with zero coverage (fully transparent)

            sub_sig = signal_arr[r0:r1, c0:c1]
            rgba = lut[sub_sig.view(np.uint16)]
            rgba[sub_mask == 0] = 0

            tile_name = f"tile_{c0}_{r0}.png"
            tile_path = tiles_dir / tile_name
            tile_png = encode_png_bytes(rgba, tw, th, comp_level=comp_level)
            tile_path.write_bytes(tile_png)

            tile_min_x = min_x + c0 * resolution_m
            tile_max_y = max_y - r0 * resolution_m
            tile_max_x = min_x + (c1 - 1) * resolution_m
            tile_min_y = max_y - (r1 - 1) * resolution_m

            corners = [
                utm30_to_latlon(tile_min_x - half, tile_max_y + half),  # NW
                utm30_to_latlon(tile_max_x + half, tile_max_y + half),  # NE
                utm30_to_latlon(tile_max_x + half, tile_min_y - half),  # SE
                utm30_to_latlon(tile_min_x - half, tile_min_y - half),  # SW
            ]
            corners_wgs84 = [[lon, lat] for lat, lon in corners]

            tile_specs.append({
                "url": f"/ultra/jobs/{out.name}/tiles/{tile_name}",
                "coordinates": corners_wgs84,
                "c0": c0,
                "r0": r0,
                "width": tw,
                "height": th,
            })

    manifest_path = out / "coverage_tiles.json"
    manifest_path.write_text(json.dumps(tile_specs, indent=2) + "\n", encoding="utf-8")
    return tile_specs


def write_png_world_file(
    *,
    out_path: str | Path,
    width: int,
    height: int,
    bounds_wgs84: dict[str, float],
) -> str:
    west = bounds_wgs84["west"]
    east = bounds_wgs84["east"]
    north = bounds_wgs84["north"]
    south = bounds_wgs84["south"]
    pixel_x = (east - west) / width
    pixel_y = (south - north) / height
    # World files store center coordinates of the upper-left pixel.
    text = (
        f"{pixel_x:.12f}\n"
        "0.000000000000\n"
        "0.000000000000\n"
        f"{pixel_y:.12f}\n"
        f"{west + pixel_x / 2:.12f}\n"
        f"{north + pixel_y / 2:.12f}\n"
    )
    path = Path(out_path)
    path.write_text(text, encoding="utf-8")
    return str(path)
