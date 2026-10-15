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


def write_coverage_png(
    *,
    signal_path: str | Path,
    mask_path: str | Path,
    out_path: str | Path,
    width: int,
    height: int,
    min_dbm: float,
    max_dbm: float = -80.0,
) -> str:
    cells = width * height
    signal_arr = np.fromfile(str(signal_path), dtype="<i2")
    mask_arr = np.fromfile(str(mask_path), dtype=np.uint8)

    if len(signal_arr) != cells:
        raise ValueError(f"signal size mismatch: expected {cells}, got {len(signal_arr)}")
    if len(mask_arr) != cells:
        raise ValueError(f"mask size mismatch: expected {cells}, got {len(mask_arr)}")

    # Vectorized LUT color mapping (orders of magnitude faster than cell-by-cell Python loop)
    lut = _build_color_lut(min_dbm, max_dbm)
    rgba = lut[signal_arr.view(np.uint16)]
    rgba[mask_arr == 0] = 0

    # Assemble raw PNG scanlines with filter type 0 (None)
    raw = np.empty((height, width * 4 + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rgba.reshape((height, width * 4))

    # Fast zlib compression (level 1 is 5-10x faster with virtually identical compression ratio on pre-filtered rasters)
    comp_level = int(os.environ.get("ULTRA_PNG_COMPRESSION_LEVEL", "1"))
    compressed = zlib.compress(raw.tobytes(), level=comp_level)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")

    path = Path(out_path)
    path.write_bytes(png)
    return str(path)


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
