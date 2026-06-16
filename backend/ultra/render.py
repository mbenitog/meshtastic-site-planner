from __future__ import annotations

from pathlib import Path
import struct
import zlib


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _color(dbm: float, min_dbm: float, max_dbm: float) -> tuple[int, int, int]:
    t = (dbm - min_dbm) / max(1.0, max_dbm - min_dbm)
    t = max(0.0, min(1.0, t))
    # Compact plasma-like ramp: dark violet -> magenta -> orange/yellow.
    if t < 0.5:
        u = t * 2
        r = round(13 + 190 * u)
        g = round(8 + 55 * u)
        b = round(135 + 35 * u)
    else:
        u = (t - 0.5) * 2
        r = round(203 + 52 * u)
        g = round(63 + 186 * u)
        b = round(170 - 140 * u)
    return r, g, b


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
    signal = Path(signal_path).read_bytes()
    mask = Path(mask_path).read_bytes()
    cells = width * height
    if len(signal) != cells * 2:
        raise ValueError(f"signal size mismatch: expected {cells * 2}, got {len(signal)}")
    if len(mask) != cells:
        raise ValueError(f"mask size mismatch: expected {cells}, got {len(mask)}")

    raw = bytearray()
    for row in range(height):
        raw.append(0)  # PNG filter type 0.
        for col in range(width):
            i = row * width + col
            dbm_x10 = struct.unpack_from("<h", signal, i * 2)[0]
            dbm = dbm_x10 / 10.0
            if mask[i] == 0 or dbm < min_dbm:
                raw.extend((0, 0, 0, 0))
            else:
                raw.extend((*_color(dbm, min_dbm, max_dbm), 180))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    png += _png_chunk(b"IEND", b"")
    path = Path(out_path)
    path.write_bytes(bytes(png))
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
