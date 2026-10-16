from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import struct
import numpy as np

from .geo import utm30_to_latlon
from .surface import SurfaceGrid


def sanitize_for_json(data: Any) -> Any:
    """Recursively convert NaN and Infinity floats to None (JSON null)."""
    if isinstance(data, (float, np.floating)):
        f_val = float(data)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return f_val
    if isinstance(data, (int, np.integer)):
        return int(data)
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [sanitize_for_json(v) for v in data]
    return data


SOURCE_LABELS = {
    0: "dtm_only",
    1: "dtm_plus_mdsn_e025",
    2: "dtm_plus_mdsn_v025",
    3: "dtm_plus_mds05",
}


@dataclass(frozen=True)
class SurfaceArtifact:
    path: str
    meta_path: str
    sources_path: str
    width: int
    height: int
    resolution_m: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_value: float
    max_value: float
    bounds_wgs84: dict[str, float]
    corners_wgs84: list[list[float]]
    mode: str
    source_counts: dict[str, int] = field(default_factory=dict)


def write_surface_artifact(grid: SurfaceGrid, out_dir: str | Path) -> SurfaceArtifact:
    """Write the projected measured surface grid plus a per-cell source mask.

    The surface binary stays little-endian int16 meters (consumed by the
    native ultra_cli). The sources binary is one byte per cell, matching the
    source-label dictionary in :data:`SOURCE_LABELS`. The metadata JSON
    embeds per-source counts so the backend can show how much of the grid
    used 2.5 m detail versus a coarser fallback.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    surface_path = out / "surface_i16le.bin"
    sources_path = out / "surface_sources_u8.bin"
    meta_path = out / "surface_meta.json"

    values_arr = np.asarray(grid.values, dtype=np.float32).reshape(-1)
    clean_vals = np.nan_to_num(values_arr, nan=0.0)
    values_i16 = np.clip(np.round(clean_vals), -32768, 32767).astype("<i2")
    surface_path.write_bytes(values_i16.tobytes())

    sources_arr = np.asarray(grid.sources, dtype=np.uint8).reshape(-1) if hasattr(grid.sources, "__len__") and len(grid.sources) > 0 else np.zeros(0, dtype=np.uint8)
    if sources_arr.size:
        sources_path.write_bytes(sources_arr.tobytes())

    counts: dict[str, int] = {label: 0 for label in SOURCE_LABELS.values()}
    if sources_arr.size > 0:
        unique_vals, unique_counts = np.unique(sources_arr, return_counts=True)
        for code, cnt in zip(unique_vals, unique_counts):
            label = SOURCE_LABELS.get(int(code), "unknown")
            counts[label] = int(cnt)

    half = grid.resolution_m / 2.0
    # Grid coordinates are sample centers; overlay/image corners need the
    # outer pixel edges so the rendered image aligns with the basemap.
    corners = [
        utm30_to_latlon(grid.min_x - half, grid.max_y + half),
        utm30_to_latlon(grid.max_x + half, grid.max_y + half),
        utm30_to_latlon(grid.max_x + half, grid.min_y - half),
        utm30_to_latlon(grid.min_x - half, grid.min_y - half),
    ]
    lats = [lat for lat, _ in corners]
    lons = [lon for _, lon in corners]

    min_val = grid.min_value
    max_val = grid.max_value
    if min_val is not None and (math.isnan(min_val) or math.isinf(min_val)):
        min_val = None
    if max_val is not None and (math.isnan(max_val) or math.isinf(max_val)):
        max_val = None

    artifact = SurfaceArtifact(
        path=str(surface_path),
        meta_path=str(meta_path),
        sources_path=str(sources_path),
        width=grid.width,
        height=grid.height,
        resolution_m=grid.resolution_m,
        min_x=grid.min_x,
        min_y=grid.min_y,
        max_x=grid.max_x,
        max_y=grid.max_y,
        min_value=min_val,
        max_value=max_val,
        bounds_wgs84={
            "north": max(lats),
            "south": min(lats),
            "east": max(lons),
            "west": min(lons),
        },
        corners_wgs84=[[lon, lat] for lat, lon in corners],
        mode=grid.mode,
        source_counts=counts,
    )
    meta_path.write_text(
        json.dumps(sanitize_for_json(asdict(artifact)), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact
