from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import struct

from .geo import utm30_to_latlon
from .surface import SurfaceGrid


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

    with surface_path.open("wb") as fp:
        for value in grid.values.flat:
            fp.write(struct.pack("<h", max(-32768, min(32767, round(value)))))
    if grid.sources.size:
        with sources_path.open("wb") as fp:
            fp.write(grid.sources.tobytes())

    counts: dict[str, int] = {label: 0 for label in SOURCE_LABELS.values()}
    for s in grid.sources.flat:
        label = SOURCE_LABELS.get(int(s), "unknown")
        counts[label] = counts.get(label, 0) + 1

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
        min_value=grid.min_value,
        max_value=grid.max_value,
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
    meta_path.write_text(json.dumps(asdict(artifact), indent=2) + "\n", encoding="utf-8")
    return artifact
