from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import struct

from .geo import utm30_to_latlon
from .surface import SurfaceGrid


@dataclass(frozen=True)
class SurfaceArtifact:
    path: str
    meta_path: str
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


def write_surface_artifact(grid: SurfaceGrid, out_dir: str | Path) -> SurfaceArtifact:
    """Write a projected measured-surface grid for the native ultra runner.

    The binary is row-major north-to-south, west-to-east signed little-endian
    int16 meters. This is deliberately simple for the first native runner; if
    sub-meter precision matters later we can version the metadata and switch to
    float32 without changing the API shape.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    surface_path = out / "surface_i16le.bin"
    meta_path = out / "surface_meta.json"

    with surface_path.open("wb") as fp:
        for value in grid.values:
            fp.write(struct.pack("<h", max(-32768, min(32767, round(value)))))

    corners = [
        utm30_to_latlon(grid.min_x, grid.max_y),
        utm30_to_latlon(grid.max_x, grid.max_y),
        utm30_to_latlon(grid.max_x, grid.min_y),
        utm30_to_latlon(grid.min_x, grid.min_y),
    ]
    lats = [lat for lat, _ in corners]
    lons = [lon for _, lon in corners]
    artifact = SurfaceArtifact(
        path=str(surface_path),
        meta_path=str(meta_path),
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
    )
    meta_path.write_text(json.dumps(asdict(artifact), indent=2) + "\n", encoding="utf-8")
    return artifact
