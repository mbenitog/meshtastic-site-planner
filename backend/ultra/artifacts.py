from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import struct

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
    )
    meta_path.write_text(json.dumps(asdict(artifact), indent=2) + "\n", encoding="utf-8")
    return artifact
