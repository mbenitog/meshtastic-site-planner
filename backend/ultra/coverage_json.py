from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError
import hashlib
import json
import subprocess
import math


API_COVERAGES_URL = "https://api-coverages.idee.es/collections"
DTM_5M_25830 = "EL.ElevationGridCoverage_25830_5_PB"


@dataclass(frozen=True)
class CoverageGrid:
    width: int
    height: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    values: list[float | None]

    @property
    def dx(self) -> float:
        return 0.0 if self.width <= 1 else (self.max_x - self.min_x) / (self.width - 1)

    @property
    def dy(self) -> float:
        return 0.0 if self.height <= 1 else (self.max_y - self.min_y) / (self.height - 1)

    @property
    def min_value(self) -> float:
        vals = [v for v in self.values if v is not None]
        return min(vals) if vals else 0.0

    @property
    def max_value(self) -> float:
        vals = [v for v in self.values if v is not None]
        return max(vals) if vals else 0.0

    def sample_nearest(self, x: float, y: float) -> float:
        if self.width == 1:
            ix = 0
        else:
            ix = round((x - self.min_x) / self.dx)
        if self.height == 1:
            iy = 0
        else:
            # CoverageJSON from this API returns y axis high-to-low for EPSG:25830.
            if self.max_y >= self.min_y:
                iy = round((self.max_y - y) / abs(self.dy))
            else:
                iy = round((y - self.max_y) / abs(self.dy))
        ix = min(max(int(ix), 0), self.width - 1)
        iy = min(max(int(iy), 0), self.height - 1)
        value = self.values[iy * self.width + ix]
        if value is not None:
            return value
        # Fill sparse null cells from the nearest measured neighbour. This is
        # preferable to creating artificial 0 m terrain holes in the RF surface.
        best: tuple[int, float] | None = None
        for j in range(self.height):
            for i in range(self.width):
                candidate = self.values[j * self.width + i]
                if candidate is None:
                    continue
                dist2 = (i - ix) * (i - ix) + (j - iy) * (j - iy)
                if best is None or dist2 < best[0]:
                    best = (dist2, candidate)
        if best is None:
            return 0.0
        return best[1]


def _cache_key(params: dict[str, str]) -> str:
    encoded = urlencode(sorted(params.items()))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_coverage_json(text: str) -> CoverageGrid:
    payload = json.loads(text)
    axes = payload["domain"]["axes"]
    x_axis = axes["x"]
    y_axis = axes["y"]
    range_obj = next(iter(payload["ranges"].values()))
    values = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in range_obj["values"]]
    width = int(x_axis["num"])
    height = int(y_axis["num"])
    if len(values) != width * height:
        raise ValueError(f"CoverageJSON size mismatch: expected {width * height}, got {len(values)}")
    y0 = float(y_axis["start"])
    y1 = float(y_axis["stop"])
    return CoverageGrid(
        width=width,
        height=height,
        min_x=min(float(x_axis["start"]), float(x_axis["stop"])),
        max_x=max(float(x_axis["start"]), float(x_axis["stop"])),
        min_y=min(y0, y1),
        max_y=max(y0, y1),
        values=values,
    )


class CoverageApiClient:
    def __init__(self, cache_dir: str | Path = ".cache/ign-coverages") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_grid(
        self,
        *,
        collection_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        timeout: float = 120.0,
    ) -> CoverageGrid:
        params = {
            "bbox": f"{min_x:.3f},{min_y:.3f},{max_x:.3f},{max_y:.3f}",
            "bbox-crs": "25830",
            "f": "json",
        }
        key_params = {"collection": collection_id, **params}
        cache_path = self.cache_dir / f"{_cache_key(key_params)}.json"
        if cache_path.exists():
            return parse_coverage_json(cache_path.read_text(encoding="utf-8"))

        url = f"{API_COVERAGES_URL}/{collection_id}/coverage?{urlencode(params, safe=',')}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=timeout) as response:
                text = response.read().decode("utf-8")
        except URLError:
            result = subprocess.run(
                ["curl", "-L", "-A", "Mozilla/5.0", "--max-time", str(int(timeout)), url],
                check=True,
                capture_output=True,
            )
            text = result.stdout.decode("utf-8")
        cache_path.write_text(text, encoding="utf-8")
        return parse_coverage_json(text)
