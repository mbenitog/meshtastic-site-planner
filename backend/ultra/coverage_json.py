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

import numpy as np


API_COVERAGES_URL = "https://api-coverages.idee.es/collections"
DTM_5M_25830 = "EL.ElevationGridCoverage_25830_5_PB"


@dataclass
class CoverageGrid:
    width: int
    height: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    values: list[float | None]

    def __post_init__(self) -> None:
        # Precompute numpy views used by the hot path so the per-cell
        # inner loop in SurfaceBuilder.build can do a single O(1) lookup
        # instead of a per-call O(width*height) scan when the cell is None.
        #
        # _values_arr: original values with None replaced by NaN. Used by
        # min/max_value and preserved exactly: a cell that was None stays
        # NaN so we can distinguish "no data" from "real 0 m".
        # _valid: boolean mask of cells that had a real value originally.
        # _filled: _values_arr with every None cell filled from the nearest
        # non-null neighbour. This is what sample_nearest returns.
        #
        # The fill is a 2-pass scan (top->bottom then left->right,
        # bottom->top then right->left) instead of an exact Euclidean
        # nearest-neighbour fill (which would require scipy.ndimage
        # .distance_transform_edt). The 2-pass is O(N) total, propagates
        # from any direction, and is good enough for terrain
        # interpolation since missing data only appears at IGN bbox edges.
        arr = np.array(
            [np.nan if v is None else float(v) for v in self.values],
            dtype=np.float64,
        ).reshape(self.height, self.width)
        valid = ~np.isnan(arr)
        filled = arr.copy()
        if not bool(valid.all()):
            # Pass 1: top then left (propagates valid rows/columns inward).
            shifted = filled[1:, :]
            mask = (~valid[1:, :]) & valid[:-1, :]
            shifted[mask] = filled[:-1, :][mask]
            filled[1:, :] = shifted
            valid[1:, :] |= valid[:-1, :]
            shifted = filled[:, 1:]
            mask = (~valid[:, 1:]) & valid[:, :-1]
            shifted[mask] = filled[:, :-1][mask]
            filled[:, 1:] = shifted
            valid[:, 1:] |= valid[:, :-1]
            # Pass 2: bottom then right.
            shifted = filled[:-1, :]
            mask = (~valid[:-1, :]) & valid[1:, :]
            shifted[mask] = filled[1:, :][mask]
            filled[:-1, :] = shifted
            valid[:-1, :] |= valid[1:, :]
            shifted = filled[:, :-1]
            mask = (~valid[:, :-1]) & valid[:, 1:]
            shifted[mask] = filled[:, 1:][mask]
            filled[:, :-1] = shifted
            valid[:, :-1] |= valid[:, 1:]
        self._values_arr = arr
        self._valid = valid
        self._filled = filled

    @property
    def dx(self) -> float:
        return 0.0 if self.width <= 1 else (self.max_x - self.min_x) / (self.width - 1)

    @property
    def dy(self) -> float:
        return 0.0 if self.height <= 1 else (self.max_y - self.min_y) / (self.height - 1)

    @property
    def min_value(self) -> float:
        if not bool(self._valid.any()):
            return 0.0
        return float(np.nanmin(self._values_arr))

    @property
    def max_value(self) -> float:
        if not bool(self._valid.any()):
            return 0.0
        return float(np.nanmax(self._values_arr))

    def _col(self, x: float) -> int:
        if self.width == 1:
            return 0
        return min(max(int(round((x - self.min_x) / self.dx)), 0), self.width - 1)

    def _row(self, y: float) -> int:
        if self.height == 1:
            return 0
        # CoverageJSON from this API returns y axis high-to-low for EPSG:25830.
        if self.max_y >= self.min_y:
            raw = round((self.max_y - y) / abs(self.dy))
        else:
            raw = round((y - self.max_y) / abs(self.dy))
        return min(max(int(raw), 0), self.height - 1)

    def sample_nearest(self, x: float, y: float) -> float:
        ix = self._col(x)
        iy = self._row(y)
        value = float(self._filled[iy, ix])
        if math.isnan(value):
            return 0.0
        return value


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

    def fetch_grid_tiled(
        self,
        *,
        collection_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        max_cells_per_request: int = 250_000,
        timeout: float = 120.0,
    ) -> CoverageGrid:
        """Fetch a bbox by splitting it into several sub-requests. The DTM
        coverage at 5 m cells typically allows ~225 cells per side, so
        large radii would otherwise be rejected by the API."""
        width_m = max_x - min_x
        height_m = max_y - min_y
        cell_size = 5.0
        cells_x = int(width_m / cell_size) + 1
        cells_y = int(height_m / cell_size) + 1
        if cells_x * cells_y <= max_cells_per_request:
            return self.fetch_grid(
                collection_id=collection_id,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                timeout=timeout,
            )
        stride_cells = int((max_cells_per_request) ** 0.5)
        stride_m = stride_cells * cell_size
        tiles: list[CoverageGrid] = []
        x = min_x
        while x < max_x - 1e-6:
            tile_max_x = min(x + stride_m, max_x)
            y = min_y
            while y < max_y - 1e-6:
                tile_max_y = min(y + stride_m, max_y)
                tiles.append(
                    self.fetch_grid(
                        collection_id=collection_id,
                        min_x=x,
                        min_y=y,
                        max_x=tile_max_x,
                        max_y=tile_max_y,
                        timeout=timeout,
                    )
                )
                y = tile_max_y
            x = tile_max_x
        return stitch_coverage_grids(tiles, min_x, min_y, max_x, max_y, cell_size)


def stitch_coverage_grids(
    tiles: list[CoverageGrid],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    cell_size: float,
) -> CoverageGrid:
    """Stitch per-tile CoverageGrids into a single grid covering
    [min_x,max_x) x [min_y,max_y) at the requested cell size."""
    ncols = max(1, int(round((max_x - min_x) / cell_size)))
    nrows = max(1, int(round((max_y - min_y) / cell_size)))
    values: list[float | None] = [None] * (ncols * nrows)
    for tile in tiles:
        if tile.width <= 0 or tile.height <= 0:
            continue
        tile_dx = tile.dx
        tile_dy = tile.dy
        if tile_dx == 0 or tile_dy == 0:
            continue
        for r in range(tile.height):
            src_y = tile.max_y - (r + 0.5) * tile_dy
            dst_row = int(round((max_y - src_y) / cell_size - 0.5))
            if dst_row < 0 or dst_row >= nrows:
                continue
            for c in range(tile.width):
                src_x = tile.min_x + (c + 0.5) * tile_dx
                dst_col = int(round((src_x - min_x) / cell_size - 0.5))
                if dst_col < 0 or dst_col >= ncols:
                    continue
                values[dst_row * ncols + dst_col] = tile.values[r * tile.width + c]
    cleaned: list[float] = []
    for v in values:
        cleaned.append(0.0 if v is None else v)
    return CoverageGrid(
        width=ncols,
        height=nrows,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        values=cleaned,
    )
