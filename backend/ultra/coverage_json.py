from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError
import concurrent.futures
import hashlib
import json
import logging
import os
import subprocess
import math

import numpy as np


logger = logging.getLogger("uvicorn.error")

API_COVERAGES_URL = "https://api-coverages.idee.es/collections"
DTM_5M_25830 = "EL.ElevationGridCoverage_25830_5_PB"
DEFAULT_DOWNLOAD_CONCURRENCY = int(os.environ.get("IGN_DOWNLOAD_CONCURRENCY", "16"))


@dataclass
class CoverageGrid:
    width: int
    height: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    values: list[float | None] | np.ndarray
    fill_on_init: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.values, np.ndarray):
            arr = np.asarray(self.values, dtype=np.float32).reshape(self.height, self.width)
        else:
            arr = np.array(
                [np.nan if v is None else float(v) for v in self.values],
                dtype=np.float32,
            ).reshape(self.height, self.width)
        valid = ~np.isnan(arr)
        if self.fill_on_init and not bool(valid.all()):
            filled = arr.copy()
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
        else:
            filled = arr
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


def parse_coverage_json(text: str, *, fill_on_init: bool = True) -> CoverageGrid:
    payload = json.loads(text)
    axes = payload["domain"]["axes"]
    x_axis = axes["x"]
    y_axis = axes["y"]
    range_obj = next(iter(payload["ranges"].values()))
    raw_vals = range_obj["values"]
    arr = np.fromiter(
        (np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in raw_vals),
        dtype=np.float32,
        count=len(raw_vals),
    )
    width = int(x_axis["num"])
    height = int(y_axis["num"])
    if arr.size != width * height:
        raise ValueError(f"CoverageJSON size mismatch: expected {width * height}, got {arr.size}")
    y0 = float(y_axis["start"])
    y1 = float(y_axis["stop"])
    return CoverageGrid(
        width=width,
        height=height,
        min_x=min(float(x_axis["start"]), float(x_axis["stop"])),
        max_x=max(float(x_axis["start"]), float(x_axis["stop"])),
        min_y=min(y0, y1),
        max_y=max(y0, y1),
        values=arr,
        fill_on_init=fill_on_init,
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
        timeout: float = 35.0,
        fill_on_init: bool = True,
    ) -> CoverageGrid:
        params = {
            "bbox": f"{min_x:.3f},{min_y:.3f},{max_x:.3f},{max_y:.3f}",
            "bbox-crs": "25830",
            "f": "json",
        }
        key_params = {"collection": collection_id, **params}
        cache_key = _cache_key(key_params)
        cache_npy = self.cache_dir / f"{cache_key}.npy"
        cache_meta = self.cache_dir / f"{cache_key}.meta.json"
        if cache_npy.exists() and cache_meta.exists():
            try:
                meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                arr = np.load(cache_npy)
                return CoverageGrid(
                    width=meta["width"],
                    height=meta["height"],
                    min_x=meta["min_x"],
                    max_x=meta["max_x"],
                    min_y=meta["min_y"],
                    max_y=meta["max_y"],
                    values=arr,
                    fill_on_init=fill_on_init,
                )
            except Exception:
                pass

        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            grid = parse_coverage_json(cache_path.read_text(encoding="utf-8"), fill_on_init=fill_on_init)
            try:
                np.save(cache_npy, grid.values)
                cache_meta.write_text(json.dumps({
                    "width": grid.width,
                    "height": grid.height,
                    "min_x": grid.min_x,
                    "max_x": grid.max_x,
                    "min_y": grid.min_y,
                    "max_y": grid.max_y,
                }), encoding="utf-8")
            except Exception:
                pass
            return grid

        url = f"{API_COVERAGES_URL}/{collection_id}/coverage?{urlencode(params, safe=',')}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last_error = None
        for attempt in range(3):
            try:
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
                grid = parse_coverage_json(text, fill_on_init=fill_on_init)
                try:
                    np.save(cache_npy, grid.values)
                    cache_meta.write_text(json.dumps({
                        "width": grid.width,
                        "height": grid.height,
                        "min_x": grid.min_x,
                        "max_x": grid.max_x,
                        "min_y": grid.min_y,
                        "max_y": grid.max_y,
                    }), encoding="utf-8")
                except Exception:
                    pass
                return grid
            except Exception as exc:
                last_error = exc
                import time
                time.sleep(1.0 * (attempt + 1))
        raise last_error

    def fetch_grid_tiled(
        self,
        *,
        collection_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        max_cells_per_request: int = 250_000,
        timeout: float = 35.0,
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
        sub_boxes: list[tuple[float, float, float, float]] = []
        x = min_x
        while x < max_x - 1e-6:
            tile_max_x = min(x + stride_m, max_x)
            y = min_y
            while y < max_y - 1e-6:
                tile_max_y = min(y + stride_m, max_y)
                sub_boxes.append((x, y, tile_max_x, tile_max_y))
                y = tile_max_y
            x = tile_max_x

        tiles: list[CoverageGrid] = []
        max_workers = min(DEFAULT_DOWNLOAD_CONCURRENCY, max(1, len(sub_boxes)))
        logger.info("  📥 [IGN DTM] Fetching '%s': %d sub-tiles (%d workers parallel)", collection_id, len(sub_boxes), max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self.fetch_grid,
                    collection_id=collection_id,
                    min_x=bx0,
                    min_y=by0,
                    max_x=bx1,
                    max_y=by1,
                    timeout=timeout,
                    fill_on_init=False,
                )
                for (bx0, by0, bx1, by1) in sub_boxes
            ]
            completed = 0
            total_sub = len(sub_boxes)
            for fut in concurrent.futures.as_completed(futures):
                tiles.append(fut.result())
                completed += 1
                if completed % 10 == 0 or completed == total_sub:
                    logger.info("  ↳ [IGN DTM] '%s' progress: %d/%d sub-tiles fetched (%.0f%%)",
                                collection_id, completed, total_sub, (completed / total_sub) * 100)
        logger.info("  📦 [IGN DTM] '%s' download complete: %d/%d tiles received, stitching...",
                    collection_id, len(tiles), len(sub_boxes))

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
    grid = np.zeros((nrows, ncols), dtype=np.float32)

    for tile in tiles:
        if tile.width <= 0 or tile.height <= 0:
            continue
        tile_dx = tile.dx
        tile_dy = tile.dy
        if tile_dx == 0 or tile_dy == 0:
            continue

        col0 = int(round((tile.min_x - min_x) / cell_size))
        row0 = int(round((max_y - tile.max_y) / cell_size))

        dst_c0 = max(0, col0)
        dst_c1 = min(ncols, col0 + tile.width)
        dst_r0 = max(0, row0)
        dst_r1 = min(nrows, row0 + tile.height)

        if dst_c1 <= dst_c0 or dst_r1 <= dst_r0:
            continue

        src_c0 = dst_c0 - col0
        src_c1 = src_c0 + (dst_c1 - dst_c0)
        src_r0 = dst_r0 - row0
        src_r1 = src_r0 + (dst_r1 - dst_r0)

        tile_data = np.asarray(tile._filled, dtype=np.float32)
        grid[dst_r0:dst_r1, dst_c0:dst_c1] = tile_data[src_r0:src_r1, src_c0:src_c1]

    return CoverageGrid(
        width=ncols,
        height=nrows,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        values=grid.ravel(),
    )
