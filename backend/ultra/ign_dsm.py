from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import subprocess
from urllib.error import URLError


import numpy as np


logger = logging.getLogger("uvicorn.error")

WCS_URL = "https://wcs-mds.idee.es/mds"
DEFAULT_DOWNLOAD_CONCURRENCY = int(os.environ.get("IGN_DOWNLOAD_CONCURRENCY", "16"))


@dataclass(frozen=True)
class ArcGrid:
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    values: np.ndarray | list[float]
    nodata: float | None = None

    @property
    def min_value(self) -> float:
        if isinstance(self.values, np.ndarray):
            return float(self.values.min()) if self.values.size else 0.0
        return min(self.values) if self.values else 0.0

    @property
    def max_value(self) -> float:
        if isinstance(self.values, np.ndarray):
            return float(self.values.max()) if self.values.size else 0.0
        return max(self.values) if self.values else 0.0


def _cache_key(params: dict[str, str]) -> str:
    encoded = urlencode(sorted(params.items()))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _extract_asc_from_multipart(text: str) -> str:
    marker = "Content-ID: coverage/out.asc"
    start = text.find(marker)
    if start == -1:
        # Some servers may return plain ArcGrid without multipart wrapping.
        return text
    body_start = text.find("\n\n", start)
    if body_start == -1:
        body_start = text.find("\r\n\r\n", start)
        sep_len = 4
    else:
        sep_len = 2
    if body_start == -1:
        raise ValueError("ArcGrid multipart body not found")
    body = text[body_start + sep_len :]
    boundary = re.search(r"\r?\n--\w+", body)
    return body[: boundary.start()].strip() if boundary else body.strip()


def parse_arcgrid(text: str) -> ArcGrid:
    asc = _extract_asc_from_multipart(text)
    header: dict[str, float] = {}
    pos = 0
    # Parse header lines without splitting the entire multi-megabyte string into millions of python objects
    for _ in range(12):
        line_start = pos
        nl = asc.find("\n", pos)
        if nl == -1:
            break
        line = asc[pos:nl].strip()
        pos = nl + 1
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() in {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value"}:
            header[parts[0].lower()] = float(parts[1])
        else:
            pos = line_start
            break

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    nodata_value = header.get("nodata_value")
    arr = np.fromstring(asc[pos:], sep=" ", dtype=np.float32)
    if arr.size != ncols * nrows:
        raise ValueError(f"ArcGrid size mismatch: expected {ncols * nrows}, got {arr.size}")
    return ArcGrid(
        ncols=ncols,
        nrows=nrows,
        xllcorner=header["xllcorner"],
        yllcorner=header["yllcorner"],
        cellsize=header["cellsize"],
        values=arr,
        nodata=nodata_value,
    )


class IgnDsmClient:
    """Small WCS client for Spain IGN DSM coverages.

    Coverage IDs currently observed:
    - mds05: absolute 5 m DSM in EPSG:25830.
    - mdsn_v025 / mdsn_e025: 2.5 m normalized DSM layers. These are heights
      above ground, so they must not be used as absolute terrain elevation by
      themselves in RF calculations.
    """

    def __init__(self, cache_dir: str | Path = ".cache/ign-dsm") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_arcgrid(
        self,
        *,
        coverage_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        timeout: float = 35.0,
    ) -> ArcGrid:
        params = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "coverageId": coverage_id,
            "subset": [f"x({min_x:.3f},{max_x:.3f})", f"y({min_y:.3f},{max_y:.3f})"],
            "format": "application/asc",
        }
        key_params = {
            "coverageId": coverage_id,
            "min_x": f"{min_x:.3f}",
            "min_y": f"{min_y:.3f}",
            "max_x": f"{max_x:.3f}",
            "max_y": f"{max_y:.3f}",
        }
        cache_key = _cache_key(key_params)
        cache_npy = self.cache_dir / f"{cache_key}.npy"
        cache_meta = self.cache_dir / f"{cache_key}.meta.json"
        if cache_npy.exists() and cache_meta.exists():
            try:
                meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                arr = np.load(cache_npy)
                return ArcGrid(
                    ncols=meta["ncols"],
                    nrows=meta["nrows"],
                    xllcorner=meta["xllcorner"],
                    yllcorner=meta["yllcorner"],
                    cellsize=meta["cellsize"],
                    values=arr,
                    nodata=meta.get("nodata"),
                )
            except Exception:
                pass

        cache_path = self.cache_dir / f"{cache_key}.asc"
        if cache_path.exists():
            grid = parse_arcgrid(cache_path.read_text(encoding="utf-8"))
            try:
                np.save(cache_npy, grid.values)
                cache_meta.write_text(json.dumps({
                    "ncols": grid.ncols,
                    "nrows": grid.nrows,
                    "xllcorner": grid.xllcorner,
                    "yllcorner": grid.yllcorner,
                    "cellsize": grid.cellsize,
                    "nodata": grid.nodata,
                }), encoding="utf-8")
            except Exception:
                pass
            return grid

        # The WCS endpoint accepts normal query encoding, but has been observed
        # to reset connections when subset parentheses are percent-encoded.
        url = f"{WCS_URL}?{urlencode(params, doseq=True, safe='(),')}"
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
        grid = parse_arcgrid(text)
        try:
            np.save(cache_npy, grid.values)
            cache_meta.write_text(json.dumps({
                "ncols": grid.ncols,
                "nrows": grid.nrows,
                "xllcorner": grid.xllcorner,
                "yllcorner": grid.yllcorner,
                "cellsize": grid.cellsize,
                "nodata": grid.nodata,
            }), encoding="utf-8")
        except Exception:
            pass
        return grid

    def fetch_arcgrid_tiled(
        self,
        *,
        coverage_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        max_cells_per_request: int = 1_100_000,
        timeout: float = 35.0,
    ) -> ArcGrid:
        """Fetch a potentially large bbox by splitting it into several WCS
        sub-requests and stitching the resulting ArcGrids into a single grid.

        This is needed because the IGN WCS endpoint rejects bboxes that would
        produce a response larger than about 1.2 million cells (i.e. ~3 km at
        2.5 m resolution). The split is based on the WCS cell size for the
        coverage (2.5 m for ``mdsn_*``, 5 m for ``mds05``). Sub-requests that
        fall outside the available data are skipped and contribute no-data
        cells in the stitched result, so the caller can fall back to a
        coarser source for those cells.
        """
        cell_size = 2.5 if coverage_id.startswith("mdsn_") else 5.0
        width_m = max_x - min_x
        height_m = max_y - min_y
        cells_x = int(width_m / cell_size) + 1
        cells_y = int(height_m / cell_size) + 1
        if cells_x * cells_y <= max_cells_per_request:
            return self.fetch_arcgrid(
                coverage_id=coverage_id,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                timeout=timeout,
            )

        # Decide a tile size in cells. Keep the cell-size-based stride so
        # adjacent tiles are easy to align.
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

        tiles: list[ArcGrid] = []
        max_workers = min(DEFAULT_DOWNLOAD_CONCURRENCY, max(1, len(sub_boxes)))
        logger.info("  📥 [IGN WCS] Fetching '%s': %d sub-tiles (%d workers parallel)", coverage_id, len(sub_boxes), max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self.fetch_arcgrid,
                    coverage_id=coverage_id,
                    min_x=bx0,
                    min_y=by0,
                    max_x=bx1,
                    max_y=by1,
                    timeout=timeout,
                )
                for (bx0, by0, bx1, by1) in sub_boxes
            ]
            completed_tiles = 0
            total_sub = len(sub_boxes)
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    tiles.append(res)
                except Exception:
                    pass  # sub-tile outside coverage; treat as no-data
                completed_tiles += 1
                if completed_tiles % 10 == 0 or completed_tiles == total_sub:
                    logger.info("  ↳ [IGN WCS] '%s' progress: %d/%d sub-tiles fetched (%.0f%%)",
                                coverage_id, completed_tiles, total_sub, (completed_tiles / total_sub) * 100)
        logger.info("  📦 [IGN WCS] '%s' download complete: %d/%d tiles received, stitching...",
                    coverage_id, len(tiles), len(sub_boxes))

        if not tiles:
            # Fall back to a single direct call so the caller still gets an
            # explicit exception (e.g. the bbox is entirely outside data).
            return self.fetch_arcgrid(
                coverage_id=coverage_id,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                timeout=timeout,
            )
        return stitch_arcgrids(tiles, min_x, min_y, max_x, max_y, cell_size)


def stitch_arcgrids(
    tiles: list[ArcGrid],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    cell_size: float,
) -> ArcGrid:
    """Stitch per-tile ArcGrids into a single grid covering [min_x,max_x) x
    [min_y,max_y) at the requested cell size. Optimized with numpy 2D block
    slicing to avoid Python loop overhead and minimize RAM usage."""
    ncols = max(1, int(round((max_x - min_x) / cell_size)))
    nrows = max(1, int(round((max_y - min_y) / cell_size)))
    grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    nodata_value: float = -9999.0

    for tile in tiles:
        t_ncols = tile.ncols
        t_nrows = tile.nrows
        if t_ncols <= 0 or t_nrows <= 0:
            continue
        t_x0 = tile.xllcorner
        t_y0 = tile.yllcorner
        t_y1 = t_y0 + t_nrows * tile.cellsize

        col0 = int(round((t_x0 - min_x) / cell_size))
        row0 = int(round((max_y - t_y1) / cell_size))

        dst_c0 = max(0, col0)
        dst_c1 = min(ncols, col0 + t_ncols)
        dst_r0 = max(0, row0)
        dst_r1 = min(nrows, row0 + t_nrows)

        if dst_c1 <= dst_c0 or dst_r1 <= dst_r0:
            continue

        src_c0 = dst_c0 - col0
        src_c1 = src_c0 + (dst_c1 - dst_c0)
        src_r0 = dst_r0 - row0
        src_r1 = src_r0 + (dst_r1 - dst_r0)

        tile_data = np.asarray(tile.values, dtype=np.float32).reshape(t_nrows, t_ncols)
        tile_chunk = tile_data[src_r0:src_r1, src_c0:src_c1]

        target_slice = grid[dst_r0:dst_r1, dst_c0:dst_c1]
        if tile.nodata is not None:
            valid_mask = tile_chunk != tile.nodata
            target_slice[valid_mask] = tile_chunk[valid_mask]
        else:
            target_slice[:] = tile_chunk

    np.nan_to_num(grid, copy=False, nan=nodata_value)

    return ArcGrid(
        ncols=ncols,
        nrows=nrows,
        xllcorner=min_x,
        yllcorner=min_y,
        cellsize=cell_size,
        values=grid.ravel(),
        nodata=nodata_value,
    )
