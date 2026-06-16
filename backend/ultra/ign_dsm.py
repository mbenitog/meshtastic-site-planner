from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import re
import subprocess
from urllib.error import URLError


WCS_URL = "https://wcs-mds.idee.es/mds"


@dataclass(frozen=True)
class ArcGrid:
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    values: list[float]
    nodata: float | None = None

    @property
    def min_value(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max_value(self) -> float:
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
    lines = [line.strip() for line in asc.splitlines() if line.strip()]
    header: dict[str, float] = {}
    data_start = 0
    for i, line in enumerate(lines):
        parts = line.split()
        key = parts[0].lower()
        if key not in {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value"}:
            data_start = i
            break
        header[key] = float(parts[1])

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    nodata_value = header.get("nodata_value")
    values: list[float] = []
    nodata = nodata_value
    for line in lines[data_start:]:
        for raw in line.split():
            value = float(raw)
            if nodata is None or value != nodata:
                values.append(value)
    if len(values) != ncols * nrows:
        raise ValueError(f"ArcGrid size mismatch: expected {ncols * nrows}, got {len(values)}")
    return ArcGrid(
        ncols=ncols,
        nrows=nrows,
        xllcorner=header["xllcorner"],
        yllcorner=header["yllcorner"],
        cellsize=header["cellsize"],
        values=values,
        nodata=nodata,
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
        timeout: float = 120.0,
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
        cache_path = self.cache_dir / f"{_cache_key(key_params)}.asc"
        if cache_path.exists():
            return parse_arcgrid(cache_path.read_text(encoding="utf-8"))

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
        return parse_arcgrid(text)

    def fetch_arcgrid_tiled(
        self,
        *,
        coverage_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        max_cells_per_request: int = 1_100_000,
        timeout: float = 120.0,
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
        tiles: list[ArcGrid] = []
        x = min_x
        while x < max_x - 1e-6:
            tile_max_x = min(x + stride_m, max_x)
            y = min_y
            while y < max_y - 1e-6:
                tile_max_y = min(y + stride_m, max_y)
                try:
                    tiles.append(
                        self.fetch_arcgrid(
                            coverage_id=coverage_id,
                            min_x=x,
                            min_y=y,
                            max_x=tile_max_x,
                            max_y=tile_max_y,
                            timeout=timeout,
                        )
                    )
                except Exception:
                    pass  # sub-tile outside coverage; treat as no-data
                y = tile_max_y
            x = tile_max_x
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
    [min_y,max_y) at the requested cell size. Tiles are assumed to be
    contiguous (or overlapping at most at the cell boundary) and aligned
    to the same cell size."""
    ncols = max(1, int(round((max_x - min_x) / cell_size)))
    nrows = max(1, int(round((max_y - min_y) / cell_size)))
    values: list[float | None] = [None] * (ncols * nrows)
    nodata_value: float | None = None
    for tile in tiles:
        t_ncols = tile.ncols
        t_nrows = tile.nrows
        t_x0 = tile.xllcorner
        t_y0 = tile.yllcorner
        if t_ncols <= 0 or t_nrows <= 0:
            continue
        t_x1 = t_x0 + (t_ncols - 1) * tile.cellsize
        t_y1 = t_y0 + (t_nrows - 1) * tile.cellsize
        col0 = max(0, int(round((t_x0 - min_x) / cell_size)))
        row0 = max(0, int(round((max_y - t_y1) / cell_size)))
        for r in range(t_nrows):
            src_y = t_y0 + (t_nrows - 1 - r) * tile.cellsize
            dst_row = max(0, int(round((max_y - src_y) / cell_size)))
            if dst_row < 0 or dst_row >= nrows:
                continue
            for c in range(t_ncols):
                src_x = t_x0 + c * tile.cellsize
                dst_col = int(round((src_x - min_x) / cell_size))
                if dst_col < 0 or dst_col >= ncols:
                    continue
                value = tile.values[r * t_ncols + c]
                values[dst_row * ncols + dst_col] = value
                if tile.nodata is not None and value == tile.nodata:
                    nodata_value = tile.nodata
    cleaned: list[float] = []
    for v in values:
        if v is None:
            cleaned.append(-9999.0)
            nodata_value = -9999.0
        else:
            cleaned.append(v)
    return ArcGrid(
        ncols=ncols,
        nrows=nrows,
        xllcorner=min_x,
        yllcorner=min_y,
        cellsize=cell_size,
        values=cleaned,
        nodata=nodata_value,
    )
