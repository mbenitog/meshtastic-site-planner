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
    values: list[float] = []
    nodata = header.get("nodata_value")
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
