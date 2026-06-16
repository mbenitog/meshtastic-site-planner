from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import subprocess

from .artifacts import SurfaceArtifact
from .geo import latlon_to_utm30
from .tiler import (
    TileSpec,
    aggregate_tiles,
    build_native_command,
    make_native_run_result,
    plan_tiles,
    tile_mask_path,
    tile_meta_path,
    tile_signal_path,
)


@dataclass(frozen=True)
class NativeRunnerInput:
    surface_meta_path: str
    surface_path: str
    output_dir: str
    frequency_mhz: float
    tx_lat: float
    tx_lon: float
    tx_height_m: float
    tx_power_w: float
    tx_gain_dbi: float
    rx_height_m: float
    rx_gain_dbi: float
    rx_sensitivity_dbm: float
    ground_dielectric: float
    ground_conductivity: float
    atmosphere_bending: float
    radio_climate: int
    polarization: int
    confidence: float
    reliability: float


@dataclass(frozen=True)
class NativeRunResult:
    status: str
    model: str
    signal_path: str | None = None
    mask_path: str | None = None
    meta_path: str | None = None
    message: str | None = None


def write_native_runner_input(
    artifact: SurfaceArtifact,
    request: dict,
    out_dir: str | Path,
) -> NativeRunnerInput:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner_input = NativeRunnerInput(
        surface_meta_path=artifact.meta_path,
        surface_path=artifact.path,
        output_dir=str(out),
        frequency_mhz=request["frequency_mhz"],
        tx_lat=request["lat"],
        tx_lon=request["lon"],
        tx_height_m=request["tx_height_m"],
        tx_power_w=request["tx_power_w"],
        tx_gain_dbi=request["tx_gain_dbi"],
        rx_height_m=request["rx_height_m"],
        rx_gain_dbi=request["rx_gain_dbi"],
        rx_sensitivity_dbm=request["rx_sensitivity_dbm"],
        ground_dielectric=request["ground_dielectric"],
        ground_conductivity=request["ground_conductivity"],
        atmosphere_bending=request["atmosphere_bending"],
        radio_climate=request["radio_climate"],
        polarization=request["polarization"],
        confidence=request["confidence"],
        reliability=request["reliability"],
    )
    path = out / "runner_input.json"
    path.write_text(json.dumps(asdict(runner_input), indent=2) + "\n", encoding="utf-8")
    return runner_input


def _run_one(
    cmd: list[str],
    *,
    timeout: float,
) -> tuple[bool, str]:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        return True, ""
    except subprocess.TimeoutExpired as exc:
        return False, f"ultra_cli timeout after {timeout:.0f}s"
    except subprocess.CalledProcessError as exc:
        return False, (exc.stderr or exc.stdout or str(exc)).strip()


def run_tiled_native_ultra(
    artifact: SurfaceArtifact,
    request: dict,
    out_dir: str | Path,
    *,
    max_tile_cells: int = 250_000,
    timeout_per_tile: float = 180.0,
) -> NativeRunResult:
    """Run the native ITM runner tile-by-tile, persist partial results to
    disk, and aggregate the full coverage output. Returns a
    :class:`NativeRunResult` in the same shape as ``run_native_ultra`` so the
    rest of the pipeline does not have to distinguish between the two paths."""
    binary = Path(os.environ.get("ULTRA_CLI", "engine/build/ultra_cli"))
    if not binary.exists():
        return NativeRunResult(
            status="missing_binary",
            model="itm_projected_grid",
            message=f"native ultra runner not found at {binary}; run engine/build_native.sh",
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_prefix = out / "coverage"
    tiles = plan_tiles(artifact.width, artifact.height, max_tile_cells=max_tile_cells)
    tx_x, tx_y = latlon_to_utm30(request["lat"], request["lon"])

    for tile in tiles:
        sig_path = tile_signal_path(out_prefix, tile)
        if sig_path.exists():
            continue  # resume support
        cmd = _build_cmd(binary, artifact, tile, request, tx_x, tx_y, out_prefix)
        ok, message = _run_one(cmd, timeout=timeout_per_tile)
        if not ok:
            return NativeRunResult(
                status="failed",
                model="itm_projected_grid",
                message=f"tile ({tile.x0},{tile.y0},{tile.width},{tile.height}) failed: {message}",
            )

    meta = aggregate_tiles(artifact=artifact, tiles=tiles, out_dir=out)
    meta["rx_sensitivity_dbm"] = request["rx_sensitivity_dbm"]
    (out / "coverage.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    payload = make_native_run_result(out_prefix)
    return NativeRunResult(**payload)


def _build_cmd(
    binary: Path,
    artifact: SurfaceArtifact,
    tile: TileSpec,
    request: dict,
    tx_x: float,
    tx_y: float,
    out_prefix: Path,
) -> list[str]:
    return [
        str(binary),
        "--surface",
        artifact.path,
        "--out",
        str(out_prefix),
        "--width",
        str(artifact.width),
        "--height",
        str(artifact.height),
        "--tile-x0",
        str(tile.x0),
        "--tile-y0",
        str(tile.y0),
        "--tile-w",
        str(tile.width),
        "--tile-h",
        str(tile.height),
        "--min-x",
        str(artifact.min_x),
        "--max-y",
        str(artifact.max_y),
        "--resolution-m",
        str(artifact.resolution_m),
        "--tx-x",
        str(tx_x),
        "--tx-y",
        str(tx_y),
        "--tx-height-m",
        str(request["tx_height_m"]),
        "--rx-height-m",
        str(request["rx_height_m"]),
        "--freq-mhz",
        str(request["frequency_mhz"]),
        "--tx-power-w",
        str(request["tx_power_w"]),
        "--tx-gain-dbi",
        str(request["tx_gain_dbi"]),
        "--rx-gain-dbi",
        str(request["rx_gain_dbi"]),
        "--rx-sensitivity-dbm",
        str(request["rx_sensitivity_dbm"]),
        "--dielect",
        str(request["ground_dielectric"]),
        "--conductivity",
        str(request["ground_conductivity"]),
        "--bend",
        str(request["atmosphere_bending"]),
        "--climate",
        str(request["radio_climate"]),
        "--pol",
        str(request["polarization"]),
        "--conf",
        str(request["confidence"]),
        "--rel",
        str(request["reliability"]),
    ]


def run_native_ultra(
    artifact: SurfaceArtifact,
    request: dict,
    out_dir: str | Path,
) -> NativeRunResult:
    """Backward-compat shim: the old single-shot runner is now expressed as a
    single full-coverage tile. New code should call
    :func:`run_tiled_native_ultra` directly.
    """
    return run_tiled_native_ultra(artifact, request, out_dir)
