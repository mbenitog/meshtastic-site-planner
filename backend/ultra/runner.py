from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import subprocess

from .artifacts import SurfaceArtifact
from .geo import latlon_to_utm30


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


def run_native_ultra(
    artifact: SurfaceArtifact,
    request: dict,
    out_dir: str | Path,
) -> NativeRunResult:
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
    tx_x, tx_y = latlon_to_utm30(request["lat"], request["lon"])
    cmd = [
        str(binary),
        "--surface",
        artifact.path,
        "--out",
        str(out_prefix),
        "--width",
        str(artifact.width),
        "--height",
        str(artifact.height),
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
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        return NativeRunResult(
            status="failed",
            model="itm_projected_grid",
            message=(exc.stderr or exc.stdout or str(exc)).strip(),
        )

    return NativeRunResult(
        status="complete",
        model="itm_projected_grid",
        signal_path=str(out_prefix) + ".signal_i16le.bin",
        mask_path=str(out_prefix) + ".mask_u8.bin",
        meta_path=str(out_prefix) + ".meta.json",
    )
