"""Standalone entry point that builds the surface and writes the per-job
artifacts. Runs in its own subprocess so the cancel endpoint can SIGTERM it
during the (otherwise non-interruptible) HTTP fetches + numpy vectorization
of the ``building_surface`` phase.

Usage:
    python -m backend.ultra.surface_worker <payload.json>

Payload schema::

    {
        "request": { ... UltraJobRequest.model_dump() ... },
        "out_dir": "/abs/path/to/job/dir"
    }

On success, writes ``<out_dir>/surface_done.flag``. The main process polls
for that file (and the absence of an error file) to know the build finished.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: surface_worker.py <payload.json>\n")
        return 2

    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    request = payload["request"]
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    error_file = out_dir / "surface_error.txt"
    done_file = out_dir / "surface_done.flag"
    # Clear any stale markers from a previous (cancelled) run.
    for stale in (error_file, done_file):
        if stale.exists():
            stale.unlink()

    try:
        from backend.ultra.surface import SurfaceBuilder
        from backend.ultra.ign_dsm import IgnDsmClient
        from backend.ultra.coverage_json import CoverageApiClient
        from backend.ultra.artifacts import write_surface_artifact
        from backend.ultra.runner import write_native_runner_input

        ign = IgnDsmClient()
        coverages = CoverageApiClient()
        builder = SurfaceBuilder(dsm_client=ign, coverage_client=coverages)

        grid = builder.build(
            lat=request["lat"],
            lon=request["lon"],
            radius_m=request["radius_km"] * 1000,
            resolution_m=request["resolution_m"],
            mode=request["surface_mode"],
        )
        artifact = write_surface_artifact(grid, out_dir)
        write_native_runner_input(artifact, request, out_dir)
    except Exception as exc:
        error_file.write_text(
            "".join(traceback.format_exception(exc)) + "\n",
            encoding="utf-8",
        )
        sys.stderr.write(f"surface_worker failed: {exc}\n")
        return 1

    done_file.write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())