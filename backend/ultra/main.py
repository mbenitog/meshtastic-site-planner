from __future__ import annotations

import json
import multiprocessing
import os
import signal
import threading
import time
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

logger = logging.getLogger("uvicorn.error")
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .geo import meter_bbox_around
from .ign_dsm import IgnDsmClient
from .surface import SurfaceBuilder, SurfaceMode
from .artifacts import SurfaceArtifact
from .runner import NativeRunnerInput, NativeRunResult
from .render import write_coverage_png, write_png_world_file
from .tiler import plan_tiles, aggregate_tiles, make_native_run_result


ULTRA_API_PREFIX = os.environ.get("ULTRA_API_PREFIX", "/ultra-api").rstrip("/") or ""
ULTRA_PUBLIC_BASE_URL = os.environ.get("ULTRA_PUBLIC_BASE_URL", "").rstrip("/")


def _public_base_url(request: Request) -> str:
    """Return the absolute base URL the browser should use to reach this API.

    Order of precedence:
      1. ``ULTRA_PUBLIC_BASE_URL`` env var (lets operators pin a public origin
         regardless of reverse-proxy headers).
      2. ``X-Forwarded-Proto`` + ``X-Forwarded-Host`` from the request, so a
         reverse proxy that preserves ``Host`` works without extra config.
      3. The request's own scheme + host as a final fallback for direct
         connections.
    """
    if ULTRA_PUBLIC_BASE_URL:
        return ULTRA_PUBLIC_BASE_URL
    fwd_proto = request.headers.get("x-forwarded-proto")
    fwd_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if fwd_proto and fwd_host:
        return f"{fwd_proto}://{fwd_host}".rstrip("/")
    return str(request.base_url).rstrip("/")


router = APIRouter()
ign = IgnDsmClient()
surface_builder = SurfaceBuilder(dsm_client=ign)
jobs: dict[str, dict] = {}
JOB_ROOT = Path(".cache/ultra-jobs")
TILE_MAX_CELLS = 250_000
SINGLE_TILE_CELL_THRESHOLD = 1_000_000
_cached_gpu_tile_cells: int | None = None


def _resolve_native_binary() -> tuple[Path, str]:
    binary = Path(os.environ.get("ULTRA_CLI", "engine/build/ultra_cli")).resolve()
    opencl_kernel = os.environ.get("ULTRA_OPENCL_KERNEL", "")
    if not opencl_kernel:
        for cand in [Path("/usr/local/share/ultra/itm_kernel.cl"), Path("engine/opencl/itm_kernel.cl")]:
            if cand.exists():
                opencl_kernel = str(cand.resolve())
                break
    if opencl_kernel:
        candidates = [
            binary.with_name("ultra_cli_opencl"),
            Path("/usr/local/bin/ultra_cli_opencl"),
            Path("engine/build/ultra_cli_opencl"),
        ]
        for candidate in candidates:
            if candidate.exists():
                binary = candidate.resolve()
                break
    return binary, opencl_kernel


_cached_gpu_info: dict | None = None


def _probe_gpu(binary_path: Path) -> dict | None:
    global _cached_gpu_info
    if _cached_gpu_info is not None:
        return _cached_gpu_info
    try:
        import subprocess
        proc = subprocess.run([str(binary_path), "--probe"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            _cached_gpu_info = json.loads(proc.stdout)
            return _cached_gpu_info
    except Exception as exc:
        logger.warning("Could not probe OpenCL device: %s", exc)
    return None


def get_optimal_tile_cells(binary_path: Path, opencl_kernel: str) -> int:
    env_override = os.environ.get("ULTRA_TILE_MAX_CELLS")
    if env_override:
        return max(50_000, int(env_override))
    if not opencl_kernel or binary_path.name != "ultra_cli_opencl":
        return TILE_MAX_CELLS
    info = _probe_gpu(binary_path)
    if not info:
        return TILE_MAX_CELLS
    vram_mb = info.get("vram_mb", 0)
    if vram_mb >= 12000:
        tile_cells = 4_000_000
    elif vram_mb >= 6000:
        tile_cells = 2_000_000
    elif vram_mb >= 3000:
        tile_cells = 1_000_000
    elif vram_mb >= 1500:
        tile_cells = 500_000
    else:
        tile_cells = 250_000
    logger.info("🎮 [GPU Auto-scale] Device: %s (%d MB VRAM) -> optimal tile size: %d cells",
                info.get("device", "GPU"), vram_mb, tile_cells)
    return tile_cells


def get_optimal_gpu_workers(
    binary_path: Path,
    opencl_kernel: str,
    total_tiles: int,
    cpu_count: int,
) -> int:
    env_override = os.environ.get("ULTRA_TILE_WORKERS")
    if env_override is not None:
        return min(max(1, int(env_override)), max(1, total_tiles))
    if not opencl_kernel or binary_path.name != "ultra_cli_opencl":
        return min(cpu_count, max(1, total_tiles))
    info = _probe_gpu(binary_path)
    if not info:
        return 1
    vram_mb = info.get("vram_mb", 0)
    max_alloc_mb = info.get("max_alloc_mb", 2048)
    if vram_mb <= 0:
        return 1

    # 85% of total VRAM budget
    vram_budget_mb = int(vram_mb * 0.85)
    # Estimate memory per worker in MB:
    # Batch buffer capped at max_alloc_mb (~2048 MB on NVIDIA) + surface/fixed buffers
    worker_mem_mb = max(512, min(max_alloc_mb + 256, 2300))
    max_workers_by_vram = max(1, vram_budget_mb // worker_mem_mb)

    workers = min(max_workers_by_vram, cpu_count, max(1, total_tiles))
    logger.info("🎮 [GPU Auto-scale] VRAM: %d MB (budget: %d MB) -> concurrency: %d workers (~%d MB/worker)",
                vram_mb, vram_budget_mb, workers, worker_mem_mb)
    return workers

# Cross-process registry of in-flight C++ subprocesses (ultra_cli) keyed by
# job_id. Backed by a multiprocessing.Manager so that worker processes
# spawned by ProcessPoolExecutor can register their children's PIDs and the
# main process can SIGTERM/SIGKILL them when cancel is requested. The
# Manager is created lazily because Manager() spawns a server process.
_active_pids_manager: multiprocessing.managers.SyncManager | None = None
_active_pids_by_job: dict | None = None  # Manager dict: job_id -> Manager list of int PIDs
_pids_registry_lock = threading.Lock()


def _ensure_pids_registry() -> dict:
    global _active_pids_manager, _active_pids_by_job
    with _pids_registry_lock:
        if _active_pids_manager is None:
            _active_pids_manager = multiprocessing.Manager()
            _active_pids_by_job = _active_pids_manager.dict()
    return _active_pids_by_job


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _terminate_job_subprocesses(job_id: str, *, grace_s: float = 3.0) -> None:
    """Send SIGTERM to every registered PID for the job, then SIGKILL to any
    that are still alive after grace_s. Safe to call when the job has no
    registered PIDs (no-op)."""
    registry = _ensure_pids_registry()
    pids = list(registry.get(job_id, []))  # snapshot
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if all(_pid_alive(pid) is False for pid in pids):
            break
        time.sleep(0.1)
    for pid in pids:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
ArtifactName = Literal[
    "job",
    "surface",
    "surface_meta",
    "runner_input",
    "coverage_signal",
    "coverage_mask",
    "coverage_meta",
    "coverage_png",
    "coverage_world",
]
ARTIFACT_FILES: dict[str, tuple[str, str]] = {
    "job": ("job.json", "application/json"),
    "surface": ("surface_i16le.bin", "application/octet-stream"),
    "surface_meta": ("surface_meta.json", "application/json"),
    "runner_input": ("runner_input.json", "application/json"),
    "coverage_signal": ("coverage.signal_i16le.bin", "application/octet-stream"),
    "coverage_mask": ("coverage.mask_u8.bin", "application/octet-stream"),
    "coverage_meta": ("coverage.meta.json", "application/json"),
    "coverage_png": ("coverage.png", "image/png"),
    "coverage_world": ("coverage.pgw", "text/plain"),
}


def job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def persist_job(job_id: str, job: dict) -> None:
    out = job_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "job.json"
    tmp = out / f"job.json.tmp.{os.getpid()}.{time.time_ns()}"
    tmp.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)


def load_job(job_id: str) -> dict | None:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        return None
    for _ in range(3):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                time.sleep(0.01)
                continue
            return json.loads(text)
        except (json.JSONDecodeError, OSError):
            time.sleep(0.01)
    return None


def artifact_urls(job_id: str, request: Request) -> dict[str, str]:
    out: dict[str, str] = {}
    artifact_dir = job_dir(job_id)
    for name, (filename, _) in ARTIFACT_FILES.items():
        if (artifact_dir / filename).exists():
            out[name] = str(
                request.url_for("get_ultra_artifact", job_id=job_id, artifact=name)
            )
    return out


def public_job(job_id: str, job: dict, request: Request) -> dict:
    return job | {"artifact_urls": artifact_urls(job_id, request)}


def set_progress(job: dict, phase: str, fraction: float) -> None:
    job["progress"] = {"phase": phase, "fraction": max(0.0, min(1.0, fraction))}


def is_cancel_requested(job_id: str) -> bool:
    """Return True if cancel was requested for this job.

    Checks the on-disk job.json first because worker processes spawned by
    ProcessPoolExecutor have a stale snapshot of the in-memory ``jobs``
    dict (from fork time) and do not see cancellations made after the
    worker was spawned. The cancel handler always ``persist_job``s before
    terminating subprocesses, so the disk is the cross-process source of
    truth. The in-memory dict is checked as a fast-path for the main
    process.
    """
    try:
        job = load_job(job_id) or jobs.get(job_id)
        return bool(job and job.get("cancel_requested"))
    except Exception:
        return False


def mark_cancelled(job_id: str, *, message: str = "Ultra job cancelled.") -> None:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        return
    job["status"] = "cancelled"
    job["message"] = message
    set_progress(job, "finalize", 1.0)
    jobs[job_id] = job
    persist_job(job_id, job)


def _run_surface_subprocess(
    job_id: str, request: dict, out_dir: Path
) -> tuple[SurfaceArtifact | None, NativeRunnerInput | None, str | None]:
    """Build the surface in a separate Python subprocess so the cancel endpoint
    can SIGTERM it during the (otherwise non-interruptible) HTTP fetches and
    numpy vectorization of the ``building_surface`` phase. Returns
    ``(artifact, runner_input, error)``; on cancellation, ``artifact`` and
    ``runner_input`` are ``None`` and ``error`` starts with ``"cancelled"``.
    """
    import subprocess
    import sys as _sys

    out_dir.mkdir(parents=True, exist_ok=True)
    payload_file = out_dir / "surface_payload.json"
    payload_file.write_text(
        json.dumps({"request": dict(request), "out_dir": str(out_dir)}),
        encoding="utf-8",
    )

    cmd = [_sys.executable, "-u", "-m", "backend.ultra.surface_worker", str(payload_file)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    registry = _ensure_pids_registry()
    job_pid_list = registry.setdefault(job_id, _active_pids_manager.list())
    job_pid_list.append(proc.pid)
    radius_km = float(request.get("radius_km", 10.0))
    # Area scales quadratically with radius: (R/10)^2. Cold downloads from public IGN servers require proportional time.
    env_timeout = os.environ.get("ULTRA_SURFACE_TIMEOUT_S")
    if env_timeout:
        timeout_s = int(env_timeout)
    else:
        timeout_s = max(900, int(600 + 450 * (radius_km / 10.0) ** 2))
    logger.info("   ↳ [Job %s] Phase 1 timeout budget: %ds (%.1f min) for %.1fkm radius",
                job_id, timeout_s, timeout_s / 60.0, radius_km)

    collected_stdout: list[str] = []
    collected_stderr: list[str] = []

    def _stream_reader(stream, storage_list, is_err=False):
        try:
            for line in iter(stream.readline, ""):
                storage_list.append(line)
                stripped = line.strip()
                if stripped:
                    if is_err and "error" in stripped.lower():
                        logger.warning("   ↳ [Job %s] [Worker] %s", job_id, stripped)
                    else:
                        logger.info("   ↳ [Job %s] %s", job_id, stripped)
        except Exception:
            pass
        finally:
            stream.close()

    import threading
    t_stdout = threading.Thread(target=_stream_reader, args=(proc.stdout, collected_stdout, False), daemon=True)
    t_stderr = threading.Thread(target=_stream_reader, args=(proc.stderr, collected_stderr, True), daemon=True)
    t_stdout.start()
    t_stderr.start()

    try:
        start_time = time.time()
        timed_out = False
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            if time.time() - start_time > timeout_s:
                timed_out = True
                proc.kill()
                proc.wait()
                break
            time.sleep(0.5)

        t_stdout.join(timeout=2.0)
        t_stderr.join(timeout=2.0)

        stdout = "".join(collected_stdout)
        stderr = "".join(collected_stderr)

        if timed_out:
            return None, None, f"surface build timed out after {timeout_s}s"

        if proc.returncode != 0:
            # If cancel was requested, the SIGTERM was what killed us; return
            # as cancelled rather than failed. The main process polls
            # is_cancel_requested (disk-backed) which returns True now.
            if is_cancel_requested(job_id):
                return None, None, "cancelled by user"
            # Surface the subprocess's traceback if it left one.
            error_trace = out_dir / "surface_error.txt"
            detail = (
                error_trace.read_text(encoding="utf-8")
                if error_trace.exists()
                else (stderr or stdout or "")[:500]
            )
            return None, None, f"surface build failed (rc={proc.returncode}): {detail}"
    finally:
        try:
            job_pid_list.remove(proc.pid)
        except ValueError:
            pass

    done_file = out_dir / "surface_done.flag"
    if not done_file.exists():
        # No marker: subprocess was killed by SIGTERM before writing it.
        return None, None, "cancelled by user"

    meta = json.loads((out_dir / "surface_meta.json").read_text(encoding="utf-8"))
    artifact = SurfaceArtifact(**meta)
    runner_input_dict = json.loads(
        (out_dir / "runner_input.json").read_text(encoding="utf-8")
    )
    runner_input = NativeRunnerInput(**runner_input_dict)
    return artifact, runner_input, None


def run_ultra_job(job_id: str, request: UltraJobRequest) -> None:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        return
    if job.get("cancel_requested"):
        mark_cancelled(job_id, message="Ultra job cancelled before execution started.")
        return
    job["status"] = "building_surface"
    job["message"] = "Building measured 2.5 m surface grid."
    set_progress(job, "terrain", 0.1)
    jobs[job_id] = job
    persist_job(job_id, job)
    logger.info("🚀 [Job %s] Starting Ultra simulation: lat=%.5f, lon=%.5f, radius=%.1fkm, res=%.1fm",
                job_id, request.lat, request.lon, request.radius_km, request.resolution_m)
    t_start = time.perf_counter()
    try:
        t_surface = time.perf_counter()
        logger.info("⏳ [Job %s] Phase 1/3: Fetching IGN surface data (DSM/DTM)...", job_id)
        artifact, runner_input, surface_error = _run_surface_subprocess(
            job_id=job_id,
            request=job["request"],
            out_dir=job_dir(job_id),
        )
        surface_dur = time.perf_counter() - t_surface
        if artifact is None:
            if surface_error and surface_error.startswith("cancelled"):
                mark_cancelled(
                    job_id,
                    message="Ultra job cancelled during surface build.",
                )
                logger.info("🛑 [Job %s] Cancelled during surface build", job_id)
                return
            job["status"] = "failed"
            set_progress(job, "finalize", 1.0)
            job["error"] = surface_error or "surface build failed"
            jobs[job_id] = job
            persist_job(job_id, job)
            logger.error("❌ [Job %s] Surface build failed: %s", job_id, job["error"])
            return
        logger.info("✅ [Job %s] Surface grid ready in %.2fs: %dx%d cells (%s MB)",
                    job_id, surface_dur, artifact.width, artifact.height,
                    round((artifact.width * artifact.height * 2) / (1024 * 1024), 2))
        job["surface"] = asdict(artifact)
        job["runner_input"] = asdict(runner_input)
        job["status"] = "running_native"
        job["message"] = "Running ITM/Longley-Rice over measured projected-grid terrain in tiles."
        set_progress(job, "compute", 0.4)
        binary, opencl_kernel = _resolve_native_binary()
        tile_max_cells = get_optimal_tile_cells(binary, opencl_kernel)
        tiles = plan_tiles(artifact.width, artifact.height, tile_max_cells)
        job["tiles"] = {
            "count": len(tiles),
            "completed": 0,
        }
        persist_job(job_id, job)
        logger.info("⏳ [Job %s] Phase 2/3: Computing ITM RF coverage across %d tiles (max %d cells/tile)...",
                    job_id, len(tiles), tile_max_cells)

        t_compute = time.perf_counter()
        native_result = _run_tiles_with_progress(
            job_id=job_id,
            artifact=artifact,
            request=job["request"],
            tiles=tiles,
        )
        compute_dur = time.perf_counter() - t_compute
        if native_result.status == "cancelled":
            mark_cancelled(job_id)
            logger.info("🛑 [Job %s] Cancelled during compute", job_id)
            return
        job["native_result"] = asdict(native_result)
        if native_result.status == "complete":
            logger.info("⏳ [Job %s] Phase 3/3: Rendering coverage overlay PNG and metadata...", job_id)
            t_render = time.perf_counter()
            png_path = write_coverage_png(
                signal_path=native_result.signal_path or "",
                mask_path=native_result.mask_path or "",
                out_path=job_dir(job_id) / "coverage.png",
                width=artifact.width,
                height=artifact.height,
                min_dbm=request.rx_sensitivity_dbm,
            )
            world_path = write_png_world_file(
                out_path=job_dir(job_id) / "coverage.pgw",
                width=artifact.width,
                height=artifact.height,
                bounds_wgs84=artifact.bounds_wgs84,
            )
            render_dur = time.perf_counter() - t_render
            total_dur = time.perf_counter() - t_start
            rate = (artifact.width * artifact.height) / max(0.01, compute_dur)
            job["coverage_png"] = {"path": png_path, "world_path": world_path}
            job["status"] = "coverage_ready"
            set_progress(job, "finalize", 1.0)
            job["message"] = (
                "Native ultra coverage is ready. Model is ITM/Longley-Rice over the measured projected 2.5 m surface grid."
            )
            logger.info("🎉 [Job %s] Done! Total: %.2fs (surface: %.2fs, compute: %.2fs @ %.0f cells/s, render: %.2fs)",
                        job_id, total_dur, surface_dur, compute_dur, rate, render_dur)
        elif native_result.status == "missing_binary":
            job["status"] = "surface_ready"
            set_progress(job, "finalize", 0.75)
            job["message"] = native_result.message
            logger.warn("⚠️ [Job %s] Missing native runner binary", job_id)
        else:
            job["status"] = "failed"
            set_progress(job, "finalize", 1.0)
            job["error"] = f"native ultra runner failed: {native_result.message}"
            logger.error("❌ [Job %s] Compute failed: %s", job_id, native_result.message)
    except Exception as exc:
        job["status"] = "failed"
        set_progress(job, "finalize", 1.0)
        job["error"] = f"surface build failed: {exc}"
        logger.error("❌ [Job %s] Error: %s", job_id, exc)
    jobs[job_id] = job
    persist_job(job_id, job)


def _run_one_tile(
    binary: str,
    artifact_dict: dict,
    tile_dict: dict,
    request: dict,
    tx_x: float,
    tx_y: float,
    out_prefix: str,
    threads: int,
    job_id: str,
    pid_registry: dict,
    opencl_kernel: str = "",
) -> None:
    """ProcessPoolExecutor worker: build the per-tile command and run it.

    Module-level (not a closure) so the executor can pickle it across
    process boundaries. All inputs are plain picklable types; the dataclass
    shapes are reconstructed from ``asdict(...)`` snapshots.

    The worker's C++ child PID is registered in ``pid_registry[job_id]``
    (a Manager list proxy) so the main process can SIGTERM/SIGKILL it on
    cancel. We use ``Popen`` + ``communicate(timeout=...)`` instead of
    ``subprocess.run`` so the cancel endpoint can interrupt the in-flight
    child even though the parent worker is still blocked.
    """
    import subprocess
    from pathlib import Path
    from .artifacts import SurfaceArtifact
    from .tiler import TileSpec
    from .runner import _build_cmd

    # Check cancel before spawning a subprocess. Otherwise a worker that
    # gets a fresh task from the executor queue (after the original 4
    # tasks were SIGTERM'd) would spawn a new ultra_cli that escapes the
    # registry entirely. The check is also done after communicate()
    # returns, but the pre-check short-circuits the spawn.
    if is_cancel_requested(job_id):
        return

    artifact = SurfaceArtifact(**artifact_dict)
    tile = TileSpec(**tile_dict)
    cmd = _build_cmd(
        Path(binary),
        artifact,
        tile,
        request,
        tx_x,
        tx_y,
        Path(out_prefix),
        threads=threads,
        opencl_kernel=opencl_kernel,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    job_pid_list = pid_registry[job_id]
    job_pid_list.append(proc.pid)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise
        if proc.returncode != 0:
            # If cancel was requested, the C++ child was SIGTERM'd by the
            # main process via _terminate_job_subprocesses. The negative
            # returncode is the signal number, not a real failure. Return
            # normally so the outer executor loop's is_cancel_requested
            # check produces the "cancelled" status instead of "failed".
            if is_cancel_requested(job_id):
                return
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=stdout, stderr=stderr,
            )
    finally:
        try:
            job_pid_list.remove(proc.pid)
        except ValueError:
            pass


def _run_tiles_with_progress(job_id, artifact, request, tiles):
    """Run the tiled native runner, persisting tile-level progress into the
    in-memory job after each completed tile so the polling client sees a
    live fraction.

    Picks one of two paths based on grid size:

    * Single-process path (``workers == 1``): one subprocess per tile, all
      cores available to the C++ runner via ``--threads cpu_count``. Used for
      small grids (under :data:`SINGLE_TILE_CELL_THRESHOLD` cells or a single
      pending tile) where the per-process startup cost would dominate.
    * Multi-process path: :class:`ProcessPoolExecutor` fans the tiles out
      across ``min(cpu_count, len(pending_tiles))`` workers, with each
      subprocess sized to ``max(1, cpu_count // workers)`` threads so the
      total threads across workers stays close to ``cpu_count`` (no
      oversubscription).
    """
    import os
    import subprocess
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor, as_completed

    binary, opencl_kernel = _resolve_native_binary()
    if not binary.exists():
        from .runner import NativeRunResult
        return NativeRunResult(
            status="missing_binary",
            model="itm_projected_grid",
            message=f"native ultra runner not found at {binary}; run engine/build_native.sh",
        )

    logger.info("⚡ [Job %s] Runner selected: %s (OpenCL Kernel: %s)",
                job_id, binary.name, opencl_kernel or "none")

    out_dir = job_dir(job_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / "coverage"
    from .geo import latlon_to_utm30
    from .runner import _build_cmd, NativeRunResult

    tx_x, tx_y = latlon_to_utm30(request["lat"], request["lon"])
    total = len(tiles)
    cpu_count = os.cpu_count() or 1
    total_cells = artifact.width * artifact.height
    avg_cells_per_tile = total_cells // max(1, total)

    pending_tiles = []
    for tile in tiles:
        sig_path = out_dir / f"coverage_x{tile.x0}_y{tile.y0}.signal_i16le.bin"
        if sig_path.exists():
            _advance_tile_progress_throttled(job_id, total, cells_per_tile=avg_cells_per_tile)
        else:
            pending_tiles.append(tile)

    is_gpu = (binary.name == "ultra_cli_opencl" and bool(opencl_kernel))
    if total_cells <= SINGLE_TILE_CELL_THRESHOLD or len(pending_tiles) <= 1:
        workers = 1
        threads_per_call = 1 if is_gpu else cpu_count
    else:
        workers = get_optimal_gpu_workers(binary, opencl_kernel, len(pending_tiles), cpu_count)
        threads_per_call = 1 if is_gpu else max(1, cpu_count // workers)

    job = jobs.get(job_id)
    if job is not None:
        tiles_meta = job.setdefault("tiles", {})
        tiles_meta["workers"] = workers
        tiles_meta["threads_per_call"] = threads_per_call
        persist_job(job_id, job)

    if is_cancel_requested(job_id):
        return NativeRunResult(
            status="cancelled",
            model="itm_projected_grid",
            message="Ultra job cancelled during tiled native execution.",
        )

    registry = _ensure_pids_registry()
    job_pid_list = registry.setdefault(job_id, _active_pids_manager.list())

    if workers == 1:
        for tile in pending_tiles:
            if is_cancel_requested(job_id):
                return NativeRunResult(
                    status="cancelled",
                    model="itm_projected_grid",
                    message="Ultra job cancelled during tiled native execution.",
                )
            cmd = _build_cmd(
                binary,
                artifact,
                tile,
                request,
                tx_x,
                tx_y,
                out_prefix,
                threads=threads_per_call,
                opencl_kernel=opencl_kernel,
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            job_pid_list.append(proc.pid)
            try:
                try:
                    stdout, stderr = proc.communicate(timeout=600)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    raise
                if proc.returncode != 0:
                    if is_cancel_requested(job_id):
                        return NativeRunResult(
                            status="cancelled",
                            model="itm_projected_grid",
                            message="Ultra job cancelled during tiled native execution.",
                        )
                    raise subprocess.CalledProcessError(
                        proc.returncode, cmd, output=stdout, stderr=stderr,
                    )
            except Exception as exc:
                return NativeRunResult(
                    status="failed",
                    model="itm_projected_grid",
                    message=f"tile ({tile.x0},{tile.y0}) failed: {exc}",
                )
            finally:
                try:
                    job_pid_list.remove(proc.pid)
                except ValueError:
                    pass
            _advance_tile_progress_throttled(job_id, total, cells_per_tile=avg_cells_per_tile)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_tile = {
                executor.submit(
                    _run_one_tile,
                    str(binary),
                    asdict(artifact),
                    asdict(tile),
                    request,
                    tx_x,
                    tx_y,
                    str(out_prefix),
                    threads_per_call,
                    job_id,
                    registry,
                    opencl_kernel,
                ): tile
                for tile in pending_tiles
            }
            for future in as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    future.result()
                except Exception as exc:
                    return NativeRunResult(
                        status="failed",
                        model="itm_projected_grid",
                        message=f"tile ({tile.x0},{tile.y0}) failed: {exc}",
                    )
                _advance_tile_progress_throttled(job_id, total, cells_per_tile=avg_cells_per_tile)
                if is_cancel_requested(job_id):
                    executor.shutdown(wait=False, cancel_futures=True)
                    return NativeRunResult(
                        status="cancelled",
                        model="itm_projected_grid",
                        message="Ultra job cancelled during tiled native execution.",
                    )

    meta = aggregate_tiles(artifact=artifact, tiles=tiles, out_dir=out_dir)
    meta["rx_sensitivity_dbm"] = request["rx_sensitivity_dbm"]
    (out_dir / "coverage.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    payload = make_native_run_result(out_prefix)
    return NativeRunResult(**payload)


def _advance_tile_progress(job_id: str, total: int) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    done = int(job.get("tiles", {}).get("completed", 0)) + 1
    job.setdefault("tiles", {})["completed"] = done
    fraction = 0.4 + 0.5 * (done / max(1, total))
    set_progress(job, "compute", fraction)
    persist_job(job_id, job)


def _advance_tile_progress_throttled(
    job_id: str,
    total: int,
    cells_per_tile: int = 0,
    *,
    min_interval_s: float = 0.25,
) -> None:
    """Like :func:`_advance_tile_progress` but coalesces disk writes.

    The in-memory ``job`` dict is updated on every call (cheap), but the
    expensive ``persist_job`` rewrite only fires when at least
    ``min_interval_s`` seconds have passed since the last persist, or when
    this is the final tile (so the on-disk state always reflects completion).
    """
    job = jobs.get(job_id)
    if not job:
        return
    done = int(job.get("tiles", {}).get("completed", 0)) + 1
    tiles_meta = job.setdefault("tiles", {})
    tiles_meta["completed"] = done
    fraction = 0.4 + 0.5 * (done / max(1, total))
    set_progress(job, "compute", fraction)
    now = time.monotonic()
    last = tiles_meta.get("last_persist_ts")
    is_last = done >= total

    first_tile_ts = tiles_meta.setdefault("first_tile_ts", now)
    elapsed = max(0.01, now - first_tile_ts)
    rate = done / elapsed
    cells_rate = (done * cells_per_tile) / elapsed if cells_per_tile > 0 else 0
    speed_str = f"~{rate:.2f} tiles/s"
    if cells_rate > 0:
        speed_str += f" | ~{int(cells_rate):,} cells/s"
    logger.info("  ↳ [Job %s] Tile %d/%d completed (%.1f%%) [%s, ~%.1fs remaining]",
                job_id, done, total, (done / total) * 100, speed_str, (total - done) / max(0.01, rate))

    if last is None or (now - last) >= min_interval_s or is_last:
        persist_job(job_id, job)
        tiles_meta["last_persist_ts"] = now


def estimate_grid(radius_km: float, resolution_m: float) -> dict[str, float | int]:
    diameter_m = radius_km * 2000
    cells_per_side = int(diameter_m / resolution_m) + 1
    cells = cells_per_side * cells_per_side
    elevation_mb = cells * 2 / (1024 * 1024)
    signal_mask_mb = cells * 2 / (1024 * 1024)
    return {
        "cells_per_side": cells_per_side,
        "cells": cells,
        "elevation_mb": round(elevation_mb, 1),
        "signal_mask_mb": round(signal_mask_mb, 1),
        "minimum_grid_mb": round(elevation_mb + signal_mask_mb, 1),
    }


class TerrainSampleRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(25.0, gt=0, le=500)
    coverage_id: Literal["mds05", "mdsn_v025", "mdsn_e025"] = "mds05"


class UltraJobRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(..., gt=0, le=30)
    frequency_mhz: float = Field(..., gt=20, le=20000)
    tx_height_m: float = Field(..., ge=0)
    rx_height_m: float = Field(..., ge=0)
    tx_power_w: float = Field(..., gt=0)
    tx_gain_dbi: float = 0.0
    rx_gain_dbi: float = 0.0
    rx_sensitivity_dbm: float = -130.0
    ground_dielectric: float = 15.0
    ground_conductivity: float = 0.005
    atmosphere_bending: float = 301.0
    radio_climate: int = Field(5, ge=1, le=7)
    polarization: int = Field(1, ge=0, le=1)
    confidence: float = Field(0.95, gt=0, le=1)
    reliability: float = Field(0.95, gt=0, le=1)
    resolution_m: Literal[2.5] = 2.5
    surface_mode: SurfaceMode = "lod_dtm_plus_buildings"


class SurfaceSampleRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(25.0, gt=0, le=500)
    resolution_m: Literal[2.5] = 2.5
    mode: SurfaceMode = "dtm_plus_buildings_2_5m"


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/terrain/sample")
def terrain_sample(request: TerrainSampleRequest) -> dict:
    min_x, min_y, max_x, max_y = meter_bbox_around(request.lat, request.lon, request.radius_m)
    try:
        grid = ign.fetch_arcgrid(
            coverage_id=request.coverage_id,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IGN DSM fetch failed: {exc}") from exc
    return {
        "coverage_id": request.coverage_id,
        "bbox_25830": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        "grid": asdict(grid) | {
            "values": None,
            "min_value": grid.min_value,
            "max_value": grid.max_value,
        },
        "note": "mdsn_* coverages are normalized heights above ground, not absolute RF terrain.",
    }


class CoverageProbeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(..., gt=0, le=10000)


@router.post("/coverage/probe")
def coverage_probe(request: CoverageProbeRequest) -> dict:
    """Return which IGN coverage layers are available for the bbox so the UI
    can pick a surface mode before submitting a job."""
    min_x, min_y, max_x, max_y = meter_bbox_around(request.lat, request.lon, request.radius_m)
    availability: dict[str, dict] = {}
    for coverage_id in ("mds05", "mdsn_v025", "mdsn_e025"):
        try:
            grid = ign.fetch_arcgrid(
                coverage_id=coverage_id,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            )
            availability[coverage_id] = {
                "available": True,
                "ncols": grid.ncols,
                "nrows": grid.nrows,
                "min_value": grid.min_value,
                "max_value": grid.max_value,
            }
        except Exception as exc:
            availability[coverage_id] = {
                "available": False,
                "error": str(exc),
            }
    mdsn_e025 = availability.get("mdsn_e025", {}).get("available", False)
    mdsn_v025 = availability.get("mdsn_v025", {}).get("available", False)
    mds05 = availability.get("mds05", {}).get("available", False)
    recommended = (
        "lod_dtm_plus_buildings"
        if mdsn_e025 or mds05
        else "dtm_only"
    )
    return {
        "bbox_25830": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        "availability": availability,
        "recommended_surface_mode": recommended,
    }


@router.post("/surface/sample")
def surface_sample(request: SurfaceSampleRequest) -> dict:
    try:
        grid = surface_builder.build(
            lat=request.lat,
            lon=request.lon,
            radius_m=request.radius_m,
            resolution_m=request.resolution_m,
            mode=request.mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IGN DSM surface build failed: {exc}") from exc
    return {
        "mode": grid.mode,
        "resolution_m": grid.resolution_m,
        "bbox_25830": {
            "min_x": grid.min_x,
            "min_y": grid.min_y,
            "max_x": grid.max_x,
            "max_y": grid.max_y,
        },
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "cells": grid.width * grid.height,
            "min_value": grid.min_value,
            "max_value": grid.max_value,
            "values": None,
        },
        "note": "2.5 m surface grid composes measured IGN DTM ground plus measured 2.5 m normalized DSM detail; no synthetic buildings are generated.",
    }


@router.post("/ultra/jobs")
def create_ultra_job(payload: UltraJobRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid4())
    estimate = estimate_grid(payload.radius_km, payload.resolution_m)
    job = {
        "status": "queued",
        "request": payload.model_dump(),
        "estimate": estimate,
        "message": "Queued for tiled projected-grid ITM execution.",
        "cancel_requested": False,
    }
    set_progress(job, "terrain", 0.02)
    jobs[job_id] = job
    persist_job(job_id, job)
    background_tasks.add_task(run_ultra_job, job_id, payload)

    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
        "estimate": estimate,
        "artifact_urls": {},
    }


@router.get("/ultra/jobs/{job_id}")
def get_ultra_job(job_id: str, request: Request) -> dict:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    jobs[job_id] = job
    return public_job(job_id, job, request)


@router.post("/ultra/jobs/{job_id}/cancel")
def cancel_ultra_job(job_id: str, request: Request) -> dict:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") in {"coverage_ready", "failed", "cancelled"}:
        jobs[job_id] = job
        return public_job(job_id, job, request)
    job["cancel_requested"] = True
    if job.get("status") == "queued":
        job["message"] = "Ultra job cancellation requested."
    jobs[job_id] = job
    persist_job(job_id, job)
    _terminate_job_subprocesses(job_id)
    return public_job(job_id, job, request)


@router.get("/ultra/jobs/{job_id}/artifacts/{artifact}")
def get_ultra_artifact(job_id: str, artifact: ArtifactName) -> FileResponse:
    job = jobs.get(job_id) or load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    filename, media_type = ARTIFACT_FILES[artifact]
    path = job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=media_type, filename=filename)


class PollingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("GET " in msg and "/ultra/jobs/" in msg and " 200" in msg)


_polling_filter = PollingLogFilter()
logging.getLogger("uvicorn.access").addFilter(_polling_filter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(_polling_filter)
    for h in access_logger.handlers:
        h.addFilter(_polling_filter)
    yield


app = FastAPI(title="Meshtastic Site Planner Ultra DSM Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=ULTRA_API_PREFIX)
