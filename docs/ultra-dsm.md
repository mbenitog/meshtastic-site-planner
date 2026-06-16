# Ultra-Resolution Spain DSM Backend

This branch keeps the browser/WASM planner for normal simulations and adds a
separate backend path for ultra-resolution Spain simulations.

## CI / GHCR

The Docker image workflow at `.github/workflows/docker-image.yml` builds and
publishes two images on every push to `hires_spain`:

- `ghcr.io/mbenitog/meshtastic-site-spain:latest` — static frontend.
- `ghcr.io/mbenitog/meshtastic-site-spain-ultra:latest` — FastAPI backend
  with the native `ultra_cli` and the `splat` submodule compiled in.

Pull and run the backend standalone with:

```bash
docker pull ghcr.io/mbenitog/meshtastic-site-spain-ultra:latest
docker run --rm -p 8000:8000 ghcr.io/mbenitog/meshtastic-site-spain-ultra:latest
```

## Data Source

IGN's DSM WCS endpoint is CORS-enabled and currently responds at:

`https://wcs-mds.idee.es/mds`

Observed coverage IDs:

- `mds05`: 5 m DSM, absolute surface heights, EPSG:25830.
- `mdsn_v025`: 2.5 m normalized DSM layer, heights above ground.
- `mdsn_e025`: 2.5 m normalized DSM layer, heights above ground.

The normalized 2.5 m layers are not absolute terrain elevations. For RF
coverage they need to be combined with terrain elevation, or used as obstruction
heights over a DTM/DSM base. Do not feed normalized values directly as absolute
terrain.

Working interpretation for ultra mode:

- The active output target is a 2.5 m RF obstruction grid.
- Use the best available measured absolute ground reference from IGN DTM.
- Add measured 2.5 m normalized DSM heights from `mdsn_e025` to represent real
  buildings/surface features at the highest available resolution.
- Optionally add measured 2.5 m `mdsn_v025` vegetation heights if we decide
  vegetation should obstruct RF at this scale.
- Compose the RF obstruction surface from measured IGN products only. Do not
  invent, procedurally generate, or simulate building heights.

## Current Prototype

Run locally:

```bash
git submodule update --init splat
python3 -m venv .venv-ultra
. .venv-ultra/bin/activate
pip install -r backend/requirements-ultra.txt
bash engine/build_native.sh
uvicorn backend.ultra.main:app --reload
```

Run the backend in Docker:

```bash
docker build -f Dockerfile.ultra -t meshtastic-site-spain-ultra:local .
docker run --rm -p 18081:8000 meshtastic-site-spain-ultra:local
```

The main `Dockerfile` remains the static frontend/nginx image. `Dockerfile.ultra`
is only for the FastAPI ultra DSM backend and bundles `/usr/local/bin/ultra_cli`.

Run the frontend and backend through Compose:

```bash
docker compose up --build app ultra-backend
```

The static frontend is exposed on `http://127.0.0.1:8080`; the ultra backend is
exposed on `http://127.0.0.1:8000`. Job/cache artifacts persist in the
`ultra-cache` Docker volume.

Probe Madrid DSM data:

```bash
curl -X POST http://127.0.0.1:8000/terrain/sample \
  -H 'content-type: application/json' \
  -d '{"lat":40.41696,"lon":-3.703508,"radius_m":25,"coverage_id":"mds05"}'
```

Probe the composed 2.5 m measured surface grid:

```bash
curl -X POST http://127.0.0.1:8000/surface/sample \
  -H 'content-type: application/json' \
  -d '{"lat":40.41696,"lon":-3.703508,"radius_m":25,"resolution_m":2.5,"mode":"dtm_plus_buildings_2_5m"}'
```

Create an ultra ITM job and download output artifacts. Job creation returns
immediately; poll the job URL until `status` is `coverage_ready`.

```bash
curl -X POST http://127.0.0.1:8000/ultra/jobs \
  -H 'content-type: application/json' \
  -d '{"lat":40.41696,"lon":-3.703508,"radius_km":0.02,"frequency_mhz":869.525,"tx_height_m":2,"rx_height_m":1,"tx_power_w":0.15,"tx_gain_dbi":3,"rx_gain_dbi":3,"rx_sensitivity_dbm":-130,"resolution_m":2.5}'

curl http://127.0.0.1:8000/ultra/jobs/<job_id>
curl -o coverage.meta.json http://127.0.0.1:8000/ultra/jobs/<job_id>/artifacts/coverage_meta
curl -o coverage.signal_i16le.bin http://127.0.0.1:8000/ultra/jobs/<job_id>/artifacts/coverage_signal
curl -o coverage.mask_u8.bin http://127.0.0.1:8000/ultra/jobs/<job_id>/artifacts/coverage_mask
curl -o coverage.png http://127.0.0.1:8000/ultra/jobs/<job_id>/artifacts/coverage_png
curl -o coverage.pgw http://127.0.0.1:8000/ultra/jobs/<job_id>/artifacts/coverage_world
```

Job responses include `artifact_urls` for browser clients. CORS is currently
permissive for the prototype so the static frontend on port `8080` can call a
backend on port `8000`.

Job responses include a coarse `progress` object with `phase` values matching
the frontend progress UI: `terrain`, `compute`, and `finalize`.

## Intended Backend Runner

The final ultra path should:

1. Fetch/cache DSM chunks from IGN WCS.
2. Build a local projected simulation grid at 2.5 m, avoiding full 1-degree
   pages.
3. Materialize `surface_i16le.bin` plus `surface_meta.json` for the native
   runner. The prototype writes signed little-endian int16 meters, row-major
   north-to-south and west-to-east.
4. Run the native C++ RF engine against that local grid.
5. Stream progress through an async job API.
6. Return GeoTIFF/COG or PNG overlay assets for the existing frontend map.

The current browser engine remains limited to 90 m and 30 m terrain pages.
The current native ultra executable is `engine/build/ultra_cli`. It uses
SPLAT!'s `point_to_point_ITM` Longley-Rice implementation over measured 2.5 m
projected-grid terrain profiles.

Sparse null cells from the measured DTM reference are filled from the nearest
valid measured neighbour in the prototype to avoid artificial zero-elevation
holes. The production runner should keep an explicit validity mask and report
any fallback filling in job metadata.

## Resource Scale

The backend job API reports a rough grid estimate before the native runner is
wired. A 2.5 m run uses about `(radius_km * 2000 / 2.5)^2` cells before any
temporary buffers or chunk overlap. Examples:

- 1 km radius: ~0.64 million cells, small.
- 10 km radius: ~64 million cells, backend-feasible.
- 30 km radius: ~576 million cells, requires careful chunking and disk-backed
  outputs.

The backend runs every supported job in the background through the FastAPI
`BackgroundTasks` worker. Jobs are split into ~250 000-cell tiles which the
native ITM runner executes in small parallel batches (up to 4 workers by
default, configurable with `ULTRA_TILE_WORKERS`), persisting per-tile
signal/mask files and aggregating them into the final coverage output.
Resume after a restart is supported because tiles that already produced a file
are skipped.

A 1 km radius around Madrid (641 601 cells, 4 tiles) typically finishes in
under a minute on a single core. Radii beyond the local IGN coverage area
fail fast at the surface-build step instead of producing an empty grid.

Each prototype job writes these files under `.cache/ultra-jobs/<job_id>/`:

- `job.json`: durable API job state.
- `surface_i16le.bin`: measured RF obstruction surface.
- `surface_meta.json`: projected grid dimensions, bounds, and min/max values.
  It also includes `bounds_wgs84` for browser overlay georeferencing.
- `runner_input.json`: native runner contract containing RF parameters and
  paths to the surface artifact.
- `coverage.signal_i16le.bin`: prototype signal output in dBm x 10.
- `coverage.mask_u8.bin`: prototype coverage mask.
- `coverage.meta.json`: prototype RF output metadata.
- `coverage.png`: browser-ready transparent coverage overlay rendered by the
  backend from the ITM signal grid.
- `coverage.pgw`: world file for `coverage.png`.
