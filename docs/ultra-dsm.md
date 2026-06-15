# Ultra-Resolution Spain DSM Backend

This branch keeps the browser/WASM planner for normal simulations and adds a
separate backend path for ultra-resolution Spain simulations.

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

- Use `mds05` as the first absolute surface-elevation source. This is measured
  DSM surface data from IGN, so buildings are real source-data elevations, not
  synthetic or simulated buildings.
- Use `mdsn_e025` only if validation confirms it is the real 2.5 m normalized
  building/surface detail layer for the same DSM product.
- Use `mdsn_v025` only if validation confirms it is the real 2.5 m normalized
  vegetation/surface detail layer and we choose to include vegetation as RF
  obstruction.
- Compose the RF obstruction surface from measured DSM products only. Do not
  invent, procedurally generate, or simulate building heights.

## Current Prototype

Run locally:

```bash
python3 -m venv .venv-ultra
. .venv-ultra/bin/activate
pip install -r backend/requirements-ultra.txt
uvicorn backend.ultra.main:app --reload
```

Probe Madrid DSM data:

```bash
curl -X POST http://127.0.0.1:8000/terrain/sample \
  -H 'content-type: application/json' \
  -d '{"lat":40.41696,"lon":-3.703508,"radius_m":25,"coverage_id":"mds05"}'
```

## Intended Backend Runner

The final ultra path should:

1. Fetch/cache DSM chunks from IGN WCS.
2. Build a local projected simulation grid at 2.5 m, avoiding full 1-degree
   pages.
3. Run the native C++ RF engine against that local grid.
4. Stream progress through an async job API.
5. Return GeoTIFF/COG or PNG overlay assets for the existing frontend map.

The current browser engine remains limited to 90 m and 30 m terrain pages.

## Resource Scale

The backend job API reports a rough grid estimate before the native runner is
wired. A 2.5 m run uses about `(radius_km * 2000 / 2.5)^2` cells before any
temporary buffers or chunk overlap. Examples:

- 1 km radius: ~0.64 million cells, small.
- 10 km radius: ~64 million cells, backend-feasible.
- 30 km radius: ~576 million cells, requires careful chunking and disk-backed
  outputs.
