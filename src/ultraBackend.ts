import type { CoverageResult } from './engine/CoverageEngine';
import type { SplatParams } from './types';

interface UltraJobResponse {
  job_id: string;
  status: string;
  artifact_urls: Record<string, string>;
}

interface UltraJobState extends UltraJobResponse {
  request: {
    lat: number;
    lon: number;
    radius_km: number;
  };
}

interface UltraCoverageMeta {
  width: number;
  height: number;
  model: string;
}

const RADIO_CLIMATE: Record<string, number> = {
  equatorial: 1,
  continental_subtropical: 2,
  maritime_subtropical: 3,
  desert: 4,
  continental_temperate: 5,
  maritime_temperate_over_land: 6,
  maritime_temperate_over_sea: 7,
};

const POLARIZATION: Record<string, number> = {
  horizontal: 0,
  vertical: 1,
};

function joinUrl(base: string, path: string): string {
  return `${base.replace(/\/$/, '')}${path}`;
}

function boundsAround(lat: number, lon: number, radiusMeters: number): CoverageResult['bounds'] {
  const deltaLat = (radiusMeters / 6378137) * (180 / Math.PI);
  const deltaLon = deltaLat / Math.cos((lat * Math.PI) / 180);
  return {
    north: lat + deltaLat,
    south: lat - deltaLat,
    east: lon + deltaLon,
    west: lon - deltaLon,
  };
}

async function readJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`${url} failed: ${response.status}`);
  return response.json() as Promise<T>;
}

async function readArrayBuffer(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url} failed: ${response.status}`);
  return response.arrayBuffer();
}

export async function runUltraBackend(
  params: SplatParams,
  signal?: AbortSignal
): Promise<CoverageResult> {
  const baseUrl = params.simulation.ultra_backend_url || 'http://127.0.0.1:8000';
  const job = await readJson<UltraJobResponse>(joinUrl(baseUrl, '/ultra/jobs'), {
    method: 'POST',
    signal,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      lat: params.transmitter.tx_lat,
      lon: params.transmitter.tx_lon,
      radius_km: params.simulation.simulation_extent,
      frequency_mhz: params.transmitter.tx_freq,
      tx_height_m: params.transmitter.tx_height,
      rx_height_m: params.receiver.rx_height,
      tx_power_w: params.transmitter.tx_power,
      tx_gain_dbi: params.transmitter.tx_gain,
      rx_gain_dbi: params.receiver.rx_gain,
      rx_sensitivity_dbm: params.receiver.rx_sensitivity,
      ground_dielectric: params.environment.ground_dielectric,
      ground_conductivity: params.environment.ground_conductivity,
      atmosphere_bending: params.environment.atmosphere_bending,
      radio_climate: RADIO_CLIMATE[params.environment.radio_climate] ?? 5,
      polarization: POLARIZATION[params.environment.polarization] ?? 1,
      confidence: params.simulation.situation_fraction / 100,
      reliability: params.simulation.time_fraction / 100,
      resolution_m: 2.5,
      surface_mode: 'dtm_plus_buildings_2_5m',
    }),
  });

  const state = await readJson<UltraJobState>(joinUrl(baseUrl, `/ultra/jobs/${job.job_id}`), { signal });
  if (state.status !== 'coverage_ready') {
    throw new Error(`Ultra backend job ${state.status}. Direct synchronous ITM jobs are limited to about 250 m radius.`);
  }

  const metaUrl = state.artifact_urls.coverage_meta;
  const signalUrl = state.artifact_urls.coverage_signal;
  if (!metaUrl || !signalUrl) throw new Error('Ultra backend did not return coverage artifacts');

  const [meta, rawSignal] = await Promise.all([
    readJson<UltraCoverageMeta>(joinUrl(baseUrl, metaUrl), { signal }),
    readArrayBuffer(joinUrl(baseUrl, signalUrl), signal),
  ]);
  const values = new Int16Array(rawSignal);
  if (values.length !== meta.width * meta.height) {
    throw new Error(`Ultra signal size mismatch: expected ${meta.width * meta.height}, got ${values.length}`);
  }

  const dbm = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) dbm[i] = values[i] / 10;

  const bounds = boundsAround(state.request.lat, state.request.lon, state.request.radius_km * 1000);
  return {
    dbm,
    width: meta.width,
    height: meta.height,
    bounds,
    pixelDegrees: (bounds.north - bounds.south) / Math.max(1, meta.height),
    stats: {
      radials: 0,
      pages: 0,
      pagesWithData: 0,
      itmWarnings: [],
      elapsedMs: 0,
      workers: 1,
    },
  };
}
