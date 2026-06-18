import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import { runUltraBackend } from '../src/ultraBackend';
import type { SplatParams } from '../src/types';

const PARAMS: SplatParams = {
  transmitter: {
    name: 'Test',
    tx_lat: 40.41696,
    tx_lon: -3.703508,
    tx_power: 0.15,
    tx_freq: 869.525,
    tx_height: 2,
    tx_gain: 3,
  },
  receiver: { rx_sensitivity: -130, rx_height: 1, rx_gain: 3, rx_loss: 2 },
  environment: {
    radio_climate: 'continental_temperate',
    polarization: 'vertical',
    clutter_height: 0.5,
    ground_dielectric: 15,
    ground_conductivity: 0.005,
    atmosphere_bending: 301,
  },
  simulation: {
    situation_fraction: 95,
    time_fraction: 95,
    simulation_extent: 0.05,
    high_resolution: true,
    ultra_backend: true,
    ultra_backend_url: 'http://ultra.test',
  },
  display: { color_scale: 'plasma', min_dbm: -130, max_dbm: -80, overlay_transparency: 50 },
};

interface MockResponseInit {
  status?: number;
  ok?: boolean;
  body?: string | ArrayBuffer;
  contentType?: string;
}

function jsonResponse(body: unknown, init: MockResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'content-type': init.contentType ?? 'application/json' },
  });
}

describe('runUltraBackend', () => {
  const originalFetch = globalThis.fetch;
  let lastOptions: RequestInit | undefined;
  let lastUrl = '';

  beforeEach(() => {
    lastOptions = undefined;
    lastUrl = '';
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('encodes ITM parameters from the UI request', async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    globalThis.fetch = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      lastUrl = String(url);
      lastOptions = init;
      if (lastUrl.endsWith('/ultra/jobs')) {
        const body = JSON.parse(String(init?.body));
        calls.push({ url: lastUrl, body });
        return jsonResponse({ job_id: 'abc', status: 'queued', artifact_urls: {}, progress: { phase: 'terrain', fraction: 0.02 } });
      }
      if (lastUrl.endsWith('/ultra/jobs/abc')) {
        return jsonResponse({
          job_id: 'abc',
          status: 'coverage_ready',
          artifact_urls: {
            coverage_meta: 'http://ultra.test/ultra-api/ultra/jobs/abc/artifacts/coverage_meta',
            surface_meta: 'http://ultra.test/ultra-api/ultra/jobs/abc/artifacts/surface_meta',
            coverage_signal: 'http://ultra.test/ultra-api/ultra/jobs/abc/artifacts/coverage_signal',
          },
        });
      }
      if (lastUrl.endsWith('coverage_meta')) {
        return jsonResponse({ width: 4, height: 4, model: 'itm_projected_grid' });
      }
      if (lastUrl.endsWith('surface_meta')) {
        return jsonResponse({
          bounds_wgs84: {
            north: 40.42, south: 40.41, east: -3.69, west: -3.71,
          },
        });
      }
      if (lastUrl.endsWith('coverage_signal')) {
        return new Response(new Int16Array(16).buffer, {
          headers: { 'content-type': 'application/octet-stream' },
        });
      }
      throw new Error(`unexpected fetch ${lastUrl}`);
    }) as unknown as typeof fetch;

    const result = await runUltraBackend(PARAMS);
    expect(result.width).toBe(4);
    expect(result.height).toBe(4);
    expect(result.dbm.length).toBe(16);
    expect(result.bounds.north).toBeCloseTo(40.42, 5);
    expect(calls[0].body.ground_dielectric).toBe(15);
    expect(calls[0].body.radio_climate).toBe(5); // continental_temperate
    expect(calls[0].body.polarization).toBe(1); // vertical
    expect(calls[0].body.confidence).toBeCloseTo(0.95, 6);
    expect(result.artifacts.coverageMetaUrl).toBe('http://ultra.test/ultra-api/ultra/jobs/abc/artifacts/coverage_meta');
  });

  it('throws when the backend job fails', async () => {
    globalThis.fetch = vi.fn(async (url: RequestInfo | URL) => {
      lastUrl = String(url);
      if (lastUrl.endsWith('/ultra/jobs')) {
        return jsonResponse({ job_id: 'xyz', status: 'queued', artifact_urls: {}, progress: { phase: 'terrain', fraction: 0.0 } });
      }
      if (lastUrl.endsWith('/ultra/jobs/xyz')) {
        return jsonResponse({
          job_id: 'xyz',
          status: 'failed',
          artifact_urls: {},
        });
      }
      throw new Error(`unexpected fetch ${lastUrl}`);
    }) as unknown as typeof fetch;
    await expect(runUltraBackend(PARAMS)).rejects.toThrow(/failed/);
  });
});
