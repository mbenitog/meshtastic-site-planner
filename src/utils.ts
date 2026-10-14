export function cloneObject(item: any) {
  return JSON.parse(JSON.stringify(item));
}

/**
 * Default base URL for the optional ultra backend.
 *
 * Always derived from the page the user is currently on so the same frontend
 * build works behind any reverse proxy: when the user opens
 * `https://site.example/foo`, this returns `https://site.example/ultra-api`,
 * which the proxy can route to the FastAPI service.
 *
 * In dev with two separate origins (frontend on :8080, backend on :8000),
 * callers should overwrite this value (the UI exposes a text input for that).
 */
export function defaultUltraBackendUrl(prefix: string = '/ultra-api'): string {
  if (typeof window === 'undefined') return prefix;
  const origin = window.location?.origin;
  if (!origin) return prefix;
  const cleanedPrefix = prefix.startsWith('/') ? prefix : `/${prefix}`;
  return `${origin}${cleanedPrefix}`;
}

/**
 * Generate a UUID v4 safely across both secure and insecure contexts
 * (e.g. HTTP over LAN IP where crypto.randomUUID is not defined by browsers).
 */
export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
