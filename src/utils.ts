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
