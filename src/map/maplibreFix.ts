import maplibregl from 'maplibre-gl';

/**
 * MapLibre GL JS bug workaround:
 *
 * When an ImageSource has dimensions that are powers of two (e.g. 512, 1024,
 * 2048, 4096), MapLibre's internal `isSizePowerOfTwo()` returns true, causing
 * the raster shader to bind the texture with MIN_FILTER = LINEAR_MIPMAP_NEAREST.
 *
 * However, ImageSource never generates mipmaps (unlike tile sources), leaving the
 * texture "incomplete" under WebGL rules. In WebGL, sampling an incomplete
 * texture always returns opaque black (0, 0, 0, 1), resulting in a solid or
 * translucent black rectangle overlaid on the map.
 *
 * Patching ImageSource.prototype.prepare to ensure its texture reports
 * `isSizePowerOfTwo() === false` forces MapLibre to use LINEAR filtering without
 * mipmaps, rendering the image cleanly regardless of whether its dimensions are
 * powers of two.
 */
export function patchMapLibreImageSource(): void {
  const proto = (maplibregl.ImageSource as any)?.prototype;
  if (!proto || proto._patchedPowerOfTwo) return;

  const origPrepare = proto.prepare;
  proto.prepare = function (this: any, ...args: any[]) {
    origPrepare.apply(this, args);
    if (this.texture && typeof this.texture.isSizePowerOfTwo === 'function') {
      this.texture.isSizePowerOfTwo = () => false;
    }
  };
  proto._patchedPowerOfTwo = true;
}

patchMapLibreImageSource();
