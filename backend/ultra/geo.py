from pyproj import Transformer


_WGS84_TO_UTM30 = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
_UTM30_TO_WGS84 = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)


def latlon_to_utm30(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert WGS84 lon-lat to EPSG:25830 meters.

    Use PROJ/pyproj instead of the earlier closed-form approximation so the
    overlay georeferencing is accurate across the full zone, including eastern
    Spain where the approximation drifted enough to visibly misalign the map.
    """
    x, y = _WGS84_TO_UTM30.transform(lon_deg, lat_deg)
    return float(x), float(y)


def utm30_to_latlon(x: float, y: float) -> tuple[float, float]:
    """Convert EPSG:25830 meters to WGS84 lat-lon."""
    lon, lat = _UTM30_TO_WGS84.transform(x, y)
    return float(lat), float(lon)


def meter_bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    x, y = latlon_to_utm30(lat, lon)
    return x - radius_m, y - radius_m, x + radius_m, y + radius_m
