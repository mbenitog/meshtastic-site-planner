import math


def latlon_to_utm30(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert WGS84/ETRS89 lon-lat to EPSG:25830-like UTM zone 30N meters.

    Spain's DSM WCS coverages use projected meter coordinates. ETRS89 and WGS84
    differ by less than a meter for this use, so this closed-form UTM transform
    avoids adding a heavyweight projection dependency to the prototype.
    """
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians(-3.0)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    aa = cos_lat * (lon - lon0)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat)
        - (35 * e2**3 / 3072) * math.sin(6 * lat)
    )

    x = k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    ) + 500000
    y = k0 * (
        m
        + n
        * tan_lat
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    return x, y


def meter_bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    x, y = latlon_to_utm30(lat, lon)
    return x - radius_m, y - radius_m, x + radius_m, y + radius_m
