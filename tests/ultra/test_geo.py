from backend.ultra.geo import latlon_to_utm30, utm30_to_latlon


def test_utm_round_trip():
    # Madrid center.
    lat, lon = 40.41696, -3.703508
    x, y = latlon_to_utm30(lat, lon)
    assert 440000 < x < 441000
    assert 4474000 < y < 4475000
    lat2, lon2 = utm30_to_latlon(x, y)
    assert abs(lat2 - lat) < 1e-6
    assert abs(lon2 - lon) < 1e-6


def test_utm_known_points():
    # At (lat, lon) = (0, -3) we are on the equator at the central meridian of
    # UTM zone 30. The transverse Mercator projection places the origin at
    # (x=500000, y=0) for these coordinates, modulo sub-meter distortion.
    x, y = latlon_to_utm30(0.0, -3.0)
    assert abs(x - 500000) < 1
    assert abs(y) < 1
