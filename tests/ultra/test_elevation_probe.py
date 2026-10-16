import numpy as np
from backend.ultra.ign_dsm import ArcGrid, stitch_arcgrids
from backend.ultra.surface import _arcgrid_lookup_filled


def test_stitch_arcgrids_alignment():
    # Tile 0: covers [0, 10] in X, [10, 20] in Y, cellsize 2.5 (4x4 cells)
    # Tile 1: covers [0, 10] in X, [0, 10] in Y, cellsize 2.5 (4x4 cells)
    data0 = np.full((4, 4), 100.0, dtype=np.float32)
    data1 = np.full((4, 4), 50.0, dtype=np.float32)

    tile0 = ArcGrid(ncols=4, nrows=4, xllcorner=0.0, yllcorner=10.0, cellsize=2.5, values=data0.ravel())
    tile1 = ArcGrid(ncols=4, nrows=4, xllcorner=0.0, yllcorner=0.0, cellsize=2.5, values=data1.ravel())

    stitched = stitch_arcgrids([tile0, tile1], min_x=0.0, min_y=0.0, max_x=10.0, max_y=20.0, cell_size=2.5)
    assert stitched.ncols == 4
    assert stitched.nrows == 8

    arr = np.asarray(stitched.values).reshape(stitched.nrows, stitched.ncols)
    # Top 4 rows should be from tile0 (100.0)
    assert np.all(arr[:4, :] == 100.0)
    # Bottom 4 rows should be from tile1 (50.0)
    assert np.all(arr[4:, :] == 50.0)


def test_arcgrid_lookup_filled_top_row():
    # Grid covering [0, 10] in X, [0, 10] in Y
    # Row 0 is [7.5, 10.0], row 1 is [5.0, 7.5], row 2 is [2.5, 5.0], row 3 is [0.0, 2.5]
    vals = np.array([
        [10.0, 10.0],
        [20.0, 20.0],
    ], dtype=np.float32)
    grid = ArcGrid(ncols=2, nrows=2, xllcorner=0.0, yllcorner=0.0, cellsize=5.0, values=vals.ravel())

    xs = np.array([2.5, 7.5])
    # Northernmost point y=9.9 should hit row 0 (10.0)
    # Southernmost point y=0.1 should hit row 1 (20.0)
    ys = np.array([9.9, 0.1])
    res = _arcgrid_lookup_filled(grid, xs, ys)
    assert np.all(res[0, :] == 10.0)
    assert np.all(res[1, :] == 20.0)
