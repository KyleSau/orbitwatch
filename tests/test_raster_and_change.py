import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from orbitwatch.eo.change import render_change_png, summarize_change
from orbitwatch.eo.raster import read_aligned_bands, snap_bounds


def _write(path, data, res, origin=(700000.0, 3840000.0)):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype.name,
        "crs": "EPSG:32610",
        "nodata": 0,
        "transform": from_origin(origin[0], origin[1], res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_snap_bounds_expands_to_pixel_grid():
    t = from_origin(700000.0, 3840000.0, 20.0, 20.0)
    assert snap_bounds((700005.0, 3839001.0, 700031.0, 3839975.0), t) == (
        700000.0,
        3839000.0,
        700040.0,
        3839980.0,
    )


def test_read_aligned_bands_upsamples_20m_onto_10m_grid(tmp_path):
    fine = np.arange(1, 101, dtype=np.uint16).reshape(10, 10)  # 10 m, 100 m x 100 m
    coarse = np.arange(1, 26, dtype=np.uint8).reshape(5, 5)  # 20 m, same extent
    _write(tmp_path / "fine.tif", fine, 10.0)
    _write(tmp_path / "coarse.tif", coarse, 20.0)
    with rasterio.open(tmp_path / "fine.tif") as src:
        from rasterio.warp import transform_bounds

        lonlat = transform_bounds(src.crs, "EPSG:4326", 700021.0, 3839941.0, 700059.0, 3839979.0)
    arrays, grid = read_aligned_bands(
        {"fine": str(tmp_path / "fine.tif"), "coarse": str(tmp_path / "coarse.tif")},
        list(lonlat),
        10.0,
    )
    # snapped outward to the 20 m grid -> 40 m x 40 m -> 4 x 4 pixels at 10 m
    assert grid.shape == arrays["fine"].shape == arrays["coarse"].shape
    rows, cols = grid.shape
    assert rows % 2 == 0 and cols % 2 == 0
    c = arrays["coarse"]
    assert (c[::2, ::2] == c[1::2, 1::2]).all()  # each 20 m pixel became a 2x2 block
    r0 = int(round((3840000.0 - grid.bounds[3]) / 10))
    c0 = int(round((grid.bounds[0] - 700000.0) / 10))
    assert (arrays["fine"] == fine[r0 : r0 + rows, c0 : c0 + cols]).all()


def test_misaligned_grids_are_refused(tmp_path):
    _write(tmp_path / "a.tif", np.ones((10, 10), np.uint16), 10.0)
    _write(tmp_path / "b.tif", np.ones((5, 5), np.uint16), 20.0, origin=(700005.0, 3840000.0))
    from rasterio.warp import transform_bounds

    lonlat = transform_bounds("EPSG:32610", "EPSG:4326", 700021.0, 3839941.0, 700059.0, 3839979.0)
    with pytest.raises(ValueError, match="not aligned"):
        read_aligned_bands(
            {"a": str(tmp_path / "a.tif"), "b": str(tmp_path / "b.tif")}, list(lonlat), 10.0
        )


def test_summarize_change_counts_and_masks():
    before = np.array([[0.5, 0.5], [0.5, np.nan]], dtype=np.float32)
    after = np.array([[0.2, 0.55], [0.8, 0.5]], dtype=np.float32)
    s = summarize_change(before, after, threshold=0.1)
    assert s["pixels_total"] == 4 and s["pixels_valid_both"] == 3
    assert s["valid_fraction"] == pytest.approx(0.75)
    assert s["pct_pixels_decreased"] == pytest.approx(33.33)
    assert s["pct_pixels_increased"] == pytest.approx(33.33)
    assert s["pct_pixels_stable"] == pytest.approx(33.33)
    assert s["delta"]["mean"] == pytest.approx((-0.3 + 0.05 + 0.3) / 3, abs=1e-4)


def test_summarize_change_all_masked():
    nan = np.full((2, 2), np.nan, np.float32)
    s = summarize_change(nan, nan, 0.1)
    assert s["pixels_valid_both"] == 0 and s["pct_pixels_decreased"] is None


def test_render_png(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.random((20, 30)).astype(np.float32)
    path = render_change_png(
        a, a * 0.8, tmp_path / "map.png", "ndvi", "A", "B", "title", "footnote", 10.0
    )
    assert path.exists() and path.stat().st_size > 1000
