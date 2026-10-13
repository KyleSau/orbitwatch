"""Synthetic Sentinel-2-like rasters for offline tests, evals and demos.

The recorded STAC metadata in ``orbitwatch/fixtures`` is real, but we never commit
real imagery. Instead each fixture scene gets small, deterministic GeoTIFFs on the
same UTM grid conventions (10 m / 20 m, nodata 0, uint16 DN) so the *exact same*
rasterio code path runs offline. Scene ids are prefixed ``FIXTURE-`` and flagged
``orbitwatch:synthetic_pixels`` so results can never be mistaken for measurements.
"""

from __future__ import annotations

import copy
import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

FIXTURE_PREFIX = "FIXTURE-"
# Covers the named AOIs in config.py with a margin.
SYNTHETIC_EXTENT_LONLAT = (-120.54, 34.628, -120.48, 34.672)
FIELD_SIZE_M = 400.0
BANDS = {"red": 10.0, "nir": 10.0, "nir08": 20.0, "swir22": 20.0, "scl": 20.0}


def _seed(scene_id: str) -> int:
    return zlib.crc32(scene_id.encode("utf-8"))


def _grid(crs: str):
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, *SYNTHETIC_EXTENT_LONLAT)
    snap = 60.0  # multiple of every band resolution, like real Sentinel-2 tile origins
    minx, maxy = np.floor(minx / snap) * snap, np.ceil(maxy / snap) * snap
    maxx, miny = np.ceil(maxx / snap) * snap, np.floor(miny / snap) * snap
    return minx, miny, maxx, maxy


def write_synthetic_scene(scene_id: str, crs: str, out_dir: Path) -> dict[str, Path]:
    """Write one GeoTIFF per band for ``scene_id``; returns band -> path."""
    out_dir = Path(out_dir) / scene_id
    paths = {band: out_dir / f"{band}.tif" for band in BANDS}
    if all(p.exists() for p in paths.values()):
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(_seed(scene_id))
    minx, miny, maxx, maxy = _grid(crs)
    nx = int(np.ceil((maxx - minx) / FIELD_SIZE_M))
    ny = int(np.ceil((maxy - miny) / FIELD_SIZE_M))
    vegetation = rng.random((ny, nx))  # 0 = bare soil, 1 = dense canopy, per "field"
    cloud_x = rng.uniform(minx, maxx)
    cloud_y = rng.uniform(miny, maxy)

    for band, res in BANDS.items():
        cols, rows = int((maxx - minx) / res), int((maxy - miny) / res)
        xs = minx + (np.arange(cols) + 0.5) * res
        ys = maxy - (np.arange(rows) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        v = vegetation[
            ((maxy - Y) // FIELD_SIZE_M).astype(int), ((X - minx) // FIELD_SIZE_M).astype(int)
        ]
        cloud = np.hypot(X - cloud_x, Y - cloud_y) < 300.0
        if band == "scl":
            data = np.where(v > 0.3, 4, 5).astype(np.uint8)  # vegetation / not vegetated
            data[np.hypot(X - cloud_x - 400, Y - cloud_y) < 250.0] = 3  # cloud shadow
            data[cloud] = 9  # cloud, high probability
            dtype = "uint8"
        else:
            refl = {
                "red": 0.12 - 0.08 * v,
                "nir": 0.20 + 0.25 * v,
                "nir08": 0.21 + 0.24 * v,
                "swir22": 0.25 - 0.15 * v,
            }[band]
            refl = refl + rng.normal(0.0, 0.004, refl.shape)
            refl[cloud] = 0.6
            data = np.clip(np.round(refl * 10000), 1, 65535).astype(np.uint16)
            dtype = "uint16"
        profile = {
            "driver": "GTiff",
            "height": rows,
            "width": cols,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": from_origin(minx, maxy, res, res),
            "nodata": 0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(paths[band], "w", **profile) as dst:
            dst.write(data, 1)
    return paths


def make_fixture_rewriter(out_dir: Path):
    """Return a function that relabels a recorded STAC item and points it at synthetic rasters."""

    def rewrite(item: dict) -> dict:
        item = copy.deepcopy(item)
        item["id"] = FIXTURE_PREFIX + item["id"]
        item["properties"]["orbitwatch:synthetic_pixels"] = True
        crs = item["properties"].get("proj:code") or f"EPSG:{item['properties']['proj:epsg']}"
        paths = write_synthetic_scene(item["id"], crs, out_dir)
        for band, path in paths.items():
            if band in item["assets"]:
                item["assets"][band]["href"] = str(path)
        return item

    return rewrite
