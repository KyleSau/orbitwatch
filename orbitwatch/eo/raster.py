"""Windowed reads from Cloud-Optimized GeoTIFFs (COGs).

A COG stores pixels in internal tiles with an index at the start of the file, so
GDAL can fetch only the byte ranges covering our small area of interest with
HTTP range requests instead of downloading a ~200 MB band file.

All bands of one Sentinel-2 tile share a UTM grid, but at 10 m and 20 m pixel
sizes. We snap the AOI outward to the 20 m grid so every band's window covers
exactly the same ground, then resample 20 m bands onto the target grid with
nearest-neighbour (each 20 m pixel becomes a 2x2 block at 10 m - no new values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

# GDAL settings that keep remote COG reads to a handful of range requests.
GDAL_COG_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # don't list the S3 "directory"
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "AWS_NO_SIGN_REQUEST": "YES",
}


@dataclass(frozen=True)
class Grid:
    crs: str
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy in CRS units (metres)
    resolution_m: float
    shape: tuple[int, int]  # rows, cols


def snap_bounds(bounds, transform) -> tuple[float, float, float, float]:
    """Expand ``bounds`` outward to whole pixels of ``transform``'s grid."""
    minx, miny, maxx, maxy = bounds
    x0, y0, res = transform.c, transform.f, transform.a
    return (
        x0 + math.floor((minx - x0) / res) * res,
        y0 - math.ceil((y0 - miny) / res) * res,
        x0 + math.ceil((maxx - x0) / res) * res,
        y0 - math.floor((y0 - maxy) / res) * res,
    )


def read_aligned_bands(
    hrefs: dict[str, str], bbox_lonlat: list[float], resolution_m: float
) -> tuple[dict[str, np.ndarray], Grid]:
    """Read the AOI window from each band onto one shared grid; returns raw DN arrays."""
    with rasterio.Env(**GDAL_COG_ENV):
        sources = {key: rasterio.open(href) for key, href in hrefs.items()}
        try:
            crs_set = {src.crs.to_string() for src in sources.values()}
            if len(crs_set) != 1:
                raise ValueError(f"Bands are in different CRSs: {crs_set}")
            coarsest = max(sources.values(), key=lambda s: s.transform.a)
            crs = coarsest.crs
            utm_bounds = transform_bounds("EPSG:4326", crs, *bbox_lonlat)
            bounds = snap_bounds(utm_bounds, coarsest.transform)
            shape = (
                round((bounds[3] - bounds[1]) / resolution_m),
                round((bounds[2] - bounds[0]) / resolution_m),
            )
            arrays: dict[str, np.ndarray] = {}
            for key, src in sources.items():
                window = from_bounds(*bounds, transform=src.transform)
                offsets = (window.col_off, window.row_off)
                if any(abs(off - round(off)) > 1e-6 for off in offsets):
                    raise ValueError(f"Band {key} grid is not aligned with the snapped AOI.")
                arrays[key] = src.read(
                    1,
                    window=window,
                    out_shape=shape,
                    boundless=True,
                    fill_value=src.nodata if src.nodata is not None else 0,
                    resampling=Resampling.nearest,
                )
        finally:
            for src in sources.values():
                src.close()
    return arrays, Grid(crs.to_string(), bounds, resolution_m, shape)


def scene_grids_match(a: Grid, b: Grid) -> bool:
    return a.crs == b.crs and a.shape == b.shape and np.allclose(a.bounds, b.bounds)
