"""End-to-end EO pipeline: scene -> windowed band reads -> index -> before/after change."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbitwatch.eo.change import render_change_png, summarize_change
from orbitwatch.eo.indices import (
    INDEX_BANDS,
    INDEX_RESOLUTION_M,
    compute_index,
    dn_to_reflectance,
    scl_valid_mask,
)
from orbitwatch.eo.raster import Grid, read_aligned_bands, scene_grids_match
from orbitwatch.eo.stac import Scene, effective_offset, validate_bbox

ATTRIBUTION = "Contains modified Copernicus Sentinel data; scenes via Element 84 Earth Search."
MIN_VALID_FRACTION = 0.5


@dataclass
class IndexResult:
    scene: Scene
    values: np.ndarray
    grid: Grid
    offset_handling: dict[str, str]
    scl_masked_fraction: float


def index_for_scene(scene: Scene, bbox: list[float], index: str) -> IndexResult:
    """Read only the AOI window of the needed bands and compute ``index``."""
    if index not in INDEX_BANDS:
        raise ValueError(f"Unknown index {index!r}; choose from {sorted(INDEX_BANDS)}.")
    first, second = INDEX_BANDS[index]
    keys = (first, second, "scl")
    missing = [k for k in keys if k not in scene.assets]
    if missing:
        raise ValueError(f"Scene {scene.id} lacks assets {missing}.")
    raw, grid = read_aligned_bands(
        {k: scene.assets[k].href for k in keys}, bbox, INDEX_RESOLUTION_M[index]
    )
    reflectance, offsets = {}, {}
    for key in (first, second):
        asset = scene.assets[key]
        offset, reason = effective_offset(scene, asset)
        reflectance[key] = dn_to_reflectance(raw[key], asset.scale, offset, asset.nodata)
        offsets[key] = reason
    valid = scl_valid_mask(raw["scl"])
    values = compute_index(index, reflectance, valid)
    return IndexResult(scene, values, grid, offsets, round(1.0 - float(valid.mean()), 4))


def compare_scenes(
    before: Scene,
    after: Scene,
    bbox: list[float],
    index: str = "ndvi",
    threshold: float = 0.1,
    png_path: Path | None = None,
) -> dict:
    """Compute ``index`` in both scenes over ``bbox`` and summarise ``after - before``."""
    validate_bbox(bbox)
    if not 0 < threshold < 2:
        raise ValueError("change_threshold must be between 0 and 2 (index units).")
    warnings: list[str] = []
    if before.datetime >= after.datetime:
        warnings.append("The 'before' scene is not earlier than the 'after' scene.")

    b = index_for_scene(before, bbox, index)
    a = index_for_scene(after, bbox, index)
    if not scene_grids_match(b.grid, a.grid):
        raise ValueError(
            f"Scenes are on different pixel grids (tiles {before.mgrs_tile} vs "
            f"{after.mgrs_tile}); choose two scenes from the same MGRS tile."
        )
    stats = summarize_change(b.values, a.values, threshold)
    if stats["valid_fraction"] < MIN_VALID_FRACTION:
        warnings.append(
            f"Only {stats['valid_fraction']:.0%} of AOI pixels are clear in both scenes "
            "(clouds/shadows/nodata); treat the statistics with caution."
        )
    synthetic = before.synthetic_pixels or after.synthetic_pixels
    if synthetic:
        warnings.append("FIXTURE DATA: pixel values are synthetic, not real measurements.")

    png = None
    if png_path is not None:
        png = render_change_png(
            b.values,
            a.values,
            Path(png_path),
            index=index,
            before_label=f"{before.id} ({before.datetime:%Y-%m-%d})",
            after_label=f"{after.id} ({after.datetime:%Y-%m-%d})",
            title=f"{index.upper()} change over bbox {[round(x, 4) for x in bbox]}"
            + ("  [SYNTHETIC FIXTURE PIXELS]" if synthetic else ""),
            footnote=f"{ATTRIBUTION} Gray = masked (cloud, shadow, cirrus, snow, nodata). "
            f"{b.grid.resolution_m:.0f} m pixels, {b.grid.crs}.",
            pixel_size_m=b.grid.resolution_m,
        )

    return {
        "index": index,
        "bbox": list(bbox),
        "before_scene": {**before.summary(), "scl_masked_fraction": b.scl_masked_fraction},
        "after_scene": {**after.summary(), "scl_masked_fraction": a.scl_masked_fraction},
        "change": stats,
        "grid": {
            "crs": b.grid.crs,
            "pixel_size_m": b.grid.resolution_m,
            "shape_rows_cols": list(b.grid.shape),
        },
        "offset_handling": {"before": b.offset_handling, "after": a.offset_handling},
        "png_path": str(png) if png else None,
        "data_source": "FIXTURES (synthetic pixels)" if synthetic else "Earth Search COGs",
        "attribution": ATTRIBUTION,
        "warnings": warnings,
    }
