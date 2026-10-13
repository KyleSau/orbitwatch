"""Pure-numpy spectral index math. No I/O, so it is trivial to unit test.

Sentinel-2 L2A pixels are stored as unsigned 16-bit digital numbers (DN).
Surface reflectance (the fraction of sunlight reflected, nominally 0..1) is
``DN * scale + offset``. Indices are *normalized differences* of two bands:

* NDVI = (NIR - Red) / (NIR + Red)        -> vegetation vigour (B08, B04; 10 m)
* NBR  = (NIR - SWIR2) / (NIR + SWIR2)    -> burn severity / moisture (B8A, B12; 20 m)

Both range from -1 to 1. Healthy vegetation reflects strongly in NIR and absorbs
red light, so NDVI is high; burned or dry surfaces reflect more SWIR, so NBR drops.
"""

from __future__ import annotations

import numpy as np

# Sentinel-2 Scene Classification Layer (SCL) classes we treat as unusable.
SCL_INVALID: dict[int, str] = {
    0: "no_data",
    1: "saturated_or_defective",
    3: "cloud_shadow",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}

INDEX_BANDS: dict[str, tuple[str, str]] = {
    "ndvi": ("nir", "red"),
    "nbr": ("nir08", "swir22"),
}
INDEX_RESOLUTION_M: dict[str, float] = {"ndvi": 10.0, "nbr": 20.0}


def dn_to_reflectance(
    dn: np.ndarray, scale: float, offset: float, nodata: float | None = 0
) -> np.ndarray:
    """Convert digital numbers to reflectance; nodata pixels become NaN.

    The offset is applied in DN units first (offset/scale = -1000 exactly for Sentinel-2),
    so a true zero stays exactly 0.0 instead of a float32 rounding error like -7e-9 that
    the negative-reflectance mask would then throw away.
    """
    refl = (dn.astype(np.float32) + np.float32(offset / scale)) * np.float32(scale)
    if nodata is not None:
        refl[dn == nodata] = np.nan
    return refl


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where either input is NaN, negative, or the sum is zero.

    Negative reflectance is physically meaningless (it appears for very dark pixels
    after offset correction) and would push the ratio outside [-1, 1], so we mask it.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    total = a + b
    bad = ~np.isfinite(a) | ~np.isfinite(b) | (a < 0) | (b < 0) | (total <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / total
    out[bad] = np.nan
    return out


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    """True where the SCL class is usable (not cloud, shadow, cirrus, snow or nodata)."""
    return ~np.isin(scl, list(SCL_INVALID))


def compute_index(index: str, reflectance: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    """Compute ``index`` from per-band reflectance arrays, masking invalid pixels."""
    if index not in INDEX_BANDS:
        raise ValueError(f"Unknown index {index!r}; choose from {sorted(INDEX_BANDS)}.")
    first, second = INDEX_BANDS[index]
    out = normalized_difference(reflectance[first], reflectance[second])
    out[~valid] = np.nan
    return out
