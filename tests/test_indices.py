import numpy as np
import pytest

from orbitwatch.eo.indices import (
    compute_index,
    dn_to_reflectance,
    normalized_difference,
    scl_valid_mask,
)


def test_dn_to_reflectance_scale_offset_and_nodata():
    dn = np.array([0, 1000, 5000], dtype=np.uint16)
    refl = dn_to_reflectance(dn, scale=0.0001, offset=-0.1, nodata=0)
    assert np.isnan(refl[0])
    assert refl[1] == 0.0  # exactly zero, so it is not mistaken for negative reflectance
    assert refl[2] == pytest.approx(0.4)


def test_dn_to_reflectance_without_offset():
    refl = dn_to_reflectance(np.array([2500], dtype=np.uint16), 0.0001, 0.0, 0)
    assert refl[0] == pytest.approx(0.25)


def test_ndvi_known_values():
    nir = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    red = np.array([0.1, 0.3, 0.6], dtype=np.float32)
    ndvi = normalized_difference(nir, red)
    assert ndvi == pytest.approx([0.4 / 0.6, 0.0, -0.4 / 0.8], abs=1e-6)


def test_normalized_difference_masks_bad_inputs():
    a = np.array([0.0, np.nan, -0.01, 0.2], dtype=np.float32)
    b = np.array([0.0, 0.1, 0.05, 0.1], dtype=np.float32)
    out = normalized_difference(a, b)
    assert np.isnan(out[:3]).all()  # zero sum, NaN input, negative reflectance
    assert out[3] == pytest.approx(1 / 3)
    assert np.nanmax(np.abs(out)) <= 1.0


def test_scl_mask_drops_clouds_shadows_and_nodata():
    scl = np.array([0, 3, 4, 5, 6, 8, 9, 10, 11])
    assert scl_valid_mask(scl).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_compute_index_applies_mask_and_band_choice():
    refl = {
        "nir08": np.full((2, 2), 0.4, np.float32),
        "swir22": np.full((2, 2), 0.2, np.float32),
    }
    valid = np.array([[True, False], [True, True]])
    nbr = compute_index("nbr", refl, valid)
    assert np.isnan(nbr[0, 1])
    assert nbr[0, 0] == pytest.approx(0.2 / 0.6)


def test_unknown_index():
    with pytest.raises(ValueError):
        compute_index("evi", {}, np.ones((1, 1), bool))
