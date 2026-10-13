import json
from types import SimpleNamespace

import pytest

from orbitwatch.context import FIXTURE_STAC_FILE
from orbitwatch.eo.stac import (
    COLLECTION,
    EarthSearchCatalog,
    FixtureCatalog,
    effective_offset,
    scene_from_item,
    validate_bbox,
)

ITEMS = json.loads(FIXTURE_STAC_FILE.read_text(encoding="utf-8"))["features"]
AOI = [-120.53, 34.635, -120.49, 34.665]


def test_scene_from_recorded_item():
    item = next(it for it in ITEMS if it["id"] == "S2C_10SGD_20260420_0_L2A")
    scene = scene_from_item(item)
    assert scene.crs == "EPSG:32610"
    assert scene.mgrs_tile == "10SGD"
    assert scene.boa_offset_applied is True
    assert set(scene.assets) == {"red", "nir", "nir08", "swir22", "scl"}
    red, swir = scene.assets["red"], scene.assets["swir22"]
    assert (red.scale, red.declared_offset, red.nodata, red.resolution_m) == (0.0001, -0.1, 0, 10)
    assert swir.resolution_m == 20
    assert red.href.startswith("https://sentinel-cogs.s3.us-west-2.amazonaws.com/")


def test_effective_offset_respects_boa_flag():
    scene = scene_from_item(ITEMS[0])
    offset, reason = effective_offset(scene, scene.assets["red"])
    assert offset == 0.0 and "already offset-corrected" in reason

    item = json.loads(json.dumps(ITEMS[0]))
    item["properties"]["earthsearch:boa_offset_applied"] = False
    unflagged = scene_from_item(item)
    offset, reason = effective_offset(unflagged, unflagged.assets["red"])
    assert offset == -0.1 and "applied" in reason


def test_earth_search_catalog_builds_query_and_sorts():
    recorded = {}

    class FakeClient:
        def search(self, **kwargs):
            recorded.update(kwargs)
            items = [SimpleNamespace(to_dict=lambda it=it: it) for it in reversed(ITEMS[:3])]
            return SimpleNamespace(items=lambda: iter(items))

    scenes = EarthSearchCatalog(client=FakeClient()).search(AOI, "2026-04-01", "2026-04-30", 20)
    assert recorded["collections"] == [COLLECTION]
    assert recorded["datetime"] == "2026-04-01/2026-04-30"
    assert recorded["query"] == {"eo:cloud_cover": {"lt": 20}}
    assert [s.datetime for s in scenes] == sorted(s.datetime for s in scenes)


def test_fixture_catalog_filters_dates_and_cloud():
    cat = FixtureCatalog(FIXTURE_STAC_FILE)
    april = cat.search(AOI, "2026-04-01", "2026-04-30", 1.0)
    assert [s.id for s in april] == ["S2C_10SGD_20260420_0_L2A"]
    assert cat.get("S2C_10SGD_20260420_0_L2A").cloud_cover_pct == pytest.approx(0.02, abs=0.01)
    with pytest.raises(ValueError):
        cat.search(AOI, "2026-04-30", "2026-04-01", 20)


@pytest.mark.parametrize("bbox", [[1, 2, 3], [-120.4, 34.6, -120.5, 34.7], [-121, 34, -120, 35]])
def test_validate_bbox_rejects_bad_or_large(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)
