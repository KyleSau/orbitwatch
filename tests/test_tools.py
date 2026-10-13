from typing import get_args

import pytest

from orbitwatch.orbits.celestrak import SUPPORTED_GROUPS
from orbitwatch.tools import (
    TOOL_NAMES,
    Group,
    ToolError,
    anthropic_tool_definitions,
    call_tool,
)

UNSUPPORTED_IN_STRICT = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
}
AOI = [-120.53, 34.635, -120.49, 34.665]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_tool_definitions_are_strict_and_closed():
    tools = anthropic_tool_definitions()
    assert [t["name"] for t in tools] == list(TOOL_NAMES)
    for tool in tools:
        assert tool["strict"] is True
        assert len(tool["description"]) > 80  # says what it does and when to call it
        schema = tool["input_schema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        for node in _walk(schema):
            assert not (UNSUPPORTED_IN_STRICT & node.keys()), (tool["name"], node)
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False


def test_group_literal_matches_celestrak_groups():
    assert set(get_args(Group)) == set(SUPPORTED_GROUPS)


def test_required_parameters():
    by_name = {t["name"]: t["input_schema"] for t in anthropic_tool_definitions()}
    assert by_name["get_tle"]["required"] == ["norad_id"]
    assert set(by_name["compare_index_between_scenes"]["required"]) == {
        "before_scene_id",
        "after_scene_id",
        "bbox",
    }
    assert "required" not in by_name["predict_passes"]  # every parameter has a default
    assert by_name["compare_index_between_scenes"]["properties"]["index"]["enum"] == ["ndvi", "nbr"]


@pytest.mark.parametrize(
    "name,args,match",
    [
        ("nope", {}, "Unknown tool"),
        ("get_tle", {}, "Invalid arguments"),
        ("get_tle", {"norad_id": 25544, "extra": 1}, "Invalid arguments"),
        ("predict_passes", {"days": 30}, "days must be"),
        ("predict_passes", {"site": "mars"}, "Unknown site"),
        (
            "search_sentinel2_scenes",
            {"bbox": [1, 2], "start_date": "2026-04-01", "end_date": "2026-04-30"},
            "bbox",
        ),
    ],
)
def test_call_tool_rejects_bad_input(toolbox, name, args, match):
    with pytest.raises(ToolError, match=match):
        call_tool(toolbox, name, args)


def test_custom_coordinates_are_labelled_custom():
    from orbitwatch.config import resolve_site

    site = resolve_site("lompoc", latitude=34.7, longitude=-120.5)
    assert site.name == "custom" and site.latitude == 34.7
    with pytest.raises(ValueError):
        resolve_site(latitude=34.7)


def test_list_places(toolbox):
    out = call_tool(toolbox, "list_places", {})
    assert {s["name"] for s in out["ground_sites"]} >= {"lompoc", "vandenberg"}


def test_search_and_compare_on_fixtures(toolbox):
    found = call_tool(
        toolbox,
        "search_sentinel2_scenes",
        {"bbox": AOI, "start_date": "2026-04-01", "end_date": "2026-04-30", "max_cloud_cover": 5},
    )
    assert found["data_source"] == "FIXTURES"
    assert all(s["scene_id"].startswith("FIXTURE-") for s in found["scenes"])
    out = call_tool(
        toolbox,
        "compare_index_between_scenes",
        {
            "before_scene_id": "FIXTURE-S2C_10SGD_20260420_0_L2A",
            "after_scene_id": "FIXTURE-S2B_10SGD_20260902_0_L2A",
            "bbox": AOI,
        },
    )
    assert out["data_source"].startswith("FIXTURES")
    assert any("synthetic" in w for w in out["warnings"])
    assert 0.5 < out["change"]["valid_fraction"] <= 1.0  # synthetic cloud is masked
    assert out["grid"]["pixel_size_m"] == 10.0
    assert out["png_path"].endswith(".png")


def test_nbr_uses_20m_grid(toolbox):
    out = call_tool(
        toolbox,
        "compare_index_between_scenes",
        {
            "before_scene_id": "FIXTURE-S2C_10SGD_20260420_0_L2A",
            "after_scene_id": "FIXTURE-S2B_10SGD_20260902_0_L2A",
            "bbox": AOI,
            "index": "nbr",
            "render_png": False,
        },
    )
    assert out["grid"]["pixel_size_m"] == 20.0 and out["png_path"] is None
