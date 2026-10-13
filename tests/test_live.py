"""Tests against real services. Skipped by default; run with `pytest --live`.

They use the normal on-disk cache so repeated runs respect CelesTrak's
one-download-per-update-cycle policy.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orbitwatch.context import DataContext
from orbitwatch.tools import Toolbox, call_tool

pytestmark = pytest.mark.live
AOI = [-120.53, 34.635, -120.49, 34.665]


@pytest.fixture
def live_toolbox(tmp_path):
    cache = Path(os.environ.get("ORBITWATCH_LIVE_CACHE", Path.home() / ".cache" / "orbitwatch"))
    return Toolbox(DataContext.live(cache_dir=cache, output_dir=tmp_path))


def test_celestrak_iss_tle_is_fresh(live_toolbox):
    out = call_tool(live_toolbox, "get_tle", {"norad_id": 25544})
    assert out["name"].startswith("ISS")
    assert abs(out["age_days"]) < 7


def test_live_passes(live_toolbox):
    out = call_tool(live_toolbox, "predict_passes", {"site": "lompoc", "days": 2})
    assert out["passes_above_threshold"] > 0


def test_earth_search_finds_known_scene(live_toolbox):
    out = call_tool(
        live_toolbox,
        "search_sentinel2_scenes",
        {"bbox": AOI, "start_date": "2026-04-01", "end_date": "2026-04-30", "max_cloud_cover": 5},
    )
    assert "S2C_10SGD_20260420_0_L2A" in [s["scene_id"] for s in out["scenes"]]


def test_live_ndvi_compare(live_toolbox):
    out = call_tool(
        live_toolbox,
        "compare_index_between_scenes",
        {
            "before_scene_id": "S2C_10SGD_20260420_0_L2A",
            "after_scene_id": "S2B_10SGD_20260902_0_L2A",
            "bbox": AOI,
            "render_png": False,
        },
    )
    assert out["change"]["valid_fraction"] > 0.5
    assert -1 <= out["change"]["before"]["mean"] <= 1


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_claude_agent_calls_pass_tool(tmp_path):
    from orbitwatch.agent.claude_agent import ClaudeAgent

    toolbox = Toolbox(DataContext.fixtures(tmp_path))
    result = ClaudeAgent(toolbox).ask("When are the next visible ISS passes over Lompoc?")
    assert "predict_passes" in result.tools_used
    assert datetime.now(UTC).year >= 2026
