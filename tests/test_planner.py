from datetime import date

import pytest

from orbitwatch.agent.grounding import check_grounding
from orbitwatch.agent.planner import (
    OFFLINE_LABEL,
    OfflinePlanner,
    choose_scene_pair,
    date_windows,
    resolve_aoi,
    resolve_satellite,
)


@pytest.mark.parametrize(
    "question,intent",
    [
        ("When are the next visible ISS passes over Lompoc?", "passes"),
        ("How did NDVI change near Vandenberg between 2026-04 and 2026-09?", "change"),
        (
            "Burn severity over [-120.53, 34.635, -120.49, 34.665] in April 2026 vs Sept 2026",
            "change",
        ),
        ("Which Sentinel-2 scenes cover Lompoc in September 2026?", "scenes"),
        ("Show me the TLE for the ISS", "tle"),
        ("Which Starlink satellites can I see tonight?", "starlink"),
        ("When is the next Starlink train over Vandenberg?", "trains"),
        ("What's the weather in Lompoc tomorrow?", None),
    ],
)
def test_classify(question, intent):
    assert OfflinePlanner.classify(question) == intent


def test_date_windows():
    assert date_windows("between 2026-04 and September 2026") == [
        (date(2026, 4, 1), date(2026, 4, 30)),
        (date(2026, 9, 1), date(2026, 9, 30)),
    ]
    assert date_windows("2026-04-01 to 2026-04-15 vs 2026-09-01 to 2026-09-15") == [
        (date(2026, 4, 1), date(2026, 4, 15)),
        (date(2026, 9, 1), date(2026, 9, 15)),
    ]
    assert date_windows("on 2026-04-20")[0] == (date(2026, 4, 13), date(2026, 4, 27))


def test_resolve_aoi_and_satellite():
    assert resolve_aoi("over [-120.5, 34.6, -120.4, 34.7]")[1] == [-120.5, 34.6, -120.4, 34.7]
    assert resolve_aoi("fields near Lompoc")[0] == "lompoc-valley-west"
    assert resolve_aoi("over Paris") is None
    assert resolve_satellite("Is Tiangong visible?") == 48274
    assert resolve_satellite("norad 20580 passes") == 20580
    assert resolve_satellite("the space station") == 25544


def test_choose_scene_pair_prefers_clear_and_same_tile():
    before = {
        "scenes": [
            {"scene_id": "a", "cloud_cover_pct": 5.0, "mgrs_tile": "10SGD", "datetime_utc": "1"},
            {"scene_id": "b", "cloud_cover_pct": 1.0, "mgrs_tile": "11SKU", "datetime_utc": "2"},
        ]
    }
    after = {
        "scenes": [
            {"scene_id": "c", "cloud_cover_pct": 0.5, "mgrs_tile": "10SGD", "datetime_utc": "3"},
            {"scene_id": "d", "cloud_cover_pct": 0.1, "mgrs_tile": "11SKU", "datetime_utc": "4"},
        ]
    }
    b, a = choose_scene_pair(before, after)
    assert (b["scene_id"], a["scene_id"]) == ("b", "d")
    with pytest.raises(ValueError):
        choose_scene_pair({"scenes": []}, after)


def test_offline_change_answer_cites_scenes_and_is_grounded(toolbox):
    q = (
        "How did vegetation (NDVI) change over the Lompoc Valley between April 2026 "
        "and September 2026?"
    )
    result = OfflinePlanner(toolbox).ask(q)
    assert result.answer.startswith(OFFLINE_LABEL)
    assert result.tools_used == [
        "search_sentinel2_scenes",
        "search_sentinel2_scenes",
        "compare_index_between_scenes",
    ]
    compare = result.tool_calls[-1].result
    for side in ("before_scene", "after_scene"):
        assert compare[side]["scene_id"] in result.answer
        assert compare[side]["date"] in result.answer
    report = check_grounding(result.answer, [c.result for c in result.tool_calls], q)
    assert report.ok, report


def test_offline_passes_answer_is_grounded(toolbox):
    q = "When are the next visible ISS passes over Lompoc?"
    result = OfflinePlanner(toolbox).ask(q)
    (call,) = result.tool_calls
    assert call.name == "predict_passes"
    assert call.arguments["visible_only"] is True and call.arguments["site"] == "lompoc"
    assert call.result["tle_epoch_utc"] in result.answer
    assert check_grounding(result.answer, [call.result], q).ok


def test_unsupported_question_calls_no_tools(toolbox):
    result = OfflinePlanner(toolbox).ask("What's the weather in Lompoc tomorrow?")
    assert result.tool_calls == [] and "outside what the offline planner supports" in result.answer
