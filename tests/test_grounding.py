from orbitwatch.agent.grounding import check_grounding

TOOL = [
    {
        "tle_epoch_utc": "2026-10-13 02:41:20 UTC",
        "passes": [
            {
                "rise_utc": "2026-10-14 02:13:05 UTC",
                "rise_local": "2026-10-13 19:13:05 PDT",
                "max_elevation_deg": 45.2,
                "rise_azimuth_deg": 312.4,
            }
        ],
        "change": {"valid_fraction": 0.9322, "delta": {"mean": -0.2113}},
        "scene_id": "S2C_10SGD_20260420_0_L2A",
    }
]


def test_grounded_answer_passes():
    answer = (
        "1. Using the TLE with epoch 2026-10-13 02:41:20 UTC, the ISS rises at 02:13:05 UTC "
        "(7:13 PM PDT) at azimuth 312 deg and peaks at 45.2 deg. Scene "
        "S2C_10SGD_20260420_0_L2A: 93.2% of pixels valid, mean NDVI change -0.21."
    )
    report = check_grounding(answer, TOOL)
    assert report.ok, report
    assert report.checked >= 6


def test_invented_number_is_flagged():
    report = check_grounding("The ISS peaks at 61.5 deg.", TOOL)
    assert report.ungrounded_numbers == ["61.5"]


def test_invented_date_and_time_are_flagged():
    report = check_grounding("Next pass 2026-10-15 at 04:00 UTC.", TOOL)
    assert report.ungrounded_dates == ["2026-10-15"]
    assert report.ungrounded_times == ["04:00"]


def test_numbers_from_question_are_allowed():
    report = check_grounding(
        "Over the next 3 days the ISS peaks at 45.2 deg.",
        TOOL,
        question="ISS passes in the next 3 days?",
    )
    assert report.ok


def test_paths_and_ids_ignored():
    report = check_grounding(
        r"Saved to C:\out\ndvi_2026_99.png for scene S2C_10SGD_20260420.", TOOL
    )
    assert report.ok


def test_product_names_with_numbers_are_not_claims():
    assert check_grounding("I can search Sentinel-2 scenes, not Landsat-9.", []).ok
