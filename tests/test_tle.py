from datetime import UTC, datetime

import pytest

from orbitwatch.orbits.tle import TleError, parse_tle_text, tle_checksum

# Real ISS element set as returned by CelesTrak on 2026-10-13.
ISS = """ISS (ZARYA)
1 25544U 98067A   26286.11204827  .00005858  00000+0  11499-3 0  9991
2 25544  51.6319  76.5194 0006724 256.4602 103.5638 15.48846742590035
"""


def test_parse_three_line_tle():
    (tle,) = parse_tle_text(ISS)
    assert tle.name == "ISS (ZARYA)"
    assert tle.norad_id == 25544
    # Day 286.11204827 of 2026 -> 13 October, 0.11204827 * 86400 s = 02:41:20.97
    assert (
        abs((tle.epoch - datetime(2026, 10, 13, 2, 41, 20, 970_000, tzinfo=UTC)).total_seconds())
        < 0.01
    )


def test_parse_two_line_tle_gets_placeholder_name():
    (tle,) = parse_tle_text("\n".join(ISS.splitlines()[1:]))
    assert tle.name == "NORAD 25544"


def test_checksums_match_last_digit():
    for line in ISS.splitlines()[1:]:
        assert tle_checksum(line) == int(line[68])


def test_corrupted_line_fails_checksum():
    bad = ISS.replace("51.6319", "51.6318")
    with pytest.raises(TleError, match="Checksum"):
        parse_tle_text(bad)


def test_truncated_record_rejected():
    with pytest.raises(TleError):
        parse_tle_text("ISS (ZARYA)\n" + ISS.splitlines()[1][:40])


def test_age_days():
    (tle,) = parse_tle_text(ISS)
    assert tle.age_days(datetime(2026, 10, 14, 2, 41, 20, 970_000, tzinfo=UTC)) == pytest.approx(
        1.0
    )
