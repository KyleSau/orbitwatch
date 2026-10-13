"""Pass prediction on a recorded TLE and a fixed window (no network)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from skyfield.api import load_file, wgs84

from orbitwatch.config import resolve_site
from orbitwatch.context import FIXTURE_CLOCK, FIXTURE_EPHEMERIS, FIXTURE_TLE_FILE
from orbitwatch.orbits.celestrak import FileTleSource
from orbitwatch.orbits.passes import SUN_DARK_ALTITUDE_DEG, predict_passes, timescale

REFERENCE = Path(__file__).parents[1] / "evals" / "pass_reference.json"
LOMPOC = resolve_site("lompoc")
END = FIXTURE_CLOCK + timedelta(days=3)


@pytest.fixture(scope="module")
def iss():
    return FileTleSource(FIXTURE_TLE_FILE, FIXTURE_CLOCK).get_tle(25544).tles[0]


@pytest.fixture(scope="module")
def eph():
    return load_file(str(FIXTURE_EPHEMERIS))


def test_geometry_invariants(iss):
    passes = predict_passes(iss, LOMPOC, FIXTURE_CLOCK, END, 10.0)
    assert len(passes) > 5
    assert [p.rise for p in passes] == sorted(p.rise for p in passes)
    for p in passes:
        assert FIXTURE_CLOCK <= p.rise < p.culmination < p.set <= END
        assert 10.0 <= p.max_elevation_deg <= 90.0
        assert 60 < p.duration_s < 15 * 60  # an ISS pass above 10 deg lasts minutes
        assert 0 <= p.rise_azimuth_deg < 360 and 0 <= p.set_azimuth_deg < 360
        assert p.visible is None  # no ephemeris supplied -> geometry only


def test_higher_threshold_is_a_subset(iss):
    low = predict_passes(iss, LOMPOC, FIXTURE_CLOCK, END, 10.0)
    high = predict_passes(iss, LOMPOC, FIXTURE_CLOCK, END, 40.0)
    assert 0 < len(high) < len(low)
    assert all(p.max_elevation_deg >= 40 for p in high)


def test_visible_means_sunlit_satellite_and_dark_sky(iss, eph):
    from skyfield.api import EarthSatellite

    passes = predict_passes(iss, LOMPOC, FIXTURE_CLOCK, END, 10.0, eph)
    visible = [p for p in passes if p.visible]
    assert visible and len(visible) < len(passes)
    ts = timescale()
    sat = EarthSatellite(iss.line1, iss.line2, iss.name, ts)
    observer = eph["earth"] + wgs84.latlon(LOMPOC.latitude, LOMPOC.longitude, LOMPOC.elevation_m)
    for p in visible:
        assert p.rise <= p.visible_start <= p.visible_end <= p.set
        t = ts.from_datetime(p.visible_start)
        sun_alt = observer.at(t).observe(eph["sun"]).apparent().altaz()[0].degrees
        assert sun_alt < SUN_DARK_ALTITUDE_DEG
        assert sat.at(t).is_sunlit(eph)


def test_matches_bruteforce_reference(iss):
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert ref["tle"][1] == iss.line1  # reference was built from this exact TLE
    passes = predict_passes(iss, LOMPOC, FIXTURE_CLOCK, END, ref["min_elevation_deg"])
    assert len(passes) == len(ref["passes"])
    for p, r in zip(passes, ref["passes"], strict=True):
        for ours, theirs in (
            (p.rise, r["rise_utc"]),
            (p.set, r["set_utc"]),
            (p.culmination, r["culmination_utc"]),
        ):
            assert abs((ours - datetime.fromisoformat(theirs)).total_seconds()) < 2.0
        assert p.max_elevation_deg == pytest.approx(r["max_elevation_deg"], abs=0.05)
