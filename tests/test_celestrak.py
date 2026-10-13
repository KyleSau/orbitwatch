from datetime import UTC, datetime, timedelta

import pytest
import requests

from orbitwatch.orbits.celestrak import CelesTrakClient, CelesTrakError

from .test_tle import ISS


class FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(params)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 13, 18, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


def test_fetch_then_serve_from_cache(tmp_path):
    session = FakeSession([FakeResponse(ISS)])
    client = CelesTrakClient(tmp_path, session=session, clock=Clock())
    first = client.get_tle(25544)
    second = client.get_tle(25544)
    assert not first.from_cache and second.from_cache
    assert len(session.calls) == 1  # one download per update cycle
    assert session.calls[0] == {"CATNR": "25544", "FORMAT": "TLE"}
    assert second.tles[0].norad_id == 25544


def test_refetch_after_ttl(tmp_path):
    clock = Clock()
    session = FakeSession([FakeResponse(ISS), FakeResponse(ISS)])
    client = CelesTrakClient(tmp_path, session=session, clock=clock)
    client.get_tle(25544)
    clock.now += timedelta(hours=3)
    assert not client.get_tle(25544).from_cache
    assert len(session.calls) == 2


def test_stale_cache_used_when_refresh_fails(tmp_path):
    clock = Clock()
    session = FakeSession([FakeResponse(ISS), FakeResponse("Forbidden", status=403)])
    client = CelesTrakClient(tmp_path, session=session, clock=clock)
    client.get_tle(25544)
    clock.now += timedelta(hours=3)
    result = client.get_tle(25544)
    assert result.stale and result.from_cache
    assert "stale" in result.warnings[0]


def test_error_without_cache(tmp_path):
    session = FakeSession([requests.ConnectionError("offline")])
    with pytest.raises(CelesTrakError, match="Could not fetch"):
        CelesTrakClient(tmp_path, session=session).get_tle(25544)


def test_no_gp_data(tmp_path):
    session = FakeSession([FakeResponse("No GP data found")])
    with pytest.raises(CelesTrakError):
        CelesTrakClient(tmp_path, session=session).get_tle(99998)


def test_six_digit_catalog_numbers_rejected(tmp_path):
    with pytest.raises(CelesTrakError, match="OMM"):
        CelesTrakClient(tmp_path, session=FakeSession([])).get_tle(100001)


def test_unsupported_group(tmp_path):
    with pytest.raises(CelesTrakError, match="Unsupported group"):
        CelesTrakClient(tmp_path, session=FakeSession([])).get_group("active")
