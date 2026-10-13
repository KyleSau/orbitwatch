"""Predict satellite passes over a ground site with Skyfield's SGP4 propagator.

A *pass* here is the interval during which the satellite is above
``min_elevation_deg`` as seen from the site. Skyfield's ``find_events`` returns
threshold crossings (rise/set) and culminations (highest point); we group them
into complete rise -> culmination -> set passes. A pass already in progress at
the start of the window, or still in progress at the end, is omitted.

*Visible* uses the classic naked-eye rule: the satellite is in sunlight while the
observer's sky is dark (Sun below -6 degrees, i.e. past civil twilight). Checking
that needs Sun/Earth positions from a JPL ephemeris; without one we report
geometry only and leave visibility as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from skyfield.api import EarthSatellite, Timescale, load, wgs84

from orbitwatch.config import GroundSite
from orbitwatch.orbits.omm import Omm
from orbitwatch.orbits.tle import Tle

SUN_DARK_ALTITUDE_DEG = -6.0  # civil twilight
VISIBILITY_STEP_S = 10.0

_TIMESCALE: Timescale | None = None


def timescale() -> Timescale:
    """Skyfield timescale using the leap-second/UT1 tables bundled with Skyfield (no network)."""
    global _TIMESCALE
    if _TIMESCALE is None:
        _TIMESCALE = load.timescale(builtin=True)
    return _TIMESCALE


def make_satellite(elements: Tle | Omm, ts: Timescale | None = None) -> EarthSatellite:
    """A Skyfield satellite from either element-set format (both are SGP4 mean elements)."""
    ts = ts or timescale()
    if isinstance(elements, Omm):
        return EarthSatellite.from_omm(ts, elements.fields)
    return EarthSatellite(elements.line1, elements.line2, elements.name, ts)


@dataclass
class Pass:
    rise: datetime
    rise_azimuth_deg: float
    culmination: datetime
    max_elevation_deg: float
    culmination_azimuth_deg: float
    set: datetime
    set_azimuth_deg: float
    visible: bool | None = None  # None when no ephemeris was supplied
    visible_start: datetime | None = None
    visible_end: datetime | None = None
    sunlit_at_culmination: bool | None = None

    @property
    def duration_s(self) -> float:
        return (self.set - self.rise).total_seconds()

    def to_dict(self, tz: str = "UTC") -> dict:
        zone = ZoneInfo(tz)

        def stamp(dt: datetime | None) -> str | None:
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else None

        def local(dt: datetime | None) -> str | None:
            return dt.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z") if dt else None

        return {
            "rise_utc": stamp(self.rise),
            "rise_local": local(self.rise),
            "rise_azimuth_deg": round(self.rise_azimuth_deg, 1),
            "culmination_utc": stamp(self.culmination),
            "culmination_local": local(self.culmination),
            "max_elevation_deg": round(self.max_elevation_deg, 1),
            "culmination_azimuth_deg": round(self.culmination_azimuth_deg, 1),
            "set_utc": stamp(self.set),
            "set_local": local(self.set),
            "set_azimuth_deg": round(self.set_azimuth_deg, 1),
            "duration_s": round(self.duration_s),
            "visible": self.visible,
            "visible_start_utc": stamp(self.visible_start),
            "visible_end_utc": stamp(self.visible_end),
            "visible_start_local": local(self.visible_start),
            "visible_end_local": local(self.visible_end),
            "sunlit_at_culmination": self.sunlit_at_culmination,
        }


def predict_passes(
    tle: Tle | Omm,
    site: GroundSite,
    start: datetime,
    end: datetime,
    min_elevation_deg: float = 10.0,
    ephemeris=None,
) -> list[Pass]:
    """Return complete passes of ``tle`` (a TLE or OMM element set) over ``site`` between
    ``start`` and ``end`` (UTC)."""
    ts = timescale()
    sat = make_satellite(tle, ts)
    observer = wgs84.latlon(site.latitude, site.longitude, elevation_m=site.elevation_m)
    t0, t1 = ts.from_datetime(start), ts.from_datetime(end)
    times, events = sat.find_events(observer, t0, t1, altitude_degrees=min_elevation_deg)

    def altaz(t):
        alt, az, _ = (sat - observer).at(t).altaz()
        return alt.degrees, az.degrees

    passes: list[Pass] = []
    rise_t = culm_t = None
    for t, event in zip(times, events, strict=True):
        if event == 0:  # rose above the threshold
            rise_t, culm_t = t, None
        elif event == 1 and rise_t is not None:  # culmination (keep the highest one)
            if culm_t is None or altaz(t)[0] > altaz(culm_t)[0]:
                culm_t = t
        elif event == 2 and rise_t is not None and culm_t is not None:  # set
            culm_alt, culm_az = altaz(culm_t)
            p = Pass(
                rise=rise_t.utc_datetime(),
                rise_azimuth_deg=altaz(rise_t)[1],
                culmination=culm_t.utc_datetime(),
                max_elevation_deg=culm_alt,
                culmination_azimuth_deg=culm_az,
                set=t.utc_datetime(),
                set_azimuth_deg=altaz(t)[1],
            )
            if ephemeris is not None:
                _add_visibility(p, sat, observer, ephemeris, ts)
            passes.append(p)
            rise_t = culm_t = None
    return passes


def _add_visibility(p: Pass, sat, observer, eph, ts: Timescale) -> None:
    """Sample the pass and mark when the satellite is sunlit while the observer is in darkness."""
    n = max(2, int(p.duration_s // VISIBILITY_STEP_S) + 1)
    offsets = np.linspace(0.0, p.duration_s, n) / 86400.0
    t = ts.tt_jd(ts.from_datetime(p.rise).tt + offsets)
    sunlit = sat.at(t).is_sunlit(eph)
    sun_alt = (eph["earth"] + observer).at(t).observe(eph["sun"]).apparent().altaz()[0].degrees
    visible = sunlit & (sun_alt < SUN_DARK_ALTITUDE_DEG)
    p.visible = bool(visible.any())
    if p.visible:
        idx = np.flatnonzero(visible)
        p.visible_start = t[idx[0]].utc_datetime()
        p.visible_end = t[idx[-1]].utc_datetime()
    tc = ts.from_datetime(p.culmination)
    p.sunlit_at_culmination = bool(sat.at(tc).is_sunlit(eph))
