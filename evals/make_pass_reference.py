"""Generate evals/pass_reference.json by brute force, independently of the code under test.

``orbitwatch.orbits.passes`` finds passes with Skyfield's event search and groups
the events. This script instead samples the satellite's elevation every second for
the whole window, finds threshold crossings by linear interpolation, and finds the
peak of each pass with a parabola through the highest sample and its neighbours.

Both use Skyfield's SGP4 propagation and the same recorded TLE, so this validates the
event finding, grouping, timing and geometry - not SGP4 itself.

    python -m evals.make_pass_reference
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
from skyfield.api import EarthSatellite, wgs84

from orbitwatch.config import resolve_site
from orbitwatch.context import FIXTURE_CLOCK, FIXTURE_TLE_FILE
from orbitwatch.orbits.celestrak import FileTleSource
from orbitwatch.orbits.passes import timescale

NORAD_ID, SITE, DAYS, MIN_EL = 25544, "lompoc", 3, 10.0
OUT = Path(__file__).parent / "pass_reference.json"


def main() -> None:
    tle = FileTleSource(FIXTURE_TLE_FILE, FIXTURE_CLOCK).get_tle(NORAD_ID).tles[0]
    site = resolve_site(SITE)
    ts = timescale()
    sat = EarthSatellite(tle.line1, tle.line2, tle.name, ts)
    observer = wgs84.latlon(site.latitude, site.longitude, elevation_m=site.elevation_m)
    t0 = ts.from_datetime(FIXTURE_CLOCK)
    seconds = np.arange(0, DAYS * 86400 + 1, dtype=float)
    times = ts.tt_jd(t0.tt + seconds / 86400.0)
    alt = (sat - observer).at(times).altaz()[0].degrees

    def at(sec: float):
        t = ts.tt_jd(t0.tt + sec / 86400.0)
        a, z, _ = (sat - observer).at(t).altaz()
        return a.degrees, z.degrees

    def crossing(i: int) -> float:  # seconds where alt crosses MIN_EL between i-1 and i
        return seconds[i - 1] + (MIN_EL - alt[i - 1]) / (alt[i] - alt[i - 1])

    above = alt >= MIN_EL
    rises = np.flatnonzero(~above[:-1] & above[1:]) + 1
    sets = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    passes = []
    for r in rises:
        later = sets[sets > r]
        if later.size == 0:
            break  # pass still in progress at the end of the window
        s = later[0]
        k = r + int(np.argmax(alt[r:s]))
        y0, y1, y2 = alt[k - 1], alt[k], alt[k + 1]
        dk = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)  # parabola vertex offset in samples
        peak_sec = seconds[k] + dk
        rise_sec, set_sec = crossing(r), crossing(s)
        passes.append(
            {
                "rise_utc": (FIXTURE_CLOCK + timedelta(seconds=rise_sec)).isoformat(),
                "culmination_utc": (FIXTURE_CLOCK + timedelta(seconds=peak_sec)).isoformat(),
                "set_utc": (FIXTURE_CLOCK + timedelta(seconds=set_sec)).isoformat(),
                "max_elevation_deg": round(float(at(peak_sec)[0]), 4),
                "rise_azimuth_deg": round(float(at(rise_sec)[1]), 3),
                "set_azimuth_deg": round(float(at(set_sec)[1]), 3),
            }
        )
    # A pass already in progress at the window start has no rise sample, so iterating over
    # rises skips it - the same convention as orbitwatch.orbits.passes.
    OUT.write_text(
        json.dumps(
            {
                "method": "brute-force 1 s elevation scan with linear threshold interpolation and "
                "parabolic peak refinement (evals/make_pass_reference.py)",
                "tle": [tle.name, tle.line1, tle.line2],
                "site": site.to_dict(),
                "window_start_utc": FIXTURE_CLOCK.isoformat(),
                "days": DAYS,
                "min_elevation_deg": MIN_EL,
                "passes": passes,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(passes)} passes to {OUT}")


if __name__ == "__main__":
    main()
