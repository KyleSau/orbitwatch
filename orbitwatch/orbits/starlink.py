"""Starlink: naked-eye-visible passes and recently launched "trains", computed cheaply.

The CelesTrak ``starlink`` group holds about 11,000 objects. Running the exact pass
finder (``passes.predict_passes``) on every one for a day would take minutes, so both
features use a two-stage design:

1. **Cheap vectorised screen.** Propagate every object with SGP4 in one ``SatrecArray``
   call, but only at coarse time steps (60 s) when the site's sky is dark (Sun below
   -6 deg, one Skyfield call for the whole window). Keep (object, time) samples that are
   near or above the elevation threshold and outside a cylindrical Earth shadow. This is
   plain numpy over ``objects x dark samples`` and takes a few seconds.
2. **Exact refinement.** Run the normal pass finder (event search, sunlit/twilight check
   sampled every 10 s against the JPL ephemeris) only on the short windows around the
   candidates. The screen is deliberately loose (5 deg elevation margin, 50 km shadow
   margin), so the exact stage makes the final call. If there are more candidates than
   ``MAX_REFINED_CANDIDATES``, the closest are checked first: with no magnitude model,
   distance at culmination is the honest proxy for "probably brighter".

**Trains.** A freshly deployed Starlink batch flies as a line of satellites ("train")
in a low insertion orbit before raising to its shell, which is what people see after a
launch from Vandenberg. Train detection uses only the elements:

* group objects by launch (international designator before the piece letter);
* keep objects below ``TRAIN_MAX_ALTITUDE_KM`` (operational shells are at ~480-570 km);
* propagate the launch's low objects to one instant and link any two within
  ``TRAIN_LINK_KM`` of each other (single-linkage clustering); a connected group of at
  least ``TRAIN_MIN_OBJECTS`` is a train;
* predict passes for the train's lead and tail satellites over the site.

These thresholds are judgment calls documented here, not published definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sgp4 import omm as sgp4_omm
from sgp4.api import Satrec, SatrecArray, jday
from skyfield.api import wgs84
from skyfield.framelib import itrs
from skyfield.sgp4lib import theta_GMST1982

from orbitwatch.config import GroundSite
from orbitwatch.orbits.omm import EARTH_EQUATORIAL_RADIUS_KM, Omm
from orbitwatch.orbits.passes import (
    SUN_DARK_ALTITUDE_DEG,
    Pass,
    make_satellite,
    predict_passes,
    timescale,
)

SCREEN_STEP_S = 60.0
SCREEN_ELEVATION_MARGIN_DEG = 5.0  # coarse samples can miss the peak; be generous
SCREEN_SHADOW_MARGIN_KM = 50.0  # cylindrical shadow is approximate; be generous
REFINE_HALF_WINDOW = timedelta(minutes=15)  # LEO passes above 10 deg last < 15 min
MAX_REFINED_CANDIDATES = 600  # bounds worst-case compute; extra candidates are counted
STALE_ELEMENTS_DAYS = 14.0  # skip objects whose elements are this old (likely decayed)
SATRECS_PER_CHUNK = 1500  # bounds memory: chunk x dark samples x 3 float64

TRAIN_MAX_ALTITUDE_KM = 450.0
TRAIN_LINK_KM = 500.0
TRAIN_MIN_OBJECTS = 5


def _satrec(o: Omm) -> Satrec:
    sat = Satrec()
    sgp4_omm.initialize(sat, o.fields)
    return sat


def _times(start: datetime, offsets_s: np.ndarray):
    """Skyfield times plus the UTC Julian date split that sgp4 expects for the same instants."""
    ts = timescale()
    t = ts.utc(
        start.year, start.month, start.day, start.hour, start.minute, start.second + offsets_s
    )
    jd0, fr0 = jday(start.year, start.month, start.day, start.hour, start.minute, start.second)
    return t, np.full(offsets_s.shape, jd0), fr0 + offsets_s / 86400.0


def _teme_to_earth_fixed(r_teme: np.ndarray, t) -> np.ndarray:
    """Rotate TEME positions (..., T, 3) to Earth-fixed by GMST (polar motion ignored,
    a few metres: irrelevant for a screen)."""
    theta, _ = theta_GMST1982(t.whole, t.ut1_fraction)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_teme[..., 0], r_teme[..., 1], r_teme[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


@dataclass
class ScreenStats:
    objects_in_catalog: int = 0
    objects_screened: int = 0
    objects_skipped_stale: int = 0
    dark_samples: int = 0
    candidates_found: int = 0
    candidates_refined: int = 0


def screen_visible_candidates(
    records: list[Omm],
    site: GroundSite,
    start: datetime,
    end: datetime,
    min_elevation_deg: float,
    ephemeris,
    stats: ScreenStats,
) -> list[tuple[int, datetime, float]]:
    """Stage 1: (record index, time near the pass peak, coarse range in km) candidates."""
    stats.objects_in_catalog = len(records)
    keep = [i for i, o in enumerate(records) if abs(o.age_days(start)) <= STALE_ELEMENTS_DAYS]
    stats.objects_skipped_stale = len(records) - len(keep)
    stats.objects_screened = len(keep)

    offsets = np.arange(0.0, (end - start).total_seconds() + 1, SCREEN_STEP_S)
    t_all, _, _ = _times(start, offsets)
    observer = wgs84.latlon(site.latitude, site.longitude, elevation_m=site.elevation_m)
    earth, sun = ephemeris["earth"], ephemeris["sun"]
    sun_alt = (earth + observer).at(t_all).observe(sun).apparent().altaz()[0].degrees
    dark = sun_alt < SUN_DARK_ALTITUDE_DEG
    stats.dark_samples = int(dark.sum())
    if not keep or not dark.any():
        return []
    offsets = offsets[dark]
    t, jd, fr = _times(start, offsets)

    obs_xyz = observer.itrs_xyz.km
    lat, lon = np.radians(site.latitude), np.radians(site.longitude)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    sun_xyz = (sun - earth).at(t).frame_xyz(itrs).km.T  # (T, 3), Earth-fixed
    sun_hat = sun_xyz / np.linalg.norm(sun_xyz, axis=1, keepdims=True)
    threshold = min_elevation_deg - SCREEN_ELEVATION_MARGIN_DEG

    candidates: list[tuple[int, datetime, float]] = []
    for c0 in range(0, len(keep), SATRECS_PER_CHUNK):
        idx = keep[c0 : c0 + SATRECS_PER_CHUNK]
        err, r, _ = SatrecArray([_satrec(records[i]) for i in idx]).sgp4(jd, fr)
        pos = _teme_to_earth_fixed(r, t)  # (n, T, 3)
        d = pos - obs_xyz
        rng = np.linalg.norm(d, axis=2)
        el = np.degrees(np.arcsin(np.einsum("ntk,k->nt", d, up) / rng))
        along = np.einsum("ntk,tk->nt", pos, sun_hat)
        perp = np.linalg.norm(pos - along[..., None] * sun_hat, axis=2)
        sunlit = (along > 0) | (perp > EARTH_EQUATORIAL_RADIUS_KM - SCREEN_SHADOW_MARGIN_KM)
        hit = (el >= threshold) & sunlit & (err == 0)
        for row in np.flatnonzero(hit.any(axis=1)):
            cols = np.flatnonzero(hit[row])
            # Split the hits into separate passes wherever there is a gap of > 10 samples.
            for run in np.split(cols, np.flatnonzero(np.diff(cols) > 10) + 1):
                peak = run[np.argmax(el[row, run])]
                when = start + timedelta(seconds=float(offsets[peak]))
                candidates.append((idx[row], when, float(rng[row, peak])))
    stats.candidates_found = len(candidates)
    return candidates


def visible_passes(
    records: list[Omm],
    site: GroundSite,
    start: datetime,
    end: datetime,
    min_elevation_deg: float,
    ephemeris,
) -> tuple[list[tuple[Omm, Pass, float]], ScreenStats]:
    """Naked-eye-visible passes of any object in ``records`` (screen, then exact check),
    as (elements, pass, range at culmination in km), in time order."""
    stats = ScreenStats()
    candidates = screen_visible_candidates(
        records, site, start, end, min_elevation_deg, ephemeris, stats
    )
    candidates.sort(key=lambda c: c[2])  # closest first
    candidates = candidates[:MAX_REFINED_CANDIDATES]
    stats.candidates_refined = len(candidates)

    ts = timescale()
    observer = wgs84.latlon(site.latitude, site.longitude, elevation_m=site.elevation_m)
    found: dict[tuple[int, datetime], tuple[Omm, Pass, float]] = {}
    for i, when, _ in candidates:
        lo, hi = max(start, when - REFINE_HALF_WINDOW), min(end, when + REFINE_HALF_WINDOW)
        for p in predict_passes(records[i], site, lo, hi, min_elevation_deg, ephemeris):
            if p.visible:
                sat = make_satellite(records[i], ts)
                rng = (sat - observer).at(ts.from_datetime(p.culmination)).distance().km
                key = (records[i].norad_id, p.culmination.replace(microsecond=0))
                found[key] = (records[i], p, float(rng))
    return sorted(found.values(), key=lambda opr: opr[1].rise), stats


# -- trains -----------------------------------------------------------------------------


@dataclass
class Train:
    launch: str
    members: list[Omm]  # ordered lead first (along the direction of motion)
    launch_low_objects: int
    along_track_span_km: float
    speed_km_s: float
    passes: list[dict] = field(default_factory=list)

    @property
    def lead(self) -> Omm:
        return self.members[0]

    @property
    def tail(self) -> Omm:
        return self.members[-1]

    @property
    def passage_estimate_s(self) -> float:
        """Rough time for the whole line to pass one point: span / orbital speed."""
        return self.along_track_span_km / self.speed_km_s


@dataclass
class LowLaunch:
    """A launch with objects still below the train altitude, clustered or not."""

    launch: str
    low_objects: int
    median_altitude_km: float
    largest_cluster: int
    median_neighbour_km: float


def _components(dist: np.ndarray, link_km: float) -> list[list[int]]:
    """Connected components of the graph 'distance <= link_km' (single linkage)."""
    n, seen, comps = len(dist), set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.flatnonzero(dist[k] <= link_km):
                if int(j) not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        comps.append(comp)
    return comps


def find_trains(
    records: list[Omm],
    at: datetime,
    max_altitude_km: float = TRAIN_MAX_ALTITUDE_KM,
    link_km: float = TRAIN_LINK_KM,
    min_objects: int = TRAIN_MIN_OBJECTS,
) -> tuple[list[Train], list[LowLaunch]]:
    """Detect trains at instant ``at``. Cheap: only low objects are propagated, once."""
    by_launch: dict[str, list[Omm]] = {}
    for o in records:
        if o.mean_altitude_km < max_altitude_km and abs(o.age_days(at)) <= STALE_ELEMENTS_DAYS:
            by_launch.setdefault(o.launch, []).append(o)

    _, jd, fr = _times(at, np.zeros(1))
    trains, low = [], []
    for launch, objs in sorted(by_launch.items()):
        if len(objs) < min_objects:
            continue
        err, r, v = SatrecArray([_satrec(o) for o in objs]).sgp4(jd, fr)
        ok = err[:, 0] == 0
        objs = [o for o, good in zip(objs, ok, strict=True) if good]
        pos, vel = r[ok, 0, :], v[ok, 0, :]
        if len(objs) < min_objects:
            continue
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        nearest = np.where(np.eye(len(objs), dtype=bool), np.inf, dist).min(axis=1)
        comp = max(_components(dist, link_km), key=len)
        low.append(
            LowLaunch(
                launch=launch,
                low_objects=len(objs),
                median_altitude_km=float(np.median([o.mean_altitude_km for o in objs])),
                largest_cluster=len(comp),
                median_neighbour_km=float(np.median(nearest)),
            )
        )
        if len(comp) < min_objects:
            continue
        p, v = pos[comp], vel[comp]
        direction = v.mean(axis=0) / np.linalg.norm(v.mean(axis=0))
        along = (p - p.mean(axis=0)) @ direction
        order = np.argsort(-along)  # largest 'along' = furthest ahead = lead
        trains.append(
            Train(
                launch=launch,
                members=[objs[comp[k]] for k in order],
                launch_low_objects=len(objs),
                along_track_span_km=float(along.max() - along.min()),
                speed_km_s=float(np.linalg.norm(v, axis=1).mean()),
            )
        )
    return trains, low


def train_passes(
    train: Train,
    site: GroundSite,
    start: datetime,
    end: datetime,
    min_elevation_deg: float,
    ephemeris,
) -> list[tuple[Pass, Pass | None]]:
    """Passes of the train's lead satellite, each paired with the tail satellite's pass
    that follows it (the tail culminates within the train's passage time + 10 min)."""
    lead = predict_passes(train.lead, site, start, end, min_elevation_deg, ephemeris)
    tail = predict_passes(
        train.tail, site, start, end + timedelta(hours=1), min_elevation_deg, ephemeris
    )
    slack = timedelta(seconds=train.passage_estimate_s + 600)
    pairs = []
    for p in lead:
        match = [
            q
            for q in tail
            if p.culmination - timedelta(minutes=2) <= q.culmination <= p.culmination + slack
        ]
        pairs.append((p, match[0] if match else None))
    return pairs
