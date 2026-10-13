"""Configuration: ground sites, areas of interest, paths and model defaults.

Everything here is plain data so it is easy to read, test and override.
Extra ground sites can be added with a TOML file named by ``ORBITWATCH_SITES_FILE``::

    [sites.my-backyard]
    latitude = 34.70
    longitude = -120.45
    elevation_m = 40
    timezone = "America/Los_Angeles"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GroundSite:
    name: str
    latitude: float  # degrees, north positive
    longitude: float  # degrees, east positive (California is negative)
    elevation_m: float  # metres above the WGS84 ellipsoid (approximated by height above sea level)
    timezone: str  # IANA name, used only to print local times
    source: str  # where the coordinates came from

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AreaOfInterest:
    name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat (WGS84 degrees)
    description: str

    def to_dict(self) -> dict:
        return {"name": self.name, "bbox": list(self.bbox), "description": self.description}


GROUND_SITES: dict[str, GroundSite] = {
    "lompoc": GroundSite(
        name="lompoc",
        latitude=34.64611,
        longitude=-120.46028,
        elevation_m=32.0,
        timezone="America/Los_Angeles",
        source="Wikipedia 'Lompoc, California' infobox: 34°38'46\"N 120°27'37\"W, 32 m",
    ),
    "vandenberg": GroundSite(
        name="vandenberg",
        latitude=34.75133,
        longitude=-120.52023,
        elevation_m=112.0,
        timezone="America/Los_Angeles",
        source="Wikipedia 'Vandenberg Space Force Base' infobox: 34.751330, -120.52023; "
        "airfield elevation 112 m",
    ),
}
DEFAULT_SITE = "lompoc"

AREAS_OF_INTEREST: dict[str, AreaOfInterest] = {
    "lompoc-valley-west": AreaOfInterest(
        name="lompoc-valley-west",
        bbox=(-120.53, 34.635, -120.49, 34.665),
        description="Farm fields in the Lompoc Valley just west of the city, between Lompoc "
        "and Vandenberg SFB (about 3.7 km x 3.3 km).",
    ),
}

# Small well-known satellites so callers (and the offline planner) can use names.
WELL_KNOWN_SATELLITES: dict[str, int] = {
    "iss": 25544,
    "zarya": 25544,
    "tiangong": 48274,
    "css": 48274,
    "tianhe": 48274,
    "hubble": 20580,
    "hst": 20580,
}

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"


def cache_dir() -> Path:
    """Directory for downloaded TLEs, the JPL ephemeris and generated rasters."""
    env = os.environ.get("ORBITWATCH_CACHE_DIR")
    path = Path(env) if env else Path.home() / ".cache" / "orbitwatch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    """Directory for generated PNG change maps."""
    env = os.environ.get("ORBITWATCH_OUTPUT_DIR")
    path = Path(env) if env else cache_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sites() -> dict[str, GroundSite]:
    """Built-in sites plus any defined in the TOML file named by ORBITWATCH_SITES_FILE."""
    sites = dict(GROUND_SITES)
    extra = os.environ.get("ORBITWATCH_SITES_FILE")
    if extra:
        data = tomllib.loads(Path(extra).read_text(encoding="utf-8"))
        for name, s in data.get("sites", {}).items():
            sites[name] = GroundSite(
                name=name,
                latitude=float(s["latitude"]),
                longitude=float(s["longitude"]),
                elevation_m=float(s.get("elevation_m", 0.0)),
                timezone=s.get("timezone", "UTC"),
                source=s.get("source", f"user sites file {extra}"),
            )
    return sites


def resolve_site(
    name: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
) -> GroundSite:
    """Pick a named site, or build an ad-hoc one when latitude and longitude are given."""
    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise ValueError("Provide both latitude and longitude, or neither.")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("latitude must be in [-90, 90] and longitude in [-180, 180].")
        return GroundSite(
            name="custom",  # never reuse a named site's label for other coordinates
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m or 0.0,
            timezone="UTC",
            source="caller-supplied coordinates",
        )
    sites = load_sites()
    key = (name or DEFAULT_SITE).strip().lower()
    if key not in sites:
        raise ValueError(f"Unknown site {name!r}. Known sites: {sorted(sites)}")
    return sites[key]
