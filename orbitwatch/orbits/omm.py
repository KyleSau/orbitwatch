"""Orbit Mean-Elements Messages (OMM): CelesTrak's JSON element sets.

OMM carries the same mean elements as a TLE (and is fitted for the same SGP4
propagator), but as named fields instead of fixed columns. orbitwatch uses it for
Starlink because the TLE format cannot represent catalog numbers above 99999, and
every object cataloged since 2026-07-11 (including the newest Starlink launches)
has one. OMM also gives the international designator (``OBJECT_ID``, e.g.
``2026-219A``) as its own field; the part before the piece letter identifies the
launch, which is what train detection groups by.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

EARTH_MU_KM3_S2 = 398600.4418  # WGS84 gravitational parameter
EARTH_EQUATORIAL_RADIUS_KM = 6378.137
SGP4_MAX_SATNUM = 339999
REQUIRED_FIELDS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "NORAD_CAT_ID",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)


class OmmError(ValueError):
    """Raised for malformed OMM records."""


@dataclass(frozen=True)
class Omm:
    name: str
    object_id: str  # international designator, e.g. "2026-219A"
    norad_id: int
    epoch: datetime
    mean_motion: float  # revolutions per day
    eccentricity: float
    inclination_deg: float
    fields: dict[str, Any] = field(compare=False, repr=False)  # the raw record, for SGP4

    @classmethod
    def from_fields(cls, rec: dict[str, Any]) -> Omm:
        missing = [k for k in REQUIRED_FIELDS if k not in rec]
        if missing:
            raise OmmError(f"OMM record is missing {missing}: {str(rec)[:120]}")
        try:
            epoch = datetime.fromisoformat(str(rec["EPOCH"])).replace(tzinfo=UTC)
            mean_motion = float(rec["MEAN_MOTION"])
            ecc = float(rec["ECCENTRICITY"])
            inc = float(rec["INCLINATION"])
            norad_id = int(rec["NORAD_CAT_ID"])
        except (TypeError, ValueError) as exc:
            raise OmmError(f"Bad numeric field in OMM record {rec.get('OBJECT_ID')}") from exc
        if not (0 < mean_motion < 20 and 0 <= ecc < 1 and 0 <= inc <= 180):
            raise OmmError(f"Implausible elements in OMM record {rec.get('OBJECT_ID')}")
        # sgp4's omm.initialize needs these three; CelesTrak always sends them but default
        # them so a hand-written record still propagates.
        full = {"CLASSIFICATION_TYPE": "U", "EPHEMERIS_TYPE": 0, "ELEMENT_SET_NO": 999}
        full.update(rec)
        full.setdefault("REV_AT_EPOCH", 0)
        if norad_id > SGP4_MAX_SATNUM:
            # sgp4 stores the catalog number in Alpha-5 (max 339999). It is only a label
            # for propagation, so 0 is safe; ``norad_id`` keeps the real number.
            full["NORAD_CAT_ID"] = 0
        return cls(
            name=str(rec["OBJECT_NAME"]),
            object_id=str(rec["OBJECT_ID"]),
            norad_id=norad_id,
            epoch=epoch,
            mean_motion=mean_motion,
            eccentricity=ecc,
            inclination_deg=inc,
            fields=full,
        )

    @property
    def launch(self) -> str:
        """Launch part of the international designator: '2026-219A' -> '2026-219'."""
        return self.object_id[:8]

    @property
    def semi_major_axis_km(self) -> float:
        n_rad_s = self.mean_motion * 2 * math.pi / 86400.0
        return (EARTH_MU_KM3_S2 / n_rad_s**2) ** (1 / 3)

    @property
    def mean_altitude_km(self) -> float:
        """Semi-major axis minus Earth's equatorial radius: a rough altitude, fine for
        near-circular low orbits (it ignores Earth's flattening, up to ~21 km at the poles)."""
        return self.semi_major_axis_km - EARTH_EQUATORIAL_RADIUS_KM

    def age_days(self, now: datetime) -> float:
        return (now - self.epoch).total_seconds() / 86400.0

    def with_name(self, name: str) -> Omm:
        """Copy with a different name (used to label withheld names, and by red-team tests)."""
        return Omm(
            name=name,
            object_id=self.object_id,
            norad_id=self.norad_id,
            epoch=self.epoch,
            mean_motion=self.mean_motion,
            eccentricity=self.eccentricity,
            inclination_deg=self.inclination_deg,
            fields={**self.fields, "OBJECT_NAME": name},
        )


def parse_omm_json(text: str) -> list[Omm]:
    """Parse CelesTrak's ``FORMAT=JSON`` response (a JSON array of OMM records)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OmmError(f"Response is not JSON: {text[:80]!r}") from exc
    if not isinstance(data, list):
        raise OmmError("Expected a JSON array of OMM records.")
    return [Omm.from_fields(rec) for rec in data]
