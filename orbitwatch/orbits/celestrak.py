"""Fetch element sets from CelesTrak's public GP API, with an on-disk cache.

CelesTrak refreshes GP data every 2 hours and enforces one download per update
cycle; repeated requests can get an IP address blocked. So we:

* cache every response on disk and reuse it for 2 hours (``DEFAULT_TTL``),
* never retry a 403/404 (the answer will not change by asking again),
* fall back to a stale cached copy, clearly flagged, if a refresh fails.

Data courtesy of CelesTrak (https://celestrak.org), Dr. T.S. Kelso.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import requests

from orbitwatch.orbits.tle import Tle, parse_tle_text

GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
DEFAULT_TTL = timedelta(hours=2)
USER_AGENT = "orbitwatch/0.1 (+https://github.com/kylesau/orbitwatch)"

# Groups exposed to tools. Names match CelesTrak's "Current Data" page.
SUPPORTED_GROUPS = ("stations", "visual", "weather", "science", "gps-ops")


class CelesTrakError(RuntimeError):
    """Raised when element sets cannot be fetched and no cached copy exists."""


@dataclass
class TleResult:
    tles: list[Tle]
    source: str  # e.g. "CelesTrak GP API" or "fixture file"
    retrieved_at: datetime  # when the data was downloaded (cache file mtime)
    from_cache: bool
    stale: bool = False
    warnings: list[str] = field(default_factory=list)


class TleSource(Protocol):
    def get_tle(self, norad_id: int) -> TleResult: ...

    def get_group(self, group: str) -> TleResult: ...


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CelesTrakClient:
    """Live TLE source backed by CelesTrak with a simple file cache."""

    def __init__(
        self,
        cache_dir: Path,
        ttl: timedelta = DEFAULT_TTL,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] = _utcnow,
        timeout_s: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "tle"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.session = session or requests.Session()
        self.clock = clock
        self.timeout_s = timeout_s

    def get_tle(self, norad_id: int) -> TleResult:
        if not 0 < int(norad_id) <= 99999:
            raise CelesTrakError(
                "TLE format only supports catalog numbers 1-99999; newer objects need OMM."
            )
        result = self._get({"CATNR": str(int(norad_id))}, f"catnr-{int(norad_id)}.tle")
        if not result.tles:
            raise CelesTrakError(f"CelesTrak returned no element set for NORAD {norad_id}.")
        return result

    def get_group(self, group: str) -> TleResult:
        group = group.lower()
        if group not in SUPPORTED_GROUPS:
            raise CelesTrakError(f"Unsupported group {group!r}; choose from {SUPPORTED_GROUPS}.")
        return self._get({"GROUP": group}, f"group-{group}.tle")

    # -- internals ---------------------------------------------------------------

    def _read_cache(self, path: Path, stale: bool, warning: str | None = None) -> TleResult:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return TleResult(
            tles=parse_tle_text(path.read_text(encoding="utf-8")),
            source="CelesTrak GP API (cached)",
            retrieved_at=mtime,
            from_cache=True,
            stale=stale,
            warnings=[warning] if warning else [],
        )

    def _get(self, params: dict[str, str], cache_name: str) -> TleResult:
        path = self.cache_dir / cache_name
        if path.exists():
            age = self.clock() - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if age < self.ttl:
                return self._read_cache(path, stale=False)
        try:
            resp = self.session.get(
                GP_URL,
                params={**params, "FORMAT": "TLE"},
                timeout=self.timeout_s,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            text = resp.text
            if re.search(r"No GP data found", text, re.IGNORECASE):
                raise CelesTrakError(f"CelesTrak: no GP data found for {params}.")
            tles = parse_tle_text(text)
        except (requests.RequestException, CelesTrakError) as exc:
            if path.exists():
                return self._read_cache(
                    path, stale=True, warning=f"Refresh failed ({exc}); using stale cached copy."
                )
            raise CelesTrakError(f"Could not fetch {params} from CelesTrak: {exc}") from exc

        now = self.clock()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)  # atomic on the same filesystem
        # The file's mtime *is* the download time; stamp it from our clock so cache age is
        # always measured on one clock (matters when a test injects a fake clock).
        os.utime(path, (now.timestamp(), now.timestamp()))
        return TleResult(tles=tles, source="CelesTrak GP API", retrieved_at=now, from_cache=False)


class FileTleSource:
    """Offline TLE source that reads a saved TLE file (used for tests, evals, demos)."""

    def __init__(self, path: Path, retrieved_at: datetime) -> None:
        self.path = Path(path)
        self.retrieved_at = retrieved_at

    def _all(self) -> list[Tle]:
        return parse_tle_text(self.path.read_text(encoding="utf-8"))

    def get_tle(self, norad_id: int) -> TleResult:
        tles = [t for t in self._all() if t.norad_id == int(norad_id)]
        if not tles:
            raise CelesTrakError(f"NORAD {norad_id} is not in fixture file {self.path.name}.")
        return TleResult(tles, f"fixture file {self.path.name}", self.retrieved_at, True)

    def get_group(self, group: str) -> TleResult:
        return TleResult(self._all(), f"fixture file {self.path.name}", self.retrieved_at, True)
