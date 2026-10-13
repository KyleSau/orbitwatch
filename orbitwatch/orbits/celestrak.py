"""Fetch element sets from CelesTrak's public GP API, with an on-disk cache.

CelesTrak refreshes GP data every 2 hours and enforces one download per update
cycle; repeated requests can get an IP address blocked. So we:

* cache every response on disk and reuse it for 2 hours (``DEFAULT_TTL``),
* never retry a 403/404 (the answer will not change by asking again),
* fall back to a stale cached copy, clearly flagged, if a refresh fails.

The same rules apply to TLE text and to OMM JSON (used for the Starlink group).

Data courtesy of CelesTrak (https://celestrak.org), Dr. T.S. Kelso.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import requests

from orbitwatch.orbits.omm import Omm, parse_omm_json
from orbitwatch.orbits.tle import Tle, parse_tle_text

GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
DEFAULT_TTL = timedelta(hours=2)
USER_AGENT = "orbitwatch/0.1 (+https://github.com/kylesau/orbitwatch)"

# Groups exposed to tools. Names match CelesTrak's "Current Data" page.
SUPPORTED_GROUPS = ("stations", "visual", "weather", "science", "gps-ops")
# Groups fetched as OMM JSON instead. Starlink is ~11k objects, many with 6-digit catalog
# numbers that the TLE format cannot hold; it is one ~5 MB download per 2-hour cycle at most.
SUPPORTED_OMM_GROUPS = ("starlink",)


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


@dataclass
class OmmResult:
    records: list[Omm]
    source: str
    retrieved_at: datetime
    from_cache: bool
    stale: bool = False
    warnings: list[str] = field(default_factory=list)


class TleSource(Protocol):
    def get_tle(self, norad_id: int) -> TleResult: ...

    def get_group(self, group: str) -> TleResult: ...

    def get_omm_group(self, group: str) -> OmmResult: ...


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CelesTrakClient:
    """Live element-set source backed by CelesTrak with a simple file cache."""

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
        result = TleResult(*self._get({"CATNR": str(int(norad_id))}, f"catnr-{int(norad_id)}.tle"))
        if not result.tles:
            raise CelesTrakError(f"CelesTrak returned no element set for NORAD {norad_id}.")
        return result

    def get_group(self, group: str) -> TleResult:
        group = group.lower()
        if group not in SUPPORTED_GROUPS:
            raise CelesTrakError(f"Unsupported group {group!r}; choose from {SUPPORTED_GROUPS}.")
        return TleResult(*self._get({"GROUP": group}, f"group-{group}.tle"))

    def get_omm_group(self, group: str) -> OmmResult:
        """A whole group as OMM JSON records (same cache and rate policy as TLEs)."""
        group = group.lower()
        if group not in SUPPORTED_OMM_GROUPS:
            raise CelesTrakError(
                f"Unsupported OMM group {group!r}; choose from {SUPPORTED_OMM_GROUPS}."
            )
        return OmmResult(
            *self._get({"GROUP": group}, f"group-{group}.json", "JSON", parse_omm_json)
        )

    # -- internals ---------------------------------------------------------------

    def _get(
        self,
        params: dict[str, str],
        cache_name: str,
        fmt: str = "TLE",
        parse: Callable[[str], list[Any]] = parse_tle_text,
    ) -> tuple[list[Any], str, datetime, bool, bool, list[str]]:
        """Positional fields of a TleResult/OmmResult: (items, source, retrieved_at,
        from_cache, stale, warnings). A download is parsed before it is cached, so a
        corrupt response never replaces a good cached copy."""
        path = self.cache_dir / cache_name
        if path.exists():
            age = self.clock() - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if age < self.ttl:
                return self._read_cache(path, parse, stale=False)
        try:
            resp = self.session.get(
                GP_URL,
                params={**params, "FORMAT": fmt},
                timeout=self.timeout_s,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            text = resp.text
            if re.search(r"No GP data found", text, re.IGNORECASE):
                raise CelesTrakError(f"CelesTrak: no GP data found for {params}.")
            items = parse(text)
        except (requests.RequestException, ValueError, CelesTrakError) as exc:
            if path.exists():
                warning = f"Refresh failed ({exc}); using stale cached copy."
                return self._read_cache(path, parse, stale=True, warning=warning)
            raise CelesTrakError(f"Could not fetch {params} from CelesTrak: {exc}") from exc

        now = self.clock()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)  # atomic on the same filesystem
        # The file's mtime *is* the download time; stamp it from our clock so cache age is
        # always measured on one clock (matters when a test injects a fake clock).
        os.utime(path, (now.timestamp(), now.timestamp()))
        return items, "CelesTrak GP API", now, False, False, []

    @staticmethod
    def _read_cache(path: Path, parse: Callable[[str], list[Any]], stale: bool, warning=None):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        items = parse(path.read_text(encoding="utf-8"))
        return items, "CelesTrak GP API (cached)", mtime, True, stale, [warning] if warning else []


class FileTleSource:
    """Offline element-set source that reads saved files (used for tests, evals, demos).

    ``omm_files`` maps OMM group names to fixtures recorded by
    ``scripts/record_fixtures.py starlink`` (gzipped JSON with the recording metadata).
    """

    def __init__(
        self, path: Path, retrieved_at: datetime, omm_files: dict[str, Path] | None = None
    ) -> None:
        self.path = Path(path)
        self.retrieved_at = retrieved_at
        self.omm_files = {k: Path(v) for k, v in (omm_files or {}).items()}

    def _all(self) -> list[Tle]:
        return parse_tle_text(self.path.read_text(encoding="utf-8"))

    def get_tle(self, norad_id: int) -> TleResult:
        tles = [t for t in self._all() if t.norad_id == int(norad_id)]
        if not tles:
            raise CelesTrakError(f"NORAD {norad_id} is not in fixture file {self.path.name}.")
        return TleResult(tles, f"fixture file {self.path.name}", self.retrieved_at, True)

    def get_group(self, group: str) -> TleResult:
        return TleResult(self._all(), f"fixture file {self.path.name}", self.retrieved_at, True)

    def get_omm_group(self, group: str) -> OmmResult:
        path = self.omm_files.get(group.lower())
        if path is None:
            raise CelesTrakError(f"No recorded OMM fixture for group {group!r}.")
        recorded = load_omm_fixture(path)
        stamp = recorded["orbitwatch:recorded"]["recorded_at_utc"]
        return OmmResult(
            records=parse_omm_json(json.dumps(recorded["records"])),
            source=f"fixture file {path.name}",
            retrieved_at=datetime.fromisoformat(stamp.replace("Z", "+00:00")),
            from_cache=True,
        )


def load_omm_fixture(path: Path) -> dict[str, Any]:
    """Read a recorded OMM fixture: ``{"orbitwatch:recorded": {...}, "records": [...]}``."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))
