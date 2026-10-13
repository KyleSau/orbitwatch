"""Where tools get their data: live services or recorded fixtures.

This is plain dependency injection. Tools never reach for a global client; they use
the ``DataContext`` they were given, so tests, evals and the offline demo can run
the exact same code with no network access.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orbitwatch import config
from orbitwatch.eo.stac import EarthSearchCatalog, FixtureCatalog, SceneCatalog
from orbitwatch.eo.synthetic import make_fixture_rewriter
from orbitwatch.orbits.celestrak import CelesTrakClient, FileTleSource, TleSource

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_TLE_FILE = FIXTURES_DIR / "stations_2026-10-13.tle"
FIXTURE_EPHEMERIS = FIXTURES_DIR / "de421_2026-10_2026-11.bsp"
FIXTURE_STAC_FILE = FIXTURES_DIR / "stac_lompoc_valley_2026.json"
# Fixed "now" for fixture mode, shortly after the recorded TLEs were downloaded.
FIXTURE_CLOCK = datetime(2026, 10, 13, 20, 0, 0, tzinfo=UTC)


@dataclass
class DataContext:
    mode: str  # "live" or "fixtures"
    tle_source: TleSource
    catalog: SceneCatalog
    load_ephemeris: Callable[[], Any]
    clock: Callable[[], datetime]
    output_dir: Path
    notes: list[str] = field(default_factory=list)

    @classmethod
    def live(cls, cache_dir: Path | None = None, output_dir: Path | None = None) -> DataContext:
        cache_root = Path(cache_dir) if cache_dir else config.cache_dir()

        @functools.cache
        def load_ephemeris():
            from skyfield.api import Loader

            # JPL DE421 (~17 MB), downloaded once into the cache directory.
            return Loader(str(cache_root / "ephemeris"), verbose=False)("de421.bsp")

        return cls(
            mode="live",
            tle_source=CelesTrakClient(cache_root),
            catalog=EarthSearchCatalog(),
            load_ephemeris=load_ephemeris,
            clock=lambda: datetime.now(UTC),
            output_dir=Path(output_dir) if output_dir else config.output_dir(),
        )

    @classmethod
    def fixtures(cls, work_dir: Path | None = None) -> DataContext:
        work = Path(work_dir) if work_dir else config.cache_dir() / "fixture-mode"
        work.mkdir(parents=True, exist_ok=True)

        @functools.cache
        def load_ephemeris():
            from skyfield.api import load_file

            return load_file(str(FIXTURE_EPHEMERIS))

        return cls(
            mode="fixtures",
            tle_source=FileTleSource(FIXTURE_TLE_FILE, retrieved_at=FIXTURE_CLOCK),
            catalog=FixtureCatalog(
                FIXTURE_STAC_FILE, rewrite_item=make_fixture_rewriter(work / "synthetic")
            ),
            load_ephemeris=load_ephemeris,
            clock=lambda: FIXTURE_CLOCK,
            output_dir=work / "outputs",
            notes=[
                "FIXTURE DATA: TLEs recorded from CelesTrak on 2026-10-13, STAC metadata "
                "recorded from Earth Search, pixel values synthetic. Clock fixed at "
                f"{FIXTURE_CLOCK:%Y-%m-%d %H:%M} UTC."
            ],
        )
