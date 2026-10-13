"""Record the offline fixtures in orbitwatch/fixtures from live services.

    python scripts/record_fixtures.py stac        # Earth Search items for the demo AOI
    python scripts/record_fixtures.py tle         # CelesTrak 'stations' group (mind the rate limit)
    python scripts/record_fixtures.py ephemeris   # 2-month excerpt of JPL DE421 (~25 KB)
    python scripts/record_fixtures.py starlink    # CelesTrak 'starlink' group as OMM JSON (gzip)
    python scripts/record_fixtures.py starlink --from-file saved.json --recorded-at 2026-...Z

Fixtures are real recorded metadata. STAC items are trimmed to the assets orbitwatch
reads (no pixels are stored). Run sparingly: CelesTrak allows one download per 2-hour
update cycle.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "orbitwatch" / "fixtures"
AOI = [-120.53, 34.635, -120.49, 34.665]
WINDOWS = ["2026-04-01/2026-04-30", "2026-09-01/2026-09-30"]
KEEP_ASSETS = {"red", "nir", "nir08", "swir22", "scl"}


def record_stac() -> None:
    from pystac_client import Client

    from orbitwatch.eo.stac import COLLECTION, EARTH_SEARCH_URL

    client = Client.open(EARTH_SEARCH_URL)
    features = []
    for window in WINDOWS:
        search = client.search(
            collections=[COLLECTION],
            bbox=AOI,
            datetime=window,
            query={"eo:cloud_cover": {"lt": 50}},
            max_items=100,
        )
        for item in search.items():
            d = item.to_dict()
            d["assets"] = {k: v for k, v in d["assets"].items() if k in KEEP_ASSETS}
            d["links"] = []
            features.append(d)
    out = {
        "type": "FeatureCollection",
        "features": features,
        "orbitwatch:recorded": {
            "source": EARTH_SEARCH_URL,
            "collection": COLLECTION,
            "bbox": AOI,
            "windows": WINDOWS,
            "max_cloud_cover_lt": 50,
            "recorded_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "Real STAC metadata trimmed to the assets orbitwatch reads.",
        },
    }
    path = FIXTURES / "stac_lompoc_valley_2026.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {len(features)} items to {path}")


def record_tle() -> None:
    import requests

    resp = requests.get(
        "https://celestrak.org/NORAD/elements/gp.php",
        params={"GROUP": "stations", "FORMAT": "TLE"},
        timeout=30,
    )
    resp.raise_for_status()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = FIXTURES / f"stations_{stamp}.tle"
    path.write_text(resp.text, encoding="utf-8")
    print(f"wrote {path} at {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}")


def record_starlink() -> None:
    """Download the Starlink group once as OMM JSON and store it gzipped, unmodified.

    ``--from-file PATH --recorded-at STAMP`` wraps a response that was already downloaded
    (so the fixture can be made without a second request inside the same 2-hour cycle).
    """
    args = sys.argv[2:]
    if "--from-file" in args:
        text = Path(args[args.index("--from-file") + 1]).read_text(encoding="utf-8")
        stamp = args[args.index("--recorded-at") + 1]
    else:
        import requests

        from orbitwatch.orbits.celestrak import GP_URL, USER_AGENT

        resp = requests.get(
            GP_URL,
            params={"GROUP": "starlink", "FORMAT": "JSON"},
            timeout=60,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        text, stamp = resp.text, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = json.loads(text)
    out = {
        "orbitwatch:recorded": {
            "source": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=JSON",
            "recorded_at_utc": stamp,
            "record_count": len(records),
            "note": "Real CelesTrak GP data (OMM), records unmodified; only re-serialised "
            "compactly and gzipped.",
        },
        "records": records,
    }
    path = FIXTURES / f"starlink_{stamp[:10]}.json.gz"
    blob = json.dumps(out, separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(blob, mtime=0))  # mtime=0: byte-stable output
    print(f"wrote {len(records)} records to {path} ({path.stat().st_size} bytes)")


def record_ephemeris() -> None:
    from skyfield.api import Loader

    from orbitwatch import config

    loader = Loader(str(config.cache_dir() / "ephemeris"), verbose=False)
    loader("de421.bsp")
    src = Path(loader.directory) / "de421.bsp"
    dst = FIXTURES / "de421_2026-10_2026-11.bsp"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jplephem",
            "excerpt",
            "2026/10/1",
            "2026/11/30",
            str(src),
            str(dst),
        ],
        check=True,
    )


if __name__ == "__main__":
    {
        "stac": record_stac,
        "tle": record_tle,
        "ephemeris": record_ephemeris,
        "starlink": record_starlink,
    }[sys.argv[1]]()
