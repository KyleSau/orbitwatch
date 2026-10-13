"""Search Sentinel-2 L2A scenes through a STAC API and normalise item metadata.

STAC (SpatioTemporal Asset Catalog) is a JSON standard for describing imagery:
each *item* is one scene with a footprint, a timestamp, properties such as cloud
cover, and *assets* (one Cloud-Optimized GeoTIFF per spectral band).

Live data comes from Element 84's Earth Search
(https://earth-search.aws.element84.com/v1, collection ``sentinel-2-l2a``),
which indexes the Sentinel-2 COGs in the AWS Open Data bucket ``sentinel-cogs``.
Contains modified Copernicus Sentinel data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_ITEMS_FETCHED = 200

# Asset keys as returned by Earth Search v1 (inspected from live items, not assumed).
BAND_ASSETS = ("red", "nir", "nir08", "swir22", "scl")


@dataclass(frozen=True)
class BandAsset:
    key: str
    href: str
    scale: float  # from raster:bands
    declared_offset: float  # from raster:bands (see effective_offset for why it may be ignored)
    nodata: float | None
    resolution_m: float


@dataclass(frozen=True)
class Scene:
    id: str
    datetime: datetime
    cloud_cover_pct: float
    mgrs_tile: str
    platform: str
    processing_baseline: str
    boa_offset_applied: bool | None
    nodata_pixel_pct: float | None
    crs: str
    bbox: tuple[float, float, float, float]
    assets: dict[str, BandAsset] = field(repr=False)
    synthetic_pixels: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "scene_id": self.id,
            "datetime_utc": self.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date": self.datetime.strftime("%Y-%m-%d"),
            "cloud_cover_pct": round(self.cloud_cover_pct, 2),
            "mgrs_tile": self.mgrs_tile,
            "platform": self.platform,
            "processing_baseline": self.processing_baseline,
            "nodata_pixel_pct": (
                round(self.nodata_pixel_pct, 2) if self.nodata_pixel_pct is not None else None
            ),
        }


def scene_from_item(item: dict[str, Any]) -> Scene:
    """Build a Scene from a STAC item dict (as returned by the API / pystac ``to_dict``)."""
    props = item["properties"]
    assets: dict[str, BandAsset] = {}
    for key in BAND_ASSETS:
        a = item.get("assets", {}).get(key)
        if a is None:
            continue
        rb = (a.get("raster:bands") or [{}])[0]
        assets[key] = BandAsset(
            key=key,
            href=a["href"],
            scale=float(rb.get("scale", 1.0)),
            declared_offset=float(rb.get("offset", 0.0)),
            nodata=rb.get("nodata"),
            resolution_m=float(rb.get("spatial_resolution") or a.get("gsd") or 10.0),
        )
    return Scene(
        id=item["id"],
        datetime=datetime.fromisoformat(props["datetime"].replace("Z", "+00:00")),
        cloud_cover_pct=float(props.get("eo:cloud_cover", float("nan"))),
        mgrs_tile=str(props.get("grid:code", "")).removeprefix("MGRS-"),
        platform=str(props.get("platform", "")),
        processing_baseline=str(props.get("s2:processing_baseline", "")),
        boa_offset_applied=props.get("earthsearch:boa_offset_applied"),
        nodata_pixel_pct=props.get("s2:nodata_pixel_percentage"),
        crs=str(props.get("proj:code") or f"EPSG:{props.get('proj:epsg')}"),
        bbox=tuple(item["bbox"]),
        assets=assets,
        synthetic_pixels=bool(props.get("orbitwatch:synthetic_pixels", False)),
    )


def effective_offset(scene: Scene, asset: BandAsset) -> tuple[float, str]:
    """Return the reflectance offset to apply to ``asset`` and a reason string.

    Since processing baseline 04.00 (Jan 2022) ESA adds +1000 to L2A digital numbers,
    so reflectance = DN * 0.0001 - 0.1. Earth Search items declare that as
    ``raster:bands`` offset -0.1. But items flagged ``earthsearch:boa_offset_applied``
    already have the +1000 removed from the pixels: in a direct comparison of the same
    acquisition processed at baseline 03.00 and 05.00 (scripts/verify_boa_offset.py)
    the raw DNs matched. Applying -0.1 again would double-subtract, so we skip it.
    """
    if scene.boa_offset_applied:
        return 0.0, (
            "earthsearch:boa_offset_applied=true -> pixels already offset-corrected; "
            f"declared offset {asset.declared_offset} not applied"
        )
    return asset.declared_offset, f"declared raster:bands offset {asset.declared_offset} applied"


def _check_dates(start: str, end: str) -> None:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if e < s:
        raise ValueError(f"end_date {end} is before start_date {start}.")


def validate_bbox(bbox: list[float] | tuple[float, ...], max_span_deg: float = 0.25) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError(f"Invalid bbox {list(bbox)}: need min < max within lon/lat ranges.")
    if max_lon - min_lon > max_span_deg or max_lat - min_lat > max_span_deg:
        raise ValueError(
            f"bbox spans more than {max_span_deg} degrees; keep areas of interest small "
            "so only a few COG tiles are read."
        )


class SceneCatalog(Protocol):
    def search(self, bbox: list[float], start: str, end: str, max_cloud: float) -> list[Scene]: ...

    def get(self, scene_id: str) -> Scene: ...


class EarthSearchCatalog:
    """Live catalog: Element 84 Earth Search via pystac-client."""

    def __init__(self, url: str = EARTH_SEARCH_URL, client=None) -> None:
        self.url = url
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from pystac_client import Client

            self._client = Client.open(self.url)
        return self._client

    def search(self, bbox: list[float], start: str, end: str, max_cloud: float) -> list[Scene]:
        _check_dates(start, end)
        search = self.client.search(
            collections=[COLLECTION],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=MAX_ITEMS_FETCHED,
        )
        scenes = [scene_from_item(it.to_dict()) for it in search.items()]
        return sorted(scenes, key=lambda s: s.datetime)

    def get(self, scene_id: str) -> Scene:
        items = list(self.client.search(collections=[COLLECTION], ids=[scene_id]).items())
        if not items:
            raise ValueError(f"Scene {scene_id!r} not found in {COLLECTION}.")
        return scene_from_item(items[0].to_dict())


def _bbox_intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class FixtureCatalog:
    """Offline catalog over a saved STAC FeatureCollection (tests, evals, demos).

    ``rewrite_item`` lets the caller swap asset hrefs (e.g. to local synthetic rasters)
    and relabel ids so fixture output can never be mistaken for real measurements.
    """

    def __init__(self, path: Path, rewrite_item=None) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data["features"]
        if rewrite_item is not None:
            items = [rewrite_item(it) for it in items]
        self._scenes = {s.id: s for s in (scene_from_item(it) for it in items)}

    def search(self, bbox: list[float], start: str, end: str, max_cloud: float) -> list[Scene]:
        _check_dates(start, end)
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        hits = [
            sc
            for sc in self._scenes.values()
            if s <= sc.datetime.date() <= e
            and sc.cloud_cover_pct < max_cloud
            and _bbox_intersects(sc.bbox, bbox)
        ]
        return sorted(hits, key=lambda sc: sc.datetime)

    def get(self, scene_id: str) -> Scene:
        if scene_id not in self._scenes:
            raise ValueError(f"Scene {scene_id!r} not in fixture catalog.")
        return self._scenes[scene_id]
