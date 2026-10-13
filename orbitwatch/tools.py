"""The tools, defined once and shared by the MCP server, the Claude agent and the offline planner.

Each tool is a method on ``Toolbox`` with typed, documented parameters. The method
signature *is* the contract:

* the MCP server registers the bound methods directly (the MCP SDK derives JSON Schema
  from the signature and uses the docstring as the description);
* the Claude agent derives its ``input_schema`` from the same signature via pydantic
  (``input_model``), so the two can never drift (a test asserts they match).

Every tool returns a JSON-serialisable dict, with numbers rounded at the source so an
answer can quote them exactly. Bad input raises ``ValueError`` with a message written
for the model to read and recover from.
"""

from __future__ import annotations

import inspect
import json
from datetime import timedelta
from functools import cache
from typing import Annotated, Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from orbitwatch import config
from orbitwatch.context import DataContext
from orbitwatch.eo.analysis import compare_scenes
from orbitwatch.eo.stac import validate_bbox
from orbitwatch.orbits.passes import predict_passes as _predict_passes

TOOL_NAMES = (
    "list_places",
    "find_satellites",
    "get_tle",
    "predict_passes",
    "search_sentinel2_scenes",
    "compare_index_between_scenes",
)

Bbox = Annotated[
    list[float],
    Field(
        description="Area of interest as [min_lon, min_lat, max_lon, max_lat] in WGS84 "
        "degrees. Keep it small (at most 0.25 degrees on a side)."
    ),
]
# Must list the same names as celestrak.SUPPORTED_GROUPS (a test checks this).
Group = Literal["stations", "visual", "weather", "science", "gps-ops"]


class ToolError(Exception):
    """A tool call failed; the message is safe to show to the model."""


class Toolbox:
    def __init__(self, ctx: DataContext) -> None:
        self.ctx = ctx

    # -- places --------------------------------------------------------------------

    def list_places(self) -> dict[str, Any]:
        """List the named ground sites (observer locations for satellite pass predictions)
        and named areas of interest (bounding boxes for Sentinel-2 analysis) that the other
        tools accept. Call this when the user mentions a place by name, such as Lompoc or
        Vandenberg, and you need its coordinates or bounding box."""
        return {
            "ground_sites": [s.to_dict() for s in config.load_sites().values()],
            "default_site": config.DEFAULT_SITE,
            "areas_of_interest": [a.to_dict() for a in config.AREAS_OF_INTEREST.values()],
        }

    # -- orbits --------------------------------------------------------------------

    def find_satellites(
        self,
        query: Annotated[
            str,
            Field(
                description="Case-insensitive part of the satellite name, e.g. 'TIANHE' or 'ISS'."
            ),
        ],
        group: Annotated[Group, Field(description="CelesTrak group to search.")] = "stations",
    ) -> dict[str, Any]:
        """Look up NORAD catalog numbers by satellite name within a CelesTrak group.
        Call this when the user names a satellite whose NORAD ID you do not know (the ISS is
        25544). Returns at most 20 matches."""
        q = query.strip().lower()
        if not q:
            raise ValueError("query must not be empty.")
        result = self.ctx.tle_source.get_group(group)
        matches = [
            {"name": t.name, "norad_id": t.norad_id} for t in result.tles if q in t.name.lower()
        ]
        return {
            "query": query,
            "group": group,
            "match_count": len(matches),
            "matches": matches[:20],
            "source": result.source,
        }

    def get_tle(
        self,
        norad_id: Annotated[
            int, Field(description="NORAD catalog number, e.g. 25544 for the ISS.")
        ],
    ) -> dict[str, Any]:
        """Fetch the current two-line element set (TLE) for one satellite from CelesTrak,
        with its epoch and age. Call this when the user asks about a satellite's orbital
        elements or how fresh its orbit data is."""
        result = self.ctx.tle_source.get_tle(norad_id)
        tle = result.tles[0]
        now = self.ctx.clock()
        return {
            "name": tle.name,
            "norad_id": tle.norad_id,
            "line1": tle.line1,
            "line2": tle.line2,
            "epoch_utc": tle.epoch.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "age_days": round(tle.age_days(now), 2),
            "source": result.source,
            "retrieved_at_utc": result.retrieved_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "warnings": result.warnings,
        }

    def predict_passes(
        self,
        norad_id: Annotated[int, Field(description="NORAD catalog number (ISS = 25544).")] = 25544,
        site: Annotated[
            str,
            Field(description="Named ground site from list_places, e.g. 'lompoc' or 'vandenberg'."),
        ] = config.DEFAULT_SITE,
        days: Annotated[
            float,
            Field(
                description="Length of the prediction window in days, "
                "starting now (0 < days <= 10)."
            ),
        ] = 3.0,
        min_elevation_deg: Annotated[
            float,
            Field(description="Only report passes rising above this elevation, 0-80 degrees."),
        ] = 10.0,
        visible_only: Annotated[
            bool,
            Field(
                description="If true, return only passes where the "
                "satellite is sunlit while the observer's sky is "
                "dark (naked-eye visible)."
            ),
        ] = False,
        latitude: Annotated[
            float | None, Field(description="Custom observer latitude; overrides site.")
        ] = None,
        longitude: Annotated[
            float | None, Field(description="Custom observer longitude; overrides site.")
        ] = None,
    ) -> dict[str, Any]:
        """Predict when a satellite passes over a ground site: rise, highest point
        (culmination) and set times in UTC and local time, maximum elevation, azimuths, and
        whether each pass is naked-eye visible. Uses the latest CelesTrak TLE and SGP4
        propagation. Call this for any question about when a satellite is overhead or can be
        seen from a place."""
        if not 0 < days <= 10:
            raise ValueError("days must be > 0 and <= 10 (SGP4 accuracy degrades with time).")
        if not 0 <= min_elevation_deg <= 80:
            raise ValueError("min_elevation_deg must be between 0 and 80.")
        ground = config.resolve_site(site, latitude, longitude)
        result = self.ctx.tle_source.get_tle(norad_id)
        tle = result.tles[0]
        start = self.ctx.clock().replace(microsecond=0)
        end = start + timedelta(days=days)
        warnings = list(result.warnings)

        try:
            eph = self.ctx.load_ephemeris()
        except Exception as exc:  # e.g. offline and not yet downloaded
            eph = None
            warnings.append(f"Ephemeris unavailable ({exc}); visibility not computed.")
            if visible_only:
                raise ValueError(
                    "visible_only needs the JPL ephemeris, which is unavailable."
                ) from exc

        passes = _predict_passes(tle, ground, start, end, min_elevation_deg, eph)
        age = tle.age_days(start)
        if abs(age) > 3:
            warnings.append(f"TLE epoch is {age:.1f} days from now; SGP4 error grows with age.")
        horizon = (end - tle.epoch).total_seconds() / 86400
        if horizon > 7:
            warnings.append(
                "Window ends more than 7 days after the TLE epoch; times may be off "
                "by minutes, especially for low orbits like the ISS."
            )
        selected = [p for p in passes if p.visible] if visible_only else passes
        return {
            "satellite": tle.name,
            "norad_id": tle.norad_id,
            "tle_epoch_utc": tle.epoch.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "tle_age_days": round(age, 2),
            "tle_source": result.source,
            "site": ground.to_dict(),
            "window_start_utc": start.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "window_end_utc": end.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "min_elevation_deg": min_elevation_deg,
            "visible_only": visible_only,
            "passes_above_threshold": len(passes),
            "visible_pass_count": sum(1 for p in passes if p.visible) if eph is not None else None,
            "passes": [p.to_dict(ground.timezone) for p in selected],
            "visibility_rule": "visible = satellite sunlit while Sun is below -6 deg at the site",
            "warnings": warnings,
        }

    # -- Earth observation ---------------------------------------------------------

    def search_sentinel2_scenes(
        self,
        bbox: Bbox,
        start_date: Annotated[str, Field(description="Start date, YYYY-MM-DD (inclusive).")],
        end_date: Annotated[str, Field(description="End date, YYYY-MM-DD (inclusive).")],
        max_cloud_cover: Annotated[
            float, Field(description="Maximum scene cloud cover in percent (0-100).")
        ] = 20.0,
        limit: Annotated[
            int, Field(description="Maximum scenes to return (1-50); the least cloudy are kept.")
        ] = 10,
    ) -> dict[str, Any]:
        """Search Sentinel-2 Level-2A (surface reflectance) scenes covering an area of interest
        within a date range, filtered by scene cloud cover, via the Earth Search STAC API.
        Returns scene IDs, acquisition dates, cloud cover and MGRS tile. Call this before
        compare_index_between_scenes to choose the before and after scenes."""
        validate_bbox(bbox)
        if not 0 <= max_cloud_cover <= 100:
            raise ValueError("max_cloud_cover must be between 0 and 100.")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50.")
        scenes = self.ctx.catalog.search(bbox, start_date, end_date, max_cloud_cover)
        kept = sorted(
            sorted(scenes, key=lambda s: s.cloud_cover_pct)[:limit], key=lambda s: s.datetime
        )
        return {
            "bbox": list(bbox),
            "start_date": start_date,
            "end_date": end_date,
            "max_cloud_cover": max_cloud_cover,
            "matched": len(scenes),
            "returned": len(kept),
            "scenes": [s.summary() for s in kept],
            "note": "cloud_cover_pct is for the whole ~110 km tile, not just the bbox; "
            "compare_index_between_scenes reports AOI-level masking.",
            "data_source": "FIXTURES" if self.ctx.mode == "fixtures" else "Earth Search STAC",
        }

    def compare_index_between_scenes(
        self,
        before_scene_id: Annotated[
            str, Field(description="Scene ID of the earlier image, from search_sentinel2_scenes.")
        ],
        after_scene_id: Annotated[
            str,
            Field(description="Scene ID of the later image (same MGRS tile as the before scene)."),
        ],
        bbox: Bbox,
        index: Annotated[
            Literal["ndvi", "nbr"],
            Field(description="'ndvi' for vegetation vigour, 'nbr' for burn severity."),
        ] = "ndvi",
        change_threshold: Annotated[
            float,
            Field(
                description="A pixel counts as changed when "
                "|after - before| exceeds this (index units)."
            ),
        ] = 0.1,
        render_png: Annotated[bool, Field(description="Also render a PNG change map.")] = True,
    ) -> dict[str, Any]:
        """Compute a spectral index (NDVI or NBR) for the area of interest in two Sentinel-2
        scenes and summarise the change (after minus before): means, percentiles and the
        percentage of pixels that decreased or increased beyond a threshold, after masking
        clouds, shadows and nodata with the scene classification layer. Reads only the needed
        window of each cloud-optimized GeoTIFF. Call this for questions about how vegetation
        or burn severity changed between two dates."""
        before = self.ctx.catalog.get(before_scene_id)
        after = self.ctx.catalog.get(after_scene_id)
        png = None
        if render_png:
            png = self.ctx.output_dir / f"{index}_change_{before.id}__{after.id}.png"
        return compare_scenes(before, after, bbox, index, change_threshold, png)


# -- schema + dispatch helpers used by the Claude agent ---------------------------------


@cache
def input_model(name: str) -> type[BaseModel]:
    """Build a pydantic model from a Toolbox method signature (extra fields forbidden)."""
    fn = getattr(Toolbox, name)
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        if pname == "self":
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (hints[pname], default)
    return create_model(f"{name}_input", __config__=ConfigDict(extra="forbid"), **fields)


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def anthropic_tool_definitions() -> list[dict[str, Any]]:
    """Tool definitions for the Messages API: strict schemas, additionalProperties false."""
    tools = []
    for name in TOOL_NAMES:
        schema = _strip_titles(input_model(name).model_json_schema())
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        tools.append(
            {
                "name": name,
                "description": inspect.getdoc(getattr(Toolbox, name)),
                "input_schema": schema,
                "strict": True,
            }
        )
    return tools


def call_tool(toolbox: Toolbox, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate ``arguments`` against the tool's schema and run it. Raises ToolError."""
    if name not in TOOL_NAMES:
        raise ToolError(f"Unknown tool {name!r}. Available: {list(TOOL_NAMES)}")
    try:
        args = input_model(name).model_validate(arguments or {})
    except ValidationError as exc:
        raise ToolError(f"Invalid arguments for {name}: {exc.errors(include_url=False)}") from exc
    try:
        return getattr(toolbox, name)(**dict(args))
    except ToolError:
        raise
    except Exception as exc:  # surface any failure to the model as an error result
        raise ToolError(f"{name} failed: {type(exc).__name__}: {exc}") from exc


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
