"""Offline, deterministic planner: answers a fixed set of question types without an LLM.

This exists so the project is demoable (and its evals runnable in CI) without an API
key. It is *scripted*: keyword rules pick an intent, regexes pull out parameters, the
same validated tools the Claude agent uses are called, and a template formats the answer
using only values the tools returned. Anything outside the supported intents gets a
clear "not supported offline" reply rather than a guess.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Any

from orbitwatch import config
from orbitwatch.agent.types import AgentResult, ToolCall
from orbitwatch.tools import Toolbox, ToolError, call_tool

OFFLINE_LABEL = "[OFFLINE MODE: scripted keyword planner, no LLM]"
SUPPORTED = (
    "satellite passes (e.g. 'When are the next visible ISS passes over Lompoc?'), "
    "TLE lookup, Sentinel-2 scene search, and NDVI/NBR change between two dates"
)

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
DATE_TOKEN = re.compile(
    r"(?P<day>\d{4}-\d{2}-\d{2})|(?P<month>\d{4}-\d{2})(?![\d-])"
    r"|(?P<named>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})",
    re.IGNORECASE,
)
BBOX_RE = re.compile(r"\[\s*" + r"\s*,\s*".join([r"(-?\d+(?:\.\d+)?)"] * 4) + r"\s*\]")


# -- parameter extraction ---------------------------------------------------------------


def date_windows(text: str) -> list[tuple[date, date]]:
    """Date expressions -> (start, end) windows, in order of appearance.

    Four full dates are read as two explicit ranges. Otherwise a month ('2026-04',
    'April 2026') becomes that whole month and a single day becomes day +/- 7 days.
    """
    tokens: list[tuple[str, date]] = []
    for m in DATE_TOKEN.finditer(text):
        if m.group("day"):
            tokens.append(("day", date.fromisoformat(m.group("day"))))
        elif m.group("month"):
            year, month = map(int, m.group("month").split("-"))
            tokens.append(("month", date(year, month, 1)))
        else:
            name, year = m.group("named").replace(".", "").split()
            tokens.append(("month", date(int(year), MONTHS[name[:3].lower()], 1)))
    days = [d for kind, d in tokens if kind == "day"]
    if len(tokens) == 4 and len(days) == 4:
        return [(days[0], days[1]), (days[2], days[3])]
    windows = []
    for kind, d in tokens:
        if kind == "month":
            last = calendar.monthrange(d.year, d.month)[1]
            windows.append((d, d.replace(day=last)))
        else:
            windows.append((d - timedelta(days=7), d + timedelta(days=7)))
    return windows


def resolve_aoi(text: str) -> tuple[str, list[float]] | None:
    m = BBOX_RE.search(text)
    if m:
        return "custom bbox", [float(x) for x in m.groups()]
    q = text.lower()
    for aoi in config.AREAS_OF_INTEREST.values():
        if aoi.name in q or aoi.name.replace("-", " ") in q:
            return aoi.name, list(aoi.bbox)
    if any(word in q for word in ("lompoc", "vandenberg", "vsfb")):
        aoi = config.AREAS_OF_INTEREST["lompoc-valley-west"]
        return aoi.name, list(aoi.bbox)
    return None


def resolve_satellite(text: str) -> int:
    q = text.lower()
    m = re.search(r"(?:norad|catalog(?: number)?|catnr)\s*#?\s*(\d{1,5})", q)
    if m:
        return int(m.group(1))
    if "space station" in q and "chinese" not in q:
        return config.WELL_KNOWN_SATELLITES["iss"]
    for name, norad_id in config.WELL_KNOWN_SATELLITES.items():
        if re.search(rf"\b{name}\b", q):
            return norad_id
    return config.WELL_KNOWN_SATELLITES["iss"]


def choose_scene_pair(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict, dict]:
    """Least-cloudy 'before' scene, then the least-cloudy 'after' scene on the same tile."""
    if not before["scenes"]:
        raise ValueError("No Sentinel-2 scenes matched the 'before' window.")
    b = min(before["scenes"], key=lambda s: (s["cloud_cover_pct"], s["datetime_utc"]))
    same_tile = [s for s in after["scenes"] if s["mgrs_tile"] == b["mgrs_tile"]]
    if not same_tile:
        raise ValueError(f"No 'after' scene on tile {b['mgrs_tile']} matched the 'after' window.")
    a = min(same_tile, key=lambda s: (s["cloud_cover_pct"], s["datetime_utc"]))
    return b, a


# -- planner --------------------------------------------------------------------------------


class OfflinePlanner:
    def __init__(self, toolbox: Toolbox) -> None:
        self.toolbox = toolbox

    def ask(self, question: str) -> AgentResult:
        result = AgentResult(
            question=question,
            answer="",
            mode="offline-scripted",
            data_mode=self.toolbox.ctx.mode,
            notes=list(self.toolbox.ctx.notes),
        )
        intent = self.classify(question)
        handler = {
            "change": self._change,
            "scenes": self._scenes,
            "tle": self._tle,
            "passes": self._passes,
        }.get(intent)
        if handler is None:
            result.answer = (
                f"{OFFLINE_LABEL}\nThis question is outside what the offline "
                f"planner supports: {SUPPORTED}. Use the Claude agent "
                "(set ANTHROPIC_API_KEY) for open-ended questions."
            )
            return result
        try:
            body = handler(question, result)
        except (ToolError, ValueError) as exc:
            body = f"Could not answer: {exc}"
        result.answer = f"{OFFLINE_LABEL}\n{body}"
        return result

    @staticmethod
    def classify(question: str) -> str | None:
        q = question.lower()
        if re.search(r"\b(ndvi|nbr|vegetation|greenness|burn(ed)?|burn severity)\b", q):
            return "change"
        if re.search(r"\b(scenes?|imagery|images?|sentinel)\b", q):
            return "scenes"
        if re.search(r"\b(tle|two-line|orbital elements?|element sets?)\b", q):
            return "tle"
        if re.search(r"\b(pass(es)?|overhead|fly(ing)? over|flyover|visible|see|spot|rise)\b", q):
            return "passes"
        return None

    def _call(self, result: AgentResult, name: str, **arguments: Any) -> dict[str, Any]:
        call = ToolCall(name=name, arguments=arguments)
        result.tool_calls.append(call)
        try:
            call.result = call_tool(self.toolbox, name, arguments)
        except ToolError as exc:
            call.error = str(exc)
            raise
        return call.result

    # -- handlers ----------------------------------------------------------------------

    def _passes(self, question: str, result: AgentResult) -> str:
        q = question.lower()
        site = "vandenberg" if re.search(r"vandenberg|vsfb", q) else "lompoc"
        days = 3.0
        if "tonight" in q:
            days = 1.0
        elif "tomorrow" in q:
            days = 2.0
        elif "week" in q:
            days = 7.0
        m = re.search(r"next (\d+) days?", q)
        if m:
            days = float(min(int(m.group(1)), 10))
        m = re.search(r"above (\d+(?:\.\d+)?)\s*(?:°|deg)", q)
        min_el = float(m.group(1)) if m else 10.0
        visible_only = bool(re.search(r"\b(visible|see|spot|naked)\b", q))
        out = self._call(
            result,
            "predict_passes",
            norad_id=resolve_satellite(question),
            site=site,
            days=days,
            min_elevation_deg=min_el,
            visible_only=visible_only,
        )
        lines = [
            f"{out['satellite']} (NORAD {out['norad_id']}) over {out['site']['name']} "
            f"({out['site']['latitude']}, {out['site']['longitude']}), "
            f"{out['window_start_utc']} to {out['window_end_utc']}, "
            f"minimum elevation {out['min_elevation_deg']} deg.",
            f"TLE epoch {out['tle_epoch_utc']} (age {out['tle_age_days']} days, "
            f"{out['tle_source']}).",
        ]
        kind = "visible passes" if visible_only else "passes"
        lines.append(
            f"{len(out['passes'])} {kind} found ({out['passes_above_threshold']} passes "
            f"above {out['min_elevation_deg']} deg in total)."
        )
        for p in out["passes"]:
            line = (
                f"- Rise {p['rise_utc']} ({p['rise_local']}) at az {p['rise_azimuth_deg']} "
                f"deg; max elevation {p['max_elevation_deg']} deg at {p['culmination_utc']}; "
                f"set {p['set_utc']} at az {p['set_azimuth_deg']} deg"
            )
            if p["visible"]:
                line += f"; visible {p['visible_start_utc']} to {p['visible_end_utc']}"
            lines.append(line + ".")
        lines += [f"Warning: {w}" for w in out["warnings"]]
        lines.append(f"Visibility rule: {out['visibility_rule']}. Orbit data: CelesTrak.")
        return "\n".join(lines)

    def _tle(self, question: str, result: AgentResult) -> str:
        out = self._call(result, "get_tle", norad_id=resolve_satellite(question))
        return "\n".join(
            [
                f"{out['name']} (NORAD {out['norad_id']}), source {out['source']}, "
                f"retrieved {out['retrieved_at_utc']}:",
                out["line1"],
                out["line2"],
                f"Epoch {out['epoch_utc']} (age {out['age_days']} days).",
                *[f"Warning: {w}" for w in out["warnings"]],
            ]
        )

    def _scenes(self, question: str, result: AgentResult) -> str:
        aoi = resolve_aoi(question)
        windows = date_windows(question)
        if aoi is None or not windows:
            raise ValueError(
                "I need an area (a named AOI, Lompoc/Vandenberg, or "
                "[min_lon, min_lat, max_lon, max_lat]) and a month or date range."
            )
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", question)
        max_cloud = float(m.group(1)) if m else 20.0
        start, end = windows[0]
        out = self._call(
            result,
            "search_sentinel2_scenes",
            bbox=aoi[1],
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            max_cloud_cover=max_cloud,
            limit=20,
        )
        lines = [
            f"Sentinel-2 L2A scenes over {aoi[0]} {aoi[1]} from {out['start_date']} to "
            f"{out['end_date']} with scene cloud cover below {out['max_cloud_cover']}%: "
            f"{out['returned']} returned of {out['matched']} matched "
            f"({out['data_source']})."
        ]
        lines += [
            f"- {s['scene_id']}  {s['date']}  cloud {s['cloud_cover_pct']}%  tile {s['mgrs_tile']}"
            for s in out["scenes"]
        ]
        lines.append(f"Note: {out['note']}")
        return "\n".join(lines)

    def _change(self, question: str, result: AgentResult) -> str:
        q = question.lower()
        index = "nbr" if re.search(r"\b(nbr|burn)", q) else "ndvi"
        aoi = resolve_aoi(question)
        windows = date_windows(question)
        if aoi is None or len(windows) < 2:
            raise ValueError(
                "I need an area and two dates or date ranges (before and after), "
                "e.g. 'between 2026-04 and 2026-09'."
            )
        name, bbox = aoi
        searches = [
            self._call(
                result,
                "search_sentinel2_scenes",
                bbox=bbox,
                start_date=s.isoformat(),
                end_date=e.isoformat(),
                max_cloud_cover=20.0,
                limit=20,
            )
            for s, e in windows[:2]
        ]
        before, after = choose_scene_pair(*searches)
        out = self._call(
            result,
            "compare_index_between_scenes",
            before_scene_id=before["scene_id"],
            after_scene_id=after["scene_id"],
            bbox=bbox,
            index=index,
            change_threshold=0.1,
            render_png=True,
        )
        c, label = out["change"], index.upper()
        lines = [
            f"{label} change over {name} {bbox} ({out['data_source']}).",
            f"Before: {before['scene_id']}, acquired {before['date']}, scene cloud cover "
            f"{before['cloud_cover_pct']}% (least cloudy of {searches[0]['matched']} candidates).",
            f"After: {after['scene_id']}, acquired {after['date']}, scene cloud cover "
            f"{after['cloud_cover_pct']}% (least cloudy on tile {after['mgrs_tile']}).",
            f"Pixels clear in both scenes: {c['pixels_valid_both']} of {c['pixels_total']} "
            f"(valid fraction {c['valid_fraction']}), {out['grid']['pixel_size_m']} m pixels.",
            f"Mean {label}: {c['before']['mean']} before, {c['after']['mean']} after; "
            f"mean change {c['delta']['mean']}, median change {c['delta']['median']} "
            f"(10th to 90th percentile {c['delta']['p10']} to {c['delta']['p90']}).",
            f"Pixels with change beyond +/-{c['threshold']}: {c['pct_pixels_decreased']}% "
            f"decreased, {c['pct_pixels_increased']}% increased, {c['pct_pixels_stable']}% "
            "within the threshold.",
        ]
        if out["png_path"]:
            lines.append(f"Change map: {out['png_path']}")
        lines += [f"Warning: {w}" for w in out["warnings"]]
        reasons = sorted(
            set(out["offset_handling"]["before"].values())
            | set(out["offset_handling"]["after"].values())
        )
        lines += [f"Reflectance offset handling: {r}" for r in reasons]
        lines.append(out["attribution"])
        return "\n".join(lines)
