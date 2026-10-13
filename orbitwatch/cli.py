"""Command-line interface: ``orbitwatch <command>``.

Every command calls the same ``Toolbox`` methods the MCP server and agents use.
``--data fixtures`` runs entirely offline against recorded fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from orbitwatch import config
from orbitwatch.context import DataContext
from orbitwatch.tools import Toolbox, ToolError, call_tool


def _toolbox(args: argparse.Namespace) -> Toolbox:
    return Toolbox(DataContext.fixtures() if args.data == "fixtures" else DataContext.live())


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _bbox(args: argparse.Namespace) -> list[float]:
    if args.bbox:
        return list(args.bbox)
    if args.aoi not in config.AREAS_OF_INTEREST:
        raise SystemExit(f"Unknown AOI {args.aoi!r}; known: {sorted(config.AREAS_OF_INTEREST)}")
    return list(config.AREAS_OF_INTEREST[args.aoi].bbox)


def _window(text: str) -> tuple[str, str]:
    start, _, end = text.partition("/")
    if not end:
        raise SystemExit(f"Expected START/END (YYYY-MM-DD/YYYY-MM-DD), got {text!r}")
    return start, end


def _satellite(value: str) -> int:
    return int(value) if value.isdigit() else config.WELL_KNOWN_SATELLITES[value.lower()]


# -- commands ---------------------------------------------------------------------------------


def cmd_places(args) -> int:
    _print_json(_toolbox(args).list_places())
    return 0


def cmd_tle(args) -> int:
    out = call_tool(_toolbox(args), "get_tle", {"norad_id": _satellite(args.satellite)})
    if args.json:
        _print_json(out)
    else:
        print(f"{out['name']}  (NORAD {out['norad_id']}, {out['source']})")
        print(out["line1"])
        print(out["line2"])
        print(f"epoch {out['epoch_utc']}  age {out['age_days']} days")
    return 0


def cmd_passes(args) -> int:
    out = call_tool(
        _toolbox(args),
        "predict_passes",
        {
            "norad_id": _satellite(args.satellite),
            "site": args.site,
            "days": args.days,
            "min_elevation_deg": args.min_el,
            "visible_only": args.visible_only,
            "latitude": args.lat,
            "longitude": args.lon,
        },
    )
    if args.json:
        _print_json(out)
        return 0
    s = out["site"]
    print(
        f"{out['satellite']} (NORAD {out['norad_id']}) over {s['name']} "
        f"({s['latitude']}, {s['longitude']}, {s['elevation_m']} m)"
    )
    print(
        f"TLE epoch {out['tle_epoch_utc']} (age {out['tle_age_days']} d) from {out['tle_source']}"
    )
    print(
        f"Window {out['window_start_utc']} -> {out['window_end_utc']}, "
        f"min elevation {out['min_elevation_deg']} deg; {out['passes_above_threshold']} passes, "
        f"{out['visible_pass_count']} visible\n"
    )
    tz = out["passes"][0]["rise_local"].split()[-1] if out["passes"] else ""
    print(
        f"{'date':<11}{'rise':>9}{'peak':>9}{'set':>9}  ({tz}){'max el':>8}{'rise az':>8}"
        f"{'set az':>8}  visible ({tz})"
    )

    def hms(stamp: str | None) -> str:
        return stamp.split()[1] if stamp else ""

    for p in out["passes"]:
        if p["visible"]:
            vis = f"{hms(p['visible_start_local'])}-{hms(p['visible_end_local'])}"
        else:
            vis = "no" if p["visible"] is False else "?"
        print(
            f"{p['rise_local'].split()[0]:<11}{hms(p['rise_local']):>9}"
            f"{hms(p['culmination_local']):>9}{hms(p['set_local']):>9}  {'':<{len(tz) + 2}}"
            f"{p['max_elevation_deg']:>8}{p['rise_azimuth_deg']:>8}{p['set_azimuth_deg']:>8}  {vis}"
        )
    for w in out["warnings"]:
        print(f"warning: {w}")
    return 0


def cmd_starlink(args) -> int:
    toolbox = _toolbox(args)
    where = {"site": args.site, "latitude": args.lat, "longitude": args.lon}
    if args.min_el is None:
        args.min_el = 30.0 if args.what == "passes" else 10.0
    if args.what == "passes":
        out = call_tool(
            toolbox,
            "starlink_visible_passes",
            {
                **where,
                "hours": args.hours,
                "min_elevation_deg": args.min_el,
                "max_results": args.top,
            },
        )
    else:
        out = call_tool(
            toolbox,
            "find_starlink_trains",
            {**where, "days": args.days, "min_elevation_deg": args.min_el},
        )
    if args.json:
        _print_json(out)
        return 0
    s = out["site"]
    print(
        f"Starlink ({out['source']}, retrieved {out['retrieved_at_utc']}) over {s['name']} "
        f"({s['latitude']}, {s['longitude']})"
    )
    print(f"Window {out['window_start_utc']} -> {out['window_end_utc']}")
    if args.what == "passes":
        print(
            f"{out['objects_screened']} of {out['objects_in_catalog']} objects screened; "
            f"{out['candidates_screened']} candidates, {out['candidates_checked_exactly']} checked "
            f"exactly; {out['visible_passes_confirmed']} visible passes above "
            f"{out['min_elevation_deg']} deg; closest {out['returned']}:\n"
        )
        print(
            f"{'name':<22}{'launch':>10}{'alt km':>8}{'range':>7}  {'rise (local)':<24}"
            f"{'max el':>7}  visible"
        )
        for p in out["passes"]:
            vis = f"{p['visible_start_local'][11:19]}-{p['visible_end_local'][11:19]}"
            print(
                f"{p['name']:<22}{p['launch']:>10}{p['mean_altitude_km']:>8}"
                f"{p['range_at_culmination_km']:>7}  {p['rise_local']:<24}"
                f"{p['max_elevation_deg']:>7}  {vis}"
            )
        print(f"\nnote: {out['brightness_note']}")
    else:
        print(f"{out['train_count']} train(s) found (criteria {out['criteria']})")
        for t in out["trains"]:
            print(
                f"\nlaunch {t['launch']}: {t['objects_in_train']} objects, "
                f"lead {t['lead']['name']}, tail {t['tail']['name']}, "
                f"span {t['along_track_span_km']} km "
                f"(~{t['passage_estimate_s']} s to pass), {t['visible_pass_count']} visible passes"
            )
            for p in t["passes"]:
                vis = "VISIBLE" if p["train_visible"] else "not visible"
                print(f"  {p['rise_local']}  max el {p['max_elevation_deg']}  {vis}")
        print(
            f"\n{out['low_orbit_launch_count']} launches have objects below the "
            "train altitude; most recent:"
        )
        for x in out["most_recent_low_orbit_launches"]:
            print(
                f"  {x['launch']}: {x['objects_in_low_orbit']} objects at "
                f"~{x['median_altitude_km']} km, largest cluster {x['largest_cluster']}, "
                f"median neighbour {x['median_neighbour_distance_km']} km"
            )
    for w in out["warnings"]:
        print(f"warning: {w}")
    return 0


def cmd_scenes(args) -> int:
    start, end = _window(args.window)
    out = call_tool(
        _toolbox(args),
        "search_sentinel2_scenes",
        {
            "bbox": _bbox(args),
            "start_date": start,
            "end_date": end,
            "max_cloud_cover": args.max_cloud,
            "limit": args.limit,
        },
    )
    if args.json:
        _print_json(out)
        return 0
    print(f"{out['returned']} of {out['matched']} scenes ({out['data_source']})")
    for sc in out["scenes"]:
        print(
            f"  {sc['scene_id']:<28} {sc['datetime_utc']}  cloud {sc['cloud_cover_pct']:>6}%  "
            f"tile {sc['mgrs_tile']}  baseline {sc['processing_baseline']}"
        )
    return 0


def cmd_change(args) -> int:
    from orbitwatch.agent.planner import choose_scene_pair

    toolbox = _toolbox(args)
    bbox = _bbox(args)
    if args.before_scene and args.after_scene:
        before_id, after_id = args.before_scene, args.after_scene
    else:
        searches = []
        for window in (args.before, args.after):
            start, end = _window(window)
            searches.append(
                call_tool(
                    toolbox,
                    "search_sentinel2_scenes",
                    {
                        "bbox": bbox,
                        "start_date": start,
                        "end_date": end,
                        "max_cloud_cover": args.max_cloud,
                        "limit": 50,
                    },
                )
            )
        before, after = choose_scene_pair(*searches)
        before_id, after_id = before["scene_id"], after["scene_id"]
    out = call_tool(
        toolbox,
        "compare_index_between_scenes",
        {
            "before_scene_id": before_id,
            "after_scene_id": after_id,
            "bbox": bbox,
            "index": args.index,
            "change_threshold": args.threshold,
            "render_png": True,
        },
    )
    if args.out and out["png_path"]:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out["png_path"], args.out)
        out["png_path"] = str(args.out)
    _print_json(out)
    return 0


def cmd_ask(args) -> int:
    from orbitwatch.observability.tracing import NULL_TRACER, Tracer

    toolbox = _toolbox(args)
    tracer = NULL_TRACER if args.no_trace else Tracer(args.trace or config.trace_file())
    if args.offline:
        from orbitwatch.agent.planner import OfflinePlanner

        result = OfflinePlanner(toolbox, tracer=tracer).ask(args.question)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "No ANTHROPIC_API_KEY set. Re-run with --offline to use the scripted "
                "planner (no LLM).",
                file=sys.stderr,
            )
            return 2
        from orbitwatch.agent.claude_agent import AgentError, ClaudeAgent

        try:
            agent = ClaudeAgent(toolbox, model=args.model, effort=args.effort, tracer=tracer)
            result = agent.ask(args.question)
        except AgentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.json:
        _print_json(result.to_dict())
        return 0
    header = "Claude agent" if result.mode == "claude" else "offline scripted planner"
    print(f"[mode: {header} | data: {result.data_mode}]")
    for note in result.notes:
        print(f"[note: {note}]")
    if args.verbose:
        for call in result.tool_calls:
            status = "error" if call.error else "ok"
            print(f"[tool] {call.name}({json.dumps(call.arguments)}) -> {status}")
    print()
    print(result.answer)
    if args.verbose and tracer.path:
        cost = "unknown" if result.cost_usd is None else f"${result.cost_usd:.4f}"
        print(
            f"\n[trace {result.trace_id} -> {tracer.path} | llm calls {result.llm_calls}, "
            f"est. cost {cost}, grounded {result.grounding_ok}]"
        )
    return 0


def cmd_trace(args) -> int:
    from orbitwatch.observability import report
    from orbitwatch.observability.tracing import load_spans

    path = Path(args.file) if args.file else config.trace_file()
    if not path.exists():
        print(f"No trace file at {path}. Run `orbitwatch ask ...` first.", file=sys.stderr)
        return 1
    spans = load_spans(path)
    summary = report.summarize(spans)
    if args.json:
        _print_json(summary)
    else:
        print(f"trace file: {path} ({len(spans)} spans)")
        print(report.format_text(summary))
    if args.png or args.html:
        png = report.render_png(summary, spans)
        if args.png:
            Path(args.png).write_bytes(png)
            print(f"wrote {args.png}")
        if args.html:
            Path(args.html).write_text(report.render_html(summary, png, str(path)), "utf-8")
            print(f"wrote {args.html}")
    return 0


def cmd_mcp(args) -> int:
    from orbitwatch.mcp_server import main as mcp_main

    return mcp_main(["--data", args.data])


# -- parser ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orbitwatch",
        description="An AI analyst for orbital and Earth-observation data.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument(
            "--data",
            choices=["live", "fixtures"],
            default="live",
            help="live services (default) or offline recorded fixtures",
        )
        sp.set_defaults(func=func)
        return sp

    add("places", cmd_places, "list named ground sites and areas of interest")

    sp = add("tle", cmd_tle, "show the current TLE for a satellite")
    sp.add_argument("satellite", nargs="?", default="25544", help="NORAD id or name (iss)")
    sp.add_argument("--json", action="store_true")

    sp = add("passes", cmd_passes, "predict satellite passes over a site")
    sp.add_argument("--sat", dest="satellite", default="25544", help="NORAD id or name (iss)")
    sp.add_argument("--site", default=config.DEFAULT_SITE)
    sp.add_argument("--lat", type=float)
    sp.add_argument("--lon", type=float)
    sp.add_argument("--days", type=float, default=3.0)
    sp.add_argument("--min-el", type=float, default=10.0)
    sp.add_argument("--visible-only", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = add("starlink", cmd_starlink, "visible Starlink passes, or launch trains")
    sp.add_argument("what", choices=["passes", "trains"])
    sp.add_argument("--site", default=config.DEFAULT_SITE)
    sp.add_argument("--lat", type=float)
    sp.add_argument("--lon", type=float)
    sp.add_argument("--hours", type=float, default=24.0, help="passes: window length")
    sp.add_argument("--days", type=float, default=3.0, help="trains: pass-prediction window")
    sp.add_argument("--min-el", type=float, default=None, help="default 30 (passes) / 10 (trains)")
    sp.add_argument("--top", type=int, default=10, help="passes: how many to show")
    sp.add_argument("--json", action="store_true")

    def add_aoi(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--aoi", default="lompoc-valley-west")
        sp.add_argument(
            "--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT")
        )
        sp.add_argument("--max-cloud", type=float, default=20.0)

    sp = add("scenes", cmd_scenes, "search Sentinel-2 L2A scenes")
    add_aoi(sp)
    sp.add_argument("--window", required=True, help="START/END, e.g. 2026-04-01/2026-04-30")
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--json", action="store_true")

    sp = add("change", cmd_change, "NDVI/NBR change between two dates")
    add_aoi(sp)
    sp.add_argument("--before", help="START/END window for the before scene")
    sp.add_argument("--after", help="START/END window for the after scene")
    sp.add_argument("--before-scene")
    sp.add_argument("--after-scene")
    sp.add_argument("--index", choices=["ndvi", "nbr"], default="ndvi")
    sp.add_argument("--threshold", type=float, default=0.1)
    sp.add_argument("--out", help="copy the PNG change map here")

    sp = add("ask", cmd_ask, "ask a question (Claude agent, or --offline scripted planner)")
    sp.add_argument("question")
    sp.add_argument(
        "--offline",
        action="store_true",
        help="no LLM: answer with the deterministic scripted planner",
    )
    sp.add_argument("--model", default=None, help=f"default {config.DEFAULT_MODEL}")
    sp.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    sp.add_argument("--json", action="store_true")
    sp.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    sp.add_argument("--trace", help="append the run's spans to this JSONL file")
    sp.add_argument("--no-trace", action="store_true", help="do not record a trace")

    sp = sub.add_parser("trace", help="summarise recorded agent traces")
    sp.add_argument("action", choices=["summary"])
    sp.add_argument("--file", help="JSONL trace file (default: the ask trace file)")
    sp.add_argument("--html", help="also write a static HTML report here")
    sp.add_argument("--png", help="also write the latency chart PNG here")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_trace)

    add("mcp", cmd_mcp, "run the MCP server over stdio")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    has_scenes = args.command == "change" and args.before_scene and args.after_scene
    if args.command == "change" and not has_scenes and not (args.before and args.after):
        raise SystemExit("change needs --before and --after windows (or explicit scene ids)")
    try:
        return args.func(args)
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
