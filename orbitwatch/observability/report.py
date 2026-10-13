"""Aggregate a JSONL trace file into run-level quality, latency and cost metrics.

    orbitwatch trace summary                       # default trace file
    orbitwatch trace summary --file evals/results/traces.jsonl --html report.html

Metrics (all computed from spans, nothing sampled):

* runs by agent mode; p50/p95 latency of runs, LLM calls and each tool;
* tokens and estimated cost (see ``pricing.py``; offline runs cost 0, unpriced = unknown);
* tool error rate = errored ``tool.call`` spans / all ``tool.call`` spans;
* grounding pass rate = runs whose answer passed the grounding check / runs checked;
* judge results, when the trace came from an eval run with a judge.
"""

from __future__ import annotations

import base64
import html
import io
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default); None for no data."""
    return round(float(np.percentile(values, q)), 1) if values else None


def _latency(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95)}


def summarize(spans: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_name[s["name"]].append(s)
    runs = by_name["agent.run"]
    tools = by_name["tool.call"]
    llm = by_name["llm.call"]
    judges = by_name["eval.judge"]

    per_tool: dict[str, list[dict]] = defaultdict(list)
    for s in tools:
        per_tool[s["attributes"].get("gen_ai.tool.name", "?")].append(s)
    tool_errors = [s for s in tools if s["status"]["code"] == "ERROR"]

    costs = [s["attributes"].get("orbitwatch.cost_usd") for s in runs]
    known = [c for c in costs if c is not None]
    tokens: Counter = Counter()
    for s in llm:
        for key, value in s["attributes"].items():
            if key.startswith("gen_ai.usage."):
                tokens[key.removeprefix("gen_ai.usage.")] += value or 0
    grounded = [
        s["attributes"]["orbitwatch.grounding.ok"]
        for s in runs
        if s["attributes"].get("orbitwatch.grounding.ok") is not None
    ]
    judge_ok = [s["attributes"].get("orbitwatch.judge.pass") for s in judges]
    judge_ok = [j for j in judge_ok if j is not None]

    return {
        "traces": len({s["trace_id"] for s in spans}),
        "runs": len(runs),
        "runs_by_mode": dict(Counter(s["attributes"].get("orbitwatch.agent.mode") for s in runs)),
        "run_errors": sum(s["status"]["code"] == "ERROR" for s in runs),
        "stop_reasons": dict(Counter(s["attributes"].get("orbitwatch.stop_reason") for s in runs)),
        "latency": {
            "run": _latency([s["duration_ms"] for s in runs]),
            "llm_call": _latency([s["duration_ms"] for s in llm]),
            "tool_call": _latency([s["duration_ms"] for s in tools]),
            "per_tool": {
                name: {
                    **_latency([s["duration_ms"] for s in group]),
                    "errors": sum(s["status"]["code"] == "ERROR" for s in group),
                }
                for name, group in sorted(per_tool.items())
            },
        },
        "llm_calls": len(llm),
        "tokens": dict(tokens),
        "cost_usd_total": round(sum(known), 6) if known else 0.0,
        "cost_usd_mean_per_run": round(sum(known) / len(known), 6) if known else None,
        "runs_with_unknown_cost": sum(c is None for c in costs),
        "tool_calls": len(tools),
        "tool_errors": len(tool_errors),
        "tool_error_rate": round(len(tool_errors) / len(tools), 4) if tools else None,
        "tool_error_types": dict(
            Counter(s["attributes"].get("error.type", "?") for s in tool_errors)
        ),
        "grounding_checked": len(grounded),
        "grounding_pass_rate": round(sum(grounded) / len(grounded), 4) if grounded else None,
        "judge_runs": len(judges),
        "judge_pass_rate": round(sum(judge_ok) / len(judge_ok), 4) if judge_ok else None,
        "judge_cost_usd": round(
            sum(s["attributes"].get("orbitwatch.cost_usd") or 0 for s in judges), 6
        ),
    }


def format_text(summary: dict[str, Any]) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lat = summary["latency"]
    lines = [
        f"runs: {summary['runs']} {summary['runs_by_mode']}  (run errors: {summary['run_errors']})",
        f"stop reasons: {summary['stop_reasons']}",
        f"run latency:  p50 {lat['run']['p50_ms']} ms   p95 {lat['run']['p95_ms']} ms",
        f"LLM calls: {summary['llm_calls']}  p50 {lat['llm_call']['p50_ms']} ms   "
        f"p95 {lat['llm_call']['p95_ms']} ms",
        f"tokens: {summary['tokens'] or 'none'}",
        f"estimated cost: ${summary['cost_usd_total']:.4f} total, "
        f"mean/run {summary['cost_usd_mean_per_run']} "
        f"(unknown-price runs: {summary['runs_with_unknown_cost']}; prices in pricing.py)",
        f"tool calls: {summary['tool_calls']}, errors {summary['tool_errors']} "
        f"(rate {pct(summary['tool_error_rate'])}) {summary['tool_error_types']}",
        f"grounding pass rate: {pct(summary['grounding_pass_rate'])} "
        f"of {summary['grounding_checked']} checked runs",
    ]
    if summary["judge_runs"]:
        lines.append(
            f"judge: {summary['judge_runs']} gradings, pass rate "
            f"{pct(summary['judge_pass_rate'])}, judge cost ${summary['judge_cost_usd']:.4f}"
        )
    lines.append("per tool:           n    p50 ms    p95 ms  errors")
    for name, t in lat["per_tool"].items():
        lines.append(f"  {name:<30}{t['n']:>3}{t['p50_ms']:>10}{t['p95_ms']:>10}{t['errors']:>8}")
    return "\n".join(lines)


# -- PNG + HTML ---------------------------------------------------------------------------

# Reference categorical palette (dataviz skill): slots 1 and 2, fixed order.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def render_png(summary: dict[str, Any], spans: list[dict[str, Any]]) -> bytes:
    """Two panels: run latency by agent mode (dots), per-tool p50/p95 latency (bars)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [s for s in spans if s["name"] == "agent.run"]
    modes = sorted({s["attributes"].get("orbitwatch.agent.mode") or "?" for s in runs})
    per_tool = summary["latency"]["per_tool"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)

    for i, mode in enumerate(modes):
        ms = [
            s["duration_ms"]
            for s in runs
            if (s["attributes"].get("orbitwatch.agent.mode") or "?") == mode
        ]
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(ms))
        a.scatter(
            ms,
            np.full(len(ms), i) + jitter,
            s=28,
            color=SERIES[i % 2],
            alpha=0.85,
            edgecolors=SURFACE,
            linewidths=1.0,
            label=mode,
        )
        p50 = percentile(ms, 50)
        a.plot([p50, p50], [i - 0.3, i + 0.3], color=INK, linewidth=2)
    a.set_yticks(range(len(modes)), modes)
    a.set_xscale("log")  # one Starlink screen next to sub-second runs: log keeps both readable
    a.set_xlabel("run latency (ms, log scale); black tick = p50", color=INK_2, fontsize=9)
    a.set_title("Agent runs", loc="left", color=INK, fontsize=11)

    names = list(per_tool)
    y = np.arange(len(names))
    b.barh(
        y - 0.18,
        [per_tool[n]["p50_ms"] for n in names],
        height=0.34,
        color=SERIES[0],
        label="p50",
        edgecolor=SURFACE,
        linewidth=1,
    )
    b.barh(
        y + 0.18,
        [per_tool[n]["p95_ms"] for n in names],
        height=0.34,
        color=SERIES[1],
        label="p95",
        edgecolor=SURFACE,
        linewidth=1,
    )
    b.set_yticks(y, names)
    b.invert_yaxis()
    b.set_xscale("log")
    b.set_xlabel("tool latency (ms, log scale)", color=INK_2, fontsize=9)
    b.set_title("Tool calls", loc="left", color=INK, fontsize=11)
    b.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def render_html(summary: dict[str, Any], png: bytes, source: str) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lat = summary["latency"]
    tiles = [
        ("Runs", str(summary["runs"])),
        ("Run latency p50 / p95", f"{lat['run']['p50_ms']} / {lat['run']['p95_ms']} ms"),
        ("Estimated cost", f"${summary['cost_usd_total']:.4f}"),
        ("Tool error rate", pct(summary["tool_error_rate"])),
        ("Grounding pass rate", pct(summary["grounding_pass_rate"])),
    ]
    if summary["judge_runs"]:
        tiles.append(("Judge pass rate", pct(summary["judge_pass_rate"])))
    tile_html = "".join(
        f'<div class="tile"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
        for k, v in tiles
    )
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{t['n']}</td><td>{t['p50_ms']}</td>"
        f"<td>{t['p95_ms']}</td><td>{t['errors']}</td></tr>"
        for n, t in lat["per_tool"].items()
    )
    img = base64.b64encode(png).decode("ascii")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>orbitwatch traces</title>
<style>
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK_2}; --line:{GRID}; }}
body {{ background:var(--surface); color:var(--ink); margin:0; padding:16px;
       font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1000px; margin:0 auto; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }}
.tile {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.k {{ color:var(--ink2); font-size:12px; }} .v {{ font-size:20px; font-weight:600; }}
img {{ max-width:100%; height:auto; margin:16px 0; }}
table {{ border-collapse:collapse; width:100%; }}
td, th {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:right; }}
td:first-child, th:first-child {{ text-align:left; }}
.note {{ color:var(--ink2); font-size:12px; }}
</style></head><body><main>
<h1>orbitwatch trace report</h1>
<p class="note">Source: {html.escape(source)}. Costs are estimates from
orbitwatch/observability/pricing.py; check them against Anthropic's current pricing.</p>
<div class="tiles">{tile_html}</div>
<img alt="Run latency by agent mode, and p50/p95 latency per tool"
 src="data:image/png;base64,{img}">
<h2>Per tool</h2>
<table><tr><th>tool</th><th>calls</th><th>p50 ms</th><th>p95 ms</th><th>errors</th></tr>
{rows}</table>
<h2>Raw summary</h2><pre>{html.escape(format_text(summary))}</pre>
</main></body></html>
"""
