"""Run the golden-set evals.

    python -m evals.run_evals                  # offline: scripted planner + fixtures (CI)
    python -m evals.run_evals --live           # Claude agent on fixtures (needs a key)
    python -m evals.run_evals --live --data live

Offline mode checks the harness end to end (tool selection, grounding, citations) and
the pass-geometry reference. Its agent cases are deterministic, so they are a regression
test of the planner and tools, not a measure of model quality; the same cases become a
model eval when run with --live.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from orbitwatch.agent.grounding import check_grounding
from orbitwatch.agent.types import AgentResult
from orbitwatch.context import DataContext
from orbitwatch.tools import Toolbox, call_tool, input_model

HERE = Path(__file__).parent


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return abs(float(expected) - float(actual)) < 1e-6
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(map(_matches, expected, actual))
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() == actual.lower()
    return expected == actual


def _with_defaults(tool: str, args: dict) -> dict:
    try:
        return dict(input_model(tool).model_validate(args))
    except Exception:
        return args


def check_case(case: dict, result: AgentResult) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    failures = []
    used = result.tools_used
    for tool in case.get("expect_tools", []):
        if tool not in used:
            failures.append(f"expected tool {tool} was not called (called: {used})")
    for tool in case.get("forbid_tools", []):
        if tool in used:
            failures.append(f"forbidden tool {tool} was called")
    for tool, expected in (case.get("expect_args") or {}).items():
        calls = [_with_defaults(tool, c.arguments) for c in result.tool_calls if c.name == tool]
        if not any(all(_matches(v, c.get(k)) for k, v in expected.items()) for c in calls):
            failures.append(f"no {tool} call matched {expected}; got {calls}")
    for c in result.tool_calls:
        if c.error:
            failures.append(f"tool error in {c.name}: {c.error}")

    outputs = [c.result for c in result.tool_calls if c.result is not None]
    answer = result.answer
    checks = case.get("checks", [])
    if "grounded" in checks:
        report = check_grounding(answer, outputs, case["question"])
        if not report.ok:
            failures.append(
                f"ungrounded: numbers={report.ungrounded_numbers} "
                f"dates={report.ungrounded_dates} times={report.ungrounded_times}"
            )
    if "cites_tle_epoch" in checks:
        epochs = [o["tle_epoch_utc"] for o in outputs if "tle_epoch_utc" in o]
        if not epochs or not any(e[:10] in answer and e[11:16] in answer for e in epochs):
            failures.append(f"TLE epoch {epochs} not cited in the answer")
    if "cites_scenes" in checks:
        compares = [o for o in outputs if "before_scene" in o]
        if not compares:
            failures.append("no compare_index_between_scenes result to cite")
        for o in compares:
            for side in ("before_scene", "after_scene"):
                sc = o[side]
                if sc["scene_id"] not in answer or sc["date"] not in answer:
                    failures.append(f"{side} {sc['scene_id']} / {sc['date']} not cited")
    return failures


def check_reference(case: dict, toolbox: Toolbox) -> list[str]:
    ref = json.loads((HERE / case["reference"]).read_text(encoding="utf-8"))
    tol = case["tolerance"]
    out = call_tool(toolbox, case["tool"], case["args"])
    got, want = out["passes"], ref["passes"]
    if len(got) != len(want):
        return [f"pass count {len(got)} != reference {len(want)}"]

    def secs(stamp: str) -> float:
        return datetime.fromisoformat(stamp.replace(" UTC", "+00:00")).timestamp()

    failures = []
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        for key in ("rise_utc", "culmination_utc", "set_utc"):
            dt = abs(secs(g[key]) - secs(w[key]))
            if dt > tol["time_s"]:
                failures.append(f"pass {i} {key} off by {dt:.1f} s")
        if abs(g["max_elevation_deg"] - w["max_elevation_deg"]) > tol["elevation_deg"] + 0.05:
            failures.append(
                f"pass {i} max elevation {g['max_elevation_deg']} vs {w['max_elevation_deg']}"
            )  # +0.05: tool rounds to 0.1 deg
        for key in ("rise_azimuth_deg", "set_azimuth_deg"):
            diff = abs((g[key] - w[key] + 180) % 360 - 180)
            if diff > tol["azimuth_deg"] + 0.05:
                failures.append(f"pass {i} {key} {g[key]} vs {w[key]}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use the Claude agent")
    parser.add_argument("--data", choices=["fixtures", "live"], default="fixtures")
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", help="run a single case id")
    args = parser.parse_args(argv)

    golden = yaml.safe_load((HERE / "golden.yaml").read_text(encoding="utf-8"))
    ctx = DataContext.live() if args.data == "live" else DataContext.fixtures()
    toolbox = Toolbox(ctx)
    if args.live:
        from orbitwatch.agent.claude_agent import ClaudeAgent

        agent = ClaudeAgent(toolbox, model=args.model)
        mode = f"claude ({agent.model})"
    else:
        from orbitwatch.agent.planner import OfflinePlanner

        agent = OfflinePlanner(toolbox)
        mode = "offline scripted planner"
    print(f"orbitwatch evals | agent: {mode} | data: {ctx.mode}\n")

    rows = []
    for case in golden["cases"]:
        if args.only and case["id"] != args.only:
            continue
        result = agent.ask(case["question"])
        failures = check_case(case, result)
        rows.append(
            {
                "id": case["id"],
                "passed": not failures,
                "failures": failures,
                "tools": result.tools_used,
                "answer": result.answer,
            }
        )
    for case in golden.get("reference_cases", []):
        if args.only and case["id"] != args.only:
            continue
        failures = check_reference(case, Toolbox(DataContext.fixtures()))
        rows.append(
            {
                "id": case["id"],
                "passed": not failures,
                "failures": failures,
                "tools": [case["tool"]],
                "answer": None,
            }
        )

    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status}  {row['id']:<40} tools={row['tools']}")
        for f in row["failures"]:
            print(f"        - {f}")
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "latest.json").write_text(
        json.dumps({"mode": mode, "data": ctx.mode, "rows": rows}, indent=1, default=str),
        encoding="utf-8",
    )
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
