"""Run the golden-set and red-team evals.

    python -m evals.run_evals                     # offline: scripted planner + fixtures (CI)
    python -m evals.run_evals --suite redteam     # just the red-team cases
    python -m evals.run_evals --judge claude      # LLM-as-judge (needs ANTHROPIC_API_KEY)
    python -m evals.run_evals --live              # Claude agent on fixtures (needs a key)
    python -m evals.run_evals --live --data live

Offline mode checks the harness end to end (tool selection, grounding, citations,
red-team refusals) and the pass-geometry reference. Its agent cases are deterministic,
so they are a regression test of the planner and tools, not a measure of model quality;
the same cases become a model eval when run with --live. The default judge is a
deterministic stub; ``--judge claude`` grades with a model and is skipped cleanly when no
API key is set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from evals.judge import make_judge, traced_grade
from evals.redteam import context_for
from orbitwatch.agent.grounding import check_grounding
from orbitwatch.agent.observe import tool_sources
from orbitwatch.agent.types import AgentResult
from orbitwatch.context import DataContext
from orbitwatch.observability.tracing import Tracer
from orbitwatch.tools import Toolbox, call_tool, input_model

HERE = Path(__file__).parent
SUITES = {"golden": "golden.yaml", "redteam": "redteam.yaml"}


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
    if not case.get("allow_tool_errors"):
        for c in result.tool_calls:
            if c.error:
                failures.append(f"tool error in {c.name}: {c.error}")

    outputs = [c.result for c in result.tool_calls if c.result is not None]
    answer = result.answer
    for pattern in case.get("expect_regex", []):
        if not re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer does not match /{pattern}/")
    for pattern in case.get("forbid_regex", []):
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer matches forbidden /{pattern}/")
    checks = case.get("checks", [])
    if "grounded" in checks:
        report = check_grounding(answer, tool_sources(result), case["question"])
        if not report.ok:
            failures.append(
                f"ungrounded: numbers={report.ungrounded_numbers} "
                f"dates={report.ungrounded_dates} times={report.ungrounded_times}"
            )
    if "cites_tle_epoch" in checks:
        epochs = [o["tle_epoch_utc"] for o in outputs if "tle_epoch_utc" in o]
        if not epochs or not any(e[:10] in answer and e[11:16] in answer for e in epochs):
            failures.append(f"TLE epoch {epochs} not cited in the answer")
    if "cites_element_epochs" in checks:
        # Starlink: every satellite quoted must come with its own element-set epoch.
        epochs = [
            p["elements_epoch_utc"]
            for o in outputs
            for p in o.get("passes", [])
            if "elements_epoch_utc" in p
        ]
        if not epochs:
            failures.append("no Starlink passes with element epochs to cite")
        missing = [e for e in epochs if e not in answer]
        if missing:
            failures.append(f"element epochs not cited: {missing}")
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


def load_cases(suite: str) -> tuple[list[dict], list[dict]]:
    """(agent cases tagged with their suite, reference cases) for 'golden'/'redteam'/'all'."""
    cases, references = [], []
    for name, filename in SUITES.items():
        if suite not in (name, "all"):
            continue
        data = yaml.safe_load((HERE / filename).read_text(encoding="utf-8"))
        cases += [{**c, "suite": name} for c in data["cases"]]
        references += [{**c, "suite": name} for c in data.get("reference_cases", [])]
    return cases, references


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use the Claude agent")
    parser.add_argument("--data", choices=["fixtures", "live"], default="fixtures")
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", choices=["all", "golden", "redteam"], default="all")
    parser.add_argument("--judge", choices=["stub", "claude", "none"], default="stub")
    parser.add_argument("--judge-model", default=None, help="default claude-sonnet-5-5")
    parser.add_argument("--only", help="run a single case id")
    args = parser.parse_args(argv)

    cases, references = load_cases(args.suite)
    base_ctx = DataContext.live() if args.data == "live" else DataContext.fixtures()
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    trace_path = results_dir / "traces.jsonl"
    trace_path.unlink(missing_ok=True)  # one file per eval run
    tracer = Tracer(trace_path)
    judge, judge_line = make_judge(args.judge, args.judge_model)

    if args.live:
        from orbitwatch.agent.claude_agent import ClaudeAgent

        def make_agent(toolbox: Toolbox):
            return ClaudeAgent(toolbox, model=args.model, tracer=tracer)

        mode = f"claude ({make_agent(Toolbox(base_ctx)).model})"
    else:
        from orbitwatch.agent.planner import OfflinePlanner

        def make_agent(toolbox: Toolbox):
            return OfflinePlanner(toolbox, tracer=tracer)

        mode = "offline scripted planner"
    print(f"orbitwatch evals | agent: {mode} | data: {base_ctx.mode} | {judge_line}\n")

    rows = []
    for case in cases:
        if args.only and case["id"] != args.only:
            continue
        agent = make_agent(Toolbox(context_for(case.get("data"), base_ctx)))
        result = agent.ask(case["question"])
        failures = check_case(case, result)
        verdict = traced_grade(judge, case, result, tracer) if judge else None
        if verdict and not verdict.passed:
            failures.append(
                f"judge ({verdict.judge}) failed: {verdict.error or verdict.scores} "
                f"{verdict.rationale}".strip()
            )
        rows.append(
            {
                "id": case["id"],
                "suite": case["suite"],
                "category": case.get("category", "golden"),
                "passed": not failures,
                "failures": failures,
                "tools": result.tools_used,
                "answer": result.answer,
                "trace_id": result.trace_id,
                "judge": verdict.__dict__ if verdict else None,
            }
        )
    for case in references:
        if args.only and case["id"] != args.only:
            continue
        failures = check_reference(case, Toolbox(DataContext.fixtures()))
        rows.append(
            {
                "id": case["id"],
                "suite": case["suite"],
                "category": "reference",
                "passed": not failures,
                "failures": failures,
                "tools": [case["tool"]],
                "answer": None,
                "trace_id": None,
                "judge": None,
            }
        )

    by_category: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status}  {row['suite']:<8}{row['id']:<38} tools={row['tools']}")
        for f in row["failures"]:
            print(f"        - {f}")
        by_category[row["category"]].append(row["passed"])
    passed = sum(r["passed"] for r in rows)
    print("\nby category: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in by_category.items()))
    print(f"{passed}/{len(rows)} passed")

    print(f"traces: {trace_path} (orbitwatch trace summary --file {trace_path.as_posix()})")
    (results_dir / "latest.json").write_text(
        json.dumps(
            {"mode": mode, "data": base_ctx.mode, "judge": judge_line, "rows": rows},
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
