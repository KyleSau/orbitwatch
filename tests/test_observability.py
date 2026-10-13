"""Tracing spans, cost estimates and the trace summary report."""

import json
from types import SimpleNamespace as NS

import pytest

from orbitwatch.agent.claude_agent import ClaudeAgent
from orbitwatch.agent.planner import OfflinePlanner
from orbitwatch.cli import main as cli_main
from orbitwatch.observability import report
from orbitwatch.observability.pricing import estimate_cost_usd
from orbitwatch.observability.tracing import Tracer, load_spans

from .test_claude_agent import FakeClient, text, tool_use


def test_spans_nest_and_are_written_as_jsonl(tmp_path):
    path = tmp_path / "t.jsonl"
    tracer = Tracer(path)
    with tracer.span("agent.run", orbitwatch__question="q") as run:
        with tracer.span("tool.call", gen_ai__tool__name="x"):
            pass
        with pytest.raises(ValueError), tracer.span("tool.call"):
            raise ValueError("boom")
    spans = load_spans(path)
    assert [s["name"] for s in spans] == ["tool.call", "tool.call", "agent.run"]  # end order
    assert {s["trace_id"] for s in spans} == {run.trace_id}
    assert spans[0]["parent_span_id"] == run.span_id and spans[2]["parent_span_id"] is None
    assert spans[1]["status"] == {"code": "ERROR", "message": "ValueError: boom"}
    assert spans[2]["attributes"]["orbitwatch.question"] == "q"
    assert all(s["duration_ms"] >= 0 for s in spans)


def test_cost_estimate_and_unknown_model():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0}
    assert estimate_cost_usd("claude-opus-5-5", usage) == pytest.approx(4.0 + 2.0)
    assert estimate_cost_usd("some-future-model", usage) is None


def test_offline_planner_run_is_traced_with_zero_cost(toolbox):
    tracer = Tracer(keep=True)
    result = OfflinePlanner(toolbox, tracer=tracer).ask("Show me the current TLE for the ISS.")
    names = [s.name for s in tracer.finished]
    assert names == ["tool.call", "grounding.check", "agent.run"]
    run = tracer.finished[-1].attributes
    assert run["orbitwatch.agent.mode"] == "offline-scripted"
    assert run["orbitwatch.cost_usd"] == 0.0 and run["orbitwatch.llm_calls"] == 0
    assert run["orbitwatch.grounding.ok"] is True and result.grounding_ok is True
    assert result.trace_id == tracer.finished[-1].trace_id


def test_tool_error_is_recorded_on_its_span(toolbox):
    tracer = Tracer(keep=True)
    OfflinePlanner(toolbox, tracer=tracer).ask("Predict passes over Vandenberg for NORAD 99999.")
    (tool,) = [s for s in tracer.finished if s.name == "tool.call"]
    assert tool.status_code == "ERROR" and tool.attributes["error.type"] == "CelesTrakError"
    run = tracer.finished[-1].attributes
    assert run["orbitwatch.tool_errors"] == 1
    assert run["orbitwatch.grounding.ok"] is True  # the refusal quotes only the error text


def test_claude_agent_spans_carry_tokens_and_cost(toolbox):
    usage = NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=5000,
               cache_creation_input_tokens=0, iterations=None)  # fmt: skip
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "list_places", {})],
           model="claude-opus-5-5", usage=usage),
        NS(stop_reason="end_turn", content=[text("Lompoc and Vandenberg.")],
           model="claude-opus-5-5", usage=usage),
    )  # fmt: skip
    tracer = Tracer(keep=True)
    result = ClaudeAgent(toolbox, client=client, tracer=tracer).ask("Which sites?")
    llm = [s for s in tracer.finished if s.name == "llm.call"]
    assert len(llm) == 2 and llm[0].attributes["gen_ai.usage.input_tokens"] == 1000
    per_call = (1000 * 4.0 + 200 * 20.0 + 5000 * 0.20) / 1e6
    assert llm[0].attributes["orbitwatch.cost_usd"] == pytest.approx(per_call)
    assert result.cost_usd == pytest.approx(2 * per_call) and result.llm_calls == 2
    assert result.usage["cache_read_input_tokens"] == 10000
    run = tracer.finished[-1]
    assert run.name == "agent.run" and run.attributes["orbitwatch.tool_calls"] == 1
    assert all(s.trace_id == run.trace_id for s in tracer.finished)


def _span(name, ms, status="OK", **attrs):
    return {
        "trace_id": "t",
        "name": name,
        "duration_ms": ms,
        "status": {"code": status, "message": None},
        "attributes": attrs,
    }


def test_summary_metrics():
    spans = [
        _span("tool.call", 10, **{"gen_ai.tool.name": "get_tle"}),
        _span("tool.call", 30, "ERROR", **{"gen_ai.tool.name": "get_tle", "error.type": "X"}),
        _span("llm.call", 900, **{"gen_ai.usage.input_tokens": 100}),
        _span("agent.run", 100, **{"orbitwatch.agent.mode": "claude", "orbitwatch.cost_usd": 0.01,
                                   "orbitwatch.grounding.ok": True}),
        _span("agent.run", 300, **{"orbitwatch.agent.mode": "claude", "orbitwatch.cost_usd": None,
                                   "orbitwatch.grounding.ok": False}),
    ]  # fmt: skip
    s = report.summarize(spans)
    assert s["runs"] == 2 and s["latency"]["run"] == {"n": 2, "p50_ms": 200.0, "p95_ms": 290.0}
    assert s["tool_error_rate"] == 0.5 and s["tool_error_types"] == {"X": 1}
    assert s["grounding_pass_rate"] == 0.5 and s["runs_with_unknown_cost"] == 1
    assert s["cost_usd_total"] == 0.01 and s["tokens"] == {"input_tokens": 100}
    png = report.render_png(s, spans)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    page = report.render_html(s, png, "x.jsonl")
    assert "Grounding pass rate" in page and "50.0%" in page


def test_cli_ask_then_trace_summary(tmp_path, capsys):
    trace = tmp_path / "runs.jsonl"
    q = "When are the next visible ISS passes over Lompoc?"
    assert cli_main(["ask", "--offline", "--data", "fixtures", "--trace", str(trace), q]) == 0
    html = tmp_path / "r.html"
    capsys.readouterr()
    assert cli_main(["trace", "summary", "--file", str(trace), "--json", "--html", str(html)]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[: out.rindex("}") + 1])
    assert summary["runs"] == 1 and summary["grounding_pass_rate"] == 1.0
    assert html.exists()
