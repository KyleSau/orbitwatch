"""Tracing hooks shared by the Claude agent and the offline planner.

Both front ends wrap ``ask`` in ``traced_run``. On the way out it runs the grounding
check on the final answer (a cheap, deterministic online quality signal) and puts the
run's totals on the root span, so ``orbitwatch trace summary`` can aggregate latency,
cost, tool errors and grounding without re-reading answers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from orbitwatch.agent.grounding import check_grounding
from orbitwatch.agent.types import AgentResult
from orbitwatch.observability.tracing import Span, Tracer


def tool_sources(result: AgentResult) -> list[dict]:
    """What an answer may quote: tool results, and tool error messages (an answer that
    explains an error may repeat what the error said)."""
    sources = [c.result for c in result.tool_calls if c.result is not None]
    return sources + [{"tool_error": c.error} for c in result.tool_calls if c.error]


@contextmanager
def traced_run(tracer: Tracer, result: AgentResult) -> Iterator[Span]:
    with tracer.span(
        "agent.run",
        orbitwatch__agent__mode=result.mode,
        orbitwatch__data_mode=result.data_mode,
        orbitwatch__question=result.question,
        gen_ai__request__model=result.model,
    ) as span:
        result.trace_id = span.trace_id
        yield span
        finish_run(tracer, span, result)


def finish_run(tracer: Tracer, span: Span, result: AgentResult) -> None:
    outputs = tool_sources(result)
    with tracer.span("grounding.check") as g:
        report = check_grounding(result.answer, outputs, result.question)
        g.set(
            orbitwatch__grounding__ok=report.ok,
            orbitwatch__grounding__checked=report.checked,
            orbitwatch__grounding__ungrounded=(
                report.ungrounded_numbers + report.ungrounded_dates + report.ungrounded_times
            )[:20],
        )
    result.grounding_ok = report.ok
    tool_errors = sum(1 for c in result.tool_calls if c.error)
    span.set(
        gen_ai__response__model=result.model,
        orbitwatch__stop_reason=result.stop_reason,
        orbitwatch__tool_calls=len(result.tool_calls),
        orbitwatch__tool_errors=tool_errors,
        orbitwatch__llm_calls=result.llm_calls,
        orbitwatch__cost_usd=result.cost_usd,
        orbitwatch__grounding__ok=report.ok,
        orbitwatch__answer_chars=len(result.answer),
        **{f"gen_ai__usage__{k}": v for k, v in result.usage.items()},
    )
