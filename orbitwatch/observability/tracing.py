"""A small span tracer that writes JSON Lines, one span per line.

Why not OpenTelemetry? The SDK plus an exporter is a lot of machinery for one
process writing one file. The record shape borrows OpenTelemetry's field names
(``trace_id``, ``span_id``, ``parent_span_id``, ``start_time_unix_nano``,
``status.code``, and the ``gen_ai.*`` semantic-convention attributes for model calls),
so the file is easy to load into anything OTel-shaped later.

Span kinds used by orbitwatch:

* ``agent.run``       one question answered (root span)
* ``llm.call``        one Messages API request (tokens, cost, stop reason)
* ``tool.call``       one tool execution (arguments, result size, errors)
* ``grounding.check`` the "every number came from a tool" check on the final answer
* ``eval.judge``      one LLM-as-judge (or stub judge) grading

A span is written when it ends, so children appear before their parent. Writing is
append-only and line-buffered; a crash loses at most the spans still open.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVICE = {"service.name": "orbitwatch", "service.version": "0.1.0"}
MAX_ATTRIBUTE_CHARS = 2000  # long strings (questions, arguments) are truncated


def _new_id(nbytes: int) -> str:
    return os.urandom(nbytes).hex()


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_ATTRIBUTE_CHARS:
        return value[:MAX_ATTRIBUTE_CHARS] + "...[truncated]"
    return value


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)
    status_code: str = "OK"
    status_message: str | None = None
    end_time_unix_nano: int | None = None
    duration_ms: float | None = None
    _t0: int = 0

    def set(self, **attributes: Any) -> None:
        """Set attributes; dots are spelled with double underscores in keyword form,
        so ``span.set(gen_ai__request__model="x")`` sets ``gen_ai.request.model``."""
        for key, value in attributes.items():
            self.attributes[key.replace("__", ".")] = _clip(value)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = _clip(value)

    def error(self, message: str, error_type: str | None = None) -> None:
        self.status_code = "ERROR"
        self.status_message = _clip(message)
        if error_type:
            self.attributes["error.type"] = error_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time_unix_nano": self.start_time_unix_nano,
            "end_time_unix_nano": self.end_time_unix_nano,
            "duration_ms": self.duration_ms,
            "status": {"code": self.status_code, "message": self.status_message},
            "attributes": self.attributes,
            "resource": SERVICE,
        }


class Tracer:
    """Creates nested spans and appends finished ones to ``path`` (if given).

    ``keep=True`` also keeps finished spans in ``self.finished`` (tests, eval runner).
    The span stack is per tracer instance: one tracer per thread of work.
    """

    def __init__(self, path: Path | str | None = None, keep: bool = False) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.finished: list[Span] = []
        self._stack: list[Span] = []

    @property
    def enabled(self) -> bool:
        return self.path is not None or self.keep

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self.current
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else _new_id(16),
            span_id=_new_id(8),
            parent_span_id=parent.span_id if parent else None,
            start_time_unix_nano=time.time_ns(),
            _t0=time.perf_counter_ns(),
        )
        span.set(**attributes)
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            if span.status_code == "OK":
                span.error(f"{type(exc).__name__}: {exc}", type(exc).__name__)
            raise
        finally:
            self._stack.pop()
            elapsed = time.perf_counter_ns() - span._t0  # monotonic, unlike wall time
            span.end_time_unix_nano = span.start_time_unix_nano + elapsed
            span.duration_ms = round(elapsed / 1e6, 3)
            self._finish(span)

    def _finish(self, span: Span) -> None:
        if self.keep:
            self.finished.append(span)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(span.to_dict(), default=str) + "\n")


NULL_TRACER = Tracer()  # records nothing; the default when no tracer is passed


def load_spans(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL trace file; malformed lines are skipped (a crash can cut one short)."""
    spans = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return spans
