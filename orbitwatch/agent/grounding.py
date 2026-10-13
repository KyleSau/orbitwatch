"""Check that every number, date and time in an answer came from a tool result.

This is the "never invent numbers" guard used by the evals. It is deliberately
simple and explainable:

1. Pull dates (YYYY-MM-DD) and clock times (HH:MM[:SS], optional AM/PM) out of the
   answer and require each to appear in a tool result (times within one minute, or
   one second when seconds are given).
2. Pull the remaining numbers and require each to match a tool-result number at the
   precision written (answer "45.2" matches tool 45.17), or that number x100
   (fraction quoted as a percentage), or a number in the question itself.

Scene IDs, file paths and list markers ("1.") are ignored. Signs are ignored too, so
"decreased by 0.21" matches a tool value of -0.21.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})")
TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([AaPp])\.?[Mm]\.?)?(?![\d:])")
NUMBER_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?![\w])")
PATH_RE = re.compile(r"\S*[\\/]\S*")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
NAME_NUMBER_RE = re.compile(r"\b[A-Za-z]+-\d+\b")  # names like "Sentinel-2", "Falcon-9"


@dataclass
class GroundingReport:
    ungrounded_numbers: list[str] = field(default_factory=list)
    ungrounded_dates: list[str] = field(default_factory=list)
    ungrounded_times: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not (self.ungrounded_numbers or self.ungrounded_dates or self.ungrounded_times)


def _walk(node: Any, numbers: list[float], strings: list[str]) -> None:
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, int | float):
        numbers.append(float(node))
    elif isinstance(node, str):
        strings.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _walk(v, numbers, strings)
    elif isinstance(node, list | tuple):
        for v in node:
            _walk(v, numbers, strings)


def _seconds_of_day(match: re.Match) -> tuple[int, bool]:
    hour, minute = int(match.group(1)), int(match.group(2))
    ampm = (match.group(4) or "").lower()
    if ampm == "p" and hour < 12:
        hour += 12
    if ampm == "a" and hour == 12:
        hour = 0
    has_seconds = match.group(3) is not None
    return hour * 3600 + minute * 60 + int(match.group(3) or 0), has_seconds


def _strip(text: str) -> str:
    text = PATH_RE.sub(" ", text)
    text = LIST_MARKER_RE.sub(" ", text)
    text = NAME_NUMBER_RE.sub(" ", text)
    text = DATE_RE.sub(" ", text)
    return TIME_RE.sub(" ", text)


def check_grounding(answer: str, tool_results: list[dict], question: str = "") -> GroundingReport:
    numbers: list[float] = []
    strings: list[str] = []
    _walk(tool_results, numbers, strings)
    source_text = "\n".join(strings)
    for s in strings:  # numbers embedded in strings (e.g. TLE lines, warnings)
        numbers.extend(abs(float(m.replace("−", "-"))) for m in NUMBER_RE.findall(_strip(s)))
    candidates = {abs(n) for n in numbers} | {abs(n) * 100 for n in numbers}
    question_numbers = {abs(float(m.replace("−", "-"))) for m in NUMBER_RE.findall(question)}
    tool_dates = set(DATE_RE.findall(source_text)) | set(DATE_RE.findall(question))
    tool_times = [_seconds_of_day(m)[0] for m in TIME_RE.finditer(source_text)]

    report = GroundingReport()
    for d in DATE_RE.findall(answer):
        report.checked += 1
        if d not in tool_dates:
            report.ungrounded_dates.append(d)
    for m in TIME_RE.finditer(PATH_RE.sub(" ", answer)):
        report.checked += 1
        secs, has_seconds = _seconds_of_day(m)
        tolerance = 1 if has_seconds else 60
        if not any(abs(secs - t) <= tolerance for t in tool_times):
            report.ungrounded_times.append(m.group(0))
    for token in NUMBER_RE.findall(_strip(answer)):
        report.checked += 1
        value = abs(float(token.replace("−", "-")))
        decimals = len(token.split(".")[1]) if "." in token else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if value in question_numbers or any(abs(value - c) <= tol for c in candidates):
            continue
        report.ungrounded_numbers.append(token)
    return report
