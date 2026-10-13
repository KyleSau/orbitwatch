"""Result types shared by the Claude agent and the offline planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class AgentResult:
    question: str
    answer: str
    mode: str  # "claude" or "offline-scripted"
    data_mode: str  # "live" or "fixtures"
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None
    notes: list[str] = field(default_factory=list)
    # Observability: filled by the agents and orbitwatch.agent.observe.
    llm_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)  # summed over LLM calls
    cost_usd: float | None = 0.0  # None when a model's price is unknown
    trace_id: str | None = None
    grounding_ok: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def tools_used(self) -> list[str]:
        return [c.name for c in self.tool_calls]
