"""Claude tool-use agent over the orbitwatch tools (official ``anthropic`` Python SDK).

Why a hand-written loop instead of the SDK's beta tool runner? The runner executes
tools automatically; here every turn must be inspected *before* any tool runs:

* ``stop_reason == "refusal"``  -> stop; never run tools from a declined turn.
* ``stop_reason == "max_tokens"`` -> stop; a tool_use block may be truncated.
* after a server-side fallback, blocks before the switch point must not be echoed.

The loop is short, and each of those rules is visible in one place. Other choices
follow the Claude API guidance for this model:

* ``thinking={"type": "adaptive"}`` and an explicit ``output_config.effort``
  (thinking cannot be disabled on claude-opus-5-5; effort is the control).
* ``tool_choice`` is left at ``auto`` (forced tool use returns a 400 on this model);
  tools are ``strict: true`` with ``additionalProperties: false`` so arguments match.
* The full ``response.content`` (including thinking blocks) is appended each turn, and
  history, system prompt and tools never change mid-conversation, so replayed thinking
  blocks stay valid and the prompt cache keeps hitting.
* All tool results go back in one user message; failures set ``is_error: true``.
* Server-side refusal fallback: beta ``server-side-fallback-2026-07-01`` with
  ``fallbacks="default"`` re-runs a declined request on Anthropic's recommended model.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from orbitwatch import config
from orbitwatch.agent.observe import traced_run
from orbitwatch.agent.types import AgentResult, ToolCall
from orbitwatch.observability.pricing import estimate_cost_usd, usage_dict
from orbitwatch.observability.tracing import NULL_TRACER, Tracer
from orbitwatch.tools import Toolbox, ToolError, anthropic_tool_definitions, call_tool, to_json

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You are orbitwatch, an analyst for orbital and Earth-observation data, \
focused on Lompoc and Vandenberg Space Force Base, California.

Rules:
- Use the tools for every factual number: pass times, elevations, azimuths, TLE epochs, \
scene IDs, acquisition dates, cloud cover and index statistics. Never estimate or invent a \
number. If no tool returned it, say you do not have it.
- For pass predictions, state the TLE epoch used, and give times exactly as the tool \
returns them (UTC and local). For Starlink, give each satellite's elements_epoch_utc.
- For Earth-observation results, cite the scene IDs, acquisition dates and cloud cover of the \
scenes you used, and the fraction of pixels that were valid in both scenes.
- Describe what the numbers show. Do not attribute a change to a cause (fire, drought, \
harvest, construction) unless a tool result establishes it.
- If a tool returns an error, explain it briefly and retry with corrected input if sensible.
- If the data source is FIXTURES, say clearly that results use recorded fixtures with \
synthetic pixel values.
- Keep answers concise and use plain text."""


class AgentError(RuntimeError):
    """The Claude API call failed in a way the user needs to act on."""


def history_safe_content(content: list[Any]) -> list[Any]:
    """Blocks to echo back into history after a possible server-side fallback.

    Per the fallback docs: thinking, redacted_thinking and tool_use blocks that appear
    before the last ``fallback`` block belong to the declined attempt and must be omitted;
    text blocks and everything after the boundary are kept. (We use no server tools, so
    unpaired server_tool_use blocks cannot occur.)
    """
    boundary = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
    if boundary < 0:
        return list(content)
    dropped = {"thinking", "redacted_thinking", "tool_use"}
    return [b for i, b in enumerate(content) if i > boundary or b.type not in dropped]


def _text_of(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if b.type == "text").strip()


class ClaudeAgent:
    def __init__(
        self,
        toolbox: Toolbox,
        client: Any | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_turns: int = 8,
        use_fallbacks: bool = True,
        tracer: Tracer = NULL_TRACER,
    ) -> None:
        self.toolbox = toolbox
        self.tracer = tracer
        self.client = client or anthropic.Anthropic()
        self.model = model or os.environ.get("ORBITWATCH_MODEL", config.DEFAULT_MODEL)
        self.effort = effort or os.environ.get("ORBITWATCH_EFFORT", config.DEFAULT_EFFORT)
        self.max_turns = max_turns
        self.use_fallbacks = use_fallbacks
        self.tools = anthropic_tool_definitions()

    def _system(self) -> str:
        ctx = self.toolbox.ctx
        source = "FIXTURES" if ctx.mode == "fixtures" else "live services"
        return (
            f"{SYSTEM_PROMPT}\n\nCurrent time (UTC): {ctx.clock():%Y-%m-%d %H:%M}. "
            f"Data source: {source}."
        )

    def _traced_create(self, system: str, messages: list[dict[str, Any]], result: AgentResult):
        """One Messages API call inside an ``llm.call`` span; adds usage and cost to the run."""
        with self.tracer.span(
            "llm.call",
            gen_ai__system="anthropic",
            gen_ai__operation__name="chat",
            gen_ai__request__model=self.model,
            orbitwatch__turn=result.llm_calls + 1,
        ) as span:
            try:
                response = self._create(system, messages)
            except AgentError as exc:
                span.error(str(exc), type(exc.__cause__).__name__ if exc.__cause__ else None)
                raise
            usage = usage_dict(getattr(response, "usage", None))
            served_by = getattr(response, "model", self.model)
            cost = estimate_cost_usd(served_by, usage)
            span.set(
                gen_ai__response__model=served_by,
                gen_ai__response__finish_reasons=[response.stop_reason],
                orbitwatch__cost_usd=cost,
                **{f"gen_ai__usage__{k}": v for k, v in usage.items()},
            )
        result.llm_calls += 1
        for k, v in usage.items():
            result.usage[k] = result.usage.get(k, 0) + v
        if cost is None or result.cost_usd is None:
            result.cost_usd = None  # one unpriced call makes the total unknown
        else:
            result.cost_usd = round(result.cost_usd + cost, 6)
        return response

    def _create(self, system: str, messages: list[dict[str, Any]]):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "tools": self.tools,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            return self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise AgentError("Authentication failed: check ANTHROPIC_API_KEY.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise AgentError(f"Permission denied for model {self.model}: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise AgentError(f"Model or endpoint not found ({self.model}): {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise AgentError("Rate limited after SDK retries; try again shortly.") from exc
        except anthropic.BadRequestError as exc:
            raise AgentError(f"Bad request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise AgentError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError("Could not reach the Anthropic API (network error).") from exc

    def ask(self, question: str) -> AgentResult:
        result = AgentResult(
            question=question,
            answer="",
            mode="claude",
            data_mode=self.toolbox.ctx.mode,
            model=self.model,
        )
        with traced_run(self.tracer, result):
            self._loop(question, result)
        return result

    def _loop(self, question: str, result: AgentResult) -> None:
        # Built once per question: the system prompt (and tools, and earlier messages) must
        # stay byte-identical across turns, or replayed thinking blocks are rejected and the
        # prompt cache misses. The clock in it must not tick between turns.
        system = self._system()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        for _ in range(self.max_turns):
            response = self._traced_create(system, messages, result)
            result.model = getattr(response, "model", self.model)
            result.stop_reason = response.stop_reason
            iterations = getattr(getattr(response, "usage", None), "iterations", None) or []
            if any(getattr(it, "type", None) == "fallback_message" for it in iterations):
                result.notes.append(f"Server-side fallback served this turn: {result.model}")

            # Check stop_reason before touching any tool_use block.
            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details else None
                result.answer = f"The model declined this request (category: {category})."
                return
            content = history_safe_content(response.content)
            if response.stop_reason == "max_tokens":
                result.answer = _text_of(content) or "Stopped: the response hit max_tokens."
                result.notes.append("Hit max_tokens; no tools from the truncated turn were run.")
                return

            messages.append({"role": "assistant", "content": content})
            if response.stop_reason == "pause_turn":
                continue  # resume the paused turn by re-sending history
            tool_uses = [b for b in content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                result.answer = _text_of(content)
                return

            tool_results = []
            for block in tool_uses:
                call = ToolCall(name=block.name, arguments=dict(block.input or {}))
                try:
                    call.result = call_tool(self.toolbox, block.name, call.arguments, self.tracer)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": to_json(call.result),
                        }
                    )
                except ToolError as exc:
                    call.error = str(exc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )
                result.tool_calls.append(call)
            messages.append({"role": "user", "content": tool_results})

        result.answer = "Stopped after the maximum number of turns without a final answer."
        result.stop_reason = "max_turns"
