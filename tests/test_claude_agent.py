"""The Claude tool-use loop, exercised against a scripted fake client (no network, no key)."""

from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from orbitwatch.agent.claude_agent import (
    FALLBACK_BETA,
    AgentError,
    ClaudeAgent,
    history_safe_content,
)


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def response(stop_reason, *content, **extra):
    return NS(
        stop_reason=stop_reason,
        content=list(content),
        model="claude-opus-5-5",
        usage=NS(iterations=extra.pop("iterations", None)),
        **extra,
    )


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # Snapshot messages: the agent appends to the same list after the call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_tool_round_trip_and_request_shape(toolbox):
    client = FakeClient(
        response("tool_use", text("Checking places."), tool_use("t1", "list_places", {})),
        response("end_turn", text("Lompoc is a known site.")),
    )
    result = ClaudeAgent(toolbox, client=client).ask("Which sites do you know?")
    assert result.answer == "Lompoc is a known site."
    assert result.tools_used == ["list_places"]

    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert "tool_choice" not in first  # forced tool use is a 400 on this model
    assert all(t["strict"] for t in first["tools"])
    assert "budget_tokens" not in str(first["thinking"])

    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][1].type == "tool_use"
    (tool_result,) = second[2]["content"]
    assert tool_result["tool_use_id"] == "t1" and "is_error" not in tool_result


def test_prefix_is_byte_identical_across_turns(toolbox):
    """System prompt and tools must not change mid-conversation (thinking-block binding)."""
    ticks = iter(range(100))
    clock = toolbox.ctx.clock
    toolbox.ctx.clock = lambda: clock().replace(minute=next(ticks) % 60)
    client = FakeClient(
        response("tool_use", tool_use("t1", "list_places", {})),
        response("tool_use", tool_use("t2", "list_places", {})),
        response("end_turn", text("done")),
    )
    ClaudeAgent(toolbox, client=client).ask("q")
    assert len({c["system"] for c in client.calls}) == 1
    assert all(c["tools"] == client.calls[0]["tools"] for c in client.calls)
    first, last = client.calls[0]["messages"], client.calls[-1]["messages"]
    assert last[: len(first)] == first  # history is append-only


def test_parallel_tool_calls_return_in_one_message_with_errors_flagged(toolbox):
    client = FakeClient(
        response(
            "tool_use",
            tool_use("a", "list_places", {}),
            tool_use("b", "get_tle", {"norad_id": "not-a-number"}),
        ),
        response("end_turn", text("done")),
    )
    result = ClaudeAgent(toolbox, client=client).ask("q")
    user_turn = client.calls[1]["messages"][-1]
    assert user_turn["role"] == "user" and len(user_turn["content"]) == 2
    ok, bad = user_turn["content"]
    assert "is_error" not in ok and bad["is_error"] is True
    assert result.tool_calls[1].error


def test_refusal_stops_without_running_tools(toolbox):
    client = FakeClient(
        response("refusal", tool_use("x", "list_places", {}), stop_details=NS(category="cyber"))
    )
    result = ClaudeAgent(toolbox, client=client).ask("q")
    assert result.stop_reason == "refusal" and "cyber" in result.answer
    assert result.tool_calls == []


def test_max_tokens_does_not_run_truncated_tool(toolbox):
    client = FakeClient(response("max_tokens", tool_use("x", "get_tle", {"norad_id": 255})))
    result = ClaudeAgent(toolbox, client=client).ask("q")
    assert result.stop_reason == "max_tokens" and result.tool_calls == []


def test_history_drops_blocks_before_fallback_boundary():
    blocks = [
        NS(type="thinking"),
        tool_use("x", "get_tle", {}),
        text("partial"),
        NS(type="fallback"),
        NS(type="thinking"),
        text("final"),
    ]
    kept = [b.type for b in history_safe_content(blocks)]
    assert kept == ["text", "fallback", "thinking", "text"]
    assert history_safe_content(blocks[:3]) == blocks[:3]  # no fallback: unchanged


def test_fallback_noted(toolbox):
    client = FakeClient(
        response(
            "end_turn",
            NS(type="fallback"),
            text("ok"),
            iterations=[NS(type="message"), NS(type="fallback_message")],
        )
    )
    result = ClaudeAgent(toolbox, client=client).ask("q")
    assert any("fallback" in n for n in result.notes)


@pytest.mark.parametrize(
    "exc,match",
    [
        (
            anthropic.AuthenticationError(
                "bad key",
                body=None,
                response=httpx2.Response(
                    401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
                ),
            ),
            "ANTHROPIC_API_KEY",
        ),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            "network",
        ),
    ],
)
def test_typed_sdk_errors_become_agent_errors(toolbox, exc, match):
    with pytest.raises(AgentError, match=match):
        ClaudeAgent(toolbox, client=FakeClient(exc)).ask("q")
