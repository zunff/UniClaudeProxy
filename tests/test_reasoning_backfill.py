"""Tests for reasoning_content backfill on chat-completions requests.

Thinking upstreams (DeepSeek, GLM) reject any tools-carrying request in which an
assistant message omits reasoning_content, and some gateways drop that field
intermittently. These tests lock in the placeholder substitution that keeps such
histories replayable.
"""

import pytest

from app.converters.anthropic_to_openai import (
    _REASONING_CONTENT_PLACEHOLDER,
    backfill_reasoning_content,
    should_backfill_reasoning_content,
    to_openai_chat_request,
)
from app.models import AnthropicRequest

TOOLS = [
    {
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
    }
]


def _request(messages, thinking=None, tools=TOOLS, output_config=None):
    """Build an AnthropicRequest from raw message dicts."""
    payload = {"model": "claude-sonnet-5-5", "max_tokens": 256, "messages": messages}
    if tools:
        payload["tools"] = tools
    if thinking is not None:
        payload["thinking"] = thinking
    if output_config is not None:
        payload["output_config"] = output_config
    return AnthropicRequest(**payload)


TOOL_TURN = [
    {"role": "user", "content": "run echo hi"},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_00_ET_poison",
                "name": "Bash",
                "input": {"command": "echo hi"},
            }
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_00_ET_poison", "content": "hi"}],
    },
]

THINKING_TURN = [
    {"role": "user", "content": "run echo hi"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should run echo hi."},
            {
                "type": "tool_use",
                "id": "call_00_plain",
                "name": "Bash",
                "input": {"command": "echo hi"},
            },
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_00_plain", "content": "hi"}],
    },
]


def test_mixed_history_gets_placeholder():
    """A thinking turn followed by a thinking-less turn is made replayable."""
    request = _request(THINKING_TURN + TOOL_TURN[1:])
    body = to_openai_chat_request(
        request, "ds/deepseek-flash", include_reasoning_content=True
    )

    assistants = [m for m in body["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert assistants[0]["reasoning_content"] == "I should run echo hi."
    assert assistants[1]["reasoning_content"] == _REASONING_CONTENT_PLACEHOLDER


def test_thinking_enabled_backfills_without_reasoning_history():
    """A fresh session with thinking on still gets the placeholder."""
    request = _request(
        TOOL_TURN,
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    body = to_openai_chat_request(
        request, "ds/deepseek-flash", include_reasoning_content=True
    )

    assistant = next(m for m in body["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == _REASONING_CONTENT_PLACEHOLDER


def test_effort_output_config_backfills():
    """output_config.effort marks the conversation as a reasoning one."""
    request = _request(TOOL_TURN, output_config={"effort": "high"})
    messages = to_openai_chat_request(
        request, "ds/deepseek-flash", include_reasoning_content=True
    )["messages"]

    assert messages[1]["reasoning_content"] == _REASONING_CONTENT_PLACEHOLDER


def test_plain_history_is_untouched():
    """Non-reasoning upstreams must not receive an unknown message field."""
    request = _request(TOOL_TURN)
    body = to_openai_chat_request(
        request, "some/plain-model", include_reasoning_content=True
    )

    assert "reasoning_content" not in body["messages"][1]


def test_no_tools_means_no_backfill():
    """Without tools the upstream cannot enforce the round-trip."""
    request = _request(
        TOOL_TURN,
        tools=None,
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    body = to_openai_chat_request(
        request, "ds/deepseek-flash", include_reasoning_content=True
    )

    assert "tools" not in body
    assert "reasoning_content" not in body["messages"][1]


def test_explicit_override_wins_over_auto():
    """Per-model config can force or suppress the backfill."""
    forced = to_openai_chat_request(
        _request(TOOL_TURN),
        "ds/deepseek-flash",
        include_reasoning_content=True,
        reasoning_backfill=True,
    )
    assert forced["messages"][1]["reasoning_content"] == _REASONING_CONTENT_PLACEHOLDER

    # Auto-detection would backfill here (thinking is enabled); the flag wins.
    suppressed = to_openai_chat_request(
        _request(TOOL_TURN, thinking={"type": "enabled", "budget_tokens": 2048}),
        "ds/deepseek-flash",
        include_reasoning_content=True,
        reasoning_backfill=False,
    )
    assert "reasoning_content" not in suppressed["messages"][1]


def test_real_reasoning_survives_suppression():
    """Suppressing the backfill never strips reasoning the client did send."""
    body = to_openai_chat_request(
        _request(THINKING_TURN),
        "ds/deepseek-flash",
        include_reasoning_content=True,
        reasoning_backfill=False,
    )
    assert body["messages"][1]["reasoning_content"] == "I should run echo hi."


def test_reasoning_content_env_kill_switch_disables_backfill(monkeypatch):
    """ANYCLAUDE_DISABLE_REASONING_CONTENT still removes the field entirely."""
    request = _request(
        TOOL_TURN,
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    body = to_openai_chat_request(
        request, "ds/deepseek-flash", include_reasoning_content=False
    )

    assert "reasoning_content" not in body["messages"][1]


@pytest.mark.parametrize(
    "messages,expected",
    [
        (THINKING_TURN, True),
        (TOOL_TURN, False),
    ],
)
def test_should_backfill_detection(messages, expected):
    """Auto-detection keys off request.thinking and existing reasoning_content."""
    request = _request(messages)
    converted = [
        {"role": "assistant", "reasoning_content": "x"}
        if msg["role"] == "assistant" and any(
            isinstance(b, dict) and b.get("type") == "thinking" for b in msg["content"]
        )
        else {"role": msg["role"]}
        for msg in messages
    ]
    assert should_backfill_reasoning_content(request, converted) is expected


def test_backfill_reports_patch_count():
    """The helper reports how many assistant turns it patched."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "b", "reasoning_content": "kept"},
    ]
    assert backfill_reasoning_content(messages) == 1
    assert messages[1]["reasoning_content"] == _REASONING_CONTENT_PLACEHOLDER
    assert messages[3]["reasoning_content"] == "kept"
