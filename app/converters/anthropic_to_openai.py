import json as _json
from typing import Any, Optional

from app.utils.images import build_image_parts

from app.models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicToolDef,
)

# Reasoning models (GLM, etc.) count thinking tokens against max_tokens.
# Claude Code permission hooks often send tiny budgets like 64; without headroom
# the model fills the quota with reasoning and returns truncated/empty content.
_REASONING_HEADROOM = 1024

# Prefix for OpenAI Responses reasoning.encrypted_content stored in Anthropic
# thinking.signature. Distinguishes our payload from Claude's native signatures.
_OA_RS_SIG_PREFIX = "oa_rs:"


def _apply_reasoning_headroom(max_tok: int | None) -> int | None:
    """Bump small max_tokens so thinking models can still emit visible text."""
    if max_tok is None:
        return None
    if max_tok < _REASONING_HEADROOM:
        return max_tok + _REASONING_HEADROOM
    return max_tok


def encode_openai_reasoning_signature(rs_id: str, encrypted_content: str) -> str:
    """Pack Responses reasoning id + encrypted_content into a thinking signature.

    Args:
        rs_id: str - OpenAI reasoning item id (e.g. 'rs_...').
        encrypted_content: str - Opaque encrypted reasoning state from OpenAI.

    Returns:
        str - Signature string safe to round-trip via Anthropic thinking blocks.
    """
    payload = _json.dumps(
        {"id": rs_id or "", "ec": encrypted_content or ""},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"{_OA_RS_SIG_PREFIX}{payload}"


def decode_openai_reasoning_signature(signature: str) -> dict[str, str] | None:
    """Unpack an OpenAI reasoning signature from an Anthropic thinking block.

    Args:
        signature: str - Value of thinking.signature from the client history.

    Returns:
        dict[str, str] | None - {"id", "ec"} if this is our encoded payload, else None.
    """
    if not signature or not isinstance(signature, str):
        return None
    if not signature.startswith(_OA_RS_SIG_PREFIX):
        return None
    try:
        data = _json.loads(signature[len(_OA_RS_SIG_PREFIX):])
    except (_json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    ec = data.get("ec") or ""
    if not ec:
        return None
    return {"id": str(data.get("id") or ""), "ec": str(ec)}


def _toolu_to_fc(tool_id: str) -> str:
    """Convert Anthropic tool ID to OpenAI Responses API function call ID.

    Args:
        tool_id: str - Anthropic tool_use ID (e.g. 'toolu_abc123').

    Returns:
        str - OpenAI Responses API ID with 'fc_' prefix.
    """
    if tool_id.startswith("fc_"):
        return tool_id
    if tool_id.startswith("toolu_"):
        return f"fc_{tool_id[6:]}"
    if tool_id.startswith("call_"):
        return f"fc_{tool_id[5:]}"
    return f"fc_{tool_id}"


def _fc_to_toolu(fc_id: str) -> str:
    """Convert OpenAI Responses API function call ID to Anthropic tool ID.

    Args:
        fc_id: str - OpenAI function call ID (e.g. 'fc_abc123').

    Returns:
        str - Anthropic tool_use ID with 'toolu_' prefix.
    """
    if fc_id.startswith("toolu_"):
        return fc_id
    if fc_id.startswith("fc_"):
        return f"toolu_{fc_id[3:]}"
    if fc_id.startswith("call_"):
        return f"toolu_{fc_id[5:]}"
    return f"toolu_{fc_id}"


def _extract_system_prompt(request: AnthropicRequest) -> Optional[str]:
    """Extract the system prompt from an Anthropic request.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.

    Returns:
        Optional[str] - The system prompt text, or None if not present.
    """
    if request.system is None:
        return None

    if isinstance(request.system, str):
        return request.system

    if isinstance(request.system, list):
        parts = []
        for block in request.system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else None

    return None


def _convert_content_to_openai_messages(content: Any) -> list[dict[str, Any]]:
    """Convert Anthropic content blocks to OpenAI message content parts.

    Args:
        content: Any - String or list of Anthropic content blocks.

    Returns:
        list[dict[str, Any]] - OpenAI-compatible content parts.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            parts.append({"type": "text", "text": block.get("text", "")})

        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                from app.utils.images import detect_media_type
                declared = source.get("media_type", "")
                data = source.get("data", "")
                media_type = detect_media_type(data, declared or "image/png")
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                        "detail": "auto",
                    },
                })
            elif source.get("type") == "url" or source.get("url"):
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": source.get("url", ""),
                        "detail": "auto",
                    },
                })

        elif block_type == "tool_use":
            pass

        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, str):
                result_text = tool_content
            elif isinstance(tool_content, list):
                result_text = " ".join(
                    b.get("text", "") for b in tool_content if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                result_text = str(tool_content)
            parts.append({"type": "text", "text": result_text})

        elif block_type == "thinking":
            pass

        else:
            text_val = block.get("text", None)
            if text_val:
                parts.append({"type": "text", "text": text_val})

    return parts if parts else [{"type": "text", "text": ""}]


ANTHROPIC_BUILTIN_TOOL_TYPES = {
    "computer_20241022",
    "text_editor_20241022",
    "bash_20241022",
    "computer_20250124",
    "text_editor_20250124",
    "bash_20250124",
}


def _convert_tools_to_openai_chat(tools: list[AnthropicToolDef]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI Chat Completions function format.

    Args:
        tools: list[AnthropicToolDef] - Anthropic tool definitions.

    Returns:
        list[dict[str, Any]] - OpenAI-compatible tool definitions.
    """
    openai_tools = []
    for tool in tools:
        tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        tool_type = tool_dict.get("type")

        if tool_type and tool_type in ANTHROPIC_BUILTIN_TOOL_TYPES:
            continue

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool_dict.get("name", ""),
                "description": tool_dict.get("description", ""),
                "parameters": tool_dict.get("input_schema", {}),
            },
        })
    return openai_tools


def _convert_tools_to_openai_responses(tools: list[AnthropicToolDef]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI Responses API function format.

    Args:
        tools: list[AnthropicToolDef] - Anthropic tool definitions.

    Returns:
        list[dict[str, Any]] - OpenAI Responses API compatible tool definitions.
    """
    openai_tools = []
    for tool in tools:
        tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        tool_type = tool_dict.get("type")

        if tool_type and tool_type in ANTHROPIC_BUILTIN_TOOL_TYPES:
            continue

        openai_tools.append({
            "type": "function",
            "name": tool_dict.get("name", ""),
            "description": tool_dict.get("description", ""),
            "parameters": tool_dict.get("input_schema", {}),
        })
    return openai_tools


def _build_chat_messages(
    request: AnthropicRequest,
    include_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI Chat Completions messages array from an Anthropic request.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.

    Returns:
        list[dict[str, Any]] - OpenAI-compatible messages array.
    """
    messages: list[dict[str, Any]] = []

    system_prompt = _extract_system_prompt(request)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in request.messages:
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        if role == "user":
            _append_user_message(messages, content)
        elif role == "assistant":
            _append_assistant_message(
                messages,
                content,
                include_reasoning_content=include_reasoning_content,
            )

    return messages


def _append_user_message(messages: list[dict[str, Any]], content: Any) -> None:
    """Append a user message to the messages list, handling tool results.

    Args:
        messages: list[dict[str, Any]] - Messages list to append to.
        content: Any - The content of the user message.
    """
    if isinstance(content, str):
        messages.append({"role": "user", "content": content})
        return

    if not isinstance(content, list):
        messages.append({"role": "user", "content": str(content)})
        return

    tool_results = []
    other_parts = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "tool_result":
            tool_results.append(block)
        else:
            other_parts.append(block)

    for tr in tool_results:
        tool_content = tr.get("content", "")
        if isinstance(tool_content, str):
            result_text = tool_content
        elif isinstance(tool_content, list):
            result_text = " ".join(
                b.get("text", "") for b in tool_content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            result_text = str(tool_content)

        messages.append({
            "role": "tool",
            "tool_call_id": tr.get("tool_use_id", ""),
            "content": result_text,
        })

    if other_parts:
        converted = _convert_content_to_openai_messages(other_parts)
        if len(converted) == 1 and converted[0].get("type") == "text":
            messages.append({"role": "user", "content": converted[0]["text"]})
        else:
            messages.append({"role": "user", "content": converted})


def _append_assistant_message(
    messages: list[dict[str, Any]],
    content: Any,
    include_reasoning_content: bool = False,
) -> None:
    """Append an assistant message to the messages list, handling tool calls.

    Args:
        messages: list[dict[str, Any]] - Messages list to append to.
        content: Any - The content of the assistant message.
    """
    if isinstance(content, str):
        messages.append({"role": "assistant", "content": content})
        return

    if not isinstance(content, list):
        messages.append({"role": "assistant", "content": str(content)})
        return

    text_parts = []
    reasoning_parts = []
    tool_calls = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            thinking_text = block.get("thinking", "")
            if thinking_text:
                reasoning_parts.append(thinking_text)
        elif block_type == "tool_use":
            import json as _json
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": _json.dumps(block.get("input", {})),
                },
            })

    msg: dict[str, Any] = {"role": "assistant"}
    combined_text = "\n".join(text_parts)
    if combined_text:
        msg["content"] = combined_text
    else:
        msg["content"] = None

    if tool_calls:
        msg["tool_calls"] = tool_calls

    if include_reasoning_content and reasoning_parts:
        msg["reasoning_content"] = "\n".join(reasoning_parts)

    messages.append(msg)


def _build_tool_summary(tools: list[Any]) -> str:
    """Build a concise summary of available tools for injection into conversation.

    Args:
        tools: list[Any] - Anthropic tool definitions.

    Returns:
        str - Formatted tool summary text.
    """
    if not tools:
        return ""

    lines = ["You have the following tools available. Use them by calling the appropriate function:\n"]
    for tool in tools:
        tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        name = tool_dict.get("name", "")
        desc = tool_dict.get("description", "")
        if name:
            if desc:
                lines.append(f"- {name}: {desc[:200]}")
            else:
                lines.append(f"- {name}")
    return "\n".join(lines)


def _build_responses_input(
    request: AnthropicRequest,
    inject_context: bool = False,
    image_mode: str = "input_image",
    image_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI Responses API input array from an Anthropic request.

    When inject_context is True, prepends a developer message containing
    the system prompt and tool summary so upstream providers that override
    instructions still see the context.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        inject_context: bool - Whether to inject system prompt and tool summary.
        image_mode: str - How to handle images: "input_image", "save_and_ref", or "strip".
        image_dir: str | None - Directory for saving images (save_and_ref mode).

    Returns:
        list[dict[str, Any]] - OpenAI Responses API compatible input items.
    """
    items: list[dict[str, Any]] = []

    if inject_context:
        system_prompt = _extract_system_prompt(request)
        tool_summary = _build_tool_summary(request.tools or [])

        injected_parts: list[str] = []
        if system_prompt:
            injected_parts.append(system_prompt)
        if tool_summary:
            injected_parts.append(tool_summary)

        if injected_parts:
            items.append({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "\n\n".join(injected_parts)}],
            })

    for msg in request.messages:
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        if role == "user":
            _append_responses_user_item(items, content, image_mode, image_dir)
        elif role == "assistant":
            _append_responses_assistant_item(items, content)

    return items


def _append_responses_user_item(
    items: list[dict[str, Any]],
    content: Any,
    image_mode: str = "input_image",
    image_dir: str | None = None,
) -> None:
    """Append a user input item for the Responses API.

    Args:
        items: list[dict[str, Any]] - Items list to append to.
        content: Any - The content of the user message.
        image_mode: str - How to handle images: "input_image", "save_and_ref", or "strip".
        image_dir: str | None - Directory for saving images (save_and_ref mode).
    """
    if isinstance(content, str):
        items.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": content}],
        })
        return

    if not isinstance(content, list):
        items.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": str(content)}],
        })
        return

    parts: list[dict[str, Any]] = []
    function_results: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            parts.append({"type": "input_text", "text": block.get("text", "")})

        elif block_type == "image":
            source = block.get("source", {})
            parts.extend(build_image_parts(source, image_mode, image_dir))

        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, str):
                result_text = tool_content
            elif isinstance(tool_content, list):
                result_text = " ".join(
                    b.get("text", "") for b in tool_content if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                result_text = str(tool_content)

            function_results.append({
                "type": "function_call_output",
                "call_id": _toolu_to_fc(block.get("tool_use_id", "")),
                "output": result_text,
            })

    for fr in function_results:
        items.append(fr)

    if parts:
        items.append({
            "type": "message",
            "role": "user",
            "content": parts,
        })


def _append_responses_assistant_item(items: list[dict[str, Any]], content: Any) -> None:
    """Append assistant output items for the Responses API.

    Preserves block order so reasoning items stay immediately before the
    function_call / message items they belong with. OpenAI requires this
    ordering when replaying encrypted_content.

    Args:
        items: list[dict[str, Any]] - Items list to append to.
        content: Any - The content of the assistant message.
    """
    if isinstance(content, str):
        items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })
        return

    if not isinstance(content, list):
        items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": str(content)}],
        })
        return

    pending_text: list[dict[str, Any]] = []

    def _flush_text() -> None:
        if not pending_text:
            return
        items.append({
            "type": "message",
            "role": "assistant",
            "content": list(pending_text),
        })
        pending_text.clear()

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "thinking":
            payload = decode_openai_reasoning_signature(block.get("signature", ""))
            if payload:
                _flush_text()
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "encrypted_content": payload["ec"],
                    "summary": [],
                }
                if payload["id"]:
                    reasoning_item["id"] = payload["id"]
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    reasoning_item["summary"] = [
                        {"type": "summary_text", "text": thinking_text},
                    ]
                items.append(reasoning_item)

        elif block_type == "text":
            text = block.get("text", "")
            if text:
                pending_text.append({"type": "output_text", "text": text})

        elif block_type == "tool_use":
            _flush_text()
            fc_id = _toolu_to_fc(block.get("id", ""))
            items.append({
                "type": "function_call",
                "id": fc_id,
                "call_id": fc_id,
                "name": block.get("name", ""),
                "arguments": _json.dumps(block.get("input", {})),
            })

    _flush_text()


def to_openai_chat_request(
    request: AnthropicRequest,
    model_id: str,
    max_output_tokens: int | None = None,
    include_reasoning_content: bool = False,
) -> dict[str, Any]:
    """Convert an Anthropic request to an OpenAI Chat Completions request body.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        model_id: str - The OpenAI model identifier.
        max_output_tokens: int | None - Optional cap on max_tokens for providers with limits.

    Returns:
        dict[str, Any] - OpenAI Chat Completions compatible request body.
    """
    max_tok = request.max_tokens
    if max_output_tokens is not None and max_tok is not None:
        max_tok = min(max_tok, max_output_tokens)
    max_tok = _apply_reasoning_headroom(max_tok)

    body: dict[str, Any] = {
        "model": model_id,
        "messages": _build_chat_messages(
            request,
            include_reasoning_content=include_reasoning_content,
        ),
        "max_tokens": max_tok,
        "stream": request.stream,
    }

    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop_sequences:
        body["stop"] = request.stop_sequences

    if request.tools:
        openai_tools = _convert_tools_to_openai_chat(request.tools)
        if openai_tools:
            body["tools"] = openai_tools
            if request.tool_choice:
                tc_type = request.tool_choice.get("type", "auto")
                if tc_type == "any":
                    body["tool_choice"] = "required"
                elif tc_type == "none":
                    body["tool_choice"] = "none"
                elif tc_type == "tool":
                    body["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice.get("name", "")},
                    }
                else:
                    body["tool_choice"] = "auto"

    if request.stream:
        body["stream_options"] = {"include_usage": True}

    return body


def to_openai_responses_request(
    request: AnthropicRequest,
    model_id: str,
    inject_context: bool = False,
    upstream_system: bool = False,
    reasoning: dict[str, Any] | None = None,
    truncation: str | None = None,
    text: dict[str, str] | None = None,
    max_output_tokens: int | None = None,
    parallel_tool_calls: bool | None = None,
    image_mode: str = "input_image",
    image_dir: str | None = None,
) -> dict[str, Any]:
    """Convert an Anthropic request to an OpenAI Responses API request body.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        model_id: str - The OpenAI model identifier.
        inject_context: bool - Inject system prompt and tool summary into input messages.
        upstream_system: bool - Provider forces its own system prompt and tools.
        reasoning: dict[str, Any] | None - Reasoning config (effort, summary).
        truncation: str | None - Truncation strategy.
        text: dict[str, str] | None - Text config (verbosity).
        max_output_tokens: int | None - Max output tokens override.
        parallel_tool_calls: bool | None - Enable parallel tool calls.
        image_mode: str - How to handle images: "input_image", "save_and_ref", or "strip".
        image_dir: str | None - Directory for saving images (save_and_ref mode).

    Returns:
        dict[str, Any] - OpenAI Responses API compatible request body.
    """
    body: dict[str, Any] = {
        "model": model_id,
        "input": _build_responses_input(request, inject_context=inject_context, image_mode=image_mode, image_dir=image_dir),
        "stream": request.stream,
        # Stateless proxy: do not rely on OpenAI server-side store; round-trip
        # reasoning.encrypted_content via Anthropic thinking.signature instead.
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }

    if not upstream_system:
        system_prompt = _extract_system_prompt(request)
        if system_prompt:
            body["instructions"] = system_prompt

    reasoning_effort = (reasoning or {}).get("effort", "none") if reasoning else None

    if reasoning_effort and reasoning_effort != "none":
        body["reasoning"] = reasoning
    elif reasoning and reasoning_effort == "none":
        body["reasoning"] = reasoning
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
    else:
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p

    if truncation:
        body["truncation"] = truncation

    if text:
        body["text"] = text

    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens

    if parallel_tool_calls is not None:
        body["parallel_tool_calls"] = parallel_tool_calls

    if not upstream_system and request.tools:
        openai_tools = _convert_tools_to_openai_responses(request.tools)
        if openai_tools:
            body["tools"] = openai_tools
            if request.tool_choice:
                tc_type = request.tool_choice.get("type", "auto")
                if tc_type == "any":
                    body["tool_choice"] = "required"
                elif tc_type == "none":
                    body["tool_choice"] = "none"
                elif tc_type == "tool":
                    body["tool_choice"] = {
                        "type": "function",
                        "name": request.tool_choice.get("name", ""),
                    }
                else:
                    body["tool_choice"] = "auto"

    return body
