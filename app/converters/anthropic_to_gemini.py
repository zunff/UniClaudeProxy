from typing import Any

from app.converters.gemini_to_anthropic import (
    decode_gemini_part_signature,
    decode_gemini_tool_id,
)
from app.models import AnthropicRequest
from app.utils.images import detect_media_type

# Fallback when replaying tool calls from another provider (e.g. load-balanced GLM).
THOUGHT_SIGNATURE_SKIP = "skip_thought_signature_validator"

# Gemini counts thinking tokens against maxOutputTokens. Clients (e.g. Claude Code
# permission hooks) often send tiny max_tokens like 64; without headroom the model
# spends the budget on thinking and returns truncated text with finishReason=MAX_TOKENS.
_GEMINI_THINKING_HEADROOM = 1024


def _extract_system_prompt(request: AnthropicRequest) -> str | None:
    """Extract system prompt from an Anthropic request.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.

    Returns:
        str | None - The extracted system prompt text, or None.
    """
    system = request.system
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
            else:
                text = getattr(block, "text", "")
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


_GEMINI_JSON_SCHEMA_KEYS = frozenset({
    "$id", "$defs", "$ref", "$anchor",
    "type", "format", "title", "description", "enum",
    "items", "prefixItems", "minItems", "maxItems",
    "minimum", "maximum", "anyOf", "oneOf",
    "properties", "additionalProperties", "required",
    "propertyOrdering",
})


def _clean_schema(schema: Any) -> dict[str, Any]:
    """Recursively strip non-Gemini JSON Schema fields.

    Args:
        schema: Any - A JSON Schema dict (or sub-dict).

    Returns:
        dict[str, Any] - Cleaned schema safe for Gemini functionDeclarations.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_JSON_SCHEMA_KEYS:
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "oneOf" and isinstance(value, list):
            # Gemini interprets oneOf with the same semantics as anyOf.
            cleaned["anyOf"] = [
                _clean_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value

    if "required" in cleaned and "properties" in cleaned:
        valid_props = set(cleaned["properties"].keys())
        cleaned["required"] = [r for r in cleaned["required"] if r in valid_props]
        if not cleaned["required"]:
            del cleaned["required"]

    if "type" not in cleaned and "properties" in cleaned:
        cleaned["type"] = "object"

    return cleaned


def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to Gemini functionDeclarations.

    Args:
        tools: list[Any] - Anthropic tool definitions.

    Returns:
        list[dict[str, Any]] - Gemini tools array with functionDeclarations.
    """
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_type = tool.get("type", "custom")
        else:
            tool_type = getattr(tool, "type", "custom")
            tool = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)

        if tool_type in ("computer_20241022", "text_editor_20241022", "bash_20241022"):
            continue

        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        schema = _clean_schema(input_schema)

        declarations.append({
            "name": name,
            "description": description,
            # parametersJsonSchema accepts the documented JSON Schema subset
            # and preserves refs, unions, ranges and additionalProperties.
            "parametersJsonSchema": schema,
        })

    if not declarations:
        return []

    return [{"functionDeclarations": declarations}]


def _build_user_parts(content: Any) -> list[dict[str, Any]]:
    """Build ordered Gemini parts from Anthropic user message content.

    Args:
        content: Any - String or list of Anthropic content blocks.

    Returns:
        list[dict[str, Any]] - Gemini user parts in their original order.
    """
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append({"text": text})

        elif block_type == "image":
            source = block.get("source", {})
            src_type = source.get("type", "")

            if src_type == "base64":
                declared = source.get("media_type", "")
                b64_data = source.get("data", "")
                media_type = detect_media_type(b64_data, declared or "image/png")
                parts.append({
                    "inlineData": {
                        "mimeType": media_type,
                        "data": b64_data,
                    }
                })

        elif block_type == "tool_result":
            tool_name = block.get("tool_use_id", "")
            tool_content = block.get("content", "")

            if isinstance(tool_content, str):
                result_text = tool_content
            elif isinstance(tool_content, list):
                result_text = " ".join(
                    b.get("text", "") for b in tool_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                result_text = str(tool_content)

            response_payload = (
                {"error": result_text}
                if block.get("is_error", False)
                else {"result": result_text}
            )
            parts.append({
                "functionResponse": {
                    "name": tool_name,
                    "response": response_payload,
                }
            })

    return parts


def _build_assistant_parts(content: Any) -> list[dict[str, Any]]:
    """Build Gemini parts from Anthropic assistant message content.

    Args:
        content: Any - String or list of Anthropic content blocks.

    Returns:
        list[dict[str, Any]] - Gemini parts array.
    """
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = getattr(block, "type", "")
            block = block.model_dump() if hasattr(block, "model_dump") else dict(block)

        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append({"text": text})

        elif block_type == "tool_use":
            name = block.get("name", "")
            input_data = block.get("input", {})
            raw_id = block.get("id", "")

            part: dict[str, Any] = {
                "functionCall": {
                    "name": name,
                    "args": input_data,
                }
            }

            metadata = decode_gemini_tool_id(raw_id)
            if metadata is not None:
                gemini_call_id = metadata.get("id", "")
                if gemini_call_id:
                    part["functionCall"]["id"] = gemini_call_id
                # A None signature explicitly means a Gemini-origin parallel
                # call that did not carry a signature; preserve that absence.
                sig = metadata.get("thoughtSignature")
                if sig:
                    part["thoughtSignature"] = sig
            else:
                # Required for Gemini 3 when history includes tool calls from
                # non-Gemini providers or clients that don't preserve signatures.
                part["thoughtSignature"] = THOUGHT_SIGNATURE_SKIP

            parts.append(part)

        elif block_type == "thinking":
            sig = decode_gemini_part_signature(block.get("signature", ""))
            if sig:
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    parts.append({
                        "text": thinking_text,
                        "thought": True,
                        "thoughtSignature": sig,
                    })
                elif parts:
                    # Signature-only marker emitted after a normal text part.
                    parts[-1]["thoughtSignature"] = sig
                else:
                    # Preserve an otherwise standalone signed empty part.
                    parts.append({"text": "", "thoughtSignature": sig})

    return parts


def _build_contents(request: AnthropicRequest) -> list[dict[str, Any]]:
    """Build Gemini contents array from Anthropic messages.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.

    Returns:
        list[dict[str, Any]] - Gemini contents array.
    """
    contents: list[dict[str, Any]] = []
    tool_id_to_call: dict[str, dict[str, str]] = {}

    for msg in request.messages:
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                for block in content:
                    b = block if isinstance(block, dict) else (block.model_dump() if hasattr(block, "model_dump") else dict(block))
                    if b.get("type") == "tool_use":
                        raw_id = b.get("id", "")
                        metadata = decode_gemini_tool_id(raw_id) or {}
                        tool_id_to_call[raw_id] = {
                            "name": b.get("name", ""),
                            "id": str(metadata.get("id") or ""),
                        }

            parts = _build_assistant_parts(content)
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "user":
            user_parts = _build_user_parts(content)
            for part in user_parts:
                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    raw_tool_use_id = fr["name"]
                    if raw_tool_use_id in tool_id_to_call:
                        call = tool_id_to_call[raw_tool_use_id]
                        fr["name"] = call["name"]
                        if call["id"]:
                            fr["id"] = call["id"]
            if user_parts:
                contents.append({"role": "user", "parts": user_parts})

    return contents


def to_gemini_request(
    request: AnthropicRequest,
    model_id: str,
) -> dict[str, Any]:
    """Convert an Anthropic request to a Gemini generateContent request body.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        model_id: str - The Gemini model identifier.

    Returns:
        dict[str, Any] - Gemini generateContent compatible request body.
    """
    body: dict[str, Any] = {
        "contents": _build_contents(request),
    }

    system_prompt = _extract_system_prompt(request)
    if system_prompt:
        body["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    gen_config: dict[str, Any] = {}
    if request.max_tokens:
        max_out = request.max_tokens
        if max_out < _GEMINI_THINKING_HEADROOM:
            max_out = max_out + _GEMINI_THINKING_HEADROOM
        gen_config["maxOutputTokens"] = max_out
    if request.temperature is not None:
        gen_config["temperature"] = request.temperature
    if request.top_p is not None:
        gen_config["topP"] = request.top_p
    if request.top_k is not None:
        gen_config["topK"] = request.top_k
    if request.stop_sequences:
        gen_config["stopSequences"] = request.stop_sequences

    if gen_config:
        body["generationConfig"] = gen_config

    if request.tools:
        gemini_tools = _convert_tools(request.tools)
        if gemini_tools:
            body["tools"] = gemini_tools
            if request.tool_choice:
                choice_type = request.tool_choice.get("type", "auto")
                function_config: dict[str, Any] = {"mode": "AUTO"}
                if choice_type == "any":
                    function_config["mode"] = "ANY"
                elif choice_type == "none":
                    function_config["mode"] = "NONE"
                elif choice_type == "tool":
                    function_config["mode"] = "ANY"
                    name = request.tool_choice.get("name", "")
                    if name:
                        function_config["allowedFunctionNames"] = [name]
                body["toolConfig"] = {
                    "functionCallingConfig": function_config,
                }

    return body
