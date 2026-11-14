import asyncio
import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import (
    AppConfig,
    ResolvedRoute,
    atomic_write_json,
    ProviderConfig,
    config_path,
    global_path,
    prices_path,
    load_config,
    reload_config,
    resolve_route,
    merge_config_files,
    split_config_files,
    read_provider_block,
    write_provider_block,
)
from app import billing
from app.watcher import ConfigWatcher
from app.converters.gemini_to_anthropic import (
    build_tool_param_index,
    from_gemini_response,
    stream_gemini_to_anthropic,
)
from app.converters.openai_to_anthropic import (
    from_openai_chat_response,
    from_openai_responses_response,
    stream_openai_chat_to_anthropic,
    stream_openai_responses_to_anthropic,
)
from app.providers import anthropic_provider, gemini_provider, openai_provider
from app.providers import web as web_providers
from app.react import transform_request as react_transform_request
from app.react import transform_response as react_transform_response
from app.react import transform_stream as react_transform_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("uniclaudeproxy")

debug_logger = logging.getLogger("uniclaudeproxy.debug")
debug_logger.setLevel(logging.DEBUG)
_debug_handler = logging.FileHandler("debug.log", mode="a", encoding="utf-8")
_debug_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
debug_logger.addHandler(_debug_handler)
debug_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: load config on startup, close clients on shutdown.

    Args:
        app: FastAPI - The FastAPI application instance.
    """
    with open("debug.log", "w", encoding="utf-8") as f:
        f.truncate(0)

    cfg = load_config()
    logger.info("UniClaudeProxy started on %s:%d", cfg.server.host, cfg.server.port)
    logger.info("Model mappings: %s", json.dumps(cfg.models, indent=2))

    def _on_config_change():
        """Callback invoked by the config watcher when config.json changes."""
        new_cfg = reload_config()
        logger.info("Hot-reloaded model mappings: %s", json.dumps(new_cfg.models, indent=2))

    watcher = ConfigWatcher([config_path(), global_path(), prices_path()], _on_config_change)
    watcher.start()

    yield

    watcher.stop()
    await openai_provider.close_client()
    await gemini_provider.close_client()
    await anthropic_provider.close_client()
    await web_providers.close()
    logger.info("UniClaudeProxy shutdown complete")


app = FastAPI(
    title="UniClaudeProxy",
    description="Anthropic API proxy that bridges to OpenAI and other providers",
    version="1.0.0",
    lifespan=lifespan,
)

LOCALHOST_ADDRESSES = {"127.0.0.1", "::1", "localhost"}


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    """Middleware that restricts access to localhost connections only.

    Rejects any request originating from a non-local IP address
    when local_only is enabled in the server configuration.
    """

    async def dispatch(self, request: Request, call_next):
        """Check if the request originates from a local address.

        Args:
            request: Request - The incoming request.
            call_next: Callable - The next middleware or route handler.

        Returns:
            Response - The response from the next handler, or 403 if blocked.
        """
        cfg = load_config()
        if cfg.server.local_only:
            client_host = request.client.host if request.client else None
            if client_host not in LOCALHOST_ADDRESSES:
                logger.warning("Blocked non-local request from %s", client_host)
                return PlainTextResponse(
                    status_code=403,
                    content="Forbidden: local_only mode is enabled. Only localhost connections are allowed.",
                )
        return await call_next(request)


app.add_middleware(LocalOnlyMiddleware)


def _accumulate_stream_usage(event: str, acc: dict[str, int]) -> None:
    """Extract usage tokens from an Anthropic SSE event into acc.

    Scans message_start (message.usage) and message_delta (usage) events,
    overwriting with the latest non-zero value so the final message_delta
    (which carries the real input/cache fields) wins.

    Args:
        event: str - One Anthropic SSE event string.
        acc: dict[str, int] - Mutable accumulator for input/output/cache tokens.
    """
    for line in event.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except (json.JSONDecodeError, TypeError):
            continue
        etype = data.get("type", "")
        if etype == "message_start":
            u = (data.get("message") or {}).get("usage") or {}
        elif etype == "message_delta":
            u = data.get("usage") or {}
        else:
            continue
        if not isinstance(u, dict):
            continue
        if u.get("input_tokens"):
            acc["input"] = u["input_tokens"]
        if u.get("output_tokens"):
            acc["output"] = u["output_tokens"]
        if u.get("cache_read_input_tokens"):
            acc["cache_read"] = u["cache_read_input_tokens"]
        if u.get("cache_creation_input_tokens"):
            acc["cache_creation"] = u["cache_creation_input_tokens"]


@app.post("/v1/messages")
async def create_message(request: Request) -> Any:
    """Handle POST /v1/messages - the Anthropic Messages API endpoint.

    Args:
        request: Request - The incoming FastAPI request object.

    Returns:
        Any - JSONResponse for non-streaming, StreamingResponse for streaming.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Invalid JSON body"},
            },
        )

    anthropic_model = body.get("model", "")
    is_stream = body.get("stream", False)

    try:
        route = resolve_route(anthropic_model)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e)},
            },
        )

    logger.info(
        "Request: model=%s -> %s/%s (type=%s, stream=%s)",
        anthropic_model,
        route.provider_name,
        route.model_id,
        route.provider_type,
        is_stream,
    )

    replacements = route.model_config.system_replacements
    if replacements:
        system = body.get("system")
        if isinstance(system, str):
            for target, replacement in replacements.items():
                system = system.replace(target, replacement)
            body["system"] = system
        elif isinstance(system, list):
            for idx, block in enumerate(system):
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "")
                    for target, replacement in replacements.items():
                        txt = txt.replace(target, replacement)
                    system[idx] = {**block, "text": txt}
            body["system"] = system

    # Web providers (browser/HTTP chat sites) have no native tool calling, so they always go through ReAct.
    use_react = route.model_config.use_react or web_providers.is_web_provider(route.provider_type)
    if use_react:
        body = react_transform_request(body)

    from app.models import AnthropicRequest

    try:
        anthropic_request = AnthropicRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": f"Request parsing error: {e}"},
            },
        )

    if route.provider_type == "claude":
        return await _handle_claude_passthrough(body, route, anthropic_model, is_stream)

    if use_react:
        return await _handle_react(anthropic_request, route, anthropic_model, is_stream)

    if is_stream:
        return await _handle_streaming(anthropic_request, route, anthropic_model)
    elif route.force_stream:
        return await _handle_force_stream_non_streaming(anthropic_request, route, anthropic_model)
    else:
        return await _handle_non_streaming(anthropic_request, route, anthropic_model)


async def _handle_claude_passthrough(
    raw_body: dict[str, Any],
    route: Any,
    anthropic_model: str,
    is_stream: bool,
) -> Any:
    """Handle requests for Anthropic passthrough providers.

    Forwards the raw body to the upstream Anthropic-compatible API
    and returns the response directly without any conversion.

    Args:
        raw_body: dict[str, Any] - The raw JSON request body.
        route: ResolvedRoute - Resolved routing information.
        anthropic_model: str - The original Anthropic model name.
        is_stream: bool - Whether the client requested streaming.

    Returns:
        Any - JSONResponse for non-streaming, StreamingResponse for streaming.
    """
    if is_stream:
        async def passthrough_generator():
            """Yield SSE events from the upstream Anthropic provider.

            Yields:
                bytes - Raw SSE event bytes from the provider.
            """
            try:
                async for chunk in anthropic_provider.send_streaming(raw_body, route):
                    # Log streaming chunks for debugging
                    debug_logger.debug("Claude passthrough streaming chunk: %s", chunk.decode("utf-8").strip())
                    # Check if the chunk contains an error response
                    chunk_str = chunk.decode("utf-8")
                    if "data: {" in chunk_str:
                        try:
                            # Extract JSON data from SSE chunk
                            data_start = chunk_str.find("data: ") + 6
                            data_end = chunk_str.rfind("\n")
                            if data_start < data_end:
                                data_str = chunk_str[data_start:data_end]
                                data = json.loads(data_str)
                                # Check if it's an error response
                                if data.get("code") != 0 and data.get("success") is False:
                                    error_event = json.dumps({
                                        "type": "error",
                                        "error": {"type": "api_error", "message": data.get("msg", "Unknown provider error")},
                                    })
                                    yield f"event: error\ndata: {error_event}\n\n".encode("utf-8")
                                    return
                        except Exception:
                            # If parsing fails, just pass the chunk through
                            pass
                    yield chunk
            except Exception as e:
                logger.error("Claude passthrough streaming error: %s", e)
                error_event = json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Provider error: {e}"},
                })
                yield f"event: error\ndata: {error_event}\n\n".encode("utf-8")

        return StreamingResponse(
            passthrough_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        raw_response = await anthropic_provider.send_non_streaming(raw_body, route)
        # Log raw response for debugging
        debug_logger.debug("Claude passthrough raw response: %s", json.dumps(raw_response))
    except Exception as e:
        logger.error("Claude passthrough error [provider=%s]: %s", route.provider_name, e)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Provider error: {e}"},
            },
        )

    # Check if the response is an error
    if raw_response.get("code") != 0 and raw_response.get("success") is False:
        logger.error("Upstream provider error [provider=%s]: %s", route.provider_name, raw_response.get("msg"))
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": raw_response.get("msg", "Unknown provider error")},
            },
        )

    # Ensure usage field exists with input_tokens and output_tokens
    if "usage" not in raw_response:
        raw_response["usage"] = {"input_tokens": 0, "output_tokens": 0}
    else:
        if "input_tokens" not in raw_response["usage"]:
            raw_response["usage"]["input_tokens"] = 0
        if "output_tokens" not in raw_response["usage"]:
            raw_response["usage"]["output_tokens"] = 0

    # Log final response for debugging
    debug_logger.debug("Claude passthrough final response: %s", json.dumps(raw_response))

    return JSONResponse(content=raw_response)


async def _handle_react(
    request: Any,
    route: Any,
    anthropic_model: str,
    is_stream: bool,
) -> Any:
    """Handle requests with ReAct-style XML tool calling.

    Routes through the normal provider flow, then post-transforms the
    response to parse XML tool calls and convert them to proper Anthropic
    tool_use content blocks.

    Args:
        request: AnthropicRequest - The parsed (ReAct-transformed) request.
        route: ResolvedRoute - Resolved routing information.
        anthropic_model: str - The original Anthropic model name.
        is_stream: bool - Whether the client requested streaming.

    Returns:
        Any - JSONResponse for non-streaming, StreamingResponse for streaming.
    """
    if is_stream:
        async def react_event_generator():
            """Generate ReAct-parsed Anthropic SSE events.

            Yields:
                str - Anthropic-formatted SSE event strings with tool_use blocks.
            """
            try:
                if web_providers.is_web_provider(route.provider_type):
                    # Already yields Anthropic SSE strings; no converter needed.
                    upstream = web_providers.send_streaming(request, route)
                elif route.provider_type == "gemini":
                    _pi = build_tool_param_index(request.tools) if request.tools else None
                    raw_stream = gemini_provider.send_streaming(request, route)
                    upstream = stream_gemini_to_anthropic(raw_stream, anthropic_model, param_index=_pi)
                elif route.use_responses:
                    raw_stream = openai_provider.send_streaming(request, route)
                    upstream = stream_openai_responses_to_anthropic(raw_stream, anthropic_model, tool_mapping=route.tool_mapping or None)
                else:
                    raw_stream = openai_provider.send_streaming(request, route)
                    upstream = stream_openai_chat_to_anthropic(raw_stream, anthropic_model)

                async for event in react_transform_stream(upstream, anthropic_model):
                    yield event

            except Exception as e:
                logger.error("ReAct streaming error [provider=%s]: %s", route.provider_name, e)
                error_event = json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Streaming error: {e}"},
                })
                yield f"event: error\ndata: {error_event}\n\n"

        return StreamingResponse(
            react_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        if web_providers.is_web_provider(route.provider_type):
            anthropic_response = await web_providers.send_non_streaming(request, route)
            anthropic_response["model"] = anthropic_model
        elif route.provider_type == "gemini":
            raw_response = await gemini_provider.send_non_streaming(request, route)
            anthropic_response = from_gemini_response(raw_response, anthropic_model)
        elif route.use_responses:
            raw_response = await openai_provider.send_non_streaming(request, route)
            anthropic_response = from_openai_responses_response(raw_response, anthropic_model)
        else:
            raw_response = await openai_provider.send_non_streaming(request, route)
            anthropic_response = from_openai_chat_response(raw_response, anthropic_model)
    except Exception as e:
        logger.error("ReAct provider error [provider=%s]: %s", route.provider_name, e)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Provider error: {e}"},
            },
        )

    anthropic_response = react_transform_response(anthropic_response, anthropic_model)

    logger.info(
        "ReAct response [provider=%s]: model=%s, stop_reason=%s, blocks=%d",
        route.provider_name,
        anthropic_model,
        anthropic_response.get("stop_reason", "unknown"),
        len(anthropic_response.get("content", [])),
    )

    return JSONResponse(content=anthropic_response)


async def _handle_force_stream_non_streaming(
    request: Any,
    route: Any,
    anthropic_model: str,
) -> JSONResponse:
    """Handle a non-streaming request by forcing stream and collecting the response.

    Used when the provider always returns SSE regardless of the stream flag.
    Forces stream=True, collects all Anthropic SSE events, and reconstructs
    a non-streaming Anthropic response JSON.

    Args:
        request: AnthropicRequest - The parsed Anthropic request.
        route: ResolvedRoute - Resolved routing information.
        anthropic_model: str - The original Anthropic model name.

    Returns:
        JSONResponse - The Anthropic-formatted non-streaming response.
    """
    original_stream = request.stream
    request.stream = True
    start = time.perf_counter()
    ttfb_sink: dict[str, float] = {}

    try:
        if route.provider_type == "gemini":
            _pi = build_tool_param_index(request.tools) if request.tools else None
            raw_stream = _track_first_byte(gemini_provider.send_streaming(request, route), ttfb_sink, start)
            converter = stream_gemini_to_anthropic(raw_stream, anthropic_model, param_index=_pi)
        elif route.use_responses:
            raw_stream = _track_first_byte(openai_provider.send_streaming(request, route), ttfb_sink, start)
            converter = stream_openai_responses_to_anthropic(raw_stream, anthropic_model, tool_mapping=route.tool_mapping or None)
        else:
            raw_stream = _track_first_byte(openai_provider.send_streaming(request, route), ttfb_sink, start)
            converter = stream_openai_chat_to_anthropic(raw_stream, anthropic_model)

        text_parts: list[str] = []
        content_blocks: list[dict] = []
        stop_reason = "end_turn"
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        current_block: dict | None = None

        async for event_str in converter:
            # Log streaming events for debugging
            debug_logger.debug("Force-stream event: %s", event_str.strip())
            for line in event_str.strip().split("\n"):
                if line.startswith("data: "):
                    try:
                        evt = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    evt_type = evt.get("type", "")

                    if evt_type == "content_block_start":
                        cb = evt.get("content_block", {})
                        if cb.get("type") == "text":
                            current_block = {"type": "text", "text": ""}
                        elif cb.get("type") == "tool_use":
                            current_block = {
                                "type": "tool_use",
                                "id": cb.get("id", ""),
                                "name": cb.get("name", ""),
                                "input": {},
                            }

                    elif evt_type == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta" and current_block and current_block["type"] == "text":
                            current_block["text"] += delta.get("text", "")
                        elif delta.get("type") == "input_json_delta" and current_block and current_block["type"] == "tool_use":
                            partial = delta.get("partial_json", "")
                            if not hasattr(current_block, "_raw_json"):
                                current_block["_raw_json"] = ""
                            current_block["_raw_json"] = current_block.get("_raw_json", "") + partial

                    elif evt_type == "content_block_stop":
                        if current_block:
                            if current_block["type"] == "tool_use" and "_raw_json" in current_block:
                                try:
                                    current_block["input"] = json.loads(current_block.pop("_raw_json"))
                                except json.JSONDecodeError:
                                    current_block.pop("_raw_json", None)
                            elif current_block["type"] == "tool_use":
                                current_block.pop("_raw_json", None)
                            content_blocks.append(current_block)
                            current_block = None

                    elif evt_type == "message_delta":
                        delta = evt.get("delta", {})
                        if "stop_reason" in delta:
                            stop_reason = delta["stop_reason"]
                        u = evt.get("usage", {})
                        if u.get("output_tokens"):
                            usage["output_tokens"] = u["output_tokens"]
                        if u.get("input_tokens"):
                            usage["input_tokens"] = u["input_tokens"]
                        if u.get("cache_read_input_tokens"):
                            usage["cache_read_input_tokens"] = u["cache_read_input_tokens"]
                        if u.get("cache_creation_input_tokens"):
                            usage["cache_creation_input_tokens"] = u["cache_creation_input_tokens"]

    except Exception as e:
        logger.error("Force-stream non-streaming error [provider=%s]: %s", route.provider_name, e)
        billing.record(route, anthropic_model, None, is_stream=True, success=False, latency_ms=(time.perf_counter() - start) * 1000, ttfb_ms=ttfb_sink.get("ttfb_ms"))
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Provider error: {e}"},
            },
        )
    finally:
        request.stream = original_stream

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    response_body = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }

    billing.record(route, anthropic_model, usage, is_stream=True, success=True, latency_ms=(time.perf_counter() - start) * 1000, ttfb_ms=ttfb_sink.get("ttfb_ms"))
    return JSONResponse(content=response_body)


async def _handle_non_streaming(
    request: Any,
    route: Any,
    anthropic_model: str,
) -> JSONResponse:
    """Handle a non-streaming request by forwarding to the provider and converting the response.

    Args:
        request: AnthropicRequest - The parsed Anthropic request.
        route: ResolvedRoute - Resolved routing information.
        anthropic_model: str - The original Anthropic model name.

    Returns:
        JSONResponse - The Anthropic-formatted response.
    """
    _gemini_pi = build_tool_param_index(request.tools) if route.provider_type == "gemini" and request.tools else None
    start = time.perf_counter()
    try:
        if route.provider_type == "gemini":
            raw_response = await gemini_provider.send_non_streaming(request, route)
        else:
            raw_response = await openai_provider.send_non_streaming(request, route)
        # Log raw response for debugging
        debug_logger.debug("Raw provider response: %s", json.dumps(raw_response))

        # Check if the response is an error
        if raw_response.get("code") != 0 and raw_response.get("success") is False:
            logger.error("Upstream provider error [provider=%s]: %s", route.provider_name, raw_response.get("msg"))
            billing.record(route, anthropic_model, None, is_stream=False, success=False, latency_ms=(time.perf_counter() - start) * 1000)
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": raw_response.get("msg", "Unknown provider error")},
                },
            )
    except Exception as e:
        logger.error("Provider request failed [provider=%s]: %s", route.provider_name, e)
        billing.record(route, anthropic_model, None, is_stream=False, success=False, latency_ms=(time.perf_counter() - start) * 1000)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Provider error: {e}"},
            },
        )

    try:
        if route.provider_type == "gemini":
            anthropic_response = from_gemini_response(raw_response, anthropic_model, param_index=_gemini_pi)
        elif route.use_responses:
            anthropic_response = from_openai_responses_response(raw_response, anthropic_model)
        else:
            anthropic_response = from_openai_chat_response(raw_response, anthropic_model)
        # Log converted response for debugging
        debug_logger.debug("Converted Anthropic response: %s", json.dumps(anthropic_response))
    except Exception as e:
        logger.error("Response conversion failed [provider=%s]: %s", route.provider_name, e)
        logger.error("Raw response that caused conversion error: %s", json.dumps(raw_response))
        billing.record(route, anthropic_model, None, is_stream=False, success=False, latency_ms=(time.perf_counter() - start) * 1000)
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Conversion error: {e}"},
            },
        )

    # Ensure usage field exists with input_tokens and output_tokens
    if "usage" not in anthropic_response:
        anthropic_response["usage"] = {"input_tokens": 0, "output_tokens": 0}
    else:
        if "input_tokens" not in anthropic_response["usage"]:
            anthropic_response["usage"]["input_tokens"] = 0
        if "output_tokens" not in anthropic_response["usage"]:
            anthropic_response["usage"]["output_tokens"] = 0

    usage = anthropic_response.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens", 0)
    logger.info(
        "Response [provider=%s]: model=%s, stop_reason=%s, output_tokens=%d, input_tokens=%d, cache_read=%d, cache_hit=%s",
        route.provider_name,
        anthropic_model,
        anthropic_response.get("stop_reason", "unknown"),
        usage.get("output_tokens", 0),
        usage.get("input_tokens", 0),
        cache_read,
        bool(cache_read),
    )
    billing.record(route, anthropic_model, usage, is_stream=False, success=True, latency_ms=(time.perf_counter() - start) * 1000)

    return JSONResponse(content=anthropic_response)


async def _track_first_byte(
    stream: Any,
    sink: dict[str, float],
    start: float,
) -> Any:
    """Pass through an upstream byte stream, recording time-to-first-byte into sink."""
    async for chunk in stream:
        if "ttfb_ms" not in sink:
            sink["ttfb_ms"] = (time.perf_counter() - start) * 1000
        yield chunk


async def _handle_streaming(
    request: Any,
    route: Any,
    anthropic_model: str,
) -> StreamingResponse:
    """Handle a streaming request by forwarding to the provider and converting SSE events.

    Args:
        request: AnthropicRequest - The parsed Anthropic request.
        route: ResolvedRoute - Resolved routing information.
        anthropic_model: str - The original Anthropic model name.

    Returns:
        StreamingResponse - SSE stream of Anthropic-formatted events.
    """
    async def event_generator():
        """Generate Anthropic SSE events from the provider's streaming response.

        Yields:
            str - Anthropic-formatted SSE event strings.
        """
        start = time.perf_counter()
        acc: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        success = True
        ttfb_sink: dict[str, float] = {}
        try:
            if route.provider_type == "gemini":
                _pi = build_tool_param_index(request.tools) if request.tools else None
                raw_stream = _track_first_byte(gemini_provider.send_streaming(request, route), ttfb_sink, start)
                converter = stream_gemini_to_anthropic(raw_stream, anthropic_model, param_index=_pi)
            elif route.use_responses:
                raw_stream = _track_first_byte(openai_provider.send_streaming(request, route), ttfb_sink, start)
                converter = stream_openai_responses_to_anthropic(raw_stream, anthropic_model, tool_mapping=route.tool_mapping or None)
            else:
                raw_stream = _track_first_byte(openai_provider.send_streaming(request, route), ttfb_sink, start)
                converter = stream_openai_chat_to_anthropic(raw_stream, anthropic_model)

            async for event in converter:
                # Log streaming events for debugging
                debug_logger.debug("Streaming event: %s", event.strip())
                _accumulate_stream_usage(event, acc)
                yield event

        except Exception as e:
            success = False
            logger.error("Streaming error [provider=%s]: %s", route.provider_name, e)
            error_event = json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": f"Streaming error: {e}"},
            })
            yield f"event: error\ndata: {error_event}\n\n"
        finally:
            usage = {
                "input_tokens": acc["input"],
                "output_tokens": acc["output"],
                "cache_read_input_tokens": acc["cache_read"],
                "cache_creation_input_tokens": acc["cache_creation"],
            }
            billing.record(route, anthropic_model, usage, is_stream=True, success=success, latency_ms=(time.perf_counter() - start) * 1000, ttfb_ms=ttfb_sink.get("ttfb_ms"))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict[str, str] - Health status.
    """
    return {"status": "ok"}


# ================= Admin Dashboard API =================

_BEIJING_TZ_ADMIN = timezone(timedelta(hours=8))


def _read_raw_config() -> dict[str, Any]:
    """Read merged global.json + config.json for the dashboard."""
    cfg = Path(config_path())
    local_raw: dict[str, Any] = {}
    if cfg.exists():
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                local_raw = data
    gpath = Path(global_path())
    global_raw: dict[str, Any] = {}
    if gpath.exists():
        with open(gpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                global_raw = data
    return merge_config_files(global_raw, local_raw)


def _write_raw_config(raw: dict[str, Any]) -> None:
    """Split merged config into global.json + config.json, then reload."""
    # Validate the merged view so neither file can be written in a broken state.
    AppConfig(**raw)
    global_out, local_out = split_config_files(raw)
    atomic_write_json(Path(global_path()), global_out)
    atomic_write_json(Path(config_path()), local_out)
    reload_config()


def _read_raw_prices() -> dict[str, Any]:
    """Read prices.json from disk as plain dict."""
    path = Path(prices_path())
    if not path.exists():
        return {"prices": {}, "fx_to_cny": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data
        return {"prices": {}, "fx_to_cny": {}}


def _write_raw_prices(raw: dict[str, Any]) -> None:
    """Atomic write prices.json + reload runtime config."""
    to_save = {k: v for k, v in raw.items() if k != "price_bindings"}
    atomic_write_json(Path(prices_path()), to_save)
    reload_config()


def _read_provider_block(name: str) -> dict[str, Any]:
    """Return the raw provider block ``name`` from config.json (mutable)."""
    return read_provider_block(name)


def _write_provider_block(name: str, block: dict[str, Any]) -> None:
    """Write back only one provider block, then hot-reload config."""
    write_provider_block(name, block)


def _require_web_provider(name: str) -> None:
    """Raise 404 unless ``name`` is a configured web provider."""
    if web_providers.site_for_provider(name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"{name} is not a web provider (check provider_type in config.json)",
        )


@app.get("/api/web/sites")
async def admin_web_sites() -> dict[str, Any]:
    """List configured web providers plus the sites that can be added in one click."""
    return {
        "sites": web_providers.configured_providers(),
        "templates": web_providers.site_templates(),
    }


@app.post("/api/web/{name}/cookies")
async def admin_web_set_cookies(name: str, request: Request) -> Any:
    """Store a cookie header on a web provider (seeds a browser site's login).

    Body: ``{"cookies": "refresh_token=...; cna=..."}`` — copied from a normal
    browser's devtools. Used by sites whose login page is captcha-protected.
    """
    _require_web_provider(name)
    try:
        body = await request.json()
        cookies = str((body or {}).get("cookies") or "").strip()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid JSON"})
    storage = ""
    try:
        storage = str((body or {}).get("storage") or "").strip()
    except Exception:
        storage = ""
    if storage:
        try:
            import json as _json

            if not isinstance(_json.loads(storage), dict):
                raise ValueError("storage 必须是一个 JSON 对象")
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"storage 解析失败: {e}"})
    block = _read_provider_block(name)
    block["cookies"] = cookies
    block["storage"] = storage
    try:
        _write_provider_block(name, block)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "cookies": len([p for p in cookies.split(";") if "=" in p])}


@app.post("/api/web/probe")
async def admin_web_probe(request: Request) -> Any:
    """Replay a pasted "Copy as cURL" once and return the head of its response.

    Adding a site needs the page's own request and the shape of its frames; this
    turns "give me a capture" into one paste. Local-only, runs with your cookie,
    against your own account.
    """
    from app.providers.web.probe import probe
    from app.providers.web.base import WebProviderError as WebError

    try:
        body = await request.json()
        curl = str((body or {}).get("curl") or "")
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid JSON"})
    if not curl.strip():
        return JSONResponse(status_code=400, content={"ok": False, "error": "curl is required"})
    try:
        return await probe(curl)
    except WebError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.exception("web probe failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/api/web/add-site")
async def admin_web_add_site(request: Request) -> Any:
    """Create a provider block for a known web site (no config.json editing).

    Body: ``{"site": "hailuo-web"}``. Idempotent: an existing block is returned
    as-is. Config-declared sites (``web-http``) are rejected — they need a
    hand-written ``web`` block.
    """
    try:
        body = await request.json()
        site_name = str((body or {}).get("site") or "").strip()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid JSON"})
    if not site_name:
        return JSONResponse(status_code=400, content={"ok": False, "error": "site is required"})
    try:
        provider = web_providers.ensure_provider(site_name)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "provider": provider}


@app.get("/api/web/{name}/status")
async def admin_web_status(name: str) -> dict[str, Any]:
    """Return browser/login (or token) state of one web provider, per account."""
    _require_web_provider(name)
    return await web_providers.get_status(name)


@app.post("/api/web/{name}/login")
async def admin_web_login(name: str, request: Request) -> Any:
    """Open this provider's browser window so the user can log in.

    The optional JSON body ``{"account": "<id>"}`` selects the account; omitted
    or empty drives the currently active one. HTTP providers have no window and
    answer with instructions for copying their token instead.
    """
    _require_web_provider(name)
    account: str | None = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            account = body.get("account")
    except Exception:
        pass
    try:
        return await web_providers.open_login_window(name, account)
    except Exception as e:
        logger.error("%s open login window failed: %s", name, e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.post("/api/web/{name}/capture-token")
async def admin_web_capture_token(name: str, request: Request) -> Any:
    """Open a login window for a direct-HTTP provider and save the token it sends.

    The user logs in and chats in the window that opens; the proxy reads the
    ``Authorization`` header the page itself sends and writes it into that
    provider's block in config.json (hot-reloaded). The optional JSON body
    ``{"timeout": <seconds>}`` bounds the wait (default 300).
    """
    _require_web_provider(name)
    timeout = 300.0
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("timeout"):
            timeout = float(body["timeout"])
    except Exception:
        pass
    try:
        return await web_providers.capture_token(name, timeout)
    except Exception as e:
        logger.error("%s token capture failed: %s", name, e)
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.post("/api/web/{name}/accounts")
async def admin_web_add_account(name: str, request: Request) -> Any:
    """Add an account id (browser site) or token (HTTP site) to a provider."""
    _require_web_provider(name)
    try:
        body = await request.json()
        account = str((body or {}).get("account") or "").strip()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid JSON"})
    if not account:
        return JSONResponse(status_code=400, content={"ok": False, "error": "account is required"})
    site = web_providers.site_for_provider(name)
    if site is not None and site.transport == "http":
        # A pasted token usually comes with "Bearer " or quotes around it.
        from app.providers.web.token_capture import clean_token

        cleaned = clean_token(account)
        if cleaned is None:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "看起来不是有效 token（太短或含空格）；复制 `authorization` 头里 Bearer 后面那段"},
            )
        account = cleaned
    block = _read_provider_block(name)
    keys = [str(k) for k in (block.get("api_keys") or [])]
    if account in keys:
        return JSONResponse(status_code=400, content={"ok": False, "error": "account already exists"})
    keys.append(account)
    block["api_keys"] = keys
    block.setdefault("active_account", keys[0])
    try:
        _write_provider_block(name, block)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "accounts": keys}


async def _delete_account(name: str, account: str) -> Any:
    """Shared implementation of the account-delete endpoints."""
    _require_web_provider(name)
    block = _read_provider_block(name)
    keys = [str(k) for k in (block.get("api_keys") or [])]
    if account not in keys:
        return JSONResponse(status_code=404, content={"ok": False, "error": "account not found"})
    keys.remove(account)
    block["api_keys"] = keys
    if block.get("active_account") == account:
        block["active_account"] = keys[0] if keys else ""
    try:
        _write_provider_block(name, block)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    try:
        await web_providers.close(name, account, timeout=5)
    except Exception as e:
        logger.warning("%s could not close browser of %s: %s", name, account, e)
    return {"ok": True, "accounts": keys, "active_account": block.get("active_account", "")}


@app.delete("/api/web/{name}/accounts")
async def admin_web_delete_account(name: str, account: str) -> Any:
    """Remove an account id (or token); its browser is closed, profile kept on disk."""
    return await _delete_account(name, account)


async def _set_active(name: str, account: str) -> Any:
    """Shared implementation of the active-account endpoints."""
    _require_web_provider(name)
    block = _read_provider_block(name)
    keys = [str(k) for k in (block.get("api_keys") or [])]
    if account and keys and account not in keys:
        return JSONResponse(status_code=400, content={"ok": False, "error": "account not in api_keys"})
    block["active_account"] = account
    try:
        _write_provider_block(name, block)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "active_account": account}


@app.put("/api/web/{name}/active-account")
async def admin_web_set_active(name: str, request: Request) -> Any:
    """Switch the active account (or token) of one web provider."""
    try:
        body = await request.json()
        account = str((body or {}).get("account") or "")
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid JSON"})
    return await _set_active(name, account)


# -- deepseek-web aliases (kept so older dashboard builds keep working) ------


@app.get("/api/deepseek-web/status")
async def admin_deepseek_web_status() -> dict[str, Any]:
    """Alias of ``/api/web/deepseek-web/status``."""
    return await web_providers.get_status("deepseek-web")


@app.post("/api/deepseek-web/login")
async def admin_deepseek_web_login(request: Request) -> Any:
    """Alias of ``/api/web/deepseek-web/login``."""
    return await admin_web_login("deepseek-web", request)


@app.post("/api/deepseek-web/accounts")
async def admin_deepseek_web_add_account(request: Request) -> Any:
    """Alias of ``/api/web/deepseek-web/accounts``."""
    return await admin_web_add_account("deepseek-web", request)


@app.delete("/api/deepseek-web/accounts/{account}")
async def admin_deepseek_web_delete_account(account: str) -> Any:
    """Alias of ``/api/web/deepseek-web/accounts``."""
    return await _delete_account("deepseek-web", account)


@app.put("/api/deepseek-web/active-account")
async def admin_deepseek_web_set_active(request: Request) -> Any:
    """Alias of ``/api/web/deepseek-web/active-account``."""
    return await admin_web_set_active("deepseek-web", request)


def _response_text(response: dict[str, Any]) -> str:
    """Collect the text blocks of an Anthropic-shaped response."""
    parts = []
    for block in response.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts).strip()


def _as_response_dict(result: Any) -> dict[str, Any]:
    """Normalise a provider result (dict or JSONResponse) to a plain dict."""
    if isinstance(result, dict):
        return result
    body = getattr(result, "body", b"") or b""
    try:
        parsed = json.loads(body)
    except Exception:
        return {"content": [{"type": "text", "text": body[:300].decode("utf-8", "replace")}]}
    return parsed if isinstance(parsed, dict) else {"content": []}


async def _dispatch_model_test(request: Any, route: Any) -> Any:
    """Send one plain (no-tools, no-ReAct) request through a provider stack."""
    if web_providers.is_web_provider(route.provider_type):
        return await web_providers.send_non_streaming(request, route)
    if route.force_stream:
        return await _handle_force_stream_non_streaming(request, route, request.model)
    if route.provider_type == "gemini":
        return from_gemini_response(await gemini_provider.send_non_streaming(request, route), request.model)
    if route.provider_type == "claude":
        return await anthropic_provider.send_non_streaming(request.model_dump(exclude_none=True), route)
    raw = await openai_provider.send_non_streaming(request, route)
    if route.use_responses:
        return from_openai_responses_response(raw, request.model)
    return from_openai_chat_response(raw, request.model)


@app.post("/api/providers/{provider}/test")
async def admin_test_provider_model(provider: str, request: Request) -> dict[str, Any]:
    """Send one tiny chat through a provider/model and report what came back.

    This exercises the real provider stack (minus tools and ReAct), so a green
    result means the key, base_url and model id actually work — not that the
    config merely parses. Always answers 200 with an ``ok`` flag, so the
    dashboard can show an upstream error verbatim.

    Body (all optional): ``{"model": "...", "prompt": "...", "max_tokens": 64,
    "timeout": 60}``.
    """
    cfg = load_config()
    body: dict[str, Any] = {}
    try:
        parsed = await request.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        pass

    model = str(body.get("model") or "")
    prov = cfg.providers.get(provider)
    if prov is None or not model or model not in prov.models:
        return {"ok": False, "error": f"unknown provider/model: {provider}/{model or '?'}"}

    prompt = str(body.get("prompt") or "ping：只回复两个字「收到」")
    try:
        max_tokens = max(8, int(body.get("max_tokens") or 64))
        timeout = float(body.get("timeout") or 60.0)
    except (TypeError, ValueError):
        max_tokens, timeout = 64, 60.0

    route = ResolvedRoute(
        provider_name=provider,
        provider=prov,
        model_id=model,
        model_config=prov.models[model],
        upstream_config=cfg.upstream,
    )
    from app.models import AnthropicRequest

    test_request = AnthropicRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            response = _as_response_dict(await _dispatch_model_test(test_request, route))
    except Exception as e:
        logger.warning("model test failed [%s/%s]: %s", provider, model, e)
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__,
        }

    text = _response_text(response)
    latency_ms = int((time.monotonic() - started) * 1000)
    if not text:
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "error": "没有返回文本: " + json.dumps(response, ensure_ascii=False)[:300],
        }
    logger.info("model test ok [%s/%s] %dms", provider, model, latency_ms)
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "text": text[:500],
        "usage": response.get("usage"),
    }


@app.get("/api/config")
async def admin_get_config() -> dict[str, Any]:
    """Return raw merged config (global.json + config.json) for dashboard editing."""
    return _read_raw_config()


@app.put("/api/config")
async def admin_put_config(request: Request) -> dict[str, Any]:
    """Save raw config.json (with validation + backup) and reload runtime."""
    try:
        raw = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Invalid JSON: {e}"})
    try:
        _write_raw_config(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "reloaded": True}


@app.get("/api/billing/prices")
async def admin_get_prices() -> dict[str, Any]:
    """List named price tables, route bindings, and all available routes."""
    raw = _read_raw_config()
    prices_raw = _read_raw_prices()
    prices = prices_raw.get("prices", {})
    price_bindings = raw.get("price_bindings")
    if price_bindings is None:
        price_bindings = (raw.get("billing") or {}).get("price_bindings")
    if not price_bindings:
        price_bindings = prices_raw.get("price_bindings", {})
    fx_to_cny = prices_raw.get("fx_to_cny") or {}
    models = raw.get("models", {})
    providers = raw.get("providers", {})

    # Collect all available provider/model_id routes
    all_routes: list[str] = []
    for pname, p in (providers or {}).items():
        for mid in ((p or {}).get("models") or {}).keys():
            all_routes.append(f"{pname}/{mid}")

    # Reverse map: price_name -> list of route keys bound to it
    bound_routes: dict[str, list[str]] = {}
    for route_key, price_name in (price_bindings or {}).items():
        bound_routes.setdefault(price_name, []).append(route_key)

    # Also compute which claude models use each route (for display)
    route_to_claude: dict[str, list[str]] = {}
    for anth_model, mapping in (models or {}).items():
        if isinstance(mapping, str):
            rks = [mapping]
        elif isinstance(mapping, list):
            rks = list(mapping)
        elif isinstance(mapping, dict):
            rks = list(mapping.keys())
        else:
            rks = []
        for rk in rks:
            route_to_claude.setdefault(rk, []).append(anth_model)

    return {
        "prices": prices,
        "price_bindings": price_bindings,
        "fx_to_cny": {str(k).upper(): float(v) for k, v in fx_to_cny.items() if v is not None},
        "bound_routes": bound_routes,
        "route_to_claude": route_to_claude,
        "all_routes": all_routes,
        "models": models,
        "providers": providers,
    }


@app.put("/api/billing/prices/{name}")
async def admin_upsert_price(name: str, request: Request) -> dict[str, Any]:
    """Create or update a named price table. name = price table ID (e.g. "deepseek-v4-flash")."""
    try:
        entry = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Invalid JSON: {e}"})
    raw = _read_raw_prices()
    raw.setdefault("prices", {})[name] = entry
    try:
        _write_raw_prices(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "name": name, "entry": entry}


@app.delete("/api/billing/prices/{name}")
async def admin_delete_price(name: str) -> dict[str, Any]:
    """Delete a named price table and remove all its route bindings."""
    prices_raw = _read_raw_prices()
    prices = prices_raw.get("prices", {})
    if name not in prices:
        return JSONResponse(status_code=404, content={"ok": False, "error": f"price table '{name}' not found"})
    del prices[name]

    # Remove all bindings pointing to this price table in config.json
    cfg_raw = _read_raw_config()
    bindings = cfg_raw.get("price_bindings")
    if bindings is None:
        bindings = cfg_raw.setdefault("price_bindings", {})
    to_remove = [k for k, v in bindings.items() if v == name]
    for k in to_remove:
        del bindings[k]

    billing_bindings = (cfg_raw.get("billing") or {}).get("price_bindings", {})
    for k in to_remove:
        billing_bindings.pop(k, None)

    # Also clean up if legacy bindings in prices.json
    if "price_bindings" in prices_raw:
        for k in to_remove:
            prices_raw["price_bindings"].pop(k, None)

    try:
        _write_raw_prices(prices_raw)
        if to_remove:
            _write_raw_config(cfg_raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "deleted": name, "removed_bindings": to_remove}


@app.put("/api/billing/bindings/{route_key:path}")
async def admin_set_binding(route_key: str, request: Request) -> dict[str, Any]:
    """Bind a provider/model_id route to a named price table in config.json.

    Request body: { "price_name": string }
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Invalid JSON: {e}"})
    price_name = body.get("price_name") if isinstance(body, dict) else None
    if not isinstance(price_name, str) or not price_name:
        return JSONResponse(status_code=400, content={"ok": False, "error": "price_name required"})
    if "/" not in route_key:
        return JSONResponse(status_code=400, content={"ok": False, "error": "route_key must be provider/model_id"})
    prices_raw = _read_raw_prices()
    prices = prices_raw.get("prices", {})
    if price_name not in prices:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"price table '{price_name}' not found"})

    cfg_raw = _read_raw_config()
    cfg_raw.setdefault("price_bindings", {})[route_key] = price_name
    if "billing" in cfg_raw and isinstance(cfg_raw["billing"], dict):
        cfg_raw["billing"].setdefault("price_bindings", {})[route_key] = price_name
    try:
        _write_raw_config(cfg_raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "route_key": route_key, "price_name": price_name}


@app.delete("/api/billing/bindings/{route_key:path}")
async def admin_delete_binding(route_key: str) -> dict[str, Any]:
    """Remove a route-to-price-table binding from config.json."""
    cfg_raw = _read_raw_config()
    bindings = cfg_raw.get("price_bindings", {})
    billing_bindings = (cfg_raw.get("billing") or {}).get("price_bindings", {})
    if route_key not in bindings and route_key not in billing_bindings:
        return JSONResponse(status_code=404, content={"ok": False, "error": f"route '{route_key}' has no binding"})
    bindings.pop(route_key, None)
    billing_bindings.pop(route_key, None)
    try:
        _write_raw_config(cfg_raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "unbound": route_key}


@app.put("/api/billing/fx")
async def admin_set_fx(request: Request) -> dict[str, Any]:
    """Update FX rates used to convert official (non-CNY) prices into CNY.

    Request body: { "USD": 7.2, ... }
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Invalid JSON: {e}"})
    if not isinstance(body, dict) or not body:
        return JSONResponse(status_code=400, content={"ok": False, "error": "fx map required"})
    fx: dict[str, float] = {}
    for k, v in body.items():
        try:
            rate = float(v)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"ok": False, "error": f"invalid rate for {k}"})
        if rate <= 0:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"rate for {k} must be > 0"})
        fx[str(k).upper()] = rate
    raw = _read_raw_prices()
    raw["fx_to_cny"] = fx
    try:
        _write_raw_prices(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "fx_to_cny": fx}


@app.put("/api/models/{claude_model}")
async def admin_set_model_mapping(claude_model: str, request: Request) -> dict[str, Any]:
    """Set or update a Claude model → provider route mapping.

    Request body: { "routes": string | string[] | Record<string, number> }
    - string: single route, e.g. "beyondpower/ds"
    - string[]: multiple routes (load balanced), e.g. ["gemini/gemini-3.7-flash", "glm-yj/glm-4.7-flash"]
    - Record<string, number>: weighted routes, e.g. {"beyondpower/ds": 1, "opencode/gpt-5.6-luna": 2}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Invalid JSON: {e}"})
    routes = body.get("routes") if isinstance(body, dict) else None
    if routes is None:
        return JSONResponse(status_code=400, content={"ok": False, "error": "routes required"})
    # Validate routes reference existing providers
    raw = _read_raw_config()
    providers = raw.get("providers") or {}
    route_list = [routes] if isinstance(routes, str) else (
        list(routes) if isinstance(routes, list) else
        list(routes.keys()) if isinstance(routes, dict) else []
    )
    for r in route_list:
        if "/" not in r:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"route '{r}' must be provider/model_id"})
        pname = r.split("/", 1)[0]
        if pname not in providers:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"provider '{pname}' not in config.providers"})
    raw.setdefault("models", {})[claude_model] = routes
    try:
        _write_raw_config(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "claude_model": claude_model, "routes": routes}


@app.delete("/api/models/{claude_model}")
async def admin_delete_model_mapping(claude_model: str) -> dict[str, Any]:
    """Remove a Claude model mapping."""
    raw = _read_raw_config()
    models = raw.get("models") or {}
    if claude_model not in models:
        return JSONResponse(status_code=404, content={"ok": False, "error": f"model '{claude_model}' not found"})
    del models[claude_model]
    try:
        _write_raw_config(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "deleted": claude_model}


@app.put("/api/providers/{provider_name}/models/{model_id}")
async def admin_upsert_provider_model(provider_name: str, model_id: str, request: Request) -> dict[str, Any]:
    """Add or update a model entry under a provider."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"Invalid JSON: {e}"})
    raw = _read_raw_config()
    providers = raw.get("providers") or {}
    if provider_name not in providers:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"provider '{provider_name}' not found"})
    models = providers[provider_name].setdefault("models", {})
    models[model_id] = body
    try:
        _write_raw_config(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "provider": provider_name, "model_id": model_id, "config": body}


@app.delete("/api/providers/{provider_name}/models/{model_id}")
async def admin_delete_provider_model(provider_name: str, model_id: str) -> dict[str, Any]:
    """Remove a model entry from a provider."""
    raw = _read_raw_config()
    providers = raw.get("providers") or {}
    if provider_name not in providers:
        return JSONResponse(status_code=404, content={"ok": False, "error": f"provider '{provider_name}' not found"})
    models = providers[provider_name].get("models") or {}
    if model_id not in models:
        return JSONResponse(status_code=404, content={"ok": False, "error": f"model '{model_id}' not found under '{provider_name}'"})
    del models[model_id]
    try:
        _write_raw_config(raw)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    return {"ok": True, "deleted": f"{provider_name}/{model_id}"}


def _date_keys_in_range(days: int) -> list[str]:
    now = datetime.now(_BEIJING_TZ_ADMIN)
    return [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)][::-1]


@app.get("/api/stats")
async def admin_stats(
    range: str = "today",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Billing stats with pre-defined time dimensions.

    Args:
        range: str - One of today|yesterday|7d|30d|custom.
        start_date, end_date: str - Required only for range=custom (YYYY-MM-DD inclusive).
    """
    now = datetime.now(_BEIJING_TZ_ADMIN)
    today_str = now.strftime("%Y-%m-%d")
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    if range == "today":
        date_keys = [today_str]
    elif range == "yesterday":
        date_keys = [yesterday_str]
    elif range == "7d":
        date_keys = _date_keys_in_range(7)
    elif range == "30d":
        date_keys = _date_keys_in_range(30)
    elif range == "custom":
        if not start_date or not end_date:
            return JSONResponse(status_code=400, content={"ok": False, "error": "start_date and end_date required for range=custom"})
        try:
            d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
            d1 = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return JSONResponse(status_code=400, content={"ok": False, "error": "invalid date format, use YYYY-MM-DD"})
        if d0 > d1:
            d0, d1 = d1, d0
        date_keys = []
        cur = d0
        while cur <= d1:
            date_keys.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
    else:
        return JSONResponse(status_code=400, content={"ok": False, "error": "range must be today|yesterday|7d|30d|custom"})

    # All aggregation now happens in SQLite via billing.get_stats_range.
    result = billing.get_stats_range(date_keys)
    return {
        "range": range,
        "date_keys": date_keys,
        "total": result["total"],
        "per_day": result["per_day"],
        "recent": result["recent"],
    }


@app.get("/api/stats/recent")
async def admin_stats_recent(
    limit: int = 50,
    date: str | None = None,
) -> dict[str, Any]:
    """Return the most recent billing records.

    Args:
        limit: int - Max records to return (1-500, default 50).
        date: str | None - Optional Beijing date filter (YYYY-MM-DD).
    """
    records = billing.get_recent(limit=limit, date=date)
    return {"records": records, "total": len(records)}


@app.get("/stats")
async def stats(date: str | None = None) -> dict[str, Any]:
    """Return billing/usage statistics.

    Args:
        date: str | None - Optional Beijing date (YYYY-MM-DD). When provided,
            returns stats for that day; otherwise returns today's stats and
            a list of available dates.

    Returns:
        dict[str, Any] - Billing stats snapshot.
    """
    return billing.get_stats(date=date)


@app.delete("/stats")
async def reset_stats(date: str | None = None) -> dict[str, bool]:
    """Clear in-memory billing aggregates (does not touch the JSONL file).

    Args:
        date: str | None - Optional Beijing date (YYYY-MM-DD). When provided,
            clears aggregates for that day only; otherwise clears all days.

    Returns:
        dict[str, bool] - Confirmation that stats were reset.
    """
    billing.reset_stats(date=date)
    payload: dict[str, bool] = {"reset": True}
    if date:
        payload["date_reset"] = True
    return payload


@app.api_route("/api/hello", methods=["GET", "HEAD"])
async def api_hello() -> dict[str, str]:
    """Claude Code connectivity probe (ANTHROPIC_BASE_URL/api/hello).

    Claude Code preflight requires HTTP 200; response body is unused.
    Mirrors the common Anthropic/Bun hello shape.
    """
    return {"message": "Hello, world!"}


@app.get("/api/hello/{name}")
async def api_hello_named(name: str) -> dict[str, str]:
    """Named variant of the Claude Code hello probe."""
    return {"message": f"Hello, {name}!"}


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
    )
