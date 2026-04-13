import json
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import config_path, load_config, reload_config, resolve_route
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

    watcher = ConfigWatcher(config_path(), _on_config_change)
    watcher.start()

    yield

    watcher.stop()
    await openai_provider.close_client()
    await gemini_provider.close_client()
    await anthropic_provider.close_client()
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

    use_react = route.model_config.use_react
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
        logger.error("Claude passthrough error: %s", e)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Provider error: {e}"},
            },
        )

    # Check if the response is an error
    if raw_response.get("code") != 0 and raw_response.get("success") is False:
        logger.error("Upstream provider error: %s", raw_response.get("msg"))
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
                if route.provider_type == "gemini":
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
                logger.error("ReAct streaming error: %s", e)
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
        if route.provider_type == "gemini":
            raw_response = await gemini_provider.send_non_streaming(request, route)
            anthropic_response = from_gemini_response(raw_response, anthropic_model)
        elif route.use_responses:
            raw_response = await openai_provider.send_non_streaming(request, route)
            anthropic_response = from_openai_responses_response(raw_response, anthropic_model)
        else:
            raw_response = await openai_provider.send_non_streaming(request, route)
            anthropic_response = from_openai_chat_response(raw_response, anthropic_model)
    except Exception as e:
        logger.error("ReAct provider error: %s", e)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": f"Provider error: {e}"},
            },
        )

    anthropic_response = react_transform_response(anthropic_response, anthropic_model)

    logger.info(
        "ReAct response: model=%s, stop_reason=%s, blocks=%d",
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

    try:
        if route.provider_type == "gemini":
            _pi = build_tool_param_index(request.tools) if request.tools else None
            raw_stream = gemini_provider.send_streaming(request, route)
            converter = stream_gemini_to_anthropic(raw_stream, anthropic_model, param_index=_pi)
        elif route.use_responses:
            raw_stream = openai_provider.send_streaming(request, route)
            converter = stream_openai_responses_to_anthropic(raw_stream, anthropic_model, tool_mapping=route.tool_mapping or None)
        else:
            raw_stream = openai_provider.send_streaming(request, route)
            converter = stream_openai_chat_to_anthropic(raw_stream, anthropic_model)

        text_parts: list[str] = []
        content_blocks: list[dict] = []
        stop_reason = "end_turn"
        usage = {"input_tokens": 0, "output_tokens": 0}
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

    except Exception as e:
        logger.error("Force-stream non-streaming error: %s", e)
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
    try:
        if route.provider_type == "gemini":
            raw_response = await gemini_provider.send_non_streaming(request, route)
        else:
            raw_response = await openai_provider.send_non_streaming(request, route)
        # Log raw response for debugging
        debug_logger.debug("Raw provider response: %s", json.dumps(raw_response))
        
        # Check if the response is an error
        if raw_response.get("code") != 0 and raw_response.get("success") is False:
            logger.error("Upstream provider error: %s", raw_response.get("msg"))
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": raw_response.get("msg", "Unknown provider error")},
                },
            )
    except Exception as e:
        logger.error("Provider request failed: %s", e)
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
        logger.error("Response conversion failed: %s", e)
        logger.error("Raw response that caused conversion error: %s", json.dumps(raw_response))
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

    logger.info(
        "Response: model=%s, stop_reason=%s, output_tokens=%d, input_tokens=%d",
        anthropic_model,
        anthropic_response.get("stop_reason", "unknown"),
        anthropic_response.get("usage", {}).get("output_tokens", 0),
        anthropic_response.get("usage", {}).get("input_tokens", 0),
    )

    return JSONResponse(content=anthropic_response)


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
        try:
            if route.provider_type == "gemini":
                _pi = build_tool_param_index(request.tools) if request.tools else None
                raw_stream = gemini_provider.send_streaming(request, route)
                async for event in stream_gemini_to_anthropic(raw_stream, anthropic_model, param_index=_pi):
                    # Log streaming events for debugging
                    debug_logger.debug("Streaming event: %s", event.strip())
                    yield event
            elif route.use_responses:
                raw_stream = openai_provider.send_streaming(request, route)
                async for event in stream_openai_responses_to_anthropic(raw_stream, anthropic_model, tool_mapping=route.tool_mapping or None):
                    # Log streaming events for debugging
                    debug_logger.debug("Streaming event: %s", event.strip())
                    yield event
            else:
                raw_stream = openai_provider.send_streaming(request, route)
                async for event in stream_openai_chat_to_anthropic(raw_stream, anthropic_model):
                    # Log streaming events for debugging
                    debug_logger.debug("Streaming event: %s", event.strip())
                    yield event

        except Exception as e:
            logger.error("Streaming error: %s", e)
            error_event = json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": f"Streaming error: {e}"},
            })
            yield f"event: error\ndata: {error_event}\n\n"

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


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
    )
