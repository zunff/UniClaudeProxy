import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import ResolvedRoute
from app.converters.anthropic_to_openai import (
    to_openai_chat_request,
    to_openai_responses_request,
)
from app.models import AnthropicRequest

logger = logging.getLogger("anyclaude.provider")
debug_logger = logging.getLogger("anyclaude.debug")

_client: httpx.AsyncClient | None = None


def _should_include_reasoning_content(route: ResolvedRoute) -> bool:
    """Whether assistant reasoning_content should be sent for chat-completions."""
    endpoint = route.endpoint_url.lower()
    provider_name = route.provider_name.lower()
    model_id = route.model_id.lower()
    return ("deepseek" in endpoint) or ("deepseek" in provider_name) or ("deepseek" in model_id)


def _log_request_body_summary(route: ResolvedRoute, body: dict[str, Any]) -> None:
    """Log a compact request summary for troubleshooting provider issues."""
    messages = body.get("messages")
    if isinstance(messages, list):
        role_counts: dict[str, int] = {}
        reasoning_messages = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            if msg.get("reasoning_content"):
                reasoning_messages += 1
        logger.info(
            "Request summary [provider=%s]: messages=%d roles=%s reasoning_messages=%d stream=%s",
            route.provider_name,
            len(messages),
            role_counts,
            reasoning_messages,
            body.get("stream"),
        )
    else:
        input_items = body.get("input")
        if isinstance(input_items, list):
            logger.info(
                "Request summary [provider=%s]: input_items=%d stream=%s",
                route.provider_name,
                len(input_items),
                body.get("stream"),
            )


def _sample_message_meta(messages: Any, limit: int = 3) -> list[dict[str, Any]]:
    """Build a tiny metadata snapshot of chat messages for error logs."""
    if not isinstance(messages, list):
        return []
    sample: list[dict[str, Any]] = []
    for msg in messages[:limit]:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        content_len = len(content) if isinstance(content, str) else -1
        sample.append({
            "role": msg.get("role"),
            "has_reasoning_content": bool(msg.get("reasoning_content")),
            "has_tool_calls": bool(msg.get("tool_calls")),
            "content_len": content_len,
        })
    return sample


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client.

    Returns:
        httpx.AsyncClient - The shared HTTP client instance.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    return _client


async def close_client() -> None:
    """Close the shared async HTTP client."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


def _build_request_body(request: AnthropicRequest, route: ResolvedRoute) -> dict[str, Any]:
    """Build the provider request body based on route configuration.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The provider-specific request body.
    """
    if route.use_responses:
        body = to_openai_responses_request(
            request,
            route.model_id,
            inject_context=route.inject_context,
            upstream_system=route.upstream_system,
            reasoning=route.reasoning or None,
            truncation=route.truncation,
            text=route.text or None,
            max_output_tokens=route.max_output_tokens,
            parallel_tool_calls=route.parallel_tool_calls,
            image_mode=route.image_mode,
            image_dir=route.image_dir,
        )
    else:
        body = to_openai_chat_request(
            request,
            route.model_id,
            max_output_tokens=route.max_output_tokens,
            include_reasoning_content=_should_include_reasoning_content(route),
        )
    
    # Add extra_body parameters
    if route.extra_body:
        body.update(route.extra_body)
    
    return body


async def send_non_streaming(
    request: AnthropicRequest,
    route: ResolvedRoute,
) -> dict[str, Any]:
    """Send a non-streaming request to an OpenAI-compatible provider.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The raw JSON response from the provider.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_request_body(request, route)
    headers = route.build_headers()
    client = await get_client()

    logger.info("Outgoing request [provider=%s] to %s", route.provider_name, route.endpoint_url)
    _log_request_body_summary(route, body)

    resp = await client.post(
        route.endpoint_url,
        json=body,
        headers=headers,
    )
    if resp.status_code >= 400:
        logger.error("Error response [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, resp.text[:1000])
        if route.provider_name.lower().find("deepseek") >= 0 or route.endpoint_url.lower().find("deepseek") >= 0:
            logger.error(
                "DeepSeek diagnostics: include_reasoning_content=%s sample_messages=%s",
                _should_include_reasoning_content(route),
                _sample_message_meta(body.get("messages", []), limit=3),
            )
    resp.raise_for_status()

    raw_text = resp.text.strip()

    if not raw_text:
        raise ValueError(f"Provider returned empty response (status {resp.status_code})")

    try:
        response_data = resp.json()
        # Check if the response contains an error
        if response_data.get("code") != 0 and response_data.get("success") is False:
            logger.error("Provider error response: %s", response_data.get("msg"))
            raise ValueError(f"Provider returned error: {response_data.get('msg', 'Unknown error')}")
        return response_data
    except Exception as e:
        logger.error("Failed to parse provider response: %s | Body: %s", e, raw_text[:500])
        raise ValueError(f"Provider returned non-JSON response: {raw_text[:200]}") from e


async def send_streaming(
    request: AnthropicRequest,
    route: ResolvedRoute,
) -> AsyncIterator[bytes]:
    """Send a streaming request to an OpenAI-compatible provider.

    The async generator keeps the HTTP connection alive for the duration
    of iteration. The httpx stream context manager closes when the
    generator is fully consumed or garbage collected.

    Args:
        request: AnthropicRequest - The incoming Anthropic-formatted request.
        route: ResolvedRoute - Resolved routing information.

    Yields:
        bytes - Raw SSE line chunks from the provider (each line terminated with newline).

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_request_body(request, route)
    headers = route.build_headers()
    headers["Accept"] = "text/event-stream"
    client = await get_client()

    logger.info("Outgoing streaming request [provider=%s] to %s", route.provider_name, route.endpoint_url)
    _log_request_body_summary(route, body)

    async with client.stream(
        "POST",
        route.endpoint_url,
        json=body,
        headers=headers,
    ) as resp:
        if resp.status_code >= 400:
            error_body = await resp.aread()
            logger.error("Stream error [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, error_body.decode("utf-8", errors="replace")[:1000])
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            yield (line + "\n").encode("utf-8")
