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

    resp = await client.post(
        route.endpoint_url,
        json=body,
        headers=headers,
    )
    if resp.status_code >= 400:
        logger.error("Error response [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, resp.text[:1000])
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
