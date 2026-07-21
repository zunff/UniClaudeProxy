import json
import logging
import asyncio
from typing import Any, AsyncIterator

import httpx

from app.config import ResolvedRoute
from app.providers.retry_utils import (
    FirstByteTimeoutError,
    policy_from_route,
    route_key,
    run_with_retry,
    should_bypass_upstream_policy,
    stream_with_retry,
)

logger = logging.getLogger("anyclaude.provider")
debug_logger = logging.getLogger("uniclaudeproxy.debug")

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client for Anthropic passthrough.

    Returns:
        httpx.AsyncClient - The shared HTTP client instance.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            headers={"User-Agent": "claude-cli/1.0.53 (external, cli)"},
        )
    return _client


async def close_client() -> None:
    """Close the shared HTTP client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None


def _build_headers(route: ResolvedRoute) -> dict[str, str]:
    """Build headers for Anthropic passthrough request.

    Args:
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, str] - Request headers.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "claude-cli/1.0.53 (external, cli)",
    }
    headers.update(route.provider.headers)
    return headers


def _build_body(raw_body: dict[str, Any], route: ResolvedRoute) -> dict[str, Any]:
    """Build the request body for Anthropic passthrough.

    Replaces the model name with the upstream model ID.

    Args:
        raw_body: dict[str, Any] - The original Anthropic request body.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The modified request body.
    """
    body = dict(raw_body)
    body["model"] = route.model_id
    # Add extra_body parameters
    if route.extra_body:
        body.update(route.extra_body)
    return body


async def send_non_streaming(
    raw_body: dict[str, Any],
    route: ResolvedRoute,
) -> dict[str, Any]:
    """Send a non-streaming passthrough request to an Anthropic-compatible provider.

    Args:
        raw_body: dict[str, Any] - The raw Anthropic request body.
        route: ResolvedRoute - Resolved routing information.

    Returns:
        dict[str, Any] - The raw JSON response from the provider.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_body(raw_body, route)
    url = route.endpoint_url
    headers = _build_headers(route)
    client = await get_client()

    # Log request details for debugging
    debug_logger.debug("Anthropic passthrough request URL: %s", url)
    debug_logger.debug("Anthropic passthrough request headers: %s", json.dumps(headers))
    debug_logger.debug("Anthropic passthrough request body: %s", json.dumps(body))

    logger.info("Outgoing Anthropic passthrough request [provider=%s] to %s", route.provider_name, url)

    bypass, matched_rule = should_bypass_upstream_policy(route)
    if bypass:
        logger.info(
            "Upstream policy bypassed [route=%s matched_rule=%s]",
            route_key(route),
            matched_rule,
        )
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            logger.error("Anthropic passthrough error [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, resp.text[:1000])
        resp.raise_for_status()
        return resp.json()

    policy = policy_from_route(route, stream=False)

    async def _send_once(first_byte_timeout_ms: int) -> dict[str, Any]:
        timeout_seconds = first_byte_timeout_ms / 1000.0
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = (await resp.aread()).decode("utf-8", errors="replace")
                logger.error("Anthropic passthrough error [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, error_body[:1000])
            resp.raise_for_status()

            chunks: list[bytes] = []
            iterator = resp.aiter_bytes()
            try:
                first_chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise FirstByteTimeoutError("Upstream did not return first byte in time") from exc
            except StopAsyncIteration:
                first_chunk = b""

            if first_chunk:
                chunks.append(first_chunk)
            async for chunk in iterator:
                chunks.append(chunk)

        raw_text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if not raw_text:
            raise ValueError("Anthropic passthrough returned empty response body")
        return json.loads(raw_text)

    return await run_with_retry(
        _send_once,
        policy=policy,
        logger=logger,
        provider_name=route.provider_name,
        model_id=route.model_id,
    )


async def send_streaming(
    raw_body: dict[str, Any],
    route: ResolvedRoute,
) -> AsyncIterator[bytes]:
    """Send a streaming passthrough request to an Anthropic-compatible provider.

    Args:
        raw_body: dict[str, Any] - The raw Anthropic request body.
        route: ResolvedRoute - Resolved routing information.

    Yields:
        bytes - Raw SSE line chunks from the provider, passed through as-is.

    Raises:
        httpx.HTTPStatusError: If the provider returns a non-2xx status.
    """
    body = _build_body(raw_body, route)
    body["stream"] = True
    url = route.endpoint_url
    headers = _build_headers(route)
    client = await get_client()

    logger.info("Outgoing Anthropic passthrough streaming request [provider=%s] to %s", route.provider_name, url)

    bypass, matched_rule = should_bypass_upstream_policy(route)
    if bypass:
        logger.info(
            "Upstream policy bypassed [route=%s matched_rule=%s]",
            route_key(route),
            matched_rule,
        )
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                logger.error("Anthropic passthrough streaming error [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, error_body.decode()[:1000])
                resp.raise_for_status()

            async for chunk in resp.aiter_bytes():
                yield chunk
        return

    policy = policy_from_route(route, stream=True)

    async def _stream_once() -> AsyncIterator[bytes]:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                logger.error("Anthropic passthrough streaming error [provider=%s] (status=%d): %s", route.provider_name, resp.status_code, error_body.decode()[:1000])
                resp.raise_for_status()

            async for chunk in resp.aiter_bytes():
                yield chunk

    async for chunk in stream_with_retry(
        _stream_once,
        policy=policy,
        logger=logger,
        provider_name=route.provider_name,
        model_id=route.model_id,
    ):
        yield chunk
