"""Public provider API shared by every web site and both transports.

The pipeline mirrors the other providers: flatten the ReAct request into one
message, run one turn on the site, then hand the text back as a minimal
Anthropic stream (one text block). Tool calls are handled upstream by
``app.react``; a site's native tool-call markup is normalised on the fly, and the
answer is cut right after a complete ``</tool_call>`` so generation stops early.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from typing import Any, AsyncIterator

from app.config import ResolvedRoute
from app.models import AnthropicRequest
from app.providers.web import registry
from app.providers.web.base import (
    ToolCallTruncator,
    estimate_tokens,
    flatten_prompt,
    message_start,
    split_request,
    stream_tail,
    text_block_start,
    text_delta,
)

logger = logging.getLogger("anyclaude")
debug_logger = logging.getLogger("anyclaude.debug")


async def answer_deltas(request: AnthropicRequest, route: ResolvedRoute) -> AsyncIterator[str]:
    """Yield answer deltas, truncated right after the first complete tool call."""
    system, turns = split_request(request)
    site = registry.site_for(route.provider_type)
    if site is None:
        raise ValueError(f"Unknown web provider type {route.provider_type!r}")
    runtime = registry.runtime_for(site.name)
    logger.info(
        "Outgoing %s request [provider=%s account=%s] messages=%d",
        site.name,
        route.provider_name,
        runtime.account_for(route.provider) or "default",
        len(turns),
    )

    normalizer = site.new_normalizer(route)
    truncator = ToolCallTruncator()
    raw = ""
    try:
        async with contextlib.aclosing(runtime.turn(system, turns, route)) as deltas:
            async for delta in deltas:
                raw += delta
                text, call_done = normalizer.feed(delta) if normalizer is not None else (delta, False)
                emit, tag_done = truncator.feed(text) if text else ("", False)
                if emit:
                    yield emit
                if call_done or tag_done:
                    logger.info("%s: tool call complete, stopping generation", site.name)
                    break
            else:
                if normalizer is not None:
                    tail = normalizer.flush()
                    if tail:
                        yield tail
    finally:
        debug_logger.info("%s RAW ANSWER: %r", site.name.upper(), raw[:6000])


async def send_streaming(request: AnthropicRequest, route: ResolvedRoute) -> AsyncIterator[str]:
    """Stream a web answer as Anthropic SSE event strings."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = estimate_tokens(flatten_prompt(request))
    started = False
    full = ""
    async with contextlib.aclosing(answer_deltas(request, route)) as deltas:
        async for delta in deltas:
            if not started:
                started = True
                yield message_start(msg_id, route.model_id, input_tokens)
                yield text_block_start()
            full += delta
            yield text_delta(delta)
    if not started:
        yield message_start(msg_id, route.model_id, input_tokens)
        yield text_block_start()
    for event in stream_tail(full):
        yield event


async def send_non_streaming(request: AnthropicRequest, route: ResolvedRoute) -> dict[str, Any]:
    """Collect a web answer into an Anthropic Messages response dict."""
    full = ""
    async with contextlib.aclosing(answer_deltas(request, route)) as deltas:
        async for delta in deltas:
            full += delta
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": route.model_id,
        "content": [{"type": "text", "text": full}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_tokens(flatten_prompt(request)),
            "output_tokens": estimate_tokens(full),
        },
    }
