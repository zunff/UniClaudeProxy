"""Web providers: one abstraction, many consumer chat sites.

A *web provider* reaches an LLM the way a human does — through a consumer chat
site with no public API. Two transports share one pipeline:

* ``browser`` (:mod:`app.providers.web.engine`) — Playwright drives the real
  page; the site's own streaming response is forwarded to Python.
* ``http`` (:mod:`app.providers.web.http`) — the site's endpoint is called
  directly with a token copied out of the logged-in browser.

Sites live in :mod:`app.providers.web.sites`: ``deepseek-web`` and ``hailuo-web``
(browser) plus ``web-http`` (any site declared in ``config.json``).
Ported DeepSeek code keeps importing from ``app.providers.deepseek_web_provider``,
which is now a thin re-export of this package.
"""

from app.providers.web.api import answer_deltas, send_non_streaming, send_streaming
from app.providers.web.base import (
    FIRST_BYTE_TIMEOUT,
    IDLE_TIMEOUT,
    TOTAL_TIMEOUT,
    Conversation,
    Normalizer,
    SseJsonParser,
    StreamParser,
    ToolCallTruncator,
    Turns,
    WebLoginRequired,
    WebProviderError,
    WebSite,
    build_full_prompt,
    build_incremental_prompt,
    flatten_prompt,
    match_conversation,
    split_request,
)
from app.providers.web.registry import (
    SITE_LIST,
    SITES,
    active_account,
    capture_token,
    close,
    configured_providers,
    ensure_provider,
    get_status,
    is_web_provider,
    looks_like_web_provider,
    open_login_window,
    runtime_for,
    site_for,
    site_for_provider,
    site_templates,
)

__all__ = [
    "FIRST_BYTE_TIMEOUT",
    "IDLE_TIMEOUT",
    "SITE_LIST",
    "SITES",
    "TOTAL_TIMEOUT",
    "Conversation",
    "Normalizer",
    "SseJsonParser",
    "StreamParser",
    "ToolCallTruncator",
    "Turns",
    "WebLoginRequired",
    "WebProviderError",
    "WebSite",
    "active_account",
    "answer_deltas",
    "capture_token",
    "build_full_prompt",
    "build_incremental_prompt",
    "close",
    "configured_providers",
    "ensure_provider",
    "flatten_prompt",
    "get_status",
    "is_web_provider",
    "looks_like_web_provider",
    "match_conversation",
    "open_login_window",
    "runtime_for",
    "send_non_streaming",
    "send_streaming",
    "site_for",
    "site_for_provider",
    "site_templates",
    "site_templates",
    "split_request",
]
