"""DeepSeek web provider — backwards-compatible entry point.

The implementation now lives in the generic web-provider package:

* :mod:`app.providers.web.base` — prompt flattening, chat reuse, SSE building
* :mod:`app.providers.web.engine` — Playwright transport (this site is a browser site)
* :mod:`app.providers.web.sites.deepseek` — DeepSeek's selectors, stream parser and DSML fallback
* :mod:`app.providers.web.api` — the Anthropic-facing request/response pipeline

This module keeps the historical names (``send_streaming``, ``get_status``,
``DeepSeekStreamParser``, ``_states`` ...) so existing callers, the admin
endpoints and the test suite keep working unchanged.

Debugging without the proxy:
    python -m app.providers.web.capture deepseek-web --login [--account NAME]
    python -m app.providers.web.capture deepseek-web --prompt "say hi"
"""

from __future__ import annotations

from typing import Any, Optional

from app.providers.web.api import send_non_streaming, send_streaming  # noqa: F401 (re-export)
from app.providers.web.base import (  # noqa: F401 (re-exported for older callers)
    Conversation,
    ToolCallTruncator,
    WebLoginRequired,
    WebProviderError,
    anthropic_events_from_text,
    build_full_prompt,
    build_incremental_prompt,
    flatten_prompt,
    match_conversation,
    split_request,
)
from app.providers.web.engine import _on_emit  # noqa: F401 (re-export)
from app.providers.web.registry import active_account as _active_account
from app.providers.web.registry import close as _registry_close
from app.providers.web.registry import get_status as _registry_status
from app.providers.web.registry import open_login_window as _registry_login
from app.providers.web.sites.deepseek import (  # noqa: F401 (re-exported for older callers)
    COMPOSER_SELECTOR,
    LEGACY_PROFILE_DIR,
    SITE,
    DeepSeekSite,
    DeepSeekStreamParser,
    DsmlNormalizer,
    dsml_to_tool_call,
    _account_id,
    _convs_for,
    _profile_dir,
    _profile_root,
    _state_for,
    _states,
)

#: Error aliases kept for callers that catch the DeepSeek names.
DeepSeekWebError = WebProviderError
DeepSeekWebLoginRequired = WebLoginRequired

DEFAULT_HOME = SITE.home
CHALLENGE_WAIT_TIMEOUT = 240.0


def active_account() -> str:
    """Return the configured active account id (for the admin endpoints)."""
    return _active_account("deepseek-web")


async def get_status(accounts: Optional[list[str]] = None) -> dict[str, Any]:
    """Report login state for the given account ids (default: the configured ones)."""
    return await _registry_status("deepseek-web", accounts)


async def open_login_window(account: str = "") -> dict[str, Any]:
    """Start one account's browser, show chat.deepseek.com and bring the window to the front."""
    return await _registry_login("deepseek-web", account)


async def close(account: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """Close one account's browser, or all of them (called on proxy shutdown)."""
    await _registry_close("deepseek-web", account, timeout)


def main() -> None:
    """Entry point for ``python -m app.providers.deepseek_web_provider``."""
    from app.providers.web.capture import main as capture_main

    capture_main(default_site="deepseek-web")


if __name__ == "__main__":
    main()
