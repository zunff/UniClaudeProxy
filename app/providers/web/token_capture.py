"""Grab a site's bearer token from a real login window.

Direct-HTTP sites (`web-http` and any module that adds one) need a token the page
normally keeps
to itself. Copying it out of devtools is the usual advice, but it is tedious and
easy to get wrong — so the proxy opens a real browser, lets the user log in
normally, and *reads the token the page itself sends*.

The page driver is injected with ``__lxTokenCapture`` set, which makes it report
the ``Authorization`` header of every request the site makes. That is exactly the
value our HTTP transport has to send, and it works for any site whose auth is a
bearer header (no per-site reverse engineering). If the site never sends a
request with an auth header, the login state is scanned in ``localStorage`` as a
fallback.

Nothing here writes config: the caller (``registry.capture_token``) saves the
token. Playwright is imported lazily, so HTTP-only setups never touch it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.providers.web.base import WebProviderError, WebSite
from app.providers.web.engine import DRIVER_JS_PATH

logger = logging.getLogger("anyclaude")

PAGE_READY_TIMEOUT = 25.0
#: How long to keep the window open waiting for the user to log in and chat.
DEFAULT_TIMEOUT = 300.0

#: Read every localStorage entry, so Python can pick a token-looking value.
_LOCAL_STORAGE_JS = """() => {
    const out = {};
    try {
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            out[k] = localStorage.getItem(k);
        }
    } catch (e) { /* storage blocked */ }
    return out;
}"""

#: Keys whose value is worth looking at when no request carried a header.
#: Deliberately narrow: a loose match picks up device ids and analytics junk and
#: saves them as if they were credentials.
_TOKEN_KEY_RE = re.compile(
    r"^(?:(?:access|auth|user|refresh|api)[_-]?)?(?:token|key)$|^authorization$",
    re.I,
)


def clean_token(raw: Any) -> Optional[str]:
    """Normalise a token as copied/stored by a page frontend.

    Handles ``"Bearer xxx"``, a JSON envelope (``{"value": "xxx"}`` /
    ``{"userToken": ...}``), and surrounding quotes/spaces.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip().strip('"').strip("'").strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if value.startswith("{"):
        try:
            inner = json.loads(value)
        except json.JSONDecodeError:
            return None
        for key in ("value", "token", "access_token", "accessToken", "userToken"):
            if isinstance(inner, dict) and isinstance(inner.get(key), str):
                return clean_token(inner[key])
        return None
    if len(value) < 16 or any(ch.isspace() for ch in value):
        return None
    return value


def looks_like_token(value: str) -> bool:
    """True for a JWT or a long opaque base64url-ish string."""
    if value.startswith("eyJ"):
        return True
    return len(value) >= 40 and re.fullmatch(r"[A-Za-z0-9._~+/=\-]+", value) is not None


def token_from_storage(storage: dict[str, Any]) -> Optional[str]:
    """Pick the most token-like value out of a page's localStorage dump.

    Requires *both* a token-ish key name and a token-shaped value: anonymous
    visitors already carry plenty of long opaque strings (device ids, analytics
    keys), and saving one of those as the API token only looks like it worked.
    """
    candidates: list[str] = []
    for key, raw in (storage or {}).items():
        if not _TOKEN_KEY_RE.match(str(key)):
            continue
        value = clean_token(raw)
        if value is None or not looks_like_token(value):
            continue
        candidates.append(value)
    if not candidates:
        return None
    return max(candidates, key=len)


async def capture_bearer_token(
    site: WebSite,
    runtime: Any,
    account: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    verify: Optional[Callable[[str], Awaitable[bool]]] = None,
) -> Optional[str]:
    """Open ``site.home``, wait for a logged-in session, return its bearer token.

    ``runtime`` provides the profile directory and the site's env settings; it is
    the site's *browser* runtime (or a throwaway one) and is only read here.

    ``verify`` is called with every candidate token; returning False means "the
    site rejected this one" (typically an anonymous visitor token seen before the
    user logged in), so the window stays open and we keep waiting.
    """
    rejected = 0

    async def _accept(candidate: Optional[str], origin: str) -> Optional[str]:
        nonlocal rejected
        if not candidate:
            return None
        if verify is not None and not await verify(candidate):
            rejected += 1
            logger.info("%s: %s token rejected by the site, still waiting for a logged-in one",
                        site.name, origin)
            return None
        logger.info("%s token captured from %s", site.name, origin)
        return candidate
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise WebProviderError(
            "playwright is not installed. Run: pip install -r requirements-deepseek-web.txt"
        ) from e

    profile = runtime.profile_dir(account)
    Path(profile).mkdir(parents=True, exist_ok=True)
    channel = runtime.env("CHANNEL", "chrome") or None

    queue: asyncio.Queue = asyncio.Queue()

    def _on_token(payload: str) -> None:
        try:
            queue.put_nowait(json.loads(payload))
        except (json.JSONDecodeError, asyncio.QueueFull):
            pass

    pw = await async_playwright().start()
    context = None
    try:
        try:
            context = await pw.chromium.launch_persistent_context(
                profile,
                channel=channel,
                headless=False,  # the user has to log in by hand
                chromium_sandbox=True,
                ignore_default_args=["--enable-automation"],
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception as e:
            raise WebProviderError(f"Failed to launch browser (channel={channel!r}): {e}") from e

        await context.expose_function("__lxToken", _on_token)
        scope = {
            "__lxSite": site.name,
            "__lxAccount": account,
            "__lxStreamPaths": [],
            "__lxTokenCapture": True,
        }
        await context.add_init_script(f"Object.assign(window, {json.dumps(scope)});")
        await context.add_init_script(path=str(DRIVER_JS_PATH))

        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(site.home, wait_until="domcontentloaded", timeout=int(PAGE_READY_TIMEOUT * 1000))
        with contextlib.suppress(Exception):
            await page.bring_to_front()
        logger.info(
            "%s token capture window open [account=%s profile=%s]; log in and send one message",
            site.name,
            account or "default",
            profile,
        )

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            with contextlib.suppress(asyncio.TimeoutError):
                msg = await asyncio.wait_for(queue.get(), timeout=5.0)
                token = await _accept(
                    clean_token(msg.get("authorization") if isinstance(msg, dict) else None),
                    "a page request",
                )
                if token:
                    return token
            # Fallback for sites that only touch their API after a user action.
            with contextlib.suppress(Exception):
                storage = await page.evaluate(_LOCAL_STORAGE_JS)
                token = await _accept(
                    token_from_storage(storage if isinstance(storage, dict) else {}),
                    "localStorage",
                )
                if token:
                    return token
        logger.warning("%s token capture timed out after %.0fs (%d candidate(s) rejected)",
                       site.name, timeout, rejected)
        return None
    finally:
        if context is not None:
            with contextlib.suppress(Exception):
                await context.close()
        with contextlib.suppress(Exception):
            await pw.stop()
