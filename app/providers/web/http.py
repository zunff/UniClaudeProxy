"""HTTP transport: call a site's chat endpoint directly with a pasted token.

No Playwright, no browser window, no page to keep alive — the request the site's
own frontend would send is rebuilt here and the stream is parsed straight from
``httpx``. This is the transport for every site whose request needs nothing but
a bearer token (or cookie) the user copies once out of their logged-in browser,
and it is what makes adding a niche site a config edit rather than a program.

The site adapter supplies ``build_http_request()`` (url, headers, body) and a
stream parser; everything around it — account selection, conversation reuse,
timeouts, error mapping — lives here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import ResolvedRoute, proxy_setting
from app.providers.web.base import (
    FIRST_BYTE_TIMEOUT,
    IDLE_TIMEOUT,
    TOTAL_TIMEOUT,
    Conversation,
    CooldownTracker,
    Turns,
    WebLoginRequired,
    WebProviderError,
    WebSite,
    build_full_prompt,
    build_incremental_prompt,
    match_conversation,
    record_conversation,
)

logger = logging.getLogger("anyclaude")

#: Status codes that mean "the token you pasted is not accepted".
AUTH_STATUS = {401, 403}
#: Status codes that count as "the site is pushing back" (cooldown-worthy).
PUSHBACK_STATUS = {429, 403}


class HttpRuntime:
    """Direct-HTTP layer for one site: token selection, turns, chat reuse."""

    def __init__(self, site: WebSite) -> None:
        self.site = site
        self.convs: dict[str, list[Conversation]] = {}
        self.lock = asyncio.Lock()
        self.last_turn_end = 0.0
        self.guard = CooldownTracker(label=site.label or site.name)

    # -- site environment -------------------------------------------------
    def env(self, suffix: str, default: str = "") -> str:
        """Read ``<PREFIX>_<SUFFIX>`` for this site."""
        if not self.site.env_prefix:
            return default
        return os.environ.get(f"{self.site.env_prefix}_{suffix}", default)

    def _env_float(self, suffix: str, default: float) -> float:
        try:
            return float(self.env(suffix, str(default)))
        except ValueError:
            return default

    def _env_int(self, suffix: str, default: int) -> int:
        try:
            return int(self.env(suffix, str(default)))
        except ValueError:
            return default

    def _refresh_guard(self) -> None:
        """Re-read the cooldown settings (env can change without a restart)."""
        self.guard.threshold = max(1, self._env_int("FAILURE_THRESHOLD", 2))
        self.guard.cooldown_seconds = max(0.0, self._env_float("COOLDOWN_MINUTES", 30.0) * 60.0)

    async def _pace(self) -> None:
        """Keep a minimum gap between turns so the token is not hammered.

        The browser transport has always done this; the direct one needs it just
        as much, because nothing else in the path slows requests down.
        """
        interval = self._env_float("MIN_INTERVAL", 3.0)
        wait = interval - (time.monotonic() - self.last_turn_end)
        if self.last_turn_end and wait > 0:
            await asyncio.sleep(wait)

    # -- accounts (one token each) ----------------------------------------
    def account_for(self, provider: Any) -> str:
        """Return the token/account this request uses (``""`` when unconfigured)."""
        if provider is None:
            return ""
        active = getattr(provider, "active_account", "") or ""
        keys = provider.resolved_api_keys() if hasattr(provider, "resolved_api_keys") else []
        if active and active in keys:
            return active
        if keys:
            return keys[0]
        return active

    @staticmethod
    def mask(token: str) -> str:
        """Short, non-reversible label for a token (logs and the dashboard)."""
        if not token:
            return ""
        return token[:4] + "…" + token[-4:] if len(token) > 12 else "…"

    def convs_for(self, account: str) -> list[Conversation]:
        """Return the tracked chats of one account."""
        return self.convs.setdefault(account, [])

    # -- turn -------------------------------------------------------------

    def _lock_timeout(self) -> float:
        """How long a request may queue behind the turn in progress."""
        try:
            return float(os.environ.get(f"{self.site.env_prefix}_LOCK_TIMEOUT", "600"))
        except (TypeError, ValueError):
            return 600.0

    @contextlib.asynccontextmanager
    async def _locked(self, account: str):
        """Hold the site lock, but never wait forever behind a stuck turn."""
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=self._lock_timeout())
        except asyncio.TimeoutError:
            raise WebProviderError(
                f"{self.site.label or self.site.name} is busy: a previous turn has been running for "
                f"more than {self._lock_timeout():.0f}s. Check the browser window, or restart the proxy "
                "to drop it."
            ) from None
        try:
            yield
        finally:
            self.lock.release()

    async def turn(self, system: str, turns: Turns, route: ResolvedRoute) -> AsyncIterator[str]:
        """POST one turn to the site and yield answer deltas.

        Continues the site-side chat when the request extends a conversation we
        already have (the site keeps the history), otherwise replays the whole
        history in a new chat.
        """
        account = self.account_for(route.provider)
        if not account:
            raise WebLoginRequired(
                f"{self.site.label or self.site.name} has no token configured. "
                f"Log in at {self.site.home}, copy the token the page uses, and put it in this "
                "provider's api_key (or api_keys)."
            )
        self._refresh_guard()
        self.guard.check(account)
        convs = self.convs_for(account)
        async with self._locked(account):
            await self._pace()
            conv = match_conversation(convs, system, turns)
            prompt = build_incremental_prompt(system, turns) if conv else build_full_prompt(system, turns)
            method, url, headers, body = self.site.build_http_request(route, system, turns, conv, prompt)
            logger.info(
                "%s turn [token=%s %s] prompt_chars=%d",
                self.site.name,
                self.mask(account),
                "continue" if conv else "new chat",
                len(prompt),
            )

            parser = self.site.new_parser(route)
            normalizer = self.site.new_normalizer(route)
            got_text = False
            failed = False
            t0 = time.monotonic()
            timeout = httpx.Timeout(connect=FIRST_BYTE_TIMEOUT, read=IDLE_TIMEOUT, write=60.0, pool=30.0)
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True, proxy=proxy_setting()
                ) as client:
                    async with client.stream(
                        method, url, headers=headers, json=body
                    ) as resp:
                        if resp.status_code >= 400:
                            detail = (await resp.aread())[:400].decode("utf-8", "replace")
                            if resp.status_code in PUSHBACK_STATUS:
                                self.guard.record(account, f"HTTP {resp.status_code}")
                            if resp.status_code in AUTH_STATUS:
                                raise WebLoginRequired(
                                    f"{self.site.label or self.site.name} rejected the token "
                                    f"(HTTP {resp.status_code}). Copy a fresh one from the logged-in browser. "
                                    f"Body: {detail!r}"
                                )
                            raise WebProviderError(
                                f"{self.site.label or self.site.name} returned HTTP {resp.status_code}: {detail!r}"
                            )

                        async def _chunks() -> AsyncIterator[str]:
                            async for chunk in resp.aiter_text():
                                yield chunk

                        deadline = t0 + TOTAL_TIMEOUT
                        stream = _chunks().__aiter__()
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise WebProviderError(f"{self.site.name} turn exceeded {TOTAL_TIMEOUT:.0f}s")
                            try:
                                chunk = await asyncio.wait_for(
                                    stream.__anext__(),
                                    timeout=IDLE_TIMEOUT if got_text else FIRST_BYTE_TIMEOUT,
                                )
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                phase = "idle" if got_text else "first byte"
                                raise WebProviderError(
                                    f"Timed out waiting for the {self.site.name} stream ({phase})"
                                ) from None

                            for kind, delta in parser.feed(chunk):
                                if kind == "text":
                                    if not got_text:
                                        got_text = True
                                        logger.info(
                                            "%s first-token latency: %.2fs",
                                            self.site.name,
                                            time.monotonic() - t0,
                                        )
                                    if normalizer is not None:
                                        emit, _done = normalizer.feed(delta)
                                        if emit:
                                            yield emit
                                    else:
                                        yield delta
                            if parser.done:
                                break

                        for kind, delta in parser.flush():
                            if kind == "text":
                                got_text = True
                                if normalizer is not None:
                                    emit, _done = normalizer.feed(delta)
                                    if emit:
                                        yield emit
                                else:
                                    yield delta
                if normalizer is not None:
                    tail = normalizer.flush()
                    if tail:
                        yield tail
                if not got_text:
                    raise WebProviderError(
                        f"{self.site.name} returned no answer text. Stream head: {parser.raw_head[:300]!r}"
                    )
                logger.info("%s turn done in %.2fs", self.site.name, time.monotonic() - t0)
            except GeneratorExit:
                raise
            except BaseException:
                failed = True
                raise
            finally:
                ref = getattr(parser, "chat_ref", None)
                if got_text and not failed:
                    record_conversation(convs, conv, system, turns, ref)
                    self.guard.success(account)
                elif conv is not None and conv in convs:
                    convs.remove(conv)
                self.last_turn_end = time.monotonic()

    # -- admin helpers ----------------------------------------------------
    async def status(self, account: str) -> dict[str, Any]:
        """Report one token's state (there is no page or login to inspect)."""
        return {
            "account": self.mask(account),
            "transport": "http",
            "profile_dir": None,
            "headless": True,
            "browser_running": False,
            "busy": self.lock.locked(),
            "logged_in": None,
            "url": self.site.home,
            "cooldown_seconds_left": round(self.guard.remaining(account)),
        }

    async def open_login_window(self, account: str, route: Optional[ResolvedRoute] = None) -> dict[str, Any]:
        """HTTP sites have no window; explain how to get a token instead."""
        raise WebProviderError(
            f"{self.site.label or self.site.name} is called directly with a token, so it has no browser "
            f"window. Open {self.site.home}, log in, and copy the token/account key it uses "
            f"({self.site.token_hint}) into this provider's api_key in config.json."
        )

    async def close(self, account: Optional[str] = None, timeout: Optional[float] = None) -> None:
        """Nothing to close (kept for API symmetry with the browser transport)."""
        return None
