"""Browser transport: drive a consumer chat page with Playwright.

One :class:`BrowserRuntime` exists per site (``deepseek-web``, ...); it owns one
browser, one persistent profile and one lock per account. ``driver.js`` forwards
the page's own streaming response to :func:`_on_emit`, which routes it to the
queue of the turn currently running on that browser.

Everything here is site-agnostic — selectors, probes and the stream parser come
from the :class:`~app.providers.web.base.WebSite` adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, AsyncIterator, Optional

from app.config import ResolvedRoute
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
debug_logger = logging.getLogger("anyclaude.debug")

DRIVER_JS_PATH = Path(__file__).with_name("driver.js")
PAGE_READY_TIMEOUT = 25.0
CHALLENGE_WAIT_TIMEOUT = 240.0  # how long to wait in place for a captcha to be solved
COMPOSER_WAIT = 10.0  # how long to look for a *visible* composer before using the first match
SUBMIT_GRACE = 12.0  # how long to allow a stream to start when the prompt may not have been sent

#: Site a driver message without ``__site`` belongs to (pre-multi-site pages).
DEFAULT_SITE = "deepseek-web"


class BrowserState:
    """Playwright handles for one account and the queue of its active turn."""

    def __init__(self, site: WebSite, account: str) -> None:
        self.site = site
        self.account = account
        self.pw: Any = None
        self.context: Any = None
        self.page: Any = None
        self.queue: Optional[asyncio.Queue] = None
        self.provider: Any = None


_RUNTIMES: dict[str, "BrowserRuntime"] = {}


def _on_emit(payload: str) -> None:
    """Binding called from a page; routes driver messages to the active turn.

    The init script is added per browser context; the payload carries the site
    and account ids so several browsers can share one callback.
    """
    try:
        msg = json.loads(payload)
    except json.JSONDecodeError:
        return
    site = msg.pop("__site", "") or DEFAULT_SITE
    account = msg.pop("__account", "")
    runtime = _RUNTIMES.get(site)
    if runtime is None:
        return
    state = runtime.states.get(account)
    queue = state.queue if state is not None else None
    if queue is None:
        return
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        pass


class BrowserRuntime:
    """Browser layer for one site: per-account browsers, serial turns, chat reuse."""

    def __init__(self, site: WebSite) -> None:
        self.site = site
        self.states: dict[str, BrowserState] = {}
        self.convs: dict[str, list[Conversation]] = {}
        self.lock = asyncio.Lock()
        self.last_turn_end = 0.0
        self._registered = False
        self.guard = CooldownTracker(label=site.label or site.name)

    # -- registry ---------------------------------------------------------
    def register(self) -> "BrowserRuntime":
        """Make this runtime reachable from the page binding."""
        _RUNTIMES[self.site.name] = self
        self._registered = True
        return self

    # -- accounts & profiles ---------------------------------------------
    def state_for(self, account: str) -> BrowserState:
        """Return (creating on demand) the browser state of one account."""
        state = self.states.get(account)
        if state is None:
            state = self.states[account] = BrowserState(self.site, account)
        return state

    def convs_for(self, account: str) -> list[Conversation]:
        """Return the tracked web chats of one account."""
        return self.convs.setdefault(account, [])

    def env(self, suffix: str, default: str = "") -> str:
        """Read ``<PREFIX>_<SUFFIX>`` for this site."""
        if not self.site.env_prefix:
            return default
        return os.environ.get(f"{self.site.env_prefix}_{suffix}", default)

    def env_flag(self, suffix: str, default: bool = False) -> bool:
        """Read a boolean site environment variable ("1" is true)."""
        raw = self.env(suffix, "1" if default else "0")
        return raw == "1"

    def env_float(self, suffix: str, default: float) -> float:
        """Read a numeric site environment variable."""
        try:
            return float(self.env(suffix, str(default)))
        except ValueError:
            return default

    def refresh_guard(self) -> None:
        """Re-read the cooldown settings (env can change without a restart)."""
        self.guard.threshold = max(1, int(self.env_float("FAILURE_THRESHOLD", 2)))
        self.guard.cooldown_seconds = max(0.0, self.env_float("COOLDOWN_MINUTES", 30.0) * 60.0)

    def profile_root(self) -> Path:
        """Base directory holding one browser profile per named account."""
        return Path(self.env("PROFILE_DIR") or Path("logs") / self.site.profile_dirname)

    @staticmethod
    def safe_dir_name(account: str) -> str:
        """Turn an account id into a directory name that is always valid.

        Account ids are user-supplied and, for HTTP sites, may be a 300-char
        token; used verbatim that becomes an illegal Windows path. Keep a
        readable prefix and a stable hash suffix instead.
        """
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", account).strip("._-")
        if len(cleaned) > 32 or cleaned != account:
            digest = hashlib.sha1(account.encode("utf-8")).hexdigest()[:8]
            cleaned = (cleaned[:24] or "acct") + "-" + digest
        return cleaned or "default"

    def profile_dir(self, account: str = "") -> str:
        """Profile directory for one account.

        The default account (empty id) keeps the site's legacy single-profile
        path so an existing login survives; named accounts get
        ``<root>/<account>`` (sanitised — see :meth:`safe_dir_name`).
        """
        base = self.env("PROFILE_DIR")
        if not account:
            return str(Path(base or self.site.legacy_profile_dir or self.profile_root()))
        return str(Path(base or self.profile_root()) / self.safe_dir_name(account))

    # -- browser lifecycle ------------------------------------------------
    async def _launch(self, state: BrowserState) -> None:
        """Start Playwright and the persistent context (first use or after a crash)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise WebProviderError(
                "playwright is not installed. Run: pip install -r requirements-deepseek-web.txt"
            ) from e

        channel = self.env("CHANNEL", "chrome") or None
        headless = self.env_flag("HEADLESS", False)

        profile = self.profile_dir(state.account)
        Path(profile).mkdir(parents=True, exist_ok=True)
        state.pw = await async_playwright().start()
        try:
            context = await state.pw.chromium.launch_persistent_context(
                profile,
                channel=channel,
                headless=headless,
                chromium_sandbox=True,  # Playwright disables Chrome's sandbox by default (shows a warning bar)
                ignore_default_args=["--enable-automation"],
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception as e:
            await self._shutdown(state)
            raise WebProviderError(f"Failed to launch browser (channel={channel!r}): {e}") from e

        await context.expose_function("__lxEmit", _on_emit)
        override = self.env("STREAM_PATHS")
        stream_paths = [p.strip() for p in override.split(",") if p.strip()] if override else list(self.site.stream_paths)
        scope = {
            "__lxSite": self.site.name,
            "__lxAccount": state.account,
            "__lxStreamPaths": stream_paths,
        }
        await context.add_init_script(f"Object.assign(window, {json.dumps(scope)});")
        await context.add_init_script(path=str(DRIVER_JS_PATH))
        await self._apply_cookies(context, state)
        await self._apply_storage(context, state)
        state.context = context
        state.page = context.pages[0] if context.pages else await context.new_page()
        logger.info(
            "%s browser started [account=%s channel=%s headless=%s profile=%s]",
            self.site.label or self.site.name,
            state.account or "default",
            channel,
            headless,
            profile,
        )

    def _cookie_items(self, provider: Any) -> list[dict[str, Any]]:
        """Turn a ``"k=v; k2=v2"`` header into Playwright cookie objects."""
        raw = (getattr(provider, "cookies", "") or "").strip()
        if not raw:
            return []
        host = urlparse(self.site.home).hostname or ""
        items: list[dict[str, Any]] = []
        for part in raw.replace(chr(10), ";").split(";"):
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            name, value = name.strip(), value.strip()
            if name and host:
                items.append({"name": name, "value": value, "domain": host, "path": "/"})
        return items

    async def _apply_cookies(self, context: Any, state: "BrowserState") -> None:
        """Log a browser-driven site in by seeding the cookies from config."""
        items = self._cookie_items(state.provider)
        if not items:
            return
        try:
            await context.add_cookies(items)
            logger.info("%s: seeded %d cookie(s) [account=%s]", self.site.name, len(items), state.account or "default")
        except Exception as e:
            logger.warning("%s could not seed cookies: %s", self.site.name, e)

    def _storage_items(self, provider: Any) -> dict[str, str]:
        """Parse the provider's ``storage`` snapshot (a JSON object of strings)."""
        raw = (getattr(provider, "storage", "") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("%s: storage snapshot is not valid JSON; ignored", self.site.name)
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in parsed.items()}

    async def _apply_storage(self, context: Any, state: "BrowserState") -> None:
        """Seed localStorage before the site's own scripts run."""
        items = self._storage_items(state.provider)
        if not items:
            return
        script = (
            "(() => { const data = " + json.dumps(items) + ";"
            " for (const [k, v] of Object.entries(data)) {"
            " try { localStorage.setItem(k, v); } catch (e) {} } })();"
        )
        try:
            await context.add_init_script(script)
            logger.info("%s: seeded %d localStorage entr(ies)", self.site.name, len(items))
        except Exception as e:
            logger.warning("%s could not seed localStorage: %s", self.site.name, e)

    async def _shutdown(self, state: BrowserState) -> None:
        """Tear down one account's browser, ignoring errors from an already-dead browser."""
        ctx, pw = state.context, state.pw
        state.context = state.page = state.pw = None
        state.queue = None
        if ctx is not None:
            with contextlib.suppress(Exception):
                await ctx.close()
        if pw is not None:
            with contextlib.suppress(Exception):
                await pw.stop()

    async def close(self, account: Optional[str] = None, timeout: Optional[float] = None) -> None:
        """Close one account's browser, or all of them (called on proxy shutdown).

        ``timeout`` bounds the wait for a turn in progress (the lock is held for
        a whole turn). On expiry the browser is left running rather than blocking
        the caller — the profile stays usable, so nothing is lost.
        """
        try:
            async with asyncio.timeout(timeout):
                async with self.lock:
                    if account is None:
                        for state in list(self.states.values()):
                            await self._shutdown(state)
                        self.states.clear()
                    else:
                        state = self.states.get(account)
                        if state is not None:
                            await self._shutdown(state)
        except TimeoutError:
            logger.warning(
                "%s close(%s) timed out waiting for the turn in progress; browser left running",
                self.site.name,
                account or "all",
            )

    async def _ensure_page(self, state: BrowserState) -> Any:
        """Return a live page for one account, (re)launching its browser if needed."""
        if state.page is not None:
            try:
                if not state.page.is_closed():
                    await asyncio.wait_for(state.page.evaluate("1"), timeout=5)
                    return state.page
            except Exception:
                logger.warning("%s page is dead, relaunching browser [account=%s]", self.site.name, state.account)
            await self._shutdown(state)
        await self._launch(state)
        return state.page

    # -- page driving -----------------------------------------------------
    def home(self, route: ResolvedRoute) -> str:
        """Site URL for this request (``base_url`` in config wins)."""
        return (route.provider.base_url or self.site.home).rstrip("/") or self.site.home

    def account_for(self, provider: Any) -> str:
        """Return the active account id of this site's provider config."""
        if provider is None:
            return ""
        account = getattr(provider, "active_account", "") or ""
        keys = provider.resolved_api_keys() if hasattr(provider, "resolved_api_keys") else []
        if account and keys and account not in keys:
            return keys[0]  # configured account was removed; fall back
        return account

    def login_message(self, account: str = "") -> str:
        """Explain how to log into this site."""
        where = self.site.label or self.site.name
        return (
            f"{where} is not logged in"
            f"{f' for account {account!r}' if account else ''}. Log in to {self.site.home} in the browser window "
            "opened by the proxy (or run the site's `--login` command), then retry. "
            f"Profile: {self.profile_dir(account)}"
        )

    async def _stop(self, page: Any) -> bool:
        """Press the page's stop button; returns False if no stop button was found."""
        return bool(await page.evaluate(self.site.stop_js))

    async def _open_chat(self, page: Any, route: ResolvedRoute, ref: Optional[str], account: str = "") -> None:
        """Open a blank chat (``ref`` None) or an existing one, and wait for the composer.

        Detects the logged-out state. Reuses the current page without reloading
        when it is already on the target conversation.
        """
        url = ref or self.home(route)
        if page.url.rstrip("/") != url.rstrip("/"):
            await page.goto(url, wait_until="domcontentloaded", timeout=int(PAGE_READY_TIMEOUT * 1000))
        deadline = time.monotonic() + PAGE_READY_TIMEOUT
        state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                state = await page.evaluate(self.site.probe_js)
            except Exception:
                await asyncio.sleep(0.3)
                continue
            if self.site.login_path and str(state.get("path", "")).startswith(self.site.login_path):
                raise WebLoginRequired(self.login_message(account))
            if state.get("composer") and state.get("logged_in", True):
                # A previous generation may still be running (e.g. the stop button was missed).
                with contextlib.suppress(Exception):
                    if await self._stop(page):
                        await asyncio.sleep(0.5)
                if ref is None:
                    await self.site.prepare_new_chat(page, route)
                return
            await asyncio.sleep(0.3)
        if state.get("logged_in") is False:
            raise WebLoginRequired(self.login_message(account))
        raise WebProviderError(
            f"{self.site.label or self.site.name} composer not found (page layout may have changed). "
            f"Check the selector {self.site.composer!r}."
        )

    async def _composer(self, page: Any) -> Any:
        """Return the composer the user would type into, not just the first match.

        ``page.locator(sel).first`` can land on a hidden editor of the same kind
        (a chat page keeps previous/overlay editors around), and everything after
        that fails silently: ``fill()`` waits for visible+editable, the fallback
        text goes nowhere, and the turn then waits for a stream that never comes.
        """
        locator = page.locator(self.site.composer)
        deadline = time.monotonic() + self.env_float("COMPOSER_WAIT", COMPOSER_WAIT)
        while True:
            count = await locator.count()
            for index in range(min(count, 8)):
                candidate = locator.nth(index)
                with contextlib.suppress(Exception):
                    if await candidate.is_visible():
                        return candidate
            if time.monotonic() > deadline:
                return locator.first
            await asyncio.sleep(0.3)

    async def _submit(self, page: Any, prompt: str) -> bool:
        """Fill the composer and send; return False when the page did not take it.

        Two attempts: a rich-text editor can lose the text if it was still
        hydrating when we typed, in which case a second pass succeeds.
        """
        composer: Any = None
        for attempt in (1, 2):
            composer = await self._composer(page)
            # Rich-text editors (TipTap/ProseMirror contenteditable) only take
            # input once focused, and some of them reject fill() outright.
            with contextlib.suppress(Exception):
                await composer.click(timeout=5000)
            try:
                await composer.fill(prompt, timeout=8000)
            except Exception as e:
                logger.info("%s fill() did not work (%s); clicking and inserting text", self.site.name, e)
                with contextlib.suppress(Exception):
                    await composer.click(timeout=3000)
                await page.keyboard.insert_text(prompt)
            await asyncio.sleep(0.5)  # let the editor commit the text
            if not (await self._composer_text(composer)).strip():
                logger.warning("%s composer stayed empty after typing (attempt %d)", self.site.name, attempt)
                continue
            debug_logger.debug("%s composer before Enter: %r", self.site.name, prompt[:60])
            await composer.press("Enter")
            await asyncio.sleep(0.8)
            # The editor usually clears (or is replaced) as soon as the turn starts,
            # so this check must never block the turn.
            if not (await self._composer_text(composer)).strip():
                return True
            if self.site.send_button_js:
                logger.warning("%s composer not cleared after Enter, clicking send button", self.site.name)
                with contextlib.suppress(Exception):
                    clicked = await page.evaluate(self.site.send_button_js)
                    logger.info("%s send button click -> %s", self.site.name, clicked)
                await asyncio.sleep(0.8)
                if not (await self._composer_text(composer)).strip():
                    return True
            logger.warning("%s composer still holds the prompt after sending", self.site.name)
        return False

    @staticmethod
    async def _composer_text(composer: Any) -> str:
        """Best-effort read of what the composer still holds (never raises).

        Covers both plain ``<textarea>``/``<input>`` composers and rich-text
        editors (TipTap/ProseMirror ``contenteditable`` divs), where the text
        lives in ``innerText`` and ``input_value`` throws.
        """
        try:
            return await composer.evaluate(
                "el => (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') "
                "? el.value : (el.innerText || el.textContent || '')",
                timeout=1500,
            )
        except Exception:
            return ""

    async def _current_chat_ref(self, page: Any) -> Optional[str]:
        """Return the page URL once the site has assigned a chat session URL."""
        marker = self.site.chat_url_marker
        for _ in range(15):
            url = page.url
            if marker and marker in url:
                return url
            if not marker:
                return url
            await asyncio.sleep(0.2)
        return None

    async def _pace(self) -> None:
        """Keep a minimum gap between turns so we do not hammer the site."""
        interval = self.env_float("MIN_INTERVAL", 3.0)
        wait = interval - (time.monotonic() - self.last_turn_end)
        if self.last_turn_end and wait > 0:
            await asyncio.sleep(wait)


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
        """Run one turn in the browser and yield answer text deltas.

        Continues an existing web chat when the request extends one we already
        have (only the newest user message is sent); otherwise opens a new chat
        and replays the whole history. Holds the site lock for the whole turn.
        If the consumer stops early (``</tool_call>`` seen, client disconnect)
        the page's stop button is pressed.
        """
        account = self.account_for(route.provider)
        self.refresh_guard()
        self.guard.check(account)
        state = self.state_for(account)
        state.provider = route.provider
        convs = self.convs_for(account)
        async with self._locked(account):
            await self._pace()
            page = await self._ensure_page(state)
            conv = match_conversation(convs, system, turns)
            prompt = build_incremental_prompt(system, turns) if conv else build_full_prompt(system, turns)
            try:
                await self._open_chat(page, route, conv.url if conv else None, account)
            except WebLoginRequired:
                raise
            except WebProviderError as e:
                if conv is None:
                    raise
                logger.warning("Could not continue %s chat (%s); replaying history in a new chat", self.site.name, e)
                convs.remove(conv)
                conv, prompt = None, build_full_prompt(system, turns)
                await self._open_chat(page, route, None, account)
            logger.info(
                "%s turn [account=%s %s] prompt_chars=%d",
                self.site.name,
                account or "default",
                "continue" if conv else "new chat",
                len(prompt),
            )

            queue: asyncio.Queue = asyncio.Queue()
            state.queue = queue
            parsers: dict[Any, Any] = {}

            def parser_for(stream_id: Any) -> Any:
                """One parser per captured stream.

                Streams must never share a parser: the answer stream is SSE, the
                page fires plenty of plain-JSON calls on the same paths, and a
                JSON body left in the buffer would corrupt the first SSE frame
                after it (silently dropping the start of the answer).
                """
                parser = parsers.get(stream_id)
                if parser is None:
                    parser = parsers[stream_id] = self.site.new_parser(route)
                return parser

            streams: dict[Any, dict[str, Any]] = {}
            finished = False
            failed = False
            t_submit = time.monotonic()
            t_first: Optional[float] = None
            got_text = False
            challenge_deadline: Optional[float] = None
            try:
                sent = await self._submit(page, prompt)
                t_submit = time.monotonic()
                while not finished:
                    limit = IDLE_TIMEOUT if got_text or t_first else FIRST_BYTE_TIMEOUT
                    if not sent and not got_text and t_first is None:
                        # The page may never have taken the prompt; fail fast
                        # instead of sitting out the full first-byte timeout.
                        limit = min(limit, SUBMIT_GRACE)
                    if challenge_deadline is not None:
                        limit = max(0.1, challenge_deadline - time.monotonic())
                    remaining_total = TOTAL_TIMEOUT - (time.monotonic() - t_submit)
                    if remaining_total <= 0:
                        raise WebProviderError(f"{self.site.name} turn exceeded {TOTAL_TIMEOUT:.0f}s")
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=min(limit, remaining_total))
                    except asyncio.TimeoutError:
                        if challenge_deadline is not None:
                            self.guard.record(account, "captcha challenge left unsolved")
                            raise WebProviderError(self.site.challenge_message()) from None
                        if not sent:
                            raise WebProviderError(
                                f"{self.site.label or self.site.name} did not accept the prompt: the composer "
                                f"still held it after Enter and the send button (selector {self.site.composer!r}). "
                                "Check the browser window — it may be showing a dialog the page needs."
                            ) from None
                        raise WebProviderError(
                            f"Timed out waiting for the {self.site.name} stream "
                            "(no request seen; the page layout or login state may have changed)"
                        ) from None

                    kind = msg.get("kind")
                    stream_id = msg.get("id")
                    info = streams.setdefault(stream_id, {"url": "", "type": "", "text": False})
                    if t_first is None and kind in ("start", "chunk"):
                        t_first = time.monotonic()
                    if kind == "error":
                        message = str(msg.get("message") or "")
                        if "429" in message or "403" in message:
                            self.guard.record(account, f"stream error {message}")
                        raise WebProviderError(f"{self.site.name} stream failed: {message}")
                    if kind == "start":
                        info["url"] = str(msg.get("url") or "")
                        info["type"] = str(msg.get("contentType") or "")
                        logger.info("%s stream #%s %s %s", self.site.name, stream_id, info["type"] or "?", info["url"])
                        debug_logger.info(
                            "%s STREAM #%s %s %s", self.site.name.upper(), stream_id, info["type"], info["url"]
                        )
                    if kind == "chunk":
                        parser = parser_for(stream_id)
                        raw_chunk = msg.get("text", "")
                        debug_logger.debug("%s chunk #%s %r", self.site.name, stream_id, raw_chunk[:200])
                        for ev_kind, delta in parser.feed(raw_chunk):
                            if ev_kind == "text":
                                info["text"] = True
                                if not got_text:
                                    got_text = True
                                    challenge_deadline = None
                                    logger.info(
                                        "%s first-token latency: %.2fs", self.site.name, time.monotonic() - t_submit
                                    )
                                yield delta
                        if parser.done:
                            finished = True
                    elif kind == "end":
                        if not info["text"] and not any(k in info["type"].lower() for k in ("event-stream", "websocket")):
                            # An unrelated JSON call answered and closed: keep listening for
                            # the real answer instead of ending the turn on this one.
                            logger.info(
                                "%s ignoring non-stream response #%s (%s)",
                                self.site.name,
                                stream_id,
                                info["type"] or "?",
                            )
                            continue
                        if not info["text"]:
                            for ev_kind, delta in parser_for(stream_id).flush():
                                if ev_kind == "text":
                                    got_text = True
                                    yield delta
                        if (
                            not got_text
                            and self.site.challenge_marker
                            and self.site.challenge_marker in parser_for(stream_id).raw_head
                        ):
                            # Do NOT reload or resubmit: that would destroy the captcha dialog the
                            # user is solving. Wait in place; the page re-sends its request once solved.
                            logger.warning(
                                "%s requires human verification; waiting up to %.0fs for it to be solved "
                                "in the browser window",
                                self.site.name,
                                CHALLENGE_WAIT_TIMEOUT,
                            )
                            challenge_deadline = time.monotonic() + CHALLENGE_WAIT_TIMEOUT
                            parsers[stream_id] = self.site.new_parser(route)
                        else:
                            finished = True

                if not got_text:
                    head = "".join(p.raw_head for p in parsers.values())[:300]
                    raise WebProviderError(f"{self.site.name} returned no answer text. Stream head: {head!r}")
                logger.info("%s turn done in %.2fs", self.site.name, time.monotonic() - t_submit)
            except GeneratorExit:
                raise  # consumer stopped early on purpose; handled in finally
            except BaseException:
                failed = True
                raise
            finally:
                state.queue = None
                if not finished:
                    # Early exit (tool call complete / client gone / error): stop server-side generation.
                    try:
                        stopped = await self._stop(page)
                        logger.info("%s stop button pressed: %s", self.site.name, stopped)
                    except Exception as e:
                        logger.warning("%s stop failed: %s", self.site.name, e)
                try:
                    if got_text and not failed:
                        record_conversation(
                            convs, conv, system, turns, await self._current_chat_ref(page)
                        )
                        self.guard.success(account)
                    elif conv is not None and conv in convs:
                        convs.remove(conv)  # state of that chat is uncertain after a failed turn
                except Exception as e:
                    logger.warning("%s could not record conversation: %s", self.site.name, e)
                self.last_turn_end = time.monotonic()

    # -- admin helpers ----------------------------------------------------
    async def status(self, account: str, route: Optional[ResolvedRoute] = None) -> dict[str, Any]:
        """Report one account's browser/login state without disturbing a running turn.

        ``logged_in`` is None when it cannot be known right now (browser not started,
        a turn is running, or the page is not on the site).
        """
        state = self.states.get(account) or BrowserState(self.site, account)
        home = self.site.home
        info: dict[str, Any] = {
            "account": account,
            "transport": "browser",
            "profile_dir": self.profile_dir(account),
            "headless": self.env_flag("HEADLESS", False),
            "browser_running": state.page is not None,
            "busy": state.queue is not None,
            "logged_in": None,
            "url": None,
            "cooldown_seconds_left": round(self.guard.remaining(account)),
        }
        if info["busy"] or state.page is None:
            return info
        async with self.lock:
            page = state.page
            if page is None or page.is_closed():
                info["browser_running"] = False
                return info
            info["url"] = page.url
            if page.url.startswith(home):
                try:
                    probe = await asyncio.wait_for(page.evaluate(self.site.probe_js), timeout=5)
                    info["logged_in"] = bool(
                        probe.get("logged_in") and not (
                            self.site.login_path and str(probe.get("path", "")).startswith(self.site.login_path)
                        )
                    )
                except Exception:
                    pass
        return info

    async def open_login_window(self, account: str, route: Optional[ResolvedRoute] = None) -> dict[str, Any]:
        """Start one account's browser, show the site and bring the window to the front.

        If a turn is running on that account, the page is left untouched (it may
        be showing a captcha the user has to solve); only the window is raised.
        """
        if self.env_flag("HEADLESS", False):
            raise WebProviderError(
                f"{self.site.env_prefix}_HEADLESS=1: the browser has no window. Unset it and restart the proxy."
            )
        state = self.state_for(account)
        if self.lock.locked():
            if state.page is not None:
                with contextlib.suppress(Exception):
                    await state.page.bring_to_front()
            return await self.status(account)
        async with self.lock:
            page = await self._ensure_page(state)
            if not page.url.startswith(self.site.home):
                await page.goto(
                    self.site.home, wait_until="domcontentloaded", timeout=int(PAGE_READY_TIMEOUT * 1000)
                )
            await page.bring_to_front()
        return await self.status(account)


def runtime_for(site: WebSite) -> BrowserRuntime:
    """Return the (single) browser runtime of a site, registering it on first use."""
    runtime = _RUNTIMES.get(site.name)
    if runtime is None or runtime.site is not site:
        runtime = BrowserRuntime(site).register()
    return runtime
