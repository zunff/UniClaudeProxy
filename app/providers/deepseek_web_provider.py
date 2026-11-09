"""DeepSeek web (chat.deepseek.com) provider driven through Playwright.

The provider does not call any private API directly. It drives the real web
page (fill the composer, press Enter) and listens to the page's *own*
streaming request via an init script (``deepseek_web_driver.js``) that wraps
``fetch`` / ``XMLHttpRequest`` and forwards raw SSE text to Python.

The web page output is converted to a minimal Anthropic SSE stream. Tool
calling is handled upstream by ``app.react`` (XML ``<tool_call>`` blocks), so
this module only has to produce a clean text stream.

Multiple accounts
    Each non-empty entry of the provider's ``api_keys`` is an *account id*.
    An account gets its own persistent browser profile, its own browser
    instance and its own tracked web chats, so several DeepSeek logins can be
    kept side by side. Which one serves requests is chosen by the provider's
    ``active_account`` (empty = the legacy single profile); a request never
    round-robins across accounts, because that would switch identity mid-chat.
    The admin dashboard lists one card per account, manages add/remove and the
    active one.

Environment variables (all optional):
    DEEPSEEK_WEB_PROFILE_DIR  Base directory for browser profiles
                              (default: logs/deepseek-web-profiles). Named accounts use
                              "<base>/<account id>"; the default account uses "<base>"
                              itself, which defaults to the pre-multi-account path
                              logs/deepseek-web-profile so an existing login survives.
    DEEPSEEK_WEB_CHANNEL      Playwright browser channel, e.g. "chrome" (default) or "" for bundled Chromium.
    DEEPSEEK_WEB_HEADLESS     "1" to run headless (default "0"; headed is needed for the first login).
    DEEPSEEK_WEB_THINKING     "1" to switch DeepThink on for new chats (default off; smart search is always turned off).
    DEEPSEEK_WEB_MIN_INTERVAL Minimum seconds between turns (default 3), per account.

Debugging without the proxy:
    python -m app.providers.deepseek_web_provider --login [--account NAME]
    python -m app.providers.deepseek_web_provider --prompt "say hi" [--account NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from app.config import ResolvedRoute, load_config
from app.models import AnthropicRequest
from app.react.prompt import format_observation_xml, format_tool_call_xml

logger = logging.getLogger("anyclaude")
debug_logger = logging.getLogger("anyclaude.debug")

DRIVER_JS_PATH = Path(__file__).with_name("deepseek_web_driver.js")
DEFAULT_HOME = "https://chat.deepseek.com"
TOOL_CALL_OPEN = "<tool_call"
TOOL_CALL_CLOSE = "</tool_call>"

COMPOSER_SELECTOR = 'textarea[name="search"], textarea'
PAGE_READY_TIMEOUT = 25.0
FIRST_BYTE_TIMEOUT = 90.0
IDLE_TIMEOUT = 120.0
TOTAL_TIMEOUT = 900.0
CHALLENGE_WAIT_TIMEOUT = 240.0  # how long to wait in place for the user to solve a captcha


class DeepSeekWebError(RuntimeError):
    """Raised when the web page cannot be driven or returns no usable answer."""


class DeepSeekWebLoginRequired(DeepSeekWebError):
    """Raised when the browser profile is not logged in to chat.deepseek.com."""


# ---------------------------------------------------------------------------
# Pure helpers (no browser): prompt flattening, stream parsing, SSE building.
# ---------------------------------------------------------------------------


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Normalise a content block (dict or pydantic model) to a dict."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return dict(block)


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic message content into plain text.

    Images are replaced by a placeholder because phase one does not upload
    attachments. Thinking blocks are dropped.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    parts: list[str] = []
    for raw in content:
        block = _block_to_dict(raw)
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            parts.append(format_tool_call_xml(block.get("name", ""), block.get("input", {})))
        elif btype == "tool_result":
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner = " ".join(
                    _block_to_dict(b).get("text", "") for b in inner if _block_to_dict(b).get("type") == "text"
                )
            parts.append(format_observation_xml(str(inner)))
        elif btype == "image":
            parts.append("[image omitted]")
    return "\n\n".join(p for p in parts if p)


def _system_to_text(system: Any) -> str:
    """Flatten an Anthropic ``system`` field (str or text blocks) into text."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            _block_to_dict(b).get("text", "") for b in system if _block_to_dict(b).get("type") == "text"
        )
    return ""


Turns = list[tuple[str, str]]


def split_request(request: AnthropicRequest) -> tuple[str, Turns]:
    """Return ``(system_text, [(role, text), ...])`` for a ReAct-transformed request."""
    system = _system_to_text(request.system).strip()
    turns = [(m.role, _content_to_text(m.content).strip()) for m in request.messages]
    return system, turns


_TOOL_HINT = " If you need a tool, end your reply with exactly one <tool_call> block and stop."


def build_full_prompt(system: str, turns: Turns) -> str:
    """Pack system prompt and the whole history into one web chat message.

    A lone user message without a system prompt is sent verbatim.
    """
    if not system and len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1]

    sections: list[str] = []
    if system:
        sections.append(f"<system>\n{system}\n</system>")
    convo = "\n\n".join(f"[{role}]\n{text}" for role, text in turns)
    sections.append(f"<conversation>\n{convo}\n</conversation>")

    tail = "Write only the next [assistant] message. Do not repeat the conversation or add role labels."
    if "<tool_call>" in system:
        tail += _TOOL_HINT
    sections.append(tail)
    return "\n\n".join(sections)


def build_incremental_prompt(system: str, turns: Turns) -> str:
    """Message for a continued web conversation: only the newest user turn."""
    text = turns[-1][1]
    if "<tool_call>" in system:
        text += "\n\n(Continue as the assistant." + _TOOL_HINT + ")"
    return text


def flatten_prompt(request: AnthropicRequest) -> str:
    """Collapse a (ReAct-transformed) request into a single full-replay message."""
    system, turns = split_request(request)
    return build_full_prompt(system, turns)


# ---------------------------------------------------------------------------
# Conversation reuse: continue the same web chat instead of replaying history.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Conversation:
    """A web chat we can continue, plus what we already sent into it."""

    url: str
    system: str
    user_turns: list[str]
    n_messages: int  # len(request.messages) of the last request answered in this chat
    last_used: float = 0.0


MAX_CONVERSATIONS = 8


def match_conversation(convs: list[Conversation], system: str, turns: Turns) -> Optional[Conversation]:
    """Find a chat whose history is a strict prefix of this request.

    The next request of the same agent loop is: previous messages, our assistant
    reply, one new user message. Assistant texts are not compared (the client
    re-serialises our tool calls), only the system prompt and the user turns.
    """
    if not turns or turns[-1][0] != "user":
        return None
    for conv in convs:
        n = conv.n_messages
        if conv.system != system or len(turns) != n + 2 or turns[n][0] != "assistant":
            continue
        if [t for r, t in turns[:n] if r == "user"] == conv.user_turns:
            return conv
    return None


class DeepSeekStreamParser:
    """Incremental parser for DeepSeek's SSE + abbreviated JSON-patch stream.

    Feed raw SSE text with :meth:`feed`; it returns ``(kind, delta)`` tuples
    where ``kind`` is ``"text"`` (answer) or ``"thinking"``. ``done`` becomes
    true once the stream signals completion.

    Frame facts (see docs of the reference project): each ``data:`` frame is a
    patch ``{"p": path, "o": op, "v": value}``; ``p`` / ``o`` are omitted when
    unchanged from the previous frame; ``o == "BATCH"`` carries a list of
    sub-patches whose paths are relative to ``p``.
    """

    def __init__(self) -> None:
        self.done = False
        self.raw_head = ""
        self._buf = ""
        self._event = ""
        self._fragments: list[dict[str, str]] = []
        self._path = ""
        self._op = "SET"
        self._legacy = {"content": "", "thinking_content": ""}

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Consume raw SSE text and return newly produced deltas."""
        if len(self.raw_head) < 500:
            self.raw_head += text[: 500 - len(self.raw_head)]
        self._buf += text
        out: list[tuple[str, str]] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line.rstrip("\r"), out)
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Process a trailing line that lacked a newline."""
        out: list[tuple[str, str]] = []
        if self._buf:
            line, self._buf = self._buf, ""
            self._handle_line(line.rstrip("\r"), out)
        return out

    def _handle_line(self, line: str, out: list[tuple[str, str]]) -> None:
        if not line:
            self._event = ""
            return
        if line.startswith(":"):
            return
        if line.startswith("event:"):
            self._event = line[6:].strip()
            if self._event in ("finish", "close"):
                self.done = True
            return
        if not line.startswith("data:"):
            return
        payload = line[5:]
        if payload.startswith(" "):
            payload = payload[1:]
        if payload.strip() == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        self._apply_frame(obj, out)

    def _apply_frame(self, obj: Any, out: list[tuple[str, str]]) -> None:
        if not isinstance(obj, dict):
            return
        if "p" in obj:
            self._path = obj["p"] or ""
            self._op = obj.get("o", "SET")
        elif "o" in obj:
            self._op = obj["o"]
        if "v" not in obj:
            return
        self._apply_patch(self._path, self._op, obj["v"], out)

    def _apply_patch(self, path: str, op: str, value: Any, out: list[tuple[str, str]]) -> None:
        if op == "BATCH" and isinstance(value, list):
            for sub in value:
                if not isinstance(sub, dict):
                    continue
                sub_path = sub.get("p", "") or ""
                full = f"{path}/{sub_path}" if path and sub_path else (path or sub_path)
                self._apply_patch(full, sub.get("o", "SET"), sub.get("v"), out)
            return

        if path == "" and isinstance(value, dict) and "response" in value:
            self._init_response(value["response"], out)
        elif path == "response" and isinstance(value, dict):
            self._init_response(value, out)
        elif path == "response/status":
            if value == "FINISHED":
                self.done = True
        elif path == "response/fragments":
            if op == "APPEND" and isinstance(value, list):
                for frag in value:
                    self._add_fragment(frag, out)
        elif path in ("response/content", "response/thinking_content"):
            self._apply_legacy(path.split("/", 1)[1], op, value, out)
        else:
            m = re.fullmatch(r"response/fragments/(-?\d+)/content", path)
            if m and isinstance(value, str):
                self._apply_fragment_content(int(m.group(1)), op, value, out)

    def _init_response(self, resp: Any, out: list[tuple[str, str]]) -> None:
        if not isinstance(resp, dict):
            return
        frags = resp.get("fragments")
        if isinstance(frags, list):
            self._fragments = []
            for frag in frags:
                self._add_fragment(frag, out)
        for key in ("thinking_content", "content"):
            if isinstance(resp.get(key), str):
                self._apply_legacy(key, "SET", resp[key], out)
        if resp.get("status") == "FINISHED":
            self.done = True

    def _add_fragment(self, frag: Any, out: list[tuple[str, str]]) -> None:
        if not isinstance(frag, dict):
            return
        entry = {"type": str(frag.get("type") or "RESPONSE"), "content": str(frag.get("content") or "")}
        self._fragments.append(entry)
        self._emit(entry["type"], entry["content"], out)

    def _apply_fragment_content(self, idx: int, op: str, value: str, out: list[tuple[str, str]]) -> None:
        if idx < 0:
            idx += len(self._fragments)
        if not 0 <= idx < len(self._fragments):
            return
        frag = self._fragments[idx]
        prev = frag["content"]
        if op == "APPEND":
            delta = value
        else:
            delta = value[len(prev):] if value.startswith(prev) else ""
        frag["content"] = prev + delta
        self._emit(frag["type"], delta, out)

    def _apply_legacy(self, key: str, op: str, value: Any, out: list[tuple[str, str]]) -> None:
        if not isinstance(value, str):
            return
        prev = self._legacy[key]
        delta = value if op == "APPEND" else (value[len(prev):] if value.startswith(prev) else "")
        self._legacy[key] = prev + delta
        self._emit("THINK" if key == "thinking_content" else "RESPONSE", delta, out)

    @staticmethod
    def _emit(frag_type: str, text: str, out: list[tuple[str, str]]) -> None:
        if not text:
            return
        if frag_type == "THINK":
            out.append(("thinking", text))
        elif frag_type == "RESPONSE":
            out.append(("text", text))


_BAR = "[｜|]"  # full-width or ASCII vertical bar
_DSML_START_RE = re.compile(rf"<{_BAR}+\s*DSML")
_DSML_END_RE = re.compile(rf"</{_BAR}+\s*DSML{_BAR}+\s*(?:function_)?calls\s*>")
# Lenient on purpose: the model sometimes closes ``invoke`` with a bare
# ``</DSML>`` or not at all, and emits stray closing tags. The invoke body runs
# to the next invoke/calls closer (or end of text); a parameter value runs to
# the next DSML tag of any kind.
_DSML_INVOKE_RE = re.compile(
    rf'<{_BAR}+\s*DSML{_BAR}+\s*invoke\s+name="([^"]+)"\s*>(.*?)'
    rf"(?:</{_BAR}+\s*DSML{_BAR}+\s*(?:invoke|(?:function_)?calls)\s*>|\Z)",
    re.S,
)
_DSML_PARAM_RE = re.compile(
    rf'<{_BAR}+\s*DSML{_BAR}+\s*parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)'
    rf"(?=</?{_BAR}+\s*DSML|\Z)",
    re.S,
)
_DSML_HOLD_RE = re.compile(rf"<{_BAR}*(?:D(?:S(?:M(?:L)?)?)?)?")


def dsml_to_tool_call(block: str) -> Optional[str]:
    """Convert DeepSeek's native tool-call markup into the ReAct ``<tool_call>`` XML.

    Observed shape (seen when the web model ignores the ReAct format)::

        <DSML calls><DSML invoke name="Write">
        <DSML parameter name="file_path" string="true">/x.py</DSML parameter>
        </DSML invoke></DSML calls>

    where ``DSML`` is wrapped in vertical bars. ``string="true"`` marks a raw
    string value, otherwise the value is JSON. Only the first invoke is used
    (ReAct allows one tool call per reply). Returns None if no invoke is found.
    """
    inv = _DSML_INVOKE_RE.search(block)
    if not inv:
        return None
    args: dict[str, Any] = {}
    for name, is_string, raw in _DSML_PARAM_RE.findall(inv.group(2)):
        if is_string == "true":
            args[name] = raw
            continue
        try:
            args[name] = json.loads(raw.strip())
        except json.JSONDecodeError:
            args[name] = raw
    return format_tool_call_xml(inv.group(1), args)


class DsmlNormalizer:
    """Rewrite a native DeepSeek tool-call block into ``<tool_call>`` XML on the fly.

    Plain text passes through. From the first DSML marker the text is buffered
    until the closing ``calls`` tag, then replaced by one ``<tool_call>`` block.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._active = False

    def feed(self, delta: str) -> tuple[str, bool]:
        """Return ``(text_to_emit, finished)``; finished means a call was completed."""
        self._buf += delta
        pre = ""
        if not self._active:
            m = _DSML_START_RE.search(self._buf)
            if m:
                pre, self._buf, self._active = self._buf[: m.start()], self._buf[m.start():], True
            else:
                idx = self._buf.rfind("<")
                if idx >= 0 and _DSML_HOLD_RE.fullmatch(self._buf[idx:]):
                    out, self._buf = self._buf[:idx], self._buf[idx:]
                else:
                    out, self._buf = self._buf, ""
                return out, False
        end = _DSML_END_RE.search(self._buf)
        if not end:
            return pre, False
        block, self._buf = self._buf[: end.end()], ""
        return pre + (dsml_to_tool_call(block) or block), True

    def flush(self) -> str:
        """Release anything still buffered at end of stream."""
        buf, self._buf = self._buf, ""
        if self._active:
            return dsml_to_tool_call(buf) or buf
        return buf


class ToolCallTruncator:
    """Cut the answer right after the first complete ``</tool_call>``.

    The web page ignores the ``</tool_call>`` stop sequence injected by the
    ReAct layer, so the model keeps writing. This finds the closing tag as soon
    as it appears so the caller can stop generation immediately.
    """

    def __init__(self) -> None:
        self._acc = ""

    def feed(self, delta: str) -> tuple[str, bool]:
        """Return ``(text_to_emit, finished)`` for one incoming delta."""
        prev_len = len(self._acc)
        self._acc += delta
        search_from = max(0, prev_len - len(TOOL_CALL_CLOSE) + 1)
        idx = self._acc.find(TOOL_CALL_CLOSE, search_from)
        if idx >= 0 and TOOL_CALL_OPEN in self._acc[:idx]:
            end = idx + len(TOOL_CALL_CLOSE)
            emit = self._acc[prev_len:end]
            self._acc = self._acc[:end]
            return emit, True
        return delta, False


def _sse(event_type: str, data: dict[str, Any]) -> str:
    """Format one Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (the web page exposes no usage numbers)."""
    return max(1, len(text) // 4) if text else 0


def anthropic_events_from_text(
    deltas: list[str],
    model: str,
    input_tokens: int = 0,
    message_id: Optional[str] = None,
) -> list[str]:
    """Build a complete minimal Anthropic SSE sequence from text deltas (test/helper)."""
    msg_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    events = [_message_start(msg_id, model, input_tokens), _text_block_start()]
    events.extend(_text_delta(d) for d in deltas if d)
    events.extend(_stream_tail("".join(deltas)))
    return events


def _message_start(msg_id: str, model: str, input_tokens: int) -> str:
    return _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })


def _text_block_start() -> str:
    return _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })


def _text_delta(text: str) -> str:
    return _sse("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    })


def _stream_tail(full_text: str) -> list[str]:
    return [
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": _estimate_tokens(full_text)},
        }),
        _sse("message_stop", {"type": "message_stop"}),
    ]


# ---------------------------------------------------------------------------
# Browser layer (Playwright). One shared persistent context, strictly serial.
# ---------------------------------------------------------------------------

_LOCK = asyncio.Lock()


class _BrowserState:
    """Playwright handles for one account and the queue of its active turn."""

    def __init__(self, account: str) -> None:
        self.account = account
        self.pw: Any = None
        self.context: Any = None
        self.page: Any = None
        self.queue: Optional[asyncio.Queue] = None


_states: dict[str, _BrowserState] = {}


def _state_for(account: str) -> _BrowserState:
    """Return (creating on demand) the browser state of one account."""
    state = _states.get(account)
    if state is None:
        state = _states[account] = _BrowserState(account)
    return state


def _on_emit(payload: str) -> None:
    """Binding called from the page; routes driver messages to the active turn.

    The init script is added per browser context; the binding payload carries
    the account id so multiple browsers can share one callback.
    """
    try:
        msg = json.loads(payload)
    except json.JSONDecodeError:
        return
    account = msg.pop("__account", "")
    state = _states.get(account)
    queue = state.queue if state is not None else None
    if queue is None:
        return
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        pass


def _account_id(provider: Any = None) -> str:
    """Return the active account id for a deepseek-web provider config.

    Empty string means the legacy single-account profile.
    """
    if provider is None:
        return ""
    account = getattr(provider, "active_account", "") or ""
    keys = provider.resolved_api_keys() if hasattr(provider, "resolved_api_keys") else []
    if account and keys and account not in keys:
        # Configured account was removed; fall back to the first key.
        return keys[0]
    return account


def active_account() -> str:
    """Return the configured active account id (for the admin endpoints)."""
    return _account_id(load_config().providers.get("deepseek-web"))


#: Profile dir used before multi-account support; the default account keeps using
#: it so an existing login is not lost.
LEGACY_PROFILE_DIR = Path("logs") / "deepseek-web-profile"


def _profile_root() -> Path:
    """Base directory holding one browser profile per named account."""
    return Path(os.environ.get("DEEPSEEK_WEB_PROFILE_DIR") or Path("logs") / "deepseek-web-profiles")


def _profile_dir(account: str = "") -> str:
    """Profile directory for one account.

    The default account (empty id) keeps the legacy single-profile path; named
    accounts get ``<root>/<account>`` so each one holds its own login.
    """
    root = Path(os.environ.get("DEEPSEEK_WEB_PROFILE_DIR") or LEGACY_PROFILE_DIR)
    if not account:
        return str(root)
    return str(_profile_root() / account)


async def _launch(state: _BrowserState) -> None:
    """Start Playwright and the persistent context (first use or after a crash)."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise DeepSeekWebError(
            "playwright is not installed. Run: pip install -r requirements-deepseek-web.txt"
        ) from e

    channel = os.environ.get("DEEPSEEK_WEB_CHANNEL", "chrome") or None
    headless = os.environ.get("DEEPSEEK_WEB_HEADLESS", "0") == "1"

    profile = _profile_dir(state.account)
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
        await _shutdown(state)
        raise DeepSeekWebError(f"Failed to launch browser (channel={channel!r}): {e}") from e

    await context.expose_function("__lxEmit", _on_emit)
    await context.add_init_script(f"window.__lxAccount = {json.dumps(state.account)};")
    await context.add_init_script(path=str(DRIVER_JS_PATH))
    state.context = context
    state.page = context.pages[0] if context.pages else await context.new_page()
    logger.info(
        "DeepSeek web browser started [account=%s channel=%s headless=%s profile=%s]",
        state.account or "default",
        channel,
        headless,
        profile,
    )


async def _shutdown(state: _BrowserState) -> None:
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


async def _close_locked(account: Optional[str]) -> None:
    async with _LOCK:
        if account is None:
            for state in list(_states.values()):
                await _shutdown(state)
            _states.clear()
        else:
            state = _states.get(account)
            if state is not None:
                await _shutdown(state)


async def close(account: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """Close one account's browser, or all of them (called on proxy shutdown).

    ``timeout`` bounds the wait for a turn in progress (the lock is held for a
    whole turn). On expiry the browser is left running rather than blocking the
    caller — the profile stays usable, so nothing is lost.
    """
    try:
        async with asyncio.timeout(timeout):
            await _close_locked(account)
    except TimeoutError:
        logger.warning(
            "DeepSeek web close(%s) timed out waiting for the turn in progress; browser left running",
            account or "all",
        )


async def _ensure_page(state: _BrowserState) -> Any:
    """Return a live page for one account, (re)launching its browser if needed."""
    if state.page is not None:
        try:
            if not state.page.is_closed():
                await asyncio.wait_for(state.page.evaluate("1"), timeout=5)
                return state.page
        except Exception:
            logger.warning("DeepSeek web page is dead, relaunching browser [account=%s]", state.account)
        await _shutdown(state)
    await _launch(state)
    return state.page


_TOGGLES_JS = """([wantThink, wantSearch]) => {
    const names = {
        think: ['\u6df1\u5ea6\u601d\u8003', 'DeepThink'],
        search: ['\u667a\u80fd\u641c\u7d22', 'Search'],
    };
    const result = {};
    for (const [key, want] of [['think', wantThink], ['search', wantSearch]]) {
        const el = [...document.querySelectorAll('.ds-toggle-button')]
            .find((b) => names[key].includes((b.innerText || '').trim()));
        if (!el) { result[key] = 'missing'; continue; }
        const on = el.className.includes('ds-toggle-button--selected');
        if (on !== want) el.click();
        result[key] = want ? 'on' : 'off';
    }
    return result;
}"""


async def _apply_toggles(page: Any) -> None:
    """Set DeepThink per DEEPSEEK_WEB_THINKING (default off) and turn web search off."""
    want_think = os.environ.get("DEEPSEEK_WEB_THINKING", "0") == "1"
    try:
        result = await page.evaluate(_TOGGLES_JS, [want_think, False])
        await asyncio.sleep(0.2)
        logger.info("DeepSeek web toggles: %s", result)
    except Exception as e:
        logger.warning("DeepSeek web could not set toggles: %s", e)


_convs_by_account: dict[str, list[Conversation]] = {}
_last_turn_end = 0.0


def _convs_for(account: str) -> list[Conversation]:
    return _convs_by_account.setdefault(account, [])


def _min_interval() -> float:
    try:
        return float(os.environ.get("DEEPSEEK_WEB_MIN_INTERVAL", "3"))
    except ValueError:
        return 3.0


async def _pace() -> None:
    """Keep a minimum gap between turns so we do not hammer the site."""
    wait = _min_interval() - (time.monotonic() - _last_turn_end)
    if _last_turn_end and wait > 0:
        await asyncio.sleep(wait)


async def _current_chat_url(page: Any) -> Optional[str]:
    """Return the page URL once the site has assigned a /a/chat/s/<id> session."""
    for _ in range(15):
        url = page.url
        if "/a/chat/s/" in url:
            return url
        await asyncio.sleep(0.2)
    return None


def _record_conversation(
    account: str, conv: Optional[Conversation], system: str, turns: Turns, url: Optional[str]
) -> None:
    """Remember (or update) the chat that now holds this request's history."""
    convs = _convs_for(account)
    now = time.monotonic()
    if conv is not None:
        conv.user_turns.append(turns[-1][1])
        conv.n_messages = len(turns)
        conv.url = url or conv.url
        conv.last_used = now
    elif url:
        convs.append(Conversation(
            url=url,
            system=system,
            user_turns=[t for r, t in turns if r == "user"],
            n_messages=len(turns),
            last_used=now,
        ))
    convs.sort(key=lambda c: c.last_used, reverse=True)
    del convs[MAX_CONVERSATIONS:]


async def _open_chat(page: Any, home: str, target_url: Optional[str], account: str = "") -> None:
    """Open a blank chat (``target_url`` None) or an existing one, and wait for the composer.

    Detects the logged-out state. Reuses the current page without reloading when it
    is already on the target conversation.
    """
    if target_url is None:
        await page.goto(home, wait_until="domcontentloaded", timeout=int(PAGE_READY_TIMEOUT * 1000))
    elif page.url.rstrip("/") != target_url.rstrip("/"):
        await page.goto(target_url, wait_until="domcontentloaded", timeout=int(PAGE_READY_TIMEOUT * 1000))
    deadline = time.monotonic() + PAGE_READY_TIMEOUT
    probe = """() => ({
        path: location.pathname,
        token: Boolean(localStorage.getItem('userToken')),
        composer: Boolean(document.querySelector('textarea[name="search"], textarea')),
    })"""
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            state = await page.evaluate(probe)
        except Exception:
            await asyncio.sleep(0.3)
            continue
        if state["path"].startswith("/sign_in"):
            raise DeepSeekWebLoginRequired(_login_message(account))
        if state["composer"] and state["token"]:
            # A previous generation may still be running (e.g. the stop button was missed).
            with contextlib.suppress(Exception):
                if await _stop_generation(page):
                    await asyncio.sleep(0.5)
            if target_url is None:
                await _apply_toggles(page)
            return
        await asyncio.sleep(0.3)
    if not state.get("token"):
        raise DeepSeekWebLoginRequired(_login_message(account))
    raise DeepSeekWebError(
        "DeepSeek composer not found (page layout may have changed). "
        'Check the selector textarea[name="search"].'
    )


_CHALLENGE_MESSAGE = (
    "DeepSeek asked for a human verification (CHALLENGE_REQUIRED) and it was not solved in time. "
    "Solve the captcha in the browser window opened by the proxy, chat normally for a moment, then retry."
)


def _login_message(account: str = "") -> str:
    return (
        "DeepSeek web is not logged in"
        f"{f' for account {account!r}' if account else ''}. Log in to chat.deepseek.com in the browser window "
        "opened by the proxy (or run `python -m app.providers.deepseek_web_provider --login "
        f"--account {account}`), then retry. Profile: {_profile_dir(account)}"
    )


async def _submit_prompt(page: Any, prompt: str) -> None:
    """Fill the composer and send with Enter, falling back to the send button."""
    composer = page.locator(COMPOSER_SELECTOR).first
    await composer.fill(prompt)
    await composer.press("Enter")
    await asyncio.sleep(0.6)
    # If the composer still holds the text, Enter was swallowed: click the send button.
    leftover = await composer.input_value()
    if leftover.strip():
        logger.warning("DeepSeek composer not cleared after Enter, clicking send button")
        await page.evaluate("""() => {
            const btn = document.querySelector('.ds-button--primary:not(.ds-button--disabled)');
            if (!btn) return false;
            const key = Object.keys(btn).find((k) => k.startsWith('__reactProps$'));
            if (key && btn[key] && typeof btn[key].onClick === 'function') {
                btn[key].onClick({ stopPropagation() {}, preventDefault() {} });
            } else { btn.click(); }
            return true;
        }""")


async def _stop_generation(page: Any) -> bool:
    """Press the page's stop button; returns False if no stop button was found."""
    return bool(await page.evaluate("""() => {
        const icon = document.querySelector('.ds-icon-stop');
        if (!icon) return false;
        const btn = icon.closest('.ds-button') || icon.closest('button') || icon.parentElement;
        const key = Object.keys(btn).find((k) => k.startsWith('__reactProps$'));
        if (key && btn[key] && typeof btn[key].onClick === 'function') {
            btn[key].onClick({ stopPropagation() {}, preventDefault() {} });
        } else { btn.click(); }
        return true;
    }"""))


async def _web_answer_deltas(system: str, turns: Turns, home: str, account: str = "") -> AsyncIterator[str]:
    """Run one turn in the browser and yield answer text deltas.

    Continues an existing web chat when the request extends one we already
    have (only the newest user message is sent); otherwise opens a new chat and
    replays the whole history. Holds the global lock for the whole turn. If the
    consumer stops early (``</tool_call>`` seen, client disconnect) the page's
    stop button is pressed.

    ``account`` selects which browser/login to drive; every account keeps its
    own browser state and its own set of tracked conversations.
    """
    global _last_turn_end
    state = _state_for(account)
    convs = _convs_for(account)
    async with _LOCK:
        await _pace()
        page = await _ensure_page(state)
        conv = match_conversation(convs, system, turns)
        prompt = build_incremental_prompt(system, turns) if conv else build_full_prompt(system, turns)
        try:
            await _open_chat(page, home, conv.url if conv else None, account)
        except DeepSeekWebLoginRequired:
            raise
        except DeepSeekWebError as e:
            if conv is None:
                raise
            logger.warning("Could not continue DeepSeek chat (%s); replaying history in a new chat", e)
            convs.remove(conv)
            conv, prompt = None, build_full_prompt(system, turns)
            await _open_chat(page, home, None, account)
        logger.info(
            "DeepSeek web turn [account=%s %s] prompt_chars=%d",
            account or "default",
            "continue" if conv else "new chat",
            len(prompt),
        )

        queue: asyncio.Queue = asyncio.Queue()
        state.queue = queue
        parser = DeepSeekStreamParser()
        finished = False
        failed = False
        t_submit = time.monotonic()
        t_first: Optional[float] = None
        got_text = False
        challenge_deadline: Optional[float] = None
        try:
            await _submit_prompt(page, prompt)
            t_submit = time.monotonic()
            while not finished:
                limit = IDLE_TIMEOUT if got_text or t_first else FIRST_BYTE_TIMEOUT
                if challenge_deadline is not None:
                    limit = max(0.1, challenge_deadline - time.monotonic())
                remaining_total = TOTAL_TIMEOUT - (time.monotonic() - t_submit)
                if remaining_total <= 0:
                    raise DeepSeekWebError(f"DeepSeek web turn exceeded {TOTAL_TIMEOUT:.0f}s")
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=min(limit, remaining_total))
                except asyncio.TimeoutError:
                    if challenge_deadline is not None:
                        raise DeepSeekWebError(_CHALLENGE_MESSAGE) from None
                    raise DeepSeekWebError(
                        "Timed out waiting for DeepSeek web stream "
                        "(no request seen; the page layout or login state may have changed)"
                    ) from None

                kind = msg.get("kind")
                if t_first is None and kind in ("start", "chunk"):
                    t_first = time.monotonic()
                if kind == "error":
                    raise DeepSeekWebError(f"DeepSeek web stream failed: {msg.get('message')}")
                if kind == "chunk":
                    for ev_kind, delta in parser.feed(msg.get("text", "")):
                        if ev_kind == "text":
                            if not got_text:
                                got_text = True
                                challenge_deadline = None
                                logger.info("DeepSeek web first-token latency: %.2fs", time.monotonic() - t_submit)
                            yield delta
                    if parser.done:
                        finished = True
                elif kind == "end":
                    for ev_kind, delta in parser.flush():
                        if ev_kind == "text":
                            got_text = True
                            yield delta
                    if not got_text and "CHALLENGE_REQUIRED" in parser.raw_head:
                        # Do NOT reload or resubmit: that would destroy the captcha dialog the user
                        # is solving. Wait in place; the page re-sends its request once solved.
                        logger.warning(
                            "DeepSeek requires human verification; waiting up to %.0fs for it to be solved "
                            "in the browser window",
                            CHALLENGE_WAIT_TIMEOUT,
                        )
                        challenge_deadline = time.monotonic() + CHALLENGE_WAIT_TIMEOUT
                        parser = DeepSeekStreamParser()
                    else:
                        finished = True

            if not got_text:
                raise DeepSeekWebError(f"DeepSeek web returned no answer text. Stream head: {parser.raw_head[:300]!r}")
            logger.info("DeepSeek web turn done in %.2fs", time.monotonic() - t_submit)
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
                    stopped = await _stop_generation(page)
                    logger.info("DeepSeek web stop button pressed: %s", stopped)
                except Exception as e:
                    logger.warning("DeepSeek web stop failed: %s", e)
            try:
                if got_text and not failed:
                    _record_conversation(account, conv, system, turns, await _current_chat_url(page))
                elif conv is not None and conv in convs:
                    convs.remove(conv)  # state of that chat is uncertain after a failed turn
            except Exception as e:
                logger.warning("DeepSeek web could not record conversation: %s", e)
            _last_turn_end = time.monotonic()


# ---------------------------------------------------------------------------
# Public provider API
# ---------------------------------------------------------------------------


async def _answer_deltas(request: AnthropicRequest, route: ResolvedRoute) -> AsyncIterator[str]:
    """Yield answer deltas, truncated right after the first complete tool call."""
    system, turns = split_request(request)
    home = (route.provider.base_url or DEFAULT_HOME).rstrip("/") or DEFAULT_HOME
    account = _account_id(route.provider)
    logger.info(
        "Outgoing DeepSeek web request [provider=%s account=%s] messages=%d",
        route.provider_name,
        account or "default",
        len(turns),
    )

    normalizer = DsmlNormalizer()
    truncator = ToolCallTruncator()
    raw = ""
    try:
        async with contextlib.aclosing(_web_answer_deltas(system, turns, home, account)) as deltas:
            async for delta in deltas:
                raw += delta
                text, call_done = normalizer.feed(delta)
                emit, tag_done = truncator.feed(text) if text else ("", False)
                if emit:
                    yield emit
                if call_done or tag_done:
                    logger.info("DeepSeek web: tool call complete, stopping generation")
                    break
            else:
                tail = normalizer.flush()
                if tail:
                    yield tail
    finally:
        debug_logger.info("DEEPSEEK WEB RAW ANSWER: %r", raw[:6000])


async def send_streaming(request: AnthropicRequest, route: ResolvedRoute) -> AsyncIterator[str]:
    """Stream a DeepSeek web answer as Anthropic SSE event strings.

    Args:
        request: AnthropicRequest - Request already transformed by app.react.
        route: ResolvedRoute - Resolved route (base_url is used as the site URL).

    Yields:
        str - Anthropic SSE events: message_start, one text block, message_delta, message_stop.

    Raises:
        DeepSeekWebLoginRequired: If the browser profile is not logged in.
        DeepSeekWebError: On any other driving failure.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = _estimate_tokens(flatten_prompt(request))
    started = False
    full = ""
    async with contextlib.aclosing(_answer_deltas(request, route)) as deltas:
        async for delta in deltas:
            if not started:
                started = True
                yield _message_start(msg_id, route.model_id, input_tokens)
                yield _text_block_start()
            full += delta
            yield _text_delta(delta)
    if not started:
        yield _message_start(msg_id, route.model_id, input_tokens)
        yield _text_block_start()
    for event in _stream_tail(full):
        yield event


async def send_non_streaming(request: AnthropicRequest, route: ResolvedRoute) -> dict[str, Any]:
    """Collect a DeepSeek web answer into an Anthropic Messages response dict."""
    full = ""
    async with contextlib.aclosing(_answer_deltas(request, route)) as deltas:
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
            "input_tokens": _estimate_tokens(flatten_prompt(request)),
            "output_tokens": _estimate_tokens(full),
        },
    }


# ---------------------------------------------------------------------------
# Admin helpers (used by the dashboard): login status and "open login window"
# ---------------------------------------------------------------------------


async def _status_for(account: str) -> dict[str, Any]:
    """Report one account's browser/login state without disturbing a running turn.

    ``logged_in`` is None when it cannot be known right now (browser not started,
    a turn is running, or the page is not on chat.deepseek.com).
    """
    state = _states.get(account) or _BrowserState(account)
    info: dict[str, Any] = {
        "account": account,
        "profile_dir": _profile_dir(account),
        "headless": os.environ.get("DEEPSEEK_WEB_HEADLESS", "0") == "1",
        "browser_running": state.page is not None,
        "busy": state.queue is not None,
        "logged_in": None,
        "url": None,
    }
    if info["busy"] or state.page is None:
        return info
    async with _LOCK:
        page = state.page
        if page is None or page.is_closed():
            info["browser_running"] = False
            return info
        info["url"] = page.url
        if page.url.startswith(DEFAULT_HOME):
            try:
                info["logged_in"] = bool(await asyncio.wait_for(page.evaluate(
                    "() => Boolean(localStorage.getItem('userToken')) && !location.pathname.startsWith('/sign_in')"
                ), timeout=5))
            except Exception:
                pass
    return info


async def get_status(accounts: Optional[list[str]] = None) -> dict[str, Any]:
    """Report login state for the given account ids (default: the configured ones).

    Returns ``{"active_account": str, "accounts": [status, ...]}``.
    """
    provider = load_config().providers.get("deepseek-web")
    active = _account_id(provider)
    if accounts is None:
        accounts = list(provider.resolved_api_keys()) if provider is not None else []
        if not accounts:
            accounts = [""]
    return {
        "active_account": active,
        "headless": os.environ.get("DEEPSEEK_WEB_HEADLESS", "0") == "1",
        "accounts": [await _status_for(a) for a in accounts],
    }


async def open_login_window(account: str = "") -> dict[str, Any]:
    """Start one account's browser, show chat.deepseek.com and bring the window to the front.

    If a turn is running on that account, the page is left untouched (it may be showing
    a captcha the user has to solve); only the window is raised.
    """
    if os.environ.get("DEEPSEEK_WEB_HEADLESS", "0") == "1":
        raise DeepSeekWebError("DEEPSEEK_WEB_HEADLESS=1: the browser has no window. Unset it and restart the proxy.")
    state = _state_for(account)
    if _LOCK.locked():
        if state.page is not None:
            with contextlib.suppress(Exception):
                await state.page.bring_to_front()
        return await _status_for(account)
    async with _LOCK:
        page = await _ensure_page(state)
        if not page.url.startswith(DEFAULT_HOME):
            await page.goto(DEFAULT_HOME, wait_until="domcontentloaded", timeout=int(PAGE_READY_TIMEOUT * 1000))
        await page.bring_to_front()
    return await _status_for(account)


# ---------------------------------------------------------------------------
# Debug CLI
# ---------------------------------------------------------------------------


async def _cli(args: argparse.Namespace) -> None:
    if args.login:
        state = _state_for(args.account)
        page = await _ensure_page(state)
        await page.goto(DEFAULT_HOME)
        print("Log in to DeepSeek in the opened browser; this command exits automatically once logged in (10 min limit).")
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                try:
                    ok = await page.evaluate(
                        "() => Boolean(localStorage.getItem('userToken')) && !location.pathname.startsWith('/sign_in')"
                    )
                except Exception:
                    ok = False  # page is navigating
                if ok:
                    await asyncio.sleep(2)  # let the profile flush cookies / storage
                    print("Logged in. Profile saved.")
                    return
                await asyncio.sleep(1)
            print("Timed out waiting for login.")
        finally:
            await close()
        return
    t0 = time.monotonic()
    first = True
    try:
        async with contextlib.aclosing(_web_answer_deltas("", [("user", args.prompt)], DEFAULT_HOME, args.account)) as deltas:
            async for delta in deltas:
                if first:
                    print(f"[first token after {time.monotonic() - t0:.2f}s]")
                    first = False
                print(delta, end="", flush=True)
        print(f"\n[total {time.monotonic() - t0:.2f}s]")
    finally:
        await close()


def main() -> None:
    """Entry point for ``python -m app.providers.deepseek_web_provider``."""
    parser = argparse.ArgumentParser(description="Debug the DeepSeek web driver without the proxy.")
    parser.add_argument("--login", action="store_true", help="Open the browser so you can log in once.")
    parser.add_argument("--prompt", default="Say hi in five words.", help="Prompt to send.")
    parser.add_argument("--account", default="", help="Account id (profile dir name); empty = default.")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
