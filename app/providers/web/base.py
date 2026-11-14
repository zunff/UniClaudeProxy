"""Shared building blocks for *web providers* (consumer chat sites without an API).

A web provider is an LLM reached the way a human reaches it: through a website
whose chat backend is not a public API. Two transports exist:

``browser``
    Playwright drives the real page (composer, Enter) and a page script
    (``driver.js``) forwards the site's *own* streaming response to Python.
    Needed when the site signs requests in JavaScript (DeepSeek, Doubao...).
``http``
    The site's JSON endpoint is called directly with a token copied out of the
    logged-in browser. No browser, no Playwright — cheaper and faster, and the
    only option for sites whose request needs nothing but a bearer token
    (most niche sites exposing an OpenAI-shaped stream).

Everything in this module is transport-agnostic and site-agnostic: prompt
flattening for ReAct (one text reply), web-chat conversation reuse, the
``</tool_call>`` truncator and the Anthropic SSE builders. A site adapter only
supplies a parser (raw stream -> ``("text" | "thinking", delta)``) and a few
hooks.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
import uuid
from typing import Any, Optional, Protocol

from app.models import AnthropicRequest
from app.react.prompt import format_observation_xml, format_tool_call_xml

logger = logging.getLogger("anyclaude")

#: Timings shared by both transports (seconds).
FIRST_BYTE_TIMEOUT = 90.0
IDLE_TIMEOUT = 120.0
TOTAL_TIMEOUT = 900.0

TOOL_CALL_OPEN = "<tool_call"
TOOL_CALL_CLOSE = "</tool_call>"


class WebProviderError(RuntimeError):
    """Raised when a web site cannot be driven or returns no usable answer."""


class WebLoginRequired(WebProviderError):
    """Raised when the configured account is not logged in to the site."""


# ---------------------------------------------------------------------------
# Prompt flattening: a web chat gets one message, not an Anthropic payload.
# ---------------------------------------------------------------------------


def block_to_dict(block: Any) -> dict[str, Any]:
    """Normalise a content block (dict or pydantic model) to a dict."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return dict(block)


def content_to_text(content: Any) -> str:
    """Flatten Anthropic message content into plain text.

    Images are replaced by a placeholder because web providers cannot upload
    attachments yet. Thinking blocks are dropped.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    parts: list[str] = []
    for raw in content:
        block = block_to_dict(raw)
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            parts.append(format_tool_call_xml(block.get("name", ""), block.get("input", {})))
        elif btype == "tool_result":
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner = " ".join(
                    block_to_dict(b).get("text", "") for b in inner if block_to_dict(b).get("type") == "text"
                )
            parts.append(format_observation_xml(str(inner)))
        elif btype == "image":
            parts.append("[image omitted]")
    return "\n\n".join(p for p in parts if p)


def system_to_text(system: Any) -> str:
    """Flatten an Anthropic ``system`` field (str or text blocks) into text."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            block_to_dict(b).get("text", "") for b in system if block_to_dict(b).get("type") == "text"
        )
    return ""


Turns = list[tuple[str, str]]


def split_request(request: AnthropicRequest) -> tuple[str, Turns]:
    """Return ``(system_text, [(role, text), ...])`` for a ReAct-transformed request."""
    system = system_to_text(request.system).strip()
    turns = [(m.role, content_to_text(m.content).strip()) for m in request.messages]
    return system, turns


_TOOL_HINT = " If you need a tool, end your reply with exactly one <tool_call> block and stop."


def wants_tool_calls(system: str) -> bool:
    """True when the ReAct layer is active for this request."""
    return "<tool_call>" in system


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
    if wants_tool_calls(system):
        tail += _TOOL_HINT
    sections.append(tail)
    return "\n\n".join(sections)


def build_incremental_prompt(system: str, turns: Turns) -> str:
    """Message for a continued web conversation: only the newest user turn."""
    text = turns[-1][1]
    if wants_tool_calls(system):
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
    """A web chat we can continue, plus what we already sent into it.

    ``url`` is opaque: a page URL for browser sites, a chat/session id for
    HTTP sites.
    """

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


def record_conversation(
    convs: list[Conversation], conv: Optional[Conversation], system: str, turns: Turns, ref: Optional[str]
) -> None:
    """Remember (or update) the chat that now holds this request's history."""
    import time

    now = time.monotonic()
    if conv is not None:
        conv.user_turns.append(turns[-1][1])
        conv.n_messages = len(turns)
        conv.url = ref or conv.url
        conv.last_used = now
    elif ref:
        # Newest first, so a tie in ``last_used`` (coarse clocks) cannot evict it.
        convs.insert(0, Conversation(
            url=ref,
            system=system,
            user_turns=[t for r, t in turns if r == "user"],
            n_messages=len(turns),
            last_used=now,
        ))
    convs.sort(key=lambda c: c.last_used, reverse=True)
    del convs[MAX_CONVERSATIONS:]


# ---------------------------------------------------------------------------
# Failure guard: stop hammering a site that is already pushing back.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CooldownTracker:
    """Disable one account for a while after a streak of *hard* failures.

    A captcha, a 429 or a 403 is the site telling us to slow down. Retrying
    through it is the single most common way these accounts get burned, so a
    streak disables the account outright: requests fail fast with a clear
    message instead of piling on more traffic.

    Only signals that mean "you are being rate limited / challenged" count.
    Timeouts and transport glitches do not — the fallback for those is the
    normal per-turn timeouts.
    """

    threshold: int = 2
    cooldown_seconds: float = 1800.0
    label: str = ""

    def __post_init__(self) -> None:
        self._failures: dict[str, int] = {}
        self._until: dict[str, float] = {}

    def remaining(self, account: str) -> float:
        """Seconds left in this account's cooldown (0 when it is usable)."""
        return max(0.0, self._until.get(account, 0.0) - time.monotonic())

    def check(self, account: str) -> None:
        """Raise when the account is cooling down, so the caller fails fast."""
        left = self.remaining(account)
        if left > 0:
            raise WebProviderError(
                f"{self.label or 'this site'} is cooling down for {left / 60:.0f} more minute(s) "
                "after repeated challenges/rate limits. Wait it out, switch the active account, "
                "or restart the proxy to clear the cooldown."
            )

    def record(self, account: str, reason: str) -> None:
        """Count one hard failure; start the cooldown once the streak is reached."""
        streak = self._failures.get(account, 0) + 1
        if streak >= self.threshold:
            self._until[account] = time.monotonic() + self.cooldown_seconds
            self._failures[account] = 0
            logger.warning(
                "%s: %s for account %s — cooling down for %.0f min after %d in a row",
                self.label or "web site",
                reason,
                account or "default",
                self.cooldown_seconds / 60,
                streak,
            )
            return
        self._failures[account] = streak
        logger.warning(
            "%s: %s for account %s (%d/%d before a cooldown)",
            self.label or "web site",
            reason,
            account or "default",
            streak,
            self.threshold,
        )

    def success(self, account: str) -> None:
        """Clear the streak after a turn the site answered normally."""
        self._failures.pop(account, None)


# ---------------------------------------------------------------------------
# Parser protocol.
# ---------------------------------------------------------------------------


class StreamParser(Protocol):
    """Incremental parser turning raw upstream text into answer deltas."""

    done: bool
    raw_head: str

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Consume raw text and return newly produced ``(kind, delta)`` tuples."""

    def flush(self) -> list[tuple[str, str]]:
        """Process a trailing fragment that lacked a terminator."""


class Normalizer(Protocol):
    """Rewrites a site's native tool-call markup into ReAct ``<tool_call>`` XML."""

    def feed(self, delta: str) -> tuple[str, bool]:
        """Return ``(text_to_emit, finished)`` for one delta."""

    def flush(self) -> str:
        """Release anything still buffered at end of stream."""


class SseJsonParser:
    """SSE line splitter; subclasses decode each frame in :meth:`handle`.

    Frame facts kept here: ``data:`` payloads, ``event:`` names, ``[DONE]``,
    comment lines, and frames split across network chunks. All JSON handling
    stays in Python so a site parser can be unit tested without a browser.
    """

    def __init__(self) -> None:
        self.done = False
        self.raw_head = ""
        self._buf = ""
        self._event = ""

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
            if self.on_event(self._event):
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
        obj = _loads(payload)
        if obj is _INVALID:
            return
        self.handle(obj, out)

    # -- hooks ------------------------------------------------------------
    def on_event(self, event: str) -> bool:
        """Handle a named SSE event; return True when it ends the stream."""
        return event in ("finish", "close", "done", "end")

    def handle(self, obj: Any, out: list[tuple[str, str]]) -> None:
        """Consume one decoded frame (default: nothing)."""
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def emit_text(text: Any, out: list[tuple[str, str]], kind: str = "text") -> None:
        """Append a delta when it is a non-empty string."""
        if isinstance(text, str) and text:
            out.append((kind, text))


class JsonLineParser(SseJsonParser):
    """Parser for sites streaming one bare JSON object per line (no ``data:``)."""

    def _handle_line(self, line: str, out: list[tuple[str, str]]) -> None:
        if not line or line.startswith(":") or line.startswith("event:") or line.startswith("id:"):
            return
        payload = line[5:] if line.startswith("data:") else line
        if payload.startswith(" "):
            payload = payload[1:]
        if payload.strip() == "[DONE]":
            self.done = True
            return
        obj = _loads(payload)
        if obj is _INVALID:
            return
        self.handle(obj, out)


_INVALID = object()


def _loads(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return _INVALID


def dig(obj: Any, path: Any) -> Any:
    """Read a nested value by dotted/int path: ``"a.b.0.c"`` or ``["a","b",0,"c"]``.

    Returns None as soon as a segment is missing, so a site spec can name an
    optional field without special-casing.
    """
    if path in (None, "", []):
        return obj
    if isinstance(path, str):
        parts: list[Any] = [p for p in re.split(r"[.\/]", path) if p != ""]
    else:
        parts = list(path)
    cur = obj
    for part in parts:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


# ---------------------------------------------------------------------------
# Tool-call handling shared by every web site.
# ---------------------------------------------------------------------------


class ToolCallTruncator:
    """Cut the answer right after the first complete ``</tool_call>``.

    A web page ignores the ``</tool_call>`` stop sequence injected by the ReAct
    layer, so the model keeps writing. This finds the closing tag as soon as it
    appears so the caller can stop generation immediately.
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


# ---------------------------------------------------------------------------
# Anthropic SSE building (web providers emit a text-only stream).
# ---------------------------------------------------------------------------


def sse(event_type: str, data: dict[str, Any]) -> str:
    """Format one Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (the web page exposes no usage numbers)."""
    return max(1, len(text) // 4) if text else 0


def message_start(msg_id: str, model: str, input_tokens: int) -> str:
    return sse("message_start", {
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


def text_block_start() -> str:
    return sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })


def text_delta(text: str) -> str:
    return sse("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    })


def stream_tail(full_text: str) -> list[str]:
    return [
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": estimate_tokens(full_text)},
        }),
        sse("message_stop", {"type": "message_stop"}),
    ]


def anthropic_events_from_text(
    deltas: list[str],
    model: str,
    input_tokens: int = 0,
    message_id: Optional[str] = None,
) -> list[str]:
    """Build a complete minimal Anthropic SSE sequence from text deltas (test/helper)."""
    msg_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    events = [message_start(msg_id, model, input_tokens), text_block_start()]
    events.extend(text_delta(d) for d in deltas if d)
    events.extend(stream_tail("".join(deltas)))
    return events


# ---------------------------------------------------------------------------
# Site description.
# ---------------------------------------------------------------------------


class WebSite:
    """Everything the engine needs to know about one consumer chat site.

    Subclasses set the class attributes and override the hooks they need. A
    browser site must provide ``composer``, ``stream_paths``, ``probe_js`` and
    :meth:`new_parser`; an HTTP site must provide :meth:`build_http_request`
    and :meth:`new_parser`. Class attributes are per-site defaults; per-provider
    overrides stay in ``config.json``.
    """

    #: Provider type used in ``config.json`` (``"provider_type"``).
    name: str = ""
    #: Human label for the dashboard.
    label: str = ""
    #: ``"browser"`` or ``"http"``.
    transport: str = "browser"
    #: Default site URL (``base_url`` in config overrides it).
    home: str = ""
    #: Prefix of the site's environment variables (``<PREFIX>_HEADLESS`` ...).
    env_prefix: str = ""
    #: Directory name for browser profiles, under ``logs/``.
    profile_dirname: str = ""
    #: Models this site offers out of the box (``{model_id: display name}``),
    #: used when the dashboard creates the provider block for the first time.
    default_models: dict[str, str] = {}

    # -- browser hooks ----------------------------------------------------
    #: CSS selector of the message composer.
    composer: str = "textarea"
    #: Path prefixes of the page requests carrying the answer stream.
    stream_paths: tuple[str, ...] = ()
    #: Pathname that means "not logged in".
    login_path: str = ""
    #: Path fragment of a per-conversation URL (enables chat continuation).
    chat_url_marker: str = ""
    #: JS expression returning ``{path, logged_in, composer}``.
    probe_js: str = """() => ({
        path: location.pathname,
        logged_in: true,
        composer: Boolean(document.querySelector('textarea')),
    })"""
    #: JS expression clicking the page's stop button; returns whether it did.
    stop_js: str = """() => false"""
    #: JS expression clicking the send button, used when Enter is swallowed.
    send_button_js: str = ""
    #: Text in the stream head meaning "the site asked for a captcha".
    challenge_marker: str = ""

    # -- http attributes --------------------------------------------------
    #: HTTP method used by the direct-API transport.
    http_method: str = "POST"
    #: Where the user finds the token the site sends (shown by the dashboard).
    token_hint: str = "localStorage / the Authorization header in devtools"

    # -- hooks ------------------------------------------------------------
    def challenge_message(self) -> str:
        """Error text for a captcha that was not solved in time."""
        return (
            f"{self.label or self.name} asked for a human verification and it was not solved in time. "
            "Solve the captcha in the browser window opened by the proxy, chat normally for a moment, then retry."
        )
    def new_parser(self, route: Any) -> Any:
        """Return a fresh stream parser for one turn."""
        raise NotImplementedError

    def new_normalizer(self, route: Any) -> Optional[Any]:
        """Return a fresh tool-call normalizer, or None when the site needs none."""
        return None

    async def prepare_new_chat(self, page: Any, route: Any) -> None:
        """Set per-turn page toggles (thinking, web search...) on a fresh chat."""

    async def after_submit(self, page: Any) -> None:
        """React to the page right after the prompt was sent (optional)."""

    def build_http_request(
        self, route: Any, system: str, turns: Turns, conv: Optional[Conversation], prompt: str
    ) -> tuple[str, str, dict[str, str], dict[str, Any]]:
        """Return ``(method, url, headers, body)`` for one direct-HTTP turn."""
        raise NotImplementedError

    def http_chat_ref(self, frame: Any) -> Optional[str]:
        """Return a conversation id carried by a frame (enables continuation)."""
        return None

    @property
    def legacy_profile_dir(self) -> Optional[str]:
        """Pre-multi-account profile path a default account keeps using."""
        return None
