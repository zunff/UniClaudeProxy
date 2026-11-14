"""DeepSeek web (chat.deepseek.com) site adapter.

The provider does not call any private API directly. It drives the real web
page (fill the composer, press Enter) and listens to the page's *own* streaming
request via the generic ``web/driver.js`` init script.

Expected in ``api_keys`` are *account ids*: each one gets its own persistent
browser profile, browser instance and tracked web chats. Which one serves
requests is chosen by ``active_account`` (empty = the default profile).

Environment variables (all optional):
    DEEPSEEK_WEB_PROFILE_DIR  Base directory for browser profiles
                              (default: logs/deepseek-web-profiles); named accounts
                              use "<base>/<account>", the default account keeps
                              logs/deepseek-web-profile so an existing login survives.
    DEEPSEEK_WEB_CHANNEL      Playwright browser channel, e.g. "chrome" (default) or "" for bundled Chromium.
    DEEPSEEK_WEB_HEADLESS     "1" to run headless (default "0"; headed is needed for the first login).
    DEEPSEEK_WEB_THINKING     "1" to switch DeepThink on for new chats (default off; smart search stays off).
    DEEPSEEK_WEB_MIN_INTERVAL Minimum seconds between turns (default 3), per account.

Debugging without the proxy:
    python -m app.providers.web.capture deepseek-web --login [--account NAME]
    python -m app.providers.web.capture deepseek-web --prompt "say hi"
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from app.providers.web.base import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    SseJsonParser,
    WebSite,
)
from app.providers.web.engine import BrowserRuntime, runtime_for
from app.react.prompt import format_tool_call_xml

logger = logging.getLogger("anyclaude")

#: Profile dir used before multi-account support; the default account keeps using
#: it so an existing login is not lost.
LEGACY_PROFILE_DIR = Path("logs") / "deepseek-web-profile"

COMPOSER_SELECTOR = 'textarea[name="search"], textarea'


class DeepSeekStreamParser(SseJsonParser):
    """Incremental parser for DeepSeek's SSE + abbreviated JSON-patch stream.

    Frame facts: each ``data:`` frame is a patch ``{"p": path, "o": op, "v":
    value}``; ``p`` / ``o`` are omitted when unchanged from the previous frame;
    ``o == "BATCH"`` carries a list of sub-patches whose paths are relative to
    ``p``. ``done`` becomes true once the stream signals completion.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fragments: list[dict[str, str]] = []
        self._path = ""
        self._op = "SET"
        self._legacy = {"content": "", "thinking_content": ""}

    # The generic line splitter feeds decoded frames here.
    def handle(self, obj: Any, out: list[tuple[str, str]]) -> None:
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
# A parameter written either natively (``<｜DSML｜ parameter ...>``) or in the
# ReAct form the model mixes in (``<parameter name="x">``).
_PARAM_ANY_RE = re.compile(
    rf'<(?:\s*{_BAR}+\s*DSML{_BAR}+\s*)?parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)'
    rf"(?=</?\s*{_BAR}*\s*(?:DSML|parameter)|\Z)",
    re.S,
)
_NAME_RE = re.compile(r"<name>\s*(.*?)\s*</name>", re.S)


def _params_from_text(text: str) -> dict[str, Any]:
    """Recover ``{name: value}`` from mixed DSML / ReAct parameter markup."""
    args: dict[str, Any] = {}
    for name, is_string, raw in _PARAM_ANY_RE.findall(text):
        if is_string == "true":
            args[name] = raw
            continue
        try:
            args[name] = json.loads(raw.strip())
        except json.JSONDecodeError:
            args[name] = raw
    return args


def _call_from_header(header: str, block: str) -> Optional[str]:
    """Build a ``<tool_call>`` for a reply that opened one but wrote DSML params.

    The web model sometimes starts in the ReAct format (``<tool_call>`` +
    ``<name>``) and then continues with its native markup. The header is held
    back so the whole call can be re-emitted canonically instead of leaving a
    half-written ``<parameter>`` tag in the stream.
    """
    m = _NAME_RE.search(header)
    if not m:
        return None
    return format_tool_call_xml(m.group(1), _params_from_text(header + block))


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
        self._header = ""  # a held-back, still-open "<tool_call> ... <name>" header

    @staticmethod
    def _hold_from(buf: str) -> Optional[int]:
        """Index the buffer must be held from: a partial marker or an open tool call."""
        candidates: list[int] = []
        idx = buf.rfind("<")
        if idx >= 0:
            tail = buf[idx:]
            # A partial marker ("<", "<|", "<|D"...) or a partial tool-call opener.
            if _DSML_HOLD_RE.fullmatch(tail) or TOOL_CALL_OPEN.startswith(tail):
                candidates.append(idx)
        open_idx = buf.rfind(TOOL_CALL_OPEN)
        if open_idx >= 0 and TOOL_CALL_CLOSE not in buf[open_idx:]:
            candidates.append(open_idx)
        return min(candidates) if candidates else None

    def feed(self, delta: str) -> tuple[str, bool]:
        """Return ``(text_to_emit, finished)``; finished means a call was completed."""
        self._buf += delta
        pre = ""
        if not self._active:
            m = _DSML_START_RE.search(self._buf)
            if m:
                head = self._buf[: m.start()]
                self._buf, self._active = self._buf[m.start():], True
                # The model opened a ReAct tool call and continued in DSML: keep the
                # header back and rebuild the whole call when it closes.
                cut = head.rfind(TOOL_CALL_OPEN)
                if cut >= 0 and TOOL_CALL_CLOSE not in head[cut:]:
                    self._header, pre = head[cut:], head[:cut]
                else:
                    pre = head
            else:
                idx = self._hold_from(self._buf)
                if idx is None:
                    out, self._buf = self._buf, ""
                    return out, False
                out, self._buf = self._buf[:idx], self._buf[idx:]
                return out, False
        end = _DSML_END_RE.search(self._buf)
        if not end:
            return pre, False
        block, self._buf = self._buf[: end.end()], ""
        call = dsml_to_tool_call(block)
        if call is None and self._header:
            call = _call_from_header(self._header, block)
        self._header = ""
        return pre + (call or block), True

    def flush(self) -> str:
        """Release anything still buffered at end of stream."""
        buf, self._buf = self._buf, ""
        if self._active:
            return dsml_to_tool_call(buf) or buf
        return self._header + buf if self._header else buf


_TOGGLES_JS = """([wantThink, wantSearch]) => {
    const names = {
        think: ['深度思考', 'DeepThink'],
        search: ['智能搜索', 'Search'],
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

_PROBE_JS = """() => ({
    path: location.pathname,
    logged_in: Boolean(localStorage.getItem('userToken')),
    composer: Boolean(document.querySelector('textarea[name="search"], textarea')),
})"""

_STOP_JS = """() => {
    const icon = document.querySelector('.ds-icon-stop');
    if (!icon) return false;
    const btn = icon.closest('.ds-button') || icon.closest('button') || icon.parentElement;
    const key = Object.keys(btn).find((k) => k.startsWith('__reactProps$'));
    if (key && btn[key] && typeof btn[key].onClick === 'function') {
        btn[key].onClick({ stopPropagation() {}, preventDefault() {} });
    } else { btn.click(); }
    return true;
}"""

_SEND_JS = """() => {
    const btn = document.querySelector('.ds-button--primary:not(.ds-button--disabled)');
    if (!btn) return false;
    const key = Object.keys(btn).find((k) => k.startsWith('__reactProps$'));
    if (key && btn[key] && typeof btn[key].onClick === 'function') {
        btn[key].onClick({ stopPropagation() {}, preventDefault() {} });
    } else { btn.click(); }
    return true;
}"""


class DeepSeekSite(WebSite):
    """chat.deepseek.com — browser transport, DSML tool-call fallback, toggles."""

    name = "deepseek-web"
    label = "DeepSeek 网页端"
    transport = "browser"
    home = "https://chat.deepseek.com"
    env_prefix = "DEEPSEEK_WEB"
    profile_dirname = "deepseek-web-profiles"
    default_models = {"deepseek-web": "DeepSeek Web（网页端）"}
    composer = COMPOSER_SELECTOR
    stream_paths = (
        "/api/v0/chat/completion",
        "/api/v0/chat/edit_message",
        "/api/v0/chat/regenerate",
    )
    login_path = "/sign_in"
    chat_url_marker = "/a/chat/s/"
    probe_js = _PROBE_JS
    stop_js = _STOP_JS
    send_button_js = _SEND_JS
    challenge_marker = "CHALLENGE_REQUIRED"

    def new_parser(self, route: Any) -> DeepSeekStreamParser:
        return DeepSeekStreamParser()

    def new_normalizer(self, route: Any) -> DsmlNormalizer:
        return DsmlNormalizer()

    async def prepare_new_chat(self, page: Any, route: Any) -> None:
        """Set DeepThink per DEEPSEEK_WEB_THINKING (default off) and turn web search off."""
        want_think = os.environ.get("DEEPSEEK_WEB_THINKING", "0") == "1"
        try:
            result = await page.evaluate(_TOGGLES_JS, [want_think, False])
            import asyncio

            await asyncio.sleep(0.2)
            logger.info("DeepSeek web toggles: %s", result)
        except Exception as e:
            logger.warning("DeepSeek web could not set toggles: %s", e)

    @property
    def legacy_profile_dir(self) -> Optional[str]:
        return str(LEGACY_PROFILE_DIR)


SITE = DeepSeekSite()

#: Browser runtime holding this site's accounts, conversations and lock.
RUNTIME: BrowserRuntime = runtime_for(SITE)

# Backwards-compatible module-level views (see app/providers/deepseek_web_provider.py).
_states = RUNTIME.states


def _state_for(account: str) -> Any:
    """Return (creating on demand) the browser state of one account."""
    return RUNTIME.state_for(account)


def _convs_for(account: str) -> list[Any]:
    """Return the tracked DeepSeek web chats of one account."""
    return RUNTIME.convs_for(account)


def _profile_root() -> Path:
    """Base directory holding one browser profile per named account."""
    return RUNTIME.profile_root()


def _profile_dir(account: str = "") -> str:
    """Profile directory for one account (default account keeps the legacy path)."""
    return RUNTIME.profile_dir(account)


def _account_id(provider: Any = None) -> str:
    """Return the active account id for a deepseek-web provider config."""
    return RUNTIME.account_for(provider)


__all__ = [
    "DeepSeekSite",
    "DeepSeekStreamParser",
    "DsmlNormalizer",
    "LEGACY_PROFILE_DIR",
    "RUNTIME",
    "SITE",
]
