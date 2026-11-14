"""海螺AI / MiniMax Agent (agent.minimax.cn) — browser transport.

Note: ``hailuoai.com`` is now MiniMax's video/image generator; the chat product is
"MiniMax Code / Agent", which lives at ``agent.minimaxi.com`` (redirects to
``agent.minimax.cn``). This adapter targets the agent site.

Why browser and not a direct HTTP call: the page signs its own requests (composite
token, device fingerprint) through
``POST /minimax-cloud/api/v1/agent/<agent>/session`` and
``POST /minimax-cloud/api/v1/session/<sid>/message`` — the second one goes to the
separate host ``agent-stream.minimax.cn`` and answers with a ``text/event-stream``
of MiniMax's own agent events. We let the page send and only read that stream.

Frame shapes seen in the wild (``data:{...}`` — no space after the colon):

    {"type":"session.created", …}
    {"type":6,"agent_message_chunk":{"msg_id":"msg-…","msg_content":"收到",
                                     "chunk_index":0,"role":"assistant",…}}
    {"type":6,"agent_message_chunk":{…,"chunk_index":1,"finish":true,"finish_reason":"stop"}}
    {"type":2,"agent_message":{"msg_id":"msg-…","msg_content":"收到","role":"assistant",
                               "usage":{"input_tokens":907,"output_tokens":3,…}}}
    {"type":"session.finish", …}

``*_chunk`` frames carry the text (one frame per burst, not per token) and the
later ``agent_message`` frame repeats the whole message, so text is deduplicated
per ``msg_id``. The user's own message is echoed on the same stream — it is
dropped by ``role``.

Discovery, if a site redesign breaks this:

    HAILUO_WEB_STREAM_PATHS="*" python -m app.providers.web.capture hailuo-web --prompt hi --raw 2>frames.log
"""

from __future__ import annotations

import logging
from typing import Any

from app.providers.web.base import SseJsonParser, WebSite

logger = logging.getLogger("anyclaude")

DEFAULT_HOME = "https://agent.minimaxi.com"


class HailuoStreamParser(SseJsonParser):
    """Parser for MiniMax Agent's own agent-event stream.

    Any object carrying ``msg_content`` is treated as content, except the echo of
    the user's own message (``role == "user"``). Keys naming reasoning
    (``agent_reasoning_chunk`` and friends) become thinking deltas instead of
    answer text, so the same parser survives a server-side frame rename.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sent: dict[str, str] = {}

    def handle(self, obj: Any, out: list[tuple[str, str]]) -> None:
        if not isinstance(obj, dict):
            return

        # MiniMax wraps errors in base_resp; a non-zero code means the turn died.
        base = obj.get("base_resp")
        if isinstance(base, dict):
            code = base.get("status_code")
            if code not in (None, 0):
                raise RuntimeError(f"海螺上游错误: {base.get('status_msg') or code}")

        kind = obj.get("type")
        if kind == "session.finish":
            self.done = True
        elif kind == "session.error":
            self.done = True
            raise RuntimeError(f"海螺会话错误: {obj.get('payload') or obj}")

        for key, payload in obj.items():
            if not isinstance(payload, dict):
                continue
            # The closing frame carries only ``finish`` — no content at all.
            if payload.get("finish"):
                self.done = True
            text = payload.get("msg_content")
            if not isinstance(text, str) or not text:
                continue
            if str(payload.get("role") or "") == "user":
                continue
            self._emit(key, payload, text, out)

    def _emit(self, key: str, payload: dict, text: str, out: list[tuple[str, str]]) -> None:
        """Emit only what has not been emitted yet for this message."""
        kind = "thinking" if ("reason" in key or "think" in key) else "text"
        bucket = f"{kind}:{payload.get('msg_id') or key}"
        prev = self._sent.get(bucket, "")
        if text == prev:
            return
        if prev and text.startswith(prev):
            delta, total = text[len(prev):], text  # cumulative snapshot
        elif prev and text.endswith(prev):
            # A cumulative snapshot whose start we never saw (the frame that
            # carried it was lost): fill in the missing prefix only.
            delta, total = text[: len(text) - len(prev)], text
        else:
            delta, total = text, prev + text  # plain delta
        self._sent[bucket] = total
        self.emit_text(delta, out, kind)


_PROBE_JS = """() => {
    const path = location.pathname;
    return {
        path,
        logged_in: !/\\/(login|signin|auth)/i.test(path),
        composer: Boolean(
            document.querySelector('div.ProseMirror[contenteditable="true"]')
            || document.querySelector('textarea')
        ),
    };
}"""

_STOP_JS = """() => {
    const btn = [...document.querySelectorAll('button')].find((b) => {
        const label = (b.getAttribute('aria-label') || b.innerText || '').trim();
        return /^(stop|\\u505c\\u6b62|\\u4e2d\\u65ad)/i.test(label);
    });
    if (!btn) return false;
    btn.click();
    return true;
}"""

_SEND_JS = """() => {
    const visible = (el) => el && el.offsetParent !== null && !el.disabled;
    // 1) an explicit send/发送 button
    const labelled = [...document.querySelectorAll('button')].filter(visible).find((b) => {
        const label = (b.getAttribute('aria-label') || b.title || b.innerText || '').trim();
        return /(send|发送|提交)/i.test(label);
    });
    if (labelled) { labelled.click(); return 'labelled'; }
    // 2) buttons inside the composer's own container (the editor is a
    //    contenteditable div, so we cannot start from a <textarea>)
    const root = document.querySelector('div.ProseMirror[contenteditable="true"]')
        || document.querySelector('textarea');
    let node = root;
    for (let i = 0; i < 6 && node; i++) {
        node = node.parentElement;
        if (!node) break;
        const btns = [...node.querySelectorAll('button')].filter(visible);
        if (btns.length) { btns[btns.length - 1].click(); return 'container'; }
    }
    // 3) any button holding only an svg (icon buttons) near the caret
    const icons = [...document.querySelectorAll('button')].filter(
        (b) => visible(b) && b.querySelector('svg') && !(b.innerText || '').trim()
    );
    if (icons.length) { icons[icons.length - 1].click(); return 'icon'; }
    return false;
}"""


class HailuoSite(WebSite):
    """海螺AI / MiniMax Agent — browser transport, agent-event parser."""

    name = "hailuo-web"
    label = "海螺AI / MiniMax Agent"
    transport = "browser"
    home = DEFAULT_HOME
    env_prefix = "HAILUO_WEB"
    profile_dirname = "hailuo-web"
    default_models = {"hailuo": "海螺AI（网页端）"}

    # 海螺的输入框是 TipTap/ProseMirror 的 contenteditable div，不是 textarea
    composer = 'div.ProseMirror[contenteditable="true"]'
    # Only the per-turn answer stream is listened to. The page also holds an
    # /api/v1/events event bus open from load (session/title/permission events);
    # its frames add nothing the answer stream does not already carry.
    # Note: no trailing slash — driver.js matches ``path.startsWith(p + '/')``.
    stream_paths = ("/minimax-cloud/api/v1/session",)
    login_path = "/login"
    chat_url_marker = ""
    probe_js = _PROBE_JS
    stop_js = _STOP_JS
    send_button_js = _SEND_JS

    def new_parser(self, route: Any) -> HailuoStreamParser:
        return HailuoStreamParser()


SITE = HailuoSite()
