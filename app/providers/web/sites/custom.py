"""Config-declared HTTP site: ``provider_type: "web-http"``.

Most niche chat sites (and self-hosted "system" backends) differ only in their
URL, a couple of headers, the body they expect and the JSON path the answer
lives at. Declaring those in ``config.json`` is enough to use the site — no new
Python module, no browser. That is what makes "one abstraction, many sites"
cheap: a protocol change is a config edit.

Example ``config.json``::

    "my-niche-site": {
      "provider_type": "web-http",
      "base_url": "https://chat.example.com",
      "api_key": "the bearer token copied from the site",
      "headers": {"X-App-Id": "web"},
      "web": {
        "url": "https://chat.example.com/api/chat",
        "method": "POST",
        "headers": {"X-Version": "1.2.3"},
        "body": {
          "model": "{{model}}",
          "stream": true,
          "messages": "{{messages}}"
        },
        "stream": {
          "frames": "sse",
          "text": "choices.0.delta.content",
          "thinking": "choices.0.delta.reasoning_content",
          "done_on": "choices.0.finish_reason",
          "chat_id": "conversation_id"
        }
      },
      "models": {"anything": {"name": "Niche chat", "upstream_model_id": "their-model-id"}}
    }

Template placeholders available in ``web.body`` (and in strings inside it):
``{{prompt}}`` the flattened conversation, ``{{messages}}`` an OpenAI-style
message list, ``{{system}}``, ``{{model}}``, ``{{api_key}}``, ``{{chat_id}}``,
``{{stream}}``.

Supported ``web.stream.frames``: ``sse`` (each ``data:`` frame is JSON),
``jsonl`` (one bare JSON object per line), ``json`` (one JSON document for the
whole answer, non-streaming endpoint).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.providers.web.base import (
    Conversation,
    JsonLineParser,
    SseJsonParser,
    Turns,
    WebProviderError,
    WebSite,
    dig,
)
from app.providers.web.http import HttpRuntime

logger = logging.getLogger("anyclaude")


class PathStreamParser(SseJsonParser):
    """SSE/JSONL parser driven by the JSON paths declared in the config."""

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        self.spec = spec or {}
        self.chat_ref: Optional[str] = None

    def handle(self, obj: Any, out: list[tuple[str, str]]) -> None:
        if not isinstance(obj, dict) and not isinstance(obj, list):
            return
        if self.spec.get("error") and dig(obj, self.spec["error"]):
            raise WebProviderError(f"Upstream error frame: {dig(obj, self.spec['error'])!r}")
        self.emit_text(dig(obj, self.spec.get("text")), out)
        if self.spec.get("thinking"):
            self.emit_text(dig(obj, self.spec["thinking"]), out, "thinking")
        elif self.spec.get("thinking_delta"):
            self.emit_text(dig(obj, self.spec["thinking_delta"]), out, "thinking")
        if self.spec.get("chat_id"):
            ref = dig(obj, self.spec["chat_id"])
            if isinstance(ref, (str, int)):
                self.chat_ref = str(ref)
        done_on = self.spec.get("done_on")
        if done_on and dig(obj, done_on) not in (None, "", False, "null"):
            self.done = True


class JsonLineSiteParser(PathStreamParser):
    """Same as :class:`PathStreamParser` but for one bare JSON object per line."""

    def _handle_line(self, line: str, out: list[tuple[str, str]]) -> None:
        JsonLineParser._handle_line(self, line, out)


class WholeJsonParser(PathStreamParser):
    """Non-streaming endpoint: buffer everything, emit the answer once at the end."""

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        self._acc = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        if len(self.raw_head) < 500:
            self.raw_head += text[: 500 - len(self.raw_head)]
        self._acc += text
        return []

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not self._acc.strip():
            return out
        try:
            obj = json.loads(self._acc)
        except json.JSONDecodeError as e:
            raise WebProviderError(f"Upstream did not return JSON: {self._acc[:300]!r}") from e
        self.handle(obj, out)
        self.done = True
        return out


def _render(value: Any, ctx: dict[str, Any]) -> Any:
    """Substitute ``{{name}}`` placeholders inside a config template."""
    if isinstance(value, str):
        if value.startswith("{{") and value.endswith("}}") and value.count("{{") == 1:
            return ctx.get(value[2:-2].strip(), "")
        for key, replacement in ctx.items():
            if isinstance(replacement, str):
                value = value.replace("{{" + key + "}}", replacement)
        return value
    if isinstance(value, list):
        return [_render(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, ctx) for k, v in value.items()}
    return value


class ConfigSite(WebSite):
    """A site described entirely by its ``web`` block in ``config.json``."""

    name = "web-http"
    label = "自定义网页端 (config)"
    transport = "http"
    token_hint = "the token the site's frontend sends (devtools → Network → request headers)"

    @staticmethod
    def _spec(route: Any) -> dict[str, Any]:
        web = getattr(route.provider, "web", None) or {}
        if not isinstance(web, dict) or not web.get("url"):
            raise WebProviderError(
                "This provider_type 'web-http' needs a 'web' block with at least a 'url' "
                "(and usually 'body' + 'stream') in config.json. See docs/web-providers.md."
            )
        return web

    def new_parser(self, route: Any) -> PathStreamParser:
        spec = self._spec(route).get("stream") or {}
        frames = str(spec.get("frames") or "sse").lower()
        if frames == "json":
            return WholeJsonParser(spec)
        if frames == "jsonl":
            return JsonLineSiteParser(spec)
        return PathStreamParser(spec)

    def http_chat_ref(self, frame: Any) -> Optional[str]:
        return None

    def build_http_request(
        self,
        route: Any,
        system: str,
        turns: Turns,
        conv: Optional[Conversation],
        prompt: str,
    ) -> tuple[str, str, dict[str, str], dict[str, Any]]:
        web = self._spec(route)
        provider = route.provider
        base = (provider.base_url or self.home).rstrip("/")
        token = _token(provider)
        model = route.model_config.upstream_model_id or route.model_id

        ctx = {
            "prompt": prompt,
            "system": system,
            "model": model,
            "api_key": token,
            "chat_id": (conv.url if conv else "") or "",
            "stream": "true",
            "messages": [
                {"role": role, "content": text} for role, text in turns
            ],
        }

        url = str(_render(web["url"], ctx))
        method = str(web.get("method") or self.http_method).upper()

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {token}",
        }
        headers.update(provider.headers or {})
        headers.update({k: str(_render(v, ctx)) for k, v in (web.get("headers") or {}).items()})

        body = _render(web.get("body") or {"messages": "{{messages}}", "stream": True}, ctx)
        if not isinstance(body, dict):
            body = {"messages": body}
        body.update(route.model_config.extra_body or {})
        return method, url, headers, body


def _token(provider: Any) -> str:
    keys = provider.resolved_api_keys() if hasattr(provider, "resolved_api_keys") else []
    active = getattr(provider, "active_account", "") or ""
    if active and active in keys:
        return active
    if keys:
        return keys[0]
    return getattr(provider, "api_key", "") or ""


SITE = ConfigSite()

#: HTTP runtime holding this site's chats and lock.
RUNTIME = HttpRuntime(SITE)
