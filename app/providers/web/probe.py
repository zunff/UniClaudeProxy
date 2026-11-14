"""Paste a cURL, see what the site actually streams back.

Adding a site needs two facts nobody can guess: the request the page sends and
the shape of the frames it gets back. "Copy as cURL" gives us the first one for
free; this module replays it once and returns the head of the response, so the
JSON path for ``web-http`` (or a new site adapter) can be read off real data
instead of guessed.

Only used by the local admin endpoint — it replays *your* request with *your*
cookie against *your* account.
"""

from __future__ import annotations

import shlex
from typing import Any, Optional

import httpx

from app.config import proxy_setting
from app.providers.web.base import WebProviderError

#: Headers that describe the transport, not the request; forwarding them breaks
#: the replay (wrong Content-Length, stale Host, chunked framing).
HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "transfer-encoding",
    "upgrade",
    "pragma",
    "cache-control",
}

MAX_HEAD_BYTES = 4000


def parse_curl(curl: str) -> tuple[str, str, dict[str, str], Optional[bytes]]:
    """Parse a "Copy as cURL" command into ``(method, url, headers, body)``.

    Handles bash/cmd quoting, ``-H/--header``, ``-X/--request``, ``-d/--data*``,
    ``--compressed``, line continuations (``\\`` or ``^``) and a leading ``$``.
    """
    text = (curl or "").strip().replace("\\\r\n", " ").replace("\\\n", " ")
    text = text.replace("^\r\n", " ").replace("^\n", " ").replace("`\r\n", " ")
    if text.startswith("$"):
        text = text[1:]
    try:
        parts = shlex.split(text, posix=True)
    except ValueError as e:
        raise WebProviderError(f"cURL 解析失败: {e}") from e
    if not parts:
        raise WebProviderError("粘贴的命令是空的")
    # Windows "Copy as cURL" yields ``curl.exe`` (sometimes with a full path).
    exe = parts[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if exe not in ("curl", "curl.exe"):
        raise WebProviderError("需要一条以 curl 开头的命令（F12 → 右键请求 → Copy as cURL）")

    url = ""
    method = ""
    headers: dict[str, str] = {}
    body: Optional[str] = None
    i = 1
    while i < len(parts):
        arg = parts[i]
        if arg in ("-H", "--header") and i + 1 < len(parts):
            raw = parts[i + 1]
            if ":" in raw:
                key, value = raw.split(":", 1)
                headers[key.strip()] = value.strip()
            i += 2
            continue
        if arg in ("-X", "--request") and i + 1 < len(parts):
            method = parts[i + 1].upper()
            i += 2
            continue
        if arg in ("-d", "--data", "--data-raw", "--data-binary", "--data-ascii") and i + 1 < len(parts):
            body = parts[i + 1]
            i += 2
            continue
        if arg in ("--url",) and i + 1 < len(parts):
            url = parts[i + 1]
            i += 2
            continue
        if arg in ("-b", "--cookie") and i + 1 < len(parts):
            headers["cookie"] = parts[i + 1]
            i += 2
            continue
        if not arg.startswith("-") and not url:
            url = arg
        i += 1

    if not url:
        raise WebProviderError("这条 cURL 里没有 URL")
    if not method:
        method = "POST" if body is not None else "GET"

    headers = {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}
    return method, url, headers, (body.encode("utf-8") if body is not None else None)


async def probe(curl: str, timeout: float = 45.0) -> dict[str, Any]:
    """Replay one cURL and return the response head (status, type, frames)."""
    method, url, headers, body = parse_curl(curl)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, proxy=proxy_setting()) as client:
            async with client.stream(method, url, headers=headers, content=body) as resp:
                collected = b""
                async for chunk in resp.aiter_bytes():
                    collected += chunk
                    if len(collected) >= MAX_HEAD_BYTES:
                        break
                return {
                    "ok": True,
                    "method": method,
                    "url": url,
                    "status": resp.status_code,
                    "content_type": resp.headers.get("content-type", ""),
                    "sent_headers": sorted(headers.keys()),
                    "head": collected.decode("utf-8", "replace")[:MAX_HEAD_BYTES],
                }
    except httpx.HTTPError as e:
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}
