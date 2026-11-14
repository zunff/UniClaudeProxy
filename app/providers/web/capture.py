"""Command-line capture tool: drive one web site without the proxy.

This is the tool for adding a site. It sends one prompt through a site adapter
and prints the answer; with ``--raw`` it also prints every chunk of the site's
own stream, which is what a new parser has to be written against.

    python -m app.providers.web.capture --list
    python -m app.providers.web.capture deepseek-web --login          # browser site
    python -m app.providers.web.capture deepseek-web --prompt "say hi"
    python -m app.providers.web.capture hailuo-web --prompt "say hi"
    python -m app.providers.web.capture deepseek-web --prompt "hi" --raw 2>frames.log

Discovery mode for an unknown site: set the provider's ``stream_paths`` to
``["*"]`` (or use ``--discover``) so the page driver forwards every streaming
response, then read the frames from ``--raw``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from typing import Any, Optional

from app.config import load_config, ProviderConfig
from app.providers.web import registry
from app.providers.web.base import WebProviderError


class _DebugProvider(ProviderConfig):
    """ProviderConfig for a site that has no block in config.json yet."""

    def __init__(self, home: str, account: str = "") -> None:  # noqa: D401 - simple container
        super().__init__(provider_type="debug", base_url=home, api_keys=[account] if account else [])


class _DebugModelConfig:
    upstream_model_id: Optional[str] = None
    extra_body: dict[str, Any] = {}
    use_react = True
    reasoning: dict[str, Any] = {}


class _DebugRoute:
    """Just enough of ResolvedRoute for one turn."""

    def __init__(self, site_name: str, home: str, account: str = "") -> None:
        self.provider = _DebugProvider(home, account)
        self.provider_name = site_name
        self.provider_type = site_name
        self.model_id = site_name
        self.model_config = _DebugModelConfig()


class _TeeParser:
    """Wraps a parser and echoes every raw chunk to stderr (``--raw``)."""

    def __init__(self, parser: Any, stream: Any) -> None:
        self._parser = parser
        self._stream = stream

    def feed(self, text: str) -> Any:
        self._stream.write(f"--- chunk {len(text)} bytes ---\n{text}\n")
        self._stream.flush()
        return self._parser.feed(text)

    def flush(self) -> Any:
        return self._parser.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parser, name)


def _provider_for(site: Any) -> tuple[Any, Any]:
    """Return ``(route, provider_name)`` for a site, using config.json when present."""
    cfg = load_config()
    for name, provider in cfg.providers.items():
        if provider.provider_type == site.name:
            route = _DebugRoute(site.name, provider.base_url or site.home)
            route.provider = provider  # real config: keeps api_key / headers / active account
            return route, name
    return _DebugRoute(site.name, site.home), site.name


async def _turn(site: Any, prompt: str, account: str, raw: bool) -> None:
    route, provider_name = _provider_for(site)
    if account:
        route.provider.api_keys = [account]
        route.provider.active_account = account
    runtime = registry.runtime_for(site.name)
    if raw:
        original = site.new_parser
        site.new_parser = lambda _route: _TeeParser(original(_route), sys.stderr)  # type: ignore[method-assign]
    t0 = time.monotonic()
    first = True
    try:
        async with contextlib.aclosing(runtime.turn("", [("user", prompt)], route)) as deltas:
            async for delta in deltas:
                if first:
                    print(f"[first token after {time.monotonic() - t0:.2f}s]")
                    first = False
                print(delta, end="", flush=True)
        print(f"\n[total {time.monotonic() - t0:.2f}s]")
    finally:
        if raw:
            site.new_parser = original  # type: ignore[method-assign]
        await registry.close(provider_name)


async def _login(site: Any, account: str) -> None:
    """Open the site and wait until the login state flips to logged in."""
    if site.transport != "browser":
        route, _ = _provider_for(site)
        print(f"{site.label or site.name} is an HTTP site: no window to open.")
        print(f"Where to get the token: {site.token_hint}")
        return
    route, provider_name = _provider_for(site)
    await registry.open_login_window(provider_name, account)
    print(f"Log in to {site.home} in the opened browser; this command exits once logged in (10 min limit).")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = await registry.get_status(provider_name)
        entry = next((a for a in status["accounts"] if a["account"] == account), None)
        if entry and entry.get("logged_in"):
            await asyncio.sleep(2)  # let the profile flush cookies / storage
            print("Logged in. Profile saved.")
            await registry.close(provider_name)
            return
        await asyncio.sleep(1)
    print("Timed out waiting for login.")
    await registry.close(provider_name)


def _print_sites() -> None:
    configured = {item["provider"] for item in registry.configured_providers()}
    print(f"{'site':<14} {'transport':<9} {'configured':<11} label")
    for site in registry.SITE_LIST:
        flag = "yes" if site.name in configured else "no"
        print(f"{site.name:<14} {site.transport:<9} {flag:<11} {site.label}")


async def _main_async(args: argparse.Namespace) -> None:
    site = registry.site_for(args.site)
    if site is None:
        raise SystemExit(f"Unknown site {args.site!r}. Try --list.")
    if args.logout or args.login:
        await _login(site, args.account)
        return
    await _turn(site, args.prompt, args.account, args.raw)


def main(default_site: Optional[str] = None) -> None:
    """Entry point for ``python -m app.providers.web.capture``."""
    parser = argparse.ArgumentParser(description="Drive one web provider site without the proxy.")
    parser.add_argument("site", nargs="?", default=default_site, help="site name, e.g. deepseek-web")
    parser.add_argument("--list", action="store_true", help="list known sites and exit")
    parser.add_argument("--login", action="store_true", help="open the browser so you can log in once")
    parser.add_argument("--logout", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prompt", default="Say hi in five words.", help="prompt to send")
    parser.add_argument("--account", default="", help="account id (profile dir name); empty = default")
    parser.add_argument("--raw", action="store_true", help="echo every raw stream chunk to stderr")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.raw else logging.DEBUG)
    if args.list:
        _print_sites()
        return
    if not args.site:
        parser.error("a site name is required (see --list)")
    try:
        asyncio.run(_main_async(args))
    except WebProviderError as e:
        raise SystemExit(f"error: {e}") from e


if __name__ == "__main__":
    main()
