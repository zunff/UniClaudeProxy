"""Site registry and the admin-facing surface of the web providers.

Adding a site = one module in :mod:`app.providers.web.sites` that subclasses
:class:`~app.providers.web.base.WebSite` and is listed in :data:`SITE_LIST`.
Optional sites are switched off from ``config.json`` (delete the provider block)
— nothing here touches the network until a request arrives.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import load_config, read_provider_block, write_provider_block
from app.providers.web.base import WebProviderError, WebSite
from app.providers.web.engine import BrowserRuntime
from app.providers.web.http import HttpRuntime
from app.providers.web.sites.custom import RUNTIME as CUSTOM_RUNTIME
from app.providers.web.sites.custom import SITE as CUSTOM_SITE
from app.providers.web.sites.deepseek import RUNTIME as DEEPSEEK_RUNTIME
from app.providers.web.sites.deepseek import SITE as DEEPSEEK_SITE
from app.providers.web.sites.hailuo import SITE as HAILUO_SITE

logger = logging.getLogger("anyclaude")

#: Every known site, in dashboard order.
SITE_LIST: tuple[WebSite, ...] = (DEEPSEEK_SITE, HAILUO_SITE, CUSTOM_SITE)

SITES: dict[str, WebSite] = {site.name: site for site in SITE_LIST}

RUNTIMES: dict[str, Any] = {
    DEEPSEEK_SITE.name: DEEPSEEK_RUNTIME,
    HAILUO_SITE.name: BrowserRuntime(HAILUO_SITE).register(),
    CUSTOM_SITE.name: CUSTOM_RUNTIME,
}


def is_web_provider(provider_type: str) -> bool:
    """True when this ``provider_type`` is served by a web site."""
    return provider_type in SITES


def looks_like_web_provider(provider_type: str) -> bool:
    """True for a ``provider_type`` that claims to be a web site (``*-web``).

    Catches blocks left behind by a removed site (or a typo): the dashboard
    lists them on the 网页端 page as broken instead of showing them as a normal
    API provider on the Models page.
    """
    return provider_type in SITES or provider_type.endswith("-web")


def site_for(provider_type: str) -> Optional[WebSite]:
    """Return the site serving a ``provider_type`` (None when unknown)."""
    return SITES.get(provider_type)


def runtime_for(site_name: str) -> Any:
    """Return the runtime (browser or HTTP) holding one site's state."""
    runtime = RUNTIMES.get(site_name)
    if runtime is None:
        raise ValueError(f"Unknown web site {site_name!r}")
    return runtime


def site_for_provider(provider_name: str) -> Optional[WebSite]:
    """Return the site behind a provider block name from ``config.json``."""
    provider = load_config().providers.get(provider_name)
    if provider is None:
        return SITES.get(provider_name)
    return SITES.get(provider.provider_type) or SITES.get(provider_name)


def configured_providers() -> list[dict[str, Any]]:
    """Describe every configured web provider (drives the dashboard's site picker)."""
    out: list[dict[str, Any]] = []
    for name, provider in load_config().providers.items():
        site = SITES.get(provider.provider_type)
        if site is None:
            if not looks_like_web_provider(provider.provider_type):
                continue
            # A block whose site is gone (or misspelt): keep it visible so the
            # user sees why requests to it fail, instead of it silently turning
            # into a normal provider on the Models page.
            out.append({
                "provider": name,
                "provider_type": provider.provider_type,
                "label": provider.provider_type,
                "transport": "unknown",
                "home": provider.base_url or "",
                "active_account": provider.active_account or "",
                "accounts": len(provider.resolved_api_keys()),
                "token_hint": None,
                "cookies_configured": bool(getattr(provider, "cookies", "")),
                "models": list(provider.models or {}),
                "unregistered": True,
            })
            continue
        runtime = runtime_for(site.name)
        out.append({
            "provider": name,
            "provider_type": site.name,
            "label": site.label or site.name,
            "transport": site.transport,
            "home": provider.base_url or site.home,
            "active_account": runtime.account_for(provider),
            "accounts": len(provider.resolved_api_keys()),
            "token_hint": site.token_hint if site.transport == "http" else None,
            "cookies_configured": bool(getattr(provider, "cookies", "")),
            # Mounted model ids: the dashboard lists a web provider's routes
            # here instead of on the Models page.
            "models": list(provider.models or {}),
        })
    out.sort(key=lambda item: item["label"])
    return out


def site_templates() -> list[dict[str, Any]]:
    """Every site the dashboard can create a provider for, plus its state."""
    configured: dict[str, str] = {}
    for name, provider in load_config().providers.items():
        if provider.provider_type in SITES:
            configured.setdefault(provider.provider_type, name)
    out: list[dict[str, Any]] = []
    for site in SITE_LIST:
        if not site.default_models:
            continue  # config-declared sites (web-http) need a hand-written block
        out.append({
            "site": site.name,
            "label": site.label or site.name,
            "transport": site.transport,
            "home": site.home,
            "configured": site.name in configured,
            "provider": configured.get(site.name),
            # Routes this site would serve once added — the mapping dropdown
            # keeps them listed even before the provider exists in config.json.
            "default_models": dict(site.default_models),
        })
    return out


def ensure_provider(site_name: str) -> str:
    """Create the provider block for a known site (idempotent); return its name.

    This is what the dashboard's "添加站点" button calls: no config.json editing,
    no CLI — the block is written from the site's own defaults and hot-reloaded.
    """
    site = SITES.get(site_name)
    if site is None or not site.default_models:
        raise WebProviderError(
            f"{site_name!r} cannot be created from the dashboard; declare it in config.json instead."
        )
    for name, provider in load_config().providers.items():
        if provider.provider_type == site.name:
            return name
    write_provider_block(site.name, {
        "provider_type": site.name,
        "base_url": site.home,
        "api_key": "",
        "models": {
            model_id: {"name": label, "use_react": True}
            for model_id, label in site.default_models.items()
        },
    })
    logger.info("created web provider block %r from the dashboard", site.name)
    return site.name


def _account_list(provider: Any) -> list[str]:
    """Accounts (browser) or tokens (HTTP) configured for a provider."""
    keys = list(provider.resolved_api_keys()) if provider is not None else []
    return keys or [""]


async def get_status(provider_name: str, accounts: Optional[list[str]] = None) -> dict[str, Any]:
    """Report browser/login (or token) state of one web provider.

    Returns ``{"provider", "provider_type", "label", "transport",
    "active_account", "accounts": [...]}``.
    """
    provider = load_config().providers.get(provider_name)
    site = site_for_provider(provider_name)
    if site is None:
        raise WebProviderError(f"{provider_name!r} is not a web provider (no provider_type in config.json)")
    runtime = runtime_for(site.name)
    pool = accounts if accounts is not None else _account_list(provider)

    entries: list[dict[str, Any]] = []
    for account in pool:
        if site.transport == "http":
            entry = await runtime.status(account)
            entry["label"] = runtime.mask(account)
            entry["account"] = account
        else:
            entry = await runtime.status(account)
            entry["label"] = account or "默认账号"
        entries.append(entry)

    result: dict[str, Any] = {
        "provider": provider_name,
        "provider_type": site.name,
        "label": site.label or site.name,
        "transport": site.transport,
        "home": (provider.base_url if provider is not None else None) or site.home,
        "active_account": runtime.account_for(provider),
        "headless": True if site.transport == "http" else runtime.env_flag("HEADLESS", False),
        "token_hint": site.token_hint if site.transport == "http" else None,
        "cookies_configured": bool(getattr(provider, "cookies", "")),
        "accounts": entries,
    }
    if site.transport == "http":
        result["active_account_label"] = runtime.mask(result["active_account"])
    return result


async def _verify_captured_token(site: Any, provider: Any, token: str, timeout: float = 30.0) -> bool:
    """Ask the site whether it accepts this token (one throwaway request).

    Returns False only on 401/403 — the signature of an anonymous visitor token
    captured before the user logged in. Network errors and unexpected response
    shapes count as "unknown", so our own guesses about the protocol can never
    block a working token.
    """
    import httpx

    from app.config import ResolvedRoute

    model_id = next(iter(getattr(provider, "models", {}) or {}), "")
    if not model_id or not hasattr(site, "build_http_request"):
        return True
    try:
        temp = provider.model_copy(deep=True)
        temp.api_key = token
        temp.api_keys = []
        temp.active_account = ""
        route = ResolvedRoute(
            provider_name=site.name,
            provider=temp,
            model_id=model_id,
            model_config=temp.models[model_id],
            upstream_config=load_config().upstream,
        )
        method, url, headers, body = site.build_http_request(route, "", [("user", "ping")], None, "ping")
        from app.config import proxy_setting

        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, proxy=proxy_setting()
        ) as client:
            async with client.stream(method, url, headers=headers, json=body) as resp:
                if resp.status_code in (401, 403):
                    logger.info("%s rejected a captured token (HTTP %d)", site.name, resp.status_code)
                    return False
                return True
    except Exception as e:
        logger.warning("%s token verification skipped: %s", site.name, e)
        return True


async def capture_token(provider_name: str, timeout: float = 300.0) -> dict[str, Any]:
    """Open a login window for an HTTP site and save the token it sends.

    This is the "log in and let the proxy keep it" path: no devtools, no copying.
    The token is written to the provider block (``api_key``, or appended to
    ``api_keys`` when several tokens are configured) and hot-reloads.
    """
    from app.providers.web.token_capture import capture_bearer_token

    provider = load_config().providers.get(provider_name)
    site = site_for_provider(provider_name)
    if site is None:
        raise WebProviderError(f"{provider_name!r} is not a web provider (no provider_type in config.json)")
    if site.transport != "http":
        raise WebProviderError(
            f"{site.label or site.name} keeps its login in a browser profile, so it has no token to copy. "
            "Use its login window instead."
        )

    runtime = runtime_for(site.name)
    # Always capture into the site's default profile: the "account" of an HTTP
    # provider is its token, and a token must never become a directory name.
    account = ""
    # A browser runtime gives us the profile dir and env settings without launching anything.
    browser = BrowserRuntime(site)
    async def _verify(candidate: str) -> bool:
        return await _verify_captured_token(site, provider, candidate)

    token = await capture_bearer_token(
        site, browser, account, timeout, verify=_verify if provider is not None else None
    )
    if not token:
        raise WebProviderError(
            "没有拿到可用的 token。常见原因：还没登录（窗口里先完成扫码/手机号登录），"
            "或者登录后没有在页面里发过消息（页面发消息时才会带上 Authorization 头）。"
            "保持窗口开着，登录并发一条消息后会自动保存。"
        )

    block = read_provider_block(provider_name)
    keys = [str(k) for k in (block.get("api_keys") or [])]
    if keys:
        if token not in keys:
            keys.append(token)
        block["api_keys"] = keys
        block["active_account"] = token
    else:
        block["api_key"] = token
    write_provider_block(provider_name, block)
    logger.info("%s: token saved to config.json [provider=%s]", site.name, provider_name)
    return {"ok": True, "provider": provider_name, "token": runtime.mask(token)}


async def open_login_window(provider_name: str, account: Optional[str] = None) -> dict[str, Any]:
    """Open one account's browser window so the user can log in (browser sites)."""
    provider = load_config().providers.get(provider_name)
    site = site_for_provider(provider_name)
    if site is None:
        raise WebProviderError(f"{provider_name!r} is not a web provider (no provider_type in config.json)")
    runtime = runtime_for(site.name)
    if account is None:
        account = runtime.account_for(provider)
    return await runtime.open_login_window(account, None)


async def close(provider_name: Optional[str] = None, account: Optional[str] = None,
                timeout: Optional[float] = None) -> None:
    """Close one provider's browser, or every web provider's (proxy shutdown)."""
    if provider_name is None:
        names = [site.name for site in SITE_LIST]
    else:
        site = site_for_provider(provider_name)
        names = [site.name if site is not None else provider_name]
    # Closing an HTTP provider is a no-op but keeps the call site uniform.
    for name in names:
        runtime = RUNTIMES.get(name)
        if runtime is None:
            continue
        try:
            await runtime.close(account, timeout)
        except Exception as e:  # pragma: no cover - shutdown must never raise
            logger.warning("%s close failed: %s", name, e)


def active_account(provider_name: str = "deepseek-web") -> str:
    """Active account id of one provider (for the admin endpoints)."""
    provider = load_config().providers.get(provider_name)
    site = site_for_provider(provider_name)
    if site is None:
        return ""
    return runtime_for(site.name).account_for(provider)


__all__ = [
    "SITES",
    "SITE_LIST",
    "RUNTIMES",
    "BrowserRuntime",
    "HttpRuntime",
    "active_account",
    "capture_token",
    "close",
    "configured_providers",
    "ensure_provider",
    "get_status",
    "is_web_provider",
    "looks_like_web_provider",
    "open_login_window",
    "runtime_for",
    "site_for",
    "site_for_provider",
    "site_templates",
]
