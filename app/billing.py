"""Billing / usage statistics with daily aggregation.

Records per-request token usage (and cost for routes that have a price table)
to an append-only JSONL log and maintains in-memory aggregates exposed via
GET /stats. Supports querying stats by date.

Scope: token usage is recorded for every provider. Cost is only computed for
routes listed in config `billing.prices` (keyed as "provider/model_id");
unlisted routes get cost=None.

Design notes:
- Config is read lazily via load_config() so hot-reload of config.json is
  picked up without restarting.
- File writes are synchronous append-a-line; cheap enough for a per-request
  log. A threading.Lock guards the in-memory aggregates.
- Peak/off-peak pricing is auto-selected by Beijing time (UTC+8).
- Daily aggregates are keyed by Beijing-date (YYYY-MM-DD) for logical grouping.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("uniclaudeproxy.billing")

_BEIJING_TZ = timezone(timedelta(hours=8))
_RECENT_MAX = 50


def _empty_stats() -> dict[str, Any]:
    return {
        "requests": 0,
        "success": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_miss_tokens": 0,
        "cost": 0.0,
        "hit_requests": 0,
        "by_model": {},
    }


def _empty_model_stats() -> dict[str, Any]:
    return {
        "requests": 0,
        "success": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_miss_tokens": 0,
        "cost": 0.0,
        "hit_requests": 0,
    }


_lock = threading.Lock()
# _daily_stats maps date_str (YYYY-MM-DD in Beijing time) -> stats dict
_daily_stats: dict[str, dict[str, Any]] = {}
_recent: list[dict[str, Any]] = []


def _get_today_key() -> str:
    """Return today's Beijing-date key (YYYY-MM-DD)."""
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")


def _get_or_create_day_stats(date_key: str) -> dict[str, Any]:
    """Get stats dict for a given date, creating if needed."""
    if date_key not in _daily_stats:
        _daily_stats[date_key] = _empty_stats()
    return _daily_stats[date_key]


def _billing_config():
    """Return the current BillingConfig (lazy read so hot-reload works)."""
    try:
        from app.config import load_config
        return load_config().billing
    except Exception:
        return None


def _is_enabled() -> bool:
    cfg = _billing_config()
    return bool(cfg and cfg.enabled)


def _is_peak(peak_hours: list[list[int]], now_bj: datetime) -> bool:
    if not peak_hours:
        return False
    h = now_bj.hour
    for interval in peak_hours:
        if len(interval) != 2:
            continue
        start, end = interval
        if start <= h < end:
            return True
    return False


def _compute_cost(
    prices: dict[str, Any],
    key: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    bindings: Optional[dict[str, str]] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Compute cost (per 1M tokens) for a route, respecting peak/off-peak.

    Lookup order:
    1. If bindings map route key -> price name, use prices[price_name]
    2. Fallback: direct lookup prices[key] (backward compat)
    """
    if not prices:
        return None, None
    # Resolve price table name via bindings, fallback to direct key
    price_name = None
    if bindings:
        price_name = bindings.get(key)
    if not price_name:
        price_name = key  # backward compat: route key IS the price key
    entry = prices.get(price_name)
    if not entry:
        return None, None
    currency = entry.get("currency")

    peak = entry.get("peak")
    offpeak = entry.get("offpeak")
    if peak and offpeak:
        peak_hours = entry.get("peak_hours") or [[9, 12], [14, 18]]
        tier = peak if _is_peak(peak_hours, datetime.now(_BEIJING_TZ)) else offpeak
    elif peak:
        tier = peak
    elif offpeak:
        tier = offpeak
    else:
        tier = entry

    p_in = float(tier.get("input", 0) or 0)
    p_cached = float(tier.get("input_cached", p_in) or 0)
    p_out = float(tier.get("output", 0) or 0)

    cache_miss = max(input_tokens - cache_read, 0)
    cost = (
        (cache_read / 1_000_000.0) * p_cached
        + (cache_miss / 1_000_000.0) * p_in
        + (output_tokens / 1_000_000.0) * p_out
    )
    return round(cost, 6), currency


def _aggregate(bucket: dict[str, Any], rec: dict[str, Any]) -> None:
    bucket["requests"] += 1
    if rec["success"]:
        bucket["success"] += 1
    bucket["input_tokens"] += rec["input_tokens"]
    bucket["output_tokens"] += rec["output_tokens"]
    bucket["cache_read_tokens"] += rec["cache_read_tokens"]
    bucket["cache_miss_tokens"] += rec["cache_miss_tokens"]
    if rec["cost"] is not None:
        bucket["cost"] = round(bucket["cost"] + rec["cost"], 6)
    if rec["cache_read_tokens"] > 0:
        bucket["hit_requests"] += 1


def record(
    route: Any,
    anthropic_model: str,
    usage: Optional[dict[str, Any]],
    is_stream: bool,
    success: bool,
    latency_ms: Optional[float] = None,
) -> None:
    """Record one request's usage. No-op when billing is disabled."""
    if not _is_enabled():
        return
    try:
        cfg = _billing_config()
        log_file = (cfg.log_file if cfg and cfg.log_file else "logs/billing.jsonl")
        prices = cfg.prices if cfg else {}
        bindings = cfg.price_bindings if cfg else None

        u = usage or {}
        input_tokens = int(u.get("input_tokens", 0) or 0)
        output_tokens = int(u.get("output_tokens", 0) or 0)
        cache_read = int(u.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(u.get("cache_creation_input_tokens", 0) or 0)
        cache_miss = max(input_tokens - cache_read, 0)

        provider_name = getattr(route, "provider_name", "") or ""
        model_id = getattr(route, "model_id", "") or ""
        key = f"{provider_name}/{model_id}" if provider_name else model_id
        cost, currency = _compute_cost(prices, key, input_tokens, output_tokens, cache_read, bindings)

        now_bj = datetime.now(_BEIJING_TZ)
        rec = {
            "ts": now_bj.isoformat(timespec="seconds"),
            "date": now_bj.strftime("%Y-%m-%d"),
            "provider": provider_name,
            "model": model_id,
            "anthropic_model": anthropic_model,
            "is_stream": is_stream,
            "success": success,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "cache_miss_tokens": cache_miss,
            "cost": cost,
            "currency": currency,
            "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        }

        try:
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("billing jsonl write failed: %s", e)

        with _lock:
            date_key = rec["date"]
            day_stats = _get_or_create_day_stats(date_key)
            _aggregate(day_stats, rec)
            bk = day_stats["by_model"].setdefault(key, _empty_model_stats())
            _aggregate(bk, rec)
            _recent.append(rec)
            if len(_recent) > _RECENT_MAX:
                del _recent[: len(_recent) - _RECENT_MAX]
    except Exception as e:
        logger.warning("billing record failed: %s", e)


def _build_view(stats: dict[str, Any]) -> dict[str, Any]:
    """Wrap a stats bucket with cache rate breakdown."""
    requests = stats.get("requests", 0)
    hit_requests = stats.get("hit_requests", 0)
    input_tokens = stats.get("input_tokens", 0)
    cache_read = stats.get("cache_read_tokens", 0)
    totals = {k: v for k, v in stats.items() if k != "by_model"}
    by_model = {k: dict(v) for k, v in stats.get("by_model", {}).items()}
    return {
        "totals": totals,
        "cache": {
            "hit_requests": hit_requests,
            "hit_rate": round(hit_requests / requests, 4) if requests else 0.0,
            "cached_token_ratio": round(cache_read / input_tokens, 4) if input_tokens else 0.0,
        },
        "by_model": by_model,
    }


def get_stats(date: Optional[str] = None) -> dict[str, Any]:
    """Return billing stats snapshot.

    Args:
        date: Optional[str] - Beijing date in ISO format (YYYY-MM-DD). When
            provided, returns stats for that day; otherwise returns today's
            stats and a list of available daily keys.
    """
    with _lock:
        if date:
            stats = _daily_stats.get(date, _empty_stats())
            view = _build_view(stats)
            view["date"] = date
            view["recent"] = [r for r in _recent if r.get("date") == date][-_RECENT_MAX:]
            return view

        today_key = _get_today_key()
        today_stats = _daily_stats.get(today_key, _empty_stats())
        today_view = _build_view(today_stats)
        today_view["date"] = today_key
        today_view["recent"] = _recent  # recent across all dates
        today_view["available_dates"] = sorted(_daily_stats.keys())
        return today_view


def reset_stats(date: Optional[str] = None) -> None:
    """Clear in-memory aggregates. Pass a date to clear only that day."""
    global _recent
    with _lock:
        if date:
            _daily_stats.pop(date, None)
            _recent = [r for r in _recent if r.get("date") != date]
        else:
            _daily_stats.clear()
            _recent = []
