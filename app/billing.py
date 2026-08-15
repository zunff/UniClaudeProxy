"""Billing / usage statistics backed by SQLite.

Records per-request token usage (and cost for routes that have a price table)
into a local SQLite database and exposes aggregated views via GET /stats and
GET /api/stats. Supports querying stats by date or date range.

Scope: token usage is recorded for every provider. Cost is only computed for
routes listed in config `billing.prices` (keyed as "provider/model_id");
unlisted routes get cost=None.

Design notes:
- Config is read lazily via load_config() so hot-reload of config.json is
  picked up without restarting.
- SQLite runs in WAL mode for read/write concurrency. A single shared
  connection (check_same_thread=False) is guarded by a threading.Lock for
  writes; reads run concurrently.
- A daemon thread runs the daily auto-cleanup at billing.cleanup_hour
  (Beijing time), deleting records older than billing.retention_days.
- On first init, a legacy logs/billing.jsonl (if present) is migrated into
  the database and renamed to `<log_file>.migrated` so it won't be re-imported.
- Peak/off-peak pricing is auto-selected by Beijing time (UTC+8).
- Daily aggregates are keyed by Beijing-date (YYYY-MM-DD) for logical grouping.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("uniclaudeproxy.billing")

_BEIJING_TZ = timezone(timedelta(hours=8))
_RECENT_MAX = 50


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


# ---------------------------------------------------------------------------
# Database bootstrap and connection
# ---------------------------------------------------------------------------

_lock = threading.Lock()  # guards writes and DDL
_conn: Optional[sqlite3.Connection] = None
_db_path: Optional[Path] = None
_cleanup_thread: Optional[threading.Thread] = None
_cleanup_stop = threading.Event()


def _resolve_db_path(db_file: str) -> Path:
    p = Path(db_file)
    if not p.is_absolute():
        # Resolve relative to project root (parent of app/ package dir)
        try:
            from app.config import config_path
            p = Path(config_path()).parent / p
        except Exception:
            p = Path.cwd() / p
    return p


_SCHEMA = """
CREATE TABLE IF NOT EXISTS billing_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    date TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    anthropic_model TEXT NOT NULL DEFAULT '',
    is_stream INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL,
    currency TEXT,
    latency_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_billing_date ON billing_records(date);
CREATE INDEX IF NOT EXISTS idx_billing_model ON billing_records(provider, model);
"""


def _init_db() -> sqlite3.Connection:
    """Open (or reuse) the shared SQLite connection, applying schema + pragmas."""
    global _conn, _db_path
    if _conn is not None:
        return _conn
    cfg = _billing_config()
    db_file = cfg.db_file if cfg and cfg.db_file else "logs/billing.db"
    p = _resolve_db_path(db_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL: readers don't block writers, durable enough for billing log.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.executescript(_SCHEMA)
    _conn = conn
    _db_path = p
    _migrate_legacy_jsonl(conn)
    _start_cleanup_thread()
    return conn


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


# ---------------------------------------------------------------------------
# Legacy JSONL migration (one-shot)
# ---------------------------------------------------------------------------

def _migrate_legacy_jsonl(conn: sqlite3.Connection) -> None:
    """Import an existing logs/billing.jsonl into the DB, then rename it."""
    cfg = _billing_config()
    if not cfg or not cfg.log_file:
        return
    p = Path(cfg.log_file)
    if not p.is_absolute():
        try:
            from app.config import config_path
            p = Path(config_path()).parent / p
        except Exception:
            p = Path.cwd() / p
    if not p.exists():
        return
    imported = 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append((
                    r.get("ts", ""),
                    r.get("date", ""),
                    r.get("provider", "") or "",
                    r.get("model", "") or "",
                    r.get("anthropic_model", "") or "",
                    1 if r.get("is_stream") else 0,
                    1 if r.get("success") else 0,
                    int(r.get("input_tokens", 0) or 0),
                    int(r.get("output_tokens", 0) or 0),
                    int(r.get("cache_read_tokens", 0) or 0),
                    int(r.get("cache_creation_tokens", 0) or 0),
                    int(r.get("cache_miss_tokens", 0) or 0),
                    float(r["cost"]) if isinstance(r.get("cost"), (int, float)) else None,
                    r.get("currency"),
                    float(r["latency_ms"]) if r.get("latency_ms") is not None else None,
                ))
                if len(rows) >= 500:
                    _bulk_insert(conn, rows)
                    imported += len(rows)
                    rows = []
            if rows:
                _bulk_insert(conn, rows)
                imported += len(rows)
        # Rename to avoid re-import on next start.
        migrated = p.with_suffix(p.suffix + ".migrated")
        try:
            p.rename(migrated)
        except OSError as e:
            logger.warning("could not rename legacy jsonl after migration: %s", e)
        logger.info("billing: migrated %d records from %s -> SQLite", imported, p.name)
    except Exception as e:
        logger.warning("billing: jsonl migration failed: %s", e)


def _bulk_insert(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.executemany(
        """INSERT INTO billing_records
           (ts, date, provider, model, anthropic_model, is_stream, success,
            input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
            cache_miss_tokens, cost, currency, latency_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

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


_DEFAULT_FX_TO_CNY = {"USD": 7.2}


def _to_cny(
    cost: float,
    currency: Optional[str],
    fx_to_cny: Optional[dict[str, float]] = None,
) -> tuple[float, str]:
    """Convert a native-currency cost into CNY using fx_to_cny rates."""
    code = (currency or "CNY").upper()
    if code == "CNY":
        return cost, "CNY"
    rates = {**_DEFAULT_FX_TO_CNY, **(fx_to_cny or {})}
    rate = rates.get(code)
    if rate is None:
        logger.warning("no fx_to_cny rate for %s; storing native currency", code)
        return cost, code
    return cost * float(rate), "CNY"


def _compute_cost(
    prices: dict[str, Any],
    key: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    bindings: Optional[dict[str, str]] = None,
    fx_to_cny: Optional[dict[str, float]] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Compute cost (per 1M tokens) for a route, respecting peak/off-peak.

    Unit prices stay in the table's native currency (e.g. official USD).
    The returned cost is converted to CNY when an FX rate is available.
    """
    if not prices:
        return None, None
    price_name = None
    if bindings:
        price_name = bindings.get(key)
    if not price_name:
        price_name = key
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
    cost, currency = _to_cny(cost, currency, fx_to_cny)
    return round(cost, 6), currency


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

_INSERT_SQL = """INSERT INTO billing_records
    (ts, date, provider, model, anthropic_model, is_stream, success,
     input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
     cache_miss_tokens, cost, currency, latency_ms)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


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
        fx_to_cny = getattr(cfg, "fx_to_cny", None) if cfg else None
        cost, currency = _compute_cost(
            prices, key, input_tokens, output_tokens, cache_read, bindings, fx_to_cny
        )

        now_bj = datetime.now(_BEIJING_TZ)
        row = (
            now_bj.isoformat(timespec="seconds"),
            now_bj.strftime("%Y-%m-%d"),
            provider_name,
            model_id,
            anthropic_model,
            1 if is_stream else 0,
            1 if success else 0,
            input_tokens,
            output_tokens,
            cache_read,
            cache_creation,
            cache_miss,
            cost,
            currency,
            round(latency_ms, 1) if latency_ms is not None else None,
        )
        conn = _init_db()
        with _lock:
            conn.execute(_INSERT_SQL, row)
    except Exception as e:
        logger.warning("billing record failed: %s", e)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_rows(rows: sqlite3.Cursor | list[sqlite3.Row]) -> dict[str, Any]:
    """Sum SQL aggregate rows (one per group) into a totals + by_model view."""
    totals = {
        "requests": 0, "success": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_miss_tokens": 0, "cost": 0.0, "hit_requests": 0,
    }
    by_model: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = f"{r['provider']}/{r['model']}".strip("/") if r["provider"] else r["model"]
        req = int(r["requests"])
        succ = int(r["success"])
        it = int(r["input_tokens"])
        ot = int(r["output_tokens"])
        cr = int(r["cache_read_tokens"])
        cm = int(r["cache_miss_tokens"])
        cost = float(r["cost"]) if r["cost"] is not None else 0.0
        hits = int(r["hit_requests"])
        totals["requests"] += req
        totals["success"] += succ
        totals["input_tokens"] += it
        totals["output_tokens"] += ot
        totals["cache_read_tokens"] += cr
        totals["cache_miss_tokens"] += cm
        totals["cost"] = round(totals["cost"] + cost, 6)
        totals["hit_requests"] += hits
        bk = by_model.setdefault(key, _empty_model_stats())
        bk["requests"] += req
        bk["success"] += succ
        bk["input_tokens"] += it
        bk["output_tokens"] += ot
        bk["cache_read_tokens"] += cr
        bk["cache_miss_tokens"] += cm
        bk["cost"] = round(bk["cost"] + cost, 6)
        bk["hit_requests"] += hits
    return {"totals": totals, "by_model": by_model}


def _build_view(stats: dict[str, Any]) -> dict[str, Any]:
    """Wrap a stats bucket with cache rate breakdown."""
    requests = stats["totals"]["requests"]
    hit_requests = stats["totals"]["hit_requests"]
    input_tokens = stats["totals"]["input_tokens"]
    cache_read = stats["totals"]["cache_read_tokens"]
    return {
        "totals": dict(stats["totals"]),
        "cache": {
            "hit_requests": hit_requests,
            "hit_rate": round(hit_requests / requests, 4) if requests else 0.0,
            "cached_token_ratio": round(cache_read / input_tokens, 4) if input_tokens else 0.0,
        },
        "by_model": {k: dict(v) for k, v in stats["by_model"].items()},
    }


_AGG_SQL = """SELECT
    provider, model,
    COUNT(*) AS requests,
    SUM(success) AS success,
    SUM(input_tokens) AS input_tokens,
    SUM(output_tokens) AS output_tokens,
    SUM(cache_read_tokens) AS cache_read_tokens,
    SUM(cache_miss_tokens) AS cache_miss_tokens,
    COALESCE(SUM(cost), 0) AS cost,
    SUM(CASE WHEN cache_read_tokens > 0 THEN 1 ELSE 0 END) AS hit_requests
FROM billing_records
WHERE date IN (%s)
GROUP BY provider, model"""


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------

def get_stats(date: Optional[str] = None) -> dict[str, Any]:
    """Return billing stats snapshot.

    Args:
        date: Optional[str] - Beijing date in ISO format (YYYY-MM-DD). When
            provided, returns stats for that day; otherwise returns today's
            stats and a list of available daily keys.
    """
    conn = _init_db()
    today_key = _get_today_key()
    if date:
        rows = conn.execute(_AGG_SQL % _placeholders(1), (date,)).fetchall()
        stats = _aggregate_rows(rows)
        view = _build_view(stats)
        view["date"] = date
        recent = conn.execute(
            "SELECT * FROM billing_records WHERE date = ? ORDER BY id DESC LIMIT ?",
            (date, _RECENT_MAX),
        ).fetchall()
        view["recent"] = [_row_to_dict(r) for r in recent]
        return view

    rows = conn.execute(_AGG_SQL % _placeholders(1), (today_key,)).fetchall()
    stats = _aggregate_rows(rows)
    view = _build_view(stats)
    view["date"] = today_key
    recent = conn.execute(
        "SELECT * FROM billing_records ORDER BY id DESC LIMIT ?", (_RECENT_MAX,)
    ).fetchall()
    view["recent"] = [_row_to_dict(r) for r in recent]
    view["available_dates"] = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM billing_records ORDER BY date DESC LIMIT 400"
        ).fetchall()
    ]
    return view


def get_stats_range(date_keys: list[str], recent_limit: int = 50) -> dict[str, Any]:
    """Return aggregated stats across multiple days, with per-day breakdown.

    Args:
        date_keys: list of YYYY-MM-DD strings (Beijing dates).
        recent_limit: max number of recent records to include.

    Returns:
        dict with:
          - total: {totals, cache, by_model} aggregated across all given days
          - per_day: {date -> {totals, cache, by_model, source="sqlite"}}
          - recent: latest N records across the queried date range
    """
    conn = _init_db()
    if not date_keys:
        empty_totals = {
            "requests": 0, "success": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_miss_tokens": 0, "cost": 0.0, "hit_requests": 0,
        }
        empty = _build_view({"totals": empty_totals, "by_model": {}})
        return {"total": empty, "per_day": {}, "recent": []}

    # Union across all requested days.
    ph = _placeholders(len(date_keys))
    union_rows = conn.execute(_AGG_SQL % ph, date_keys).fetchall()
    union_stats = _aggregate_rows(union_rows)
    total_view = _build_view(union_stats)

    per_day: dict[str, Any] = {}
    for dk in date_keys:
        day_rows = conn.execute(_AGG_SQL % _placeholders(1), (dk,)).fetchall()
        day_stats = _aggregate_rows(day_rows)
        day_view = _build_view(day_stats)
        day_view["date"] = dk
        day_view["source"] = "sqlite"
        per_day[dk] = day_view

    # Recent records across the queried date range.
    recent_rows = conn.execute(
        f"SELECT * FROM billing_records WHERE date IN ({ph}) ORDER BY id DESC LIMIT ?",
        date_keys + [recent_limit],
    ).fetchall()
    recent = [_row_to_dict(r) for r in recent_rows]

    return {"total": total_view, "per_day": per_day, "recent": recent}


def get_recent(limit: int = 50, date: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the most recent billing records.

    Args:
        limit: max number of records to return (default 50, max 500).
        date: optional Beijing date filter (YYYY-MM-DD).
    """
    conn = _init_db()
    limit = min(max(limit, 1), 500)
    if date:
        rows = conn.execute(
            "SELECT * FROM billing_records WHERE date = ? ORDER BY id DESC LIMIT ?",
            (date, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM billing_records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def reset_stats(date: Optional[str] = None) -> None:
    """Delete records. Pass a date to clear only that day."""
    conn = _init_db()
    with _lock:
        if date:
            conn.execute("DELETE FROM billing_records WHERE date = ?", (date,))
        else:
            conn.execute("DELETE FROM billing_records")


def _row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["is_stream"] = bool(d.get("is_stream"))
    d["success"] = bool(d.get("success"))
    return d


def _get_today_key() -> str:
    """Return today's Beijing-date key (YYYY-MM-DD)."""
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Auto-cleanup scheduler
# ---------------------------------------------------------------------------

def _seconds_until_next_run(cleanup_hour: int) -> float:
    """Seconds from now (Beijing time) until the next scheduled cleanup run."""
    now = datetime.now(_BEIJING_TZ)
    target = now.replace(hour=cleanup_hour % 24, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _run_cleanup() -> None:
    """Delete records older than retention_days, then vacuum-compact."""
    cfg = _billing_config()
    if not cfg or not cfg.retention_days or cfg.retention_days <= 0:
        return
    cutoff = (datetime.now(_BEIJING_TZ) - timedelta(days=cfg.retention_days)).strftime("%Y-%m-%d")
    conn = _init_db()
    try:
        with _lock:
            cur = conn.execute("DELETE FROM billing_records WHERE date < ?", (cutoff,))
            deleted = cur.rowcount
        # Reclaim disk space (WAL allows this without blocking readers long).
        with _lock:
            conn.execute("VACUUM;")
        logger.info("billing cleanup: deleted %d records older than %s (retention=%dd)",
                    deleted, cutoff, cfg.retention_days)
    except Exception as e:
        logger.warning("billing cleanup failed: %s", e)


def _cleanup_loop() -> None:
    """Daemon thread: sleep until cleanup_hour, run, repeat every 24h."""
    while not _cleanup_stop.is_set():
        cfg = _billing_config()
        hour = cfg.cleanup_hour if cfg else 3
        delay = _seconds_until_next_run(hour)
        # Wake up at most every hour to re-check config / stop signal.
        while delay > 0:
            step = min(delay, 3600.0)
            if _cleanup_stop.wait(step):
                return
            delay -= step
        _run_cleanup()


def _start_cleanup_thread() -> None:
    global _cleanup_thread
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return
    cfg = _billing_config()
    if cfg and cfg.retention_days and cfg.retention_days > 0:
        _cleanup_stop.clear()
        _cleanup_thread = threading.Thread(target=_cleanup_loop, name="billing-cleanup", daemon=True)
        _cleanup_thread.start()
        logger.info("billing cleanup scheduler started (retention=%dd, hour=%d)",
                    cfg.retention_days, cfg.cleanup_hour)


def shutdown() -> None:
    """Signal the cleanup thread to stop. Safe to call multiple times."""
    _cleanup_stop.set()
