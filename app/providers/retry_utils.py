import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

import httpx

from app.config import ResolvedRoute

T = TypeVar("T")


class FirstByteTimeoutError(TimeoutError):
    """Raised when upstream fails to return the first byte in time."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry and timeout settings for upstream calls."""

    first_byte_timeout_ms: int
    max_attempts: int
    interval_ms: int
    total_timeout_ms: int


def route_key(route: ResolvedRoute) -> str:
    """Build canonical provider/model route key."""
    return f"{route.provider_name}/{route.model_id}"


def match_disabled_route(route: ResolvedRoute) -> str | None:
    """Return matched disabled route rule if this route should bypass upstream policy."""
    route_key_value = route_key(route)
    provider_wildcard = f"{route.provider_name}/*"
    for item in route.upstream_config.disabled_routes:
        rule = item.strip()
        if not rule:
            continue
        if rule == route_key_value:
            return rule
        if rule == provider_wildcard:
            return rule
    return None


def should_bypass_upstream_policy(route: ResolvedRoute) -> tuple[bool, str | None]:
    """Whether current route should skip first-byte timeout and retry policy."""
    matched = match_disabled_route(route)
    return matched is not None, matched


def policy_from_route(route: ResolvedRoute, *, stream: bool) -> RetryPolicy:
    """Build retry policy from resolved route config."""
    first_byte_timeout_ms = (
        route.upstream_config.stream.first_byte_timeout_ms
        if stream
        else route.upstream_config.non_stream.first_byte_timeout_ms
    )
    retry_cfg = route.upstream_config.retry
    return RetryPolicy(
        first_byte_timeout_ms=first_byte_timeout_ms,
        max_attempts=max(1, retry_cfg.max_attempts),
        interval_ms=max(0, retry_cfg.interval_ms),
        total_timeout_ms=max(1, retry_cfg.total_timeout_ms),
    )


def is_retryable_error(exc: Exception) -> bool:
    """Whether an exception should trigger retry."""
    if isinstance(exc, FirstByteTimeoutError):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else 0
        return status >= 500
    return False


async def run_with_retry(
    operation: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    logger: Any,
    provider_name: str,
    model_id: str,
) -> T:
    """Run non-stream operation with first-byte timeout + fixed-interval retry."""
    start = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if elapsed_ms >= policy.total_timeout_ms:
            break
        try:
            return await operation(policy.first_byte_timeout_ms)
        except Exception as exc:
            last_exc = exc
            retryable = is_retryable_error(exc)
            attempt_exhausted = attempt >= policy.max_attempts
            elapsed_ms = int((time.monotonic() - start) * 1000)
            will_hit_total_timeout = (elapsed_ms + policy.interval_ms) >= policy.total_timeout_ms
            should_retry = retryable and not attempt_exhausted and not will_hit_total_timeout

            if isinstance(exc, FirstByteTimeoutError):
                logger.warning(
                    "First-byte timeout; aborting and retrying "
                    "[provider=%s model=%s attempt=%d/%d first_byte_timeout_ms=%d retry_interval_ms=%d elapsed_ms=%d next_retry_in_ms=%d]",
                    provider_name,
                    model_id,
                    attempt,
                    policy.max_attempts,
                    policy.first_byte_timeout_ms,
                    policy.interval_ms,
                    elapsed_ms,
                    policy.interval_ms if should_retry else 0,
                )

            if not should_retry:
                raise
            await asyncio.sleep(policy.interval_ms / 1000.0)

    if last_exc is not None:
        raise last_exc
    raise TimeoutError("Upstream retry aborted due to total timeout budget")


async def stream_with_retry(
    stream_factory: Callable[[], AsyncIterator[bytes]],
    *,
    policy: RetryPolicy,
    logger: Any,
    provider_name: str,
    model_id: str,
) -> AsyncIterator[bytes]:
    """Run stream operation with retry before first byte only."""
    start = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if elapsed_ms >= policy.total_timeout_ms:
            break

        stream = stream_factory()
        iterator = stream.__aiter__()
        try:
            first_chunk = await asyncio.wait_for(
                iterator.__anext__(), timeout=policy.first_byte_timeout_ms / 1000.0
            )
        except asyncio.TimeoutError as exc:
            await _close_stream_safely(stream)
            last_exc = FirstByteTimeoutError("Upstream did not return first byte in time")
            attempt_exhausted = attempt >= policy.max_attempts
            elapsed_ms = int((time.monotonic() - start) * 1000)
            will_hit_total_timeout = (elapsed_ms + policy.interval_ms) >= policy.total_timeout_ms
            should_retry = not attempt_exhausted and not will_hit_total_timeout
            logger.warning(
                "First-byte timeout; aborting and retrying "
                "[provider=%s model=%s attempt=%d/%d first_byte_timeout_ms=%d retry_interval_ms=%d elapsed_ms=%d next_retry_in_ms=%d]",
                provider_name,
                model_id,
                attempt,
                policy.max_attempts,
                policy.first_byte_timeout_ms,
                policy.interval_ms,
                elapsed_ms,
                policy.interval_ms if should_retry else 0,
            )
            if not should_retry:
                raise last_exc from exc
            await asyncio.sleep(policy.interval_ms / 1000.0)
            continue
        except StopAsyncIteration:
            await _close_stream_safely(stream)
            return
        except Exception as exc:
            await _close_stream_safely(stream)
            last_exc = exc
            retryable = is_retryable_error(exc)
            attempt_exhausted = attempt >= policy.max_attempts
            elapsed_ms = int((time.monotonic() - start) * 1000)
            will_hit_total_timeout = (elapsed_ms + policy.interval_ms) >= policy.total_timeout_ms
            should_retry = retryable and not attempt_exhausted and not will_hit_total_timeout
            if not should_retry:
                raise
            await asyncio.sleep(policy.interval_ms / 1000.0)
            continue

        yield first_chunk
        async for chunk in iterator:
            yield chunk
        return

    if last_exc is not None:
        raise last_exc
    raise TimeoutError("Upstream streaming retry aborted due to total timeout budget")


async def _close_stream_safely(stream: AsyncIterator[bytes]) -> None:
    """Close async generator/stream iterator if supported."""
    close = getattr(stream, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:
            return
