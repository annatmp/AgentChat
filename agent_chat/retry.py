"""
Retry with exponential backoff, replacing the blanket `time.sleep(3)`.

Two requirements from TODO/EXPERIMENT_DESIGN shape this:

1. *Log that a retry happened.* SDK-internal retries are invisible, so the
   clients are built with `max_retries=0` and retrying happens here, where the
   attempt count can land in the run record.
2. *Don't silently retry until success.* Robustness differences between
   strategies are a legitimate finding, so a call that exhausts its attempts
   raises and gets recorded as an error rather than being papered over.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

import anthropic
import openai

T = TypeVar("T")

# Transport-level failures: retryable regardless of status code.
_RETRYABLE_TYPES = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APIConnectionError,
)
# Anything carrying an HTTP status: retryable only on 429 / 5xx.
_STATUS_TYPES = (anthropic.APIStatusError, openai.APIStatusError)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    if isinstance(exc, _STATUS_TYPES):
        status = getattr(exc, "status_code", 0) or 0
        return status == 429 or status >= 500
    return False


def _retry_after(exc: BaseException) -> float | None:
    """Honour the provider's own backoff hint when it sends one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_allowed: Callable[[], bool] | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> tuple[T, int]:
    """
    Run `fn`, retrying retryable API errors. Returns `(result, retries)`.

    `retry_allowed` is the guard for streaming: a retry restarts the stream, so
    if tokens have already been emitted to the terminal, retrying would print
    the turn twice. Callers pass `lambda: not tokens_seen` and a mid-stream
    failure is raised instead — recorded as an errored call.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt - 1
        except BaseException as exc:
            last_attempt = attempt == max_attempts
            if last_attempt or not is_retryable(exc):
                raise
            if retry_allowed is not None and not retry_allowed():
                raise
            backoff = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            delay = min(_retry_after(exc) or backoff, max_delay)
            if on_retry:
                on_retry(attempt, delay, exc)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
