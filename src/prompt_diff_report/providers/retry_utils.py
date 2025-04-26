"""Reusable tenacity-based retry decorators.

*Restored after accidental deletion.*

Functions
---------
- **`rate_limit_retry`** - 429 レートリミット専用リトライ。
- **`llm_retry_decorator`** - 任意の *predicate* で戻り値を検査してリトライ。
"""

from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
)

from .exceptions import ProviderRateLimitError

F = TypeVar('F', bound=Callable[..., object])


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def rate_limit_retry(*, max_attempts: int = 3, wait_seconds: int = 1) -> Callable[[F], F]:
    """Retry when :class:`ProviderRateLimitError` is raised.

    Parameters
    ----------
    max_attempts : int, default 3
        リトライ回数上限。
    wait_seconds : int, default 1
        各リトライ間の固定待機時間

    """
    return retry(
        retry=retry_if_exception_type(ProviderRateLimitError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_seconds),
        reraise=True,
    )


def llm_retry_decorator(
    predicate: Callable[[object], bool],
    *,
    max_attempts: int = 5,
    wait_seconds: int = 1,
) -> Callable[[F], F]:
    """Return a decorator that retries while *predicate(result)* is **True**."""
    return retry(
        retry=retry_if_result(predicate),
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_seconds),
        reraise=True,
    )
