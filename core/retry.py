import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    retryable: tuple[type[Exception], ...] = (Exception,),
    label: str = "operation",
) -> Callable[[F], F]:
    """Decorator factory: exponential backoff retry with configurable exceptions."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "[%s] Failed after %d attempts: %s",
                            label,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "[%s] Attempt %d/%d failed (%s). Retrying in %.1fs.",
                        label,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator
