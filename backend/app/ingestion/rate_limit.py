from __future__ import annotations

from collections.abc import Callable

class RateLimiter:
    """Min-interval limiter: at most `rate_limit_rps` starts per second, no bursts."""

    def __init__(
        self,
        rate_limit_rps: float,
        *,
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        if rate_limit_rps <= 0:
            self._interval = 0.0
        else:
            self._interval = 1.0 / rate_limit_rps
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_allowed = 0.0
        self._started = False

    def wait(self) -> None:
        now = self._monotonic()
        if not self._started:
            self._started = True
            self._next_allowed = now + self._interval
            return
        delay = self._next_allowed - now
        if delay > 0:
            self._sleep(delay)
            now = self._monotonic()
            if now < self._next_allowed:
                now = self._next_allowed
        self._next_allowed = now + self._interval
