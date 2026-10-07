from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float
    last_updated: float


class TokenBucketRateLimiter:
    """In-memory thread-safe token-bucket rate limiter keyed by client IP."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int | None = None,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self._time_func = time_func
        self.requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", str(requests_per_minute)))
        burst_val = burst if burst is not None else int(os.getenv("RATE_LIMIT_BURST", str(self.requests_per_minute)))
        self.burst = burst_val
        self.refill_rate = self.requests_per_minute / 60.0
        self.capacity = float(self.burst)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def check_rate_limit(self, key: str, tokens: float = 1.0) -> tuple[bool, int]:
        """Check if request is allowed for the given key.

        Returns:
            (allowed: bool, retry_after: int)
        """
        now = self._time_func()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate,
                    tokens=self.capacity,
                    last_updated=now,
                )
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.last_updated)
                bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate)
                bucket.last_updated = now

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True, 0
            else:
                needed = tokens - bucket.tokens
                retry_after = math.ceil(needed / bucket.refill_rate) if bucket.refill_rate > 0 else 60
                return False, max(1, retry_after)

    def reset(self) -> None:
        """Clear all buckets (useful for test isolation)."""
        with self._lock:
            self._buckets.clear()
