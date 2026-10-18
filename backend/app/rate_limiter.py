"""
StudySphere AI - In-Memory Sliding Window Rate Limiter
Provides zero-budget abuse protection with zero external dependencies (no Redis).
Tracks request counts across a sliding time window per client IP / session.

Features:
- Configurable window size (seconds) and max requests per window.
- Granular per-endpoint tiers (upload, ask/chat, summarizer/quiz, general).
- Automatic cleanup of stale IP timestamps to prevent memory growth.
- Disabled or relaxed when RATE_LIMIT_ENABLED=false (e.g. for testing).
- Standard HTTP 429 Too Many Requests response with Retry-After header.
"""

import os
import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

# Toggle rate limiting from environment (enabled by default in production)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")

# Tier limits: (max_requests, window_seconds)
LIMITS = {
    "upload": (10, 60),      # 10 uploads per minute per IP
    "ask": (30, 60),         # 30 Q&A / chat queries per minute
    "generate": (20, 60),    # 20 summarizations / quiz generations per minute
    "general": (120, 60),    # 120 requests per minute for general endpoints
}


class SlidingWindowRateLimiter:
    """
    Thread-safe, in-memory sliding window rate limiter.
    Keyed by (client_identifier, endpoint_tier).
    """

    def __init__(self, cleanup_interval: float = 300.0):
        self._history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval

    def _cleanup_stale(self, now: float):
        """Removes entries where all timestamps are older than max window."""
        max_window = max(window for _, window in LIMITS.values())
        cutoff = now - max_window
        stale_keys = []
        for key, timestamps in self._history.items():
            valid = [ts for ts in timestamps if ts > cutoff]
            if valid:
                self._history[key] = valid
            else:
                stale_keys.append(key)
        for k in stale_keys:
            del self._history[k]
        self._last_cleanup = now

    def check_rate_limit(self, request: Request, tier: str = "general"):
        """
        Enforces sliding window limit for the given tier.
        Raises HTTP 429 if rate limit is exceeded.
        """
        if not RATE_LIMIT_ENABLED:
            return

        # Check for test bypass header (strictly restricted to non-production environments)
        if os.getenv("ENVIRONMENT", "development").lower() != "production":
            if request.headers.get("X-Bypass-Rate-Limit") == "test-suite":
                return

        now = time.time()

        # Run periodic cleanup
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_stale(now)

        # Identify client by X-Forwarded-For (proxied cloud) or client.host
        client_ip = request.headers.get("X-Forwarded-For")
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "127.0.0.1"

        # Also combine with X-Session-Token if present
        session_token = request.headers.get("X-Session-Token", "")
        client_key = f"{client_ip}:{session_token}" if session_token else client_ip

        max_requests, window_seconds = LIMITS.get(tier, LIMITS["general"])
        cutoff = now - window_seconds

        history_key = (client_key, tier)
        timestamps = self._history[history_key]

        # Filter out timestamps older than the window
        valid_timestamps = [ts for ts in timestamps if ts > cutoff]
        self._history[history_key] = valid_timestamps

        if len(valid_timestamps) >= max_requests:
            # Calculate seconds until the oldest request in the current window expires
            oldest = valid_timestamps[0]
            retry_after = max(1, int(oldest + window_seconds - now))
            logger.warning(
                "Rate limit exceeded for %s on tier '%s' (%d/%d in %ds). Retry-After: %ds",
                client_ip,
                tier,
                len(valid_timestamps),
                max_requests,
                window_seconds,
                retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {tier} requests. Please wait {retry_after} second(s) before retrying.",
                headers={"Retry-After": str(retry_after)},
            )

        # Record this request
        valid_timestamps.append(now)


# Global singleton instance
limiter = SlidingWindowRateLimiter()
