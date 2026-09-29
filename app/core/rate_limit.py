"""
A fixed-window rate limiter for the API.

Why it exists
-------------
``API_KEY`` is a shared secret, not a per-user identity. There is no account to
lock out, no per-user quota to enforce, and nothing that stops one client from
spending the configured provider's entire token budget in a minute. Every route
that calls an LLM is metered by the *provider*, so the failure mode is a burst
that exhausts a token bucket and then returns 429s to the legitimate operator --
which is exactly what happened to this application's own test runs against a
free tier.

Design, and what it deliberately is not
--------------------------------------
A fixed window per client key, held in memory:

* **No new dependency.** A token bucket is about thirty lines; a library would be
  more code than the thing it replaced.
* **In-memory, per process.** Honest about its scope: with several workers each
  process counts separately, so the effective limit is ``limit x workers``. A
  shared store would fix that and would also make the limiter a new thing that
  can fail, so the limit is documented as per-process rather than presented as
  global.
* **Bounded memory.** At most ``max_clients`` keys are tracked, and the least
  recently used are evicted. Without that, an unauthenticated flood of spoofed
  client keys would grow the table without limit -- the limiter would become the
  denial-of-service vector it exists to prevent.
* **The expensive routes only.** The LLM-backed endpoints are the ones with a
  per-request cost; rate-limiting ``/health`` would break a container healthcheck
  and rate-limiting static reads would achieve nothing.

Off by default, because a limiter that silently starts rejecting a single-user
local deployment is worse than no limiter. ``RATE_LIMIT_ENABLED=true`` turns it on.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class RateLimitSettings:
    """The limit itself, and whether it applies at all."""

    enabled: bool = False
    requests: int = 60
    window_seconds: float = 60.0
    #: Paths exempt from limiting. The container healthcheck runs every ten
    #: seconds, so a limit low enough to be useful would trip it.
    exempt_paths: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"/health", "/api/v1/resume/health", "/docs", "/redoc", "/openapi.json"}
        )
    )

    def as_headers(self) -> dict[str, int]:
        return {
            "RATE_LIMIT_REQUESTS": self.requests,
            "RATE_LIMIT_WINDOW_SECONDS": int(self.window_seconds),
        }


@dataclass
class _Window:
    started: float
    count: int = 0


class RateLimiter:
    """
    Fixed-window counter, keyed by client.

    Fixed rather than sliding because a sliding window needs a timestamp per
    request and this needs one timestamp per key. The trade is a boundary burst
    of up to twice the limit across a window edge, which is acceptable for
    protecting a token budget and not acceptable as a security control -- and it
    is not claimed to be one.
    """

    def __init__(self, settings: RateLimitSettings, max_clients: int = 4096) -> None:
        self.settings = settings
        self.max_clients = max_clients
        self._windows: OrderedDict[str, _Window] = OrderedDict()

    def check(self, key: str, now: float | None = None) -> tuple[bool, dict[str, str]]:
        """
        Record a request against ``key``.

        Returns ``(allowed, headers)``. The headers are the standard
        ``X-RateLimit-*`` set plus ``Retry-After`` when the request was refused,
        so a client can back off correctly rather than guessing.
        """
        moment = time.monotonic() if now is None else now
        limit = max(1, self.settings.requests)
        window = max(0.001, self.settings.window_seconds)

        window_state = self._windows.get(key)

        if window_state is None or (moment - window_state.started) >= window:
            window_state = _Window(started=moment, count=1)
            self._windows[key] = window_state
            self._evict_if_needed()
        else:
            window_state.count += 1
            # Re-insert so the ordering reflects recency, which is what the
            # eviction below relies on.
            self._windows.move_to_end(key)

        remaining = max(0, limit - window_state.count)
        reset_in = max(0.0, window - (moment - window_state.started))

        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(math.ceil(reset_in))),
        }

        allowed = window_state.count <= limit

        if not allowed:
            headers["Retry-After"] = str(max(1, int(math.ceil(reset_in))))

        return allowed, headers

    def _evict_if_needed(self) -> None:
        """
        Keep the table bounded.

        Without this, an attacker who never presents a valid key can still send
        requests with a fresh spoofed client key and grow the table until the
        process runs out of memory. The oldest key goes first, which is the one
        least likely to belong to an active client.
        """
        while len(self._windows) > self.max_clients:
            self._windows.popitem(last=False)

    def reset(self) -> None:
        """Forget every window. For tests, and for an operator who has just
        changed the limit and does not want to wait out the old window."""
        self._windows.clear()

    def client_key(self, request, api_key: str | None) -> str:
        """
        Identify the caller.

        The presented API key is preferred: it is the one thing a legitimate
        client always has, so every legitimate client gets its own budget even
        behind a shared NAT. The remote address is the fallback, which means
        anonymous traffic from one address shares a budget -- the correct default
        when there is no identity to separate on.
        """
        if api_key:
            # A short digest, so the limiter's table never holds a credential.
            import hashlib

            return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]

        client = request.client
        host = getattr(client, "host", None) if client else None
        return f"addr:{host or 'unknown'}"
