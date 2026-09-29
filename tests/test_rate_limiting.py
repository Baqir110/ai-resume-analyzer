"""
Rate limiting: the limit, the headers, the bounds, and the key handling.

The properties that matter, and why each is asserted:

* **The limit is enforced.** A burst past the limit is refused, and the refusal
  comes back as 429 rather than as a generic error.
* **The table is bounded.** A limiter that can be grown without limit by an
  unauthenticated flood is the denial-of-service vector it was added to prevent,
  so eviction is part of the contract, not an optimisation.
* **The table never holds a credential.** It stores a digest. A table of API
  keys would turn a heap dump or a memory log into a credential leak.
* **The health endpoints are exempt.** A container healthcheck runs every ten
  seconds; a limit low enough to be useful would trip it and the orchestrator
  would kill a healthy container.
* **It is off by default.** A limiter that silently starts rejecting a
  single-user local deployment is worse than no limiter.

The middleware is exercised through the real ASGI app, because a limiter that
works in isolation and is wired up in the wrong order still protects nothing.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import RateLimiter, RateLimitSettings

# ===========================================================================
# The limiter itself
# ===========================================================================


def test_requests_within_the_limit_are_allowed():
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=5, window_seconds=60))

    for _ in range(5):
        allowed, _ = limiter.check("client")
        assert allowed


def test_the_request_past_the_limit_is_refused():
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=5, window_seconds=60))

    for _ in range(5):
        limiter.check("client")

    allowed, headers = limiter.check("client")

    assert allowed is False
    assert headers["X-RateLimit-Remaining"] == "0"
    assert int(headers["Retry-After"]) >= 1, "a refusal must say how long to wait"


def test_the_window_resets():
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=2, window_seconds=60))

    assert limiter.check("client", now=0.0)[0]
    assert limiter.check("client", now=1.0)[0]
    assert limiter.check("client", now=2.0)[0] is False

    # Inside the window, still refused.
    assert limiter.check("client", now=30.0)[0] is False

    # Past it, allowed again.
    assert limiter.check("client", now=61.0)[0] is True


def test_clients_are_counted_separately():
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=2, window_seconds=60))

    for _ in range(2):
        limiter.check("alice")

    assert limiter.check("alice", now=1.0)[0] is False
    assert (
        limiter.check("bob", now=1.0)[0] is True
    ), "one client's burst must not spend another's budget"


def test_the_table_is_bounded():
    """
    The security property. Without a bound, spoofed client keys grow the table
    until the process runs out of memory.
    """
    limiter = RateLimiter(
        RateLimitSettings(enabled=True, requests=10, window_seconds=60), max_clients=64
    )

    for index in range(1000):
        limiter.check(f"spoofed-{index}", now=float(index))

    assert len(limiter._windows) <= 64, f"the table grew to {len(limiter._windows)} entries"


def test_an_evicted_client_is_not_penalised_for_being_evicted():
    """
    Eviction drops the least recently used key. A client that was evicted is
    counted afresh, which is the correct trade: forgetting is better than
    refusing a legitimate client forever.
    """
    limiter = RateLimiter(
        RateLimitSettings(enabled=True, requests=1, window_seconds=600), max_clients=2
    )

    limiter.check("first", now=0.0)
    assert limiter.check("first", now=1.0)[0] is False

    for index in range(10):
        limiter.check(f"other-{index}", now=float(index + 2))

    assert (
        limiter.check("first", now=100.0)[0] is True
    ), "an evicted client should get a fresh window, not a refusal"


def test_headers_report_the_limit_and_the_reset():
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=10, window_seconds=30))

    _, headers = limiter.check("client", now=0.0)

    assert headers["X-RateLimit-Limit"] == "10"
    assert headers["X-RateLimit-Remaining"] == "9"
    assert int(headers["X-RateLimit-Reset"]) == 30


def test_a_zero_or_negative_setting_still_yields_a_usable_limit():
    """
    A typo like RATE_LIMIT_REQUESTS=0 should not make every request fail with a
    division error or a 0-limit header.
    """
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=0, window_seconds=0))

    allowed, headers = limiter.check("client")

    assert allowed is True
    assert int(headers["X-RateLimit-Limit"]) >= 1


# ===========================================================================
# Key handling
# ===========================================================================


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    def __init__(self, host: str) -> None:
        self.client = _FakeClient(host)


def test_the_key_is_a_digest_and_never_the_credential():
    """
    The table must not hold the API key. A digest separates clients just as well
    and cannot be read back into a credential.
    """
    limiter = RateLimiter(RateLimitSettings(enabled=True))
    secret = "sk-a-real-looking-key-value-0123456789"

    key = limiter.client_key(_FakeRequest("127.0.0.1"), secret)

    assert secret not in key
    assert "key:" in key
    assert len(key) < len(secret) + 8


def test_the_same_key_always_maps_to_the_same_digest():
    limiter = RateLimiter(RateLimitSettings(enabled=True))
    secret = "sk-a-real-looking-key-value-0123456789"

    first = limiter.client_key(_FakeRequest("127.0.0.1"), secret)
    second = limiter.client_key(_FakeRequest("10.0.0.9"), secret)

    assert first == second, (
        "the same credential must map to the same bucket regardless of address, "
        "or a client behind a changing address gets a fresh budget each time"
    )


def test_without_a_key_the_address_is_used():
    limiter = RateLimiter(RateLimitSettings(enabled=True))

    key = limiter.client_key(_FakeRequest("10.1.2.3"), None)

    assert key == "addr:10.1.2.3"


def test_different_addresses_are_different_clients():
    limiter = RateLimiter(RateLimitSettings(enabled=True))

    assert limiter.client_key(_FakeRequest("10.1.2.3"), None) != limiter.client_key(
        _FakeRequest("10.1.2.4"), None
    )


# ===========================================================================
# Through the application
# ===========================================================================


@pytest.fixture
def limited_client(monkeypatch):
    """The real app, with the limiter switched on and its counters cleared."""
    from app import main as main_module

    monkeypatch.setattr(main_module.limiter.settings, "enabled", True)
    monkeypatch.setattr(main_module.limiter.settings, "requests", 5)
    monkeypatch.setattr(main_module.limiter.settings, "window_seconds", 60.0)
    main_module.limiter.reset()

    with TestClient(main_module.app) as client:
        yield client

    main_module.limiter.reset()


def test_a_burst_is_refused_with_429(limited_client):
    """
    End to end, through the real middleware chain.

    A cheap unauthenticated route is used so the assertion is about the limiter
    and not about an LLM call.
    """
    statuses = []
    for _ in range(8):
        statuses.append(limited_client.get("/api/v1/resume/model-catalog").status_code)

    assert 429 in statuses, f"no request was limited; statuses were {statuses}"


def test_the_refusal_explains_itself(limited_client):
    for _ in range(5):
        limited_client.get("/api/v1/resume/model-catalog")

    response = limited_client.get("/api/v1/resume/model-catalog")

    assert response.status_code == 429
    assert (
        "RATE_LIMIT_REQUESTS" in response.json()["detail"]
    ), "the message should say what to change, not just that a limit was hit"


def test_a_refusal_carries_retry_after(limited_client):
    for _ in range(5):
        limited_client.get("/api/v1/resume/model-catalog")

    response = limited_client.get("/api/v1/resume/model-catalog")

    assert "Retry-After" in response.headers
    assert "X-RateLimit-Limit" in response.headers


def test_allowed_requests_carry_the_counters(limited_client):
    response = limited_client.get("/api/v1/resume/model-catalog")

    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "5"
    assert int(response.headers["X-RateLimit-Remaining"]) < 5


def test_health_is_exempt(limited_client):
    """
    A container healthcheck every ten seconds must not trip a limit worth
    having; otherwise the orchestrator kills a healthy container.
    """
    statuses = [limited_client.get("/api/v1/resume/health").status_code for _ in range(30)]

    assert set(statuses) == {200}, f"health was rate limited: {set(statuses)}"


def test_the_limiter_is_off_by_default(monkeypatch):
    """
    A default-on limiter would start refusing a single-user local deployment,
    which is the deployment this project is built for.
    """
    from app.core.config import Settings

    settings = Settings()
    assert settings.RATE_LIMIT_ENABLED is False

    fresh = Settings(_env_file=None)
    assert fresh.RATE_LIMIT_ENABLED is False


def test_disabling_it_lets_a_burst_through(monkeypatch):
    from app import main as main_module

    monkeypatch.setattr(main_module.limiter.settings, "enabled", False)
    main_module.limiter.reset()

    with TestClient(main_module.app) as client:
        statuses = [client.get("/api/v1/resume/model-catalog").status_code for _ in range(20)]

    assert 429 not in statuses, "the limiter applied while disabled"


def test_the_settings_default_to_something_usable():
    defaults = RateLimitSettings()

    assert defaults.enabled is False
    assert defaults.requests > 0
    assert defaults.window_seconds > 0
    assert "/health" in defaults.exempt_paths
    assert "/api/v1/resume/health" in defaults.exempt_paths


def test_a_real_clock_advances_the_window():
    """
    The limiter is called without ``now`` in production, so the real clock path
    is the one that has to work.
    """
    limiter = RateLimiter(RateLimitSettings(enabled=True, requests=2, window_seconds=0.05))

    assert limiter.check("client")[0]
    assert limiter.check("client")[0]
    assert limiter.check("client")[0] is False

    time.sleep(0.06)

    assert limiter.check("client")[0] is True
