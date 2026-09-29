"""
Rate limits: classification, the stated duration, and the cooldown it causes.

Three defects, all found by running the full CV pipeline against a provider
whose free tier actually rate-limits, and all of which turned a served request
into a failed one:

1. ``"billing"`` was a marker for an exhausted credit balance. Groq's rate-limit
   body ends with an upgrade link containing that word, so a retryable 429 was
   classified as a non-retryable credits problem -- the router gave up, and told
   the operator to buy quota for a request that only needed to wait.

2. ``"402"`` was matched as a *text* substring, so a request id or a token count
   containing those three digits classified a failure as a payment problem. A
   status code is the authority for that and is consulted first.

3. Every retryable failure caused a flat 600-second cooldown. A rate limit that
   the provider said would clear in 5.9 seconds took the provider out of
   rotation for ten minutes, so the emergency pass fired against a healthy
   provider.

Each is asserted against a real provider body rather than a paraphrase, because
the bug was in the gap between the intended wording and the wording that
arrived.
"""

from __future__ import annotations

import pytest

from app.services.llm import provider as provider_module
from app.services.llm.provider import (
    _RETRYABLE_CATEGORIES,
    _SAME_REQUEST_RETRYABLE_CATEGORIES,
    LLM_ERROR_CREDITS,
    LLM_ERROR_RATE_LIMIT,
    LLMService,
    _env_float,
    _env_int,
    classify_llm_error,
    rate_limit_retry_after,
)

#: Verbatim from Groq, which is where the misclassification was found. The
#: upgrade link at the end is the point: "billing" appears in the marketing tail
#: of a rate limit.
GROQ_RATE_LIMIT = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_01m2dx` service tier `on_demand` "
    "on tokens per minute (TPM): Limit 8000, Used 5458, Requested 3328. Please "
    "try again in 5.895s. Need more tokens? Upgrade to Dev Tier today at "
    "https://console.groq.com/settings/billing', 'type': 'tokens', "
    "'code': 'rate_limit_exceeded'}}"
)


class FakeHTTPError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        if status is not None:
            self.status_code = status


class WithRetryAfterHeader(FakeHTTPError):
    def __init__(self, message: str, header: str) -> None:
        super().__init__(message)

        response = type("_R", (), {"headers": {"Retry-After": header}})()
        self.response = response


class WithRetryAfterAttribute(FakeHTTPError):
    def __init__(self, message: str, seconds: float) -> None:
        super().__init__(message)
        self.retry_after_seconds = seconds


# ===========================================================================
# 1. Classification
# ===========================================================================


def test_a_rate_limit_with_an_upgrade_link_is_a_rate_limit():
    """
    The regression. This exact body classified as ``insufficient_credits``,
    which is non-retryable, so the router abandoned a request that would have
    succeeded a few seconds later.
    """
    category = classify_llm_error(FakeHTTPError(GROQ_RATE_LIMIT, 429), "groq")

    assert (
        category == LLM_ERROR_RATE_LIMIT
    ), f"a body that says 'Rate limit reached' classified as {category!r}"
    assert category in _RETRYABLE_CATEGORIES, "a rate limit must be retried"
    assert category in _SAME_REQUEST_RETRYABLE_CATEGORIES


def test_a_spent_balance_is_still_a_spent_balance():
    """
    The other direction. Tightening the markers must not have made every 429
    retryable, because a 402 needs a different remedy than a wait.
    """
    category = classify_llm_error(
        FakeHTTPError("Error code: 402 - {'message': 'Insufficient Balance'}", 402),
        "deepseek",
    )
    assert category == LLM_ERROR_CREDITS
    assert category not in _RETRYABLE_CATEGORIES


def test_a_quota_exhausted_429_is_not_retried():
    """
    Gemini's wording for "your plan needs changing", which is not a rate limit
    that a backoff will clear.
    """
    category = classify_llm_error(
        FakeHTTPError(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
            "'You exceeded your current quota, please check your plan and "
            "billing details.'}}",
            429,
        ),
        "gemini",
    )
    assert category == LLM_ERROR_CREDITS


def test_a_429_that_says_the_balance_is_out_is_credits():
    category = classify_llm_error(
        FakeHTTPError("429 insufficient credits on this account", 429), "groq"
    )
    assert category == LLM_ERROR_CREDITS


def test_three_digits_in_a_request_id_are_not_a_payment_problem():
    """
    ``"402"`` as a text substring matched request ids and token counts.
    """
    category = classify_llm_error(FakeHTTPError("request abc402def was rejected", 400), "groq")
    assert (
        category != LLM_ERROR_CREDITS
    ), "'402' inside an unrelated string must not classify as a spent balance"


# ===========================================================================
# 2. The stated duration
# ===========================================================================


def test_the_duration_is_read_from_the_body():
    assert rate_limit_retry_after(FakeHTTPError(GROQ_RATE_LIMIT, 429)) == pytest.approx(
        5.895, abs=0.001
    )


def test_a_retry_after_header_is_honoured():
    assert rate_limit_retry_after(WithRetryAfterHeader("429", "12")) == 12.0


def test_an_explicit_attribute_is_honoured():
    assert rate_limit_retry_after(WithRetryAfterAttribute("429", 7.0)) == 7.0


def test_no_duration_means_no_guess():
    """
    Returning ``None`` lets the caller use its own short backoff. Inventing a
    number here would silently substitute a guess for the provider's knowledge.
    """
    assert rate_limit_retry_after(FakeHTTPError("429 too many requests", 429)) is None


def test_a_non_rate_limit_is_never_asked_for_a_duration():
    assert rate_limit_retry_after(FakeHTTPError("500 Internal Server Error", 500)) is None


# ===========================================================================
# 3. The cooldown
# ===========================================================================


@pytest.fixture
def clear_exhaustion():
    LLMService._exhausted_models.clear()
    yield
    LLMService._exhausted_models.clear()


def test_a_rate_limits_cooldown_is_the_duration_it_stated(clear_exhaustion):
    """
    The second defect: a 600-second blackout for a 5.9-second rate limit.
    """
    LLMService._mark_exhausted(
        "groq", "test-model", rate_limit_retry_after(FakeHTTPError(GROQ_RATE_LIMIT, 429))
    )

    remaining = LLMService._shortest_exhaustion_remaining()

    assert 5.0 < remaining < 6.5, (
        f"cooldown was {remaining:.1f}s; the provider asked for 5.895s and the "
        f"flat default is {LLMService._cooldown_seconds}s"
    )


def test_a_failure_with_no_stated_duration_keeps_the_flat_cooldown(clear_exhaustion):
    """
    A 500 has nothing to read. Guessing a duration would be worse than the
    existing default, so the default stands.
    """
    LLMService._mark_exhausted("groq", "test-model", None)

    remaining = LLMService._shortest_exhaustion_remaining()

    assert (
        remaining > LLMService._cooldown_seconds * 0.9
    ), f"expected the {LLMService._cooldown_seconds}s default, got {remaining:.1f}s"


def test_an_absurd_stated_duration_is_clamped(clear_exhaustion, monkeypatch):
    """
    A provider quoting an hour must not hang the router for an hour.
    """
    monkeypatch.setenv("LLM_RATE_LIMIT_BACKOFF_SECONDS", "10")

    LLMService._mark_exhausted("groq", "test-model", 3600.0)

    remaining = LLMService._shortest_exhaustion_remaining()

    assert remaining <= 10.5, f"cooldown was {remaining:.1f}s, ceiling is 10s"


def test_a_sub_second_duration_still_produces_a_usable_cooldown(clear_exhaustion):
    """
    Rounding a provider's "0.4s" to nothing would make the next call land inside
    the window it was told to avoid.
    """
    LLMService._mark_exhausted("groq", "test-model", 0.4)

    remaining = LLMService._shortest_exhaustion_remaining()

    assert 0.5 < remaining < 2.0, f"cooldown was {remaining:.2f}s"


def test_no_cooldowns_means_nothing_to_wait_for(clear_exhaustion):
    assert LLMService._shortest_exhaustion_remaining() == float("inf")


# ===========================================================================
# 4. The helpers
# ===========================================================================


def test_the_numeric_environment_helpers_are_defined_once():
    """
    ``_env_int`` and ``_env_bool`` were each defined twice, identically. The
    second shadowed the first, so the duplication was invisible to every test.
    """
    import inspect

    source = inspect.getsource(provider_module)

    for name in ("_env_int", "_env_bool", "_env_float"):
        assert source.count(f"\ndef {name}(") == 1, f"{name} is defined more than once"


def test_env_float_reads_and_falls_back(monkeypatch):
    monkeypatch.setenv("_PROBE", "3.25")
    assert _env_float("_PROBE", 1.0) == 3.25

    monkeypatch.setenv("_PROBE", "not-a-number")
    assert _env_float("_PROBE", 9.0) == 9.0

    monkeypatch.delenv("_PROBE", raising=False)
    assert _env_float("_PROBE", 4.0) == 4.0


def test_env_int_still_behaves(monkeypatch):
    """The deduplication must not have changed the integer reader."""
    monkeypatch.setenv("_PROBE", "7")
    assert _env_int("_PROBE", 1) == 7

    monkeypatch.setenv("_PROBE", "nope")
    assert _env_int("_PROBE", 3) == 3
