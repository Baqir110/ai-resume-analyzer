# app/services/jobs/error_recovery.py
"""Error classification + backoff for the apply pipeline (Section 20).

The spec is explicit that a flat try/except-and-give-up isn't enough --
different failure classes need different responses:
  - CAPTCHA / anti-bot: never retry (Section 21) -- retrying a CAPTCHA
    wall just wastes time and looks more bot-like, not less.
  - Timeouts, network blips, temporary server errors (5xx): worth a
    retry with backoff, since these are often gone a few seconds later.
  - Validation errors, "unknown field", "missing required question":
    NOT worth blindly retrying -- the form will reject it identically
    every time. These should surface for a human/the answer engine to
    fix, not spin.

This intentionally works at the orchestrator boundary (classifying
whatever error message/exception the existing browser_use_applier.py
pipeline already produces) rather than reaching into that file's
internals -- Section 20 also asks for "alternative selectors, DOM
inspection, semantic field matching," which are browser_use's own
agent's job at the page level (it already does semantic field
matching via its LLM-driven action loop); duplicating that logic here
would fight the existing, working implementation rather than extend it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailureClass(str, Enum):
    CAPTCHA = "captcha"  # never retry
    TRANSIENT = "transient"  # retry with backoff
    PERMANENT = "permanent"  # don't retry, needs a fix (bad answer, broken selector, etc.)
    UNKNOWN = "unknown"  # be conservative, retry a limited number of times


_CAPTCHA_PATTERNS = re.compile(
    r"captcha|recaptcha|hcaptcha|cloudflare|request blocked|bot detection|"
    r"human verification|access denied|are you a robot",
    re.IGNORECASE,
)
_TRANSIENT_PATTERNS = re.compile(
    r"timeout|timed out|connection (reset|refused|error)|network error|"
    r"temporarily unavailable|5\d\d|service unavailable|gateway timeout|"
    r"stale element|element not (found|interactable)|navigation failed",
    re.IGNORECASE,
)
_PERMANENT_PATTERNS = re.compile(
    r"validation error|required field|invalid (email|phone|format)|"
    r"file too large|unsupported file type|application already submitted|"
    r"position (has been |is )?(closed|filled|expired)|unknown question",
    re.IGNORECASE,
)


def classify_failure(message: str) -> FailureClass:
    if not message:
        return FailureClass.UNKNOWN
    if _CAPTCHA_PATTERNS.search(message):
        return FailureClass.CAPTCHA
    if _PERMANENT_PATTERNS.search(message):
        return FailureClass.PERMANENT
    if _TRANSIENT_PATTERNS.search(message):
        return FailureClass.TRANSIENT
    return FailureClass.UNKNOWN


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    # UNKNOWN failures get fewer attempts than confirmed-TRANSIENT ones --
    # conservative, since we don't actually know retrying will help.
    max_attempts_unknown: int = 2


async def run_with_recovery(
    action: Callable[[], Awaitable[T]],
    error_message_extractor: Callable[[T], str],
    policy: RetryPolicy | None = None,
) -> tuple[T, int, FailureClass | None]:
    """Runs `action`, retrying with backoff based on the failure class
    found in the result's error message (via `error_message_extractor`).

    Returns (final_result, attempts_made, last_failure_class). A CAPTCHA
    or PERMANENT classification stops immediately -- attempts_made will
    be 1 in that case, by design.
    """
    policy = policy or RetryPolicy()
    attempt = 0

    while True:
        attempt += 1
        result = await action()
        message = error_message_extractor(result) or ""
        if not message:
            return result, attempt, None  # success, no error to classify

        failure_class = classify_failure(message)

        if failure_class in (FailureClass.CAPTCHA, FailureClass.PERMANENT):
            logger.info(
                "Failure classified as %s (attempt %d) -- not retrying: %s",
                failure_class.value,
                attempt,
                message,
            )
            return result, attempt, failure_class

        limit = (
            policy.max_attempts_unknown
            if failure_class == FailureClass.UNKNOWN
            else policy.max_attempts
        )
        if attempt >= limit:
            logger.info(
                "Failure classified as %s, exhausted %d attempts -- giving up: %s",
                failure_class.value,
                attempt,
                message,
            )
            return result, attempt, failure_class

        delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2 ** (attempt - 1)))
        logger.info(
            "Failure classified as %s (attempt %d/%d) -- retrying in %.1fs: %s",
            failure_class.value,
            attempt,
            limit,
            delay,
            message,
        )
        await asyncio.sleep(delay)


__all__ = ["FailureClass", "RetryPolicy", "classify_failure", "run_with_recovery"]
