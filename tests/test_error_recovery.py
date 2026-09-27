# tests/test_error_recovery.py
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.services.jobs.error_recovery import (
    FailureClass,
    RetryPolicy,
    classify_failure,
    run_with_recovery,
)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Blocked by Cloudflare Request Blocked", FailureClass.CAPTCHA),
        ("Please solve the reCAPTCHA to continue", FailureClass.CAPTCHA),
        ("Connection timed out after 30s", FailureClass.TRANSIENT),
        ("502 Bad Gateway", FailureClass.TRANSIENT),
        ("Stale element reference", FailureClass.TRANSIENT),
        ("Validation error: email format invalid", FailureClass.PERMANENT),
        ("This position has been closed", FailureClass.PERMANENT),
        ("Something totally unexpected happened", FailureClass.UNKNOWN),
        ("", FailureClass.UNKNOWN),
    ],
)
def test_classify_failure(message, expected):
    assert classify_failure(message) == expected


@pytest.mark.asyncio
async def test_captcha_never_retries():
    action = AsyncMock(return_value={"error_message": "Cloudflare captcha detected"})
    with patch(
        "app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        result, attempts, failure = await run_with_recovery(
            action, lambda r: r.get("error_message", ""), RetryPolicy(max_attempts=5)
        )
    assert attempts == 1
    assert failure == FailureClass.CAPTCHA
    mock_sleep.assert_not_called()
    action.assert_awaited_once()


@pytest.mark.asyncio
async def test_permanent_failure_never_retries():
    action = AsyncMock(return_value={"error_message": "Validation error: required field missing"})
    with patch("app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock):
        result, attempts, failure = await run_with_recovery(
            action, lambda r: r.get("error_message", ""), RetryPolicy(max_attempts=5)
        )
    assert attempts == 1
    assert failure == FailureClass.PERMANENT


@pytest.mark.asyncio
async def test_transient_failure_retries_up_to_limit_then_gives_up():
    action = AsyncMock(return_value={"error_message": "Connection timed out"})
    with patch(
        "app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        result, attempts, failure = await run_with_recovery(
            action, lambda r: r.get("error_message", ""), RetryPolicy(max_attempts=3)
        )
    assert attempts == 3
    assert failure == FailureClass.TRANSIENT
    assert mock_sleep.await_count == 2  # slept between attempts 1->2 and 2->3, not after the last


@pytest.mark.asyncio
async def test_transient_failure_succeeds_on_second_attempt():
    action = AsyncMock(
        side_effect=[
            {"error_message": "Timeout waiting for page load"},
            {"error_message": "", "success": True},
        ]
    )
    with patch("app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock):
        result, attempts, failure = await run_with_recovery(
            action, lambda r: r.get("error_message", ""), RetryPolicy(max_attempts=3)
        )
    assert attempts == 2
    assert failure is None
    assert result["success"] is True


@pytest.mark.asyncio
async def test_unknown_failure_gets_fewer_attempts_than_transient():
    action = AsyncMock(return_value={"error_message": "Something totally unexpected"})
    with patch("app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock):
        result, attempts, failure = await run_with_recovery(
            action,
            lambda r: r.get("error_message", ""),
            RetryPolicy(max_attempts=5, max_attempts_unknown=2),
        )
    assert attempts == 2
    assert failure == FailureClass.UNKNOWN


@pytest.mark.asyncio
async def test_backoff_delay_grows_exponentially():
    action = AsyncMock(return_value={"error_message": "network error"})
    with patch(
        "app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        await run_with_recovery(
            action,
            lambda r: r.get("error_message", ""),
            RetryPolicy(max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=100.0),
        )
    delays = [call.args[0] for call in mock_sleep.await_args_list]
    assert delays == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_success_on_first_try_never_sleeps():
    action = AsyncMock(return_value={"error_message": ""})
    with patch(
        "app.services.jobs.error_recovery.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        result, attempts, failure = await run_with_recovery(
            action, lambda r: r.get("error_message", "")
        )
    assert attempts == 1
    assert failure is None
    mock_sleep.assert_not_called()
