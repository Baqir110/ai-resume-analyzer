"""
Shared pytest configuration.

Provides:
  - Automatic skipping of `@pytest.mark.integration` tests when no real
    gateway key is present in the environment.
  - A fixture (`has_real_key`) that tests can use to branch if needed.

To run only the hermetic unit tests (CI behavior):
    pytest -m "not integration"

To run only integration tests:
    pytest -m integration

To run everything:
    pytest
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Keep tests independent of any developer's ignored live applicant profile.
# The fixture contains only synthetic data and is safe for CI.
os.environ["APPLICANT_PROFILE_PATH"] = str(
    Path(__file__).resolve().parent / "fixtures" / "applicant_profile.yaml"
)

# Stub values that CI or example files might set. If we see one of these,
# treat the key as absent — the real service would reject it.
_STUB_VALUES = {
    "",
    "test",
    "test_key",
    "test_key_ci",
    "dummy",
    "your_experiential_org_key",
    "your_gemini_key",
    "your_groq_key",
    "your_openai_key",
}


def _live_integration_opt_in() -> bool:
    return os.getenv("RUN_LIVE_INTEGRATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _has_real_gateway_key() -> bool:
    """True only when live integration was explicitly opted into."""
    if not _live_integration_opt_in():
        return False
    key = os.getenv("EXPLABS_API_KEY", "").strip() or os.getenv("EXPERIENTIAL_ORG_KEY", "").strip()
    return bool(key) and key not in _STUB_VALUES


def _has_any_real_provider_key() -> bool:
    """True only when live integration was explicitly opted into."""
    if not _live_integration_opt_in():
        return False
    for env_name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        value = os.getenv(env_name, "").strip()
        if value and value not in _STUB_VALUES:
            return True
    return False


@pytest.fixture(scope="session")
def has_real_key() -> bool:
    """Session-wide fixture: True if any real key is present."""
    return _has_real_gateway_key() or _has_any_real_provider_key()


@pytest.fixture(autouse=True)
def _override_api_auth_for_unit_tests():
    """Keep hermetic unit tests independent of deployment credentials.

    Production requests still go through the real API-key dependency. Tests
    that exercise authentication explicitly remove this override first.
    """

    from app.core.security import require_api_key
    from app.main import app

    previous = app.dependency_overrides.get(require_api_key)
    app.dependency_overrides[require_api_key] = lambda: None
    yield
    if previous is None:
        app.dependency_overrides.pop(require_api_key, None)
    else:
        app.dependency_overrides[require_api_key] = previous


def pytest_collection_modifyitems(config, items):
    """
    Auto-skip `@pytest.mark.integration` tests when no real key is
    configured. This makes CI and local runs behave consistently without
    per-test skipif decorators.
    """
    if _has_real_gateway_key() or _has_any_real_provider_key():
        return

    skip_integration = pytest.mark.skip(
        reason="integration test requires a real provider/gateway key",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
