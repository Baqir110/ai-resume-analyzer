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

import pytest

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


def _has_real_gateway_key() -> bool:
    """True only if a plausible (non-stub) gateway key is configured."""
    key = os.getenv("EXPLABS_API_KEY", "").strip() or os.getenv("EXPERIENTIAL_ORG_KEY", "").strip()
    return bool(key) and key not in _STUB_VALUES


def _has_any_real_provider_key() -> bool:
    """True if any native provider key looks real."""
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
