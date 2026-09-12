"""
Integration test: real LLM gateway call.

Marked with @pytest.mark.integration. Auto-skipped by conftest.py when no
real gateway key is present. To force it locally:
    pytest -m integration
"""

import pytest

from app.services.llm.provider import LLMService


@pytest.mark.integration
def test_gateway_connection():
    """Live smoke test — verifies the gateway accepts a real prompt."""
    assert LLMService is not None

    response = LLMService.generate(
        prompt="Reply with exactly: Gateway connection successful!",
        provider="experiential",
        route_mode="experiential",
    )

    assert response, "gateway returned an empty response"
    assert response.strip(), "gateway returned whitespace-only"
    assert (
        "successful" in response.lower()
    ), f"gateway responded but did not echo the expected phrase: {response!r}"
