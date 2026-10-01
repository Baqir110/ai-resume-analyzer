"""
A mistyped CV layout must be answered as a client error.

The generator already rejected an unknown layout name, but it raised ``ValueError``
from inside a ``try`` that only handled ``FactualValidationError``, so it escaped
to Starlette's default handler. A request that mistyped one character of a layout
name came back as ``500 Internal Server Error`` with no indication of what was
wrong -- the same response a broken TeX installation would produce.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

_RESUME = (
    "Hans Mueller\nh@x.de\nBerufserfahrung: Engineer\nAusbildung: B.Sc.\n"
    "Faehigkeiten: Python, Kubernetes, Docker. " * 10
)
_JD = "Python engineer with Kubernetes and Docker experience. " * 5


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def headers():
    return {"X-API-Key": str(settings.API_KEY)}


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/resume/generate-german-cv",
        "/api/v1/resume/generate-tex-cv",
    ],
)
def test_an_unknown_layout_is_a_400_naming_the_valid_ones(client, headers, endpoint):
    response = client.post(
        endpoint,
        data={"job_description": _JD, "layout_style": "not_a_real_layout"},
        files={"resume_file": ("cv.txt", _RESUME, "text/plain")},
        headers=headers,
    )

    assert response.status_code == 400, (
        "a mistyped layout is a client mistake, not a server failure"
    )
    detail = response.json()["detail"]
    assert detail["code"] == "unknown_layout"
    assert detail["requested"] == "not_a_real_layout"
    # The point of the 400 is that it is actionable.
    assert "german_minimal_ats" in detail["available"]
    assert "international_ats" in detail["available"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/resume/generate-german-cv",
        "/api/v1/resume/generate-tex-cv",
    ],
)
@pytest.mark.parametrize("style", ["auto", "auto_detect", "german_minimal_ats"])
def test_the_accepted_layout_values_are_still_accepted(client, headers, endpoint, style):
    """
    The guard on the guard. Validation must reject only genuinely unknown names,
    so the auto-select aliases and a real template have to get past it. The
    request may still fail later for unrelated reasons; what matters is that it
    is not rejected as an unknown layout.
    """
    response = client.post(
        endpoint,
        data={"job_description": _JD, "layout_style": style},
        files={"resume_file": ("cv.txt", _RESUME, "text/plain")},
        headers=headers,
    )

    assert response.status_code != 400 or response.json()["detail"]["code"] != "unknown_layout", (
        f"{style!r} is a valid layout value and must not be rejected"
    )


def test_template_style_takes_precedence_and_is_validated(client, headers):
    """
    ``template_style`` overrides ``layout_style``. The override must be the value
    that gets validated -- otherwise an unknown layout hides behind a good one in
    ``layout_style`` and reaches the generator.
    """
    response = client.post(
        "/api/v1/resume/generate-tex-cv",
        data={
            "job_description": _JD,
            "layout_style": "german_minimal_ats",
            "template_style": "bogus_from_template_field",
        },
        files={"resume_file": ("cv.txt", _RESUME, "text/plain")},
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unknown_layout"
    assert response.json()["detail"]["requested"] == "bogus_from_template_field"
