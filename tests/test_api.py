"""
Unit tests: FastAPI endpoints.

The /analyze endpoint calls optimize_resume_bullets() when missing skills are
found. We patch the underlying LLM call so this test never touches the
network. This makes it run identically in CI and locally.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_resume_analysis_endpoint(monkeypatch):
    # Patch the retry wrapper so no real LLM call happens. Returning a
    # static string exercises the full endpoint code path.
    monkeypatch.setattr(
        "app.services.cv.optimizer._call_llm_with_retry",
        lambda *_args, **_kwargs: "Mocked bullet rewrite for tests.",
    )

    jd = (
        "We are looking for an engineer with Python, FastAPI, Docker, " "and PostgreSQL experience."
    )
    resume_content = (
        "Experienced software developer skilled in Python, FastAPI, " "and Docker containerization."
    )

    response = client.post(
        "/api/v1/resume/analyze",
        data={"job_description": jd},
        files={
            "resume_file": (
                "resume.txt",
                resume_content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "ats_match_score" in data
    assert "Python" in data["matching_skills"]
    assert "FastAPI" in data["matching_skills"]
    assert "PostgreSQL" in data["missing_skills"]


# ---------------------------------------------------------------------------
# The ATS breakdown and the suggestions that go with it
# ---------------------------------------------------------------------------


def test_analysis_returns_a_transparent_breakdown(monkeypatch):
    """The score must be reconstructible from the per-category numbers."""
    monkeypatch.setattr(
        "app.services.cv.optimizer._call_llm_with_retry",
        lambda *_args, **_kwargs: "Mocked bullet rewrite for tests.",
    )

    response = client.post(
        "/api/v1/resume/analyze",
        data={
            "job_description": (
                "Senior Platform Engineer. Must have Kubernetes, Docker and "
                "Terraform experience. 5+ years."
            )
        },
        files={
            "resume_file": (
                "resume.txt",
                b"Jane Doe\njane@example.com\n\nSenior Platform Engineer\n\n"
                b"EXPERIENCE\nEngineer at Northwind, 2020 - 2026\n"
                b"- Ran Kubernetes on AWS with Terraform and Docker.\n\n"
                b"EDUCATION\nBSc Computer Science\n\nSKILLS\nKubernetes Docker "
                b"Terraform AWS Python\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    breakdown = data["ats_breakdown"]

    assert breakdown["ats_score"] == pytest.approx(
        sum(
            category["weighted_points"]
            for category in breakdown["categories"].values()
            if not category.get("not_measured")
        ),
        abs=0.2,
    )
    # No PDF exists at this point, so that category must say so rather than
    # reporting a zero it did not earn.
    assert breakdown["categories"]["pdf_parsing"]["not_measured"] is True
    assert breakdown["pre_generation"] is True

    for suggestion in data["ats_suggestions"]:
        assert suggestion["issue"]
        assert suggestion["action"]


def test_analysis_recommends_a_layout(monkeypatch):
    """Requirements 28/30: a recommendation, with a reason and a length judgement."""
    monkeypatch.setattr(
        "app.services.cv.optimizer._call_llm_with_retry",
        lambda *_args, **_kwargs: "Mocked bullet rewrite for tests.",
    )

    response = client.post(
        "/api/v1/resume/analyze",
        data={"job_description": "Senior Platform Engineer with Kubernetes experience."},
        files={
            "resume_file": (
                "resume.txt",
                b"Jane Doe\njane@example.com\n\nEXPERIENCE\nEngineer, 2020-2026\n\n"
                b"EDUCATION\nBSc\n\nSKILLS\nKubernetes\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    rec = response.json()["layout_recommendation"]
    assert rec["recommended_layout"]
    assert rec["reason"]
    assert rec["ats_safety"]
    assert rec["page_guidance"]["basis"]


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------


def _pdf_upload():
    from tests.test_ats_scoring_helpers import minimal_pdf

    return {"pdf_file": ("cv.pdf", minimal_pdf(), "application/pdf")}


def test_validate_ats_activates_the_pdf_category():
    """The final score must include the checks the first score had to skip."""
    response = client.post(
        "/api/v1/resume/validate-ats",
        data={"job_description": "Senior Platform Engineer with Kubernetes."},
        files=_pdf_upload(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["pdf_parsing"]["measured"] is True
    assert data["pdf_parsing"]["score"] is not None
    assert data["breakdown"]["pre_generation"] is False
    assert 0 <= data["ats_score"] <= 100


def test_validate_ats_reports_the_layout_used():
    response = client.post(
        "/api/v1/resume/validate-ats",
        data={"job_description": "Senior Platform Engineer.", "layout": "international_ats"},
        files=_pdf_upload(),
    )

    assert response.status_code == 200
    layout = response.json()["layout"]
    assert layout["requested"] == "international_ats"
    assert layout["known"] is True
    assert layout["ats_safety"]


def test_validate_ats_reports_an_unknown_layout_rather_than_failing():
    response = client.post(
        "/api/v1/resume/validate-ats",
        data={"job_description": "Senior Platform Engineer.", "layout": "not_a_layout"},
        files=_pdf_upload(),
    )

    assert response.status_code == 200
    layout = response.json()["layout"]
    assert layout["known"] is False
    assert layout["detail"]


def test_validate_ats_rejects_a_non_pdf():
    response = client.post(
        "/api/v1/resume/validate-ats",
        data={"job_description": "Engineer."},
        files={"pdf_file": ("cv.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_validate_ats_scores_a_truncated_pdf_as_zero_not_as_a_crash():
    response = client.post(
        "/api/v1/resume/validate-ats",
        data={"job_description": "Engineer."},
        files={"pdf_file": ("cv.pdf", b"%PDF-1.4 but truncated", "application/pdf")},
    )

    # A bad document is an answer, not a failed request.
    assert response.status_code == 200
    assert response.json()["pdf_parsing"]["score"] == 0.0


def test_improvement_loop_endpoint_reports_before_and_after():
    response = client.post(
        "/api/v1/resume/improvement-loop",
        data={"job_description": "Senior Platform Engineer. Must have Kubernetes, Terraform."},
        files={
            "resume_file": (
                "resume.txt",
                b"Sam Smith\nsam@example.com\n\nEXPERIENCE\nEngineer, 2020-2026\n"
                b"- Built things.\n\nEDUCATION\nBSc\n\nSKILLS\nPython\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert "initial_score" in result
    assert "final_score" in result
    assert result["explanation"]
    # The loop must not claim an improvement it did not make.
    if result["total_rounds"] == 0:
        assert result["total_improvement"] == 0
