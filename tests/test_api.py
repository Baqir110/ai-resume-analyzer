"""
Unit tests: FastAPI endpoints.

The /analyze endpoint calls optimize_resume_bullets() when missing skills are
found. We patch the underlying LLM call so this test never touches the
network. This makes it run identically in CI and locally.
"""

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
        "We are looking for an engineer with Python, FastAPI, Docker, "
        "and PostgreSQL experience."
    )
    resume_content = (
        "Experienced software developer skilled in Python, FastAPI, "
        "and Docker containerization."
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
