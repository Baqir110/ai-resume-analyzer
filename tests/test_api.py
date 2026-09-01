from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_resume_analysis_endpoint():
    jd = "We are looking for an engineer with Python, FastAPI, Docker, and PostgreSQL experience."
    resume_content = "Experienced software developer skilled in Python, FastAPI, and Docker containerization."

    response = client.post(
        "/api/v1/resume/analyze",
        data={"job_description": jd},
        files={
            "resume_file": ("resume.txt", resume_content.encode("utf-8"), "text/plain")
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "ats_match_score" in data
    assert "Python" in data["matching_skills"]
    assert "FastAPI" in data["matching_skills"]
    assert "PostgreSQL" in data["missing_skills"]
