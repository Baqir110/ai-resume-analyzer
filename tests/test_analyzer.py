import pytest
from app.services.analyzer import analyze_resume_content, load_skills_taxonomy


def test_load_skills_taxonomy():
    skills = load_skills_taxonomy()
    assert isinstance(skills, list)
    assert len(skills) > 0
    assert "python" in [s.lower() for s in skills]


def test_analyze_resume_content_matching():
    job_desc = "Looking for a Python software engineer with FastAPI, Docker, and PostgreSQL experience."
    resume = "I am a Python engineer who works with FastAPI and Docker daily."

    results = analyze_resume_content(resume, job_desc)

    assert "ats_match_score" in results
    assert results["ats_match_score"] > 0
    assert "Python" in results["matching_skills"]
    assert "FastAPI" in results["matching_skills"]
    assert "PostgreSQL" in results["missing_skills"]
    assert len(results["improvement_suggestions"]) > 0


def test_analyze_resume_content_empty_overlap():
    job_desc = "Seeking a Java developer with Spring Boot and AWS."
    resume = "Experienced graphic designer skilled in Photoshop and Illustrator."

    results = analyze_resume_content(resume, job_desc)

    assert results["ats_match_score"] == 0.0
    assert len(results["matching_skills"]) == 0
