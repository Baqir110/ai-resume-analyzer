"""Tests for app.services.jobs.answer_engine."""

import pytest

from app.services.jobs.answer_engine import CANONICAL_QUESTIONS, answer_question

# ---------------------------------------------------------------------------
# Minimal profile that mirrors the real applicant_profile.yaml structure
# ---------------------------------------------------------------------------

SAMPLE_PROFILE = {
    "personal": {
        "first_name": "Test",
        "last_name": "Candidate",
        "full_name": "Test Candidate",
        "email": "test@example.com",
        "phone": "+49 152 00000000",
        "linkedin": "https://www.linkedin.com/in/test/",
        "github": "https://github.com/testuser",
    },
    "work_authorization": {
        "authorized_to_work": True,
        "requires_sponsorship": False,
        "citizenship": "Pakistan",
        "visa_status": "Student visa (Germany)",
    },
    "experience": {
        "total_years": 3,
        "skills": {
            "Python": 3,
            "Docker": 2,
            "Kubernetes": 1,
            "AWS": 2,
        },
    },
    "education": [
        {
            "institution": "Otto-Friedrich-Universitat Bamberg",
            "degree": "M.Sc. in International Software Systems Science",
            "current": True,
        }
    ],
    "free_text": {
        "salary_expectation": "Negotiable, based on role and location",
        "availability": "Immediately",
        "notice_period": "2 weeks",
        "willing_to_relocate": "Yes",
        "german_level": "B1",
        "english_level": "IELTS 8.0",
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_work_authorization_question_matched():
    """'Are you authorized to work in Germany?' should return a non-empty string."""
    result = answer_question(
        "Are you authorized to work in Germany?",
        SAMPLE_PROFILE,
    )
    assert result, "Expected a non-empty work-authorization answer"
    assert result in ("Yes", "No")


def test_salary_question_matched():
    """'What is your expected salary?' should return the profile salary expectation."""
    result = answer_question(
        "What is your expected salary?",
        SAMPLE_PROFILE,
    )
    assert result, "Expected a non-empty salary answer"
    assert "negotiable" in result.lower() or len(result) > 3


def test_python_years_matched():
    """'How many years of Python experience?' should return something."""
    result = answer_question(
        "How many years of Python experience do you have?",
        SAMPLE_PROFILE,
    )
    assert result, "Expected a non-empty Python experience answer"


def test_unknown_question_returns_empty():
    """'Do you have a driver's license type B?' has no canonical match -> ''."""
    result = answer_question(
        "Do you have a driver's license type B?",
        SAMPLE_PROFILE,
    )
    assert result == "", f"Expected empty string for unmatched question, got: {result!r}"


def test_canonical_coverage():
    """CANONICAL_QUESTIONS must define at least 15 question types."""
    assert (
        len(CANONICAL_QUESTIONS) >= 15
    ), f"Expected at least 15 canonical question types, found {len(CANONICAL_QUESTIONS)}"


def test_work_auth_no_profile_requires_manual_review():
    """Missing safety-critical authorization must never become an affirmative answer."""
    result = answer_question(
        "Are you authorized to work in Germany?",
        {},
    )
    assert result == "Manual review required"


def test_german_level_question():
    """Questions about German proficiency should return the level."""
    result = answer_question(
        "What is your German language level?",
        SAMPLE_PROFILE,
    )
    assert result == "B1"


def test_notice_period_question():
    """'What is your notice period?' should return the profile value."""
    result = answer_question(
        "What is your notice period?",
        SAMPLE_PROFILE,
    )
    assert result, "Expected a non-empty notice period answer"


def test_empty_question_returns_empty():
    """Empty question string should return empty string."""
    result = answer_question("", SAMPLE_PROFILE)
    assert result == ""


def test_invalid_profile_type_returns_empty():
    """Non-dict profile should return empty string without raising."""
    result = answer_question("What is your salary expectation?", None)
    assert result == ""


def test_linkedin_question():
    """'linkedin' keyword should resolve to the profile URL."""
    result = answer_question("What is your LinkedIn profile URL?", SAMPLE_PROFILE)
    assert "linkedin" in result.lower()
