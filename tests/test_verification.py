# tests/test_verification.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.jobs.verification import verify_submission


def test_verified_with_independent_browser_evidence():
    result = verify_submission(
        {
            "submitted": True,
            "browser_application": {
                "submission_verified": True,
                "submission_status": "submitted",
                "confirmation_message": "Thank you for applying to ACME!",
            },
        }
    )
    assert result.verified is True
    assert result.evidence


def test_model_prose_and_confirmation_url_are_not_independent():
    result = verify_submission(
        {
            "submitted": True,
            "browser_application": {
                "final_result": "Vielen Dank für Ihre Bewerbung.",
                "final_url": "https://acme.example/apply/thank-you",
            },
        }
    )
    assert result.verified is False


def test_verified_with_structured_confirmation_id():
    result = verify_submission({"submitted": True, "confirmation_id": "APP-12345"})
    assert result.verified is True


def test_not_verified_on_bare_success_claim_alone():
    result = verify_submission(
        {
            "submitted": True,
            "browser_application": {
                "final_result": "Done.",
                "final_url": "https://acme.example/apply",
            },
        }
    )
    assert result.verified is False
    assert "not trusted" in result.reason.lower()


def test_not_verified_when_nothing_was_submitted():
    result = verify_submission({"submitted": False})
    assert result.verified is False
    assert "no submission" in result.reason.lower()


def test_confirmation_phrase_without_page_evidence_is_rejected():
    result = verify_submission(
        {
            "submitted": True,
            "browser_application": {"final_result": "APPLICATION SUCCESSFULLY SUBMITTED"},
        }
    )
    assert result.verified is False
