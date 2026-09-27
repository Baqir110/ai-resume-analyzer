# tests/test_package_validator.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.jobs.agent_schemas import NormalizedJob
from app.services.jobs.backend_submitter import ApplicationPackage
from app.services.jobs.package_validator import validate_package


def _write_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)


def _job(**overrides):
    defaults = dict(job_id="1", title="DevOps Engineer", company="ACME")
    defaults.update(overrides)
    return NormalizedJob(**defaults)


def test_valid_package_passes(tmp_path):
    cv = tmp_path / "cv.pdf"
    _write_pdf(cv)
    package = ApplicationPackage(
        job=_job(),
        cv_path=str(cv),
        cover_letter_path=None,
        answers={"salary_expectation": "Negotiable"},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package)
    assert result.valid is True
    assert result.errors == []


def test_non_pdf_content_with_pdf_extension_fails(tmp_path):
    cv = tmp_path / "fake.pdf"
    cv.write_bytes(b"plain text, not a PDF")
    package = ApplicationPackage(
        job=_job(),
        cv_path=str(cv),
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package)
    assert result.valid is False
    assert any("PDF" in error for error in result.errors)


def test_missing_cv_file_fails(tmp_path):
    package = ApplicationPackage(
        job=_job(),
        cv_path=str(tmp_path / "does_not_exist.pdf"),
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package)
    assert result.valid is False
    assert any("does not exist" in e for e in result.errors)


def test_empty_cv_file_fails(tmp_path):
    cv = tmp_path / "empty.pdf"
    cv.write_bytes(b"")
    package = ApplicationPackage(
        job=_job(),
        cv_path=str(cv),
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package)
    assert result.valid is False
    assert any("empty" in e for e in result.errors)


def test_missing_candidate_identity_fails(tmp_path):
    cv = tmp_path / "cv.pdf"
    _write_pdf(cv)
    package = ApplicationPackage(
        job=_job(),
        cv_path=str(cv),
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package)
    assert result.valid is False
    assert any("first_name" in e for e in result.errors)


def test_required_cover_letter_missing_fails(tmp_path):
    cv = tmp_path / "cv.pdf"
    _write_pdf(cv)
    package = ApplicationPackage(
        job=_job(),
        cv_path=str(cv),
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package, require_cover_letter=True)
    assert result.valid is False
    assert any("cover letter" in e.lower() for e in result.errors)


def test_placeholder_answer_fails():
    package = ApplicationPackage(
        job=_job(),
        cv_path="/tmp/x.pdf",
        cover_letter_path=None,
        answers={"why_this_company": "{{company_name}} is great"},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package, required_answer_keys=["why_this_company"])
    assert result.valid is False
    assert any("placeholder" in e.lower() for e in result.errors)


def test_missing_required_answer_fails():
    package = ApplicationPackage(
        job=_job(),
        cv_path="/tmp/x.pdf",
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package, required_answer_keys=["salary_expectation"])
    assert result.valid is False
    assert any("salary_expectation" in e for e in result.errors)


def test_unknown_company_or_title_fails_closed(tmp_path):
    cv = tmp_path / "cv.pdf"
    _write_pdf(cv)
    package = ApplicationPackage(
        job=_job(company="Unknown", title="Unknown"),
        cv_path=str(cv),
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "A", "last_name": "B", "email": "a@b.com"}},
    )
    result = validate_package(package)
    assert result.valid is False
    assert any("company" in error.lower() for error in result.errors)
    assert any("title" in error.lower() for error in result.errors)
