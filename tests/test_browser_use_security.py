"""Focused safety tests for the Browser Use application worker."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.jobs.browser_use_applier import (
    BrowserUseApplier,
    _configured_browser_profile_dir,
    validate_job_url,
    validate_upload_path,
)
from app.services.jobs.verification import verify_submission


def _write_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,hello",
        "http://localhost/jobs/1",
        "http://127.0.0.1/jobs/1",
        "https://10.0.0.1/jobs/1",
        "https://[::1]/jobs/1",
        "https://2130706433/jobs/1",
        "http://0177.0.0.1/jobs/1",
    ],
)
def test_job_url_rejects_non_web_and_local_destinations(url: str):
    with pytest.raises(ValueError):
        validate_job_url(url, resolve_dns=False)


def test_job_url_rejects_dns_rebinding_to_private_address(monkeypatch):
    def private_answer(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", private_answer)
    with pytest.raises(ValueError):
        validate_job_url("https://jobs.example.test/apply")


def test_upload_path_is_confined_and_pdf_only(tmp_path: Path):
    root = tmp_path / "uploads"
    root.mkdir()
    resume = root / "resume.pdf"
    _write_pdf(resume)

    assert validate_upload_path(resume, allowed_root=root) == resume.resolve()

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(ValueError):
        validate_upload_path(outside, allowed_root=root)

    wrong_type = root / "resume.txt"
    wrong_type.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_upload_path(wrong_type, allowed_root=root)


def test_browser_llm_endpoint_rejects_private_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("EXPLABS_API_KEY", "synthetic-key")
    with pytest.raises(ValueError, match="public HTTPS"):
        BrowserUseApplier()._read_key(
            {
                "kind": "openai-compatible",
                "base_url_env": "OPENAI_BASE_URL",
                "api_key_env": "EXPLABS_API_KEY",
            }
        )


def test_prompt_minimizes_sensitive_pii_by_default(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "personal": {
                    "first_name": "Test",
                    "last_name": "Candidate",
                    "date_of_birth": "1990-01-01",
                    "address": "Private Street 1",
                    "citizenship": "Testland",
                }
            }
        ),
        encoding="utf-8",
    )
    applier = BrowserUseApplier(profile_path=profile_path, allowed_upload_root=tmp_path)
    prompt = applier._build_task_prompt(
        job_url="https://jobs.example.test/apply",
        resume_path=str(tmp_path / "resume.pdf"),
        cover_letter_path=None,
        why_this_company="",
        auto_submit=False,
    )
    assert "1990-01-01" not in prompt
    assert "Private Street 1" not in prompt
    assert "Testland" not in prompt


def test_prompt_never_contains_portal_credentials(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "personal": {"first_name": "Test", "email": "applicant@example.test"},
                "portal_accounts": {
                    "email": "portal@example.test",
                    "portal_account_password": "FAKE_PORTAL_SECRET",
                },
            }
        ),
        encoding="utf-8",
    )
    applier = BrowserUseApplier(profile_path=profile_path, allowed_upload_root=tmp_path)

    prompt = applier._build_task_prompt(
        job_url="https://jobs.example.test/apply",
        resume_path=str(tmp_path / "resume.pdf"),
        cover_letter_path=None,
        why_this_company="",
        auto_submit=False,
    )

    assert "FAKE_PORTAL_SECRET" not in prompt
    assert "portal@example.test" not in prompt
    assert "Do NOT click any final" in prompt
    assert "Never create an account" in prompt


def test_legacy_live_browser_profile_is_not_reused(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHROME_USER_DATA_DIR", str(tmp_path / "real-browser-profile"))
    monkeypatch.delenv("BROWSER_AGENT_USER_DATA_DIR", raising=False)
    monkeypatch.delenv("BROWSER_AGENT_ALLOW_USER_PROFILE", raising=False)
    assert _configured_browser_profile_dir() is None


def test_persistent_profile_requires_explicit_app_owned_opt_in(monkeypatch, tmp_path: Path):
    import app.services.jobs.browser_use_applier as module

    profile_dir = tmp_path / "data" / "browser_profiles" / "worker"
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("BROWSER_AGENT_USER_DATA_DIR", str(profile_dir))
    monkeypatch.setenv("BROWSER_AGENT_ALLOW_USER_PROFILE", "true")

    assert _configured_browser_profile_dir() == str(profile_dir.resolve())
    assert profile_dir.is_dir()


class _FakeHistory:
    def __init__(self, result: str, url: str):
        self._result = result
        self._url = url
        self.history = []

    def final_result(self) -> str:
        return self._result

    def urls(self) -> list[str]:
        return [self._url]


class _FakePage:
    def __init__(self, url: str, text: str):
        self._url = url
        self._text = text

    async def get_url(self) -> str:
        return self._url

    async def get_title(self) -> str:
        return "Application"

    async def evaluate(self, _script: str) -> str:
        return self._text


class _FakeBrowser:
    def __init__(self, url: str, text: str):
        self._page = _FakePage(url, text)

    async def get_current_page(self):
        return self._page


@pytest.mark.asyncio
async def test_generic_final_result_is_not_classified_as_submitted(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("personal: {}\n", encoding="utf-8")
    applier = BrowserUseApplier(profile_path=profile_path, allowed_upload_root=tmp_path)
    history = _FakeHistory("Done.", "https://jobs.example.test/apply")
    browser = _FakeBrowser("https://jobs.example.test/apply", "Form filled")

    submitted, verified, needs_review, status, _, _ = await applier._classify_submission(
        history=history,
        browser=browser,
        auto_submit=True,
    )

    assert submitted is False
    assert verified is False
    assert needs_review is False
    assert status == "not_submitted"

    negative = await applier._classify_submission(
        history=_FakeHistory(
            "I did not submit; the application submitted button was not clicked",
            "https://jobs.example.test/apply",
        ),
        browser=browser,
        auto_submit=True,
    )
    assert negative[0] is False


@pytest.mark.asyncio
async def test_explicit_confirmation_is_classified_and_fill_only_is_not(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("personal: {}\n", encoding="utf-8")
    applier = BrowserUseApplier(profile_path=profile_path, allowed_upload_root=tmp_path)
    history = _FakeHistory(
        "Application submitted successfully", "https://jobs.example.test/thank-you"
    )
    browser = _FakeBrowser(
        "https://jobs.example.test/thank-you",
        "__STRUCTURED_SUBMISSION_EVIDENCE__:APP-123\nThank you for applying",
    )

    submitted, verified, needs_review, status, message, _ = await applier._classify_submission(
        history=history,
        browser=browser,
        auto_submit=True,
    )
    assert submitted is True
    assert verified is True
    assert needs_review is False
    assert status == "submitted"
    assert "thank you for applying" in message.casefold()

    model_only = await applier._classify_submission(
        history=_FakeHistory(
            "Application submitted successfully", "https://jobs.example.test/apply"
        ),
        browser=_FakeBrowser("https://jobs.example.test/apply", "Form filled"),
        auto_submit=True,
    )
    assert model_only[0] is True
    assert model_only[1] is False
    assert model_only[2] is True
    assert model_only[3] == "verification_required"
    downstream = verify_submission(
        {
            "submitted": model_only[0],
            "browser_application": {
                "confirmation_message": model_only[4],
                "final_url": model_only[5],
            },
        }
    )
    assert downstream.verified is False

    fill_only = await applier._classify_submission(
        history=_FakeHistory(
            "Application submitted successfully", "https://jobs.example.test/thank-you"
        ),
        browser=browser,
        auto_submit=False,
    )
    assert fill_only[0] is False
    assert fill_only[3] == "ready_for_review"


@pytest.mark.asyncio
async def test_invalid_job_url_stops_before_browser_creation(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("personal: {}\n", encoding="utf-8")
    resume = tmp_path / "resume.pdf"
    _write_pdf(resume)
    applier = BrowserUseApplier(profile_path=profile_path, allowed_upload_root=tmp_path)

    with patch(
        "app.services.jobs.browser_use_applier._make_browser", new=AsyncMock()
    ) as browser_factory:
        result = await applier.apply("file:///not-allowed", str(resume))

    assert result.success is False
    assert result.submitted is False
    browser_factory.assert_not_called()
