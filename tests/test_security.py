"""Security regression tests for URL, path, and API-key boundaries."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import (
    UnsafeInputError,
    require_api_key,
    validate_local_file,
    validate_public_http_url,
)
from app.main import app


def _use_real_auth() -> None:
    app.dependency_overrides.pop(require_api_key, None)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1:8000/admin",
        "http://localhost/admin",
        "http://[::1]/admin",
        "http://2130706433/admin",
        "https://user:password@example.com/job",
    ],
)
def test_outbound_url_rejects_unsafe_destinations(url):
    with pytest.raises(UnsafeInputError):
        validate_public_http_url(url, resolve_dns=False)


def test_outbound_url_accepts_public_literal_without_dns():
    assert validate_public_http_url("https://example.com/job", resolve_dns=False)


def test_file_validation_prevents_root_escape(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    allowed = root / "resume.pdf"
    allowed.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(settings, "ALLOWED_FILE_ROOTS", [str(root)])

    assert validate_local_file(allowed, allowed_extensions={".pdf"}) == allowed.resolve()
    with pytest.raises(UnsafeInputError):
        validate_local_file(tmp_path / "outside.pdf", allowed_extensions={".pdf"})


def test_generated_pdfs_get_unique_immutable_paths(tmp_path, monkeypatch):
    from app.services.jobs import full_pipeline

    monkeypatch.setattr(full_pipeline, "OUTPUT_DIR", tmp_path)
    first = full_pipeline.save_pdf(b"%PDF-fake", "same.pdf")
    second = full_pipeline.save_pdf(b"%PDF-fake", "same.pdf")
    assert first != second
    assert first.read_bytes() == second.read_bytes() == b"%PDF-fake"


def test_diff_preview_escapes_user_html():
    from app.services.cv.diff_preview import DiffPreviewService

    result = DiffPreviewService.generate_word_diff(
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
    )
    assert "<script>" not in result["diff_html"]
    assert "<img" not in result["diff_html"]
    assert "&lt;script&gt;" in result["diff_html"]


def test_openapi_advertises_api_key_security_scheme():
    schema = app.openapi()
    assert "APIKeyHeader" in schema["components"]["securitySchemes"]
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"


def test_protected_jobs_fail_closed_without_configured_key(monkeypatch):
    _use_real_auth()
    monkeypatch.setattr(settings, "API_KEY", "")
    response = TestClient(app).post(
        "/api/v1/jobs/fetch-jd",
        json={"job_url": "https://example.com/job"},
    )
    assert response.status_code == 503


def test_protected_jobs_require_matching_key(monkeypatch):
    _use_real_auth()
    monkeypatch.setattr(settings, "API_KEY", "test-secret-for-auth")
    response = TestClient(app).post(
        "/api/v1/jobs/fetch-jd",
        json={"job_url": "https://example.com/job"},
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401
