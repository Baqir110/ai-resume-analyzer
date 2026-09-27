# app/services/jobs/package_validator.py
"""Application package validation (Section 15).

Before any submission attempt -- backend or browser -- the assembled
ApplicationPackage must pass this gate. This is deliberately independent
of backend_submitter.py and full_pipeline.py: it doesn't generate
anything, it only checks what was already produced, so a bug in CV
generation or answer-mapping gets caught here rather than silently
reaching a real application form.

Checks performed (each maps to one line of Section 15's checklist):
  - CV exists and is a real, non-empty, readable file
  - Cover letter, if the policy requires one, exists
  - Candidate identity fields are present and non-empty
  - Company/job identity on the package matches the job being applied to
  - Every required answer key the job needs has a non-empty value
  - No answer value looks like a placeholder/unfilled template artifact
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.core.config import settings
from app.services.jobs.agent_schemas import fingerprint_job
from app.services.jobs.backend_submitter import ApplicationPackage
from app.services.jobs.dedupe import canonicalize_url

_PLACEHOLDER_PATTERNS = [
    re.compile(r"\{\{.*?\}\}"),  # {{template_var}}
    re.compile(r"\[.*?insert.*?\]", re.IGNORECASE),
    re.compile(r"\bTODO\b"),
    re.compile(r"\bLOREM IPSUM\b", re.IGNORECASE),
    re.compile(r"\b(?:manual review required|unknown|not provided)\b", re.IGNORECASE),
    re.compile(r"^\s*$"),  # blank
]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


def _looks_like_placeholder(value: str) -> bool:
    return any(p.search(value) for p in _PLACEHOLDER_PATTERNS)


def _validate_pdf(path: Path, label: str, errors: list[str]) -> None:
    """Verify that a generated artifact is a readable, non-empty PDF."""

    if path.suffix.casefold() != ".pdf":
        errors.append(f"{label} must have a .pdf extension")
        return
    try:
        max_bytes = max(1_024, int(getattr(settings, "MAX_PACKAGE_BYTES", 25 * 1024 * 1024)))
        if path.stat().st_size > max_bytes:
            errors.append(f"{label} exceeds the package size limit")
            return
        with path.open("rb") as stream:
            header = stream.read(1024)
        if b"%PDF-" not in header:
            errors.append(f"{label} is not a PDF document")
            return
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            errors.append(f"{label} is encrypted and cannot be validated")
            return
        if len(reader.pages) < 1:
            errors.append(f"{label} contains no pages")
    except Exception:
        errors.append(f"{label} is not a readable PDF document")


def validate_package(
    package: ApplicationPackage,
    required_answer_keys: list[str] | None = None,
    require_cover_letter: bool = False,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # 1. CV exists, is a real file, and isn't empty.
    cv_path = Path(package.cv_path) if package.cv_path else None
    if not cv_path or not cv_path.exists():
        errors.append(f"CV file does not exist: {package.cv_path!r}")
    elif not cv_path.is_file():
        errors.append("CV path is not a regular file")
    elif cv_path.stat().st_size == 0:
        errors.append(f"CV file is empty: {package.cv_path}")
    else:
        _validate_pdf(cv_path, "CV", errors)

    # 2. Cover letter, if policy requires it.
    if require_cover_letter:
        cl_path = Path(package.cover_letter_path) if package.cover_letter_path else None
        if not cl_path or not cl_path.exists():
            errors.append("Cover letter is required by policy but missing")
        elif not cl_path.is_file():
            errors.append("Cover letter path is not a regular file")
        elif cl_path.stat().st_size == 0:
            errors.append(f"Cover letter file is empty: {package.cover_letter_path}")
        else:
            _validate_pdf(cl_path, "Cover letter", errors)

    # 3. Candidate identity present.
    personal = (package.profile or {}).get("personal", {}) or {}
    for field_name in ("first_name", "last_name", "email"):
        value = personal.get(field_name)
        if not value or not str(value).strip():
            errors.append(f"Candidate profile missing required field: {field_name}")
    email = str(personal.get("email") or "").strip()
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        errors.append("Candidate profile email is invalid")

    # 4. Company/job identity sanity -- the job on the package must
    #    actually have a company and title, or we'd be submitting a
    #    document with no idea what it's for.
    if not package.job.company or package.job.company.strip().casefold() in {"unknown", "n/a", ""}:
        errors.append("Job company is missing or unknown")
    if not package.job.title or package.job.title.strip().casefold() in {"unknown", "n/a", ""}:
        errors.append("Job title is missing or unknown")
    if package.expected_fingerprint:
        actual_fingerprint = package.job.fingerprint or fingerprint_job(
            package.job.company,
            package.job.title,
            package.job.location,
            url=canonicalize_url(package.job.application_url or package.job.original_url),
        )
        if actual_fingerprint != package.expected_fingerprint:
            errors.append("Application package is bound to a different job")

    # 5. Every required answer key has a non-empty, non-placeholder value.
    for key in required_answer_keys or []:
        value = package.answers.get(key)
        if value is None or not str(value).strip():
            errors.append(f"Required answer '{key}' is missing or empty")
        elif _looks_like_placeholder(str(value)):
            errors.append(f"Answer for '{key}' looks like an unfilled placeholder: {value!r}")

    # 6. Scan all supplied answers for placeholder artifacts even if not
    #    strictly required -- catches template bugs before they reach a
    #    real form.
    for key, value in package.answers.items():
        if (
            value
            and _looks_like_placeholder(str(value))
            and key not in (required_answer_keys or [])
        ):
            warnings.append(f"Answer for '{key}' looks like a template placeholder: {value!r}")

    return ValidationResult(valid=(len(errors) == 0), errors=errors, warnings=warnings)


__all__ = ["ValidationResult", "validate_package"]
