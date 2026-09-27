# app/services/jobs/agent_schemas.py
"""Shared data model for the autonomous job agent: normalized jobs,
match results, and the application state machine.

Kept dependency-free (stdlib only) so it can be imported and unit
tested without pulling in jobspy / playwright / browser-use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Normalized job
# ---------------------------------------------------------------------------


@dataclass
class NormalizedJob:
    """Unified job schema. Every discovery source must map into this."""

    job_id: str
    title: str
    company: str
    location: str = ""
    country: str = ""
    remote_type: str = ""  # onsite / hybrid / remote
    employment_type: str = ""  # full_time / part_time / contract / internship
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: str = ""
    description: str = ""
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    preferred_requirements: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    experience_required_years: Optional[float] = None
    education_required: str = ""
    work_authorization: str = ""
    visa_sponsorship_offered: Optional[bool] = None
    application_url: str = ""
    source: str = (
        ""  # jobspy_indeed / jobspy_linkedin / greenhouse / lever / ashby / workable / web_search
    )
    source_job_id: str = ""
    original_url: str = ""
    discovered_at: str = field(default_factory=_now_iso)
    deadline: str = ""
    company_url: str = ""
    status: str = "DISCOVERED"
    fingerprint: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("raw", None)
        return d


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    overall_score: float
    technical_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    location_score: float = 0.0
    language_score: float = 0.0
    seniority_score: float = 0.0
    matched: list[str] = field(default_factory=list)
    partially_matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    decision: str = "SKIP"  # APPLY | SKIP
    priority: str = "SKIP"  # HIGH_PRIORITY | GOOD_MATCH | LOW_PRIORITY | SKIP
    skip_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Application state machine
# ---------------------------------------------------------------------------


class ApplicationState(str, Enum):
    DISCOVERED = "DISCOVERED"
    ANALYZING = "ANALYZING"
    MATCHED = "MATCHED"
    SKIPPED = "SKIPPED"
    DUPLICATE = "DUPLICATE"
    PREPARING = "PREPARING"
    CV_GENERATED = "CV_GENERATED"
    COVER_LETTER_GENERATED = "COVER_LETTER_GENERATED"
    READY_FOR_APPLICATION = "READY_FOR_APPLICATION"
    APPLYING = "APPLYING"
    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
    CAPTCHA_REQUIRED = "CAPTCHA_REQUIRED"
    READY_TO_SUBMIT = "READY_TO_SUBMIT"
    SUBMITTED = "SUBMITTED"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    RETRYING = "RETRYING"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    VERIFIED = "VERIFIED"
    INTERVIEW = "INTERVIEW"
    REJECTED = "REJECTED"
    OFFER = "OFFER"
    WITHDRAWN = "WITHDRAWN"


# Allowed transitions. Anything not listed here is rejected by the
# state machine so a bug can't silently skip steps (e.g. DISCOVERED
# straight to SUBMITTED).
VALID_TRANSITIONS: dict[ApplicationState, set[ApplicationState]] = {
    ApplicationState.DISCOVERED: {
        ApplicationState.ANALYZING,
        ApplicationState.DUPLICATE,
    },
    ApplicationState.ANALYZING: {
        ApplicationState.MATCHED,
        ApplicationState.SKIPPED,
    },
    ApplicationState.MATCHED: {
        ApplicationState.PREPARING,
        ApplicationState.SKIPPED,
    },
    ApplicationState.PREPARING: {
        ApplicationState.CV_GENERATED,
        ApplicationState.READY_FOR_APPLICATION,
        ApplicationState.SUBMISSION_FAILED,
        ApplicationState.CAPTCHA_REQUIRED,
    },
    ApplicationState.CV_GENERATED: {
        ApplicationState.COVER_LETTER_GENERATED,
        ApplicationState.READY_FOR_APPLICATION,
        ApplicationState.SUBMISSION_FAILED,
    },
    ApplicationState.COVER_LETTER_GENERATED: {
        ApplicationState.READY_FOR_APPLICATION,
        ApplicationState.SUBMISSION_FAILED,
    },
    ApplicationState.READY_FOR_APPLICATION: {
        ApplicationState.APPLYING,
        ApplicationState.SUBMISSION_FAILED,
        ApplicationState.WITHDRAWN,
    },
    ApplicationState.APPLYING: {
        ApplicationState.WAITING_FOR_INPUT,
        ApplicationState.CAPTCHA_REQUIRED,
        ApplicationState.READY_TO_SUBMIT,
        ApplicationState.SUBMITTED,
        ApplicationState.SUBMISSION_FAILED,
    },
    ApplicationState.WAITING_FOR_INPUT: {
        ApplicationState.APPLYING,
        ApplicationState.SUBMISSION_FAILED,
    },
    ApplicationState.CAPTCHA_REQUIRED: {
        ApplicationState.RETRYING,
        ApplicationState.SUBMISSION_FAILED,
    },
    ApplicationState.READY_TO_SUBMIT: {
        ApplicationState.SUBMITTED,
        ApplicationState.SUBMISSION_FAILED,
        ApplicationState.RETRYING,
        ApplicationState.WITHDRAWN,
    },
    ApplicationState.SUBMITTED: {
        ApplicationState.VERIFIED,
        ApplicationState.VERIFICATION_FAILED,
    },
    ApplicationState.SUBMISSION_FAILED: {
        ApplicationState.RETRYING,
        ApplicationState.WITHDRAWN,
    },
    ApplicationState.RETRYING: {
        ApplicationState.PREPARING,
        ApplicationState.APPLYING,
        ApplicationState.SUBMISSION_FAILED,
    },
    ApplicationState.VERIFICATION_FAILED: {
        ApplicationState.RETRYING,
        ApplicationState.SUBMITTED,
        ApplicationState.VERIFIED,
        ApplicationState.WITHDRAWN,
    },
    ApplicationState.VERIFIED: {
        ApplicationState.INTERVIEW,
        ApplicationState.REJECTED,
    },
    ApplicationState.INTERVIEW: {
        ApplicationState.OFFER,
        ApplicationState.REJECTED,
    },
    ApplicationState.SKIPPED: set(),
    ApplicationState.DUPLICATE: set(),
    ApplicationState.REJECTED: set(),
    ApplicationState.OFFER: {ApplicationState.WITHDRAWN},
    ApplicationState.WITHDRAWN: set(),
}


class InvalidTransition(Exception):
    """Raised when the agent tries to skip or reorder pipeline stages."""


def assert_valid_transition(from_state: ApplicationState, to_state: ApplicationState) -> None:
    if to_state not in VALID_TRANSITIONS.get(from_state, set()):
        raise InvalidTransition(f"{from_state} -> {to_state} is not allowed")


def sha256_fingerprint(*parts: str) -> str:
    """Return the complete SHA-256 digest for an unambiguous key.

    Fingerprints are persisted as durable queue identifiers, so truncating
    them creates an avoidable collision surface and makes the same logical
    job hash differently depending on which helper produced it.
    """
    key = "|".join(str(part) for part in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def fingerprint_job(company: str, title: str, location: str, url: str = "") -> str:
    """Stable SHA-256 fingerprint for a normalized job.

    Supplying a canonical application URL makes the URL the primary key.
    Without one, company/title/location form the stable cross-board key.
    """
    if url:
        return sha256_fingerprint(url.strip())
    return sha256_fingerprint(
        "|".join(
            (
                _normalize_token(company),
                _normalize_token(title),
                _normalize_token(location),
            )
        )
    )


def _normalize_token(value: str) -> str:
    return " ".join((value or "").lower().split())


__all__ = [
    "NormalizedJob",
    "MatchResult",
    "ApplicationState",
    "VALID_TRANSITIONS",
    "InvalidTransition",
    "assert_valid_transition",
    "sha256_fingerprint",
    "fingerprint_job",
]
