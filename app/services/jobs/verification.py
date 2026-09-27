"""Conservative submission verification.

A browser agent's prose and a URL string are not independent proof that an
application was accepted.  Only a structured confirmation identifier or the
applier's independently collected page-state classification is accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    verified: bool
    evidence: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"verified": self.verified, "evidence": self.evidence, "reason": self.reason}


def verify_submission(pipeline_result: dict[str, Any]) -> VerificationResult:
    """Verify a claimed submission using structured, independent evidence."""

    browser = pipeline_result.get("browser_application", {}) or {}
    claimed = bool(pipeline_result.get("submitted") or browser.get("submitted"))
    if not claimed:
        return VerificationResult(verified=False, reason="No submission was claimed")

    evidence: list[str] = []

    # Backend form submitters can provide a server-issued identifier.  A
    # generic confirmation string is intentionally not enough.
    application_id = (
        pipeline_result.get("application_id")
        or pipeline_result.get("confirmation_id")
        or browser.get("application_id")
        or browser.get("confirmation_id")
    )
    if application_id:
        evidence.append(f"structured application identifier: {str(application_id)[:120]}")

    # BrowserUseApplier sets this only after inspecting the current page state
    # (confirmation DOM or a validated confirmation-page transition).  Model
    # history and final_result text never set this flag by themselves.
    if (
        browser.get("submission_verified") is True
        and browser.get("submission_status") == "submitted"
    ):
        evidence.append("independent browser confirmation-page evidence")

    if evidence:
        return VerificationResult(verified=True, evidence=evidence)

    return VerificationResult(
        verified=False,
        reason=(
            "Submission was claimed but no structured identifier or independent "
            "browser confirmation evidence was available; agent prose and URL "
            "patterns are not trusted."
        ),
    )


__all__ = ["VerificationResult", "verify_submission"]
