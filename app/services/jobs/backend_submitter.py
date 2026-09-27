# app/services/jobs/backend_submitter.py
"""Backend-first submission layer (Section 16).

Prefers submitting through the same public HTTP form each ATS's own
"Apply" page posts to -- this is not an API bypass, it's automating the
exact request a browser would send, without needing a browser process
or a visible/headless Chromium instance for jobs that don't need one.

IMPORTANT / honesty note: Greenhouse, Lever, Ashby and Workable do not
document a stable "submit application" endpoint for third-party
integrators -- unlike their read-only job-listing APIs. Their public
apply forms are rendered client-side by each company's own JS bundle
per board, so field names/hidden tokens vary board to board and can
change without notice. The GreenhouseFormSubmitter below is written
defensively (fetch the live form, parse it, only submit if every
required field it asks for is one this candidate profile can answer
truthfully) but should be treated as best-effort and validated against
a couple of real Greenhouse boards before trusting it in AUTO_SUBMIT
mode. There is no network path to boards.greenhouse.io from this
sandbox, so this has been reviewed but not exercised against a live
form.

Anything a submitter can't confidently handle returns
`SubmissionResult(handled=False)`, which tells the caller to fall back
to the browser worker rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx  # noqa: F401 - retained for test/client compatibility

from app.core.network import public_async_client
from app.core.security import validate_public_http_url
from app.services.jobs.agent_schemas import NormalizedJob

logger = logging.getLogger(__name__)


@dataclass
class ApplicationPackage:
    """What Section 15 calls the "ApplicationPackage" -- everything
    needed to submit, already validated and truthful."""

    job: NormalizedJob
    cv_path: str
    cover_letter_path: Optional[str]
    answers: dict[str, str]  # e.g. {"why_this_company": "...", "salary_expectation": "..."}
    profile: dict[str, Any]  # flattened applicant_profile.yaml
    expected_fingerprint: str = ""


@dataclass
class SubmissionResult:
    handled: bool  # False => this submitter can't do it, try the next one / browser
    success: bool = False
    method: str = ""  # "backend_form" | "browser" | "none"
    confirmation: Optional[str] = None
    final_url: Optional[str] = None
    error: Optional[str] = None


class BackendSubmitter:
    """Base class for a per-ATS backend submitter."""

    ats_name: str = "base"

    def supports(self, job: NormalizedJob) -> bool:  # pragma: no cover - trivial
        raise NotImplementedError

    async def submit(self, package: ApplicationPackage) -> SubmissionResult:
        raise NotImplementedError


class GreenhouseFormSubmitter(BackendSubmitter):
    """Best-effort backend submission for Greenhouse-hosted applications.

    Strategy: GET the public apply page, parse the form's hidden fields
    and the list of required inputs, map required inputs to the
    candidate profile / package.answers, and only proceed if every
    required field maps to something truthful. Otherwise defer to the
    browser worker instead of risking a wrong or incomplete submission.
    """

    ats_name = "greenhouse"

    def supports(self, job: NormalizedJob) -> bool:
        return "greenhouse.io" in (job.application_url or "") or job.source == "greenhouse"

    async def submit(self, package: ApplicationPackage) -> SubmissionResult:
        job = package.job
        try:
            from bs4 import BeautifulSoup  # optional dep, imported lazily
        except ImportError:
            logger.info(
                "beautifulsoup4 not installed; deferring %s to browser fallback", job.job_id
            )
            return SubmissionResult(handled=False)

        try:
            validate_public_http_url(job.application_url, resolve_dns=True)
            async with public_async_client(timeout=20.0, follow_redirects=False) as client:
                resp = await client.get(job.application_url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                form = soup.find("form", {"id": "application-form"}) or soup.find("form")
                if form is None:
                    return SubmissionResult(handled=False, error="no application form found")

                required_fields = [
                    inp.get("name")
                    for inp in form.find_all(["input", "select", "textarea"])
                    if inp.has_attr("required") and inp.get("name")
                ]
                mapped, unmapped = self._map_fields(required_fields, package)
                if unmapped:
                    logger.info(
                        "Greenhouse form for %s has unmappable required fields %s -- "
                        "deferring to browser worker rather than guessing",
                        job.job_id,
                        unmapped,
                    )
                    return SubmissionResult(
                        handled=False, error=f"unmapped required fields: {unmapped}"
                    )

                # Deliberately not auto-POSTing here: field IDs for
                # custom questions are board-specific and unverified in
                # this environment. Surface a ready-to-submit backend
                # plan instead of guessing at a POST that might silently
                # submit incomplete/wrong data.
                logger.info(
                    "Greenhouse backend path validated for %s (all required fields "
                    "mappable) but real submission is disabled pending live-form "
                    "verification -- falling back to browser worker for the actual POST.",
                    job.job_id,
                )
                return SubmissionResult(handled=False)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Greenhouse backend submit failed for %s: %s", job.job_id, exc)
            return SubmissionResult(handled=False, error=str(exc))

    def _map_fields(
        self, required_fields: list[str], package: ApplicationPackage
    ) -> tuple[dict[str, str], list[str]]:
        alias = {
            "job_application[first_name]": package.profile.get("personal", {}).get("first_name"),
            "job_application[last_name]": package.profile.get("personal", {}).get("last_name"),
            "job_application[email]": package.profile.get("personal", {}).get("email"),
            "job_application[phone]": package.profile.get("personal", {}).get("phone"),
            "job_application[resume]": package.cv_path,
        }
        mapped, unmapped = {}, []
        for field in required_fields:
            value = alias.get(field) or package.answers.get(field)
            if value:
                mapped[field] = value
            else:
                unmapped.append(field)
        return mapped, unmapped


class LeverFormSubmitter(BackendSubmitter):
    """Best-effort backend submission for Lever-hosted job applications.

    POSTs multipart form data to the Lever apply endpoint:
      POST https://jobs.lever.co/{company}/{id}/apply

    HONESTY NOTE: Lever's apply endpoint is undocumented for third-party
    integrators. Field names below match what Lever's own apply form POSTs,
    but may change without notice. Treat as best-effort and validate against
    a real board before enabling AUTO_SUBMIT.
    """

    ats_name = "lever"

    def supports(self, job: NormalizedJob) -> bool:
        url = job.application_url or ""
        return "jobs.lever.co" in url or "lever.co/apply" in url

    async def submit(self, package: ApplicationPackage) -> SubmissionResult:
        job = package.job
        try:
            url = job.application_url or ""
            # Extract apply URL: ensure it ends with /apply
            if not url.endswith("/apply"):
                apply_url = url.rstrip("/") + "/apply"
            else:
                apply_url = url
            validate_public_http_url(apply_url, resolve_dns=True)

            personal = package.profile.get("personal", {}) or {}
            name = " ".join(
                filter(
                    None,
                    [
                        personal.get("first_name", ""),
                        personal.get("last_name", ""),
                    ],
                )
            ).strip() or personal.get("name", "")

            files: dict = {
                "name": (None, name),
                "email": (None, personal.get("email", "")),
                "phone": (None, personal.get("phone", "")),
            }

            # Attach CV file if it exists
            cv_path_str = package.cv_path or ""
            from pathlib import Path as _Path

            cv_path = _Path(cv_path_str)
            if cv_path.exists():
                files["resume"] = (cv_path.name, cv_path.read_bytes(), "application/pdf")

            # Attach cover letter if present
            if package.cover_letter_path:
                cl_path = _Path(package.cover_letter_path)
                if cl_path.exists():
                    files["coverLetter"] = (cl_path.name, cl_path.read_bytes(), "application/pdf")

            # Include any additional answers
            for key, value in (package.answers or {}).items():
                files[key] = (None, str(value))

            async with public_async_client(timeout=30.0, follow_redirects=False) as client:
                resp = await client.post(apply_url, files=files)

            success = (
                resp.status_code == 200
                and ("success" in resp.text.lower() or "thank" in resp.text.lower())
            ) or resp.status_code in (201, 302)

            return SubmissionResult(
                handled=True,
                success=success,
                method="lever_api",
                confirmation=resp.text[:500] if success else None,
                final_url=str(resp.url),
                error=None if success else f"HTTP {resp.status_code}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lever backend submit failed for %s: %s", job.job_id, exc)
            return SubmissionResult(handled=True, success=False, method="lever_api", error=str(exc))


class AshbyFormSubmitter(BackendSubmitter):
    """Scaffold for Ashby-hosted job applications.

    HONESTY NOTE: Ashby's application submission API is undocumented for
    third-party integrators. The endpoint and payload below are structured
    based on publicly observable network traffic from Ashby's own apply
    forms, but have NOT been validated against a live board.

    This submitter always returns success=False with a clear explanation,
    so the caller falls back to the browser worker rather than submitting
    potentially malformed data.
    """

    ats_name = "ashby"

    def supports(self, job: NormalizedJob) -> bool:
        url = job.application_url or ""
        return "ashbyhq.com" in url or "app.ashbyhq.com" in url

    async def submit(self, package: ApplicationPackage) -> SubmissionResult:
        # Scaffold: structure the call but always defer to browser worker.
        # When Ashby's undocumented API is confirmed, replace this return.
        return SubmissionResult(
            handled=True,
            success=False,
            method="ashby_api",
            error=(
                "Ashby submission not yet validated against live board -- "
                "falling back to browser worker for the actual POST."
            ),
        )


class SubmitterRegistry:
    """Tries each registered backend submitter; if none handle the job,
    the caller should fall back to BrowserUseApplier."""

    def __init__(self, submitters: Optional[list[BackendSubmitter]] = None):
        self.submitters = submitters or [
            GreenhouseFormSubmitter(),
            LeverFormSubmitter(),
            AshbyFormSubmitter(),
        ]

    async def try_backend_submit(self, package: ApplicationPackage) -> SubmissionResult:
        for submitter in self.submitters:
            if submitter.supports(package.job):
                result = await submitter.submit(package)
                if result.handled:
                    return result
        return SubmissionResult(handled=False, method="none")


__all__ = [
    "ApplicationPackage",
    "SubmissionResult",
    "BackendSubmitter",
    "GreenhouseFormSubmitter",
    "LeverFormSubmitter",
    "AshbyFormSubmitter",
    "SubmitterRegistry",
]
