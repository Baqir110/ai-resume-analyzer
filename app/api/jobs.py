"""Job discovery and AI-driven application endpoints."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import (
    UnsafeInputError,
    require_api_key,
    validate_job_url,
    validate_local_file,
    validate_pdf_file,
)
from app.services.jobs.browser_use_applier import ApplicationResult, BrowserUseApplier
from app.services.jobs.full_pipeline import run_full_application
from app.services.jobs.job_metadata import fetch_job_metadata
from app.services.parsing.resume_parser import extract_text_from_file

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(require_api_key)],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPLICATIONS_LOG = PROJECT_ROOT / "data" / "job_applications.jsonl"
PROFILE_PATH = Path(settings.APPLICANT_PROFILE_PATH)
if not PROFILE_PATH.is_absolute():
    PROFILE_PATH = PROJECT_ROOT / PROFILE_PATH
_PROFILE_WRITE_LOCK = threading.RLock()


def _validate_job_url(value: str) -> str:
    try:
        return validate_job_url(value, resolve_dns=True)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_upload_path(value: str, *, cover_letter: bool = False) -> str:
    try:
        path = validate_local_file(
            value,
            allowed_extensions={".pdf"},
        )
        path = validate_pdf_file(path)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Keep the check explicit so adding another upload policy later cannot
    # accidentally broaden the accepted document types.
    if not cover_letter and path.suffix.casefold() != ".pdf":
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")
    return str(path)


def _validate_upload_filename(filename: str | None) -> str:
    name = (filename or "").strip()
    suffix = Path(name).suffix.casefold()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=400, detail="Resume must be a PDF, DOCX, or TXT file")
    if any(ord(char) < 32 for char in name):
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    return name


def _public_generated_path(value: object) -> str | None:
    """Avoid exposing absolute server paths in API responses."""

    if not value:
        return None
    try:
        path = Path(str(value)).resolve()
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except (OSError, ValueError):
        return Path(str(value)).name


def _safe_error(exc: Exception) -> str:
    """Return a bounded, non-sensitive API error message."""

    text = str(exc).strip().replace("\n", " ")
    return (text[:300] + "...") if len(text) > 300 else (text or "Operation failed")


class ApplyRequest(BaseModel):
    job_url: str
    resume_path: str
    cover_letter_path: Optional[str] = None
    why_this_company: str = Field(default="", max_length=10_000)
    company_name: str = Field(default="", max_length=300)
    role_title: str = Field(default="", max_length=300)
    max_steps: int = Field(default=40, ge=1, le=100)
    headless: bool = True
    agent_llm: str = "auto"
    auto_submit: bool = False
    handle_email_verification: bool = False
    verification_timeout: int = Field(default=120, ge=0, le=900)


class ApplyResponse(BaseModel):
    success: bool
    job_url: str
    steps_taken: int
    paused_for_human: bool = False
    captcha_encountered: bool = False
    submitted: bool = False
    verification_required: bool = False
    verification_handled: bool = False
    confirmation_message: str = ""
    error_message: Optional[str] = None
    final_url: Optional[str] = None
    history_summary: list[str] = Field(default_factory=list)


class FetchJdRequest(BaseModel):
    job_url: str


class AnswerQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    answer: str = Field(min_length=1, max_length=10_000)


def _log_application(req: ApplyRequest, result: ApplicationResult) -> None:
    APPLICATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_url": req.job_url[:4096],
        "company": req.company_name[:300],
        "role": req.role_title[:300],
        "success": result.success,
        "captcha": result.captcha_encountered,
        "submitted": result.submitted,
        "verification_required": result.verification_required,
        "verification_handled": result.verification_handled,
        "confirmation": result.confirmation_message[:1000],
        "steps": result.steps_taken,
        "final_url": (result.final_url or "")[:4096],
        "error": (result.error_message or "")[:1000],
    }
    from app.core.event_log import _write_record

    _write_record(APPLICATIONS_LOG, entry)


@router.post("/fetch-jd")
async def fetch_jd(req: FetchJdRequest):
    """Fetch JD, company, and role from a public job posting URL."""
    safe_url = _validate_job_url(req.job_url)
    try:
        meta = await fetch_job_metadata(safe_url)
    except Exception as exc:
        logger.warning("fetch-jd failed for %s: %s", safe_url, _safe_error(exc))
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from exc

    return {
        "ok": True,
        "job_url": req.job_url,
        "company": meta.company,
        "role": meta.role,
        "location": meta.location,
        "source_ats": meta.source_ats,
        "jd_text": meta.jd_text,
        "chars": len(meta.jd_text),
    }


@router.post("/apply", response_model=ApplyResponse)
async def apply_to_job(req: ApplyRequest):
    """Fill a form using an allowlisted local PDF; submission is opt-in."""
    safe_url = _validate_job_url(req.job_url)
    safe_resume = _validate_upload_path(req.resume_path)
    safe_cover = (
        _validate_upload_path(req.cover_letter_path, cover_letter=True)
        if req.cover_letter_path
        else None
    )
    try:
        applier = BrowserUseApplier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="data/applicant_profile.yaml is missing.",
        ) from exc

    try:
        result = await applier.apply(
            job_url=safe_url,
            resume_path=safe_resume,
            cover_letter_path=safe_cover,
            why_this_company=req.why_this_company,
            company_name=req.company_name,
            role_title=req.role_title,
            max_steps=req.max_steps,
            headless=req.headless,
            agent_llm=req.agent_llm,
            auto_submit=req.auto_submit,
            handle_email_verification=req.handle_email_verification,
            verification_timeout=req.verification_timeout,
        )
    except Exception as exc:
        logger.exception("apply_to_job crashed")
        raise HTTPException(status_code=500, detail="Application worker failed") from exc

    _log_application(req, result)

    return ApplyResponse(
        success=result.success,
        job_url=result.job_url,
        steps_taken=result.steps_taken,
        paused_for_human=result.paused_for_human,
        captcha_encountered=result.captcha_encountered,
        submitted=result.submitted,
        verification_required=result.verification_required,
        verification_handled=result.verification_handled,
        confirmation_message=result.confirmation_message,
        error_message=result.error_message,
        final_url=result.final_url,
        history_summary=result.history_summary,
    )


@router.post("/apply-with-tailored-cv")
async def apply_with_tailored_cv(
    job_url: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form(""),
    role_title: str = Form(""),
    why_this_company: str = Form(""),
    cover_letter_template: str = Form("classic_professional"),
    cover_letter_tone: str = Form("formal"),
    layout_style: str = Form("auto"),
    max_steps: int = Form(40, ge=1, le=100),
    provider: str | None = Form(None),
    model_name: str | None = Form(None),
    route_mode: str | None = Form(None),
    agent_llm: str = Form("auto"),
    auto_submit: bool = Form(False),
    handle_email_verification: bool = Form(False),
    verification_timeout: int = Form(120, ge=0, le=900),
    resume_file: UploadFile = File(...),
):
    """Tailor a CV and fill a form; final submission is opt-in."""
    safe_url = _validate_job_url(job_url)
    if len(job_description) > settings.MAX_JOB_DESCRIPTION_CHARS:
        raise HTTPException(status_code=413, detail="Job description is too large")
    _validate_upload_filename(resume_file.filename)

    try:
        resume_bytes = await resume_file.read(settings.MAX_FILE_SIZE + 1)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read upload",
        ) from exc
    if len(resume_bytes) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded resume is too large")
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty.")

    try:
        await resume_file.seek(0)
        resume_text = await extract_text_from_file(resume_file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from resume",
        ) from exc

    if not resume_text.strip():
        raise HTTPException(
            status_code=400, detail="No text could be extracted from the uploaded resume."
        )

    job: dict = {
        "title": role_title or "Unknown Role",
        "company": company_name or "Unknown Company",
        "description": job_description,
        "job_url": safe_url,
    }

    try:
        result = await run_full_application(
            job=job,
            resume_text=resume_text,
            model_name=model_name or "gpt-5.6-luna",
            agent_llm=agent_llm,
            auto_submit=auto_submit,
            provider=provider,
            route_mode=route_mode,
            max_steps=max_steps,
            verification_timeout=verification_timeout,
        )
    except Exception as exc:
        logger.exception("Tailored application pipeline failed")
        raise HTTPException(status_code=500, detail="Application pipeline failed") from exc

    pipeline_ok = result.get("success", False)
    app = result.get("browser_application")

    def _serialise_app(a: Optional[ApplicationResult]) -> Optional[dict]:
        if a is None:
            return None
        return {
            "success": a.success,
            "steps_taken": a.steps_taken,
            "paused_for_human": getattr(a, "paused_for_human", False),
            "captcha_encountered": a.captcha_encountered,
            "submitted": a.submitted,
            "verification_required": a.verification_required,
            "verification_handled": a.verification_handled,
            "confirmation_message": a.confirmation_message,
            "final_url": a.final_url,
            "history_summary": a.history_summary,
        }

    if not pipeline_ok:
        return {
            "ok": False,
            "stage_failed": result.get("status") or "unknown",
            "error": "Pipeline failed; check browser_application for details.",
            "cv_path": _public_generated_path(result.get("cv_path")),
            "cover_letter_path": _public_generated_path(result.get("cover_letter_path")),
            "pdf_bytes": None,
            "application": _serialise_app(app),
        }

    return {
        "ok": True,
        "stage_failed": None,
        "error": None,
        "cv_path": _public_generated_path(result.get("cv_path")),
        "cover_letter_path": _public_generated_path(result.get("cover_letter_path")),
        "pdf_bytes": None,
        "application": _serialise_app(app),
    }


@router.post("/answer-question")
async def answer_question(req: AnswerQuestionRequest):
    """Save an answer for a missing question into data/applicant_profile.yaml."""
    if PROFILE_PATH.resolve() == (PROJECT_ROOT / "data" / "applicant_profile.yaml").resolve():
        raise HTTPException(
            status_code=503,
            detail="Configure APPLICANT_PROFILE_PATH before saving answers",
        )
    if not PROFILE_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="data/applicant_profile.yaml missing.",
        )

    try:
        with _PROFILE_WRITE_LOCK:
            with PROFILE_PATH.open("r", encoding="utf-8") as f:
                profile = yaml.safe_load(f) or {}
            if not isinstance(profile, dict):
                raise ValueError("Applicant profile must be a mapping")

            free_text = profile.setdefault("free_text", {})
            if not isinstance(free_text, dict):
                raise ValueError("Applicant profile free_text must be a mapping")
            free_text[req.question.strip()] = req.answer.strip()

            fd, temporary_name = tempfile.mkstemp(
                prefix=f"{PROFILE_PATH.name}.",
                suffix=".tmp",
                dir=str(PROFILE_PATH.parent),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary_name, PROFILE_PATH)
            finally:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

        return {
            "ok": True,
            "message": "Answer saved.",
            "question": req.question.strip(),
        }
    except Exception as exc:
        logger.exception("Failed to update applicant profile")
        raise HTTPException(status_code=500, detail="Could not save answer") from exc


class AutoSearchApplyRequest(BaseModel):
    search_term: str = Field(default="Software Engineer", min_length=1, max_length=200)
    location: str = Field(default="Germany", min_length=1, max_length=200)
    resume_path: str = "data/resume.pdf"
    max_applications: int = Field(default=3, ge=1, le=20)
    auto_submit: bool = False


@router.post("/auto-discover-and-apply")
async def auto_discover_and_apply(req: AutoSearchApplyRequest):
    """Automatically find matching jobs in Germany and run applications end-to-end."""
    try:
        safe_resume = _validate_upload_path(req.resume_path)
        from app.services.jobs.auto_runner import run_automated_job_hunting

        results = await run_automated_job_hunting(
            search_term=req.search_term,
            location=req.location,
            resume_path=safe_resume,
            max_applications=req.max_applications,
            auto_submit=req.auto_submit,
        )

        return {
            "ok": True,
            "processed": len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Auto discover and apply failed")
        raise HTTPException(status_code=500, detail="Automated job search failed") from exc


@router.get("/applications")
async def list_applications(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    if not APPLICATIONS_LOG.exists():
        return {"applications": []}
    try:
        lines = APPLICATIONS_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"applications": []}
    rows = []
    for line in lines[-max(1, limit) :]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"applications": list(reversed(rows))}
