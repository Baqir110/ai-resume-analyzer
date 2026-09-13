"""Job discovery and AI-driven application endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.jobs.browser_use_applier import ApplicationResult, BrowserUseApplier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPLICATIONS_LOG = PROJECT_ROOT / "data" / "job_applications.jsonl"


class ApplyRequest(BaseModel):
    job_url: str
    resume_path: str
    cover_letter_path: Optional[str] = None
    why_this_company: str = ""
    company_name: str = ""
    role_title: str = ""
    max_steps: int = 40
    headless: bool = False


class ApplyResponse(BaseModel):
    success: bool
    job_url: str
    steps_taken: int
    paused_for_human: bool = False
    captcha_encountered: bool = False
    error_message: Optional[str] = None
    final_url: Optional[str] = None
    history_summary: list[str] = []


def _log_application(req: ApplyRequest, result: ApplicationResult) -> None:
    """Append one line per attempt to the applications log."""
    APPLICATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_url": req.job_url,
        "company": req.company_name,
        "role": req.role_title,
        "success": result.success,
        "captcha": result.captcha_encountered,
        "steps": result.steps_taken,
        "final_url": result.final_url,
        "error": result.error_message,
    }
    with APPLICATIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@router.post("/apply", response_model=ApplyResponse)
async def apply_to_job(req: ApplyRequest):
    """
    Fill a job application form with Browser Use.

    Does NOT click Submit. The browser stays open (headless=False)
    so you can review the filled form and submit it yourself.
    """
    try:
        applier = BrowserUseApplier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "data/applicant_profile.yaml is missing. " "Create it before using /jobs/apply."
            ),
        ) from exc

    try:
        result = await applier.apply(
            job_url=req.job_url,
            resume_path=req.resume_path,
            cover_letter_path=req.cover_letter_path,
            why_this_company=req.why_this_company,
            company_name=req.company_name,
            role_title=req.role_title,
            max_steps=req.max_steps,
            headless=req.headless,
        )
    except Exception as exc:
        logger.exception("apply_to_job crashed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _log_application(req, result)

    return ApplyResponse(
        success=result.success,
        job_url=result.job_url,
        steps_taken=result.steps_taken,
        paused_for_human=result.paused_for_human,
        captcha_encountered=result.captcha_encountered,
        error_message=result.error_message,
        final_url=result.final_url,
        history_summary=result.history_summary,
    )


@router.get("/applications")
async def list_applications(limit: int = 50):
    """Return the most recent application attempts."""
    if not APPLICATIONS_LOG.exists():
        return {"applications": []}

    lines = APPLICATIONS_LOG.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-max(1, limit) :]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"applications": list(reversed(rows))}
