import asyncio
import functools
import io
import logging
import re
import time
from datetime import datetime, timezone

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.utils import decode_suggestions as _decode_suggestions
from app.core.config import settings
from app.core.event_log import log_event, new_request_id, set_request_id
from app.core.security import UnsafeInputError, require_api_key, validate_job_url
from app.models.schemas import AnalysisResponse
from app.services.analysis.ats_analyzer import analyze_resume_content
from app.services.bulk.bulk_analyzer import BulkAnalyzerService
from app.services.career.audit_matrix import AuditMatrixService
from app.services.career.cover_letter import CoverLetterService
from app.services.career.interview_prep import InterviewPrepService
from app.services.career.linkedin_optimizer import LinkedInOptimizerService
from app.services.cv.diff_preview import DiffPreviewService
from app.services.cv.latex_generator import (
    FactualValidationError,
    compile_latex_to_pdf,
    generate_german_latex_content,
)
from app.services.cv.optimizer import (
    auto_select_layout,
    generate_full_tailored_cv,
    optimize_resume_bullets,
    suggest_best_cv_format,
)
from app.services.cv.pdf_compiler import LaTeXCompilationError, LaTeXSourceError, PDFLayoutError
from app.services.llm import quota_tracker
from app.services.llm.provider import LOG_PATH, LLMService
from app.services.parsing.resume_parser import extract_text_from_file
from app.services.tracking.tracker import DB_PATH as TRACKER_DB_PATH
from app.services.tracking.tracker import ApplicationTrackerService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])
_PROTECTED = [Depends(require_api_key)]


# ============================================================
# PIPELINE LOGGING DECORATOR
# ============================================================


def _log_pipeline(operation: str):
    """Wrap an endpoint with pipeline_started / pipeline_completed events."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            rid = new_request_id()
            set_request_id(rid)
            started = time.perf_counter()

            _jd = kwargs.get("job_description") or ""
            if len(_jd) > settings.MAX_JOB_DESCRIPTION_CHARS:
                raise HTTPException(status_code=413, detail="Job description is too large")
            _uf = kwargs.get("resume_file")
            _fn = ""
            _ext = ""
            if _uf is not None:
                _fn = getattr(_uf, "filename", "") or ""
                if _fn:
                    _ext = _fn.rsplit(".", 1)[-1].lower()

            logger.info(
                "[%s] START req=%s jd_chars=%d file=%s", operation, rid, len(_jd), _ext or "?"
            )

            log_event(
                "pipeline",
                "pipeline_started",
                rid,
                operation=operation,
                jd_chars=len(_jd),
                file_type=_ext,
            )

            try:
                result = await fn(*args, **kwargs)
                ms = round((time.perf_counter() - started) * 1000, 1)
                logger.info("[%s] OK req=%s duration_ms=%s", operation, rid, ms)
                log_event(
                    "pipeline",
                    "pipeline_completed",
                    rid,
                    operation=operation,
                    status="success",
                    duration_ms=ms,
                )
                return result
            except HTTPException as exc:
                if isinstance(exc.__cause__, PDFLayoutError):
                    layout_error = exc.__cause__
                    exc = HTTPException(
                        status_code=422,
                        detail={
                            "code": getattr(layout_error, "code", "pdf_page_limit_exceeded"),
                            "pages": layout_error.pages,
                            "message": str(layout_error),
                        },
                    )
                ms = round((time.perf_counter() - started) * 1000, 1)
                log_event(
                    "pipeline",
                    "pipeline_failed",
                    rid,
                    operation=operation,
                    status="http_error",
                    status_code=exc.status_code,
                    duration_ms=ms,
                    error=str(exc.detail)[:300],
                )
                raise exc
            except Exception as exc:
                ms = round((time.perf_counter() - started) * 1000, 1)
                logger.error(
                    "[%s] ERROR req=%s duration_ms=%s type=%s",
                    operation,
                    rid,
                    ms,
                    type(exc).__name__,
                )
                log_event(
                    "pipeline",
                    "pipeline_failed",
                    rid,
                    operation=operation,
                    status="error",
                    duration_ms=ms,
                    error=str(exc)[:300],
                )
                raise

        return wrapper

    return decorator


# ============================================================
# REQUEST SCHEMAS
# ============================================================


class BulletDiffRequest(BaseModel):
    original_bullets: list[str]
    optimized_bullets: list[str]


class TrackerCreateRequest(BaseModel):
    company_name: str = Field(min_length=1, max_length=300)
    job_title: str = Field(min_length=1, max_length=300)
    job_url: str | None = Field(default="", max_length=4096)
    ats_score: int | None = Field(default=0, ge=0, le=100)
    status: str | None = Field(default="Saved", max_length=64)
    notes: str | None = Field(default="", max_length=10_000)


class TrackerStatusUpdate(BaseModel):
    status: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9 _-]+$")


# ============================================================
# FLAGSHIP 1: VISUAL DIFF & BULK ANALYSIS
# ============================================================


@router.post("/diff-preview")
async def diff_preview(payload: BulletDiffRequest):
    try:
        diffs = await asyncio.to_thread(
            DiffPreviewService.compare_bullet_lists,
            payload.original_bullets,
            payload.optimized_bullets,
        )
        return {"status": "success", "diffs": diffs}
    except Exception:
        logger.exception("Diff preview failed")
        raise HTTPException(status_code=500, detail="Diff preview failed")


@router.post("/analyze-bulk")
async def analyze_bulk(
    job_description: str = Form(...),
    resume_files: list[UploadFile] = File(...),
):
    if not resume_files:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")
    if len(resume_files) > 20:
        raise HTTPException(status_code=413, detail="Too many resume files")

    try:
        files_data = []
        total_size = 0
        for file in resume_files:
            if getattr(file, "size", None) is not None and int(file.size) > settings.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="Uploaded file is too large")
            content = await file.read(settings.MAX_FILE_SIZE + 1)
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="Uploaded file is too large")
            total_size += len(content)
            if total_size > settings.MAX_FILE_SIZE * 5:
                raise HTTPException(status_code=413, detail="Batch upload is too large")
            files_data.append((content, file.filename))

        ranked_candidates = await BulkAnalyzerService.process_batch(files_data, job_description)

        return {
            "status": "success",
            "total_processed": len(ranked_candidates),
            "rankings": ranked_candidates,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Bulk analysis failed")
        raise HTTPException(status_code=500, detail="Bulk analysis failed")


# ============================================================
# FLAGSHIP 2: ATS AUDIT MATRIX
# ============================================================


@router.post("/audit-matrix")
async def audit_matrix_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    file_ext = resume_file.filename.split(".")[-1] if resume_file.filename else "pdf"
    audit_result = await asyncio.to_thread(
        AuditMatrixService.run_full_audit,
        resume_text=resume_text,
        job_description=job_description,
        file_type=file_ext,
    )
    return {"status": "success", "data": audit_result}


# ============================================================
# FLAGSHIP 3: COVER LETTER & OUTREACH
# ============================================================


@router.post("/generate-cover-letter")
async def generate_cover_letter_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    company_name: str | None = Form("Target Company"),
    tone: str | None = Form("formal"),
    template: str | None = Form("classic_professional"),
    provider: str | None = Form(None),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    result = await asyncio.to_thread(
        CoverLetterService.generate_cover_letter_and_outreach,
        resume_text=resume_text,
        job_description=job_description,
        company_name=company_name or "Target Company",
        tone=tone or "formal",
        template=template or "classic_professional",
        provider=provider,
    )
    return {"status": "success", "data": result}


@router.post("/generate-cover-letter-pdf")
@_log_pipeline("generate-cover-letter-pdf")
async def generate_cover_letter_pdf_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    company_name: str | None = Form("Target Company"),
    tone: str | None = Form("formal"),
    template: str | None = Form("classic_professional"),
    provider: str | None = Form(None),
    model_name: str | None = Form(None),
    route_mode: str | None = Form(None),
    language: str | None = Form("auto"),
):
    from app.services.career.cover_letter_pdf import (
        compile_cover_letter_pdf,
        generate_cover_letter_latex,
    )

    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    try:
        latex_code, lang_used = await asyncio.to_thread(
            generate_cover_letter_latex,
            resume_text=resume_text,
            job_description=job_description,
            company_name=company_name or "Target Company",
            template_style=template or "classic_professional",
            tone=tone or "formal",
            provider=provider,
            model_name=model_name,
            route_mode=route_mode,
            language=language or "auto",
        )
    except Exception as exc:
        logger.exception("Cover-letter generation failed")
        raise HTTPException(
            status_code=500,
            detail="Cover letter generation failed",
        ) from exc

    try:
        pdf_bytes = await asyncio.to_thread(compile_cover_letter_pdf, latex_code)
    except Exception as exc:
        logger.exception("Cover-letter PDF compilation failed")
        raise HTTPException(
            status_code=500,
            detail="Cover letter PDF compilation failed",
        ) from exc

    safe_company = re.sub(r"[^A-Za-z0-9_-]+", "_", company_name or "company")[:80]

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f"attachment; filename=Cover_Letter_{lang_used}_{safe_company}.pdf"
            )
        },
    )


# ============================================================
# FLAGSHIP 4: INTERVIEW PREP & GAP DEFENSE
# ============================================================


@router.post("/interview-prep")
async def interview_prep_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    family: str | None = Form("technical"),
    provider: str | None = Form(None),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    analysis = await asyncio.to_thread(analyze_resume_content, resume_text, job_description)
    missing_skills = analysis.get("missing_skills") or []

    prep_data = await asyncio.to_thread(
        InterviewPrepService.generate_interview_prep,
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        family=family or "technical",
        provider=provider,
    )
    return {"status": "success", "data": prep_data}


# ============================================================
# FLAGSHIP 5: LINKEDIN PROFILE OPTIMIZER
# ============================================================


@router.post("/linkedin-optimize")
async def linkedin_optimize_endpoint(
    resume_file: UploadFile = File(...),
    target_role: str | None = Form("Software Engineer"),
    provider: str | None = Form(None),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    optimized = await asyncio.to_thread(
        LinkedInOptimizerService.optimize_profile,
        resume_text=resume_text,
        target_role=target_role or "Software Engineer",
        provider=provider,
    )
    return {"status": "success", "data": optimized}


# ============================================================
# FLAGSHIP 6: APPLICATION PIPELINE TRACKER
# ============================================================


@router.get("/tracker/applications", dependencies=_PROTECTED)
async def list_applications_endpoint():
    return {
        "status": "success",
        "applications": ApplicationTrackerService.list_applications(),
    }


@router.post("/tracker/applications", dependencies=_PROTECTED)
async def create_application_endpoint(payload: TrackerCreateRequest):
    safe_job_url = payload.job_url or ""
    if safe_job_url:
        try:
            safe_job_url = validate_job_url(safe_job_url, resolve_dns=False)
        except UnsafeInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    app_data = ApplicationTrackerService.create_application(
        company_name=payload.company_name.strip(),
        job_title=payload.job_title.strip(),
        job_url=safe_job_url,
        ats_score=payload.ats_score or 0,
        status=payload.status or "Saved",
        notes=payload.notes or "",
    )
    return {"status": "success", "data": app_data}


@router.patch("/tracker/applications/{app_id}", dependencies=_PROTECTED)
async def update_status_endpoint(app_id: int, payload: TrackerStatusUpdate):
    updated = ApplicationTrackerService.update_status(app_id, payload.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Application record not found.")
    return {"status": "success", "message": "Application status updated."}


@router.delete("/tracker/applications/{app_id}", dependencies=_PROTECTED)
async def delete_application_endpoint(app_id: int):
    deleted = ApplicationTrackerService.delete_application(app_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Application record not found.")
    return {"status": "success", "message": "Application deleted."}


# ============================================================
# DOCX BUILDER
# ============================================================


def build_ats_docx_resume(markdown_resume: str) -> bytes:
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    for line in markdown_resume.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("# "):
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(stripped[2:])
            run.bold = True
            run.font.size = Pt(18)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        elif stripped.startswith("## "):
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(stripped[3:].upper())
            run.bold = True
            run.font.size = Pt(12)

        elif stripped.startswith(("* ", "- ")):
            paragraph = doc.add_paragraph(style="List Bullet")
            run = paragraph.add_run(stripped[2:].strip())
            run.font.size = Pt(10)

        else:
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(stripped)
            run.font.size = Pt(10)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEALTH
# ============================================================


@router.get("/health")
async def health_check():
    return {"status": "ok"}


# ============================================================
# USAGE & MODEL CATALOG
# ============================================================


@router.get("/usage-summary")
async def usage_summary(period: str = "all"):
    period_map = {
        "today": 24,
        "24h": 24,
        "7d": 24 * 7,
        "30d": 24 * 30,
        "all": None,
    }
    hours = period_map.get((period or "all").lower(), None)

    summary = LLMService.get_usage_summary(since_hours=hours)

    return {
        "status": "success",
        "period": period,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }


@router.get("/model-catalog")
async def model_catalog(provider: str | None = None):
    """
    List the models this deployment can actually use.

    With no ``provider`` the whole catalog is returned, so the endpoint no
    longer hard-codes a single gateway and therefore no longer reports a
    provider the operator may not even have configured.
    """
    try:
        rows = LLMService.get_model_catalog(provider)
        return {
            "status": "success",
            "provider": provider,
            "count": len(rows),
            "models": rows,
        }
    except Exception:
        logger.exception("Model catalog request failed")
        raise HTTPException(status_code=500, detail="Model catalog unavailable")


@router.get("/model-discovery")
async def model_discovery(provider: str | None = None):
    """
    List the models a provider currently offers.

    Complements ``/model-catalog``, which reports what this deployment is
    *configured* to send. This one asks the provider what it currently *has*,
    which is how a free-tier or aggregator model list is kept honest without
    editing code: availability changes, so the answer is read live.

    Discovery is always optional. A provider that exposes no listing, or that
    cannot be reached, still returns its configured model with a ``status`` that
    says why, so this endpoint can never be the reason a CV cannot be generated.
    """
    try:
        if provider:
            return LLMService.list_models(provider)

        result = LLMService.list_models()
        providers = result.get("providers", {})
        return {
            "status": "success",
            "count": sum(len(rows.get("models", [])) for rows in providers.values()),
            "configured": sum(1 for rows in providers.values() if rows.get("configured")),
            "total": len(providers),
            "providers": providers,
        }
    except Exception:
        logger.exception("Model discovery request failed")
        raise HTTPException(status_code=500, detail="Model discovery unavailable")


@router.post("/validate-pdf")
async def validate_pdf_endpoint(pdf_file: UploadFile = File(...)):
    """
    Check a generated PDF and report what was verified.

    Accepts any PDF, so a CV generated earlier can be re-checked without
    regenerating it. The report is the same one the generation path runs
    internally; nothing is asserted here that is not measured there.

    Returns ``valid: false`` with a reason rather than an HTTP error, because a
    document that failed validation is a legitimate answer to the question, not
    a failed request.
    """
    from app.services.cv.pdf_validation import (
        PDFValidationError,
        extract_pdf_text,
        find_latex_artifacts,
        validate_pdf_content,
    )

    filename = (pdf_file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only a .pdf file can be validated.",
        )

    try:
        payload = await pdf_file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file.")

    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(payload) > 32 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The PDF is too large to validate.",
        )

    if not payload.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="That file is not a PDF.",
        )

    report: dict = {
        "filename": filename,
        "bytes": len(payload),
        "valid": False,
        "status": "unknown",
        "pages": None,
        "text_chars": None,
        "sections_found": [],
        "sections_missing": [],
        "latex_artifacts": [],
        "replacement_chars": 0,
        "content_retention": None,
        "problems": [],
        "detail": "",
    }

    try:
        # No expected_text: this is a file from outside, so there is nothing
        # honest to compare it against. Retention is reported as absent rather
        # than guessed.
        result = validate_pdf_content(payload, expected_pages=None)
    except PDFValidationError as exc:
        report.update(exc.report or {})
        report["valid"] = False
        report["status"] = "rejected"
        report["detail"] = str(exc)
        return report
    except Exception:
        # A parser failure is a result, not a crash.
        report["status"] = "unreadable"
        report["detail"] = "The PDF could not be parsed."
        return report

    try:
        report["latex_artifacts"] = find_latex_artifacts(extract_pdf_text(payload))
    except Exception:
        report["latex_artifacts"] = []

    report.update(result)
    report["valid"] = not result.get("problems")
    report["status"] = "ok" if report["valid"] else "rejected"
    if report["valid"]:
        report["detail"] = (
            f"{result.get('pages')} page(s), "
            f"{result.get('text_chars')} characters extracted, "
            f"no problems found."
        )
    return report


@router.get("/pipeline-metrics")
async def pipeline_metrics(limit: int = 40):
    """
    Recent generation metrics, read from the pipeline event log.

    The log already records provider, model, per-attempt LLM duration, token
    counts, retries and PDF compilation time. This surfaces the most recent
    ``limit`` events so a UI can show them, grouped into a readable summary.

    Read-only and tolerant: a missing or unreadable log returns an empty list
    rather than an error, because a metrics panel is never worth failing a page
    render over.
    """
    import json as _json

    # LOG_PATH, not a re-derived path: the writer and the reader must agree,
    # and the writer is the service that owns the constant. Reading a second,
    # independently-resolved path is how a metrics panel ends up permanently
    # empty while generation works fine.
    log_path = LOG_PATH

    if not log_path.exists():
        return {
            "status": "no_log",
            "detail": "No processing log has been written yet.",
            "events": [],
            "summary": {},
        }

    try:
        raw = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {"status": "unreadable", "events": [], "summary": {}}

    events: list[dict] = []
    for line in raw[-max(1, min(limit, 500)) :]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_json.loads(line))
        except Exception:
            continue

    # A request that failed at attempt 1 and succeeded at attempt 2 is one
    # generation with one retry, not two generations. Counting distinct request
    # ids is what makes the number meaningful.
    requests: dict[str, dict] = {}

    for event in events:
        kind = event.get("event")
        request_id = event.get("request_id")

        if not request_id or not isinstance(kind, str):
            continue

        entry = requests.setdefault(
            request_id,
            {
                "request_id": request_id,
                "task": event.get("task"),
                "mode": event.get("mode"),
                "provider": None,
                "model": None,
                "attempts": 0,
                "retries": 0,
                "llm_duration_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "fallbacks": [],
                "outcome": "in_progress",
                "started_at": event.get("timestamp"),
            },
        )

        entry["provider"] = event.get("provider") or entry["provider"]
        entry["model"] = event.get("model") or entry["model"]

        if kind == "request_started":
            entry["attempts"] += 1
            if entry["attempts"] > 1:
                entry["retries"] = entry["attempts"] - 1
            if event.get("fallback_from"):
                entry["fallbacks"].append(
                    {
                        "from": event.get("fallback_from"),
                        "reason": event.get("fallback_reason"),
                    }
                )
        elif kind == "request_completed":
            entry["llm_duration_ms"] += float(event.get("duration_ms") or 0.0)
            entry["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
            entry["completion_tokens"] += int(event.get("completion_tokens") or 0)
            entry["outcome"] = "ok"
        elif kind == "request_failed":
            entry["llm_duration_ms"] += float(event.get("duration_ms") or 0.0)
            entry["outcome"] = "failed"
            entry["error"] = event.get("error")

    pdf_events = [event for event in events if str(event.get("event", "")).startswith("pdf_")]

    latest_pdf = pdf_events[-1] if pdf_events else {}

    completed = [row for row in requests.values() if row["outcome"] == "ok"]

    summary = {
        "generations": len(requests),
        "succeeded": len(completed),
        "llm_calls": sum(row["attempts"] for row in requests.values()),
        "retries": sum(row["retries"] for row in requests.values()),
        "llm_duration_ms": round(sum(row["llm_duration_ms"] for row in requests.values()), 1),
        "prompt_tokens": sum(row["prompt_tokens"] for row in requests.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in requests.values()),
    }

    if latest_pdf.get("duration_ms") is not None:
        summary["pdf_duration_ms"] = latest_pdf.get("duration_ms")

    if latest_pdf.get("event") == "pdf_compiled":
        summary["pdf"] = {
            "bytes": latest_pdf.get("bytes"),
            "pages": latest_pdf.get("pages"),
            "text_chars": latest_pdf.get("text_chars"),
            "sections_found": latest_pdf.get("sections_found"),
            "content_retention": latest_pdf.get("content_retention"),
            "duration_ms": latest_pdf.get("duration_ms"),
            "valid": True,
        }
    elif latest_pdf.get("event") in {
        "pdf_content_rejected",
        "pdf_compilation_failed",
    }:
        summary["pdf"] = {
            "valid": False,
            "event": latest_pdf.get("event"),
            "error": latest_pdf.get("error"),
            "problems": latest_pdf.get("problems"),
            "duration_ms": latest_pdf.get("duration_ms"),
        }

    return {
        "status": "ok",
        "summary": summary,
        "requests": list(reversed(list(requests.values())))[:20],
        "pdf_events": pdf_events[-10:],
    }


# ============================================================
# QUOTA / USAGE LIMITS
# ============================================================


@router.get("/quota-status")
async def quota_status(provider: str | None = None):
    try:
        if provider:
            data = quota_tracker.build_quota_status(
                provider=provider,
                log_path=LOG_PATH,
            )
            return {
                "status": "success",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "providers": {provider.lower(): data},
            }

        data = quota_tracker.build_all_provider_quota_status(log_path=LOG_PATH)
        return {
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "providers": data,
        }
    except Exception:
        logger.exception("Quota status request failed")
        raise HTTPException(status_code=500, detail="Quota status unavailable")


@router.get("/quota-events")
async def quota_events(provider: str | None = None, limit: int = 50):
    limit = max(1, min(limit, 200))
    events = quota_tracker.recent_rate_limit_events(provider=provider, limit=limit)
    return {
        "status": "success",
        "count": len(events),
        "events": events,
    }


# ============================================================
# ANALYTICS
# ============================================================


@router.get("/analytics/summary", dependencies=_PROTECTED)
async def analytics_summary(period: str = "30d"):
    """Aggregated analytics over the pipeline log and the tracker DB."""
    period_map = {
        "today": 24,
        "24h": 24,
        "7d": 24 * 7,
        "30d": 24 * 30,
        "90d": 24 * 90,
        "all": None,
    }
    hours = period_map.get((period or "30d").lower(), 24 * 30)

    from app.services.analytics.dashboard_aggregator import compute_analytics

    data = compute_analytics(
        log_path=LOG_PATH,
        db_path=TRACKER_DB_PATH,
        since_hours=hours,
    )
    return {
        "status": "success",
        "period": period,
        "data": data,
    }


@router.get("/career-options")
async def career_options():
    """Catalog of cover-letter templates and interview families."""
    from app.services.career.cover_letter_templates import list_templates
    from app.services.career.interview_questions import list_families

    return {
        "status": "success",
        "cover_letter_templates": list_templates(),
        "interview_families": list_families(),
    }


# ============================================================
# ANALYZE RESUME
# ============================================================


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
@_log_pipeline("analyze")
async def analyze_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: str | None = Form(None),
    model_name: str | None = Form(None),
    route_mode: str | None = Form(None),
):
    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty.",
        )

    resume_text = await extract_text_from_file(resume_file)

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable text from resume.",
        )

    results = await asyncio.to_thread(
        analyze_resume_content,
        resume_text=resume_text,
        job_description=job_description,
    )

    log_event(
        "analysis",
        "analysis_completed",
        ats_score=results.get("ats_match_score"),
        missing_count=len(results.get("missing_skills") or []),
        matched_count=len(results.get("matching_skills") or []),
        missing_skills=results.get("missing_skills") or [],
        matching_skills=results.get("matching_skills") or [],
    )

    results["recommendation"] = suggest_best_cv_format(
        job_description=job_description,
        resume_text=resume_text,
    )

    if results.get("missing_skills"):
        try:
            _layout = auto_select_layout(job_description, resume_text)

            rewrite = await asyncio.to_thread(
                optimize_resume_bullets,
                resume_text=resume_text,
                job_description=job_description,
                missing_skills=results["missing_skills"],
                provider=provider,
                model_name=model_name,
                route_mode=route_mode,
                layout_style=_layout,
            )

            results.setdefault("improvement_suggestions", []).append(
                f"AI Bullet Point Rewrite " f"({(provider or 'auto').upper()}):\n\n" f"{rewrite}"
            )

        except Exception:
            logger.exception("AI bullet rewrite unavailable")
            results.setdefault("improvement_suggestions", []).append(
                "AI bullet rewrite unavailable; manual review is required."
            )

    return AnalysisResponse(
        status="success",
        ats_match_score=results["ats_match_score"],
        keyword_density_score=results["keyword_density_score"],
        matching_skills=results["matching_skills"],
        missing_skills=results["missing_skills"],
        improvement_suggestions=results["improvement_suggestions"],
        recommendation=results.get("recommendation"),
        resume_text=resume_text,
    )


# ============================================================
# FULL TAILORED DOCX GENERATION
# ============================================================


@router.post("/generate-full")
@_log_pipeline("generate-full")
async def generate_full_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: str | None = Form(None),
    model_name: str | None = Form(None),
    route_mode: str | None = Form(None),
    layout_style: str | None = Form("auto"),
    improvement_suggestions: str | None = Form(None),
):
    resume_text = await extract_text_from_file(resume_file)

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable text from resume.",
        )

    suggestions = _decode_suggestions(improvement_suggestions)

    analysis = await asyncio.to_thread(
        analyze_resume_content,
        resume_text=resume_text,
        job_description=job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    _layout = layout_style or "auto"
    if _layout.strip().lower() in ("auto", "auto_detect", ""):
        _layout = auto_select_layout(job_description, resume_text)

    tailored = await asyncio.to_thread(
        generate_full_tailored_cv,
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider,
        model_name=model_name,
        route_mode=route_mode,
        improvement_suggestions=suggestions,
        layout_style=_layout,
    )

    docx_bytes = await asyncio.to_thread(build_ats_docx_resume, tailored)

    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        headers={"Content-Disposition": ("attachment; filename=Tailored_Optimized_Resume.docx")},
    )


# ============================================================
# GERMAN PDF LEBENSLAUF
# ============================================================


@router.post("/generate-german-cv")
@_log_pipeline("generate-german-cv")
async def generate_german_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    layout_style: str | None = Form("auto"),
    template_style: str | None = Form(None),
    provider: str | None = Form(None),
    model_name: str | None = Form(None),
    route_mode: str | None = Form(None),
    improvement_suggestions: str | None = Form(None),
):
    selected_style = template_style or layout_style or "auto"

    resume_text = await extract_text_from_file(resume_file)

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable text from resume.",
        )

    suggestions = _decode_suggestions(improvement_suggestions)

    analysis = await asyncio.to_thread(
        analyze_resume_content,
        resume_text=resume_text,
        job_description=job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    try:
        latex_code = await asyncio.to_thread(
            generate_german_latex_content,
            resume_text=resume_text,
            job_description=job_description,
            missing_skills=missing_skills,
            provider=provider,
            model_name=model_name,
            route_mode=route_mode,
            layout_style=selected_style,
            improvement_suggestions=suggestions,
        )
    except FactualValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "factual_validation_failed",
                "violations": exc.violations,
            },
        ) from exc

    try:
        pdf_bytes = await asyncio.to_thread(compile_latex_to_pdf, latex_code)
    except PDFLayoutError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": exc.code,
                "pages": exc.pages,
                "message": str(exc),
            },
        ) from exc
    except LaTeXCompilationError as exc:
        # The compiler's own diagnostic is already redacted by
        # pdf_compiler, and returning it is the whole point: a bare
        # "LaTeX compilation failed" made an unescaped "&" in generated CV
        # text indistinguishable from a broken TeX installation.
        logger.error(
            "LaTeX compilation failed: category=%s line=%s detail=%s",
            exc.category,
            exc.line,
            exc.detail or exc,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": f"latex_{exc.category}",
                "message": str(exc),
            },
        ) from exc
    except LaTeXSourceError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "latex_source_invalid",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("LaTeX compilation failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "latex_compilation_failed",
                "message": "LaTeX compilation failed.",
            },
        ) from exc

    safe_style = re.sub(r"[^A-Za-z0-9_-]+", "_", selected_style)[:60] or "cv"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": (f"attachment; filename=CV_{safe_style}.pdf")},
    )


# ============================================================
# GERMAN LATEX TEX SOURCE
# ============================================================


@router.post("/generate-tex-cv")
@_log_pipeline("generate-tex-cv")
async def generate_tex_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    layout_style: str | None = Form("auto"),
    template_style: str | None = Form(None),
    provider: str | None = Form(None),
    model_name: str | None = Form(None),
    route_mode: str | None = Form(None),
    improvement_suggestions: str | None = Form(None),
):
    selected_style = template_style or layout_style or "auto"

    resume_text = await extract_text_from_file(resume_file)

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable text from resume.",
        )

    suggestions = _decode_suggestions(improvement_suggestions)

    analysis = await asyncio.to_thread(
        analyze_resume_content,
        resume_text=resume_text,
        job_description=job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    try:
        latex_code = await asyncio.to_thread(
            generate_german_latex_content,
            resume_text=resume_text,
            job_description=job_description,
            missing_skills=missing_skills,
            provider=provider,
            model_name=model_name,
            route_mode=route_mode,
            layout_style=selected_style,
            improvement_suggestions=suggestions,
        )
    except FactualValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "factual_validation_failed",
                "violations": exc.violations,
            },
        ) from exc

    safe_style = re.sub(r"[^A-Za-z0-9_-]+", "_", selected_style)[:60] or "cv"
    return StreamingResponse(
        io.BytesIO(latex_code.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": (f"attachment; filename=CV_{safe_style}.tex")},
    )


# ============================================================
# BACKEND LLM STATUS & LOGGING
# ============================================================


@router.get("/backend-status", dependencies=_PROTECTED)
async def backend_status():
    """
    Report which LLM providers are usable and how requests are being routed.

    ``routing`` makes the active LLM_MODE visible without reading .env, and
    ``ollama`` turns "the model name is wrong" into an explicit message instead
    of a 404 during generation. Neither contains a credential: provider names,
    model names and booleans only.
    """
    import shutil

    from app.services.cv.latex_generator import (
        CV_LAYOUT_COLUMNS,
        CV_TEMPLATES,
        required_language_for_layout,
    )

    pdflatex = shutil.which("pdflatex")

    layouts = [
        {
            "name": name,
            "language": required_language_for_layout(name),
            "columns": CV_LAYOUT_COLUMNS.get(name, "single"),
        }
        for name in sorted(CV_TEMPLATES)
    ]

    return {
        "status": "online",
        "providers": LLMService.provider_status(),
        "routing": LLMService.route_info(),
        "ollama": LLMService.ollama_health(),
        "log_file": LOG_PATH.name,
        # Reported by the process that will actually run the compiler, which is
        # not necessarily the process rendering the dashboard.
        "document_toolchain": {
            "pdflatex_available": bool(pdflatex),
            "pdflatex_path": pdflatex or "",
            "pdf_generation": bool(pdflatex),
            "note": ("" if pdflatex else "Install MiKTeX or TeX Live and put pdflatex on PATH."),
        },
        "layouts": layouts,
    }


@router.get("/processing-log", dependencies=_PROTECTED)
async def processing_log(limit: int = 100):
    limit = max(1, min(limit, 500))
    logs = LLMService.recent_logs(limit)

    return {
        "status": "success",
        "count": len(logs),
        "logs": logs,
    }


@router.delete("/processing-log", dependencies=_PROTECTED)
async def clear_processing_log():
    LLMService.clear_logs()

    return {
        "status": "success",
        "message": "Backend processing log cleared.",
    }
