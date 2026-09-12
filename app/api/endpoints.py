import functools
import io
import time
from datetime import datetime, timezone

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.utils import decode_suggestions as _decode_suggestions
from app.core.event_log import log_event, new_request_id, set_request_id
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
from app.services.llm import quota_tracker
from app.services.llm.provider import LOG_PATH, LLMService
from app.services.parsing.resume_parser import extract_text_from_file
from app.services.tracking.tracker import DB_PATH as TRACKER_DB_PATH
from app.services.tracking.tracker import ApplicationTrackerService

router = APIRouter()


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
            _uf = kwargs.get("resume_file")
            _ext = ""
            if _uf is not None and getattr(_uf, "filename", None):
                _ext = _uf.filename.rsplit(".", 1)[-1].lower()

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
                log_event(
                    "pipeline",
                    "pipeline_completed",
                    rid,
                    operation=operation,
                    status="success",
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                return result
            except HTTPException as exc:
                log_event(
                    "pipeline",
                    "pipeline_failed",
                    rid,
                    operation=operation,
                    status="http_error",
                    status_code=exc.status_code,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                    error=str(exc.detail)[:300],
                )
                raise
            except Exception as exc:
                log_event(
                    "pipeline",
                    "pipeline_failed",
                    rid,
                    operation=operation,
                    status="error",
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
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
    company_name: str
    job_title: str
    job_url: str | None = ""
    ats_score: int | None = 0
    status: str | None = "Saved"
    notes: str | None = ""


class TrackerStatusUpdate(BaseModel):
    status: str


# ============================================================
# FLAGSHIP 1: VISUAL DIFF & BULK ANALYSIS
# ============================================================


@router.post("/diff-preview")
async def diff_preview(payload: BulletDiffRequest):
    try:
        diffs = DiffPreviewService.compare_bullet_lists(
            payload.original_bullets, payload.optimized_bullets
        )
        return {"status": "success", "diffs": diffs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze-bulk")
async def analyze_bulk(
    job_description: str = Form(...),
    resume_files: list[UploadFile] = File(...),
):
    if not resume_files:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")

    try:
        files_data = []
        for file in resume_files:
            content = await file.read()
            files_data.append((content, file.filename))

        ranked_candidates = await BulkAnalyzerService.process_batch(files_data, job_description)

        return {
            "status": "success",
            "total_processed": len(ranked_candidates),
            "rankings": ranked_candidates,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
    audit_result = AuditMatrixService.run_full_audit(
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
    provider: str | None = Form("experiential"),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    result = CoverLetterService.generate_cover_letter_and_outreach(
        resume_text=resume_text,
        job_description=job_description,
        company_name=company_name or "Target Company",
        tone=tone or "formal",
        template=template or "classic_professional",
        provider=provider or "experiential",
    )
    return {"status": "success", "data": result}


# ============================================================
# FLAGSHIP 4: INTERVIEW PREP & GAP DEFENSE
# ============================================================


@router.post("/interview-prep")
async def interview_prep_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    family: str | None = Form("technical"),
    provider: str | None = Form("experiential"),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    analysis = analyze_resume_content(resume_text, job_description)
    missing_skills = analysis.get("missing_skills") or []

    prep_data = InterviewPrepService.generate_interview_prep(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        family=family or "technical",
        provider=provider or "experiential",
    )
    return {"status": "success", "data": prep_data}


# ============================================================
# FLAGSHIP 5: LINKEDIN PROFILE OPTIMIZER
# ============================================================


@router.post("/linkedin-optimize")
async def linkedin_optimize_endpoint(
    resume_file: UploadFile = File(...),
    target_role: str | None = Form("Software Engineer"),
    provider: str | None = Form("experiential"),
):
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text.")

    optimized = LinkedInOptimizerService.optimize_profile(
        resume_text=resume_text,
        target_role=target_role or "Software Engineer",
        provider=provider or "experiential",
    )
    return {"status": "success", "data": optimized}


# ============================================================
# FLAGSHIP 6: APPLICATION PIPELINE TRACKER
# ============================================================


@router.get("/tracker/applications")
async def list_applications_endpoint():
    return {
        "status": "success",
        "applications": ApplicationTrackerService.list_applications(),
    }


@router.post("/tracker/applications")
async def create_application_endpoint(payload: TrackerCreateRequest):
    app_data = ApplicationTrackerService.create_application(
        company_name=payload.company_name,
        job_title=payload.job_title,
        job_url=payload.job_url,
        ats_score=payload.ats_score or 0,
        status=payload.status or "Saved",
        notes=payload.notes or "",
    )
    return {"status": "success", "data": app_data}


@router.patch("/tracker/applications/{app_id}")
async def update_status_endpoint(app_id: int, payload: TrackerStatusUpdate):
    updated = ApplicationTrackerService.update_status(app_id, payload.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Application record not found.")
    return {"status": "success", "message": "Application status updated."}


@router.delete("/tracker/applications/{app_id}")
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
            paragraph = doc.add_paragraph(
                stripped[2:].strip(),
                style="List Bullet",
            )
            paragraph.style.font.size = Pt(10)

        else:
            paragraph = doc.add_paragraph(stripped)
            paragraph.style.font.size = Pt(10)

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
async def model_catalog(provider: str = "experiential"):
    try:
        rows = LLMService.get_model_catalog(provider)
        return {
            "status": "success",
            "provider": provider,
            "count": len(rows),
            "models": rows,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@router.get("/analytics/summary")
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
    provider: str | None = Form("experiential"),
    model_name: str | None = Form(None),
    route_mode: str | None = Form("experiential"),
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

    results = analyze_resume_content(
        resume_text=resume_text,
        job_description=job_description,
    )

    log_event(
        "analysis",
        "analysis_completed",
        ats_score=results.get("ats_match_score"),
        missing_count=len(results.get("missing_skills") or []),
        matched_count=len(results.get("matching_skills") or []),
    )

    results["recommendation"] = suggest_best_cv_format(
        job_description=job_description,
        resume_text=resume_text,
    )

    if results.get("missing_skills"):
        try:
            # Auto layout detection based on JD language
            _layout = auto_select_layout(job_description, resume_text)

            print(
                f"[AI DEBUG] /analyze "
                f"provider={provider!r} "
                f"model_name={model_name!r} "
                f"route_mode={route_mode!r} "
                f"layout={_layout!r} "
                f"missing_skills={results.get('missing_skills')!r}"
            )

            rewrite = optimize_resume_bullets(
                resume_text=resume_text,
                job_description=job_description,
                missing_skills=results["missing_skills"],
                provider=provider or "experiential",
                model_name=model_name,
                route_mode=route_mode or "experiential",
                layout_style=_layout,
            )

            results.setdefault("improvement_suggestions", []).append(
                f"AI Bullet Point Rewrite "
                f"({(provider or 'experiential').upper()}):\n\n"
                f"{rewrite}"
            )

        except Exception as exc:
            print(f"[AI DEBUG] /analyze bullet rewrite failed: {exc}")

            results.setdefault("improvement_suggestions", []).append(
                f"AI bullet rewrite unavailable: {exc}"
            )

    return AnalysisResponse(
        status="success",
        ats_match_score=results["ats_match_score"],
        keyword_density_score=results["keyword_density_score"],
        matching_skills=results["matching_skills"],
        missing_skills=results["missing_skills"],
        improvement_suggestions=results["improvement_suggestions"],
        recommendation=results.get("recommendation"),
    )


# ============================================================
# FULL TAILORED DOCX GENERATION
# ============================================================


@router.post("/generate-full")
@_log_pipeline("generate-full")
async def generate_full_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: str | None = Form("experiential"),
    model_name: str | None = Form(None),
    route_mode: str | None = Form("experiential"),
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

    analysis = analyze_resume_content(
        resume_text=resume_text,
        job_description=job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    # Auto-select layout from JD language (or use explicit override)
    _layout = layout_style or "auto"
    if _layout.strip().lower() in ("auto", "auto_detect", ""):
        _layout = auto_select_layout(job_description, resume_text)

    tailored = generate_full_tailored_cv(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider or "experiential",
        model_name=model_name,
        route_mode=route_mode or "experiential",
        improvement_suggestions=suggestions,
        layout_style=_layout,
    )

    docx_bytes = build_ats_docx_resume(tailored)

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
    provider: str | None = Form("experiential"),
    model_name: str | None = Form(None),
    route_mode: str | None = Form("experiential"),
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

    analysis = analyze_resume_content(
        resume_text=resume_text,
        job_description=job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    try:
        latex_code = generate_german_latex_content(
            resume_text=resume_text,
            job_description=job_description,
            missing_skills=missing_skills,
            provider=provider or "experiential",
            model_name=model_name,
            route_mode=route_mode or "experiential",
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
        pdf_bytes = compile_latex_to_pdf(latex_code)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LaTeX compilation failed.\n\n{exc}",
        ) from exc

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": (f"attachment; filename=CV_{selected_style}.pdf")},
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
    provider: str | None = Form("experiential"),
    model_name: str | None = Form(None),
    route_mode: str | None = Form("experiential"),
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

    analysis = analyze_resume_content(
        resume_text=resume_text,
        job_description=job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    try:
        latex_code = generate_german_latex_content(
            resume_text=resume_text,
            job_description=job_description,
            missing_skills=missing_skills,
            provider=provider or "experiential",
            model_name=model_name,
            route_mode=route_mode or "experiential",
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

    return StreamingResponse(
        io.BytesIO(latex_code.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": (f"attachment; filename=CV_{selected_style}.tex")},
    )


# ============================================================
# BACKEND LLM STATUS & LOGGING
# ============================================================


@router.get("/backend-status")
async def backend_status():
    return {
        "status": "online",
        "providers": LLMService.provider_status(),
        "log_file": str(LOG_PATH),
    }


@router.get("/processing-log")
async def processing_log(limit: int = 100):
    limit = max(1, min(limit, 500))
    logs = LLMService.recent_logs(limit)

    return {
        "status": "success",
        "count": len(logs),
        "logs": logs,
    }


@router.delete("/processing-log")
async def clear_processing_log():
    LLMService.clear_logs()

    return {
        "status": "success",
        "message": "Backend processing log cleared.",
    }
