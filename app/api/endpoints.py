import io
from typing import Optional

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
)

from app.models.schemas import AnalysisResponse

from app.services.analyzer import (
    analyze_resume_content,
)

from app.services.parser import (
    extract_text_from_file,
)

from app.services.optimizer import (
    optimize_resume_bullets,
    generate_full_tailored_cv,
    suggest_best_cv_format,
)

from app.services.latex_generator import (
    generate_german_latex_content,
    compile_latex_to_pdf,
)

from app.services.llm_provider import (
    LLMService,
)

router = APIRouter()


# ============================================================
# DOCX
# ============================================================


def build_ats_docx_resume(
    markdown_resume: str,
) -> bytes:

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
# ANALYZE
# ============================================================


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
async def analyze_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: Optional[str] = Form("gemini"),
    model_name: Optional[str] = Form(None),
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
        resume_text,
        job_description,
    )

    results["recommendation"] = suggest_best_cv_format(job_description)

    if results.get("missing_skills"):

        try:

            rewrite = optimize_resume_bullets(
                resume_text=resume_text,
                job_description=job_description,
                missing_skills=results["missing_skills"],
                provider=provider or "gemini",
            )

            results.setdefault(
                "improvement_suggestions",
                [],
            ).append(
                f"AI Bullet Point Rewrite "
                f"({(provider or 'gemini').upper()}):\n\n"
                f"{rewrite}"
            )

        except Exception as exc:

            results.setdefault(
                "improvement_suggestions",
                [],
            ).append(f"AI bullet rewrite unavailable: {exc}")

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
# FULL DOCX
# ============================================================


@router.post("/generate-full")
async def generate_full_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: Optional[str] = Form("gemini"),
    model_name: Optional[str] = Form(None),
):

    resume_text = await extract_text_from_file(resume_file)

    analysis = analyze_resume_content(
        resume_text,
        job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    tailored = generate_full_tailored_cv(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider or "gemini",
    )

    docx_bytes = build_ats_docx_resume(tailored)

    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type=(
            "application/vnd.openxmlformats-" "officedocument.wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": "attachment; "
            "filename=Tailored_Optimized_Resume.docx"
        },
    )


# ============================================================
# GERMAN PDF
# ============================================================


@router.post("/generate-german-cv")
async def generate_german_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    layout_style: Optional[str] = Form("german_corporate"),
    template_style: Optional[str] = Form(None),
    provider: Optional[str] = Form("gemini"),
    model_name: Optional[str] = Form(None),
):

    selected_style = template_style or layout_style or "german_corporate"

    resume_text = await extract_text_from_file(resume_file)

    analysis = analyze_resume_content(
        resume_text,
        job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    latex_code = generate_german_latex_content(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider or "gemini",
        model_name=model_name,
        layout_style=selected_style,
    )

    try:

        pdf_bytes = compile_latex_to_pdf(latex_code)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=("LaTeX compilation failed.\n\n" f"{exc}"),
        ) from exc

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; "
            f"filename=Lebenslauf_Muhammad_Baqir_"
            f"{selected_style}.pdf"
        },
    )


# ============================================================
# TEX
# ============================================================


@router.post("/generate-tex-cv")
async def generate_tex_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    layout_style: Optional[str] = Form("german_corporate"),
    template_style: Optional[str] = Form(None),
    provider: Optional[str] = Form("gemini"),
    model_name: Optional[str] = Form(None),
):

    selected_style = template_style or layout_style or "german_corporate"

    resume_text = await extract_text_from_file(resume_file)

    analysis = analyze_resume_content(
        resume_text,
        job_description,
    )

    missing_skills = analysis.get("missing_skills") or []

    latex_code = generate_german_latex_content(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider or "gemini",
        model_name=model_name,
        layout_style=selected_style,
    )

    return StreamingResponse(
        io.BytesIO(latex_code.encode("utf-8")),
        media_type="text/plain",
        headers={
            "Content-Disposition": "attachment; "
            f"filename=Lebenslauf_Muhammad_Baqir_"
            f"{selected_style}.tex"
        },
    )


# ============================================================
# BACKEND LLM STATUS
# ============================================================


@router.get("/backend-status")
async def backend_status():

    return {
        "status": "online",
        "providers": LLMService.provider_status(),
        "log_file": str(LLMService.LOG_PATH),
    }


# ============================================================
# BACKEND PROCESSING LOG
# ============================================================


@router.get("/processing-log")
async def processing_log(
    limit: int = 100,
):

    limit = max(
        1,
        min(limit, 500),
    )

    return {
        "status": "success",
        "count": len(LLMService.recent_logs(limit)),
        "logs": LLMService.recent_logs(limit),
    }


# ============================================================
# CLEAR PROCESSING LOG
# ============================================================


@router.delete("/processing-log")
async def clear_processing_log():

    LLMService.clear_logs()

    return {
        "status": "success",
        "message": "Backend processing log cleared.",
    }
