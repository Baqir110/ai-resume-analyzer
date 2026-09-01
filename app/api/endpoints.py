import io
from typing import Optional
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.models.schemas import AnalysisResponse, ErrorResponse
from app.services.analyzer import analyze_resume_content
from app.services.parser import extract_text_from_file
from app.services.optimizer import (
    optimize_resume_bullets,
    generate_full_tailored_cv,
    suggest_best_cv_format,
)
from app.services.latex_generator import (
    generate_german_latex_content,
    compile_latex_to_pdf,
)

router = APIRouter()


def build_ats_docx_resume(markdown_resume: str) -> bytes:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    lines = markdown_resume.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            run = p.add_run(stripped.replace("# ", ""))
            run.bold = True
            run.font.size = Pt(18)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif stripped.startswith("## "):
            p = doc.add_paragraph()
            run = p.add_run(stripped.replace("## ", "").upper())
            run.bold = True
            run.font.size = Pt(12)
        elif stripped.startswith("* ") or stripped.startswith("- "):
            clean_bullet = stripped.lstrip("*").lstrip("-").strip()
            p = doc.add_paragraph(clean_bullet, style="List Bullet")
            p.style.font.size = Pt(10)
        else:
            p = doc.add_paragraph(stripped)
            p.style.font.size = Pt(10)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: Optional[str] = Form("gemini"),
    api_key: Optional[str] = Form(None),
):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(
            status_code=400, detail="Could not extract readable text from resume file."
        )

    results = analyze_resume_content(resume_text, job_description)
    results["recommendation"] = suggest_best_cv_format(job_description)

    if results["missing_skills"]:
        try:
            llm_rewrite = optimize_resume_bullets(
                resume_text=resume_text,
                job_description=job_description,
                missing_skills=results["missing_skills"],
                provider=provider,
                api_key=api_key if api_key and api_key.strip() else None,
            )
            results["improvement_suggestions"].append(
                f"🤖 AI Bullet Point Rewrite ({provider.upper()}):\n\n{llm_rewrite}"
            )
        except Exception as e:
            results["improvement_suggestions"].append(
                f"Note: Could not run {provider.upper()} rewriter ({str(e)})."
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


@router.post("/generate-full")
async def generate_full_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    provider: Optional[str] = Form("gemini"),
    api_key: Optional[str] = Form(None),
):
    resume_text = await extract_text_from_file(resume_file)
    analysis = analyze_resume_content(resume_text, job_description)
    missing_skills = analysis.get("missing_skills", [])

    tailored_markdown = generate_full_tailored_cv(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider,
        api_key=api_key if api_key and api_key.strip() else None,
    )

    docx_bytes = build_ats_docx_resume(tailored_markdown)
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": "attachment; filename=Tailored_Optimized_Resume.docx"
        },
    )


@router.post("/generate-german-cv")
async def generate_german_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    layout_style: Optional[str] = Form("german_corporate"),
    template_style: Optional[str] = Form(None),
    provider: Optional[str] = Form("gemini"),
    api_key: Optional[str] = Form(None),
):
    selected_style = template_style or layout_style or "german_corporate"
    resume_text = await extract_text_from_file(resume_file)
    analysis = analyze_resume_content(resume_text, job_description)
    missing_skills = analysis.get("missing_skills", [])

    latex_code = generate_german_latex_content(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider,
        api_key=api_key if api_key and api_key.strip() else None,
        layout_style=selected_style,
    )

    try:
        pdf_bytes = compile_latex_to_pdf(latex_code)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=Lebenslauf_Muhammad_Baqir_{selected_style}.pdf"
            },
        )
    except Exception as compile_err:
        raise HTTPException(
            status_code=500,
            detail=f"LaTeX Compilation Error: pdflatex command failed or is not available on system PATH.\n\nError details:\n{str(compile_err)}"
        )


@router.post("/generate-tex-cv")
async def generate_tex_cv_endpoint(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    layout_style: Optional[str] = Form("german_corporate"),
    template_style: Optional[str] = Form(None),
    provider: Optional[str] = Form("gemini"),
    api_key: Optional[str] = Form(None),
):
    selected_style = template_style or layout_style or "german_corporate"
    resume_text = await extract_text_from_file(resume_file)
    analysis = analyze_resume_content(resume_text, job_description)
    missing_skills = analysis.get("missing_skills", [])

    latex_code = generate_german_latex_content(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_skills,
        provider=provider,
        api_key=api_key if api_key and api_key.strip() else None,
        layout_style=selected_style,
    )

    return StreamingResponse(
        io.BytesIO(latex_code.encode("utf-8")),
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=Lebenslauf_Muhammad_Baqir_{selected_style}.tex"
        },
    )