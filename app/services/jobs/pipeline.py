# app/services/jobs/pipeline.py
from pathlib import Path

from app.services.analysis.ats_analyzer import analyze_resume_content as analyze_resume
from app.services.career.cover_letter_pdf import (
    compile_cover_letter_pdf,
    generate_cover_letter_latex,
)
from app.services.cv.latex_generator import compile_latex_to_pdf, generate_german_latex_content
from app.services.jobs.browser_use_applier import BrowserUseApplier

OUTPUT_DIR = Path("data/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_pdf(pdf_bytes: bytes, name: str) -> str:
    """Save generated PDF bytes to disk and return the path."""
    path = OUTPUT_DIR / f"{name}.pdf"
    path.write_bytes(pdf_bytes)
    return str(path)


async def full_pipeline(job: dict, resume_text: str, profile: dict):
    """
    Discover → Score → Tailor → Generate → Apply.
    """
    # 1. Analyzer
    analysis = analyze_resume(resume_text, job["description"])
    if analysis.score < 7.0:
        return {"skipped": True, "reason": "below threshold"}

    # 2. CV generator
    latex = generate_german_latex_content(
        resume_text=resume_text,
        job_description=job["description"],
        missing_skills=analysis.missing_skills,
        layout_style=analysis.recommended_layout,
    )
    pdf_bytes = compile_latex_to_pdf(latex)
    resume_path = save_pdf(pdf_bytes, job["id"])

    # 3. Cover letter generator
    cover_latex, lang = generate_cover_letter_latex(
        resume_text=resume_text,
        job_description=job["description"],
        company_name=job["company"],
    )
    cover_bytes = compile_cover_letter_pdf(cover_latex)
    cover_path = save_pdf(cover_bytes, f"{job['id']}_cover")

    # 4. "Why this company" — placeholder until a real generator exists.
    #    Replace with your real function when available.
    why = (
        f"I am excited about the opportunity at {job['company']} "
        f"because it aligns with my background and the {job['title']} role."
    )

    # 5. Fill the form
    applier = BrowserUseApplier()
    return await applier.apply(
        job_url=job["apply_url"],
        resume_path=resume_path,
        cover_letter_path=cover_path,
        why_this_company=why,
        company_name=job["company"],
        role_title=job["title"],
    )
