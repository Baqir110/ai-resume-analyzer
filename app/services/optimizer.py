# optimizer.py
from typing import List, Dict, Any
from app.services.llm_provider import LLMService


def suggest_best_cv_format(
    job_description: str,
) -> Dict[str, Any]:

    jd_lower = (job_description or "").lower()

    is_german = any(
        word in jd_lower
        for word in [
            "deutsch",
            "lebenslauf",
            "aufgaben",
            "profil",
            "anforderungen",
            "standort",
        ]
    )

    is_traditional = any(
        word in jd_lower
        for word in [
            "gmbh",
            "ag",
            "behörde",
            "behoerde",
            "versicherung",
            "bank",
            "mittelstand",
        ]
    )

    if is_german:
        if is_traditional:
            return {
                "recommended_format": "german_classic_pdf",
                "label": "German Classic Single-Column PDF",
                "reason": "German posting for a traditional company. "
                "A formal single-column structure is recommended.",
                "ats_safety": "Sehr hoch",
                "confidence": 0.90,
                "layout": "german_classic",
            }

        return {
            "recommended_format": "german_modern_pdf",
            "label": "German Modern Executive PDF",
            "reason": "German technology/startup role. "
            "A modern professional layout is appropriate.",
            "ats_safety": "Hoch",
            "confidence": 0.85,
            "layout": "german_modern",
        }

    return {
        "recommended_format": "international_docx",
        "label": "International ATS Standard DOCX",
        "reason": "International/English posting. "
        "A simple ATS-compatible DOCX structure is recommended.",
        "ats_safety": "Sehr hoch",
        "confidence": 0.90,
        "layout": "international_ats",
    }


def optimize_resume_bullets(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "gemini",
) -> str:

    prompt = f"""
You are an expert ATS Resume Coach.

Target Job Description:
{job_description}

Critical Missing Skills:
{", ".join(missing_skills)}

Current Resume:
{resume_text[:5000]}

Instructions:
1. Enhance existing resume bullets to target missing keywords naturally.
2. Never remove existing original accomplishments or change degree names/facts.
3. Use: Action + Context/Technology + Result/Metric.
4. Give Before and After versions.
5. Keep the answer concise.
6. Return plain Markdown only.
"""

    return LLMService.generate(
        prompt=prompt,
        provider=provider,
    )


def generate_full_tailored_cv(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "gemini",
) -> str:

    prompt = f"""
You are an expert ATS CV Optimization Specialist.

YOUR GOAL:
Maximize the ATS match score against the target job description by enriching the candidate's existing content.

TARGET JOB DESCRIPTION:
{job_description}

MISSING SKILLS TO INTEGRATE:
{", ".join(missing_skills)}

ORIGINAL RESUME TEXT:
{resume_text}

STRICT CONSTRAINTS & ACCURACY RULES:
1. PRESERVE ALL ORIGINAL FACTS: Do not delete, shorten, or change existing company names, degree titles, dates, locations, or project titles.
2. ADDITIVE ENHANCEMENTS ONLY: You MAY expand existing experience entries, education details, and existing projects by naturally adding technical depth, methodology, tools, and missing keywords required by the job.
3. NO FABRICATED JOBS OR PROJECTS: Do not invent new company entries or brand-new side projects from scratch. Only enrich existing ones.
4. KEYWORD INTEGRATION: Seamlessly blend missing keywords into the Technical Skills section and into relevant bullet points of existing roles and projects.
5. DO NOT PRUNE: Preserve all existing accomplishments and bullet points. Do not cut details to save space.

Return exactly:

# Candidate Name

## Professional Summary

## Technical Skills

## Professional Experience

## Projects

## Education

## Languages & Certifications

Return Markdown only.
"""

    return LLMService.generate(
        prompt=prompt,
        provider=provider,
    )


def generate_cv_html_payload(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    layout_style: str = "german_modern",
    provider: str = "gemini",
) -> str:

    prompt = f"""
You are an expert German CV writer and HTML document designer.

Target Job:
{job_description}

Missing Skills:
{", ".join(missing_skills)}

Original Resume:
{resume_text}

Layout:
{layout_style}

Create a professional German CV.

Rules:
- Never delete or invent core facts.
- Preserve every project and employment entry.
- Enrich existing bullets with missing job keywords.
- Preserve degree titles, dates, and education.
- Use valid HTML only.
- No Markdown or code fences.

Return:

<div class="cv-container">
...
</div>
"""

    html = LLMService.generate(
        prompt=prompt,
        provider=provider,
    )

    return html.replace("```html", "").replace("```", "").strip()
