from typing import Any, Dict, List
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
                "reason": "German posting for a traditional organization. A formal single-column structure is recommended.",
                "ats_safety": "Sehr hoch",
                "confidence": 0.90,
                "layout": "german_classic",
            }

        return {
            "recommended_format": "german_modern_pdf",
            "label": "German Modern Executive PDF",
            "reason": "German technology or startup role. A modern professional layout is appropriate.",
            "ats_safety": "Hoch",
            "confidence": 0.85,
            "layout": "german_modern",
        }

    return {
        "recommended_format": "international_docx",
        "label": "International ATS Standard DOCX",
        "reason": "International or English posting. A clean ATS-compatible structure is recommended.",
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
    prompt = f"""You are an expert ATS Resume Coach.

### FEW-SHOT EXAMPLES OF HIGH-IMPACT REWRITES:
Before: "Worked on Python projects."
After: "Developed and deployed Python-based REST APIs using FastAPI and PostgreSQL, improving backend response times by 30%."

Before: "Responsible for setting up server monitoring."
After: "Configured system-wide infrastructure monitoring using Prometheus and Grafana, reducing system downtime by 20%."

### TARGET JOB DESCRIPTION:
{job_description}

### CRITICAL MISSING SKILLS TO INTEGRATE:
{", ".join(missing_skills) if missing_skills else "None provided."}

### CURRENT RESUME TEXT:
{resume_text[:5000]}

### STEP-BY-STEP INSTRUCTIONS:
1. STEP 1 (ANALYSIS): Compare the current resume bullets against the target job requirements.
2. STEP 2 (ENRICHMENT): Enhance existing bullet points by adding relevant technologies, context, missing skills, and quantifiable metrics (Action + Context/Tool + Result).
3. STEP 3 (FACT PRESERVATION): Never alter original employment dates, company names, or degree details.

Return plain Markdown displaying the Before and After comparisons.
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
    prompt = f"""You are an expert ATS CV Optimization Specialist.

### TARGET JOB DESCRIPTION:
{job_description}

### CRITICAL MISSING SKILLS TO INTEGRATE:
{", ".join(missing_skills) if missing_skills else "None provided."}

### ORIGINAL RESUME TEXT:
{resume_text}

### STEP-BY-STEP WORKFLOW:
1. STEP 1 (MAPPING): Map existing candidate experiences and accomplishments to the target job requirements.
2. STEP 2 (SKILLS ENRICHMENT): Expand the Technical Skills section with missing target keywords.
3. STEP 3 (ADDITIVE EXPERIENCE ENRICHMENT): Expand existing bullet points in work history and projects with technical context and methodologies mentioned in the job posting.
4. STEP 4 (FACT PRESERVATION): Retain all original company names, dates, degrees, and project entries completely intact. Do NOT prune or shorten accomplishments.

Return strictly Markdown following this structure:

# Candidate Name

## Professional Summary

## Technical Skills

## Professional Experience

## Projects

## Education

## Languages & Certifications
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
    prompt = f"""You are an expert German CV document designer.

Target Job:
{job_description}

Missing Skills:
{", ".join(missing_skills) if missing_skills else "None"}

Original Resume:
{resume_text}

Layout Style:
{layout_style}

### RULES:
1. Preserve every project, degree, and employment entry.
2. Enrich existing bullets naturally with missing job keywords.
3. Output valid HTML inside <div class="cv-container">...</div>.
4. Do NOT output code fences or conversational text.
"""

    html = LLMService.generate(
        prompt=prompt,
        provider=provider,
    )

    return html.replace("```html", "").replace("```", "").strip()
