from typing import List, Optional, Dict
from app.services.llm_provider import LLMService

def suggest_best_cv_format(job_description: str) -> Dict[str, str]:
    """Analyzes job description and recommends the optimal CV template style."""
    jd_lower = job_description.lower()
    is_german = any(w in jd_lower for w in ["deutsch", "lebenslauf", "aufgaben", "profil", "anforderungen", "standort", "gmbh", "ag"])
    is_tech_startup = any(w in jd_lower for w in ["startup", "scaleup", "cloud native", "agile", "modern stack"])
    is_enterprise = any(w in jd_lower for w in ["konzern", "bank", "versicherung", "behoerde", "traditional"])

    if is_german:
        if is_tech_startup:
            return {
                "recommended_format": "german_modern",
                "label": "🇩🇪 German Modern Two-Column Visual Sidebar",
                "reason": "Target posting is a German tech/startup role. A modern two-column layout with a skills sidebar stands out visually."
            }
        elif is_enterprise:
            return {
                "recommended_format": "german_classic",
                "label": "🇩🇪 German Classic Conservative Single-Column",
                "reason": "Target posting is at a traditional German enterprise. A conservative single-column layout is preferred."
            }
        else:
            return {
                "recommended_format": "german_corporate",
                "label": "🇩🇪 Custom Corporate Slate Navy (100% ATS Single-Column)",
                "reason": "Target posting is in German. Your custom single-column Slate Navy template gives 100% ATS readability with modern styling."
            }
    else:
        return {
            "recommended_format": "international_ats",
            "label": "🌐 International ATS Standard (.DOCX / .PDF)",
            "reason": "Target posting is an international/English role optimized for automated ATS software."
        }


def optimize_resume_bullets(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "gemini",
    api_key: Optional[str] = None
) -> str:
    """Generates targeted bullet point rewrites to integrate missing skills."""
    prompt = f"""
You are an expert ATS Resume Coach. Optimize the candidate's resume content for the target job description.

Target Job Description:
{job_description}

Critical Missing Skills to Integrate:
{', '.join(missing_skills)}

Current Resume Text Excerpt:
{resume_text[:2000]}

Instructions:
1. Rewrite 2 to 4 bullet points to naturally incorporate missing skills.
2. Follow formula: [Action Verb] + [Context/Tools Used] + [Measurable Result/Metric].
3. Return clear before-and-after comparisons with brief explanations.
"""
    return LLMService.generate(prompt=prompt, provider=provider, api_key=api_key)


def generate_full_tailored_cv(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "gemini",
    api_key: Optional[str] = None
) -> str:
    """Generates complete tailored resume in clean Markdown format for DOCX conversion."""
    prompt = f"""
You are an Executive Resume Writer. Rewrite the candidate's entire resume tailored for this position.

Target Job Description:
{job_description}

Critical Skills to Integrate Naturally:
{', '.join(missing_skills)}

Candidate's Original Resume Text:
{resume_text}

Strict Formatting Rules:
1. Retain all factual information (name, contact info, companies, dates, education).
2. You MUST include ALL projects and work experience entries present in the candidate's original resume. Do NOT omit or summarize away any project.
3. Rewrite Professional Summary to directly match job requirements.
4. Rewrite ALL Work Experience bullet points using: [Action Verb] + [Context/Tools] + [Metric].
5. Output clean Markdown using this exact header order:
   # Candidate Name
   ## Contact Information
   ## Professional Summary
   ## Technical Skills
   ## Professional Experience
   ## Projects
   ## Education
"""
    return LLMService.generate(prompt=prompt, provider=provider, api_key=api_key)