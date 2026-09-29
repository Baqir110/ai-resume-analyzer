"""
Intelligent CV layout recommendation engine.

Analyzes the job description, resume content, and candidate profile to
recommend the most appropriate CV layout. The recommendation is based on
measurable characteristics of both the job and the candidate, not on
visual appearance alone.

Each layout has metadata describing its ATS safety, structure, best use
cases, and parsing risk. The recommender matches these traits against the
detected job/candidate profile.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.cv.optimizer import detect_jd_language

logger = logging.getLogger(__name__)

# Layout metadata — based on actual template implementation
LAYOUT_METADATA: dict[str, dict[str, Any]] = {
    "german_corporate": {
        "name": "German Corporate",
        "ats_safety": "High",
        "ats_safety_score": 90,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["technical", "business", "german_market"],
        "recommended_industries": ["IT", "Engineering", "Finance", "Consulting"],
        "recommended_seniority": ["mid", "senior"],
        "typical_pages": 1,
        "parsing_risk": "low",
        "language": "de",
        "description": "Serif body, a single accent colour and a rule under each section. The conventional German application CV.",
    },
    "german_ats": {
        "name": "German ATS",
        "ats_safety": "Very High",
        "ats_safety_score": 95,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["ats_focused", "german_market", "keyword_heavy"],
        "recommended_industries": ["IT", "Engineering", "Administration"],
        "recommended_seniority": ["junior", "mid", "senior"],
        "typical_pages": 1,
        "parsing_risk": "very_low",
        "language": "de",
        "description": "Plain and unambiguous, built for keyword extraction. No colour emphasis.",
    },
    "german_classic": {
        "name": "German Classic",
        "ats_safety": "High",
        "ats_safety_score": 88,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["traditional", "conservative", "german_market"],
        "recommended_industries": ["Finance", "Legal", "Government", "Mittelstand"],
        "recommended_seniority": ["mid", "senior"],
        "typical_pages": 1,
        "parsing_risk": "low",
        "language": "de",
        "description": "Serif throughout with ruled section heads. The most conservative option.",
    },
    "german_modern": {
        "name": "German Modern",
        "ats_safety": "High",
        "ats_safety_score": 85,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["modern", "tech", "german_market"],
        "recommended_industries": ["IT", "Startups", "Media"],
        "recommended_seniority": ["junior", "mid"],
        "typical_pages": 1,
        "parsing_risk": "low",
        "language": "de",
        "description": "Sans-serif with a coloured header block. Reads as more current than Classic.",
    },
    "german_minimal_ats": {
        "name": "German Minimal ATS",
        "ats_safety": "Maximum",
        "ats_safety_score": 100,
        "structure": "single_column",
        "content_density": "high",
        "best_for": ["ats_focused", "maximum_compatibility", "german_market"],
        "recommended_industries": ["IT", "Engineering", "Any"],
        "recommended_seniority": ["junior", "mid", "senior"],
        "typical_pages": 1,
        "parsing_risk": "minimal",
        "language": "de",
        "description": "Pure black throughout, no colour at all. Survives monochrome printing and aggressive parsers.",
    },
    "international_ats": {
        "name": "International ATS",
        "ats_safety": "Maximum",
        "ats_safety_score": 100,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["ats_focused", "international", "english_market", "keyword_heavy"],
        "recommended_industries": ["IT", "Engineering", "Finance", "Consulting", "Any"],
        "recommended_seniority": ["junior", "mid", "senior"],
        "typical_pages": 1,
        "parsing_risk": "minimal",
        "language": "en",
        "description": "English ATS layout with a brand colour and a full-width measure.",
    },
    "academic": {
        "name": "Academic",
        "ats_safety": "High",
        "ats_safety_score": 85,
        "structure": "single_column",
        "content_density": "high",
        "best_for": ["academic", "research", "education", "publications"],
        "recommended_industries": ["Academia", "Research", "Education", "Science"],
        "recommended_seniority": ["mid", "senior"],
        "typical_pages": 2,
        "parsing_risk": "low",
        "language": "en",
        "description": "Serif with a centred header and numbered sections. Suits research and teaching roles.",
    },
    "technical_lead": {
        "name": "Technical Lead",
        "ats_safety": "High",
        "ats_safety_score": 88,
        "structure": "single_column",
        "content_density": "high",
        "best_for": ["technical", "leadership", "engineering", "senior"],
        "recommended_industries": ["IT", "Engineering", "Software", "DevOps"],
        "recommended_seniority": ["senior"],
        "typical_pages": 1,
        "parsing_risk": "low",
        "language": "en",
        "description": "Sans-serif, dense skills block, a rule between entries. Suits engineering leadership.",
    },
    "standard": {
        "name": "Standard",
        "ats_safety": "High",
        "ats_safety_score": 90,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["general", "professional", "neutral"],
        "recommended_industries": ["Any"],
        "recommended_seniority": ["junior", "mid", "senior"],
        "typical_pages": 1,
        "parsing_risk": "low",
        "language": "en",
        "description": "The neutral default: blue section heads, ruled, one column.",
    },
    "hr_executive_gold": {
        "name": "HR Executive Gold",
        "ats_safety": "Medium",
        "ats_safety_score": 75,
        "structure": "single_column",
        "content_density": "medium",
        "best_for": ["executive", "hr", "leadership", "senior"],
        "recommended_industries": ["HR", "Executive", "Management", "Finance"],
        "recommended_seniority": ["senior"],
        "typical_pages": 1,
        "parsing_risk": "medium",
        "language": "en",
        "description": "Serif with a gold accent and uppercase section heads.",
    },
}

# Industry detection patterns
_INDUSTRY_PATTERNS: dict[str, list[str]] = {
    "IT": [
        "software",
        "developer",
        "engineer",
        "programming",
        "code",
        "api",
        "cloud",
        "devops",
        "it ",
        "daten",
        "informatik",
    ],
    "Engineering": [
        "engineering",
        "mechanical",
        "electrical",
        "civil",
        "ingenieur",
        "maschinenbau",
    ],
    "Finance": ["finance", "banking", "accounting", "finanz", "bank", "buchhaltung"],
    "Consulting": ["consulting", "beratung", "advisory", "strategy"],
    "Healthcare": ["health", "medical", "nursing", "pflege", "medizin", "krankenhaus"],
    "Academia": ["research", "university", "professor", "forschung", "universität", "lehre"],
    "HR": ["hr", "human resources", "recruiting", "personal", "talent"],
    "Marketing": ["marketing", "seo", "content", "social media", "werbung"],
    "Legal": ["legal", "law", "attorney", "recht", "anwalt"],
    "Government": ["government", "public", "behörde", "verwaltung", "öffentlich"],
}

# Seniority detection
_SENIORITY_PATTERNS = {
    "junior": ["junior", "entry", "trainee", "praktikant", "werkstudent", "graduate"],
    "mid": ["mid", "regular", "experienced", "berufserfahren", "professional"],
    "senior": ["senior", "lead", "principal", "staff", "head", "leiter", "manager", "director"],
}

# Role type detection
_ROLE_TYPE_PATTERNS = {
    "technical": [
        "developer",
        "engineer",
        "programmer",
        "devops",
        "sre",
        "administrator",
        "techniker",
        "entwickler",
    ],
    "non_technical": ["manager", "consultant", "analyst", "coordrator", "berater", "manager"],
    "academic": ["researcher", "professor", "lecturer", "wissenschaftler", "dozent"],
}


def _detect_industry(job_description: str) -> str:
    """Detect the industry from the job description."""
    jd_lower = job_description.lower()
    scores: dict[str, int] = {}
    for industry, keywords in _INDUSTRY_PATTERNS.items():
        scores[industry] = sum(1 for kw in keywords if kw in jd_lower)
    if not scores or max(scores.values()) == 0:
        return "Any"
    return max(scores, key=scores.get)


def _detect_seniority(job_description: str) -> str:
    """Detect the seniority level from the job description."""
    jd_lower = job_description.lower()
    for level, keywords in _SENIORITY_PATTERNS.items():
        if any(kw in jd_lower for kw in keywords):
            return level
    return "mid"


def _detect_role_type(job_description: str) -> str:
    """Detect whether the role is technical, non-technical, or academic."""
    jd_lower = job_description.lower()
    scores: dict[str, int] = {}
    for role_type, keywords in _ROLE_TYPE_PATTERNS.items():
        scores[role_type] = sum(1 for kw in keywords if kw in jd_lower)
    if not scores or max(scores.values()) == 0:
        return "non_technical"
    return max(scores, key=scores.get)


def _estimate_content_volume(resume_text: str) -> dict[str, Any]:
    """Estimate the content volume of the resume."""
    char_count = len(resume_text)
    word_count = len(resume_text.split())
    line_count = len(resume_text.splitlines())
    lower = resume_text.lower()

    # Count sections
    section_count = 0
    for keyword in ["experience", "education", "skills", "projects", "certifications", "languages"]:
        if keyword in lower:
            section_count += 1

    # How many distinct skills the CV claims. A long comma-separated skills line
    # is the single biggest driver of whether a CV needs one page or two, so it
    # is counted rather than inferred from the overall word count.
    skill_count = 0
    for marker in ("skills", "kenntnisse", "fähigkeiten", "competencies", "technical skills"):
        index = lower.find(marker)
        if index < 0:
            continue
        for line in resume_text[index : index + 1200].splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            # A short all-caps line is the next heading, not a skill.
            if len(stripped.split()) <= 4 and stripped == stripped.upper():
                break
            skill_count += len(
                [part for part in re.split(r"[,;/|•]|\band\b", stripped) if part.strip()]
            )
        break

    # How many dated roles the CV describes, which decides how much of the page is
    # given over to work history.
    entry_count = len(
        re.findall(r"(?:19|20)\d{2}\s*(?:-|–|—|to|bis|until)\s*(?:[A-Za-z0-9]|$)", resume_text)
    )

    # ~500 words per page is the conventional density for a single-column CV.
    estimated_pages = max(1, round(word_count / 500))

    return {
        "chars": char_count,
        "words": word_count,
        "lines": line_count,
        "sections": section_count,
        "skills": skill_count,
        "entries": entry_count,
        "estimated_pages": estimated_pages,
        "is_content_heavy": word_count > 600 or section_count >= 5,
    }


def _score_layout_fit(
    layout_id: str,
    industry: str,
    seniority: str,
    role_type: str,
    content_volume: dict[str, Any],
    jd_language: str,
    ats_priority: bool,
) -> dict[str, Any]:
    """
    Score how well a layout fits the detected profile.

    Returns a fit score (0-100) and reasons for the score.
    """
    meta = LAYOUT_METADATA.get(layout_id, {})
    if not meta:
        return {"fit_score": 0, "reasons": ["Unknown layout"]}

    score = 50.0  # Base score
    reasons: list[str] = []

    # Language match
    layout_lang = meta.get("language", "any")
    if layout_lang == jd_language:
        score += 20.0
        reasons.append(f"Language matches ({jd_language})")
    elif layout_lang == "any":
        score += 10.0
    else:
        score -= 30.0
        reasons.append(f"Language mismatch: layout is {layout_lang}, JD is {jd_language}")

    # ATS priority
    ats_safety_score = meta.get("ats_safety_score", 50)
    if ats_priority:
        if ats_safety_score >= 95:
            score += 20.0
            reasons.append("Maximum ATS safety")
        elif ats_safety_score >= 85:
            score += 10.0
            reasons.append("High ATS safety")
        else:
            score -= 10.0
            reasons.append("Lower ATS safety than alternatives")

    # Industry match
    recommended_industries = meta.get("recommended_industries", [])
    if industry in recommended_industries or "Any" in recommended_industries:
        score += 10.0
        reasons.append(f"Recommended for {industry}")

    # Seniority match
    recommended_seniority = meta.get("recommended_seniority", [])
    if seniority in recommended_seniority:
        score += 10.0
        reasons.append(f"Suits {seniority}-level roles")

    # Content density
    content_density = meta.get("content_density", "medium")
    if content_volume.get("is_content_heavy"):
        if content_density == "high":
            score += 10.0
            reasons.append("Handles high content volume well")
        elif content_density == "medium":
            score += 0.0
        else:
            score -= 10.0
            reasons.append("May not handle high content volume well")

    # Role type
    best_for = meta.get("best_for", [])
    if role_type == "technical" and "technical" in best_for:
        score += 10.0
        reasons.append("Designed for technical roles")
    elif role_type == "academic" and "academic" in best_for:
        score += 10.0
        reasons.append("Designed for academic roles")

    # Parsing risk
    parsing_risk = meta.get("parsing_risk", "low")
    if parsing_risk in ("minimal", "very_low"):
        score += 5.0
        reasons.append("Very low parsing risk")

    score = max(0.0, min(100.0, score))
    return {"fit_score": round(score, 1), "reasons": reasons}


def recommend_layout(
    job_description: str,
    resume_text: str = "",
    ats_priority: bool = True,
) -> dict[str, Any]:
    """
    Recommend the best CV layout for the given job and resume.

    Analyzes industry, seniority, role type, content volume, and ATS
    requirements to recommend the most appropriate layout.

    Returns the recommended layout with reasons and alternatives.
    """
    logger.info("Generating layout recommendation.")

    # Detect characteristics
    jd_language = detect_jd_language(job_description)
    industry = _detect_industry(job_description)
    seniority = _detect_seniority(job_description)
    role_type = _detect_role_type(job_description)
    content_volume = _estimate_content_volume(resume_text)

    # Score all layouts
    layout_scores: list[dict[str, Any]] = []
    for layout_id, meta in LAYOUT_METADATA.items():
        fit = _score_layout_fit(
            layout_id,
            industry,
            seniority,
            role_type,
            content_volume,
            jd_language,
            ats_priority,
        )
        layout_scores.append(
            {
                "layout_id": layout_id,
                "name": meta.get("name", layout_id),
                "fit_score": fit["fit_score"],
                "reasons": fit["reasons"],
                "ats_safety": meta.get("ats_safety", "Unknown"),
                "ats_safety_score": meta.get("ats_safety_score", 0),
                "structure": meta.get("structure", "unknown"),
                "language": meta.get("language", "any"),
                "description": meta.get("description", ""),
                "best_for": meta.get("best_for", []),
                "recommended_industries": meta.get("recommended_industries", []),
                "recommended_seniority": meta.get("recommended_seniority", []),
                "typical_pages": meta.get("typical_pages", 1),
                "parsing_risk": meta.get("parsing_risk", "unknown"),
            }
        )

    # Sort by fit score descending
    layout_scores.sort(key=lambda x: x["fit_score"], reverse=True)

    # Top recommendation
    recommended = layout_scores[0] if layout_scores else None

    # Alternatives (next best, excluding the recommended)
    alternatives = layout_scores[1:4] if len(layout_scores) > 1 else []

    # Build recommendation reason
    if recommended:
        reason_parts = []
        if jd_language != "unknown":
            reason_parts.append(
                f"the job description is in {'German' if jd_language == 'de' else 'English'}"
            )
        if industry != "Any":
            reason_parts.append(f"the role is in {industry}")
        if seniority != "mid":
            reason_parts.append(f"the role is {seniority}-level")
        if ats_priority:
            reason_parts.append("ATS compatibility is prioritized")

        reason = "Recommended because " + ", ".join(reason_parts) + "."
        if recommended["reasons"]:
            reason += " " + " ".join(recommended["reasons"][:3]) + "."
    else:
        reason = "No recommendation could be generated."

    # One page or two, stated as a measurement rather than a preference. The word
    # count is the CV's own, so the figure is reproducible; what a role *expects* is
    # the editorial part, and it is phrased as a judgement rather than a fact.
    words = content_volume.get("words", 0)
    if words == 0:
        page_guidance = {
            "recommended_pages": None,
            "basis": "no CV text was available to measure",
        }
    else:
        estimated = content_volume.get("estimated_pages", 1)
        senior_role = seniority in {"senior", "executive", "lead"}
        academic = role_type == "academic"
        if academic:
            # Academic CVs list publications and papers. Those are not padding, so
            # length here is expected rather than a signal to cut.
            recommended_pages = max(estimated, 2)
            basis = (
                f"{words} words across {content_volume.get('sections', 0)} sections. "
                "Academic CVs are judged on publications, which cannot be cut, so "
                "length is expected here."
            )
        else:
            recommended_pages = max(estimated, 2 if senior_role else 1)
            seniority_note = (
                "This is a senior role, which normally expects more than one page, "
                "so there may be more to add."
                if senior_role
                else "One page is the safer target; cut the oldest roles first."
            )
            if words < 350:
                basis = (
                    f"{words} words - a short CV. One page is right, and a denser "
                    f"layout would only make it harder to read. {seniority_note}"
                )
            else:
                basis = (
                    f"{words} words across {content_volume.get('sections', 0)} sections, "
                    f"{content_volume.get('entries', 0)} dated role(s) and about "
                    f"{content_volume.get('skills', 0)} listed skills. {seniority_note}"
                )
        page_guidance = {"recommended_pages": recommended_pages, "basis": basis}

    return {
        "recommended_layout": recommended["layout_id"] if recommended else None,
        "recommended_name": recommended["name"] if recommended else None,
        "reason": reason,
        "ats_safety": recommended["ats_safety"] if recommended else None,
        "ats_safety_score": recommended["ats_safety_score"] if recommended else None,
        "alternatives": alternatives,
        "all_layouts": layout_scores,
        "page_guidance": page_guidance,
        "detected_profile": {
            "industry": industry,
            "seniority": seniority,
            "role_type": role_type,
            "jd_language": jd_language,
            "content_volume": content_volume,
        },
    }


def get_layout_metadata(layout_id: str) -> dict[str, Any]:
    """Get metadata for a specific layout."""
    return LAYOUT_METADATA.get(layout_id, {})


def get_all_layouts_metadata() -> dict[str, dict[str, Any]]:
    """Get metadata for all layouts."""
    return LAYOUT_METADATA.copy()


__all__ = [
    "LAYOUT_METADATA",
    "get_all_layouts_metadata",
    "get_layout_metadata",
    "recommend_layout",
]
