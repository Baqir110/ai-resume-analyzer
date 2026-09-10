# app/services/suggestions.py
from typing import List

# Only these prefixes represent actionable edits the LLM can apply.
_ACTIONABLE_PREFIXES = (
    "CRITICAL MATCH GAP:",
    "MODERATE MATCH GAP:",
    "GOOD ALIGNMENT:",
    "HIGH ALIGNMENT:",
    "KEYWORD INJECTION:",
    "FULL KEYWORD COVERAGE:",
    "LANGUAGE MISMATCH",
)

# These are meta / verification / formatting advice — shown in the UI, but NOT
# passed to the LLM (the LLM already applies structural ATS rules via the
# generation prompt itself).
_META_PREFIXES = (
    "VERIFICATION:",
    "FORMATTING RULE",
    "OUTSTANDING ATS MATCH:",
)


def generate_recommendations(
    ats_score: float, missing_skills: List[str], matching_skills: List[str]
) -> List[str]:
    """
    Generates actionable advice to achieve 100% ATS score compliance.
    """
    recommendations = []

    # 1. Tiered ATS Match Score Feedback
    if ats_score < 40.0:
        recommendations.append(
            "CRITICAL MATCH GAP: Match score is low. Explicitly weave missing target frameworks and operational tools directly into your primary job bullet points."
        )
    elif ats_score < 60.0:
        recommendations.append(
            "MODERATE MATCH GAP: Core technical terms are missing. Contextualize key tools within project outcomes across your experience section."
        )
    elif ats_score < 75.0:
        recommendations.append(
            "GOOD ALIGNMENT: Strong baseline match. To reach 100%, list specialized secondary skills in both the Technical Skills list and job descriptions."
        )
    elif ats_score < 90.0:
        recommendations.append(
            "HIGH ALIGNMENT: Excellent match! Ensure target skills appear near strong action verbs in accomplishment bullets alongside quantifiable metrics."
        )
    else:
        recommendations.append(
            "OUTSTANDING ATS MATCH: 100% Optimization reached! Keep document layout single-column and free of tables, headers/footers, or complex graphics."
        )

    # 2. Precise Keyword Injection Plan
    if missing_skills:
        top_missing = missing_skills[:7]
        skills_str = ", ".join(top_missing)
        recommendations.append(
            f"KEYWORD INJECTION: Explicitly add these terms to both Technical Skills and Experience entries: {skills_str}."
        )
        recommendations.append(
            "VERIFICATION: After editing, scan your CV to confirm every skill from the job description appears at least once. Use the ATS score tool to validate."
        )
    else:
        recommendations.append(
            "FULL KEYWORD COVERAGE: All primary job skills were detected in your resume."
        )

    # 3. Structural & Parsing Rules for 100% Compliance
    recommendations.append(
        "FORMATTING RULE FOR 100% ATS COMPLIANCE: Use standard headings (e.g., 'Work Experience', 'Technical Skills', 'Education'). Avoid tables, text boxes, or dual-column sidebar layouts that split text flow."
    )

    return recommendations


def extract_actionable_suggestions(recommendations: List[str]) -> List[str]:
    """
    Returns only the recommendations that should be fed into the CV-generation
    prompt as hard constraints. Drops meta-advice (verification steps, generic
    formatting rules) that the LLM cannot meaningfully act on.
    """
    if not recommendations:
        return []

    actionable: List[str] = []
    for rec in recommendations:
        text = (rec or "").strip()
        if not text:
            continue

        # Skip pure meta items
        if any(text.startswith(p) for p in _META_PREFIXES):
            continue

        # Keep anything that's clearly actionable
        if any(text.startswith(p) for p in _ACTIONABLE_PREFIXES):
            actionable.append(text)
            continue

        # Unknown prefix — keep it; err on the side of inclusion so future
        # suggestion types flow through automatically.
        actionable.append(text)

    return actionable
