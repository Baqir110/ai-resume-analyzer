# app/services/analysis/suggestions.py

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
    ats_score: float, missing_skills: list[str], matching_skills: list[str]
) -> list[str]:
    """
    Generates actionable advice to achieve 100% ATS score compliance.

    User-friendly language (requirement 32): simple, clear, actionable.
    No ATS jargon without explanation.
    """
    recommendations = []

    # 1. Tiered ATS Match Score Feedback
    if ats_score < 40.0:
        recommendations.append(
            "CRITICAL MATCH GAP: Your CV has significant gaps compared to the job description. "
            "Focus on adding relevant skills and experience that you actually have."
        )
    elif ats_score < 60.0:
        recommendations.append(
            "MODERATE MATCH GAP: Some key skills are missing. "
            "Add relevant keywords from the job description where supported by your actual experience."
        )
    elif ats_score < 75.0:
        recommendations.append(
            "GOOD ALIGNMENT: Your CV matches the job well. "
            "To improve further, ensure all relevant skills are clearly listed."
        )
    elif ats_score < 90.0:
        recommendations.append(
            "HIGH ALIGNMENT: Excellent match! "
            "Make sure your key skills appear near strong action verbs in your experience section."
        )
    else:
        recommendations.append(
            "OUTSTANDING ATS MATCH: Your CV is highly compatible with the job description. "
            "Keep the document layout single-column and free of tables or complex graphics."
        )

    # 2. Precise Keyword Injection Plan
    if missing_skills:
        top_missing = missing_skills[:7]
        skills_str = ", ".join(top_missing)
        recommendations.append(
            f"KEYWORD INJECTION: Consider adding these terms to your CV: {skills_str}. "
            "Only add skills you actually have experience with."
        )
        recommendations.append(
            "VERIFICATION: After editing, scan your CV to confirm every skill from the job description appears at least once."
        )
    else:
        recommendations.append(
            "FULL KEYWORD COVERAGE: All primary job skills were detected in your resume."
        )

    # 3. Structural & Parsing Rules for 100% Compliance
    recommendations.append(
        "FORMATTING RULE: Use standard headings (e.g., 'Work Experience', 'Technical Skills', 'Education'). "
        "Avoid tables, text boxes, or dual-column layouts that may confuse ATS systems."
    )

    return recommendations


def extract_actionable_suggestions(recommendations: list[str]) -> list[str]:
    """
    Returns only the recommendations that should be fed into the CV-generation
    prompt as hard constraints. Drops meta-advice (verification steps, generic
    formatting rules) that the LLM cannot meaningfully act on.
    """
    if not recommendations:
        return []

    actionable: list[str] = []
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
