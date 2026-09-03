# suggestions.py
from typing import List


def generate_recommendations(
    ats_score: float, missing_skills: List[str], matching_skills: List[str]
) -> List[str]:
    """
    Generates actionable optimization advice based on resume analysis.
    Enforces additive enrichment to reach 100% ATS alignment without altering facts.
    """
    recommendations = []

    # Score-based tier recommendations
    if ats_score < 40.0:
        recommendations.append(
            "CRITICAL GAP: Low ATS alignment detected. Expand your existing work experience and project entries by adding exact tools, frameworks, and methodologies required by the job posting."
        )
    elif ats_score < 70.0:
        recommendations.append(
            "MODERATE ALIGNMENT: Weave missing technical keywords directly into your existing experience bullet points. Ensure tools are contextualized within project accomplishments."
        )
    else:
        recommendations.append(
            "STRONG ALIGNMENT: Excellent key term match! Verify that missing niche skills are integrated into both your Technical Skills block and relevant accomplishment descriptions."
        )

    # Missing skill integration
    if missing_skills:
        top_missing = missing_skills[:7]
        skills_str = ", ".join(top_missing)
        recommendations.append(
            f"KEYWORD INJECTION: Add these missing core technical terms to your Technical Skills section and existing project descriptions: {skills_str}."
        )

    # Domain depth & tool coverage
    if len(matching_skills) < 6:
        recommendations.append(
            "TOOL DEPTH: Expand technical specificity. Explicitly name version control systems, deployment environments, databases, and monitoring tools mentioned in the job description."
        )
    else:
        recommendations.append(
            "ATS OPTIMIZATION: Ensure all matched technical terms appear naturally within detailed result metrics (e.g., impact, system scale, or optimization outcomes)."
        )

    return recommendations
