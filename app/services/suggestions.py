from typing import List, Dict, Any


def generate_recommendations(
    ats_score: float, missing_skills: List[str], matching_skills: List[str]
) -> List[str]:
    """
    Generates actionable optimization advice based on resume analysis.
    """
    recommendations = []

    # Score-based advice
    if ats_score < 40.0:
        recommendations.append(
            "Your resume content has low alignment with the job description. Consider re-writing bullet points to target key job requirements."
        )
    elif ats_score < 65.0:
        recommendations.append(
            "Moderate alignment found. Try integrating missing core technical skills into your work experience section."
        )

    # Missing skill advice
    if missing_skills:
        top_missing = missing_skills[:3]
        skills_str = ", ".join(top_missing)
        recommendations.append(
            f"Add critical missing keywords to your Technical Skills section: {skills_str}."
        )

    # Formatting and keyword advice
    if len(matching_skills) < 5:
        recommendations.append(
            "Include more domain-specific tool names and frameworks mentioned in the job description to improve ATS parsing."
        )
    else:
        recommendations.append(
            "Good key term match! Ensure keywords appear naturally inside achievements rather than just a standalone skills list."
        )

    return recommendations
