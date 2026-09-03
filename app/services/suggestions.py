# app/services/suggestions.py
from typing import List


def generate_recommendations(
    ats_score: float, missing_skills: List[str], matching_skills: List[str]
) -> List[str]:
    """
    Generates high-impact, actionable optimization advice based on resume analysis.
    Enforces additive enrichment to maximize ATS alignment without altering original facts.
    """
    recommendations = []

    # ------------------------------------------------------------
    # 1. Tiered ATS Alignment Advice
    # ------------------------------------------------------------
    if ats_score < 40.0:
        recommendations.append(
            "CRITICAL ALIGNMENT GAP: Low match score detected. Expand your existing work history and project entries by integrating exact tools, frameworks, and methodologies required by the job posting."
        )
    elif ats_score < 60.0:
        recommendations.append(
            "MODERATE ALIGNMENT GAP: Core technical terms are missing. Weave target job keywords directly into your experience bullet points, ensuring tools are contextualized within project deliverables."
        )
    elif ats_score < 75.0:
        recommendations.append(
            "GOOD ALIGNMENT: Strong keyword foundation. Strengthen your resume by ensuring missing specialized tools and secondary requirements appear in both your Technical Skills block and job descriptions."
        )
    elif ats_score < 90.0:
        recommendations.append(
            "HIGH ALIGNMENT: Excellent keyword match! Verify that all target technologies appear naturally inside high-impact accomplishment bullets alongside measurable results."
        )
    else:
        recommendations.append(
            "OUTSTANDING ALIGNMENT: Near-perfect ATS match! Ensure formatting is clean, single-column, and free of tables or complex graphics to ensure smooth ATS parsing."
        )

    # ------------------------------------------------------------
    # 2. Targeted Keyword Injection Advice
    # ------------------------------------------------------------
    if missing_skills:
        top_missing = missing_skills[:7]
        skills_str = ", ".join(top_missing)
        recommendations.append(
            f"KEYWORD INJECTION: Explicitly add these missing technical terms to your Technical Skills section and relevant project descriptions: {skills_str}."
        )
    else:
        recommendations.append(
            "FULL KEYWORD COVERAGE: All primary technical skills extracted from the job posting were successfully detected in your CV."
        )

    # ------------------------------------------------------------
    # 3. Technical Depth & Achievement Quality Checks
    # ------------------------------------------------------------
    num_matched = len(matching_skills)

    if num_matched < 5 and missing_skills:
        recommendations.append(
            "TECHNICAL SPECIFICITY: Increase depth by explicitly naming version control systems, deployment environments, databases, CI/CD pipelines, and monitoring tools mentioned in the job posting."
        )
    elif num_matched >= 5:
        recommendations.append(
            "QUANTIFIABLE IMPACT: Combine matched technical skills with measurable metrics (e.g., system scale, performance optimizations, latency reductions, or ticket resolution efficiency)."
        )

    # ------------------------------------------------------------
    # 4. Fact Preservation & ATS Formatting Guidance
    # ------------------------------------------------------------
    recommendations.append(
        "FORMATTING & ACCURACY RULE: Maintain 100% factual integrity. Add missing technical keywords as additive context to existing roles without changing company names, job titles, employment dates, or degree titles."
    )

    return recommendations
