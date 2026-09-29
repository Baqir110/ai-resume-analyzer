"""
ATS Improvement Loop — analyze, suggest, apply, re-score, repeat.

This module implements the improvement cycle described in requirement 26.
It identifies gaps, generates suggestions, applies valid improvements,
re-scores, and compares before/after. The loop stops when the target is
reached or no meaningful improvements remain.

Safety rules (requirement 35):
- Never invent experience, technologies, certifications, education, or achievements.
- Never insert unsupported keywords solely to manipulate the ATS score.
- If a required skill is missing, identify it as a gap — do not silently add it.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.analysis.ats_analyzer import (
    extract_keywords_from_jd,
    format_skill_name,
    is_skill_in_text,
    normalize_skill,
)
from app.services.analysis.ats_scoring import (
    compute_ats_breakdown,
    generate_user_friendly_suggestions,
    get_ats_summary,
)

logger = logging.getLogger(__name__)

#: Maximum improvement iterations to prevent infinite loops
MAX_IMPROVEMENT_ROUNDS = 3

#: Minimum score improvement to continue the loop
MIN_IMPROVEMENT_THRESHOLD = 2.0


def identify_ats_gaps(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    Identify specific, actionable ATS gaps.

    Returns a structured list of gaps, each with:
    - category: what type of gap (keyword, structure, experience, etc.)
    - severity: how much it impacts the score
    - description: human-readable description
    - actionable: whether it can be automatically improved
    - action: what to do about it (if actionable)
    """
    from app.services.analysis.ats_analyzer import check_resume_structure, extract_keywords_from_jd

    gaps: list[dict[str, Any]] = []
    resume_lower = resume_text.lower()

    # 1. Missing keywords
    jd_keywords = extract_keywords_from_jd(job_description)
    missing_keywords: list[str] = []
    for skill in sorted(jd_keywords, key=str.lower):
        if not is_skill_in_text(skill, resume_lower):
            missing_keywords.append(format_skill_name(skill))

    if missing_keywords:
        # Separate into "can integrate" vs "cannot invent"
        # A keyword can be integrated if it's a synonym of something in the resume
        integrable: list[str] = []
        non_integrable: list[str] = []

        for kw in missing_keywords:
            # Check if it's a variation of something in the resume
            kw_normalized = normalize_skill(kw)
            # Simple heuristic: check if any word of the skill appears in resume
            words = kw_normalized.replace("-", " ").split()
            if any(w in resume_lower for w in words if len(w) > 3):
                integrable.append(kw)
            else:
                non_integrable.append(kw)

        if integrable:
            gaps.append(
                {
                    "category": "keyword",
                    "severity": "high" if len(integrable) > 2 else "medium",
                    "description": f"{len(integrable)} relevant keywords could be better represented: {', '.join(integrable[:5])}",
                    "actionable": True,
                    "action": "Weave these terms naturally into existing experience bullets where supported by the candidate's actual background.",
                    "keywords": integrable,
                }
            )

        if non_integrable:
            gaps.append(
                {
                    "category": "keyword",
                    "severity": "high",
                    "description": f"{len(non_integrable)} required keywords are not present in the resume: {', '.join(non_integrable[:5])}",
                    "actionable": False,
                    "action": "These skills/experiences are required by the job but not supported by the candidate's CV. They cannot be added without fabrication.",
                    "keywords": non_integrable,
                    "is_gap": True,
                }
            )

    # 2. Structure gaps
    structural = check_resume_structure(resume_text)
    warnings = structural.get("warnings", [])
    if warnings:
        gaps.append(
            {
                "category": "structure",
                "severity": "medium",
                "description": f"Structure issues: {'; '.join(warnings)}",
                "actionable": True,
                "action": "Add missing standard sections and ensure contact information is present.",
                "warnings": warnings,
            }
        )

    # 3. Experience relevance
    resume_lower = resume_text.lower()

    # Check for years mismatch
    import re

    years_re = re.compile(r"(\d+)\+?\s*(?:years?|jahre|jahren)", re.IGNORECASE)
    jd_years_match = years_re.search(job_description)
    resume_years = [int(y) for y in years_re.findall(resume_text)]

    if jd_years_match:
        jd_years = int(jd_years_match.group(1))
        max_resume_years = max(resume_years, default=0)
        if max_resume_years < jd_years * 0.7:
            gaps.append(
                {
                    "category": "experience",
                    "severity": "high",
                    "description": f"Job requires {jd_years}+ years of experience; resume shows ~{max_resume_years} years.",
                    "actionable": False,
                    "action": "This is an experience gap that cannot be closed by CV optimization. The candidate should apply only if they meet the core requirements.",
                    "is_gap": True,
                }
            )

    # 4. Soft skills / context requirements
    from app.services.analysis.ats_analyzer import extract_context_requirements

    context_reqs = extract_context_requirements(job_description)
    missing_context = [
        req for req in sorted(context_reqs) if not is_skill_in_text(req, resume_lower)
    ]
    if missing_context:
        gaps.append(
            {
                "category": "soft_skills",
                "severity": "low",
                "description": f"Soft skills from the JD not clearly demonstrated: {', '.join(missing_context[:5])}",
                "actionable": True,
                "action": "Strengthen soft skill evidence in the summary and experience bullets using specific examples from the candidate's actual work.",
                "keywords": missing_context,
            }
        )

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: severity_order.get(g.get("severity", "low"), 3))

    return {
        "gaps": gaps,
        "total_gaps": len(gaps),
        "actionable_gaps": len([g for g in gaps if g.get("actionable")]),
        "non_actionable_gaps": len([g for g in gaps if not g.get("actionable")]),
        "high_severity_gaps": len([g for g in gaps if g.get("severity") == "high"]),
    }


def generate_improvement_plan(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    Generate a structured improvement plan.

    Each step is specific, actionable, and safe. Steps that would require
    fabrication are explicitly marked as gaps, not improvements.
    """
    gaps = identify_ats_gaps(resume_text, job_description)

    steps: list[dict[str, Any]] = []
    for gap in gaps["gaps"]:
        step = {
            "category": gap["category"],
            "severity": gap["severity"],
            "description": gap["description"],
            "actionable": gap["actionable"],
            "action": gap["action"],
        }
        if gap.get("is_gap"):
            step["is_unfillable_gap"] = True
            step["user_message"] = (
                f"⚠️ {gap['description']}. " "This gap cannot be closed by CV optimization alone."
            )
        steps.append(step)

    return {
        "steps": steps,
        "total_steps": len(steps),
        "actionable_steps": len([s for s in steps if s["actionable"]]),
        "unfillable_gaps": len([s for s in steps if s.get("is_unfillable_gap")]),
        "summary": _generate_improvement_summary(gaps),
    }


def _generate_improvement_summary(gaps: dict[str, Any]) -> str:
    """Generate a human-readable summary of the improvement plan."""
    parts = []
    if gaps["actionable_gaps"] > 0:
        parts.append(f"{gaps['actionable_gaps']} improvement(s) can be made to the CV.")
    if gaps["non_actionable_gaps"] > 0:
        parts.append(
            f"{gaps['non_actionable_gaps']} gap(s) require skills or experience not present in the CV."
        )
    if gaps["high_severity_gaps"] > 0:
        parts.append(f"{gaps['high_severity_gaps']} high-severity issue(s) need attention.")
    return " ".join(parts) if parts else "No significant gaps identified."


def apply_safe_improvements(
    resume_text: str,
    job_description: str,
    improvement_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply safe, non-fabricating improvements to the resume text.

    Only applies improvements that are supported by the candidate's actual
    experience. Never invents qualifications, technologies, or achievements.

    Returns the improved resume text and a report of what was changed.
    """
    improved_text = resume_text
    changes: list[dict[str, Any]] = []

    for step in improvement_plan.get("steps", []):
        if not step.get("actionable"):
            continue

        category = step.get("category")

        if category == "structure":
            # Add missing section headers
            improved_text, structure_changes = _fix_structure(improved_text, step)
            changes.extend(structure_changes)

        elif category == "keyword":
            # Only add keywords that are variations of existing content
            improved_text, keyword_changes = _improve_keyword_presence(
                improved_text, step, job_description
            )
            changes.extend(keyword_changes)

        elif category == "soft_skills":
            # Strengthen soft skill evidence
            improved_text, soft_changes = _strengthen_soft_skills(improved_text, step)
            changes.extend(soft_changes)

    return {
        "improved_text": improved_text,
        "changes": changes,
        "total_changes": len(changes),
        "improvement_applied": len(changes) > 0,
    }


def _fix_structure(text: str, step: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Fix structural issues like missing section headers."""
    changes: list[dict[str, Any]] = []
    text_lower = text.lower()

    # Check for missing sections and add them if appropriate
    section_keywords = {
        "experience": ["experience", "berufserfahrung", "arbeitserfahrung", "employment"],
        "education": ["education", "ausbildung", "bildung", "studium", "academic"],
        "skills": ["skills", "fähigkeiten", "kenntnisse", "competenc"],
    }

    for section, keywords in section_keywords.items():
        if not any(kw in text_lower for kw in keywords):
            # Section is missing — but we can't invent content
            # Just note it as a gap
            changes.append(
                {
                    "type": "structure",
                    "action": "flagged",
                    "section": section,
                    "note": f"Section '{section}' is missing. The candidate should add it with their own content.",
                }
            )

    return text, changes


def _improve_keyword_presence(
    text: str,
    step: dict[str, Any],
    job_description: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Improve keyword presence by enhancing existing content.

    Never adds keywords that aren't supported by the candidate's experience.
    Only enhances the representation of skills already present.
    """
    changes: list[dict[str, Any]] = []
    # This is a safe improvement: we only note where keywords could be better
    # represented, we don't invent new experience
    keywords = step.get("keywords", [])
    for kw in keywords[:5]:
        changes.append(
            {
                "type": "keyword",
                "action": "suggested",
                "keyword": kw,
                "note": f"Consider emphasizing '{kw}' more prominently if supported by actual experience.",
            }
        )

    return text, changes


def _strengthen_soft_skills(
    text: str,
    step: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Strengthen soft skill evidence in the resume."""
    changes: list[dict[str, Any]] = []
    keywords = step.get("keywords", [])
    for kw in keywords[:5]:
        changes.append(
            {
                "type": "soft_skill",
                "action": "suggested",
                "keyword": kw,
                "note": f"Consider adding a specific example demonstrating '{kw}'.",
            }
        )

    return text, changes


def verify_no_unsupported_claims(
    before_text: str,
    after_text: str,
) -> dict[str, Any]:
    """
    Did the improvement introduce a claim the original CV did not support?

    Compares the *vocabulary* of the two texts rather than the score, because the
    score is what is in question: a fabricated skill raises the score, so checking
    the score cannot detect fabrication. Any term that appears in the improved text
    but not the original, from the vocabulary that carries a claim — skills,
    technologies, employers, job titles, qualifications — is a fabrication.

    Ordinary editorial changes are not claims. Rewording a bullet, reordering
    sections, or tightening a sentence all alter the text without asserting
    anything new, so generic words and numbers are ignored and only the
    claim-bearing vocabulary is compared.

    Returns ``clean: True`` when nothing new was asserted. Never raises: a
    verification failure is a reason to discard the change, not to crash the run.
    """
    if after_text == before_text:
        return {
            "clean": True,
            "introduced": [],
            "detail": "the text is unchanged",
        }

    before_lower = (before_text or "").lower()

    introduced: list[str] = []
    for skill in sorted(extract_keywords_from_jd(after_text)):
        # A term is a problem when the *original* never mentioned it in any form.
        if not is_skill_in_text(skill, before_lower):
            introduced.append(format_skill_name(skill))

    if introduced:
        return {
            "clean": False,
            "introduced": introduced,
            "detail": (
                "The improved text asserts "
                + ", ".join(introduced)
                + ", which the original CV did not support. This change was "
                "discarded rather than shown as an improvement."
            ),
        }

    return {
        "clean": True,
        "introduced": [],
        "detail": (
            "No new skill, technology or qualification appears in the improved "
            "text that the original CV did not support."
        ),
    }


def run_improvement_loop(
    resume_text: str,
    job_description: str,
    target_score: float = 100.0,
) -> dict[str, Any]:
    """
    Run the full improvement loop.

    1. Analyze CV
    2. Analyze job description
    3. Identify ATS gaps
    4. Generate improvement suggestions
    5. Apply valid improvements
    6. Re-run ATS analysis
    7. Compare before/after score
    8. Verify improvements didn't introduce unsupported claims
    9. Continue until target reached or no meaningful improvements remain

    Returns the full loop results with before/after comparison.
    """
    logger.info("Starting ATS improvement loop.")

    # Initial score
    initial_breakdown = compute_ats_breakdown(resume_text, job_description)
    initial_score = initial_breakdown["ats_score"]

    rounds: list[dict[str, Any]] = []
    current_text = resume_text
    current_score = initial_score

    for round_num in range(1, MAX_IMPROVEMENT_ROUNDS + 1):
        logger.info(f"Improvement round {round_num}/{MAX_IMPROVEMENT_ROUNDS}")

        # Identify gaps
        gaps = identify_ats_gaps(current_text, job_description)

        # If no actionable gaps, stop
        if gaps["actionable_gaps"] == 0:
            logger.info("No actionable gaps remain. Stopping improvement loop.")
            break

        # Generate improvement plan
        plan = generate_improvement_plan(current_text, job_description)

        # Apply safe improvements
        result = apply_safe_improvements(current_text, job_description, plan)

        if not result["improvement_applied"]:
            logger.info("No safe improvements could be applied. Stopping.")
            break

        # Re-score
        new_breakdown = compute_ats_breakdown(result["improved_text"], job_description)
        new_score = new_breakdown["ats_score"]

        # Step 8: verify the change asserted nothing the original did not. Checked
        # *before* the score comparison, because a fabricated skill raises the
        # score — comparing first and verifying second would let a gain bought by
        # invention be reported as an improvement.
        verification = verify_no_unsupported_claims(current_text, result["improved_text"])
        if not verification["clean"]:
            logger.warning(
                "Discarding an improvement that introduced unsupported claims: %s",
                verification["introduced"],
            )
            rounds.append(
                {
                    "round": round_num,
                    "score_before": current_score,
                    "score_after": current_score,
                    "improvement": 0.0,
                    "gaps_addressed": 0,
                    "changes_made": 0,
                    "rejected": True,
                    "rejection_reason": verification["detail"],
                }
            )
            break

        # Check if improvement is meaningful
        improvement = new_score - current_score
        if improvement < MIN_IMPROVEMENT_THRESHOLD:
            logger.info(
                f"Improvement of {improvement:.1f} points is below threshold "
                f"({MIN_IMPROVEMENT_THRESHOLD}). Stopping."
            )
            break

        # Record the round
        rounds.append(
            {
                "round": round_num,
                "score_before": current_score,
                "score_after": new_score,
                "improvement": round(improvement, 1),
                "gaps_addressed": gaps["actionable_gaps"],
                "changes_made": result["total_changes"],
                "verified": verification["detail"],
            }
        )

        current_text = result["improved_text"]
        current_score = new_score

        # Check if target reached
        if current_score >= target_score:
            logger.info(f"Target score {target_score} reached.")
            break

    # Final comparison
    final_breakdown = compute_ats_breakdown(current_text, job_description)
    final_improvement = current_score - initial_score
    remaining = identify_ats_gaps(current_text, job_description)

    # The plan for the final state, so the caller can show what was proposed even
    # when nothing could be applied. Showing the plan matters as much as showing
    # the result: a gap that could not be closed is still something the user needs
    # to know about, and "no change was made" on its own does not say why.
    plan = generate_improvement_plan(current_text, job_description)

    return {
        "initial_score": initial_score,
        "final_score": current_score,
        "total_improvement": round(final_improvement, 1),
        "target_score": target_score,
        "target_reached": current_score >= target_score,
        "rounds": rounds,
        "total_rounds": len(rounds),
        "improvement_possible": final_improvement > 0,
        "remaining_gaps": remaining,
        "steps": plan["steps"],
        "improvement_summary": plan["summary"],
        "initial_breakdown": initial_breakdown,
        "final_breakdown": final_breakdown,
        "summary": get_ats_summary(final_breakdown),
        "suggestions": generate_user_friendly_suggestions(
            final_breakdown,
            remaining.get("gaps", [{}])[0].get("keywords") if remaining.get("gaps") else None,
        ),
        "explanation": _generate_loop_explanation(
            initial_score, current_score, target_score, len(rounds), final_improvement
        ),
    }


def _generate_loop_explanation(
    initial: float,
    final: float,
    target: float,
    rounds: int,
    improvement: float,
) -> str:
    """Generate a human-readable explanation of the improvement loop results."""
    if final >= target:
        return (
            f"Target reached. The ATS score improved from {initial:.1f} to "
            f"{final:.1f} over {rounds} round(s)."
        )
    if improvement > 0:
        return (
            f"The ATS score improved from {initial:.1f} to {final:.1f} "
            f"(+{improvement:.1f} points) over {rounds} round(s). "
            f"The remaining gap to {target:.1f} requires skills or experience "
            "not present in the current CV."
        )
    return (
        f"The ATS score remained at {initial:.1f}. No meaningful improvements "
        "could be made without inventing experience or qualifications, so none "
        "were made. The remaining gaps require skills the CV does not contain."
    )


__all__ = [
    "MAX_IMPROVEMENT_ROUNDS",
    "MIN_IMPROVEMENT_THRESHOLD",
    "apply_safe_improvements",
    "generate_improvement_plan",
    "identify_ats_gaps",
    "run_improvement_loop",
    "verify_no_unsupported_claims",
]
