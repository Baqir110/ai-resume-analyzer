import re
from typing import Dict, Any, List
from app.services.analyzer import analyze_resume_content


class AuditMatrixService:
    # Common soft skills and domain nouns
    SOFT_SKILLS_TAXONOMY = {
        "agile",
        "scrum",
        "cross-functional",
        "leadership",
        "stakeholder management",
        "communication",
        "problem solving",
        "collaboration",
        "slas",
        "time management",
        "critical thinking",
        "mentorship",
        "project management",
        "adaptability",
    }

    @classmethod
    def run_full_audit(
        cls, resume_text: str, job_description: str, file_type: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Calculates a detailed 4-part audit breakdown of the candidate's resume.
        """
        # 1. Hard & Soft Skill Evaluation
        basic_analysis = analyze_resume_content(resume_text, job_description)
        matching_skills = basic_analysis.get("matching_skills", [])

        jd_words = set(re.findall(r"\b[a-zA-Z-]+\b", job_description.lower()))
        resume_words = set(re.findall(r"\b[a-zA-Z-]+\b", resume_text.lower()))

        matched_soft = list(
            cls.SOFT_SKILLS_TAXONOMY.intersection(jd_words).intersection(resume_words)
        )
        missing_soft = list(
            cls.SOFT_SKILLS_TAXONOMY.intersection(jd_words) - resume_words
        )

        soft_score = (
            int(
                (len(matched_soft) / max(1, len(matched_soft) + len(missing_soft)))
                * 100
            )
            if (matched_soft or missing_soft)
            else 80
        )

        # 2. Measurable Impact Check (Percentages, numbers, currency, metrics)
        metric_patterns = [
            r"\b\d+%\b",  # 30%
            r"\$\d+(?:,\d+)*(?:\.\d+)?\b",  # $50,000
            r"\b\d+\s*(?:ms|s|sec|min|hrs|x|k|m|b)\b",  # 200ms, 5x, 10k
            r"\b(?:increased|decreased|reduced|improved|grew|saved)\s+[^.\n]*?\b\d+\b",  # reduced latency by 40
        ]

        lines = [
            line.strip()
            for line in resume_text.splitlines()
            if line.strip().startswith(("*", "-", "•")) or len(line.strip()) > 30
        ]
        quantified_lines = 0

        for line in lines:
            if any(
                re.search(pattern, line, re.IGNORECASE) for pattern in metric_patterns
            ):
                quantified_lines += 1

        impact_score = (
            int((quantified_lines / max(1, len(lines))) * 100) if lines else 50
        )

        # 3. Formatting & ATS Safety Check
        format_issues = []
        format_score = 100

        # Check for table artifacts or complex column characters
        if "\t\t" in resume_text or "   " in resume_text:
            format_issues.append(
                "Possible multi-column or table structure detected (may disrupt standard ATS parsers)."
            )
            format_score -= 15

        if len(resume_text.strip()) < 400:
            format_issues.append(
                "Resume text appears extremely short or unparseable (check for scanned image PDFs)."
            )
            format_score -= 35

        # Non-ASCII character check (fancy bullets / special symbols)
        non_ascii_chars = set(re.findall(r"[^\x00-\x7F]", resume_text))
        if len(non_ascii_chars) > 5:
            format_issues.append(
                f"Contains non-standard special characters ({', '.join(list(non_ascii_chars)[:3])}) which can distort parsing."
            )
            format_score -= 10

        format_score = max(10, format_score)

        return {
            "overall_ats_score": basic_analysis.get("ats_match_score", 0),
            "audit_breakdown": {
                "hard_skills": {
                    "score": basic_analysis.get("keyword_density_score", 0),
                    "matched": matching_skills,
                    "missing": basic_analysis.get("missing_skills", []),
                },
                "soft_skills_and_domain": {
                    "score": soft_score,
                    "matched": matched_soft,
                    "missing": missing_soft,
                },
                "measurable_impact": {
                    "score": impact_score,
                    "total_bullet_points": len(lines),
                    "quantified_bullet_points": quantified_lines,
                    "feedback": f"Found metrics in {quantified_lines} out of {len(lines)} key entries.",
                },
                "formatting_safety": {
                    "score": format_score,
                    "issues": (
                        format_issues
                        if format_issues
                        else ["No severe ATS parsing obstacles detected."]
                    ),
                },
            },
        }
