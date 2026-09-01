import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.suggestions import generate_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def format_skill_name(skill: str) -> str:
    """Formats skill names with correct capitalization."""
    special_cases = {
        "fastapi": "FastAPI",
        "postgresql": "PostgreSQL",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "scikit-learn": "Scikit-Learn",
        "ci/cd": "CI/CD",
        "html": "HTML",
        "css": "CSS",
        "sql": "SQL",
        "aws": "AWS",
        "gcp": "GCP",
    }
    return special_cases.get(skill.lower(), skill.capitalize())


def is_skill_in_text(skill: str, text: str) -> bool:
    """Checks if a skill is present in text using boundary checks."""
    escaped = re.escape(skill)
    pattern = r"(?:^|[\s,./()\-:+])" + escaped + r"(?:$|[\s,./()\-:+])"
    return bool(re.search(pattern, text))


def load_skills_taxonomy() -> List[str]:
    """Loads skill keywords dynamically from JSON config or falls back to standard list."""
    taxonomy_path = BASE_DIR / settings.SKILL_TAXONOMY_PATH
    if taxonomy_path.exists():
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                skills = []
                for category in data.values():
                    skills.extend(category)
                return list(set(skills))
        except Exception:
            pass

    return [
        "python",
        "fastapi",
        "docker",
        "kubernetes",
        "sql",
        "postgresql",
        "git",
        "aws",
        "react",
        "javascript",
        "scikit-learn",
        "pandas",
        "numpy",
        "linux",
        "ci/cd",
        "terraform",
        "pytest",
        "rest",
        "ansible",
        "gcp",
        "azure",
    ]


def analyze_resume_content(resume_text: str, job_description: str) -> Dict[str, Any]:
    # 1. Skill Extraction
    skills_taxonomy = load_skills_taxonomy()
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    matching_skills = []
    missing_skills = []

    for skill in skills_taxonomy:
        in_jd = is_skill_in_text(skill, jd_lower)
        in_resume = is_skill_in_text(skill, resume_lower)

        if in_jd:
            formatted_name = format_skill_name(skill)
            if in_resume:
                matching_skills.append(formatted_name)
            else:
                missing_skills.append(formatted_name)

    total_jd_skills = len(matching_skills) + len(missing_skills)
    keyword_density = round(len(matching_skills) / max(1, total_jd_skills) * 100, 2)

    # 2. Textual TF-IDF Similarity (Filtered to meaningful content)
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=500
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        raw_similarity = (
            float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]) * 100
        )
    except Exception:
        raw_similarity = 0.0

    # 3. Hybrid ATS Score Calculation
    # Formula: 70% Skill Coverage + 30% Contextual Text Similarity
    skill_score = keyword_density
    context_score = min(
        100.0, raw_similarity * 2.5
    )  # Scale text similarity appropriately

    final_ats_score = round((0.70 * skill_score) + (0.30 * context_score), 2)

    # 4. Generate Recommendations
    suggestions = generate_recommendations(
        final_ats_score, missing_skills, matching_skills
    )

    if len(resume_text.split()) < 200:
        suggestions.append(
            "Resume body text is relatively short; expand on project accomplishments and metrics."
        )

    return {
        "ats_match_score": final_ats_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "keyword_density_score": keyword_density,
        "improvement_suggestions": suggestions,
    }
