# analyzer.py
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.suggestions import generate_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def format_skill_name(skill: str) -> str:
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
    escaped = re.escape(skill)
    pattern = r"(?:^|[\s,./()\-:+])" + escaped + r"(?:$|[\s,./()\-:+])"
    return bool(re.search(pattern, text, re.IGNORECASE))


@lru_cache(maxsize=1)
def load_skills_taxonomy() -> List[str]:
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


def extract_keywords_from_jd(job_description: str) -> Set[str]:
    """Extracts both taxonomy skills and key technical terms from the Job Description."""
    taxonomy = set(load_skills_taxonomy())
    jd_lower = job_description.lower()

    extracted = set()
    for skill in taxonomy:
        if is_skill_in_text(skill, jd_lower):
            extracted.add(skill.lower())

    # Extract additional technical words (alphanumeric phrases)
    words = re.findall(r"\b[a-zA-Z0-9\+#\.\-]{2,}\b", jd_lower)
    for w in words:
        if w in taxonomy:
            extracted.add(w)

    return extracted


def analyze_resume_content(resume_text: str, job_description: str) -> Dict[str, Any]:
    jd_keywords = extract_keywords_from_jd(job_description)
    resume_lower = resume_text.lower()

    matching_skills = []
    missing_skills = []

    for skill in jd_keywords:
        formatted_name = format_skill_name(skill)
        if is_skill_in_text(skill, resume_lower):
            matching_skills.append(formatted_name)
        else:
            missing_skills.append(formatted_name)

    total_jd_skills = len(matching_skills) + len(missing_skills)
    keyword_density = (
        round(len(matching_skills) / max(1, total_jd_skills) * 100, 2)
        if total_jd_skills > 0
        else 100.0
    )

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

    skill_score = keyword_density
    context_score = min(100.0, raw_similarity * 2.5)

    final_ats_score = round((0.70 * skill_score) + (0.30 * context_score), 2)

    suggestions = generate_recommendations(
        final_ats_score, missing_skills, matching_skills
    )

    return {
        "ats_match_score": final_ats_score,
        "matching_skills": sorted(list(set(matching_skills))),
        "missing_skills": sorted(list(set(missing_skills))),
        "keyword_density_score": keyword_density,
        "improvement_suggestions": suggestions,
    }
