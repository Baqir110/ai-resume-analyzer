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

# Multilingual General Business & Corporate Stop Words (German + English)
# Used to prevent non-technical nouns from populating skill gaps.
GENERAL_STOP_WORDS = {
    # German corporate & section terms
    "about",
    "abschluss",
    "acht",
    "ansatz",
    "applicant",
    "arbeit",
    "arbeitgeber",
    "arbeitsplatz",
    "arbeitsumgebung",
    "arbeitszeitregelungen",
    "aufgabenschwerpunkte",
    "ausbau",
    "ausgangslage",
    "bachelor",
    "background",
    "benefits",
    "berater",
    "beratungsunternehmen",
    "bestandteil",
    "branche",
    "chart",
    "come",
    "communities",
    "consulting",
    "core",
    "das",
    "deine",
    "deshalb",
    "dies",
    "egal",
    "englischkenntnisse",
    "feedback",
    "folge",
    "fortune",
    "frankfurt",
    "get",
    "hauptsitz",
    "herz",
    "informatik",
    "interesse",
    "jahre",
    "kaffee",
    "kollegen",
    "kultur",
    "kunden",
    "laufendes",
    "learns",
    "lunch",
    "man",
    "master",
    "masterstudium",
    "methoden",
    "mehr",
    "mitarbeitenden",
    "mitarbeiter",
    "netzwerks",
    "obst",
    "offices",
    "ort",
    "player",
    "potenzial",
    "profitiere",
    "reise",
    "seattle",
    "skills",
    "slalom",
    "slalomer",
    "slalomobwohl",
    "spezialisierung",
    "stakeholder",
    "standort",
    "standorte",
    "strategie",
    "studiengangs",
    "studienverpflichtungen",
    "stunden",
    "teamevents",
    "teams",
    "technologie",
    "ticket",
    "time",
    "unlock",
    "unser",
    "values",
    "vielfalt",
    "vordergrund",
    "wachstum",
    "was",
    "weg",
    "welt",
    "wenn",
    "wir",
    "wort",
    "zentrale",
    "ziel",
    "ziele",
    "und",
    "mit",
    "für",
    # English general business terms
    "the",
    "this",
    "with",
    "from",
    "your",
    "have",
    "must",
    "will",
    "work",
    "team",
    "role",
    "good",
    "years",
    "office",
    "company",
    "candidate",
    "ability",
    "experience",
}

# Common tech synonyms mapping
SYNONYM_MAP = {
    "k8s": "kubernetes",
    "kubernetes": "k8s",
    "postgres": "postgresql",
    "postgresql": "postgres",
    "js": "javascript",
    "javascript": "js",
    "ts": "typescript",
    "typescript": "ts",
    "aws": "amazon web services",
    "reactjs": "react",
    "nodejs": "node.js",
}

SPECIAL_CASES = {
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
    "c++": "C++",
    "c#": "C#",
    "mongodb": "MongoDB",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
}

# Regex pattern for technical terms with special characters (e.g. C++, CI/CD, Node.js, REST-API)
TECH_PATTERN = re.compile(
    r"\b[A-Za-z0-9+#.-]*(?:[+#.-][A-Za-z0-9]+)+\b|\b[C][+#]{1,2}\b", re.IGNORECASE
)


def format_skill_name(skill: str) -> str:
    return SPECIAL_CASES.get(skill.lower(), skill.title())


def is_skill_in_text(skill: str, text_lower: str) -> bool:
    """Safely checks for skill or synonym matches using escaped boundaries."""
    skill_lower = skill.lower()

    escaped = re.escape(skill_lower)
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"

    if re.search(pattern, text_lower):
        return True

    synonym = SYNONYM_MAP.get(skill_lower)
    if synonym:
        syn_escaped = re.escape(synonym)
        syn_pattern = r"(?<![a-zA-Z0-9])" + syn_escaped + r"(?![a-zA-Z0-9])"
        if re.search(syn_pattern, text_lower):
            return True

    return False


@lru_cache(maxsize=1)
def load_skills_taxonomy() -> List[str]:
    """Loads taxonomy JSON with multiple path fallback checks."""
    possible_paths = [
        Path(settings.SKILL_TAXONOMY_PATH),
        BASE_DIR / settings.SKILL_TAXONOMY_PATH,
        BASE_DIR / "data" / "skills.json",
        Path("data/skills.json"),
    ]

    for taxonomy_path in possible_paths:
        if taxonomy_path.exists():
            try:
                with open(taxonomy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    skills = []
                    for category in data.values():
                        if isinstance(category, list):
                            skills.extend(category)
                    if skills:
                        return list(set(skills))
            except Exception:
                continue

    # Fallback taxonomy array
    return [
        "python",
        "javascript",
        "typescript",
        "java",
        "c++",
        "c#",
        "go",
        "rust",
        "sql",
        "html",
        "css",
        "fastapi",
        "django",
        "flask",
        "react",
        "angular",
        "vue",
        "pandas",
        "numpy",
        "scikit-learn",
        "pytorch",
        "tensorflow",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "git",
        "github",
        "gitlab",
        "ci/cd",
        "terraform",
        "ansible",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
    ]


def is_valid_tech_term(term: str, taxonomy: Set[str]) -> bool:
    """Validates if an extracted candidate word is a genuine technical term."""
    term_lower = term.lower()

    # 1. Reject if listed in stop words
    if term_lower in GENERAL_STOP_WORDS:
        return False

    # 2. Accept if explicitly part of taxonomy
    if term_lower in taxonomy:
        return True

    # 3. Accept terms containing technical special operators
    if any(char in term for char in ["+", "#", "/", ".", "-"]):
        return True

    return False


def extract_keywords_from_jd(job_description: str) -> Set[str]:
    """Universal skill extractor working across German and English JDs."""
    taxonomy = set(load_skills_taxonomy())
    jd_lower = job_description.lower()
    extracted = set()

    # Step A: Direct match against Taxonomy
    for skill in taxonomy:
        if is_skill_in_text(skill, jd_lower):
            extracted.add(skill.lower())

    # Step B: Extract terms matching tech special character patterns (e.g., CI/CD, C++, REST-API)
    tech_matches = TECH_PATTERN.findall(job_description)
    for match in tech_matches:
        if is_valid_tech_term(match, taxonomy):
            extracted.add(match.lower())

    # Step C: Extract capitalized candidate words filtered against stop words
    capitalized_words = re.findall(r"\b[A-Z][a-zA-Z0-9+#.-]{2,}\b", job_description)
    for word in capitalized_words:
        if is_valid_tech_term(word, taxonomy):
            extracted.add(word.lower())

    return extracted


def analyze_resume_content(resume_text: str, job_description: str) -> Dict[str, Any]:
    jd_keywords = extract_keywords_from_jd(job_description)
    resume_lower = resume_text.lower()

    matching_skills_set = set()
    missing_skills_set = set()

    for skill in jd_keywords:
        formatted_name = format_skill_name(skill)
        if is_skill_in_text(skill, resume_lower):
            matching_skills_set.add(formatted_name)
        else:
            missing_skills_set.add(formatted_name)

    matching_skills = sorted(list(matching_skills_set))
    missing_skills = sorted(list(missing_skills_set))

    total_jd_skills = len(matching_skills) + len(missing_skills)

    if total_jd_skills > 0:
        keyword_density = round((len(matching_skills) / total_jd_skills) * 100, 2)
    else:
        keyword_density = 0.0

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

    if total_jd_skills > 0:
        final_ats_score = round((0.70 * skill_score) + (0.30 * context_score), 2)
    else:
        final_ats_score = round(context_score, 2)

    suggestions = generate_recommendations(
        final_ats_score, missing_skills, matching_skills
    )

    return {
        "ats_match_score": final_ats_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "keyword_density_score": keyword_density,
        "improvement_suggestions": suggestions,
    }
