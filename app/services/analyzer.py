import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv

# 1. Load environment variables before initializing Hugging Face / Sentence Transformers
load_dotenv()

# Setup module logger
logger = logging.getLogger(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Machine Learning Enhancements with Graceful Fallbacks
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as exc:
    nlp = None
    logger.warning(
        f"spaCy model failed to load. Falling back to rule-based parsing: {exc}"
    )

try:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as exc:
    embedder = None
    logger.warning(
        f"SentenceTransformer failed to load. Semantic embeddings disabled: {exc}"
    )

from app.core.config import settings
from app.services.suggestions import generate_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================
# GENERAL STOP WORDS
# ============================================================

GENERAL_STOP_WORDS = {
    # English
    "about",
    "ability",
    "applicant",
    "background",
    "benefits",
    "candidate",
    "company",
    "consulting",
    "core",
    "experience",
    "feedback",
    "from",
    "good",
    "have",
    "job",
    "must",
    "office",
    "role",
    "skills",
    "team",
    "the",
    "this",
    "with",
    "work",
    "years",
    "your",
    # German
    "abschluss",
    "arbeit",
    "arbeitgeber",
    "arbeitsplatz",
    "arbeitsumgebung",
    "aufgabenschwerpunkte",
    "ausbau",
    "bachelor",
    "berater",
    "beratungsunternehmen",
    "bestandteil",
    "deine",
    "dies",
    "englischkenntnisse",
    "informatik",
    "interesse",
    "jahre",
    "kollegen",
    "kunden",
    "laufendes",
    "master",
    "masterstudium",
    "methoden",
    "mitarbeiter",
    "mitarbeitenden",
    "ort",
    "skills",
    "spezialisierung",
    "stakeholder",
    "standort",
    "standorte",
    "studiengangs",
    "studienverpflichtungen",
    "teamevents",
    "teams",
    "technologie",
    "ticket",
    "stunden",
    "ziel",
    "ziele",
    "und",
    "mit",
    "für",
}


# ============================================================
# SYNONYMS
# ============================================================

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
    "amazon web services": "aws",
    "reactjs": "react",
    "react": "reactjs",
    "nodejs": "node.js",
    "node.js": "nodejs",
    "terraform": "infrastructure as code",
    "infrastructure-as-code": "infrastructure as code",
}


# ============================================================
# DISPLAY NAMES
# ============================================================

SPECIAL_CASES = {
    "fastapi": "FastAPI",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "scikit-learn": "Scikit-Learn",
    "ci/cd": "CI/CD",
    "html": "HTML",
    "css": "CSS",
    "sql": "SQL",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "c++": "C++",
    "c#": "C#",
    "mongodb": "MongoDB",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "github": "GitHub",
    "gitlab": "GitLab",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "prometheus": "Prometheus",
    "grafana": "Grafana",
    "cloudwatch": "CloudWatch",
    "redis": "Redis",
    "mysql": "MySQL",
    "elasticsearch": "Elasticsearch",
    "rest": "REST",
    "rest-api": "REST API",
    "api": "API",
    "infrastructure-as-code": "Infrastructure-as-Code",
    "infrastructure as code": "Infrastructure-as-Code",
    "cross-channel": "Cross-Channel",
    "solution-oriented": "Solution-Oriented",
}


# ============================================================
# STRICT TECHNICAL TAXONOMY
# ============================================================

DEFAULT_TECHNICAL_SKILLS = {
    # Programming
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "go",
    "rust",
    "php",
    "ruby",
    "kotlin",
    "swift",
    # Web
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "django",
    "flask",
    "fastapi",
    # Data / ML
    "sql",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    # Databases
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    # Cloud
    "aws",
    "gcp",
    "azure",
    "cloudwatch",
    # DevOps
    "docker",
    "kubernetes",
    "k8s",
    "terraform",
    "ansible",
    "helm",
    "jenkins",
    "github actions",
    "gitlab ci",
    "ci/cd",
    # Version control
    "git",
    "github",
    "gitlab",
    "bitbucket",
    # Monitoring / observability
    "prometheus",
    "grafana",
    "datadog",
    "opentelemetry",
    "elk",
    "logging",
    "monitoring",
    "observability",
    # APIs / architecture
    "rest",
    "rest-api",
    "api",
    "microservices",
    # Infrastructure
    "infrastructure as code",
    "infrastructure-as-code",
    # AI / ML operations
    "mlops",
    "llmops",
    "model deployment",
    "model serving",
    "ai operations",
}


# ============================================================
# NON-TECHNICAL JD TERMS
# ============================================================

NON_TECHNICAL_TERMS = {
    "cross-channel",
    "solution-oriented",
    "structured",
    "analytical",
    "dynamic",
    "interdisciplinary",
    "communication",
    "teamwork",
    "collaboration",
    "reliable",
    "reliably",
    "innovation",
    "customer",
    "customers",
    "product management",
    "engineering teams",
    "existing system landscapes",
    "cost control",
    "finops",
}


# ============================================================
# TECHNICAL PATTERNS
# ============================================================

TECH_PATTERN = re.compile(
    r"""
    (?:
        \bC\+\+\b
        |
        \bC#\b
        |
        \bCI/CD\b
        |
        \bNode\.js\b
        |
        \bREST[- ]API\b
        |
        \bInfrastructure[- ]as[- ]Code\b
        |
        \bGitHub Actions\b
        |
        \bGitLab CI\b
        |
        \bMachine Learning\b
        |
        \bMLOps\b
        |
        \bLLMOps\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# FORMAT & NORMALIZE HELPER FUNCTIONS
# ============================================================


def format_skill_name(skill: str) -> str:
    skill = skill.strip()
    normalized = skill.lower()
    if normalized in SPECIAL_CASES:
        return SPECIAL_CASES[normalized]
    return skill.title()


def normalize_skill(skill: str) -> str:
    skill = skill.strip().lower()
    skill = re.sub(r"\s+", " ", skill)
    skill = skill.replace("infrastructure as code", "infrastructure-as-code")
    return skill


def is_skill_in_text(skill: str, text_lower: str) -> bool:
    """Safely checks whether a skill exists in text handling aliases & hyphens."""
    skill_lower = normalize_skill(skill)
    escaped = re.escape(skill_lower)
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"

    if re.search(pattern, text_lower, flags=re.IGNORECASE):
        return True

    if " " in skill_lower:
        alternate = skill_lower.replace(" ", "-")
        alt_pattern = r"(?<![a-zA-Z0-9])" + re.escape(alternate) + r"(?![a-zA-Z0-9])"
        if re.search(alt_pattern, text_lower, flags=re.IGNORECASE):
            return True

    if "-" in skill_lower:
        alternate = skill_lower.replace("-", " ")
        alt_pattern = r"(?<![a-zA-Z0-9])" + re.escape(alternate) + r"(?![a-zA-Z0-9])"
        if re.search(alt_pattern, text_lower, flags=re.IGNORECASE):
            return True

    synonym = SYNONYM_MAP.get(skill_lower)
    if synonym:
        synonym = normalize_skill(synonym)
        syn_pattern = r"(?<![a-zA-Z0-9])" + re.escape(synonym) + r"(?![a-zA-Z0-9])"
        if re.search(syn_pattern, text_lower, flags=re.IGNORECASE):
            return True

    return False


# ============================================================
# LOAD TAXONOMY
# ============================================================


@lru_cache(maxsize=1)
def load_skills_taxonomy() -> List[str]:
    """Loads skill taxonomy from file or fallback dictionary."""
    possible_paths = [
        Path(getattr(settings, "SKILL_TAXONOMY_PATH", "data/skills.json")),
        BASE_DIR / getattr(settings, "SKILL_TAXONOMY_PATH", "data/skills.json"),
        BASE_DIR / "data" / "skills.json",
        Path("data/skills.json"),
    ]

    for taxonomy_path in possible_paths:
        if not taxonomy_path.exists():
            continue
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            skills = []
            if isinstance(data, dict):
                for category in data.values():
                    if isinstance(category, list):
                        skills.extend(category)
            elif isinstance(data, list):
                skills.extend(data)

            cleaned = {
                normalize_skill(skill) for skill in skills if isinstance(skill, str)
            }

            technical = cleaned & DEFAULT_TECHNICAL_SKILLS
            if technical:
                logger.info(
                    f"Loaded custom skill taxonomy from {taxonomy_path} with {len(cleaned)} skills."
                )
                return sorted(cleaned)
        except Exception as exc:
            logger.error(f"Failed loading taxonomy file at {taxonomy_path}: {exc}")
            continue

    logger.info("Using default hardcoded technical skills taxonomy.")
    return sorted(DEFAULT_TECHNICAL_SKILLS)


def is_valid_tech_term(term: str, taxonomy: Set[str]) -> bool:
    normalized = normalize_skill(term)
    if normalized in GENERAL_STOP_WORDS or normalized in NON_TECHNICAL_TERMS:
        return False
    if normalized in taxonomy:
        return True

    synonym = SYNONYM_MAP.get(normalized)
    if synonym and normalize_skill(synonym) in taxonomy:
        return True

    for match in TECH_PATTERN.findall(term):
        if normalize_skill(match) == normalized:
            return True

    return False


# ============================================================
# EXTRACTION LOGIC (RULE-BASED + SPACY NER)
# ============================================================


def extract_keywords_from_jd(job_description: str) -> Set[str]:
    """Extracts technical skills using Taxonomy + Pattern Matching + spaCy NER."""
    taxonomy = set(load_skills_taxonomy())
    jd_lower = job_description.lower()
    extracted = set()

    # Direct Taxonomy Evaluation
    for skill in sorted(taxonomy, key=len, reverse=True):
        if is_skill_in_text(skill, jd_lower):
            extracted.add(normalize_skill(skill))

    # Pattern Extraction
    for match in TECH_PATTERN.findall(job_description):
        normalized = normalize_skill(match)
        if is_valid_tech_term(normalized, taxonomy):
            extracted.add(normalized)

    # ML spaCy NER Entity Extraction (if available)
    if nlp and job_description:
        try:
            doc = nlp(job_description)
            spacy_count = 0
            for token in doc:
                clean_tok = normalize_skill(token.text)
                if clean_tok in taxonomy:
                    extracted.add(clean_tok)
                    spacy_count += 1
            for chunk in doc.noun_chunks:
                clean_chunk = normalize_skill(chunk.text)
                if clean_chunk in taxonomy:
                    extracted.add(clean_chunk)
                    spacy_count += 1
            logger.debug(
                f"spaCy NER identified {spacy_count} potential technical entities."
            )
        except Exception as exc:
            logger.debug(f"spaCy extraction skipped due to error: {exc}")

    logger.info(f"Extracted {len(extracted)} technical skills from Job Description.")
    return extracted


def extract_context_requirements(job_description: str) -> Set[str]:
    jd_lower = job_description.lower()
    found = set()
    for term in NON_TECHNICAL_TERMS:
        if is_skill_in_text(term, jd_lower):
            found.add(term)
    logger.debug(f"Extracted {len(found)} non-technical context requirements.")
    return found


# ============================================================
# ML SIMILARITY ENGINES (TF-IDF + SBERT EMBEDDINGS)
# ============================================================


def calculate_context_similarity(resume_text: str, job_description: str) -> float:
    """Computes TF-IDF N-gram Cosine Similarity."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000,
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        score = float(similarity * 100)
        logger.debug(f"TF-IDF cosine similarity score: {score:.2f}%")
        return score
    except Exception as exc:
        logger.error(f"TF-IDF similarity calculation failed: {exc}")
        return 0.0


def calculate_semantic_similarity(resume_text: str, job_description: str) -> float:
    """Computes Dense Context Similarity using SBERT Embeddings."""
    if not embedder or not resume_text or not job_description:
        logger.debug(
            "SBERT embedder not active or text empty. Skipping semantic similarity."
        )
        return 0.0
    try:
        embeddings = embedder.encode([resume_text, job_description])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        score = float(sim * 100)
        logger.debug(f"SentenceTransformers semantic similarity score: {score:.2f}%")
        return score
    except Exception as exc:
        logger.error(f"Semantic similarity calculation failed: {exc}")
        return 0.0


def check_resume_structure(resume_text: str) -> Dict[str, Any]:
    """Rule-based heuristic checks for contact info and standard headers."""
    text_lower = (resume_text or "").lower()
    warnings = []
    score = 100.0

    required_sections = ["experience", "education", "skills"]
    for sec in required_sections:
        if sec not in text_lower:
            warnings.append(f"Missing standard section header: '{sec.capitalize()}'")
            score -= 15.0

    if "@" not in text_lower:
        warnings.append("No email address detected in resume text.")
        score -= 10.0

    if len(resume_text or "") < 300:
        warnings.append("Resume text appears extremely brief.")
        score -= 20.0

    final_struct_score = max(0.0, score)
    logger.info(
        f"Resume structural health score: {final_struct_score:.1f}% ({len(warnings)} warnings)"
    )
    return {
        "structure_score": final_struct_score,
        "warnings": warnings,
    }


# ============================================================
# MASTER UNIFIED ANALYZER
# ============================================================


def analyze_resume_content(
    resume_text: str,
    job_description: str,
) -> Dict[str, Any]:
    """
    Unified entrypoint performing keyword analysis, TF-IDF vector matching,
    SBERT embeddings, and structural checks in a single call.
    """
    logger.info("Beginning resume vs job description analysis.")
    resume_text = resume_text or ""
    job_description = job_description or ""
    resume_lower = resume_text.lower()

    # 1. Technical Keyword Skill Extraction
    jd_keywords = extract_keywords_from_jd(job_description)
    matching_skills_set = set()
    missing_skills_set = set()

    for skill in jd_keywords:
        formatted_name = format_skill_name(skill)
        if is_skill_in_text(skill, resume_lower):
            matching_skills_set.add(formatted_name)
        else:
            missing_skills_set.add(formatted_name)

    matching_skills = sorted(matching_skills_set, key=str.lower)
    missing_skills = sorted(missing_skills_set, key=str.lower)

    logger.info(
        f"Skill Alignment: {len(matching_skills)} matched, {len(missing_skills)} missing."
    )

    # 2. Context Requirements
    context_requirements = extract_context_requirements(job_description)
    context_matches = [
        format_skill_name(req)
        for req in sorted(context_requirements)
        if is_skill_in_text(req, resume_lower)
    ]

    # 3. Keyword Density Score
    total_jd_skills = len(matching_skills) + len(missing_skills)
    keyword_density = (
        round((len(matching_skills) / total_jd_skills) * 100, 2)
        if total_jd_skills > 0
        else 0.0
    )

    # 4. Similarities (TF-IDF + SBERT)
    raw_tfidf = calculate_context_similarity(resume_text, job_description)
    raw_semantic = calculate_semantic_similarity(resume_text, job_description)

    # Normalize context similarity score
    context_score = min(100.0, raw_tfidf * 2.5)
    if raw_semantic > 0.0:
        context_score = (context_score * 0.5) + (raw_semantic * 0.5)

    # 5. Structural Checks
    structural_flags = check_resume_structure(resume_text)

    # 6. Final Blended ATS Score
    # 60% Keyword Alignment + 25% Context Similarity + 15% Structural Health
    if total_jd_skills > 0:
        final_ats_score = round(
            (0.60 * keyword_density)
            + (0.25 * context_score)
            + (0.15 * structural_flags["structure_score"]),
            2,
        )
    else:
        final_ats_score = round(
            (0.80 * context_score) + (0.20 * structural_flags["structure_score"]), 2
        )

    final_ats_score = min(100.0, max(0.0, final_ats_score))
    logger.info(f"Final blended ATS score calculated: {final_ats_score:.2f}%")

    # 7. Generate Recommendations
    suggestions = generate_recommendations(
        final_ats_score,
        missing_skills,
        matching_skills,
    )

    for warning in structural_flags.get("warnings", []):
        suggestions.append(f"FORMATTING WARNING: {warning}")

    return {
        "ats_match_score": final_ats_score,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "keyword_density_score": keyword_density,
        "context_similarity_score": round(context_score, 2),
        "tfidf_similarity": round(raw_tfidf, 2),
        "semantic_similarity": round(raw_semantic, 2),
        "context_requirements": sorted(context_requirements),
        "context_matches": sorted(context_matches),
        "improvement_suggestions": suggestions,
        "structural_check": structural_flags,
    }
