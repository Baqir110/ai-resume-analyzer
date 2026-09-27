"""Semantic application-answer engine.

Maps free-text form questions to pre-computed answers from the
applicant profile, using keyword-overlap matching (no external ML).

Usage::

    from app.services.jobs.answer_engine import answer_question, CANONICAL_QUESTIONS

    answer = answer_question("Are you authorized to work in Germany?", profile)
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["answer_question", "CANONICAL_QUESTIONS"]


# ============================================================================
# HELPERS
# ============================================================================


def _get(profile: dict, *keys: str, default: str = "") -> str:
    """Safely traverse nested dicts; return default when any key is absent."""
    node: Any = profile
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return str(node).strip() if node is not None else default


def _skill_years(profile: dict, skill: str) -> str:
    """Look up years for a named skill in experience.skills dict."""
    skills: Any = profile.get("experience", {})
    if isinstance(skills, dict):
        skills = skills.get("skills", {})
    if isinstance(skills, dict):
        lower_skill = skill.lower()
        for key, value in skills.items():
            if key.lower() == lower_skill:
                return str(value)
    return ""


def _total_experience_years(profile: dict) -> str:
    """Use experience.total_years if present."""
    exp = profile.get("experience", {})
    if isinstance(exp, dict):
        total = exp.get("total_years")
        if total is not None:
            return str(total)
    return ""


# ============================================================================
# CANONICAL QUESTION TYPES
# ============================================================================

CANONICAL_QUESTIONS: list = [
    {
        "name": "work_authorization",
        "keywords": [
            "authorized",
            "work permit",
            "visa",
            "right to work",
            "work in germany",
            "arbeitserlaubnis",
            "work authorization",
            "legally authorized",
            "eligible to work",
        ],
        "profile_fn": lambda p: (
            "Yes"
            if (p.get("work_authorization") or {}).get("authorized_to_work") is True
            else "No"
            if (p.get("work_authorization") or {}).get("authorized_to_work") is False
            else "Manual review required"
        ),
        "fallback": "Manual review required",
    },
    {
        "name": "requires_sponsorship",
        "keywords": [
            "sponsorship",
            "visa sponsorship",
            "require sponsorship",
            "need sponsorship",
            "sponsor",
        ],
        "profile_fn": lambda p: (
            "No" if not (p.get("work_authorization") or {}).get("requires_sponsorship") else "Yes"
        ),
        "fallback": "Manual review required",
    },
    {
        "name": "salary_expectation",
        "keywords": [
            "salary",
            "compensation",
            "gehalt",
            "erwartung",
            "salary expectation",
            "expected salary",
            "pay expectation",
            "remuneration",
            "lohn",
        ],
        "profile_fn": lambda p: _get(p, "free_text", "salary_expectation"),
        "fallback": "Manual review required",
    },
    {
        "name": "notice_period",
        "keywords": [
            "notice",
            "notice period",
            "kuendigungsfrist",
            "available",
            "availability",
            "start date",
            "when can you start",
            "earliest start",
        ],
        "profile_fn": lambda p: (
            _get(p, "free_text", "notice_period") or _get(p, "free_text", "availability")
        ),
        "fallback": "Manual review required",
    },
    {
        "name": "location_willing",
        "keywords": [
            "relocate",
            "relocation",
            "move",
            "umzug",
            "willing to relocate",
            "open to relocation",
        ],
        "profile_fn": lambda p: _get(p, "free_text", "willing_to_relocate"),
        "fallback": "Manual review required",
    },
    {
        "name": "remote",
        "keywords": [
            "remote",
            "home office",
            "homeoffice",
            "work from home",
            "remote work",
            "telecommute",
        ],
        "profile_fn": lambda p: _get(p, "free_text", "remote_preference")
        or "Manual review required",
        "fallback": "Manual review required",
    },
    {
        "name": "education_level",
        "keywords": [
            "degree",
            "education",
            "abschluss",
            "bachelor",
            "master",
            "highest degree",
            "educational background",
            "qualification",
        ],
        "profile_fn": lambda p: (
            next(
                (e.get("degree", "") for e in (p.get("education") or []) if e.get("degree")),
                "",
            )
        ),
        "fallback": "Manual review required",
    },
    {
        "name": "linkedin_url",
        "keywords": ["linkedin"],
        "profile_fn": lambda p: _get(p, "personal", "linkedin"),
        "fallback": "",
    },
    {
        "name": "github_url",
        "keywords": ["github"],
        "profile_fn": lambda p: _get(p, "personal", "github"),
        "fallback": "",
    },
    {
        "name": "phone",
        "keywords": ["phone", "telefon", "mobile", "telefonnummer", "phone number"],
        "profile_fn": lambda p: _get(p, "personal", "phone"),
        "fallback": "",
    },
    {
        "name": "email",
        "keywords": ["email", "e-mail", "email address"],
        "profile_fn": lambda p: _get(p, "personal", "email"),
        "fallback": "",
    },
    {
        "name": "first_name",
        "keywords": ["first name", "vorname", "given name"],
        "profile_fn": lambda p: _get(p, "personal", "first_name"),
        "fallback": "",
    },
    {
        "name": "last_name",
        "keywords": ["last name", "nachname", "surname", "family name"],
        "profile_fn": lambda p: _get(p, "personal", "last_name"),
        "fallback": "",
    },
    {
        "name": "years_python",
        "keywords": [
            "python",
            "years of python",
            "python experience",
            "experience python",
            "how many years python",
        ],
        "profile_fn": lambda p: _skill_years(p, "python"),
        "fallback": "",
    },
    {
        "name": "years_experience_total",
        "keywords": [
            "total experience",
            "how many years",
            "years of work",
            "years of experience",
            "overall experience",
            "gesamterfahrung",
            "berufserfahrung",
        ],
        "profile_fn": _total_experience_years,
        "fallback": "",
    },
    {
        "name": "years_kubernetes",
        "keywords": [
            "kubernetes",
            "k8s",
            "years of kubernetes",
            "kubernetes experience",
        ],
        "profile_fn": lambda p: _skill_years(p, "kubernetes"),
        "fallback": "",
    },
    {
        "name": "years_docker",
        "keywords": [
            "docker",
            "years of docker",
            "docker experience",
            "containerization",
        ],
        "profile_fn": lambda p: _skill_years(p, "docker"),
        "fallback": "",
    },
    {
        "name": "years_aws",
        "keywords": [
            "aws",
            "amazon web services",
            "years of aws",
            "aws experience",
            "cloud experience aws",
        ],
        "profile_fn": lambda p: _skill_years(p, "aws"),
        "fallback": "",
    },
    {
        "name": "language_german",
        "keywords": [
            "german",
            "deutsch",
            "deutschkenntnisse",
            "german language",
            "german level",
            "deutsch level",
            "german proficiency",
        ],
        "profile_fn": lambda p: _get(p, "free_text", "german_level"),
        "fallback": "Manual review required",
    },
    {
        "name": "cover_letter_text",
        "keywords": [
            "cover letter",
            "anschreiben",
            "motivationsschreiben",
            "letter of motivation",
            "motivation letter",
        ],
        "profile_fn": lambda p: (
            _get(p, "free_text", "cover_letter_intro") or _get(p, "free_text", "why_this_company")
        ),
        "fallback": "",
    },
]


# ============================================================================
# MATCHING
# ============================================================================

_MIN_SCORE = 0.15


def _compute_match_score(question_lower: str, keywords: list) -> float:
    """Simple overlap ratio: matched keywords / total keywords."""
    if not keywords:
        return 0.0
    matched = sum(1 for kw in keywords if kw.lower() in question_lower)
    return matched / len(keywords)


def _best_match(question_lower: str):
    """Return the canonical question dict with the highest overlap score."""
    best = None
    best_score = _MIN_SCORE - 1e-9

    for cq in CANONICAL_QUESTIONS:
        score = _compute_match_score(question_lower, cq["keywords"])
        if score > best_score:
            best_score = score
            best = cq

    if best_score >= _MIN_SCORE:
        return best
    return None


# ============================================================================
# PUBLIC API
# ============================================================================


def answer_question(
    question: str,
    profile: dict,
    job: Optional[dict] = None,
) -> str:
    """Return a pre-computed answer for *question* drawn from *profile*.

    Parameters
    ----------
    question:
        The raw form question text (any language).
    profile:
        Applicant profile dict, typically loaded from applicant_profile.yaml.
    job:
        Optional job dict (reserved for future context enrichment).

    Returns
    -------
    str
        The answer string, or ``""`` if no canonical type matches (score < 0.15).
        An empty return means the caller should fall back to the LLM with full
        context rather than guessing.
    """
    if not question or not isinstance(profile, dict):
        return ""

    question_lower = question.lower()
    matched = _best_match(question_lower)

    if matched is None:
        return ""

    try:
        value = matched["profile_fn"](profile)
    except Exception:
        value = ""

    if value:
        return str(value).strip()

    fallback = matched.get("fallback", "")
    return str(fallback).strip() if fallback else ""


def build_canonical_answers(profile: dict) -> dict:
    """Return {name: answer} for every canonical question type.

    Entries with empty answers are omitted so the task prompt stays concise.
    """
    result = {}
    for cq in CANONICAL_QUESTIONS:
        try:
            value = cq["profile_fn"](profile)
        except Exception:
            value = ""
        if not value:
            value = cq.get("fallback", "")
        normalized = str(value).strip()
        if normalized and normalized.casefold() not in {
            "manual review required",
            "unknown",
            "n/a",
        }:
            result[cq["name"]] = normalized
    return result
