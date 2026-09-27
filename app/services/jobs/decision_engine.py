# app/services/jobs/decision_engine.py
"""Autonomous apply/skip decision engine.

Two layers, per spec:
  1. Hard rules -- disqualifying facts that should skip a job
     regardless of how good the score looks (visa, salary floor,
     incompatible location, language far above candidate level, etc).
  2. Weighted scoring -- technical/experience/education/location/
     language dimensions blended into an overall score used to
     prioritize (not just filter) the remaining jobs.

Pure stdlib logic, deliberately decoupled from the LLM/ATS analyzer so
it's cheap to run *before* spending an LLM call, and unit-testable
without any of the heavy optional dependencies (sentence-transformers,
browser-use, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.services.jobs.agent_schemas import MatchResult, NormalizedJob


@dataclass
class CandidateProfile:
    """Minimal fields the decision engine needs. Built from
    data/applicant_profile.yaml + config -- see profile_manager.py
    for the loader that fills this in."""

    skills: set[str] = field(default_factory=set)
    years_experience: float = 0.0
    education_level: str = ""  # bachelor / master / phd
    languages: dict[str, str] = field(default_factory=dict)  # {"german": "B1", "english": "C1"}
    allowed_locations: set[str] = field(
        default_factory=set
    )  # normalized city/region names; empty = any
    remote_ok: bool = True
    requires_sponsorship: bool = False
    citizenship: str = ""
    work_authorized_countries: set[str] = field(default_factory=set)
    work_authorization_known: bool = False
    authorized_to_work: bool | None = None
    minimum_salary: Optional[float] = None
    salary_currency: str = "EUR"
    willing_to_relocate: bool = True


@dataclass
class MatchConfig:
    weight_technical: float = 0.36
    weight_experience: float = 0.18
    weight_education: float = 0.09
    weight_location: float = 0.135
    weight_language: float = 0.135
    weight_seniority: float = 0.10
    min_match_score: float = 60.0
    high_priority_threshold: float = 85.0
    good_match_threshold: float = 70.0
    low_priority_threshold: float = 60.0
    # A missing *preferred* skill never disqualifies on its own; this
    # only affects how many hard-required skills may be absent before
    # the job is skipped outright.
    max_missing_required_skills: int = 2
    # 0 = unknown/skip seniority scoring; otherwise candidate's years of experience.
    candidate_years_experience: int = 0

    # CEFR ordering used to compare "required B2" vs "candidate B1".
    _CEFR_ORDER = ["a1", "a2", "b1", "b2", "c1", "c2", "native"]


_LANGUAGE_LEVEL_ORDER = ["a1", "a2", "b1", "b2", "c1", "c2", "native"]


def _level_rank(level: str) -> int:
    level = (level or "").strip().lower()
    return _LANGUAGE_LEVEL_ORDER.index(level) if level in _LANGUAGE_LEVEL_ORDER else -1


# ---------------------------------------------------------------------------
# Hard rules
# ---------------------------------------------------------------------------


def check_hard_rules(
    job: NormalizedJob, profile: CandidateProfile, config: MatchConfig
) -> list[str]:
    """Returns a list of human-readable skip reasons. Empty list = passes."""
    reasons: list[str] = []

    # Visa / sponsorship
    if job.visa_sponsorship_offered is False and profile.requires_sponsorship:
        reasons.append("Job explicitly does not offer visa sponsorship, candidate requires it")

    # Work authorization / country restriction. Unknown authorization is a
    # hard stop for autonomous applications; never infer permission.
    if job.work_authorization:
        wa = job.work_authorization.lower()
        required_countries = _extract_countries(wa)
        if not profile.work_authorization_known:
            reasons.append(
                f"Work authorization is unknown; manual review required: '{job.work_authorization}'"
            )
        elif profile.authorized_to_work is False:
            reasons.append(
                f"Candidate is not authorized for the requested location: '{job.work_authorization}'"
            )
        elif profile.work_authorized_countries:
            if required_countries and not (profile.work_authorized_countries & required_countries):
                reasons.append(f"Mandatory work authorization not met: '{job.work_authorization}'")
        elif not profile.requires_sponsorship and required_countries:
            reasons.append(
                f"Work authorization countries are not configured: '{job.work_authorization}'"
            )

    # Location
    if profile.allowed_locations and job.remote_type.lower() != "remote":
        job_loc = (job.location or "").lower()
        if job_loc and not any(loc in job_loc for loc in profile.allowed_locations):
            if not (
                profile.willing_to_relocate
                and job.country.lower()
                in {c.split(",")[0].strip() for c in profile.allowed_locations}
            ):
                reasons.append(f"Location '{job.location}' outside candidate's allowed locations")

    if job.remote_type.lower() == "onsite" and not profile.remote_ok and profile.allowed_locations:
        pass  # already covered by location check above

    # Language requirement far above candidate level
    for lang, required_level in _extract_language_requirements(job):
        candidate_level = profile.languages.get(lang.lower())
        if candidate_level is None:
            continue
        req_rank, cand_rank = _level_rank(required_level), _level_rank(candidate_level)
        if req_rank >= 0 and cand_rank >= 0 and req_rank - cand_rank >= 2:
            reasons.append(
                f"Required {lang} level {required_level.upper()} is well above "
                f"candidate's {candidate_level.upper()}"
            )

    # Salary floor
    if profile.minimum_salary and job.salary_max:
        if job.salary_max < profile.minimum_salary:
            reasons.append(
                f"Salary max {job.salary_max} {job.salary_currency} below "
                f"candidate minimum {profile.minimum_salary} {profile.salary_currency}"
            )

    # Clearly impossible experience requirement
    if job.experience_required_years and profile.years_experience:
        if job.experience_required_years - profile.years_experience >= 5:
            reasons.append(
                f"Requires {job.experience_required_years}+ years, candidate has "
                f"{profile.years_experience}"
            )

    return reasons


def _extract_countries(text: str) -> set[str]:
    return {t.strip() for t in text.replace("/", ",").split(",") if t.strip()}


def _extract_language_requirements(job: NormalizedJob) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for lang in job.languages:
        parts = lang.split()
        if len(parts) >= 2 and _level_rank(parts[-1]) >= 0:
            out.append((" ".join(parts[:-1]), parts[-1]))
    return out


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------


def score_job(
    job: NormalizedJob,
    profile: CandidateProfile,
    config: MatchConfig,
    matched_skills: Optional[list[str]] = None,
    missing_skills: Optional[list[str]] = None,
    partial_skills: Optional[list[str]] = None,
) -> MatchResult:
    """Blend dimension scores into an overall 0-100 score.

    Skill matching itself is delegated to the caller (typically the
    existing ATS analyzer, which already does skill extraction well)
    so this function stays a pure aggregator -- easy to unit test with
    hand-built inputs, and reusable regardless of which skill-matcher
    produced the lists.
    """
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    partial_skills = partial_skills or []

    required_total = len(matched_skills) + len(missing_skills) + len(partial_skills)
    technical_score = (
        100.0 * (len(matched_skills) + 0.5 * len(partial_skills)) / required_total
        if required_total
        else 50.0
    )

    experience_score = _ratio_score(profile.years_experience, job.experience_required_years)
    education_score = _education_score(profile.education_level, job.education_required)
    location_score = _location_score(job, profile)
    language_score = _language_score(job, profile)
    seniority_score = _seniority_score(job, config)

    overall = (
        technical_score * config.weight_technical
        + experience_score * config.weight_experience
        + education_score * config.weight_education
        + location_score * config.weight_location
        + language_score * config.weight_language
        + seniority_score * config.weight_seniority
    )

    result = MatchResult(
        overall_score=round(overall, 1),
        technical_score=round(technical_score, 1),
        experience_score=round(experience_score, 1),
        education_score=round(education_score, 1),
        location_score=round(location_score, 1),
        language_score=round(language_score, 1),
        seniority_score=round(seniority_score, 1),
        matched=matched_skills,
        partially_matched=partial_skills,
        missing=missing_skills,
    )
    return result


def _ratio_score(have: float, required: Optional[float]) -> float:
    if not required or required <= 0:
        return 100.0
    if have >= required:
        return 100.0
    return max(0.0, 100.0 * have / required)


def _education_score(candidate_level: str, required_level: str) -> float:
    order = ["highschool", "bachelor", "master", "phd"]
    if not required_level:
        return 100.0
    req = required_level.lower()
    req_idx = next((i for i, o in enumerate(order) if o in req), None)
    cand_idx = next((i for i, o in enumerate(order) if o in candidate_level.lower()), None)
    if req_idx is None or cand_idx is None:
        return 75.0  # unknown -> neutral-ish, don't let it dominate the score
    return 100.0 if cand_idx >= req_idx else max(0.0, 100.0 - 30.0 * (req_idx - cand_idx))


def _location_score(job: NormalizedJob, profile: CandidateProfile) -> float:
    if job.remote_type.lower() == "remote":
        return 100.0
    if not profile.allowed_locations:
        return 100.0
    job_loc = (job.location or "").lower()
    if any(loc in job_loc for loc in profile.allowed_locations):
        return 100.0
    return 60.0 if profile.willing_to_relocate else 20.0


def _language_score(job: NormalizedJob, profile: CandidateProfile) -> float:
    reqs = _extract_language_requirements(job)
    if not reqs:
        return 100.0
    scores = []
    for lang, level in reqs:
        cand_level = profile.languages.get(lang.lower())
        if cand_level is None:
            scores.append(75.0)  # unknown requirement, don't penalize hard
            continue
        gap = _level_rank(level) - _level_rank(cand_level)
        scores.append(100.0 if gap <= 0 else max(0.0, 100.0 - 25.0 * gap))
    return sum(scores) / len(scores)


def _detect_job_seniority(job: NormalizedJob) -> str:
    """Returns 'HIGH', 'MID', or 'LOW' based on title and description."""
    text = f"{job.title} {job.description}".lower()
    # HIGH seniority indicators
    if re.search(
        r"\b(senior|lead|principal|architect|staff)\b" r"|\b([5-9]|[1-9][0-9])\+\s*years?\b",
        text,
    ):
        return "HIGH"
    # LOW seniority indicators
    if re.search(
        r"\b(junior|entry[- ]level|entry\s+level)\b" r"|\b(0[-–][23]|1[-–]3)\s*years?\b",
        text,
    ):
        return "LOW"
    return "MID"


def _seniority_score(job: NormalizedJob, config: MatchConfig) -> float:
    """Score 0-100 measuring seniority fit between candidate and job.

    Returns the neutral score (75) when candidate experience is unknown
    (``candidate_years_experience == 0``).
    """
    years = config.candidate_years_experience
    if years <= 0:
        return 75.0  # unknown -> neutral, don't penalise or reward

    # Infer candidate seniority level from years.
    if years >= 5:
        cand_level = "HIGH"
    elif years >= 3:
        cand_level = "MID"
    else:
        cand_level = "LOW"

    job_level = _detect_job_seniority(job)

    _RANK = {"LOW": 0, "MID": 1, "HIGH": 2}
    job_rank = _RANK[job_level]
    cand_rank = _RANK[cand_level]

    if cand_rank > job_rank:
        return 80.0  # overqualified but not disqualifying
    if cand_rank == job_rank:
        return 100.0  # perfect seniority fit
    gap = job_rank - cand_rank
    if gap == 1:
        return 70.0  # one level below requirements
    return 20.0  # two levels below (e.g. job=HIGH, candidate=LOW)


def _priority_bucket(score: float, config: MatchConfig) -> str:
    if score >= config.high_priority_threshold:
        return "HIGH_PRIORITY"
    if score >= config.good_match_threshold:
        return "GOOD_MATCH"
    if score >= config.low_priority_threshold:
        return "LOW_PRIORITY"
    return "SKIP"


# ---------------------------------------------------------------------------
# Top-level decision
# ---------------------------------------------------------------------------


def decide(
    job: NormalizedJob,
    profile: CandidateProfile,
    config: MatchConfig,
    matched_skills: Optional[list[str]] = None,
    missing_skills: Optional[list[str]] = None,
    partial_skills: Optional[list[str]] = None,
) -> MatchResult:
    """Single entry point: applies hard rules first (cheap, no LLM
    needed), then scores the job if it survives."""
    hard_fail_reasons = check_hard_rules(job, profile, config)
    if hard_fail_reasons:
        return MatchResult(
            overall_score=0.0,
            decision="SKIP",
            priority="SKIP",
            skip_reasons=hard_fail_reasons,
        )

    result = score_job(job, profile, config, matched_skills, missing_skills, partial_skills)

    too_many_missing = len(missing_skills or []) > config.max_missing_required_skills
    if result.overall_score < config.min_match_score or too_many_missing:
        result.decision = "SKIP"
        result.priority = "SKIP"
        if too_many_missing:
            result.skip_reasons.append(
                f"{len(missing_skills or [])} required skills missing "
                f"(max allowed: {config.max_missing_required_skills})"
            )
        if result.overall_score < config.min_match_score:
            result.skip_reasons.append(
                f"Overall score {result.overall_score} below minimum {config.min_match_score}"
            )
        return result

    result.decision = "APPLY"
    result.priority = _priority_bucket(result.overall_score, config)
    return result


__all__ = [
    "CandidateProfile",
    "MatchConfig",
    "check_hard_rules",
    "score_job",
    "decide",
]
