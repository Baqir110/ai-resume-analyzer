# app/services/jobs/profile_manager.py
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_WRITE_LOCK = threading.RLock()


def _configured_profile_path() -> Path:
    raw = Path(getattr(settings, "APPLICANT_PROFILE_PATH", "data/applicant_profile.local.yaml"))
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    return raw


PROFILE_PATH = _configured_profile_path()


def load_raw_profile() -> dict[str, Any]:
    """Load the local profile, falling back to the safe template.

    A live profile is intentionally not tracked in Git.  The fallback keeps
    the application importable in a fresh checkout while ensuring missing
    personal data fails validation rather than being invented.
    """
    candidates = [_configured_profile_path()]
    safe_template = PROJECT_ROOT / "data" / "applicant_profile.yaml"
    if safe_template not in candidates:
        candidates.append(safe_template)
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            return loaded if isinstance(loaded, dict) else {}
    return {}


def _years_of_experience(raw: dict[str, Any]) -> float:
    """Sum experience-entry durations. Best-effort MM/YYYY parsing;
    an entry the parser can't read is skipped rather than guessed."""
    total_months = 0.0
    for entry in raw.get("experience", []) or []:
        start = _parse_month(entry.get("start_date", ""))
        end = (
            datetime.now(timezone.utc)
            if entry.get("current")
            else _parse_month(entry.get("end_date", ""))
        )
        if start and end and end >= start:
            total_months += (end.year - start.year) * 12 + (end.month - start.month)
    return round(total_months / 12, 1)


def _parse_month(value: str):
    value = (value or "").strip()
    for fmt in ("%m/%Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def load_candidate_profile(config: Any = None):
    """Build a decision_engine.CandidateProfile from applicant_profile.yaml
    + app config, so the decision engine never has to parse YAML itself.
    """
    from app.services.jobs.decision_engine import CandidateProfile

    raw = load_raw_profile()
    work_auth = raw.get("work_authorization", {}) or {}
    if not isinstance(work_auth, dict):
        work_auth = {}
    education = raw.get("education", []) or []
    raw_authorized = work_auth.get("authorized_to_work") if isinstance(work_auth, dict) else None
    if isinstance(raw_authorized, str):
        normalized_auth = raw_authorized.strip().casefold()
        authorized_value = (
            True
            if normalized_auth in {"true", "yes", "1"}
            else False
            if normalized_auth in {"false", "no", "0"}
            else None
        )
    elif isinstance(raw_authorized, bool):
        authorized_value = raw_authorized
    else:
        authorized_value = None
    raw_countries = (
        work_auth.get("work_authorized_countries") or work_auth.get("authorized_countries") or []
    )
    if isinstance(raw_countries, str):
        raw_countries = [raw_countries]
    authorized_countries = {
        str(country).strip().casefold() for country in (raw_countries or []) if str(country).strip()
    }
    authorization_known = authorized_value is not None

    highest_degree = ""
    for entry in education:
        degree = (entry.get("degree") or "").lower()
        if "m.sc" in degree or "master" in degree:
            highest_degree = "master"
            break
        if "b.sc" in degree or "bachelor" in degree:
            highest_degree = highest_degree or "bachelor"
        if "phd" in degree or "dr." in degree:
            highest_degree = "phd"
            break

    languages = {}
    if config is not None:
        languages = {k.lower(): v.lower() for k, v in getattr(config, "LANGUAGES", {}).items()}

    allowed_locations = set()
    if config is not None:
        allowed_locations = {loc.lower() for loc in getattr(config, "TARGET_LOCATIONS", [])}

    return CandidateProfile(
        skills=set(),  # skill matching stays with ats_analyzer; decision_engine takes matched/missing lists directly
        years_experience=_years_of_experience(raw),
        education_level=highest_degree,
        languages=languages,
        allowed_locations=allowed_locations,
        remote_ok=(getattr(config, "REMOTE_PREFERENCE", "hybrid_ok") != "onsite_ok")
        if config
        else True,
        requires_sponsorship=bool(work_auth.get("requires_sponsorship", False)),
        citizenship=(work_auth.get("citizenship") or "").lower(),
        work_authorized_countries=authorized_countries,
        work_authorization_known=authorization_known,
        authorized_to_work=authorized_value,
        minimum_salary=getattr(config, "MINIMUM_SALARY", None) if config else None,
        salary_currency=getattr(config, "SALARY_CURRENCY", "EUR") if config else "EUR",
        willing_to_relocate=True,
    )


def build_free_text_answers(raw_profile: dict[str, Any] | None = None) -> dict[str, str]:
    """Flat question->answer map for the application-answer engine
    (Section 14). Combines the structured profile with any saved
    free_text answers, so common questions never need an LLM guess."""
    raw = raw_profile if raw_profile is not None else load_raw_profile()
    personal = raw.get("personal", {}) or {}
    work_auth = raw.get("work_authorization", {}) or {}

    answers = {
        "first_name": personal.get("first_name", ""),
        "last_name": personal.get("last_name", ""),
        "full_name": personal.get("full_name", ""),
        "email": personal.get("email", ""),
        "phone": personal.get("phone", ""),
        "city": personal.get("city", ""),
        "country": personal.get("country", ""),
        "linkedin": personal.get("linkedin", ""),
        "github": personal.get("github", ""),
        "authorized_to_work": "Yes" if work_auth.get("authorized_to_work") else "No",
        "requires_sponsorship": "Yes" if work_auth.get("requires_sponsorship") else "No",
    }
    answers.update(raw.get("free_text", {}) or {})
    return answers


def save_custom_answer(question: str, answer: str) -> None:
    """Save a custom answer only to an explicitly configured local profile."""
    target = _configured_profile_path()
    if target == PROJECT_ROOT / "data" / "applicant_profile.yaml":
        raise RuntimeError("Configure APPLICANT_PROFILE_PATH before saving answers")
    if not target.exists():
        raise FileNotFoundError(f"Local applicant profile not found: {target}")

    with _PROFILE_WRITE_LOCK:
        with target.open("r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}
        if not isinstance(profile, dict):
            raise ValueError("Applicant profile must be a mapping")
        free_text = profile.setdefault("free_text", {})
        if not isinstance(free_text, dict):
            raise ValueError("Applicant profile free_text must be a mapping")
        free_text[question] = answer

        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_name, target)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
