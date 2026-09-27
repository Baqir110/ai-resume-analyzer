# app/services/jobs/finder.py
"""
Automated job search engine targeting tech roles in Germany.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from typing import Any, Dict, List

from app.core.security import UnsafeInputError, validate_public_http_url

logger = logging.getLogger(__name__)

# jobspy uses multiprocessing internally. On Windows, importing gensim/sklearn
# weights before jobspy corrupts its multiprocessing context and causes os._exit().
# Fix: run jobspy in a clean subprocess that has no prior model imports.
_JOBSPY_WORKER = r"""
import json
import math
import sys
import warnings

from jobspy import scrape_jobs

warnings.filterwarnings("ignore")

search_term = sys.argv[1]
location = sys.argv[2]
results = int(sys.argv[3])
hours = int(sys.argv[4])


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


try:
    frame = scrape_jobs(
        site_name=["indeed", "linkedin", "google"],
        search_term=search_term,
        location=location,
        results_wanted=results,
        hours_old=hours,
        country_indeed="germany",
    )
    jobs = []
    for _, row in frame.iterrows():
        url = safe_text(row.get("job_url")) or safe_text(row.get("site_url"))
        if not url:
            continue
        jobs.append(
            {
                "job_url": url,
                "title": safe_text(row.get("title")) or search_term,
                "company": safe_text(row.get("company")) or "Unknown",
                "location": safe_text(row.get("location")) or location,
                "description": safe_text(row.get("description")),
                "site": safe_text(row.get("site")) or "unknown",
            }
        )
    print(json.dumps(jobs))
except Exception:
    print(json.dumps([]))
"""


def _run_jobspy_subprocess(
    search_term: str,
    location: str,
    results_wanted: int,
    hours_old: int,
) -> list[dict]:
    """Run jobspy in an isolated subprocess to avoid gensim/sklearn interference."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _JOBSPY_WORKER,
                search_term,
                location,
                str(results_wanted),
                str(hours_old),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("jobspy subprocess exited with status %s", result.returncode)
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        if output_lines:
            payload = json.loads(output_lines[-1])
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
    except subprocess.TimeoutExpired:
        logger.warning("jobspy subprocess timed out for '%s' in %s", search_term, location)
    except Exception as exc:
        logger.warning("jobspy subprocess error: %s", exc)
    return []


def _safe_str(value: Any) -> str:
    """Coerce a jobspy dataframe cell to str, handling NaN floats.

    pandas represents missing text cells as `float('nan')`, and NaN is
    truthy in Python, so `value or ""` does not catch it -- this was the
    source of the `'float' object has no attribute 'strip'` crash.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _clean_job_result(
    raw: object,
    search_term: str,
    default_location: str,
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    job_url = _safe_str(raw.get("job_url")) or _safe_str(raw.get("site_url"))
    try:
        job_url = validate_public_http_url(job_url, resolve_dns=False)
    except UnsafeInputError:
        return None
    return {
        "job_url": job_url,
        "title": _safe_str(raw.get("title")) or search_term,
        "company": _safe_str(raw.get("company")) or "Unknown",
        "location": _safe_str(raw.get("location")) or default_location,
        "description": _safe_str(raw.get("description")),
        "site": _safe_str(raw.get("site")) or "unknown",
    }


def get_profile_target_title() -> str:
    """Read the target role title from the local profile when available."""
    from app.services.jobs.profile_manager import load_raw_profile

    profile = load_raw_profile() or {}
    experience = profile.get("experience", [])
    if isinstance(experience, list) and experience:
        first = experience[0]
        if isinstance(first, dict) and first.get("title"):
            return str(first["title"])
    personal = profile.get("personal", {})
    if isinstance(personal, dict) and personal.get("target_title"):
        return str(personal["target_title"])
    return "DevOps Engineer"


def find_jobs_germany(
    search_term: str | None = None,
    location: str = "Germany",
    results_wanted: int = 20,
    hours_old: int = 72,
) -> List[Dict[str, Any]]:
    """
    Scrapes job platforms in Germany (Indeed, LinkedIn, Google Jobs).
    """
    term = (search_term or get_profile_target_title()).strip()
    location = location.strip()
    if not term or len(term) > 200 or not location or len(location) > 200:
        raise ValueError("search_term and location must be between 1 and 200 characters")
    results_wanted = max(1, min(int(results_wanted), 100))
    hours_old = max(1, min(int(hours_old), 24 * 30))
    logger.info("Starting automated job search for '%s' in %s...", term, location)
    valid_jobs: List[Dict[str, Any]] = []

    raw_jobs = _run_jobspy_subprocess(term, location, results_wanted, hours_old)
    for raw_job in raw_jobs:
        cleaned = _clean_job_result(raw_job, term, location)
        if cleaned is not None:
            valid_jobs.append(cleaned)

    # Fallback: broaden to all of Germany if primary location returned nothing
    if not valid_jobs and location.lower() not in ("germany", "deutschland"):
        fallback_location = "Germany"
        logger.info(
            "Zero listings for '%s' in '%s'. Retrying with location='%s'...",
            term,
            location,
            fallback_location,
        )
        raw_jobs = _run_jobspy_subprocess(term, fallback_location, results_wanted, 168)
        for raw_job in raw_jobs:
            cleaned = _clean_job_result(raw_job, term, fallback_location)
            if cleaned is not None:
                valid_jobs.append(cleaned)

    logger.info("Found %d valid job postings.", len(valid_jobs))
    return valid_jobs
