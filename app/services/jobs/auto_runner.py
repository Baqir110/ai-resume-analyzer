"""Batch job-search runner for the automated application pipeline."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from app.core.config import settings
from app.core.security import UnsafeInputError, validate_public_http_url
from app.services.jobs.agent_schemas import ApplicationState, NormalizedJob
from app.services.jobs.dedupe import Deduplicator, compute_dedupe_key
from app.services.jobs.finder import _safe_str, find_jobs_germany
from app.services.jobs.orchestrator import JobAgentOrchestrator, run_application_with_validation
from app.services.tracking.state_machine import AUTOMATION_QUEUE_STATES, ApplicationStateMachine

logger = logging.getLogger(__name__)


# ============================================================================
# PROVIDER / MODEL DETECTION
# ============================================================================


def _detect_provider(model_name: str) -> str:
    """Infer provider from a model name."""
    model = (model_name or "").strip().lower()

    if "experiential" in model:
        return "experiential"

    if "luna" in model or "astra" in model or "terra" in model:
        return "experiential"

    if "fable" in model:
        return "experiential"

    if "qwen3.8" in model:
        return "experiential"

    if "deepseek-v4" in model:
        return "experiential"

    if "gemini" in model:
        return "gemini"

    if "openai" in model or "gpt-" in model:
        return "openai"

    if "claude" in model or "anthropic" in model:
        return "claude"

    if "groq" in model:
        return "groq"

    if "openrouter" in model:
        return "openrouter"

    if "ollama" in model:
        return "ollama"

    if "deepseek" in model:
        return "deepseek"

    return "gemini"


def _detect_route_mode(provider: str) -> str:
    """Determine routing mode."""
    provider = (provider or "").strip().lower()

    if provider == "experiential":
        return "experiential"

    return "direct"


# ============================================================================
# RESUME
# ============================================================================


def _extract_resume_text(resume_path: Path) -> str:
    """Extract text from the base resume PDF."""
    try:
        reader = PdfReader(str(resume_path))
    except Exception as exc:
        raise RuntimeError(f"Could not open resume PDF '{resume_path}': {exc}") from exc

    pages: List[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Could not extract one resume page: %s",
                exc,
            )
            text = ""

        if text.strip():
            pages.append(text)

    resume_text = "\n".join(pages).strip()

    if not resume_text:
        raise ValueError(f"No text could be extracted from resume: {resume_path}")

    return resume_text


# ============================================================================
# JOB NORMALIZATION
# ============================================================================


def _normalize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JobSpy output into the pipeline format without NaN strings."""
    job_url = _safe_str(job.get("job_url")) or _safe_str(job.get("url"))
    application_url = (
        _safe_str(job.get("application_url")) or _safe_str(job.get("apply_url")) or job_url
    )
    return {
        "job_id": _safe_str(job.get("job_id")) or _safe_str(job.get("id")) or job_url,
        "title": _safe_str(job.get("title")) or "Unknown",
        "company": _safe_str(job.get("company")) or "Unknown",
        "description": _safe_str(job.get("description")) or _safe_str(job.get("job_description")),
        "job_url": job_url,
        "application_url": application_url,
        "location": _safe_str(job.get("location")),
        "site": _safe_str(job.get("site")),
        "job_type": _safe_str(job.get("job_type")),
        "date_posted": _safe_str(job.get("date_posted")),
        "raw": job,
    }


def _queue_job(job: Dict[str, Any]) -> NormalizedJob:
    url = str(job.get("application_url") or job.get("apply_url") or job.get("job_url") or "")
    return NormalizedJob(
        job_id=str(job.get("job_id") or job.get("id") or url),
        title=str(job.get("title") or "Unknown"),
        company=str(job.get("company") or "Unknown"),
        location=str(job.get("location") or ""),
        description=str(job.get("description") or job.get("job_description") or ""),
        application_url=url,
        source=str(job.get("site") or job.get("source") or ""),
    )


def _job_from_record(record: dict[str, Any]) -> NormalizedJob:
    return NormalizedJob(
        job_id=str(record.get("id") or ""),
        title=str(record.get("job_title") or "Unknown"),
        company=str(record.get("company_name") or "Unknown"),
        location=str(record.get("location") or ""),
        description=str(record.get("job_description") or ""),
        application_url=str(record.get("job_url") or ""),
        source=str(record.get("source") or ""),
        fingerprint=str(record.get("fingerprint") or ""),
    )


def _pipeline_payload(job: NormalizedJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "title": job.title,
        "company": job.company,
        "description": job.description,
        "job_url": job.application_url,
        "application_url": job.application_url,
        "location": job.location,
        "site": job.source,
    }


# ============================================================================
# MAIN RUNNER
# ============================================================================


async def run_automated_job_hunting(
    search_term: str = "DevOps Engineer",
    location: str = "Germany",
    resume_path: str = "data/resume.pdf",
    max_applications: int = 5,
    auto_submit: bool = False,
    model_name: str = "gemini-2.5-flash",
    agent_llm: str = "direct-gemini",
    provider: Optional[str] = None,
    route_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search jobs and run the complete application pipeline."""

    if max_applications < 1:
        raise ValueError("max_applications must be at least 1")
    daily_limit = min(max_applications, max(0, int(settings.MAX_DAILY_APPLICATIONS)))

    resume_file = Path(resume_path)

    if not resume_file.exists():
        raise FileNotFoundError(f"Resume not found: {resume_file.resolve()}")

    resume_text = _extract_resume_text(resume_file)

    provider = provider or _detect_provider(model_name)

    route_mode = route_mode or _detect_route_mode(provider)

    logger.info(
        "LLM configuration: provider=%s, " "route_mode=%s, text_model=%s, " "agent_model=%s",
        provider,
        route_mode,
        model_name,
        agent_llm,
    )

    logger.info(
        "Starting automated job search for '%s' in %s...",
        search_term,
        location,
    )

    jobs = await asyncio.to_thread(
        find_jobs_germany,
        search_term=search_term,
        location=location,
        results_wanted=max(
            max_applications * 2,
            max_applications,
        ),
    )

    if not jobs:
        logger.warning("No new valid jobs were found; persistent queue will be recovered")

    logger.info(
        "Found %d valid job postings.",
        len(jobs),
    )

    results: List[Dict[str, Any]] = []
    processed_count = 0

    known = ApplicationStateMachine.known_fingerprints(exclude_states=AUTOMATION_QUEUE_STATES)
    known_soft = ApplicationStateMachine.known_dedupe_keys(exclude_states=AUTOMATION_QUEUE_STATES)
    dedupe = Deduplicator(
        existing_fingerprints=known,
        existing_soft_fingerprints=known_soft,
    )
    candidates: List[NormalizedJob] = []
    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            logger.warning("Skipping invalid non-dictionary job result")
            continue
        candidate = _queue_job(_normalize_job(raw_job))
        if not candidate.application_url:
            logger.warning(
                "Skipping '%s' at %s: no application URL.",
                candidate.title,
                candidate.company,
            )
            continue
        try:
            validate_public_http_url(candidate.application_url, resolve_dns=False)
        except UnsafeInputError as exc:
            logger.warning("Skipping unsafe application URL: %s", exc)
            continue
        candidates.append(candidate)

    queued_by_fingerprint: dict[str, NormalizedJob] = {
        job.fingerprint: job for job in dedupe.dedupe_batch(candidates)
    }
    for job in list(queued_by_fingerprint.values()):
        record = ApplicationStateMachine.create(
            company_name=job.company,
            job_title=job.title,
            job_url=job.application_url,
            fingerprint=job.fingerprint,
            dedupe_key=compute_dedupe_key(job),
            source=job.source,
            job_description=job.description,
            location=job.location,
        )
        job.fingerprint = str(record.get("fingerprint") or job.fingerprint)
        state = ApplicationState(record.get("state") or "DISCOVERED")
        if state not in AUTOMATION_QUEUE_STATES:
            queued_by_fingerprint.pop(job.fingerprint, None)
        else:
            queued_by_fingerprint[job.fingerprint] = _job_from_record(record)

    for record in ApplicationStateMachine.list_automation_queue():
        fingerprint = str(record.get("fingerprint") or "")
        queued_by_fingerprint.setdefault(fingerprint, _job_from_record(record))

    recorder = JobAgentOrchestrator(
        SimpleNamespace(
            MINIMUM_MATCH_SCORE=60.0,
            HIGH_PRIORITY_THRESHOLD=85.0,
            GOOD_MATCH_THRESHOLD=70.0,
            MAX_MISSING_REQUIRED_SKILLS=2,
            MAX_CONCURRENT_BROWSER_WORKERS=1,
            MAX_DAILY_APPLICATIONS=daily_limit,
            AUTOMATIC_APPLY=True,
            AUTOMATIC_SUBMIT=auto_submit,
            FOLLOW_UP_DAYS=14,
        ),
        resume_path=str(resume_file),
    )

    for queue_job in queued_by_fingerprint.values():
        if processed_count >= max_applications:
            break

        job = _pipeline_payload(queue_job)
        job_title = job["title"]
        company_name = job["company"]
        job_url = job["application_url"]
        record = ApplicationStateMachine.find_by_fingerprint(queue_job.fingerprint)
        if record is None:
            continue
        app_id = int(record["id"])
        state = ApplicationState(record.get("state") or "DISCOVERED")

        _app_id, match_result = await recorder._analyze_and_transition(queue_job, resume_text)
        if match_result is None or match_result.decision != "APPLY":
            logger.info("SKIP %s at %s after scoring", job_title, company_name)
            continue
        current_record = ApplicationStateMachine.get(app_id)
        if current_record is None:
            continue
        state = ApplicationState(current_record.get("state") or state.value)

        if not ApplicationStateMachine.reserve_daily_application(
            app_id,
            daily_limit,
        ):
            if ApplicationStateMachine.count_daily_applications() >= daily_limit:
                results.append(
                    {
                        "status": "daily_budget_exhausted",
                        "success": False,
                        "job_id": queue_job.job_id,
                        "job_title": job_title,
                        "company": company_name,
                        "job_url": job_url,
                    }
                )
                break
            # This application was already claimed by another worker. Keep it
            # queued and continue with jobs that still have capacity.
            continue

        if state == ApplicationState.MATCHED:
            ApplicationStateMachine.transition(
                app_id, ApplicationState.PREPARING, "starting pipeline"
            )

        processed_count += 1

        logger.info(
            "Processing job %d/%d: %s at %s",
            processed_count,
            max_applications,
            job_title,
            company_name,
        )

        try:
            result = await run_application_with_validation(
                job=job,
                resume_text=resume_text,
                profile={},
                model_name=model_name,
                agent_llm=agent_llm,
                auto_submit=auto_submit,
                provider=provider,
                route_mode=route_mode,
            )

            if not isinstance(result, dict):
                result = {
                    "status": "completed",
                    "success": False,
                    "result": str(result),
                }

            result.setdefault(
                "job_id",
                job.get("job_id"),
            )
            result.setdefault(
                "job_title",
                job_title,
            )
            result.setdefault(
                "company",
                company_name,
            )
            result.setdefault(
                "job_url",
                job_url,
            )
            recorder._record_pipeline_result(app_id, result, queue_job)

            results.append(result)

            logger.info(
                "Finished job %d/%d: %s at %s | " "ATS %.2f%% -> %.2f%%",
                processed_count,
                max_applications,
                job_title,
                company_name,
                result.get(
                    "original_ats_score",
                    0.0,
                ),
                result.get(
                    "final_ats_score",
                    0.0,
                ),
            )

        except Exception:
            logger.exception(
                "Application pipeline failed for '%s' at '%s'.",
                job_title,
                company_name,
            )
            failure_result = {
                "status": "error",
                "success": False,
                "submitted": False,
                "error": "Application pipeline failed",
            }
            recorder._record_pipeline_result(app_id, failure_result, queue_job)
            results.append(
                {
                    **failure_result,
                    "job_id": job.get("job_id"),
                    "job_title": job_title,
                    "company": company_name,
                    "job_url": job_url,
                }
            )

    logger.info(
        "Automated job hunting finished. " "Processed %d jobs.",
        processed_count,
    )

    return results


__all__ = [
    "run_automated_job_hunting",
    "_detect_provider",
    "_detect_route_mode",
]
