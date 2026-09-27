# app/services/jobs/orchestrator.py
"""Ties discovery, dedupe, decision engine, existing CV/cover-letter/
browser pipeline, and the state machine into one autonomous run.

This is the "zero-question autonomous mode" entry point (Section 19):
every decision is made from config + candidate profile, nothing here
calls input(). Errors in one job never stop the batch (Section 38).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.services.analysis.ats_analyzer import analyze_resume
from app.services.jobs.agent_schemas import ApplicationState, NormalizedJob
from app.services.jobs.decision_engine import MatchConfig, decide
from app.services.jobs.dedupe import Deduplicator, compute_dedupe_key
from app.services.jobs.discovery import DiscoveryConfig, DiscoveryEngine
from app.services.jobs.profile_manager import (
    build_free_text_answers,
    load_candidate_profile,
    load_raw_profile,
)
from app.services.tracking.state_machine import ApplicationStateMachine

logger = logging.getLogger(__name__)


class PackageValidationError(RuntimeError):
    """Raised when an application package is unsafe to send to a browser."""


_BrowserRunner = Callable[..., Awaitable[dict[str, Any]]]
_BrowserGate = Callable[[tuple[Any, ...], dict[str, Any]], Awaitable[dict[str, Any]]]
_browser_gate: ContextVar[Optional[_BrowserGate]] = ContextVar(
    "job_application_browser_gate", default=None
)
_gate_install_lock = threading.Lock()
_real_browser_runner: Optional[_BrowserRunner] = None


def _install_browser_package_gate() -> None:
    """Install a task-local gate at full_pipeline's browser boundary.

    ``full_pipeline`` combines document generation and browser execution in a
    single public function.  The dispatcher validates the exact generated CV,
    cover letter, profile, and answers immediately before delegating to the
    real browser worker.  A ContextVar keeps concurrent application tasks
    isolated, while calls without a gate continue to the original runner.
    """
    global _real_browser_runner

    from app.services.jobs import full_pipeline

    current = full_pipeline._run_browser_application
    if getattr(current, "_package_gate_dispatcher", False):
        return
    with _gate_install_lock:
        current = full_pipeline._run_browser_application
        if getattr(current, "_package_gate_dispatcher", False):
            return
        _real_browser_runner = current

        async def dispatcher(*args: Any, **kwargs: Any) -> dict[str, Any]:
            gate = _browser_gate.get()
            if gate is not None:
                return await gate(args, kwargs)
            assert _real_browser_runner is not None
            return await _real_browser_runner(*args, **kwargs)

        dispatcher._package_gate_dispatcher = True  # type: ignore[attr-defined]
        full_pipeline._run_browser_application = dispatcher


def _normalized_job_from_payload(payload: dict[str, Any]) -> NormalizedJob:
    return NormalizedJob(
        job_id=str(payload.get("job_id") or payload.get("id") or ""),
        title=str(payload.get("title") or payload.get("job_title") or ""),
        company=str(payload.get("company") or payload.get("company_name") or ""),
        location=str(payload.get("location") or ""),
        description=str(payload.get("description") or payload.get("job_description") or ""),
        application_url=str(
            payload.get("application_url")
            or payload.get("apply_url")
            or payload.get("job_url")
            or payload.get("url")
            or ""
        ),
    )


def _job_from_record(record: dict[str, Any]) -> NormalizedJob:
    return NormalizedJob(
        job_id=str(record.get("id") or ""),
        title=str(record.get("job_title") or "Unknown"),
        company=str(record.get("company_name") or "Unknown"),
        location=str(record.get("location") or ""),
        description=str(record.get("job_description") or ""),
        application_url=str(record.get("job_url") or ""),
        original_url=str(record.get("job_url") or ""),
        source=str(record.get("source") or ""),
        fingerprint=str(record.get("fingerprint") or ""),
    )


def _validate_application_package(
    job: NormalizedJob,
    cv_path: str,
    cover_letter_path: str | None,
    profile: dict[str, Any],
    answers: dict[str, str],
    require_cover_letter: bool,
):
    from app.services.jobs.backend_submitter import ApplicationPackage
    from app.services.jobs.package_validator import validate_package

    package = ApplicationPackage(
        job=job,
        cv_path=cv_path,
        cover_letter_path=cover_letter_path,
        answers=answers,
        profile=profile,
        expected_fingerprint=job.fingerprint,
    )
    return validate_package(
        package,
        require_cover_letter=require_cover_letter,
    )


async def run_application_with_validation(
    job: dict[str, Any],
    resume_text: str,
    profile: dict[str, Any] | None = None,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """Run the existing pipeline with a fail-closed pre-browser package gate."""
    from app.services.jobs import full_pipeline

    _install_browser_package_gate()
    normalized_job = _normalized_job_from_payload(job)
    raw_profile = profile if profile else load_raw_profile()
    answers = build_free_text_answers(raw_profile)
    description = normalized_job.description
    require_cover_letter = bool(
        getattr(full_pipeline, "_cover_letter_required", lambda _text: False)(description)
    )

    async def gate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        bound = dict(zip(("application_url",), args))
        bound.update(kwargs)
        cv_path = str(bound.get("resume_path") or "")
        cover_value = bound.get("cover_letter_path")
        cover_path = str(cover_value) if cover_value else None
        validation = _validate_application_package(
            normalized_job,
            cv_path,
            cover_path,
            raw_profile,
            answers,
            require_cover_letter,
        )
        if not validation.valid:
            raise PackageValidationError("Package validation error")
        assert _real_browser_runner is not None
        browser_result = await _real_browser_runner(*args, **kwargs)
        if not isinstance(browser_result, dict):
            browser_result = dict(browser_result or {})
        browser_result["_package_validated"] = True
        return browser_result

    token = _browser_gate.set(gate)
    try:
        result = await full_pipeline.run_full_application(
            job=job,
            resume_text=resume_text,
            profile=raw_profile,
            **pipeline_kwargs,
        )
    finally:
        _browser_gate.reset(token)

    if not isinstance(result, dict):
        raise TypeError("Application pipeline must return a dictionary")

    # Revalidate after generation as a TOCTOU/postcondition check.  This also
    # keeps mocked/test pipelines fail-closed when they do not traverse the
    # browser boundary.
    validation = _validate_application_package(
        normalized_job,
        str(result.get("cv_path") or ""),
        str(result.get("cover_letter_path") or "") or None,
        raw_profile,
        answers,
        bool(result.get("cover_letter_required")),
    )
    if not validation.valid:
        raise PackageValidationError("Package validation error")
    return result


class JobAgentOrchestrator:
    def __init__(self, config: Any, resume_path: str = "data/resume.pdf"):
        self.config = config
        self.resume_path = resume_path
        self.match_config = MatchConfig(
            min_match_score=config.MINIMUM_MATCH_SCORE,
            high_priority_threshold=config.HIGH_PRIORITY_THRESHOLD,
            good_match_threshold=config.GOOD_MATCH_THRESHOLD,
            max_missing_required_skills=config.MAX_MISSING_REQUIRED_SKILLS,
        )
        self._browser_semaphore = asyncio.Semaphore(max(1, config.MAX_CONCURRENT_BROWSER_WORKERS))

    # ------------------------------------------------------------------
    # DISCOVER
    # ------------------------------------------------------------------

    async def discover(self) -> list[NormalizedJob]:
        from app.services.observability.structured_logger import log_event
        from app.services.tracking.live_status import set_stage

        set_stage("DISCOVERY")
        log_event("DISCOVERY_STARTED")

        from app.services.tracking.state_machine import AUTOMATION_QUEUE_STATES

        known = ApplicationStateMachine.known_fingerprints(exclude_states=AUTOMATION_QUEUE_STATES)
        dedupe = Deduplicator(existing_fingerprints=known)

        _KNOWN_SOURCES = {"jobspy", "ats_direct", "arbeitsagentur", "web_search"}
        unknown = set(self.config.SOURCES_ENABLED) - _KNOWN_SOURCES
        if unknown:
            logger.warning(
                "SOURCES_ENABLED contains unrecognized source(s) %s -- these are "
                "silently ignored. Known sources: %s. StepStone and XING have no "
                "public discovery API and are not implemented.",
                sorted(unknown),
                sorted(_KNOWN_SOURCES),
            )

        company_boards = []
        from app.services.jobs.company_boards import CompanyBoard

        for ats, handle, name in getattr(self.config, "COMPANY_BOARDS", []):
            company_boards.append(CompanyBoard(ats=ats, handle=handle, company_name=name))

        engine = DiscoveryEngine()
        all_new_jobs: list[NormalizedJob] = []

        # Build geo-location list — strip "Remote" which jobspy cannot geocode.
        # If all locations are "remote" or the list is empty, fall back to Germany.
        geo_locations = [
            loc
            for loc in self.config.TARGET_LOCATIONS
            if loc.lower() not in ("remote", "worldwide")
        ] or ["Germany"]

        for role in self.config.TARGET_ROLES:
            for location in geo_locations:
                disc_config = DiscoveryConfig(
                    search_term=role,
                    location=location,
                    results_wanted=30,
                    hours_old=168,  # 7 days — wider window catches quieter periods
                    company_boards=company_boards,
                    enable_jobspy="jobspy" in self.config.SOURCES_ENABLED,
                    enable_ats_direct="ats_direct" in self.config.SOURCES_ENABLED,
                    enable_web_search="web_search" in self.config.SOURCES_ENABLED,
                    enable_arbeitsagentur="arbeitsagentur" in self.config.SOURCES_ENABLED,
                )
                jobs = await engine.discover_all(disc_config, dedupe=dedupe)
                all_new_jobs.extend(jobs)

        queued_by_fingerprint: dict[str, NormalizedJob] = {}
        for job in all_new_jobs:
            record = ApplicationStateMachine.create(
                company_name=job.company,
                job_title=job.title,
                job_url=job.application_url or job.original_url,
                fingerprint=job.fingerprint,
                dedupe_key=compute_dedupe_key(job),
                source=job.source,
                job_description=job.description,
                location=job.location,
            )
            job.fingerprint = str(record.get("fingerprint") or job.fingerprint)
            state = ApplicationState(record.get("state") or "DISCOVERED")
            if state not in AUTOMATION_QUEUE_STATES:
                continue
            queued_by_fingerprint[job.fingerprint] = _job_from_record(record)
            log_event(
                "JOB_FOUND",
                title=job.title,
                company=job.company,
                source=job.source,
            )

        # Rehydrate persistent queue entries even when a transient source
        # outage means the job was not returned in this discovery pass.  An
        # upsert above enriches these rows; this step only preserves work.
        for record in ApplicationStateMachine.list_automation_queue():
            fingerprint = str(record.get("fingerprint") or "")
            if fingerprint in queued_by_fingerprint:
                continue
            queued_by_fingerprint[fingerprint] = _job_from_record(record)

        queued_jobs = list(queued_by_fingerprint.values())
        log_event("DISCOVERY_COMPLETE", jobs_found=len(queued_jobs))
        logger.info("Discovery queue contains %d job(s)", len(queued_jobs))
        return queued_jobs

    # ------------------------------------------------------------------
    # ANALYZE + DECIDE
    # ------------------------------------------------------------------

    def analyze_and_decide(self, job: NormalizedJob, resume_text: str) -> tuple[Any, dict]:
        analysis = analyze_resume(resume_text=resume_text, job_description=job.description)
        profile = load_candidate_profile(self.config)
        result = decide(
            job,
            profile,
            self.match_config,
            matched_skills=analysis.get("matching_skills", []),
            missing_skills=analysis.get("missing_skills", []),
        )
        return result, analysis

    async def _analyze_and_transition(
        self, job: NormalizedJob, resume_text: str
    ) -> tuple[Optional[int], Any]:
        """Runs decision_engine on one job and applies the corresponding
        ANALYZING -> MATCHED/SKIPPED transition. Shared by run_once() and
        analyze_only() so --analyze exercises the exact same scoring path
        as a real run, just without the apply step after it.

        Returns (application_id, MatchResult) on success, or (None, None)
        if there's no tracker record or analysis raised.
        """
        record = ApplicationStateMachine.find_by_fingerprint(job.fingerprint)
        if record is None:
            logger.warning("No tracker record for job %s -- skipping", job.job_id)
            return None, None
        app_id = record["id"]

        from app.services.tracking.live_status import set_stage

        set_stage("ANALYZING", source=job.source, job_title=job.title, company=job.company)

        current_state = ApplicationState(record.get("state") or ApplicationState.DISCOVERED.value)
        from app.services.tracking.state_machine import AUTOMATION_QUEUE_STATES

        if current_state not in AUTOMATION_QUEUE_STATES:
            logger.info(
                "Application %s is in %s and is not an active queue item",
                app_id,
                current_state.value,
            )
            return app_id, None

        try:
            if current_state == ApplicationState.DISCOVERED:
                ApplicationStateMachine.transition(app_id, ApplicationState.ANALYZING, "scoring")
                current_state = ApplicationState.ANALYZING
            match_result, _analysis = self.analyze_and_decide(job, resume_text)
            ApplicationStateMachine.update_match_score(app_id, match_result.overall_score)
        except Exception:
            logger.exception("Analysis failed for %s at %s", job.title, job.company)
            return app_id, None

        if match_result.decision != "APPLY":
            if current_state != ApplicationState.SKIPPED:
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.SKIPPED,
                    "; ".join(match_result.skip_reasons) or "score below threshold",
                )
            from app.services.observability.structured_logger import log_event

            log_event(
                "DECISION",
                application_id=app_id,
                decision="SKIP",
                score=match_result.overall_score,
                reasons=match_result.skip_reasons,
            )
        else:
            if current_state == ApplicationState.ANALYZING:
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.MATCHED,
                    f"score={match_result.overall_score} priority={match_result.priority}",
                )
            from app.services.observability.structured_logger import log_event

            log_event(
                "DECISION",
                application_id=app_id,
                decision="APPLY",
                score=match_result.overall_score,
                priority=match_result.priority,
            )
        return app_id, match_result

    async def analyze_only(self) -> list[dict[str, Any]]:
        """Section 36 --analyze: discover + score + decide, but never
        generates a CV or touches the browser. Useful for previewing what
        the agent *would* do before trusting --prepare/--apply."""
        resume_file = Path(self.resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume not found: {resume_file.resolve()}")
        resume_text = _extract_resume_text(resume_file)

        discovered_jobs = await self.discover()
        rows: list[dict[str, Any]] = []
        for job in discovered_jobs:
            _app_id, match_result = await self._analyze_and_transition(job, resume_text)
            if match_result is None:
                rows.append(
                    {
                        "title": job.title,
                        "company": job.company,
                        "decision": "ERROR",
                        "priority": "-",
                        "score": 0.0,
                        "reasons": [],
                    }
                )
                continue
            rows.append(
                {
                    "title": job.title,
                    "company": job.company,
                    "decision": match_result.decision,
                    "priority": match_result.priority,
                    "score": match_result.overall_score,
                    "reasons": match_result.skip_reasons,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # FULL AUTONOMOUS PASS: discover -> analyze -> decide -> apply
    # ------------------------------------------------------------------

    async def run_once(self) -> list[dict[str, Any]]:
        resume_file = Path(self.resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume not found: {resume_file.resolve()}")
        resume_text = _extract_resume_text(resume_file)

        discovered_jobs = await self.discover()

        results: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(max(1, self.config.MAX_CONCURRENT_BROWSER_WORKERS))

        async def process(job: NormalizedJob) -> Optional[dict[str, Any]]:
            app_id, match_result = await self._analyze_and_transition(job, resume_text)
            if app_id is None:
                return None
            if match_result is None:
                return {"job": job.as_dict(), "decision": "ERROR"}

            if match_result.decision != "APPLY":
                return {
                    "job": job.as_dict(),
                    "decision": "SKIP",
                    "reasons": match_result.skip_reasons,
                }

            if not self.config.AUTOMATIC_APPLY:
                return {
                    "job": job.as_dict(),
                    "decision": "MATCHED_NOT_APPLIED",
                    "match": match_result.as_dict(),
                }
            try:
                reserved = ApplicationStateMachine.reserve_daily_application(
                    app_id,
                    max(0, int(self.config.MAX_DAILY_APPLICATIONS)),
                )
                if not reserved:
                    return {
                        "job": job.as_dict(),
                        "decision": "DAILY_BUDGET_EXHAUSTED",
                        "match": match_result.as_dict(),
                    }

                pipeline_result = await self._execute_pipeline(app_id, job, resume_text, semaphore)
            except Exception:
                logger.exception("Application %s could not be processed", app_id)
                return {
                    "job": job.as_dict(),
                    "decision": "ERROR",
                    "error": "Application processing failed",
                }
            if pipeline_result is None:
                return {
                    "job": job.as_dict(),
                    "decision": "APPLY",
                    "error": "pipeline failed, see logs",
                }
            pipeline_result["job"] = job.as_dict()
            pipeline_result["match"] = match_result.as_dict()
            return pipeline_result

        for job in discovered_jobs:
            outcome = await process(job)
            if outcome:
                results.append(outcome)

        used_today = ApplicationStateMachine.count_daily_applications()
        logger.info(
            "Autonomous pass complete: %d jobs queued, %d processed, "
            "%d UTC-day application slot(s) used",
            len(discovered_jobs),
            len(results),
            used_today,
        )
        from app.services.tracking.live_status import set_stage

        set_stage("IDLE")
        return results

    async def _execute_pipeline(
        self,
        app_id: int,
        job: NormalizedJob,
        resume_text: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[dict[str, Any]]:
        """Runs the existing CV/cover-letter/browser pipeline for one job
        and records the outcome in the state machine. Shared by the normal
        autonomous pass and --retry-failed so both go through identical
        logic.

        Section 20: failures are classified (CAPTCHA/permanent never
        retry; transient retries with backoff; unknown gets a couple of
        conservative attempts) via error_recovery.py, rather than a flat
        try/except-and-give-up.
        """
        from app.services.jobs.error_recovery import RetryPolicy, run_with_recovery

        async def attempt() -> dict[str, Any]:
            try:
                # The full pipeline is imported lazily, but browser execution
                # remains behind the package-validation boundary.
                return await run_application_with_validation(
                    job={
                        "job_id": job.job_id,
                        "title": job.title,
                        "company": job.company,
                        "job_url": job.application_url or job.original_url,
                        "description": job.description,
                    },
                    resume_text=resume_text,
                    profile={},
                    auto_submit=self.config.AUTOMATIC_SUBMIT,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline raised for %s at %s", job.title, job.company)
                return {"success": False, "submitted": False, "error": str(exc)}

        def extract_error(result: dict[str, Any]) -> str:
            browser = result.get("browser_application", {}) or {}
            if result.get("success") or result.get("submitted"):
                return ""  # no error to classify -- treat as success
            return str(result.get("error") or browser.get("error_message") or "")

        async with semaphore:
            ApplicationStateMachine.transition(
                app_id, ApplicationState.PREPARING, "starting pipeline"
            )
            from app.services.tracking.live_status import set_stage

            set_stage("PREPARING", source=job.source, job_title=job.title, company=job.company)
            pipeline_result, attempts, failure_class = await run_with_recovery(
                attempt,
                extract_error,
                RetryPolicy(max_attempts=3, max_attempts_unknown=2),
            )
            set_stage(
                "APPLYING",
                source=job.source,
                job_title=job.title,
                company=job.company,
                retry_count=attempts - 1,
                current_error=extract_error(pipeline_result) if failure_class else "",
            )

            if failure_class is not None and not (
                pipeline_result.get("success") or pipeline_result.get("submitted")
            ):
                reason = extract_error(pipeline_result)
                logger.info(
                    "Pipeline for %s at %s failed after %d attempt(s), classified as %s: %s",
                    job.title,
                    job.company,
                    attempts,
                    failure_class.value,
                    reason,
                )
                if failure_class.value == "captcha":
                    ApplicationStateMachine.transition(
                        app_id, ApplicationState.CAPTCHA_REQUIRED, reason
                    )
                else:
                    ApplicationStateMachine.transition(
                        app_id,
                        ApplicationState.SUBMISSION_FAILED,
                        f"[{failure_class.value}, {attempts} attempt(s)] {reason}",
                    )
                return None

        self._record_pipeline_result(app_id, pipeline_result, job)
        return pipeline_result

    # ------------------------------------------------------------------
    # RETRY
    # ------------------------------------------------------------------

    MAX_RETRY_ATTEMPTS = 3

    async def retry_failed(self) -> list[dict[str, Any]]:
        """Reconstructs each SUBMISSION_FAILED job from its stored tracker
        record (title/company/url/description -- no re-discovery needed,
        since discovery would just dedupe it away as already-known) and
        re-runs the pipeline. Applications that have already hit
        MAX_RETRY_ATTEMPTS are left alone rather than retried forever."""
        resume_file = Path(self.resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume not found: {resume_file.resolve()}")
        resume_text = _extract_resume_text(resume_file)

        failed = ApplicationStateMachine.list_by_state(ApplicationState.SUBMISSION_FAILED)
        semaphore = asyncio.Semaphore(max(1, self.config.MAX_CONCURRENT_BROWSER_WORKERS))
        results: list[dict[str, Any]] = []

        for record in failed:
            app_id = record["id"]
            retry_count = record.get("retry_count") or 0
            if retry_count >= self.MAX_RETRY_ATTEMPTS:
                logger.info(
                    "Application %s already retried %d times (max %d) -- leaving as "
                    "SUBMISSION_FAILED for manual review",
                    app_id,
                    retry_count,
                    self.MAX_RETRY_ATTEMPTS,
                )
                continue

            job = _job_from_record(record)

            try:
                claimed = ApplicationStateMachine.begin_retry(
                    app_id,
                    "automatic retry",
                    daily_limit=max(0, int(self.config.MAX_DAILY_APPLICATIONS)),
                )
            except Exception:
                logger.exception("Could not atomically begin retry for application %s", app_id)
                continue
            if claimed is None:
                logger.info("Application %s retry deferred by UTC-day budget", app_id)
                results.append(
                    {
                        "job": job.as_dict(),
                        "decision": "RETRY_BUDGET_DEFERRED",
                    }
                )
                continue

            pipeline_result = await self._execute_pipeline(app_id, job, resume_text, semaphore)
            if pipeline_result is not None:
                pipeline_result["job"] = job.as_dict()
                results.append(pipeline_result)

        logger.info("Retry pass complete: %d applications retried", len(results))
        return results

    def _record_pipeline_result(
        self,
        app_id: int,
        result: dict[str, Any],
        job: Optional["NormalizedJob"] = None,
    ) -> None:
        from app.services.observability.structured_logger import log_event

        def fail_closed(reason: str) -> None:
            current = ApplicationStateMachine.get(app_id)
            if current is None:
                logger.error("Cannot record failure for missing application %s", app_id)
                return
            current_state = ApplicationState(
                current.get("state") or ApplicationState.DISCOVERED.value
            )
            if current_state == ApplicationState.SUBMISSION_FAILED:
                return
            try:
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.SUBMISSION_FAILED,
                    reason,
                )
            except Exception:
                logger.exception("Could not record fail-closed state for application %s", app_id)
                return
            log_event(
                "SUBMISSION_FAILED",
                application_id=app_id,
                error=reason,
            )

        record = ApplicationStateMachine.get(app_id)
        if record is None:
            logger.error("Cannot record pipeline result for missing application %s", app_id)
            return
        if job is None:
            job = _job_from_record(record)

        # This is deliberately the first state-affecting operation.  Missing
        # identity, a validator exception, or any invalid package leaves the
        # application failed and never opens a browser-facing state.
        try:
            raw_profile = load_raw_profile() or {}
            answers = build_free_text_answers(raw_profile)
            validation = _validate_application_package(
                job,
                str(result.get("cv_path") or ""),
                str(result.get("cover_letter_path") or "") or None,
                raw_profile,
                answers,
                bool(result.get("cover_letter_required")),
            )
        except Exception:
            logger.exception("Package validation could not complete for application %s", app_id)
            fail_closed("Package validation error")
            return

        if not validation.valid:
            logger.warning(
                "Application %s failed package validation (%d error(s))",
                app_id,
                len(validation.errors),
            )
            fail_closed("Package validation error")
            return
        if validation.warnings:
            logger.warning(
                "Application %s has %d package validation warning(s)",
                app_id,
                len(validation.warnings),
            )

        browser = result.get("browser_application", {}) or {}
        cv_generated = bool(result.get("cv_path"))
        cover_generated = bool(result.get("cover_letter_path"))
        submitted = bool(result.get("submitted"))
        captcha = "captcha" in str(browser.get("error_message", "")).lower()

        try:
            current_state = ApplicationState(
                record.get("state") or ApplicationState.DISCOVERED.value
            )
            if current_state not in {
                ApplicationState.PREPARING,
                ApplicationState.RETRYING,
                ApplicationState.CV_GENERATED,
                ApplicationState.COVER_LETTER_GENERATED,
                ApplicationState.READY_FOR_APPLICATION,
            }:
                return

            if cv_generated:
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.CV_GENERATED,
                    f"ats {result.get('original_ats_score')}->{result.get('final_ats_score')}",
                )
                ApplicationStateMachine.record_document(
                    app_id,
                    "cv",
                    str(result["cv_path"]),
                    version=str(result.get("selected_cv_version") or ""),
                )
                log_event(
                    "CV_GENERATED",
                    application_id=app_id,
                    ats_before=result.get("original_ats_score"),
                    ats_after=result.get("final_ats_score"),
                )

            if cover_generated:
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.COVER_LETTER_GENERATED,
                    "cover letter ready",
                )
                ApplicationStateMachine.record_document(
                    app_id, "cover_letter", str(result["cover_letter_path"])
                )
                log_event("COVER_LETTER_GENERATED", application_id=app_id)

            ApplicationStateMachine.transition(
                app_id, ApplicationState.READY_FOR_APPLICATION, "package prepared"
            )
            log_event("APPLICATION_PREPARED", application_id=app_id)
            ApplicationStateMachine.transition(
                app_id, ApplicationState.APPLYING, "browser result received"
            )
            log_event("BROWSER_STARTED", application_id=app_id)

            ApplicationStateMachine.record_browser_session(
                app_id,
                final_url=str(browser.get("final_url") or ""),
                screenshot_path=str(browser.get("screenshot_path") or ""),
                success=bool(browser.get("success") or result.get("success")),
                error_message=str(browser.get("error_message") or result.get("error") or ""),
            )

            if captcha:
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.CAPTCHA_REQUIRED,
                    "captcha encountered",
                )
                log_event("CAPTCHA_REQUIRED", application_id=app_id)
            elif submitted:
                from app.services.jobs.verification import verify_submission

                verification = verify_submission(result)
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.SUBMITTED,
                    "auto-submitted",
                )
                if verification.verified:
                    ApplicationStateMachine.transition(
                        app_id,
                        ApplicationState.VERIFIED,
                        "submission evidence verified",
                    )
                    log_event(
                        "SUBMITTED",
                        application_id=app_id,
                        evidence_count=len(verification.evidence),
                    )
                    try:
                        from app.services.tracking.analytics import schedule_follow_up

                        schedule_follow_up(app_id, getattr(self.config, "FOLLOW_UP_DAYS", 14))
                    except Exception:
                        logger.exception("Could not schedule follow-up for application %s", app_id)
                else:
                    ApplicationStateMachine.transition(
                        app_id,
                        ApplicationState.VERIFICATION_FAILED,
                        verification.reason,
                    )
                    log_event(
                        "VERIFICATION_FAILED",
                        application_id=app_id,
                        reason=verification.reason,
                    )
            elif result.get("success"):
                ApplicationStateMachine.transition(
                    app_id,
                    ApplicationState.READY_TO_SUBMIT,
                    "awaiting manual submit",
                )
                log_event("READY_TO_SUBMIT", application_id=app_id)
            else:
                fail_closed(
                    str(browser.get("error_message") or result.get("error") or "unknown failure")
                )
        except Exception:
            logger.exception(
                "State transition failed while recording pipeline result for app %s",
                app_id,
            )


def _extract_resume_text(resume_file: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(resume_file))
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(f"No text extracted from resume: {resume_file}")
    return text


__all__ = [
    "JobAgentOrchestrator",
    "PackageValidationError",
    "run_application_with_validation",
]
