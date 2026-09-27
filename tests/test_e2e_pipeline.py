"""End-to-end mock pipeline test — no network, no LLM, no browser.

Exercises the orchestrator's prepare loop end-to-end using mocked
dependencies so the full state-machine flow (DISCOVERED → MATCHED →
PREPARING → CV_GENERATED → ...) is exercised without any real I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.services.jobs.agent_schemas import ApplicationState, NormalizedJob

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(job_id: str, title: str = "DevOps Engineer") -> NormalizedJob:
    return NormalizedJob(
        job_id=job_id,
        title=title,
        company="Acme GmbH",
        location="Munich",
        description="We need DevOps. Kubernetes, Docker, CI/CD.",
        application_url=f"https://boards.greenhouse.io/acme/jobs/{job_id}",
        source="greenhouse",
    )


def _fake_run_result(success: bool = True) -> dict:
    return {
        "success": success,
        "submitted": False,
        "cv_path": "/tmp/cv.pdf",
        "original_ats_score": 60.0,
        "final_ats_score": 75.0,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_jobs():
    return [_make_job("job-001"), _make_job("job-002")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_mock_run(tmp_path, two_jobs):
    """Both jobs go through prepare(); both get tracker records in a
    non-DISCOVERED state; event log has entries for each job."""
    try:
        from app.services.jobs.orchestrator import JobOrchestrator
    except ImportError:
        pytest.skip("orchestrator module not available")

    mock_run = AsyncMock(return_value=_fake_run_result())
    mock_resume_text = MagicMock(return_value="resume text with python docker kubernetes")
    mock_analysis = MagicMock(
        return_value={
            "matching_skills": ["python", "docker"],
            "missing_skills": [],
        }
    )

    db_path = tmp_path / "test_agent.db"
    with (
        patch(
            "app.services.jobs.orchestrator.run_full_application",
            mock_run,
        ),
        patch(
            "app.services.jobs.orchestrator.JobOrchestrator._extract_resume_text",
            mock_resume_text,
        ),
        patch(
            "app.services.jobs.orchestrator.analyze_resume",
            mock_analysis,
        ),
    ):
        orchestrator = JobOrchestrator(db_path=str(db_path))
        await orchestrator.prepare(two_jobs)

    for job in two_jobs:
        record = orchestrator.tracker.get(job.job_id)
        assert record is not None, f"No tracker record for {job.job_id}"
        assert (
            record.state != ApplicationState.DISCOVERED
        ), f"Job {job.job_id} is still in DISCOVERED state after prepare()"

    # Event log must have entries for both jobs
    log_entries = orchestrator.event_log.get_all()
    job_ids_in_log = {e.job_id for e in log_entries}
    for job in two_jobs:
        assert job.job_id in job_ids_in_log, f"No event log entry for {job.job_id}"


@pytest.mark.asyncio
async def test_pipeline_continues_after_one_job_fails(tmp_path, two_jobs):
    """If one job's run_full_application raises, the other job still
    completes -- the pipeline must not abort on a single failure."""
    try:
        from app.services.jobs.orchestrator import JobOrchestrator
    except ImportError:
        pytest.skip("orchestrator module not available")

    job_a, job_b = two_jobs
    call_count = {"n": 0}

    async def selective_run(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure on first job")
        return _fake_run_result()

    mock_resume_text = MagicMock(return_value="resume text")
    mock_analysis = MagicMock(
        return_value={
            "matching_skills": ["python"],
            "missing_skills": [],
        }
    )

    db_path = tmp_path / "test_agent_fail.db"
    with (
        patch(
            "app.services.jobs.orchestrator.run_full_application",
            selective_run,
        ),
        patch(
            "app.services.jobs.orchestrator.JobOrchestrator._extract_resume_text",
            mock_resume_text,
        ),
        patch(
            "app.services.jobs.orchestrator.analyze_resume",
            mock_analysis,
        ),
    ):
        orchestrator = JobOrchestrator(db_path=str(db_path))
        # Should not raise even though job_a fails
        await orchestrator.prepare(two_jobs)

    # job_b must still have a tracker record (not stuck in DISCOVERED)
    record_b = orchestrator.tracker.get(job_b.job_id)
    assert record_b is not None, "job_b has no tracker record"
    assert (
        record_b.state != ApplicationState.DISCOVERED
    ), "job_b is still DISCOVERED -- pipeline did not continue after job_a failure"
