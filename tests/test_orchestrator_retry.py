# tests/test_orchestrator_retry.py
"""Tests the retry_failed() flow end-to-end against a scratch SQLite DB,
with run_full_application mocked out (no jobspy/browser-use needed).

Also guards the RETRYING -> PREPARING state-machine transition that a
previous version of this code got wrong.

Section 15 additionally verifies that the application package validation
gate is enforced before an application can proceed.
"""

import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

os.environ.setdefault("DATABASE_PATH", "/tmp/test_orchestrator_retry.db")

# browser-use (and its Playwright/Chromium chain) isn't installed in every
# dev environment. full_pipeline.py imports it at module level for the
# browser-automation step, which we mock out entirely in these tests
# anyway -- so stub the package rather than requiring a multi-GB install
# just to exercise the orchestration/state-machine logic around it.
if "browser_use" not in sys.modules:
    import types

    stub = types.ModuleType("browser_use")
    for name in (
        "Agent",
        "Browser",
        "ChatAnthropic",
        "ChatGoogle",
        "ChatOpenAI",
    ):
        setattr(stub, name, type(name, (), {}))
    sys.modules["browser_use"] = stub

from app.services.jobs.agent_schemas import ApplicationState


def _write_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)


from app.services.jobs.orchestrator import JobAgentOrchestrator
from app.services.tracking.state_machine import ApplicationStateMachine


class FakeConfig:
    MINIMUM_MATCH_SCORE = 60.0
    HIGH_PRIORITY_THRESHOLD = 85.0
    GOOD_MATCH_THRESHOLD = 70.0
    MAX_MISSING_REQUIRED_SKILLS = 2
    MAX_CONCURRENT_BROWSER_WORKERS = 1
    MAX_DAILY_APPLICATIONS = 10
    AUTOMATIC_APPLY = True
    AUTOMATIC_SUBMIT = False
    SOURCES_ENABLED = ["jobspy", "ats_direct", "arbeitsagentur"]
    TARGET_ROLES = ["DevOps Engineer"]
    TARGET_LOCATIONS = ["Bamberg"]
    COMPANY_BOARDS = []


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "orchestrator_retry.db"

    from app.services.tracking import tracker

    monkeypatch.setattr(tracker, "DB_PATH", db_path)

    ApplicationStateMachine._migrated = False

    import app.services.tracking.live_status as live_status

    live_status._TABLE_READY = False

    yield

    # Clean up file descriptors for Windows SQLite test isolation
    try:
        conn = sqlite3.connect(db_path)
        conn.close()
    except Exception:
        pass


def _resume_stub(tmp_path) -> str:
    p = tmp_path / "resume.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    # The resume contents are never actually parsed in these tests because
    # _extract_resume_text is mocked where the retry flow needs it.
    return str(p)


@pytest.mark.asyncio
async def test_retry_failed_reprocesses_and_can_succeed(tmp_path):
    """A failed application can be retried and reach READY_TO_SUBMIT.

    This test focuses on retry orchestration/state transitions. The actual
    package validation behavior is tested separately below, so the validator
    is mocked as successful here.
    """
    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    record = ApplicationStateMachine.create(
        company_name="ACME",
        job_title="DevOps Engineer",
        job_url="https://acme.example/job/1",
        fingerprint="fp-retry-1",
        source="greenhouse",
        job_description="Kubernetes, Docker, Python",
    )

    for state in [
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ]:
        ApplicationStateMachine.transition(
            record["id"],
            state,
            "setup",
        )

    ApplicationStateMachine.transition(
        record["id"],
        ApplicationState.SUBMISSION_FAILED,
        "browser crashed",
    )

    from app.services.jobs.package_validator import ValidationResult

    with (
        patch(
            "app.services.jobs.orchestrator._extract_resume_text",
            return_value="resume text",
        ),
        patch(
            "app.services.jobs.full_pipeline.run_full_application",
            new_callable=AsyncMock,
        ) as mock_pipeline,
        patch(
            "app.services.jobs.package_validator.validate_package",
            return_value=ValidationResult(valid=True),
        ),
    ):
        mock_pipeline.return_value = {
            "success": True,
            "submitted": False,
            "cv_path": str(tmp_path / "cv.pdf"),
            "original_ats_score": 60.0,
            "final_ats_score": 70.0,
        }

        results = await orchestrator.retry_failed()

    assert len(results) == 1

    updated = ApplicationStateMachine.find_by_fingerprint("fp-retry-1")

    assert updated["state"] == ApplicationState.READY_TO_SUBMIT.value
    assert updated["retry_count"] == 1

    history_states = [event["to_state"] for event in ApplicationStateMachine.history(record["id"])]

    # The retry must legally pass through:
    #
    # RETRYING -> PREPARING
    #
    # rather than skipping directly to APPLYING/SUBMITTED.
    retry_idx = history_states.index("RETRYING")

    assert history_states[retry_idx + 1] == "PREPARING"


@pytest.mark.asyncio
async def test_analyze_only_scores_without_touching_pipeline(tmp_path):
    """--analyze must score and transition state, but never call the
    CV/browser pipeline."""
    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    from app.services.jobs.agent_schemas import NormalizedJob
    from app.services.jobs.discovery import DiscoveryConfig, DiscoveryEngine, JobSource

    class OneJobSource(JobSource):
        name = "fake"

        async def discover(self, config):
            return [
                NormalizedJob(
                    job_id="x1",
                    title="DevOps Engineer",
                    company="ACME",
                    application_url="https://acme.example/job/x1",
                    description="Kubernetes Docker Python Terraform AWS",
                )
            ]

    with (
        patch(
            "app.services.jobs.orchestrator._extract_resume_text",
            return_value="resume text",
        ),
        patch(
            "app.services.jobs.orchestrator.analyze_resume",
            return_value={
                "matching_skills": ["kubernetes", "docker", "python"],
                "missing_skills": [],
            },
        ),
        patch.object(
            __import__(
                "app.services.jobs.orchestrator",
                fromlist=["DiscoveryEngine"],
            ),
            "DiscoveryEngine",
            lambda: DiscoveryEngine(sources=[OneJobSource()]),
        ),
        patch(
            "app.services.jobs.full_pipeline.run_full_application",
            new_callable=AsyncMock,
        ) as mock_pipeline,
    ):
        rows = await orchestrator.analyze_only()

    mock_pipeline.assert_not_called()

    assert len(rows) == 1
    assert rows[0]["decision"] == "APPLY"
    assert rows[0]["score"] > 0

    record = ApplicationStateMachine.find_by_fingerprint(
        __import__(
            "app.services.jobs.dedupe",
            fromlist=["compute_fingerprint"],
        ).compute_fingerprint(
            NormalizedJob(
                job_id="x1",
                title="DevOps Engineer",
                company="ACME",
                application_url="https://acme.example/job/x1",
            )
        )
    )

    assert record is not None
    assert record["state"] == "MATCHED"


@pytest.mark.asyncio
async def test_unverified_submission_flagged_not_trusted(tmp_path):
    """End-to-end: pipeline claims submitted=True with no supporting
    evidence -- orchestrator must record SUBMITTED (the click happened)
    then immediately flag VERIFICATION_FAILED, not silently trust it."""
    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    record = ApplicationStateMachine.create(
        company_name="ACME",
        job_title="DevOps Engineer",
        job_url="https://acme.example/job/3",
        fingerprint="fp-verify-1",
        source="greenhouse",
        job_description="Kubernetes",
    )

    for state in [
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ]:
        ApplicationStateMachine.transition(
            record["id"],
            state,
            "setup",
        )

    cv_file = tmp_path / "unverified_cv.pdf"
    _write_pdf(cv_file)
    orchestrator._record_pipeline_result(
        record["id"],
        {
            "success": True,
            "submitted": True,
            "cv_path": str(cv_file),
            "browser_application": {
                "final_result": "Done.",
                "final_url": "https://acme.example/apply",
            },
        },
    )

    updated = ApplicationStateMachine.find_by_fingerprint("fp-verify-1")

    assert updated["state"] == ApplicationState.VERIFICATION_FAILED.value

    history_states = [event["to_state"] for event in ApplicationStateMachine.history(record["id"])]

    # The click/submission was still recorded as having happened.
    assert "SUBMITTED" in history_states


@pytest.mark.asyncio
async def test_verified_submission_reaches_verified_state(tmp_path):
    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    record = ApplicationStateMachine.create(
        company_name="ACME",
        job_title="DevOps Engineer",
        job_url="https://acme.example/job/4",
        fingerprint="fp-verify-2",
        source="greenhouse",
        job_description="Kubernetes",
    )

    for state in [
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ]:
        ApplicationStateMachine.transition(
            record["id"],
            state,
            "setup",
        )

    cv_file = tmp_path / "verified_cv.pdf"
    _write_pdf(cv_file)
    orchestrator._record_pipeline_result(
        record["id"],
        {
            "success": True,
            "submitted": True,
            "cv_path": str(cv_file),
            "browser_application": {
                "final_result": "Thank you for applying!",
                "submission_verified": True,
                "submission_status": "submitted",
            },
        },
    )

    updated = ApplicationStateMachine.find_by_fingerprint("fp-verify-2")

    assert updated["state"] == ApplicationState.VERIFIED.value
    history_states = [event["to_state"] for event in ApplicationStateMachine.history(record["id"])]
    assert "SUBMITTED" in history_states


@pytest.mark.asyncio
async def test_retry_failed_respects_max_attempts(tmp_path):
    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    record = ApplicationStateMachine.create(
        company_name="Beta",
        job_title="MLOps Engineer",
        job_url="https://beta.example/job/2",
        fingerprint="fp-retry-2",
        source="lever",
        job_description="MLOps, AWS",
    )

    for state in [
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ]:
        ApplicationStateMachine.transition(
            record["id"],
            state,
            "setup",
        )

    ApplicationStateMachine.transition(
        record["id"],
        ApplicationState.SUBMISSION_FAILED,
        "captcha then failure",
    )

    # Simulate it already having exhausted retries.
    for _ in range(orchestrator.MAX_RETRY_ATTEMPTS):
        ApplicationStateMachine.increment_retry_count(record["id"])

    with (
        patch(
            "app.services.jobs.orchestrator._extract_resume_text",
            return_value="resume text",
        ),
        patch(
            "app.services.jobs.full_pipeline.run_full_application",
            new_callable=AsyncMock,
        ) as mock_pipeline,
    ):
        results = await orchestrator.retry_failed()

    assert results == []

    mock_pipeline.assert_not_called()

    updated = ApplicationStateMachine.find_by_fingerprint("fp-retry-2")

    assert updated["state"] == ApplicationState.SUBMISSION_FAILED.value


@pytest.mark.asyncio
async def test_discovery_preserves_an_existing_queued_job(tmp_path):
    """Rediscovering queued work must upsert it, not remove it from the queue."""
    from app.services.jobs.agent_schemas import NormalizedJob
    from app.services.jobs.discovery import DiscoveryEngine, JobSource

    class OneJobSource(JobSource):
        name = "fake_queue"

        async def discover(self, config):
            return [
                NormalizedJob(
                    job_id="queue-1",
                    title="Platform Engineer",
                    company="Example",
                    location="Berlin",
                    application_url="https://example.com/jobs/queue-1",
                    description="Python, Kubernetes, Terraform",
                )
            ]

    orchestrator = JobAgentOrchestrator(FakeConfig(), resume_path=_resume_stub(tmp_path))
    with patch.object(
        __import__(
            "app.services.jobs.orchestrator",
            fromlist=["DiscoveryEngine"],
        ),
        "DiscoveryEngine",
        lambda: DiscoveryEngine(sources=[OneJobSource()]),
    ):
        first = await orchestrator.discover()
        first_record = ApplicationStateMachine.find_by_fingerprint(first[0].fingerprint)
        ApplicationStateMachine.transition(first_record["id"], ApplicationState.ANALYZING, "test")
        ApplicationStateMachine.transition(first_record["id"], ApplicationState.MATCHED, "test")

        second = await orchestrator.discover()

    assert len(first) == len(second) == 1
    assert second[0].fingerprint == first[0].fingerprint
    updated = ApplicationStateMachine.find_by_fingerprint(first[0].fingerprint)
    assert updated["state"] == ApplicationState.MATCHED.value
    from app.services.tracking.tracker import ApplicationTrackerService

    with ApplicationTrackerService._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_invalid_package_is_rejected_before_browser_runner(tmp_path):
    """The browser worker is never reached when the generated package is invalid."""
    import app.services.jobs.orchestrator as orchestrator_module
    from app.services.jobs import full_pipeline
    from app.services.jobs.package_validator import ValidationResult

    orchestrator_module._install_browser_package_gate()
    browser_runner = AsyncMock(return_value={})

    async def fake_pipeline(**kwargs):
        return await full_pipeline._run_browser_application(
            application_url="https://example.com/jobs/1",
            resume_path=str(tmp_path / "missing.pdf"),
            cover_letter_path=None,
            why_this_company="test",
            company_name="Example",
            role_title="Platform Engineer",
            agent_llm="test",
            auto_submit=False,
        )

    with (
        patch.object(
            full_pipeline,
            "run_full_application",
            side_effect=fake_pipeline,
        ),
        patch(
            "app.services.jobs.package_validator.validate_package",
            return_value=ValidationResult(valid=False, errors=["invalid"]),
        ),
        patch.object(orchestrator_module, "_real_browser_runner", browser_runner),
    ):
        with pytest.raises(orchestrator_module.PackageValidationError):
            await orchestrator_module.run_application_with_validation(
                job={
                    "job_id": "gate-1",
                    "title": "Platform Engineer",
                    "company": "Example",
                    "job_url": "https://example.com/jobs/1",
                    "description": "Python",
                },
                resume_text="resume",
                profile={
                    "personal": {
                        "first_name": "Test",
                        "last_name": "User",
                        "email": "test@example.invalid",
                    }
                },
            )

    browser_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_package_validator_exception_fails_closed(tmp_path):
    orchestrator = JobAgentOrchestrator(FakeConfig(), resume_path=_resume_stub(tmp_path))
    record = ApplicationStateMachine.create(
        company_name="Example",
        job_title="Platform Engineer",
        job_url="https://example.com/jobs/validator-error",
        fingerprint="fp-validator-error",
        source="greenhouse",
        job_description="Python",
    )
    for state in (
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ):
        ApplicationStateMachine.transition(record["id"], state, "setup")

    cv_file = tmp_path / "validator_error.pdf"
    _write_pdf(cv_file)
    with patch(
        "app.services.jobs.package_validator.validate_package",
        side_effect=RuntimeError("validator unavailable"),
    ):
        orchestrator._record_pipeline_result(
            record["id"],
            {
                "success": True,
                "submitted": False,
                "cv_path": str(cv_file),
                "browser_application": {},
            },
        )

    updated = ApplicationStateMachine.get(record["id"])
    assert updated["state"] == ApplicationState.SUBMISSION_FAILED.value
    states = [event["to_state"] for event in ApplicationStateMachine.history(record["id"])]
    assert "READY_FOR_APPLICATION" not in states
    assert "APPLYING" not in states


# ---------------------------------------------------------------------------
# Section 15 — package_validator integration with _record_pipeline_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_package_fails_before_ready_for_application(tmp_path):
    """If validate_package() returns invalid, _record_pipeline_result must
    immediately transition to SUBMISSION_FAILED and NOT proceed to
    READY_FOR_APPLICATION or any submission state."""
    from app.services.jobs.agent_schemas import NormalizedJob
    from app.services.jobs.package_validator import ValidationResult

    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    record = ApplicationStateMachine.create(
        company_name="Corrupt",
        job_title="Data Engineer",
        job_url="https://corrupt.example/job/1",
        fingerprint="fp-validate-fail-1",
        source="greenhouse",
        job_description="Spark, Kafka",
    )

    for state in [
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ]:
        ApplicationStateMachine.transition(
            record["id"],
            state,
            "setup",
        )

    job = NormalizedJob(
        job_id="vf-1",
        title="Data Engineer",
        company="Corrupt",
        application_url="https://corrupt.example/job/1",
        description="Spark, Kafka",
    )

    # Produce a non-empty CV file so the check gets past the
    # "cv_generated" gate.
    cv_file = tmp_path / "cv.pdf"
    _write_pdf(cv_file)

    # Mock validate_package to return invalid. This simulates a
    # corrupted CV or another package validation failure.
    invalid_result = ValidationResult(
        valid=False,
        errors=["CV file is empty: /tmp/cv.pdf"],
    )

    with patch(
        "app.services.jobs.package_validator.validate_package",
        return_value=invalid_result,
    ):
        orchestrator._record_pipeline_result(
            record["id"],
            {
                "success": True,
                "submitted": False,
                "cv_path": str(cv_file),
                "browser_application": {},
            },
            job=job,
        )

    updated = ApplicationStateMachine.find_by_fingerprint("fp-validate-fail-1")

    assert (
        updated["state"] == ApplicationState.SUBMISSION_FAILED.value
    ), f"Expected SUBMISSION_FAILED, got {updated['state']}"

    history_states = [event["to_state"] for event in ApplicationStateMachine.history(record["id"])]

    assert "READY_FOR_APPLICATION" not in history_states, (
        "Must not reach READY_FOR_APPLICATION when " "package validation fails"
    )


@pytest.mark.asyncio
async def test_valid_package_passes_validation_gate(tmp_path):
    """When validate_package() returns valid, execution must proceed past
    the validation gate and reach READY_TO_SUBMIT."""
    from app.services.jobs.agent_schemas import NormalizedJob
    from app.services.jobs.package_validator import ValidationResult

    orchestrator = JobAgentOrchestrator(
        FakeConfig(),
        resume_path=_resume_stub(tmp_path),
    )

    record = ApplicationStateMachine.create(
        company_name="GoodCorp",
        job_title="Backend Engineer",
        job_url="https://good.example/job/1",
        fingerprint="fp-validate-pass-1",
        source="lever",
        job_description="Python, FastAPI",
    )

    for state in [
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
    ]:
        ApplicationStateMachine.transition(
            record["id"],
            state,
            "setup",
        )

    job = NormalizedJob(
        job_id="vp-1",
        title="Backend Engineer",
        company="GoodCorp",
        application_url="https://good.example/job/1",
        description="Python, FastAPI",
    )

    cv_file = tmp_path / "cv.pdf"
    _write_pdf(cv_file)

    valid_result = ValidationResult(valid=True)

    with patch(
        "app.services.jobs.package_validator.validate_package",
        return_value=valid_result,
    ):
        orchestrator._record_pipeline_result(
            record["id"],
            {
                "success": True,
                "submitted": False,
                "cv_path": str(cv_file),
                "browser_application": {},
            },
            job=job,
        )

    history_states = [event["to_state"] for event in ApplicationStateMachine.history(record["id"])]

    assert "READY_FOR_APPLICATION" in history_states, (
        "Must reach READY_FOR_APPLICATION when " "package validation passes"
    )

    updated = ApplicationStateMachine.find_by_fingerprint("fp-validate-pass-1")

    assert (
        updated["state"] == ApplicationState.READY_TO_SUBMIT.value
    ), f"Expected READY_TO_SUBMIT, got {updated['state']}"
