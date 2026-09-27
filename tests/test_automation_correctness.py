"""Focused regressions for persistent queue and SQLite automation state."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from app.services.jobs.agent_schemas import (
    ApplicationState,
    NormalizedJob,
    fingerprint_job,
    sha256_fingerprint,
)
from app.services.jobs.dedupe import canonicalize_url, compute_dedupe_key, compute_fingerprint
from app.services.tracking import tracker
from app.services.tracking.state_machine import ApplicationStateMachine


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    db_path = tmp_path / "automation-state.db"
    monkeypatch.setattr(tracker, "DB_PATH", db_path)
    ApplicationStateMachine._migrated = False
    ApplicationStateMachine._migrated_path = None
    return db_path


def _fingerprint(character: str) -> str:
    return character * 64


def test_fingerprints_are_full_sha256_and_stable_across_tracking_urls():
    first = NormalizedJob(
        job_id="one",
        title="Platform Engineer",
        company="Example GmbH",
        location="Berlin, Germany",
        application_url="https://EXAMPLE.com/jobs/1?utm_source=test",
    )
    second = NormalizedJob(
        job_id="two",
        title="Platform Engineer",
        company="Example",
        location="Berlin",
        application_url="https://example.com/jobs/1/",
    )

    first_fp = compute_fingerprint(first)
    assert len(first_fp) == 64
    assert first_fp == compute_fingerprint(second)
    assert first_fp == fingerprint_job(
        "Example GmbH",
        "Platform Engineer",
        "Berlin, Germany",
        url=canonicalize_url(first.application_url),
    )
    assert (
        fingerprint_job("Example", "Platform Engineer", "Berlin")
        == hashlib.sha256(b"example|platform engineer|berlin").hexdigest()
    )
    assert compute_dedupe_key(first) == compute_dedupe_key(second)


def test_create_is_idempotent_and_does_not_reset_queued_state(state_db):
    first = ApplicationStateMachine.create(
        company_name="Example",
        job_title="Platform Engineer",
        job_url="https://example.com/jobs/1",
        fingerprint=_fingerprint("a"),
        source="jobspy",
        location="Berlin",
    )
    ApplicationStateMachine.transition(first["id"], ApplicationState.ANALYZING, "test")
    ApplicationStateMachine.transition(first["id"], ApplicationState.MATCHED, "test")

    second = ApplicationStateMachine.create(
        company_name="Example",
        job_title="Platform Engineer",
        job_url="https://example.com/jobs/1",
        fingerprint=_fingerprint("a"),
        source="greenhouse",
        job_description="A richer description",
        location="Berlin",
    )

    assert second["id"] == first["id"]
    assert second["state"] == ApplicationState.MATCHED.value
    assert second["source"] == "greenhouse"
    assert second["job_description"] == "A richer description"

    third = ApplicationStateMachine.create(
        company_name="Example",
        job_title="Platform Engineer",
        job_url="https://example.com/jobs/1",
        fingerprint=_fingerprint("a"),
        source="jobspy",
        job_description="short",
        location="Berlin",
    )
    assert third["job_description"] == "A richer description"
    assert third["source"] == "greenhouse"

    with tracker.ApplicationTrackerService._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 1
        discovered_events = conn.execute(
            "SELECT COUNT(*) FROM application_events WHERE to_state = 'DISCOVERED'"
        ).fetchone()[0]
    assert discovered_events == 1


def test_concurrent_upsert_creates_one_application_and_event(state_db):
    def create_record(_):
        return ApplicationStateMachine.create(
            company_name="Concurrent",
            job_title="Engineer",
            job_url="https://example.com/jobs/concurrent",
            fingerprint=_fingerprint("7"),
            location="Berlin",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(create_record, range(8)))

    assert len({record["id"] for record in records}) == 1
    application_id = records[0]["id"]
    with tracker.ApplicationTrackerService._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM application_events WHERE application_id = ?",
                (application_id,),
            ).fetchone()[0]
            == 1
        )


def test_transition_and_event_commit_atomically(state_db):
    record = ApplicationStateMachine.create(
        company_name="Example",
        job_title="Platform Engineer",
        job_url="https://example.com/jobs/atomic",
        fingerprint=_fingerprint("b"),
        location="Berlin",
    )
    with tracker.ApplicationTrackerService._get_connection() as conn:
        conn.execute("""
            CREATE TRIGGER reject_transition_event
            BEFORE INSERT ON application_events
            BEGIN
                SELECT RAISE(ABORT, 'event insert rejected');
            END
            """)
        conn.commit()

    with pytest.raises(Exception, match="event insert rejected"):
        ApplicationStateMachine.transition(record["id"], ApplicationState.ANALYZING, "test")

    assert ApplicationStateMachine.get(record["id"])["state"] == "DISCOVERED"
    assert [event["to_state"] for event in ApplicationStateMachine.history(record["id"])] == [
        ApplicationState.DISCOVERED.value
    ]


def test_daily_budget_uses_utc_day_and_does_not_double_reserve(state_db):
    first = ApplicationStateMachine.create(
        company_name="First",
        job_title="Engineer",
        job_url="https://example.com/jobs/first",
        fingerprint=_fingerprint("c"),
        location="Berlin",
    )
    second = ApplicationStateMachine.create(
        company_name="Second",
        job_title="Engineer",
        job_url="https://example.com/jobs/second",
        fingerprint=_fingerprint("d"),
        location="Munich",
    )

    assert ApplicationStateMachine.reserve_daily_application(first["id"], 1, "2026-09-23")
    assert not ApplicationStateMachine.reserve_daily_application(first["id"], 1, "2026-09-23")
    assert not ApplicationStateMachine.reserve_daily_application(second["id"], 1, "2026-09-23")
    assert ApplicationStateMachine.reserve_daily_application(second["id"], 1, "2026-09-24")
    assert ApplicationStateMachine.count_daily_applications("2026-09-23") == 1
    assert ApplicationStateMachine.count_daily_applications("2026-09-24") == 1


def test_retry_claim_and_budget_reservation_are_atomic(state_db):
    record = ApplicationStateMachine.create(
        company_name="Retry",
        job_title="Engineer",
        job_url="https://example.com/jobs/retry",
        fingerprint=_fingerprint("9"),
        location="Berlin",
    )
    for state in (
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.PREPARING,
        ApplicationState.SUBMISSION_FAILED,
    ):
        ApplicationStateMachine.transition(record["id"], state)

    assert (
        ApplicationStateMachine.begin_retry(record["id"], daily_limit=0, utc_day="2026-09-23")
        is None
    )
    unchanged = ApplicationStateMachine.get(record["id"])
    assert unchanged["state"] == ApplicationState.SUBMISSION_FAILED.value
    assert unchanged["retry_count"] == 0
    assert ApplicationStateMachine.count_daily_applications("2026-09-23") == 0


def test_daily_budget_reservation_is_atomic_across_threads(state_db):
    applications = [
        ApplicationStateMachine.create(
            company_name=f"Company {index}",
            job_title="Engineer",
            job_url=f"https://example.com/jobs/{index}",
            fingerprint=_fingerprint("e")[:-1] + str(index % 10),
            location=f"City {index}",
        )
        for index in range(8)
    ]
    # Ensure every generated fingerprint is unique despite the compact helper.
    with tracker.ApplicationTrackerService._get_connection() as conn:
        for index, record in enumerate(applications):
            conn.execute(
                "UPDATE applications SET fingerprint = ? WHERE id = ?",
                (sha256_fingerprint("fingerprint", str(index)), record["id"]),
            )
            record["fingerprint"] = sha256_fingerprint("fingerprint", str(index))
        conn.commit()

    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(
            pool.map(
                lambda record: ApplicationStateMachine.reserve_daily_application(
                    record["id"], 3, "2026-09-23"
                ),
                applications,
            )
        )

    assert sum(reservations) == 3
    assert ApplicationStateMachine.count_daily_applications("2026-09-23") == 3


def test_legacy_duplicate_migration_keeps_all_rows(state_db):
    tracker.ApplicationTrackerService.init_db()
    with tracker.ApplicationTrackerService._get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO applications (company_name, job_title, job_url, status)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    "Legacy",
                    "Engineer",
                    "https://example.com/legacy?utm_source=test",
                    "Saved",
                ),
                (
                    "Legacy",
                    "Engineer",
                    "https://example.com/legacy",
                    "Saved",
                ),
            ],
        )
        conn.commit()

    ApplicationStateMachine._ensure_schema()

    expected = compute_fingerprint(
        NormalizedJob(
            job_id="legacy",
            company="Legacy",
            title="Engineer",
            application_url="https://example.com/legacy?utm_source=test",
        )
    )
    with tracker.ApplicationTrackerService._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 2
        keyed_rows = conn.execute(
            "SELECT fingerprint FROM applications WHERE fingerprint IS NOT NULL"
        ).fetchall()
    assert [row["fingerprint"] for row in keyed_rows] == [expected]
    assert ApplicationStateMachine.list_automation_queue() == []


@pytest.mark.asyncio
async def test_auto_runner_recovers_queue_when_discovery_returns_no_jobs(
    state_db, tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    import app.services.jobs.auto_runner as auto_runner

    record = ApplicationStateMachine.create(
        company_name="Queued",
        job_title="Engineer",
        job_url="https://example.com/jobs/queued",
        fingerprint=_fingerprint("8"),
        location="Berlin",
    )
    ApplicationStateMachine.transition(record["id"], ApplicationState.ANALYZING)
    ApplicationStateMachine.transition(record["id"], ApplicationState.MATCHED)

    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4 test")
    pipeline = AsyncMock(return_value={})
    monkeypatch.setattr(auto_runner, "find_jobs_germany", lambda **_: [])
    monkeypatch.setattr(auto_runner, "_extract_resume_text", lambda _: "resume")
    monkeypatch.setattr(auto_runner, "run_application_with_validation", pipeline)

    await auto_runner.run_automated_job_hunting(
        resume_path=str(resume),
        max_applications=1,
    )

    pipeline.assert_awaited_once()


def test_finder_rejects_nan_and_accepts_fallback_url():
    from app.services.jobs.auto_runner import _normalize_job
    from app.services.jobs.finder import _clean_job_result, _safe_str

    assert _safe_str(float("nan")) == ""
    assert _normalize_job({"job_url": float("nan"), "title": float("nan")})["application_url"] == ""
    cleaned = _clean_job_result(
        {
            "job_url": float("nan"),
            "site_url": "https://example.com/jobs/1",
            "title": float("nan"),
            "company": "Example",
            "location": "Berlin",
        },
        "Platform Engineer",
        "Germany",
    )
    assert cleaned == {
        "job_url": "https://example.com/jobs/1",
        "title": "Platform Engineer",
        "company": "Example",
        "location": "Berlin",
        "description": "",
        "site": "unknown",
    }


@pytest.mark.asyncio
async def test_scheduler_uses_utc_slots_once_per_day():
    from app.services.jobs.scheduler import AgentScheduler, ScheduleConfig

    now = datetime(2026, 9, 23, 0, 0, tzinfo=timezone.utc)
    fired = 0

    async def on_discover():
        nonlocal fired
        fired += 1

    async def on_apply():
        raise AssertionError("apply slot was not configured")

    scheduler = AgentScheduler(
        ScheduleConfig(discover_times=["00:00"], apply_times=[]),
        on_discover,
        on_apply,
        clock=lambda: now,
    )
    await scheduler._tick()
    await scheduler._tick()
    assert fired == 1

    now = datetime(2026, 9, 24, 0, 0, tzinfo=timezone.utc)
    await scheduler._tick()
    assert fired == 2
