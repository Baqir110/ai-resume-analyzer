# tests/test_discovery.py
"""Integration test for DiscoveryEngine: multiple sources feeding into
one deduplicated list, and graceful continuation when a source fails.
Sources are mocked -- no real network calls to job boards.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.services.jobs.agent_schemas import NormalizedJob
from app.services.jobs.discovery import DiscoveryConfig, DiscoveryEngine, JobSource


class FakeSourceA(JobSource):
    name = "fake_a"

    async def discover(self, config):
        return [
            NormalizedJob(
                job_id="a1",
                title="DevOps Engineer",
                company="ACME GmbH",
                location="Bamberg, Germany",
                application_url="https://a.example/job/1",
                description="short desc",
                source="fake_a",
            ),
            NormalizedJob(
                job_id="a2",
                title="Data Scientist",
                company="Beta AG",
                location="Berlin, Germany",
                application_url="https://a.example/job/2",
                description="another job",
                source="fake_a",
            ),
        ]


class FakeSourceB(JobSource):
    """Returns the *same* job as source A (different URL/source) with a
    richer description, plus one genuinely new job."""

    name = "fake_b"

    async def discover(self, config):
        return [
            NormalizedJob(
                job_id="a1",
                title="DevOps Engineer",
                company="ACME GmbH",
                location="Bamberg, Germany",
                application_url="https://boards.greenhouse.io/acme/jobs/1",
                description="short desc",
                source="greenhouse",
            ),
            NormalizedJob(
                job_id="b2",
                title="MLOps Engineer",
                company="Gamma",
                location="Munich, Germany",
                application_url="https://a.example/job/3",
                description="third job",
                source="fake_b",
            ),
        ]


class FailingSource(JobSource):
    name = "failing"

    async def discover(self, config):
        raise RuntimeError("simulated source outage")


@pytest.mark.asyncio
async def test_discovery_engine_dedupes_across_sources():
    engine = DiscoveryEngine(sources=[FakeSourceA(), FakeSourceB()])
    jobs = await engine.discover_all(DiscoveryConfig())

    titles = sorted(j.title for j in jobs)
    assert titles == ["DevOps Engineer", "Data Scientist", "MLOps Engineer"].__class__(
        sorted(["DevOps Engineer", "Data Scientist", "MLOps Engineer"])
    )
    assert len(jobs) == 3  # 4 raw, 1 true duplicate collapsed

    devops = next(j for j in jobs if j.title == "DevOps Engineer")
    assert devops.source == "greenhouse"  # richer record won


@pytest.mark.asyncio
async def test_discovery_engine_continues_when_one_source_fails():
    engine = DiscoveryEngine(sources=[FakeSourceA(), FailingSource()])
    jobs = await engine.discover_all(DiscoveryConfig())
    assert len(jobs) == 2  # FakeSourceA's jobs still came through


@pytest.mark.asyncio
async def test_discovery_prioritizes_jobs_from_better_performing_sources(tmp_path, monkeypatch):
    """Section 26/27: a source with a proven interview track record
    should have its jobs sorted ahead of an equally-sized but
    zero-response source."""
    from app.services.jobs.agent_schemas import ApplicationState
    from app.services.tracking import tracker
    from app.services.tracking.state_machine import ApplicationStateMachine

    db_path = tmp_path / "priority.db"
    monkeypatch.setattr(tracker, "DB_PATH", db_path)
    ApplicationStateMachine._migrated = False

    # Build up a track record: "greenhouse" gets interviews, "jobspy_indeed" doesn't.
    for i in range(3):
        rec = ApplicationStateMachine.create(
            company_name=f"GH{i}",
            job_title="DevOps Engineer",
            job_url=f"https://gh.example/{i}",
            fingerprint=f"gh-fp-{i}",
            source="greenhouse",
        )
        for state in [
            ApplicationState.ANALYZING,
            ApplicationState.MATCHED,
            ApplicationState.PREPARING,
            ApplicationState.CV_GENERATED,
            ApplicationState.READY_FOR_APPLICATION,
            ApplicationState.APPLYING,
            ApplicationState.SUBMITTED,
            ApplicationState.VERIFIED,
        ]:
            ApplicationStateMachine.transition(rec["id"], state, "setup")
        if i < 2:
            ApplicationStateMachine.transition(
                rec["id"], ApplicationState.INTERVIEW, "got interview"
            )

    for i in range(3):
        rec = ApplicationStateMachine.create(
            company_name=f"IN{i}",
            job_title="DevOps Engineer",
            job_url=f"https://indeed.example/{i}",
            fingerprint=f"indeed-fp-{i}",
            source="jobspy_indeed",
        )
        for state in [
            ApplicationState.ANALYZING,
            ApplicationState.MATCHED,
            ApplicationState.PREPARING,
            ApplicationState.CV_GENERATED,
            ApplicationState.READY_FOR_APPLICATION,
            ApplicationState.APPLYING,
            ApplicationState.SUBMITTED,
            ApplicationState.VERIFIED,
        ]:
            ApplicationStateMachine.transition(rec["id"], state, "setup")
        # no interviews for this source

    class IndeedFirstSource(JobSource):
        name = "indeed_first"

        async def discover(self, config):
            return [
                NormalizedJob(
                    job_id="new-indeed",
                    title="DevOps Engineer",
                    company="NewCo",
                    application_url="https://indeed.example/new",
                    source="jobspy_indeed",
                ),
                NormalizedJob(
                    job_id="new-gh",
                    title="MLOps Engineer",
                    company="OtherCo",
                    application_url="https://gh.example/new",
                    source="greenhouse",
                ),
            ]

    engine = DiscoveryEngine(sources=[IndeedFirstSource()])
    jobs = await engine.discover_all(DiscoveryConfig())

    assert [j.source for j in jobs] == ["greenhouse", "jobspy_indeed"]
