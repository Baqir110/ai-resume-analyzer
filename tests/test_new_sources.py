# tests/test_new_sources.py
"""Parsing tests for the newly added discovery sources. Each test feeds
a hand-built response matching the vendor's *documented* schema through
an httpx.MockTransport, so the parsing logic is genuinely exercised
without needing real network access to these hosts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from app.services.jobs.arbeitsagentur import search_arbeitsagentur_jobs
from app.services.jobs.company_boards import (
    CompanyBoard,
    list_ashby_jobs,
    list_greenhouse_jobs,
    list_lever_jobs,
    list_personio_jobs,
    list_recruitee_jobs,
    list_smartrecruiters_jobs,
    list_workable_jobs,
)


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_greenhouse_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 555,
                        "title": "Site Reliability Engineer",
                        "location": {"name": "Remote - Europe"},
                        "content": "<p>We need an SRE.</p>",
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/555",
                    }
                ]
            },
        )

    board = CompanyBoard(ats="greenhouse", handle="acme", company_name="Acme")
    async with _client_with(handler) as client:
        jobs = await list_greenhouse_jobs(client, board)
    assert len(jobs) == 1
    assert jobs[0].title == "Site Reliability Engineer"
    assert jobs[0].source == "greenhouse"
    assert jobs[0].application_url.endswith("/jobs/555")


@pytest.mark.asyncio
async def test_lever_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": "lev-1",
                    "text": "Cloud Infrastructure Engineer",
                    "categories": {"location": "Berlin", "commitment": "Full-time"},
                    "descriptionPlain": "Own our cloud infra.",
                    "applyUrl": "https://jobs.lever.co/acme/lev-1/apply",
                    "hostedUrl": "https://jobs.lever.co/acme/lev-1",
                }
            ],
        )

    board = CompanyBoard(ats="lever", handle="acme", company_name="Acme")
    async with _client_with(handler) as client:
        jobs = await list_lever_jobs(client, board)
    assert len(jobs) == 1
    assert jobs[0].location == "Berlin"
    assert jobs[0].employment_type == "Full-time"
    assert jobs[0].source == "lever"


@pytest.mark.asyncio
async def test_ashby_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": "ash-1",
                        "title": "MLOps Engineer",
                        "location": "Remote",
                        "isRemote": True,
                        "descriptionPlain": "MLOps role.",
                        "jobUrl": "https://jobs.ashbyhq.com/acme/ash-1",
                    }
                ]
            },
        )

    board = CompanyBoard(ats="ashby", handle="acme", company_name="Acme")
    async with _client_with(handler) as client:
        jobs = await list_ashby_jobs(client, board)
    assert len(jobs) == 1
    assert jobs[0].remote_type == "remote"
    assert jobs[0].source == "ashby"


@pytest.mark.asyncio
async def test_workable_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "shortcode": "wk-1",
                        "title": "Support Engineer",
                        "location": {"city": "Hamburg"},
                        "employment_type": "full_time",
                        "description": "Support our customers.",
                        "url": "https://apply.workable.com/acme/j/wk-1/",
                    }
                ]
            },
        )

    board = CompanyBoard(ats="workable", handle="acme", company_name="Acme")
    async with _client_with(handler) as client:
        jobs = await list_workable_jobs(client, board)
    assert len(jobs) == 1
    assert jobs[0].location == "Hamburg"
    assert jobs[0].source == "workable"


@pytest.mark.asyncio
async def test_personio_parsing():
    xml_body = """<?xml version="1.0" encoding="UTF-8"?>
    <workzag-position>
        <position>
            <id>77</id>
            <name>IT Administrator</name>
            <office>Bamberg</office>
            <employmentType>full-time</employmentType>
            <jobDescriptions>Manage internal IT.</jobDescriptions>
            <applyUrl>https://acme.jobs.personio.de/job/77</applyUrl>
        </position>
    </workzag-position>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=xml_body, headers={"content-type": "application/xml"})

    board = CompanyBoard(ats="personio", handle="acme", company_name="Acme")
    async with _client_with(handler) as client:
        jobs = await list_personio_jobs(client, board)
    assert len(jobs) == 1
    assert jobs[0].title == "IT Administrator"
    assert jobs[0].source == "personio"


@pytest.mark.asyncio
async def test_personio_malformed_xml_returns_empty_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not xml at all {{{")

    board = CompanyBoard(ats="personio", handle="acme")
    async with _client_with(handler) as client:
        jobs = await list_personio_jobs(client, board)
    assert jobs == []


@pytest.mark.asyncio
async def test_smartrecruiters_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "id": "abc123",
                        "name": "Backend Engineer",
                        "location": {
                            "city": "Berlin",
                            "region": "BE",
                            "country": "de",
                            "remote": False,
                        },
                        "applyUrl": "https://careers.smartrecruiters.com/Acme/backend-engineer",
                        "ref": "https://api.smartrecruiters.com/v1/companies/Acme/postings/abc123",
                    }
                ]
            },
        )

    board = CompanyBoard(ats="smartrecruiters", handle="Acme", company_name="Acme Corp")
    async with _client_with(handler) as client:
        jobs = await list_smartrecruiters_jobs(client, board)

    assert len(jobs) == 1
    assert jobs[0].title == "Backend Engineer"
    assert jobs[0].company == "Acme Corp"
    assert "Berlin" in jobs[0].location
    assert jobs[0].source == "smartrecruiters"


@pytest.mark.asyncio
async def test_recruitee_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "offers": [
                    {
                        "id": 42,
                        "title": "Platform Engineer",
                        "location": "Munich",
                        "remote": True,
                        "employment_type_code": "full_time",
                        "description": "Build our platform.",
                        "careers_url": "https://acme.recruitee.com/o/platform-engineer",
                    }
                ]
            },
        )

    board = CompanyBoard(ats="recruitee", handle="acme")
    async with _client_with(handler) as client:
        jobs = await list_recruitee_jobs(client, board)

    assert len(jobs) == 1
    assert jobs[0].title == "Platform Engineer"
    assert jobs[0].remote_type == "remote"
    assert jobs[0].source == "recruitee"


@pytest.mark.asyncio
async def test_smartrecruiters_handles_malformed_entry_gracefully():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"id": None, "location": None}]})

    board = CompanyBoard(ats="smartrecruiters", handle="Acme")
    async with _client_with(handler) as client:
        jobs = await list_smartrecruiters_jobs(client, board)
    # Should not raise; a malformed entry with no name still parses (falls
    # back to "Unknown"), since only truly broken shapes are skipped.
    assert len(jobs) == 1
    assert jobs[0].title == "Unknown"


@pytest.mark.asyncio
async def test_smartrecruiters_source_failure_returns_empty_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    board = CompanyBoard(ats="smartrecruiters", handle="Acme")
    async with _client_with(handler) as client:
        jobs = await list_smartrecruiters_jobs(client, board)
    assert jobs == []  # Section 19/38: source failure never raises


@pytest.mark.asyncio
async def test_arbeitsagentur_parsing(monkeypatch):
    import app.services.jobs.arbeitsagentur as aa

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == aa.PUBLIC_API_KEY
        return httpx.Response(
            200,
            json={
                "stellenangebote": [
                    {
                        "refnr": "10000-1234567890-S",
                        "titel": "DevOps Ingenieur (m/w/d)",
                        "arbeitgeber": "Muster GmbH",
                        "arbeitsort": {"ort": "Bamberg", "region": "Bayern", "land": "Deutschland"},
                        "stellenbeschreibung": "Kubernetes und Docker Erfahrung gewuenscht.",
                        "aktuelleVeroeffentlichungsdatum": "2026-09-01",
                    }
                ]
            },
        )

    async def fake_client_factory(*args, **kwargs):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            **{k: v for k, v in kwargs.items() if k not in ("timeout", "headers")},
        )

    # Patch httpx.AsyncClient used inside the module to route through our mock transport.
    original_client = httpx.AsyncClient

    class PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(aa.httpx, "AsyncClient", PatchedClient)

    jobs = await search_arbeitsagentur_jobs(query="DevOps Engineer", location="Bamberg")
    assert len(jobs) == 1
    assert jobs[0].company == "Muster GmbH"
    assert jobs[0].country == "Germany"
    assert jobs[0].source == "arbeitsagentur"
    assert "Bamberg" in jobs[0].location


# ---------------------------------------------------------------------------
# Task 2: WorkdaySource tests
# ---------------------------------------------------------------------------

from app.services.jobs.company_boards import list_workday_jobs


@pytest.mark.asyncio
async def test_workday_parses_jobs_from_mock_response():
    """Verify WorkdaySource correctly parses a Workday CXS API response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert "wday/cxs" in str(request.url)
        return httpx.Response(
            200,
            json={
                "jobPostings": [
                    {
                        "title": "DevOps Engineer (m/w/d)",
                        "locationsText": "Munich, Bavaria",
                        "bulletFields": ["5+ years experience", "Kubernetes required"],
                        "externalPath": "/en-US/Siemens/job/Munich/DevOps-Engineer_JR12345",
                    },
                    {
                        "title": "MLOps Engineer",
                        "locationsText": "Berlin",
                        "bulletFields": ["Python", "MLflow"],
                        "externalPath": "/en-US/Siemens/job/Berlin/MLOps-Engineer_JR67890",
                    },
                ]
            },
        )

    board = {
        "company": "Siemens",
        "tenant": "siemens",
        "board": "Siemens",
        "instance": "wd3",
    }
    async with _client_with(handler) as client:
        jobs = await list_workday_jobs(client, board)

    assert len(jobs) == 2
    assert jobs[0].title == "DevOps Engineer (m/w/d)"
    assert jobs[0].location == "Munich, Bavaria"
    assert "Kubernetes required" in jobs[0].description
    assert jobs[0].source == "workday"
    assert jobs[0].company == "Siemens"
    assert "siemens" in jobs[0].application_url
    assert jobs[1].title == "MLOps Engineer"


@pytest.mark.asyncio
async def test_workday_handles_empty_response_gracefully():
    """An empty jobPostings array returns [] without raising."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jobPostings": []})

    board = {"company": "Acme", "tenant": "acme", "board": "Acme", "instance": "wd1"}
    async with _client_with(handler) as client:
        jobs = await list_workday_jobs(client, board)

    assert jobs == []


@pytest.mark.asyncio
async def test_workday_handles_server_error_gracefully():
    """A 500 response returns [] without raising."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    board = {"company": "Acme", "tenant": "acme", "board": "Acme", "instance": "wd1"}
    async with _client_with(handler) as client:
        jobs = await list_workday_jobs(client, board)

    assert jobs == []
