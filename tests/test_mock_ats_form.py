# tests/test_mock_ats_form.py
"""Section 35: mock application form for browser/backend-submission
testing, so field-mapping logic is validated against realistic HTML
instead of only being reasoned about. This is the part of
backend_submitter.py most likely to be wrong in practice (real forms
vary board to board), so it's worth pinning down with a concrete test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from app.services.jobs.agent_schemas import NormalizedJob
from app.services.jobs.backend_submitter import ApplicationPackage, GreenhouseFormSubmitter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_greenhouse_form.html"


def _make_job() -> NormalizedJob:
    return NormalizedJob(
        job_id="gh-1",
        title="DevOps Engineer",
        company="ACME",
        application_url="https://boards.greenhouse.io/acme/jobs/1",
        source="greenhouse",
    )


def _make_package(answers: dict | None = None) -> ApplicationPackage:
    return ApplicationPackage(
        job=_make_job(),
        cv_path="/tmp/cv.pdf",
        cover_letter_path=None,
        answers=answers or {},
        profile={
            "personal": {
                "first_name": "Test",
                "last_name": "Candidate",
                "email": "test@example.com",
                "phone": "+49 152 00000000",
            }
        },
    )


@pytest.mark.asyncio
async def test_supports_matches_greenhouse_urls():
    submitter = GreenhouseFormSubmitter()
    assert submitter.supports(_make_job()) is True
    other = _make_job()
    other.application_url = "https://de.indeed.com/viewjob?jk=1"
    other.source = ""
    assert submitter.supports(other) is False


@pytest.mark.asyncio
async def test_defers_to_browser_when_required_custom_questions_unmapped():
    """The form has two required custom questions with no matching
    profile/answers data -- the submitter must recognize it can't fill
    them and defer, not guess or submit incomplete data."""
    html = FIXTURE_PATH.read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    submitter = GreenhouseFormSubmitter()
    package = _make_package(answers={})  # no answers for the custom questions

    import app.services.jobs.backend_submitter as bs_module

    original_client = httpx.AsyncClient

    class PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    orig = bs_module.httpx.AsyncClient
    bs_module.httpx.AsyncClient = PatchedClient
    try:
        result = await submitter.submit(package)
    finally:
        bs_module.httpx.AsyncClient = orig

    assert result.handled is False
    assert "unmapped" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_field_mapping_identifies_all_required_fields_correctly():
    """Direct unit test of _map_fields against the fixture's required
    field list, independent of the network layer."""
    from bs4 import BeautifulSoup

    html = FIXTURE_PATH.read_text()
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", {"id": "application-form"})
    required_fields = [
        inp.get("name")
        for inp in form.find_all(["input", "select", "textarea"])
        if inp.has_attr("required") and inp.get("name")
    ]

    # Sanity check the fixture itself has the shape we expect before
    # trusting conclusions drawn from it.
    assert "job_application[first_name]" in required_fields
    assert "job_application[resume]" in required_fields
    assert (
        len(required_fields) == 7
    )  # first, last, email, phone, resume + 2 custom (optional linkedin excluded)

    submitter = GreenhouseFormSubmitter()
    package = _make_package(answers={})
    mapped, unmapped = submitter._map_fields(required_fields, package)

    assert "job_application[first_name]" in mapped
    assert "job_application[email]" in mapped
    assert "job_application[resume]" in mapped
    # The two custom select/textarea questions have no mapping source.
    assert len(unmapped) == 2


@pytest.mark.asyncio
async def test_field_mapping_succeeds_when_custom_answers_supplied():
    """If the caller *does* supply answers for the custom questions
    (e.g. from an LLM-driven answer engine), the submitter should
    recognize the form as fully mappable."""
    from bs4 import BeautifulSoup

    html = FIXTURE_PATH.read_text()
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", {"id": "application-form"})
    required_fields = [
        inp.get("name")
        for inp in form.find_all(["input", "select", "textarea"])
        if inp.has_attr("required") and inp.get("name")
    ]

    submitter = GreenhouseFormSubmitter()
    package = _make_package(
        answers={
            "job_application[answers_attributes][0][text_value]": "yes",
            "job_application[answers_attributes][1][text_value]": "Immediately",
        }
    )
    mapped, unmapped = submitter._map_fields(required_fields, package)
    assert unmapped == []
    assert len(mapped) == len(required_fields)


# ---------------------------------------------------------------------------
# Task 3: LeverFormSubmitter and AshbyFormSubmitter tests
# ---------------------------------------------------------------------------

from app.services.jobs.backend_submitter import AshbyFormSubmitter, LeverFormSubmitter


def _make_lever_job(url: str = "https://jobs.lever.co/acme/abc-123") -> NormalizedJob:
    return NormalizedJob(
        job_id="lever-abc-123",
        title="Platform Engineer",
        company="Acme",
        application_url=url,
        source="lever",
    )


def _make_ashby_job(url: str = "https://jobs.ashbyhq.com/acme/xyz-789") -> NormalizedJob:
    return NormalizedJob(
        job_id="ashby-xyz-789",
        title="Backend Engineer",
        company="Acme",
        application_url=url,
        source="ashby",
    )


# -- supports() tests --------------------------------------------------------


def test_lever_submitter_supports_lever_url():
    submitter = LeverFormSubmitter()
    assert submitter.supports(_make_lever_job()) is True


def test_lever_submitter_not_for_greenhouse():
    submitter = LeverFormSubmitter()
    gh_job = NormalizedJob(
        job_id="gh-1",
        title="DevOps Engineer",
        company="Acme",
        application_url="https://boards.greenhouse.io/acme/jobs/1",
        source="greenhouse",
    )
    assert submitter.supports(gh_job) is False


def test_lever_submitter_supports_lever_apply_url():
    submitter = LeverFormSubmitter()
    job = _make_lever_job(url="https://jobs.lever.co/acme/abc-123/apply")
    assert submitter.supports(job) is True


def test_ashby_submitter_supports_ashby_url():
    submitter = AshbyFormSubmitter()
    assert submitter.supports(_make_ashby_job()) is True


def test_ashby_submitter_supports_app_ashby_url():
    submitter = AshbyFormSubmitter()
    job = _make_ashby_job(url="https://app.ashbyhq.com/acme/xyz-789")
    assert submitter.supports(job) is True


def test_ashby_submitter_not_for_lever():
    submitter = AshbyFormSubmitter()
    assert submitter.supports(_make_lever_job()) is False


# -- submit() tests ----------------------------------------------------------


@pytest.mark.asyncio
async def test_ashby_submitter_returns_scaffold_result():
    """AshbyFormSubmitter is an honest scaffold -- it returns handled=True,
    success=False with a message rather than silently submitting garbage."""
    submitter = AshbyFormSubmitter()
    package = ApplicationPackage(
        job=_make_ashby_job(),
        cv_path="/tmp/cv.pdf",
        cover_letter_path=None,
        answers={},
        profile={"personal": {"first_name": "Test", "email": "t@example.com"}},
    )
    result = await submitter.submit(package)
    assert result.handled is True
    assert result.success is False
    assert result.method == "ashby_api"
    assert result.error is not None
    assert (
        "not yet validated" in result.error.lower()
        or "scaffold" in result.error.lower()
        or "browser" in result.error.lower()
    )


@pytest.mark.asyncio
async def test_lever_submitter_handles_network_error_gracefully():
    """If the network call fails, LeverFormSubmitter returns handled=True,
    success=False (never raises)."""
    import app.services.jobs.backend_submitter as bs_module

    def bad_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    original_client = bs_module.httpx.AsyncClient

    class PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(bad_handler)
            super().__init__(*args, **kwargs)

    bs_module.httpx.AsyncClient = PatchedClient
    try:
        submitter = LeverFormSubmitter()
        package = ApplicationPackage(
            job=_make_lever_job(),
            cv_path="/tmp/cv.pdf",
            cover_letter_path=None,
            answers={},
            profile={"personal": {"first_name": "Test", "email": "t@example.com"}},
        )
        result = await submitter.submit(package)
    finally:
        bs_module.httpx.AsyncClient = original_client

    assert result.handled is True
    assert result.success is False
    assert result.error is not None
