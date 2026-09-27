# tests/test_agent_core.py
"""Unit tests for the new autonomous-agent core: dedupe, decision
engine, and the application state machine. Pure stdlib -- no jobspy /
browser-use / sentence-transformers required, so this runs anywhere.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.services.jobs.agent_schemas import (
    ApplicationState,
    InvalidTransition,
    NormalizedJob,
    assert_valid_transition,
)
from app.services.jobs.decision_engine import (
    CandidateProfile,
    MatchConfig,
    check_hard_rules,
    decide,
)
from app.services.jobs.dedupe import (
    Deduplicator,
    canonicalize_url,
    normalize_company,
    normalize_title,
)
from app.services.tracking.state_machine import ApplicationStateMachine


def make_job(**overrides) -> NormalizedJob:
    defaults = dict(
        job_id="1",
        title="Senior DevOps Engineer (m/w/d)",
        company="ACME GmbH",
        location="Bamberg, Germany",
        country="Germany",
        remote_type="hybrid",
        description="Kubernetes, Docker, Python, Terraform",
        application_url="https://de.indeed.com/viewjob?jk=abc123&utm_source=share",
    )
    defaults.update(overrides)
    return NormalizedJob(**defaults)


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


def test_normalize_company_strips_legal_suffix():
    assert normalize_company("ACME GmbH") == "acme"
    assert normalize_company("Beta Inc.") == "beta"


def test_normalize_title_strips_gender_suffix_and_synonyms():
    assert normalize_title("Senior DevOps Engineer (m/w/d)") == "senior devops engineer"
    assert normalize_title("Sr. Backend Engineer") == "senior backend engineer"


def test_canonicalize_url_strips_tracking_params():
    a = canonicalize_url("https://Example.com/jobs/123?utm_source=x&ref=y")
    b = canonicalize_url("https://example.com/jobs/123/")
    assert a == b


def test_dedupe_batch_keeps_distinct_postings_across_sources():
    job_indeed = make_job(
        job_id="indeed-1",
        source="jobspy_indeed",
        application_url="https://de.indeed.com/viewjob?jk=abc123",
        description="short",
    )
    job_company_site = make_job(
        job_id="acme-42",
        source="greenhouse",
        application_url="https://boards.greenhouse.io/acme/jobs/42",
        description="much longer and more complete description with details",
    )
    dedupe = Deduplicator()
    unique = dedupe.dedupe_batch([job_indeed, job_company_site])

    # Different URLs/ATS identifiers are distinct postings even when the
    # company/title/location happen to match.
    assert len(unique) == 2


def test_dedupe_batch_keeps_genuinely_different_jobs():
    job_a = make_job(
        job_id="1",
        title="DevOps Engineer",
        company="ACME",
        application_url="https://acme.example/jobs/1",
    )
    job_b = make_job(
        job_id="2",
        title="Data Scientist",
        company="Beta",
        application_url="https://beta.example/jobs/2",
    )
    dedupe = Deduplicator()
    unique = dedupe.dedupe_batch([job_a, job_b])
    assert len(unique) == 2


def test_dedupe_respects_previously_seen_fingerprints():
    job = make_job()
    dedupe = Deduplicator()
    dedupe.register(job)

    dedupe2 = Deduplicator(existing_fingerprints=set())
    # Simulate persisted fingerprint from a previous run
    from app.services.jobs.dedupe import compute_fingerprint

    fp = compute_fingerprint(job)
    dedupe2 = Deduplicator(existing_fingerprints={fp})
    assert dedupe2.is_duplicate(job) is True


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


def make_profile(**overrides) -> CandidateProfile:
    defaults = dict(
        skills={"python", "docker", "kubernetes"},
        years_experience=2.0,
        education_level="master",
        languages={"english": "c1", "german": "b1"},
        allowed_locations={"bamberg", "erlangen", "nürnberg"},
        remote_ok=True,
        requires_sponsorship=False,
        citizenship="pakistan",
        minimum_salary=None,
        willing_to_relocate=True,
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def test_hard_rule_skips_when_sponsorship_required_but_not_offered():
    job = make_job(visa_sponsorship_offered=False)
    profile = make_profile(requires_sponsorship=True)
    reasons = check_hard_rules(job, profile, MatchConfig())
    assert any("sponsorship" in r.lower() for r in reasons)


def test_hard_rule_skips_when_salary_below_minimum():
    job = make_job(salary_max=30000, salary_currency="EUR")
    profile = make_profile(minimum_salary=50000)
    reasons = check_hard_rules(job, profile, MatchConfig())
    assert any("salary" in r.lower() for r in reasons)


def test_hard_rule_skips_impossible_experience_gap():
    job = make_job(experience_required_years=10)
    profile = make_profile(years_experience=1)
    reasons = check_hard_rules(job, profile, MatchConfig())
    assert any("years" in r.lower() for r in reasons)


def test_hard_rule_does_not_skip_on_missing_preferred_skill_alone():
    job = make_job()  # no visa/salary/experience red flags
    profile = make_profile()
    reasons = check_hard_rules(job, profile, MatchConfig())
    assert reasons == []


def test_decide_apply_for_strong_match():
    job = make_job()
    profile = make_profile()
    result = decide(
        job,
        profile,
        MatchConfig(),
        matched_skills=["python", "docker", "kubernetes"],
        missing_skills=[],
        partial_skills=[],
    )
    assert result.decision == "APPLY"
    assert result.priority in {"HIGH_PRIORITY", "GOOD_MATCH"}
    assert result.overall_score >= MatchConfig().min_match_score


def test_decide_skip_for_weak_technical_match():
    job = make_job()
    profile = make_profile()
    result = decide(
        job,
        profile,
        MatchConfig(),
        matched_skills=["python"],
        missing_skills=["kubernetes", "terraform", "aws", "azure"],
        partial_skills=[],
    )
    assert result.decision == "SKIP"
    assert result.skip_reasons  # should explain why


def test_decide_skip_short_circuits_on_hard_rule_before_scoring():
    job = make_job(visa_sponsorship_offered=False)
    profile = make_profile(requires_sponsorship=True)
    result = decide(job, profile, MatchConfig(), matched_skills=["python", "docker", "kubernetes"])
    assert result.decision == "SKIP"
    assert result.overall_score == 0.0  # never even scored


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_valid_transition_allowed():
    assert_valid_transition(ApplicationState.DISCOVERED, ApplicationState.ANALYZING)
    assert_valid_transition(ApplicationState.MATCHED, ApplicationState.PREPARING)
    assert_valid_transition(ApplicationState.READY_TO_SUBMIT, ApplicationState.SUBMITTED)


def test_invalid_transition_rejected():
    with pytest.raises(InvalidTransition):
        assert_valid_transition(ApplicationState.DISCOVERED, ApplicationState.SUBMITTED)


def test_terminal_states_have_no_outgoing_transitions_except_offer():
    from app.services.jobs.agent_schemas import VALID_TRANSITIONS

    assert VALID_TRANSITIONS[ApplicationState.SKIPPED] == set()
    assert VALID_TRANSITIONS[ApplicationState.REJECTED] == set()


def test_document_and_browser_session_tracking(tmp_path, monkeypatch):
    from app.services.tracking import tracker

    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "doc_tracking.db")
    ApplicationStateMachine._migrated = False

    rec = ApplicationStateMachine.create(
        company_name="ACME",
        job_title="DevOps Engineer",
        job_url="https://x.com/1",
        fingerprint="fp-doc-1",
        source="greenhouse",
    )
    ApplicationStateMachine.record_document(
        rec["id"], "cv", "/tmp/cv_v1.pdf", version="tailored_v1"
    )
    ApplicationStateMachine.record_document(
        rec["id"], "cv", "/tmp/cv_v2.pdf", version="tailored_v2"
    )
    ApplicationStateMachine.record_document(rec["id"], "cover_letter", "/tmp/cl.pdf")

    docs = ApplicationStateMachine.documents(rec["id"])
    assert len(docs) == 3  # both CV versions kept, not overwritten
    assert [d["doc_type"] for d in docs] == ["cv", "cv", "cover_letter"]

    ApplicationStateMachine.record_browser_session(
        rec["id"], final_url="https://acme.example/apply", success=False, error_message="timeout"
    )
    ApplicationStateMachine.record_browser_session(
        rec["id"], final_url="https://acme.example/thank-you", success=True
    )
    sessions = ApplicationStateMachine.browser_sessions(rec["id"])
    assert len(sessions) == 2
    assert bool(sessions[0]["success"]) is False
    assert bool(sessions[1]["success"]) is True


# ---------------------------------------------------------------------------
# Section 31 — ApplicationResult.screenshot_path field exists
# ---------------------------------------------------------------------------


def test_application_result_has_screenshot_path_field():
    """ApplicationResult must expose a screenshot_path field so the
    orchestrator's record_browser_session call can actually store it.
    Previously the dataclass had no such field, meaning the column was
    always empty in practice."""
    import sys as _sys
    import types

    # browser_use may not be installed; stub it so the import works.
    if "browser_use" not in _sys.modules:
        stub = types.ModuleType("browser_use")
        for _name in (
            "Agent",
            "Browser",
            "ChatAnthropic",
            "ChatGoogle",
            "ChatOpenAI",
        ):
            setattr(stub, _name, type(_name, (), {}))
        _sys.modules["browser_use"] = stub

    from app.services.jobs.browser_use_applier import ApplicationResult

    # Default should be None (no screenshot captured yet).
    r = ApplicationResult(success=False, job_url="https://example.com")
    assert hasattr(r, "screenshot_path"), "ApplicationResult must have a screenshot_path field"
    assert r.screenshot_path is None

    # Field should accept a real path string.
    r2 = ApplicationResult(
        success=True,
        job_url="https://example.com",
        screenshot_path="/data/screenshots/submitted_20260101.png",
    )
    assert r2.screenshot_path == "/data/screenshots/submitted_20260101.png"


@pytest.mark.asyncio
async def test_capture_screenshot_helper_stores_file(tmp_path):
    """_capture_screenshot must call page.screenshot() and return the path.
    Tested with a mock page and browser so no real browser is needed."""
    import sys as _sys
    import types

    if "browser_use" not in _sys.modules:
        stub = types.ModuleType("browser_use")
        for _name in (
            "Agent",
            "Browser",
            "ChatAnthropic",
            "ChatGoogle",
            "ChatOpenAI",
        ):
            setattr(stub, _name, type(_name, (), {}))
        _sys.modules["browser_use"] = stub

    from unittest.mock import AsyncMock, MagicMock, patch

    from app.services.jobs.browser_use_applier import BrowserUseApplier

    # Build a fake page whose screenshot() writes a real file.
    fake_page = AsyncMock()

    async def fake_screenshot(path):  # noqa: D401
        Path(path).write_bytes(b"\x89PNG\r\n")

    fake_page.screenshot.side_effect = fake_screenshot

    fake_browser = AsyncMock()
    fake_browser.get_current_page.return_value = fake_page

    # Redirect the screenshots dir to tmp_path so we don't pollute the repo.
    with patch("app.services.jobs.browser_use_applier.PROJECT_ROOT", tmp_path):
        path = await BrowserUseApplier._capture_screenshot(fake_browser, label="test")

    assert path, "Expected a non-empty screenshot path"
    assert Path(path).exists(), f"Screenshot file must exist at {path}"
    assert path.endswith(".png")


@pytest.mark.asyncio
async def test_capture_screenshot_returns_empty_string_on_failure():
    """_capture_screenshot must never raise — it must return '' if anything
    goes wrong (browser already closed, Playwright unavailable, etc.)."""
    import sys as _sys
    import types

    if "browser_use" not in _sys.modules:
        stub = types.ModuleType("browser_use")
        for _name in (
            "Agent",
            "Browser",
            "ChatAnthropic",
            "ChatGoogle",
            "ChatOpenAI",
        ):
            setattr(stub, _name, type(_name, (), {}))
        _sys.modules["browser_use"] = stub

    from unittest.mock import AsyncMock

    from app.services.jobs.browser_use_applier import BrowserUseApplier

    exploding_browser = AsyncMock()
    exploding_browser.get_current_page.side_effect = RuntimeError("browser already closed")

    path = await BrowserUseApplier._capture_screenshot(exploding_browser, label="fail_test")
    assert path == "", f"Expected empty string on failure, got {path!r}"


# ---------------------------------------------------------------------------
# Semantic dedup
# ---------------------------------------------------------------------------


def test_semantic_dedup_keeps_distinct_postings_with_same_description():
    """Identical descriptions do not prove that two requisitions are the same."""
    pytest.importorskip("sklearn", reason="sklearn not installed; semantic dedup skip")

    from app.services.jobs.dedupe import semantic_dedup

    shared_desc = (
        "We are looking for an experienced Python developer to work on "
        "cloud infrastructure using Kubernetes, Terraform, and AWS. "
        "You will design scalable microservices and mentor junior engineers."
    )
    job_a = make_job(
        job_id="sem-1",
        company="ACME GmbH",
        source="greenhouse",
        application_url="https://boards.greenhouse.io/acme/jobs/1001",
        description=shared_desc,
    )
    job_b = make_job(
        job_id="sem-2",
        company="ACME GmbH",
        source="web_search",
        application_url="https://careers.acme.example/jobs/1001",
        description=shared_desc,
    )

    # Both have different fingerprints (different URLs) but same description.
    assert job_a.application_url != job_b.application_url

    result = semantic_dedup([job_a, job_b], threshold=0.85)
    assert len(result) == 2


def test_semantic_dedup_keeps_genuinely_different_descriptions():
    """Jobs with low description similarity must not be collapsed even
    when they share the same company."""
    pytest.importorskip("sklearn", reason="sklearn not installed; semantic dedup skip")

    from app.services.jobs.dedupe import semantic_dedup

    job_a = make_job(
        job_id="sem-3",
        company="ACME GmbH",
        description="Python backend developer with Django and REST API experience.",
    )
    job_b = make_job(
        job_id="sem-4",
        company="ACME GmbH",
        description="UX designer proficient in Figma and user research methodologies.",
    )

    result = semantic_dedup([job_a, job_b], threshold=0.85)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Seniority scoring
# ---------------------------------------------------------------------------


def test_seniority_score_senior_job_junior_candidate():
    """A junior candidate (2 years) applying to a senior role should receive
    a low seniority score."""
    from app.services.jobs.decision_engine import MatchConfig, score_job

    job = make_job(title="Senior Software Engineer", description="5+ years required, lead teams")
    profile = make_profile(years_experience=2.0)
    config = MatchConfig(candidate_years_experience=2)

    result = score_job(
        job,
        profile,
        config,
        matched_skills=["python", "docker", "kubernetes"],
        missing_skills=[],
        partial_skills=[],
    )
    # Senior job + junior candidate (2 yrs) -> seniority_score should be 20
    assert result.seniority_score == 20.0


def test_seniority_score_unknown_candidate_is_neutral():
    """When candidate_years_experience is 0 (unknown), seniority score
    should be neutral (75) and not dominate the overall score."""
    from app.services.jobs.decision_engine import MatchConfig, score_job

    job = make_job(title="Senior Software Engineer", description="5+ years required")
    profile = make_profile()
    config = MatchConfig(candidate_years_experience=0)

    result = score_job(
        job,
        profile,
        config,
        matched_skills=["python", "docker", "kubernetes"],
        missing_skills=[],
        partial_skills=[],
    )
    assert result.seniority_score == 75.0
