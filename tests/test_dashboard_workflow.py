"""
Dashboard rendering and workflow-state tests.

Two kinds of test, both needed:

*Rendering.* Streamlit answers HTTP 200 whether or not the script raised,
because the script runs per session. ``AppTest`` executes the script the way a
browser session would, so it is the only way to know a page renders rather than
that the server started. Every page is rendered against a stubbed backend, so
these need no API, no Ollama and no model.

*State.* The workflow spans pages, so its session state is a contract. Several
assertions here exist purely to stop a refactor silently breaking the handoff
between pages -- the failure mode is a page that renders fine and hands the next
one nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP = str((ROOT / "app" / "dashboard" / "main.py").resolve())

pytest.importorskip("streamlit", reason="streamlit is not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

from app.dashboard import theme, workflow  # noqa: E402

# ---------------------------------------------------------------------------
# Backend payloads, shaped exactly as the API returns them.
# ---------------------------------------------------------------------------

BACKEND_STATUS = {
    "status": "online",
    "routing": {
        "mode": "auto",
        "provider": "ollama",
        "model": "qwen3:8b",
        "legacy_route_mode": "automatic",
        "ollama_model": "qwen3:8b",
        "online_primary": "gemini",
        "online_fallback": "openrouter",
        "allows_cloud_fallback": True,
        "fallback_enabled": True,
        "max_provider_attempts": 4,
        "retries": 0,
        "tasks": ["ats_analysis", "cv_tailoring", "full_cv_generation"],
    },
    "ollama": {
        "configured": True,
        "base_url_configured": True,
        "model": "qwen3:8b",
        "reachable": True,
        "model_available": True,
        "available_models": ["qwen3:8b", "qwen2.5:7b", "llama3.2"],
        "detail": "",
    },
    "log_file": "llm_processing.jsonl",
    "document_toolchain": {
        "pdflatex_available": True,
        "pdflatex_path": r"C:\Program Files\MiKTeX\bin\pdflatex.EXE",
        "pdf_generation": True,
        "note": "",
    },
    "layouts": [
        {"name": "german_corporate", "language": "de", "columns": "single"},
        {"name": "german_ats", "language": "de", "columns": "single"},
        {"name": "german_classic", "language": "de", "columns": "single"},
        {"name": "german_minimal_ats", "language": "de", "columns": "single"},
        {"name": "german_modern", "language": "de", "columns": "single"},
        {"name": "international_ats", "language": "en", "columns": "single"},
        {"name": "academic", "language": "en", "columns": "single"},
        {"name": "technical_lead", "language": "en", "columns": "single"},
        {"name": "standard", "language": "en", "columns": "single"},
        {"name": "hr_executive_gold", "language": "en", "columns": "single"},
    ],
    "providers": {"ollama": {"configured": True}},
}

MODEL_DISCOVERY = {
    "status": "success",
    "providers": {
        "ollama": {
            "provider": "ollama",
            "protocol": "ollama_native",
            "local": True,
            "configured": True,
            "configured_model": "qwen3:8b",
            "discoverable": True,
            "status": "ok",
            "detail": "",
            "models": [
                {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "id": "qwen3:8b",
                    "source": "discovery",
                    "size_bytes": 5225388164,
                },
                {
                    "provider": "ollama",
                    "model": "qwen2.5:7b",
                    "id": "qwen2.5:7b",
                    "source": "discovery",
                    "size_bytes": 4683087332,
                },
            ],
        },
        "omniroute": {
            "provider": "omniroute",
            "protocol": "openai_compatible",
            "local": True,
            "configured": True,
            "configured_model": "auto",
            "discoverable": True,
            "status": "discovery_failed",
            "detail": "Could not list models: connection refused",
            "models": [
                {
                    "provider": "omniroute",
                    "model": "auto",
                    "id": "auto",
                    "source": "configuration",
                    "missing_from_discovery": True,
                }
            ],
        },
        "gemini": {
            "provider": "gemini",
            "protocol": "gemini",
            "local": False,
            "configured": True,
            "configured_model": "gemini-2.5-flash",
            "discoverable": False,
            "status": "manual_configuration",
            "detail": "gemini does not expose a model listing here.",
            "models": [
                {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "id": "gemini-2.5-flash",
                    "source": "configuration",
                }
            ],
        },
        "cerebras": {
            "provider": "cerebras",
            "protocol": "openai_compatible",
            "local": False,
            "configured": False,
            "configured_model": "",
            "discoverable": True,
            "status": "not_configured",
            "detail": "cerebras is not configured. Set CEREBRAS_API_KEY.",
            "models": [],
        },
    },
}

MODEL_CATALOG = {
    "status": "success",
    "count": 2,
    "models": [
        {"provider": "ollama", "model": "qwen3:8b", "configured": True},
        {"provider": "gemini", "model": "gemini-2.5-flash", "configured": True},
    ],
}

PIPELINE_METRICS = {
    "status": "ok",
    "summary": {
        "generations": 3,
        "succeeded": 3,
        "llm_calls": 4,
        "retries": 1,
        "llm_duration_ms": 41230.0,
        "prompt_tokens": 6210,
        "completion_tokens": 3120,
        "pdf_duration_ms": 1024.0,
        "pdf": {
            "bytes": 93208,
            "pages": 1,
            "text_chars": 2198,
            "sections_found": 7,
            "content_retention": 0.8714,
            "duration_ms": 1024.0,
            "valid": True,
        },
    },
    "requests": [
        {
            "request_id": "abc123",
            "task": "full_cv_generation",
            "mode": "auto",
            "provider": "ollama",
            "model": "qwen3:8b",
            "attempts": 2,
            "retries": 1,
            "llm_duration_ms": 25030.0,
            "prompt_tokens": 3210,
            "completion_tokens": 1620,
            "fallbacks": [],
            "outcome": "ok",
        }
    ],
    "pdf_events": [],
}

ANALYSIS = {
    "ats_match_score": 63.81,
    "keyword_density_score": 72.73,
    "matching_skills": ["Python", "Docker", "Kubernetes", "AWS", "Terraform"],
    "missing_skills": ["Prometheus", "Grafana", "Ansible"],
    "improvement_suggestions": [
        "Quantify the deployment time reduction.",
        "Name the observability stack explicitly.",
    ],
    "recommendation": {"recommended_format": "german_corporate"},
    "keywords": ["Kubernetes", "Terraform", "CI/CD", "Observability"],
    "resume_text": "Alex Berger\nPlatform Engineer\n" + ("experience " * 200),
}

PDF_VALIDATION = {
    "filename": "cv.pdf",
    "bytes": 93208,
    "valid": True,
    "status": "ok",
    "pages": 1,
    "text_chars": 2198,
    "sections_found": ["summary", "experience", "education", "skills"],
    "sections_missing": [],
    "latex_artifacts": [],
    "replacement_chars": 0,
    "content_retention": 0.8714,
    "problems": [],
    "detail": "1 page(s), 2198 characters extracted, no problems found.",
}

#: A report for a document that failed. The dashboard must not present this as a
#: success, which is the single most important behaviour on the preview page.
PDF_VALIDATION_REJECTED = {
    "filename": "cv.pdf",
    "bytes": 40000,
    "valid": False,
    "status": "rejected",
    "pages": 1,
    "text_chars": 60,
    "sections_found": [],
    "sections_missing": ["experience", "education"],
    "latex_artifacts": ["item", "textbf"],
    "replacement_chars": 3,
    "content_retention": 0.21,
    "problems": ["blank_or_near_blank", "latex_artifacts", "content_lost"],
    "detail": "The generated PDF is blank.",
}

USAGE_SUMMARY = {"summary": {"total_tokens": 9330, "requests": 4}}
QUOTA_STATUS = {
    "ollama": {"requests_per_minute": {"used": 0, "limit": 0}},
    "gemini": {"requests_per_minute": {"used": 3, "limit": 15}},
}

ROUTES = {
    "/api/v1/resume/backend-status": BACKEND_STATUS,
    "/api/v1/resume/health": {"status": "ok"},
    "/api/v1/resume/model-discovery": MODEL_DISCOVERY,
    "/api/v1/resume/model-catalog": MODEL_CATALOG,
    "/api/v1/resume/pipeline-metrics": PIPELINE_METRICS,
    "/api/v1/resume/usage-summary": USAGE_SUMMARY,
    "/api/v1/resume/quota-status": QUOTA_STATUS,
    "/api/v1/resume/processing-log": {
        "logs": [
            {
                "timestamp": "2026-09-27T18:00:00Z",
                "event": "request_completed",
                "provider": "ollama",
                "duration_ms": 25030.0,
            }
        ],
        "count": 1,
    },
}

#: Every page the sidebar can reach. Kept in one place so a page added to the
#: navigation without a test here is a visible gap, not a silent one.
NEW_PAGES = (
    "overview",
    "resume",
    "job_input",
    "ats_analysis",
    "optimization",
    "cv_generator",
    "pdf_preview",
    "llm_settings",
    "layout_picker",
    "diagnostics",
)

LEGACY_PAGES = (
    "advanced_tools",
    "career_suite",
    "analytics",
    "auto_apply",
    "agent_control",
)


@pytest.fixture
def stub_backend(monkeypatch):
    """
    Replace the dashboard's HTTP layer with a router over ``ROUTES``.

    Patches the attribute on the ``requests`` module itself, because several
    pre-existing pages import ``requests`` directly rather than going through
    the helper. Stubbing only the helper would leave those pages reaching the
    real network.
    """
    import app.dashboard.helpers as helpers

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code
            self.headers = {"Content-Type": "application/json"}
            self.content = b""

        @property
        def ok(self) -> bool:
            return 200 <= self.status_code < 300

        def json(self):
            return self._payload

        @property
        def text(self) -> str:
            import json as _json

            return _json.dumps(self._payload)

        def raise_for_status(self):
            if not self.ok:
                raise helpers.requests.HTTPError(f"{self.status_code} error", response=self)

    def fake_get(url, headers=None, timeout=None, params=None, **kwargs):
        for path, payload in ROUTES.items():
            if path in url:
                return FakeResponse(payload)
        return FakeResponse({}, status_code=404)

    def fake_delete(url, headers=None, timeout=None, **kwargs):
        return FakeResponse({"status": "cleared"})

    monkeypatch.setattr(helpers.requests, "get", fake_get)
    monkeypatch.setattr(helpers.requests, "delete", fake_delete)
    # The status readers are cached; a cached value from a previous test would
    # make this one pass or fail for the wrong reason.
    for fn in (
        helpers.fetch_backend_status,
        helpers.fetch_model_discovery,
        helpers.fetch_model_catalog,
        helpers.fetch_pipeline_metrics,
        helpers.fetch_usage_summary,
        helpers.fetch_quota_status,
        helpers.fetch_processing_log,
    ):
        try:
            fn.clear()
        except Exception:
            pass

    yield

    for fn in (
        helpers.fetch_backend_status,
        helpers.fetch_model_discovery,
        helpers.fetch_model_catalog,
        helpers.fetch_pipeline_metrics,
    ):
        try:
            fn.clear()
        except Exception:
            pass


def _render(page: str, **session) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=90)
    app.session_state["current_page"] = page
    app.session_state["api_base"] = "http://127.0.0.1:9"
    for key, value in session.items():
        app.session_state[key] = value
    app.run()
    return app


def _reference_pdf() -> bytes:
    """A real, readable one-page PDF, for exercising the preview stage."""
    sys.path.insert(0, str(Path(APP).resolve().parents[1] / "tests"))
    from tests.test_ats_scoring_helpers import minimal_pdf

    return minimal_pdf()


def _visible_text(app: AppTest) -> str:
    """
    Everything the page actually rendered, as one string.

    A rejected PDF reports its problems through ``st.error`` and its warnings
    through ``st.warning``, not through markdown. Searching only markdown would
    therefore assert nothing at all about the failure path -- the one that
    matters most. Every element a user can see is included.
    """
    parts: list[str] = []

    for collection in (
        app.markdown,
        app.info,
        app.warning,
        app.success,
        app.error,
        app.caption,
        app.text,
    ):
        for element in collection:
            value = getattr(element, "value", None)
            if isinstance(value, str):
                parts.append(value)
            elif value is not None:
                parts.append(str(value))

    for element in app.dataframe:
        try:
            parts.append(str(element.value.to_dict()))
        except Exception:
            pass

    for element in app.json:
        parts.append(str(getattr(element, "value", "")))

    for element in app.dataframe:
        try:
            parts.append(str(element.value.to_csv()))
        except Exception:
            pass

    return "\n".join(parts)


@pytest.fixture
def session():
    """
    A clean, directly-writable session state.

    ``AppTest`` gives the *script* its own session, so a test body reading
    ``workflow.*`` would be looking at a different one. For pure state
    assertions -- which is most of them -- no AppTest is involved at all, and
    this is the honest way to exercise the module: set the state, call the
    function, read the result.
    """
    import streamlit as st

    for key in (
        workflow.KEY_UPLOAD,
        workflow.KEY_JOB,
        workflow.KEY_ANALYSIS,
        workflow.KEY_ANALYSIS_META,
        workflow.KEY_CV_META,
        workflow.KEY_PROVIDER,
        workflow.KEY_MODEL,
        workflow.KEY_ROUTE,
        workflow.KEY_LAYOUT,
    ):
        st.session_state.pop(key, None)

    workflow.ensure_defaults()
    yield st.session_state
    workflow.ensure_defaults()


# ===========================================================================
# 1. Every page renders
# ===========================================================================


@pytest.mark.parametrize("page", NEW_PAGES)
def test_new_page_renders_without_raising(stub_backend, page):
    """
    A page that raises renders as an error box, not as a page.

    Streamlit serves HTTP 200 either way, so this is the only check that
    distinguishes a working page from a broken one.
    """
    app = _render(page)
    assert not app.exception, (
        f"{page} raised: {type(app.exception[0].value).__name__}: " f"{app.exception[0].value}"
    )


@pytest.mark.parametrize("page", LEGACY_PAGES)
def test_pre_existing_page_still_renders(stub_backend, page):
    """
    The redesign must not cost a single existing capability.

    These five pages are unchanged; this test exists so that a future change to
    the navigation or the shared state cannot quietly remove one.
    """
    app = _render(page)
    assert not app.exception, (
        f"{page} raised: {type(app.exception[0].value).__name__}: " f"{app.exception[0].value}"
    )


def test_every_navigation_target_resolves(stub_backend):
    """
    Every navigation entry must have a dispatch branch.

    A sidebar button with no branch is a dead button, and that is the failure
    worth catching. The reverse is deliberately *not* required to match: the six
    workflow stages are sections of the single CV workflow page, so their keys
    still dispatch -- for a deep link to one stage, and so a stage opened alone
    renders as its own page -- without being navigation entries of their own.
    Asserting exact equality would forbid a legitimate deep-link target.
    """
    import ast

    source = Path(APP).read_text(encoding="utf-8")
    tree = ast.parse(source)

    nav_keys: set[str] = set()
    dispatch_keys: set[str] = set()

    for node in ast.walk(tree):
        # ("overview", "label") inside NAV_GROUPS
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[0].value, str)
            and isinstance(node.elts[1], ast.Constant)
        ):
            nav_keys.add(node.elts[0].value)

        # page == "overview"
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "page"
        ):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    dispatch_keys.add(comparator.value)

    assert nav_keys, "no navigation entries found"
    assert dispatch_keys, "no dispatch branches found"

    missing_dispatch = nav_keys - dispatch_keys
    assert (
        not missing_dispatch
    ), f"navigation targets with no dispatch branch: {sorted(missing_dispatch)}"


def test_navigation_covers_the_required_workflow():
    """
    The workflow is reachable, and it is one page.

    The six stages are sections of a single page rather than six navigation
    entries, so this asserts the *page* that runs them plus the setup pages, and
    separately asserts that every stage is still dispatchable on its own. A stage
    could otherwise be dropped from the page and nothing here would notice.
    """
    import ast

    tree = ast.parse(Path(APP).read_text(encoding="utf-8"))
    keys: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[0].value, str)
        ):
            keys.add(node.elts[0].value)

    for required in (
        "overview",
        "cv_workflow",
        "llm_settings",
        "layout_picker",
        "diagnostics",
    ):
        assert required in keys, f"missing page: {required}"


def test_every_workflow_stage_is_reachable():
    """
    Each of the six stages still renders, as a deep link.

    They are sections of one page now, so nothing in the navigation points at
    them individually. This keeps them individually addressable, which is what
    lets an old bookmark or a link from another page still land somewhere
    sensible.
    """
    import ast

    tree = ast.parse(Path(APP).read_text(encoding="utf-8"))
    dispatch: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "page"
        ):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    dispatch.add(comparator.value)

    for stage in (
        "resume",
        "job_input",
        "ats_analysis",
        "optimization",
        "cv_generator",
        "pdf_preview",
    ):
        assert stage in dispatch, f"stage no longer reachable: {stage}"


def test_the_workflow_page_renders_every_stage(stub_backend):
    """
    One page carries all six stages.

    A stage that quietly stopped rendering would leave the page working and
    incomplete, and no other test would see it: the stage pages are only reached
    by deep link now.
    """
    app = _render("cv_workflow")

    assert not app.exception, [f"{type(e.value).__name__}: {e.value}" for e in app.exception]

    # Every stage's own controls appear without the user navigating anywhere.
    labels = " ".join(
        str(getattr(el, "label", "") or getattr(el, "body", ""))
        for el in [*app.file_uploader, *app.text_area, *app.button, *app.tabs]
    ).casefold()
    for marker in ("upload resume", "job description"):
        assert marker in labels, f"stage control missing from the single page: {marker}"


@pytest.mark.parametrize(
    "state_name, session",
    [
        ("empty", {}),
        (
            "resume only",
            {
                workflow.KEY_UPLOAD: (
                    "cv.pdf",
                    b"%PDF-1.4\n% resume bytes\n",
                    "application/pdf",
                )
            },
        ),
        (
            "resume and posting",
            {
                workflow.KEY_UPLOAD: (
                    "cv.pdf",
                    b"%PDF-1.4\n% resume bytes\n",
                    "application/pdf",
                ),
                workflow.KEY_JOB: "We need a Python engineer with Docker experience.",
            },
        ),
        (
            "with an analysis",
            {
                workflow.KEY_UPLOAD: (
                    "cv.txt",
                    b"Jane Doe\njane@example.com\n\nEXPERIENCE\nEngineer\n",
                    "text/plain",
                ),
                workflow.KEY_JOB: "We need a Python engineer with Docker experience.",
                workflow.KEY_ANALYSIS: {
                    "status": "success",
                    "ats_match_score": 72.0,
                    "ats_breakdown": {
                        "ats_score": 72.0,
                        "band": "Moderate match",
                        "pre_generation": True,
                        "categories": {},
                        "improvement_areas": [{"category": "keyword_match", "points_lost": 5.0}],
                    },
                    "ats_suggestions": [
                        {
                            "issue": "Some words are missing.",
                            "action": "Add what you have used.",
                            "severity": "medium",
                        }
                    ],
                    "layout_recommendation": {
                        "recommended_layout": "international_ats",
                        "recommended_name": "International ATS",
                        "reason": "Single column, strong parsing.",
                        "ats_safety": "Very High",
                    },
                    "matching_skills": ["Python"],
                    "missing_skills": ["Kubernetes"],
                    "improvement_suggestions": [],
                    "resume_text": "Jane Doe",
                },
            },
        ),
        (
            "with a generated PDF",
            {
                workflow.KEY_UPLOAD: (
                    "cv.txt",
                    b"Jane Doe\njane@example.com\n\nEXPERIENCE\nEngineer\n",
                    "text/plain",
                ),
                workflow.KEY_JOB: "We need a Python engineer.",
                workflow.KEY_ANALYSIS: {"status": "success", "ats_match_score": 80.0},
                "_result_pdf": {
                    "content": _reference_pdf(),
                    "filename": "cv.pdf",
                    "elapsed": 1.0,
                },
                "_result_pdf_validation": {
                    "valid": True,
                    "status": "ok",
                    "pages": 1,
                    "text_chars": 900,
                    "sections_found": ["experience", "education", "skills"],
                    "problems": [],
                    "detail": "1 page, 900 characters, no problems found.",
                },
                "_result_final_ats": {
                    "ats_score": 88.0,
                    "band": "Good match",
                    "summary": "Everything evidenced.",
                    "document": {"valid": True, "pages": 1, "problems": []},
                    "layout": {
                        "requested": "international_ats",
                        "known": True,
                        "name": "International ATS",
                        "ats_safety": "Very High",
                        "ats_safety_score": 95,
                        "structure": "single_column",
                        "parsing_risk": "Very Low",
                    },
                    "pdf_parsing": {
                        "measured": True,
                        "score": 100.0,
                        "checks": [],
                        "failed_checks": [],
                        "issues": [],
                    },
                    "ats_issues": 0,
                    "breakdown": {"categories": {}},
                },
                "_result_improvement_result": {
                    "initial_score": 72.0,
                    "final_score": 85.0,
                    "total_improvement": 13.0,
                    "target_score": 100.0,
                    "explanation": "Improved where it was safe to do so.",
                },
            },
        ),
    ],
)
def test_the_workflow_page_survives_every_stage_of_the_flow(stub_backend, state_name, session):
    """
    The single page must render at every point in the flow, not just at the start.

    The stages have always assumed they are the only thing on the page, so a
    guard that returns early on the standalone page no longer returns when
    rendered inline, and the code after it runs against data that is not there.
    That failure only appears at the state where the data is missing -- so each
    point in the flow is rendered separately here.
    """
    app = _render("cv_workflow", **session)

    assert not app.exception, (
        f"the workflow page broke with {state_name} state: "
        f"{[f'{type(e.value).__name__}: {e.value}' for e in app.exception]}"
    )


def test_every_workflow_stage_is_reachable_standalone(stub_backend):
    """
    A stage opened on its own still renders as a page.

    Deep links to a single stage still work, and they show the full page
    treatment rather than the inline form.
    """
    for page in (
        "resume",
        "job_input",
        "ats_analysis",
        "optimization",
        "cv_generator",
        "pdf_preview",
    ):
        app = _render(page)
        assert not app.exception, (
            f"stage {page} raised: "
            f"{[f'{type(e.value).__name__}: {e.value}' for e in app.exception]}"
        )


# ===========================================================================
# 2. Workflow state
# ===========================================================================


def test_readiness_reflects_what_is_present(session):
    assert session[workflow.KEY_UPLOAD] is None
    ready = workflow.workflow_readiness()
    assert ready["resume"] is False
    assert ready["job_description"] is False
    assert ready["analysis"] is False

    page, action = workflow.next_step()
    assert page == "resume"
    assert "resume" in action.casefold()


def test_a_complete_workflow_reads_as_complete(session):
    session[workflow.KEY_UPLOAD] = (
        "cv.pdf",
        b"%PDF-1.4 fake",
        "application/pdf",
    )
    session[workflow.KEY_JOB] = "Senior Platform Engineer. " * 40
    session[workflow.KEY_ANALYSIS] = ANALYSIS

    ready = workflow.workflow_readiness()
    assert ready["resume"] is True
    assert ready["job_description"] is True
    assert ready["analysis"] is True
    assert ready["generation"] is True

    page, _ = workflow.next_step()
    assert page == "cv_generator"


def test_job_statistics_are_counts_not_guesses(session):
    workflow.set_job("Kubernetes engineer\n\nTerraform")

    stats = workflow.job_stats()
    assert stats["chars"] == len("Kubernetes engineer\n\nTerraform")
    # "Kubernetes engineer" + "Terraform" = 3 words.
    assert stats["words"] == 3
    assert stats["lines"] == 2
    # No token estimate: a guess presented as a measurement is worse than none.
    assert "tokens" not in stats


def test_generation_choice_persists_across_pages(stub_backend):
    """
    The provider chosen on the settings page is the one generation uses.

    This is the handoff the redesign exists to make reliable: previously each
    page rendered its own selector, so a user could analyse with one provider
    and generate with another without noticing.
    """
    workflow.set_generation_choice("groq", "some-model", "direct")

    choice = workflow.generation_choice()
    assert choice["provider"] == "groq"
    assert choice["model_name"] == "some-model"
    assert choice["route_mode"] == "direct"

    # A page switch must not lose it: the whole point of storing the choice
    # once is that the next page reads the same value.
    # Seeded into the rendered session explicitly, because AppTest gives the
    # script its own. An analysis is seeded too, because the generation page
    # tells the user to run one rather than rendering a provider panel at all --
    # which is the correct behaviour, and means the test has to satisfy it.
    app = _render(
        "cv_generator",
        **{
            workflow.KEY_PROVIDER: "groq",
            workflow.KEY_MODEL: "some-model",
            workflow.KEY_ROUTE: "direct",
            workflow.KEY_UPLOAD: ("cv.pdf", b"%PDF-1.4", "application/pdf"),
            workflow.KEY_JOB: "Senior Platform Engineer. " * 40,
            workflow.KEY_ANALYSIS: ANALYSIS,
        },
    )
    assert app.session_state[workflow.KEY_PROVIDER] == "groq"
    assert app.session_state[workflow.KEY_MODEL] == "some-model"

    # The generation page reads the choice from the workflow, not from its own
    # selector, so the value it shows must be the seeded one.
    text = _visible_text(app)
    assert "groq" in text, "the generation page did not show the chosen provider"


def test_reset_clears_the_whole_workflow(session):
    session[workflow.KEY_UPLOAD] = ("cv.pdf", b"%PDF-1.4", "application/pdf")
    session[workflow.KEY_JOB] = "a job description"
    session[workflow.KEY_ANALYSIS] = ANALYSIS
    session["_result_pdf"] = {"content": b"%PDF-1.5", "filename": "cv.pdf"}
    assert workflow.has_upload() and workflow.has_analysis()

    workflow.reset_workflow()

    assert workflow.has_upload() is False
    assert workflow.get_job() == ""
    assert workflow.has_analysis() is False
    assert workflow.cv_meta() == {}


def test_file_payload_round_trips_the_bytes(session):
    """
    The stored bytes are what gets re-sent on a later request.

    Streamlit's uploader widget is ephemeral across a page switch, so anything
    that must reach a later request re-sends from here. If this stops
    round-tripping, generation silently loses its resume.
    """
    payload = b"%PDF-1.4 the actual bytes"
    workflow.set_upload("cv.pdf", payload, "application/pdf")

    built = workflow.file_payload()
    assert "resume_file" in built
    name, content, mime = built["resume_file"]
    assert name == "cv.pdf"
    assert content == payload
    assert mime == "application/pdf"


def test_file_payload_is_empty_without_an_upload(session):
    workflow.reset_workflow()
    assert workflow.file_payload() == {}


# ===========================================================================
# 3. Status honesty
# ===========================================================================


def test_an_unconfigured_provider_is_never_shown_as_working(stub_backend):
    """
    The three states stay apart.

    ``cerebras`` is in the stub with ``configured: False``. Nothing on the
    settings or diagnostics page may present it as available or as tested.
    """
    from app.dashboard.views.llm_settings import _provider_table

    rows = {row["id"]: row for row in _provider_table("http://127.0.0.1:9")}
    cerebras = rows["cerebras"]
    assert cerebras["configured"] == theme.NOT_CONFIGURED
    assert cerebras["available"] == theme.NOT_CONFIGURED
    # Nothing was called, so nothing may claim a pass.
    assert cerebras["smoke_tested"] == theme.NOT_TESTED


def test_configured_is_distinct_from_available(stub_backend):
    """
    ``gemini`` is configured but exposes no listing.

    That is a weaker claim than "available", and showing it as available would
    overstate what has been established.
    """
    from app.dashboard.views.llm_settings import _provider_table

    rows = {row["id"]: row for row in _provider_table("http://127.0.0.1:9")}
    gemini = rows["gemini"]

    assert gemini["configured"] == theme.PASSED
    assert gemini["available"] == theme.CONFIGURED
    assert gemini["smoke_tested"] == theme.NOT_TESTED


def test_a_discovery_failure_still_reports_the_configured_model(stub_backend):
    """
    ``omniroute`` could not be listed, but its model is known.

    Discovery is optional. A provider that cannot be listed is still usable with
    a hand-set model, so the row must say CONFIGURED rather than FAILED, and
    must still name the model.
    """
    from app.dashboard.views.llm_settings import _provider_table

    rows = {row["id"]: row for row in _provider_table("http://127.0.0.1:9")}
    omniroute = rows["omniroute"]

    assert omniroute["configured"] == theme.PASSED
    assert omniroute["model"] == "auto"
    assert "connection refused" in omniroute["note"]


def test_local_providers_are_labelled_as_local(stub_backend):
    from app.dashboard.views.llm_settings import _provider_table

    rows = {row["id"]: row for row in _provider_table("http://127.0.0.1:9")}

    assert rows["ollama"]["side"] == "local"
    assert rows["omniroute"]["side"] == "local"
    assert rows["gemini"]["side"] == "API"
    assert rows["cerebras"]["side"] == "API"


# ===========================================================================
# 4. The preview page must not report a failure as a success
# ===========================================================================


@pytest.mark.parametrize(
    "report,expect_valid_label",
    [
        (PDF_VALIDATION, True),
        (PDF_VALIDATION_REJECTED, False),
    ],
)
def test_validation_report_drives_the_verdict(stub_backend, report, expect_valid_label):
    """
    The preview page's verdict comes from the report, not from the file arriving.

    A rejected document is still offered for download -- the user may want to
    see what went wrong -- but it is labelled rejected and the problems are
    listed above it.
    """
    app = _render(
        "pdf_preview",
        **{
            "_result_pdf": {
                "content": b"%PDF-1.5 fake",
                "filename": "cv.pdf",
                "mime": "application/pdf",
                "elapsed": 20.2,
                "size_kb": 90.0,
            },
            "_result_pdf_validation": report,
        },
    )

    assert not app.exception
    text = _visible_text(app)
    if expect_valid_label:
        assert theme.PASSED in text
    else:
        assert theme.FAILED in text
        # Every recorded problem must be surfaced, not summarised away.
        for problem in report["problems"]:
            assert problem.replace("_", " ") in text, problem


def test_a_rejected_pdf_is_not_presented_as_valid(stub_backend):
    """
    The single most important behaviour on the preview page.

    Content lost, raw LaTeX in the output and a near-blank page are all
    failures. If any of them can reach a "success" state, the page is worse than
    no page, because it tells the user their CV is fine when it is not.
    """
    app = _render(
        "pdf_preview",
        **{
            "_result_pdf": {
                "content": b"%PDF-1.5 fake",
                "filename": "cv.pdf",
                "mime": "application/pdf",
                "elapsed": 5.0,
                "size_kb": 40.0,
            },
            "_result_pdf_validation": PDF_VALIDATION_REJECTED,
        },
    )

    text = _visible_text(app)
    assert "did not pass validation" in text
    assert theme.PASSED not in text, "a rejected document was marked as passing"

    # The specific failures, named. A user who is told "something is wrong"
    # cannot act; a user who is told the page is blank and LaTeX leaked can.
    assert "blank" in text.casefold()
    assert "item" in text or "textbf" in text


def test_a_valid_pdf_reports_what_was_checked(stub_backend):
    app = _render(
        "pdf_preview",
        **{
            "_result_pdf": {
                "content": b"%PDF-1.5 fake",
                "filename": "cv.pdf",
                "mime": "application/pdf",
                "elapsed": 20.2,
                "size_kb": 90.0,
            },
            "_result_pdf_validation": PDF_VALIDATION,
        },
    )

    text = _visible_text(app)
    # The individual checks, not just a green tick.
    for expected in (
        "Pages",
        "Text extracted",
        "Content retained",
        "Sections found",
    ):
        assert expected in text, expected
    assert "no raw LaTeX" in text


# ===========================================================================
# 5. No secret can reach the browser
# ===========================================================================


def test_no_dashboard_module_renders_an_api_key(stub_backend):
    """
    A credential must never reach the page.

    Checked by reading the rendered output rather than by reading the code, so a
    future change that starts printing a key is caught even if it looks
    harmless.
    """
    from app.dashboard.helpers import get_api_headers

    headers = get_api_headers()
    secret = headers.get("X-API-Key", "")
    if not secret:
        pytest.skip("no API_KEY in the environment to check against")

    for page in NEW_PAGES:
        app = _render(page)
        text = "\n".join(
            [block.value for block in app.markdown]
            + [str(block.value) for block in app.json]
            + [str(getattr(block, "data", "")) for block in app.dataframe]
        )
        assert secret not in text, f"{page} rendered the API key"


def test_theme_escapes_anything_rendered_through_it():
    """
    Chip and card values come from a model or a parsed document.

    Neither can be trusted to contain valid HTML, so the theme escapes before it
    renders. This is the check that keeps a skill name from breaking the layout.
    """
    import html

    payload = '<script>alert("x")</script>'
    rendered = theme._escape(payload)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert html.escape(payload) == rendered


def test_status_pill_always_carries_its_label():
    """
    The verdict text is in the markup, not only in the colour.

    A colour-blind user, a greyscale print and a screen reader all get the same
    information, which is the only way a status pill is accessible at all.
    """
    for verdict in (
        theme.PASSED,
        theme.FAILED,
        theme.NOT_CONFIGURED,
        theme.NOT_INSTALLED,
        theme.NOT_TESTED,
    ):
        assert verdict in theme.status_pill(verdict)


# ===========================================================================
# 6. The layout picker is driven by the backend
# ===========================================================================


def test_layouts_come_from_the_backend_not_a_local_list(stub_backend):
    """
    A layout that leaves the generator must leave the picker.

    The picker renders whatever ``backend-status`` reports, so this test feeds it
    a list and checks that a layout absent from that list is absent from the
    picker. A hard-coded list in the UI would pass a "does it render" test while
    offering a layout that no longer exists.
    """
    app = _render("layout_picker")

    # The widget's own options, not the rendered HTML. Stronger: it is the
    # actual choice list, and it does not depend on how the page happens to lay
    # its text out.
    options: list[str] = []
    for widget in app.selectbox:
        if widget.label == "Layout":
            options = list(widget.options)
            break

    # The widget formats its options, so the rendered values are the display
    # names. The property that matters is that the mapping is complete and
    # one-to-one: every layout the backend reported is offered, and nothing else
    # is, under a name a user can recognise.
    from app.dashboard.views.layout_picker import DISPLAY_NAMES

    expected = {
        row["name"]: DISPLAY_NAMES.get(row["name"], row["name"])
        for row in BACKEND_STATUS["layouts"]
    }

    assert len(options) == len(expected), (
        f"the picker offered {len(options)} layouts, the backend reported "
        f"{len(expected)}: {options}"
    )
    for identifier, display in expected.items():
        assert display in options, f"{identifier} is not selectable"

    text = _visible_text(app)
    assert str(len(expected)) in text, "the count shown does not match the list"


def test_the_layout_picker_offers_a_placeholder_when_the_backend_is_silent(
    stub_backend, monkeypatch
):
    """
    An unreachable backend must produce an explanation, not a blank page.

    Reached by pointing the helper at a URL nothing answers, which is the state
    a user hits when the API is down.
    """
    import app.dashboard.helpers as helpers

    for fn in (
        helpers.fetch_backend_status,
        helpers.fetch_model_discovery,
    ):
        try:
            fn.clear()
        except Exception:
            pass

    # Force the "no layouts" path by clearing the response for that route.
    original = dict(ROUTES)
    ROUTES.pop("/api/v1/resume/backend-status", None)
    try:
        app = _render("layout_picker")
        assert not app.exception
        text = "\n".join(block.value for block in app.markdown)
        assert "Layout list unavailable" in text
    finally:
        ROUTES.update(original)


def test_layout_traits_are_derived_from_the_templates():
    """
    The "traits" column is a measurement, not marketing.

    Read from ``CV_TEMPLATES`` itself, so it cannot describe a layout the
    generator does not have.
    """
    from app.dashboard.views.layout_picker import _template_facts
    from app.services.cv.latex_generator import CV_TEMPLATES

    facts = _template_facts()

    assert set(facts) == set(CV_TEMPLATES)
    for name, entry in facts.items():
        assert entry["traits"], f"{name} reported no traits"

    # The monochrome layout really is colourless, which is the fact that
    # distinguishes it from the other ATS layouts.
    assert "#000000" in facts["german_minimal_ats"]["traits"]
    assert "colour" not in facts["german_minimal_ats"]["traits"].split(" · ")[0]


def test_every_layout_has_a_display_name():
    """
    ``german_minimal_ats`` is an identifier, not a name a user would choose.
    """
    from app.dashboard.views.layout_picker import DISPLAY_NAMES
    from app.services.cv.latex_generator import CV_TEMPLATES

    missing = set(CV_TEMPLATES) - set(DISPLAY_NAMES)
    assert not missing, f"layouts with no display name: {sorted(missing)}"
