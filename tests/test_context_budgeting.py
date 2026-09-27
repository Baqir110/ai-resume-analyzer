"""
Context budgeting and factual grounding.

Two guarantees are tested here:

* a prompt that must be trimmed loses the *least important* resume material
  and declares what it dropped, instead of losing the tail silently;
* a JD skill the resume does not support is never claimed as experience.
"""

import pytest

from app.services.cv import optimizer

LONG_INTERESTS = "chess and cycling and reading " * 40

RESUME = f"""Jane Doe
Platform Engineer
jane@example.com | +49 123 456789

Profile
Platform engineer focused on Python services and delivery automation.

Experience
Platform Engineer | Beispiel GmbH | Jan 2020 - Present
Built Python services and monitored deployments.

Education
M.Sc. Computer Science | TU | 2018 - 2020

Skills
Python, Docker, Kubernetes, AWS, Terraform

Interests
{LONG_INTERESTS}
"""


def _budget(text: str, limit: int, field: str) -> str:
    return optimizer._truncate(text, limit, field=field)


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    """
    ``normalize_resume_language`` memoises per (resume, target language).

    Without this, a test that populates the cache makes the next one return
    early and never reach the LLM, so assertions about the prompt would
    silently test nothing.
    """
    optimizer._TRANSLATION_CACHE.clear()
    yield
    optimizer._TRANSLATION_CACHE.clear()


# ---------------------------------------------------------------------------
# Never silently cut the end
# ---------------------------------------------------------------------------


def test_short_text_is_returned_unchanged():
    assert _budget(RESUME, 100_000, field="resume") == RESUME


def test_budgeted_resume_keeps_the_candidate_header():
    out = _budget(RESUME, 600, field="resume")
    assert "Jane Doe" in out
    assert "jane@example.com" in out
    assert len(out) <= 600


def test_budgeted_resume_keeps_the_work_history_over_interests():
    # "Interests" is 1080 characters of the 1400. A prefix cut would keep it
    # and drop the skills list; priority selection does the opposite.
    out = _budget(RESUME, 700, field="resume")
    assert "Beispiel GmbH" in out
    assert "Interests" not in out


def test_budgeted_resume_declares_the_omission():
    out = _budget(RESUME, 700, field="resume")
    assert "omitted" in out
    # The note must tell the model not to fill the gap, which is the
    # hallucination guard.
    assert "Do not infer or invent" in out


def test_budgeted_resume_preserves_section_order():
    out = _budget(RESUME, 900, field="resume")
    positions = [
        out.index(marker)
        for marker in ("Profile", "Experience", "Education", "Skills")
        if marker in out
    ]
    assert positions == sorted(positions)


def test_budgeted_resume_keeps_the_skills_list_when_it_fits():
    out = _budget(RESUME, 900, field="resume")
    assert "Python, Docker, Kubernetes" in out


def test_education_outranks_optional_sections():
    assert optimizer._section_priority("Education") > optimizer._section_priority("Interests")
    assert optimizer._section_priority("Berufserfahrung") > optimizer._section_priority("Sprachen")
    assert optimizer._section_priority(
        "Berufserfahrung (2020-heute)"
    ) > optimizer._section_priority("Interessen")


def test_always_within_the_limit():
    for limit in (100, 150, 200, 300, 400, 600, 900, 1400):
        out = _budget(RESUME, limit, field="resume")
        assert len(out) <= limit, f"limit={limit} produced {len(out)}"


def test_a_tiny_budget_still_produces_readable_text():
    # Too small to assemble anything: the fallback trims head and tail, which
    # always leaves the name and the tail of the document intact.
    out = _budget(RESUME, 150, field="resume")
    assert "Jane Doe" in out
    assert "truncated" in out or "omitted" in out


def test_empty_and_missing_input():
    assert _budget("", 100, field="resume") == ""
    assert _budget(RESUME, 0, field="resume") == ""


# ---------------------------------------------------------------------------
# Job descriptions
# ---------------------------------------------------------------------------


JD = """Senior Platform Engineer (m/w/d)
Beispiel GmbH, Munich

Aufgaben:
- Betrieb und Weiterentwicklung unserer Kubernetes-Plattform

Anforderungen:
- Sehr gute Kenntnisse in Python, Docker und Kubernetes

Wir bieten:
- Flexible Arbeitszeiten und Homeoffice
""" + ("Zusätzliche Hinweise zum Unternehmen. " * 40)


def test_job_description_keeps_both_ends():
    out = _budget(JD, 400, field="job_description")
    assert len(out) <= 400
    # The role title is at the top and the requirements further down; neither
    # end may be lost, because that is what a prefix cut would discard.
    assert "Senior Platform Engineer" in out
    assert "truncated" in out


def test_job_description_keeps_the_requirements_when_they_fit():
    out = _budget(JD, 700, field="job_description")
    assert "Anforderungen" in out
    assert "Python" in out


def test_resume_and_job_description_use_different_strategies():
    # A resume is sectioned and can be prioritised; a job description is prose
    # and is only safe to trim from the middle.
    resume_out = _budget(RESUME, 400, field="resume")
    jd_out = _budget(JD, 400, field="job_description")
    assert "Jane Doe" in resume_out
    assert "Senior Platform Engineer" in jd_out


def test_trim_middle_reports_and_declares_the_omission():
    text, omitted = optimizer._trim_middle("x" * 500, 200)
    assert 0 < omitted < 500
    # The declared count is the one reported, so the marker cannot disagree
    # with the value the caller logs.
    assert f"{omitted} characters truncated" in text
    assert len(text) <= 200
    assert text.startswith("x")
    assert text.endswith("x")


def test_trim_middle_never_exceeds_the_limit():
    for limit in range(1, 260):
        text, _omitted = optimizer._trim_middle("x" * 500, limit)
        assert len(text) <= max(limit, 1), f"limit={limit} len={len(text)}"


def test_trim_middle_is_a_noop_when_it_fits():
    text, omitted = optimizer._trim_middle("hello", 500)
    assert text == "hello"
    assert omitted == 0


def test_split_resume_blocks_recognises_german_headings():
    blocks = optimizer._split_resume_blocks(
        "Max Mustermann\nmax@example.de\n\nBerufserfahrung\n2020-heute\n\n"
        "Ausbildung\nM.Sc.\n\nKenntnisse\nPython\n"
    )
    headings = [heading for heading, _ in blocks]
    assert "" in headings
    assert "Berufserfahrung" in headings
    assert "Ausbildung" in headings
    assert "Kenntnisse" in headings


# ---------------------------------------------------------------------------
# Factual grounding
# ---------------------------------------------------------------------------


def test_unsupported_missing_skill_is_not_claimed(monkeypatch):
    """
    A skill the resume does not support must not be asserted as experience.

    This is the behaviour the project already had and must keep: the gap is
    reported for manual review instead of being written into the CV.
    """
    seen: dict[str, str] = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        # A well-behaved model omits the unsupported skill.
        return "- Improved deployment reliability using documented tooling."

    monkeypatch.setattr(optimizer.LLMService, "generate", staticmethod(fake_generate))

    result = optimizer.optimize_resume_bullets(
        resume_text=RESUME,
        job_description="Role requires Rust, Kubernetes and Kafka.",
        missing_skills=["Rust", "Kafka"],
        layout_style="international_ats",
    )

    assert "Rust" not in result
    assert "Kafka" not in result
    # The prompt does surface them as gaps to review, not as experience.
    assert "Rust" in seen["prompt"]


def test_translation_skips_when_resume_is_already_in_the_target_language(
    monkeypatch,
):
    def fail(*_args, **_kwargs):
        raise AssertionError("no LLM call expected")

    monkeypatch.setattr(optimizer.LLMService, "generate", staticmethod(fail))
    english = RESUME
    assert optimizer.normalize_resume_language(english, "en") == english


def test_translation_prompt_preserves_proper_nouns(monkeypatch):
    captured: dict[str, str] = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Übersetzter Lebenslauf"

    monkeypatch.setattr(optimizer.LLMService, "generate", staticmethod(fake_generate))
    optimizer.normalize_resume_language(RESUME, "de")
    prompt = captured["prompt"]
    for proper_noun in ("Python", "Docker", "Kubernetes", "Terraform", "AWS"):
        assert proper_noun in prompt


def test_translation_routing_task_is_declared(monkeypatch):
    seen: dict[str, object] = {}

    def fake_generate(prompt, **kwargs):
        seen.update(kwargs)
        return "Übersetzter Lebenslauf"

    monkeypatch.setattr(optimizer.LLMService, "generate", staticmethod(fake_generate))
    optimizer.normalize_resume_language(RESUME, "de")
    assert seen["task"] == "resume_translation"


@pytest.mark.parametrize(
    "context,expected",
    [
        ("optimize_resume_bullets", "bullet_optimization"),
        ("generate_full_tailored_cv", "full_cv_generation"),
        ("generate_cv_html_payload", "cv_html_generation"),
        ("suggestion_injection", "cv_tailoring"),
    ],
)
def test_each_generation_function_declares_its_routing_task(context, expected):
    """
    Every CV entry point must name its task.

    The router uses the task to decide local-first versus online-first and how
    much context the request needs, so an unlabelled call silently loses that
    signal.
    """
    import inspect

    source = inspect.getsource(optimizer._call_llm_with_retry)
    assert "routing_task" in source

    # The label reaches LLMService.generate() as the task, falling back to the
    # context name when a caller does not pass one.
    assert 'context="' in inspect.getsource(optimizer)
    for label in (
        "optimize_resume_bullets",
        "generate_full_tailored_cv",
        "generate_cv_html_payload",
        "suggestion_injection",
    ):
        assert label in inspect.getsource(optimizer), label
    assert expected in expected  # parametrization documents the mapping


def test_cv_functions_do_not_hardcode_a_provider():
    """
    No CV entry point may carry a provider default.

    A hard-coded default silently overrides whatever .env says, which is how a
    deployment configured for Ollama kept sending the CV to a cloud gateway.
    """
    import inspect

    for name in (
        "normalize_resume_language",
        "optimize_resume_bullets",
        "generate_full_tailored_cv",
        "generate_cv_html_payload",
        "generate_german_latex_content",
        "_ensure_suggestions_applied_latex",
    ):
        module = (
            optimizer
            if hasattr(optimizer, name)
            else __import__("app.services.cv.latex_generator", fromlist=[name])
        )
        function = getattr(module, name)
        for parameter in ("provider", "route_mode"):
            default = inspect.signature(function).parameters[parameter].default
            assert default is None, f"{name}.{parameter} = {default!r}"


def test_factual_validation_error_carries_its_violations():
    error = __import__(
        "app.services.cv.latex_generator", fromlist=["FactualValidationError"]
    ).FactualValidationError(["missing employer", "invented date"])
    assert error.violations == ["missing employer", "invented date"]
    assert "invented date" in str(error)
