"""
Documentation that has to stay true.

A configuration file and a README are the two things a user trusts before
running anything. Both drift silently: a variable added to the registry is
simply absent from `.env.example`, so a user configuring that provider has
nothing to copy, and no error -- just a provider that is never used.

These tests make the drift a test failure instead.

No network, no credentials, no LLM.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services.llm.registry import PROVIDER_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
README = ROOT / "README.md"

#: Variables the application reads that are not provider-specific.
SHARED_VARIABLES = (
    "LLM_MODE",
    "LLM_PROVIDER",
    "LLM_ROUTE_MODE",
    "LLM_FALLBACK_ENABLED",
    "LLM_MAX_PROVIDER_ATTEMPTS",
    "LLM_RETRIES",
    "LLM_RETRY_BACKOFF_SECONDS",
    "LLM_MAX_TOKENS",
    "LLM_REQUEST_TIMEOUT_SECONDS",
    "LLM_PROCESSING_LOG",
)


@pytest.fixture(scope="module")
def env_example() -> str:
    return ENV_EXAMPLE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


# ===========================================================================
# .env.example
# ===========================================================================


def test_env_example_exists_and_is_not_empty(env_example):
    assert env_example.strip(), ".env.example is empty"


@pytest.mark.parametrize("provider", sorted(PROVIDER_REGISTRY))
def test_every_provider_is_documented_in_env_example(env_example, provider):
    """
    A provider with no documented configuration is a provider nobody can use.

    The user has no template to copy and no error to act on: the provider is
    simply never selected.
    """
    spec = PROVIDER_REGISTRY[provider]

    variables = [
        spec.base_url_env,
        spec.model_env,
        spec.max_tokens_env,
        *spec.key_env,
        spec.timeout_env,
    ]

    missing = sorted(
        {
            name
            for name in variables
            # The shared timeout is documented once rather than per provider.
            if name and name != "LLM_REQUEST_TIMEOUT_SECONDS" and name not in env_example
        }
    )

    assert not missing, f"{provider}: {', '.join(missing)} " f"not documented in .env.example"


@pytest.mark.parametrize("variable", SHARED_VARIABLES)
def test_shared_variables_are_documented(env_example, variable):
    assert (
        variable in env_example
    ), f"{variable} is read by the application but absent from .env.example"


def test_no_secret_appears_in_env_example(env_example):
    """
    A real credential committed here would be a credential leaked.

    Placeholders and empty values are fine; anything with the shape of a key is
    not.
    """
    suspicious = (
        "sk-",
        "AIza",
        "gsk_",
        "xai-",
        "hf_",
        "ghp_",
    )

    for marker in suspicious:
        assert marker not in env_example, (
            f".env.example contains something shaped like a credential " f"({marker!r})"
        )

    # No line may assign a value that looks filled in where a key is expected.
    for line in env_example.split("\n"):
        match = re.match(r"\s*([A-Z_]*(?:API_KEY|TOKEN|SECRET|ORG_KEY))\s*=\s*(\S+)", line)
        if not match:
            continue
        name, value = match.groups()
        assert value in {
            "",
            "replace-with-a-long-random-secret",
            "your_experiential_org_key",
        }, f"{name} has a value in .env.example: {value!r}"


def test_env_example_states_that_model_ids_are_not_permanent(env_example):
    """
    The single most important caveat about free tiers.

    A comment that says a model is free is a claim that goes stale. The
    example has to say the identifiers are examples.
    """
    lowered = env_example.casefold()
    assert "not promises" in lowered or "change" in lowered
    assert "free" in lowered
    assert "permanently free" in lowered or "not a promise" in lowered


def test_env_example_explains_the_shim_window(env_example):
    """
    The most surprising property of the recommended configuration.

    A user who sets the documented `/v1` URL and a 2048 ceiling will otherwise
    have no idea why a long request skips the local provider.
    """
    assert "2048" in env_example
    assert "num_ctx" in env_example
    assert "api/generate" in env_example and "/v1" in env_example


def test_env_example_names_the_smoke_test(env_example):
    assert "scripts.llm_smoke" in env_example


def test_every_script_env_example_recommends_exists(env_example):
    """
    A recommended command has to run.

    ``.env.example`` points users at scripts to verify their setup. A reference
    to a script that has been deleted is worse than no reference: the user types
    it, gets a module-not-found traceback, and concludes their configuration is
    broken. This asserts the referenced modules are on disk, so removing a script
    without clearing its documentation is a test failure.
    """
    referenced = set(re.findall(r"scripts\.([A-Za-z0-9_]+)", env_example))

    assert referenced, "no script referenced; the fixture is not testing anything"

    missing = sorted(
        name for name in referenced if not (ROOT / "scripts" / f"{name}.py").is_file()
    )
    assert not missing, f".env.example recommends scripts that do not exist: {missing}"


# ===========================================================================
# README
# ===========================================================================


def test_readme_has_a_free_local_section(readme):
    """
    The section has to be findable from the table of contents.
    """
    assert "## LLM Providers" in readme
    assert "### Free / Local LLM Options" in readme
    assert "#free--local-llm-options" in readme, "the TOC has no anchor for the free/local section"


def test_readme_distinguishes_the_three_cost_tiers(readme):
    for heading in (
        "#### LOCAL FREE",
        "#### FREE-TIER API",
        "#### PAID API",
    ):
        assert heading in readme, f"missing cost tier: {heading}"


def test_readme_lists_every_registered_provider(readme):
    missing = [name for name in PROVIDER_REGISTRY if name.casefold() not in readme.casefold()]
    assert not missing, f"undocumented providers: {missing}"


def test_readme_documents_omniroute_as_a_gateway(readme):
    """
    A gateway, not a model, with an optional key.
    """
    assert "gateway, not a model" in readme.casefold()
    assert "OMNIROUTE_BASE_URL=http://localhost:20128/v1" in readme
    assert "OMNIROUTE_MODEL=auto" in readme
    assert "OMNIROUTE_MAX_TOKENS=2048" in readme
    assert "Authorization" in readme, "the README should say what happens when the key is blank"


def test_readme_does_not_claim_anything_is_permanently_free(readme):
    """
    The requirement, stated as a test.

    Free quotas and model catalogues change. A README sentence asserting that a
    model is free will be wrong eventually and nobody will update it.
    """
    for match in re.finditer(r"[^.]*\bpermanently free\b[^.]*\.", readme, re.I):
        sentence = match.group(0)
        assert (
            "not" in sentence.casefold() or "no " in sentence.casefold()
        ), f"the README asserts permanent availability: {sentence.strip()!r}"


def test_readme_documents_the_privacy_switch(readme):
    """
    The setting that actually guarantees a CV never leaves the machine.
    """
    assert "LLM_MODE=ollama" in readme
    assert "never leaves" in readme.casefold() or "never leaves your machine" in readme


def test_readme_documents_model_switching_without_code_changes(readme):
    """
    The headline requirement, shown as configuration.
    """
    for example in (
        "LLM_PROVIDER=ollama",
        "LLM_PROVIDER=omniroute",
        "LLM_PROVIDER=gemini",
        "OLLAMA_MODEL=qwen2.5:7b",
    ):
        assert example in readme, f"missing switching example: {example}"


def test_readme_documents_the_retry_and_fallback_contract(readme):
    assert "LLM_FALLBACK_ENABLED" in readme
    assert "LLM_MAX_PROVIDER_ATTEMPTS" in readme
    assert "LLM_RETRIES" in readme
    assert (
        "one attempt" in readme.casefold()
    ), "the README must state that LLM_RETRIES=0 means one total attempt"


def test_readme_documents_output_ceilings_per_provider(readme):
    for provider in ("ollama", "omniroute", "gemini", "groq", "openrouter"):
        assert f"{provider.upper()}_MAX_TOKENS" in readme, provider


def test_readme_documents_model_discovery_and_its_optionality(readme):
    assert "Model discovery" in readme
    assert "never required" in readme.casefold() or "optional" in readme.casefold()


def test_readme_documents_the_smoke_test_safety_rules(readme):
    """
    The rule that makes it acceptable to run: no CV, no secrets in the output.
    """
    assert "python -m scripts.llm_smoke" in readme
    assert "never" in readme.casefold()
    lowered = readme.casefold()
    assert "cv or a job description" in lowered or "your cv" in lowered


def test_readme_documents_layout_validation(readme):
    """
    A supported layout is one that was actually produced and checked.
    """
    lowered = readme.casefold()
    assert "content-validated" in lowered or "content validated" in lowered
    assert "structurally identical" in lowered


def test_readme_troubleshooting_names_the_empty_content_cause(readme):
    """
    The most-reported failure, with the actual cause and the actual fix.
    """
    lowered = readme.casefold()
    assert "empty content" in lowered
    assert "reasoning" in lowered
    assert "finish_reason" in lowered


# ===========================================================================
# The measured model status table
# ===========================================================================


def test_readme_has_a_measured_status_table(readme):
    """
    The table is a record of runs, so it has to look like one.

    A capability list and a measurement record are different things: the first
    says what might work, the second says what was tried and what happened. Only
    the second is useful to a reader deciding what to run.
    """
    assert "### Verified status" in readme
    assert "#### Local models" in readme
    assert "#### API providers" in readme
    assert "scripts.model_matrix" in readme


def test_status_table_keeps_the_three_states_apart(readme):
    """
    AVAILABLE, smoke tested and full pipeline are three different claims.

    A model whose listing answered has not been shown to work. A model that
    answered a smoke prompt has not been shown to produce a CV. Collapsing the
    columns would make the table claim more than it measured.
    """
    for column in ("AVAILABLE", "Smoke tested", "Full CV pipeline"):
        assert column in readme, f"missing column: {column}"


def test_no_status_row_leaves_a_test_column_blank(readme):
    """
    Every test cell says PASS, FAIL or NOT TESTED.

    A blank cell reads as "fine, just untidy". An explicit NOT TESTED reads as
    "nobody has checked this", which is the truth and is what stops a reader
    assuming otherwise.
    """
    start = readme.find("#### Local models")
    assert start > 0
    end = readme.find("#### API providers")
    assert end > start
    block = readme[start:end]

    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 6 or cells[0].startswith("---"):
            continue

        model = cells[0].strip("`")
        if model in {"Model", ""}:
            continue

        smoke = cells[3]
        pipeline = cells[4]

        for name, value in (("smoke", smoke), ("pipeline", pipeline)):
            assert value, f"{model}: the {name} column is blank"
            assert any(
                marker in value for marker in ("PASS", "FAIL", "NOT TESTED", "NOT INSTALLED")
            ), f"{model}: the {name} column says {value!r}"


def test_a_pipeline_pass_is_always_spelled_out(readme):
    """
    A pass is claimed only with the word PASS.

    Catches the most likely future edit: adding a row to the table without
    running the test, or softening a failure into something that reads like a
    success. ``FAIL`` and ``NOT TESTED`` are legitimate values -- the check is
    that the cell says exactly one of the four, not that it mentions a pass.
    """
    allowed = {"**PASS**", "PASS", "FAIL", "NOT TESTED", "NOT INSTALLED"}

    start = readme.find("#### Local models")
    end = readme.find("#### API providers")
    block = readme[start:end]

    checked = 0
    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 6 or cells[0].startswith("---"):
            continue
        model = cells[0].strip("`")
        if model in {"Model", ""}:
            continue

        assert cells[4] in allowed, (
            f"{model}: the pipeline column says {cells[4]!r}, which is not one "
            f"of {sorted(allowed)}"
        )
        checked += 1

    assert checked >= 5, f"only {checked} local model rows were checked"


def test_status_table_states_that_local_models_passed(readme):
    """
    The headline result, stated rather than left for the reader to tally.

    Local models completing the real pipeline is the single most useful fact in
    the table.
    """
    assert "completed the entire pipeline" in readme.casefold()
    assert "qwen3:8b" in readme


def test_status_table_does_not_claim_anything_is_permanently_free(readme):
    """
    The requirement, stated as a test.

    Free quotas and model catalogues change. A sentence claiming a model is
    free will be wrong eventually and nobody will update it, because nothing
    fails when it does.
    """
    for match in re.finditer(r"[^.\n]*\bpermanently free\b[^.\n]*\.", readme, re.I):
        sentence = match.group(0)
        assert (
            "not" in sentence.casefold()
        ), f"the README asserts permanent availability: {sentence.strip()!r}"


def test_status_table_marks_untested_rows_explicitly(readme):
    start = readme.find("### Verified status")
    assert start > 0
    block = readme[start : start + 12_000]
    assert "NOT TESTED" in block
    assert "NOT CONFIGURED" in block


def test_readme_documents_the_dashboard_workflow(readme):
    """
    The dashboard is one workflow page, and the README says which stages it has.

    The expected stage names are read from the application rather than written
    out here, so renaming a stage fails this test instead of quietly leaving the
    README describing a workflow the app no longer has. That is the drift this
    file exists to catch: a real page name and a documented one drifting apart
    with no error anywhere.
    """
    from app.dashboard.views.workflow_page import STAGES

    assert "### One-Page CV Workflow" in readme

    for _number, label, _key, _glyph in STAGES:
        assert label in readme, f"undocumented workflow stage: {label}"

    # The other navigation entry, and the configuration pages that are
    # deliberately not stages of the workflow.
    for page in ("Overview", "layout", "diagnostics"):
        assert page.casefold() in readme.casefold(), page


def test_readme_does_not_describe_the_old_multi_page_workflow(readme):
    """
    The six-page description is gone from the README, not just supplemented.

    The old text could survive an edit that adds the new section without
    removing the old one, leaving the README describing two different dashboards
    at once. This asserts the specific thing that has changed rather than the
    general claim, so it fails on the stale sentence instead of on a rewording.
    """
    for stale in (
        "### Guided Dashboard",
        "Overview → Resume",
        "Continue to job description",
    ):
        assert stale not in readme, f"README still describes the old page flow: {stale!r}"


def test_readme_explains_the_dashboard_status_vocabulary(readme):
    """
    CONFIGURED, AVAILABLE and PASSED are documented, not just used.
    """
    lowered = readme.casefold()
    for verdict in ("configured", "available", "passed"):
        assert verdict in lowered, verdict
    assert "never presented as a success" in lowered or ("not presented as a success" in lowered)
