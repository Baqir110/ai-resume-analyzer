"""
Central LLM routing: modes, tasks, fallback, classification, context budget.

Every test here is a unit test. Provider calls are mocked, so no API key and no
network access is required. The tests that need a real local model live in
``test_ollama_qwen3_generation.py`` and skip when Ollama is absent.
"""

import pytest

from app.services.llm import provider as provider_module
from app.services.llm.provider import (
    KNOWN_TASKS,
    LLM_ERROR_AUTH,
    LLM_ERROR_CONNECTION,
    LLM_ERROR_CONTEXT_TOO_LARGE,
    LLM_ERROR_CREDITS,
    LLM_ERROR_MALFORMED,
    LLM_ERROR_MODEL_UNAVAILABLE,
    LLM_ERROR_RATE_LIMIT,
    LLM_ERROR_SERVER,
    LLM_ERROR_TIMEOUT,
    LLMService,
    classify_llm_error,
    normalize_task,
    task_policy,
)

ALL_ENV = (
    "LLM_MODE",
    "LLM_PROVIDER",
    "LLM_ROUTE_MODE",
    "LLM_PROVIDER_CHAIN",
    "LLM_RETRIES",
    "ONLINE_PRIMARY_PROVIDER",
    "ONLINE_FALLBACK_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_MAX_CONTEXT_TOKENS",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "EXPLABS_API_KEY",
    "EXPERIENTIAL_ORG_KEY",
    "OPENAI_BASE_URL",
)


class FakeStatus(Exception):
    """Stand-in for an SDK HTTP error that carries a status code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _reset_exhausted() -> None:
    """Clear the cross-request provider cooldown map.

    ``_state_lock`` is a plain ``threading.Lock``, so nothing that acquires it
    itself may run inside this block.
    """
    with provider_module.LLMService._state_lock:
        provider_module.LLMService._exhausted_models.clear()


@pytest.fixture
def clean_env(monkeypatch):
    """Start from a known-empty LLM configuration."""
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    # Reset the cross-request exhaustion cache so one test cannot suppress a
    # candidate in the next.
    _reset_exhausted()
    yield monkeypatch
    _reset_exhausted()


@pytest.fixture
def local_only(clean_env):
    """Ollama configured, nothing else."""
    clean_env.setenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    clean_env.setenv("OLLAMA_MODEL", "qwen3:8b")
    return clean_env


@pytest.fixture
def hybrid(clean_env):
    """Ollama plus two online providers, matching the recommended setup."""
    clean_env.setenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    clean_env.setenv("OLLAMA_MODEL", "qwen3:8b")
    clean_env.setenv("GEMINI_API_KEY", "test-gemini-key")
    clean_env.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    clean_env.setenv("ONLINE_PRIMARY_PROVIDER", "gemini")
    clean_env.setenv("ONLINE_FALLBACK_PROVIDER", "openrouter")
    return clean_env


def _candidates(**kwargs):
    return LLMService._build_candidates(**kwargs)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_mode_defaults_to_auto(clean_env):
    assert LLMService._default_mode() == "auto"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("auto", "auto"),
        ("automatic", "auto"),
        ("ollama", "ollama"),
        ("local", "ollama"),
        ("offline", "ollama"),
        ("private", "ollama"),
        ("online", "online"),
        ("cloud", "online"),
    ],
)
def test_llm_mode_values(clean_env, value, expected):
    clean_env.setenv("LLM_MODE", value)
    assert LLMService._default_mode() == expected


def test_unknown_mode_falls_back_to_auto(clean_env):
    clean_env.setenv("LLM_MODE", "nonsense")
    assert LLMService._default_mode() == "auto"


def test_llm_mode_takes_precedence_over_legacy_route_mode(clean_env):
    clean_env.setenv("LLM_MODE", "auto")
    clean_env.setenv("LLM_ROUTE_MODE", "direct")
    clean_env.setenv("LLM_PROVIDER", "ollama")
    assert LLMService._default_mode() == "auto"


def test_legacy_direct_route_mode_maps_to_local(clean_env):
    clean_env.setenv("LLM_ROUTE_MODE", "direct")
    clean_env.setenv("LLM_PROVIDER", "ollama")
    assert LLMService._default_mode() == "ollama"


def test_legacy_automatic_route_mode_maps_to_auto(clean_env):
    clean_env.setenv("LLM_ROUTE_MODE", "automatic")
    assert LLMService._default_mode() == "auto"


def test_legacy_gateway_route_mode_still_maps_to_experiential(clean_env):
    clean_env.setenv("LLM_ROUTE_MODE", "gateway")
    assert LLMService._default_mode() == "auto"


def test_mode_is_resolved_in_exactly_one_place(clean_env):
    """
    There must be a single mode -> route mapping.

    A second implementation once lived in ``_default_route_mode`` and folded
    ``online`` into ``automatic``, which includes Ollama in the candidate list.
    That silently broke the one guarantee the mode exists for: a CV would have
    been sent to a cloud provider in the mode that promises it will not be.
    """
    assert not hasattr(LLMService, "_default_llm_mode")

    clean_env.setenv("LLM_MODE", "online")
    assert LLMService._default_route_mode() == "automatic"
    assert LLMService._mode_to_route(LLMService._default_mode()) == "online"

    clean_env.setenv("LLM_MODE", "ollama")
    assert LLMService._mode_to_route(LLMService._default_mode()) == "local"

    clean_env.setenv("LLM_MODE", "auto")
    assert LLMService._mode_to_route(LLMService._default_mode()) == "automatic"


@pytest.mark.parametrize(
    "mode,expected",
    [("auto", "automatic"), ("ollama", "local"), ("online", "online")],
)
def test_every_mode_maps_to_a_route_that_keeps_its_promise(clean_env, mode, expected):
    clean_env.setenv("LLM_MODE", mode)
    clean_env.setenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    clean_env.setenv("OLLAMA_MODEL", "qwen3:8b")
    clean_env.setenv("GEMINI_API_KEY", "k")

    route = LLMService._mode_to_route(LLMService._default_mode())
    assert route == expected

    providers = {
        p
        for p, _ in LLMService._build_candidates(
            provider=None,
            model=None,
            route_mode=route,
            task="full_cv_generation",
            prompt="a prompt",
        )
    }
    if mode == "ollama":
        assert providers == {"ollama"}
    if mode == "online":
        assert "ollama" not in providers
        assert "gemini" in providers
    if mode == "auto":
        assert providers == {"ollama", "gemini"}


def test_route_info_reports_the_active_configuration(hybrid):
    info = LLMService.route_info()
    assert info["mode"] == "auto"
    assert info["ollama_model"] == "qwen3:8b"
    assert info["allows_cloud_fallback"] is True
    assert info["online_primary"] == "gemini"
    assert "cv_tailoring" in info["tasks"]


def test_route_info_never_contains_a_secret(hybrid):
    rendered = repr(LLMService.route_info())
    assert "test-gemini-key" not in rendered
    assert "test-openrouter-key" not in rendered


# ---------------------------------------------------------------------------
# LLM_MODE=ollama -- never leaves the machine
# ---------------------------------------------------------------------------


def test_ollama_mode_yields_exactly_one_local_candidate(hybrid):
    hybrid.setenv("LLM_MODE", "ollama")
    candidates = _candidates(provider=None, model=None, route_mode="local", task="cv_tailoring")
    assert candidates == [("ollama", "qwen3:8b")]


def test_ollama_mode_uses_the_configured_model(local_only):
    # Any model Ollama serves must work through configuration alone; nothing in
    # the router may hard-code qwen3:8b.
    local_only.setenv("OLLAMA_MODEL", "llama3.3:70b")
    assert _candidates(provider=None, model=None, route_mode="local") == [
        ("ollama", "llama3.3:70b")
    ]


@pytest.mark.parametrize(
    "model",
    ["qwen3:8b", "glm-4.7-flash:latest", "qwen3.6:latest", "llama3.3:70b"],
)
def test_every_ollama_model_can_be_configured(local_only, model):
    local_only.setenv("OLLAMA_MODEL", model)
    assert LLMService.get_default_model("ollama") == model
    assert _candidates(provider=None, model=None, route_mode="local") == [("ollama", model)]


def test_ollama_mode_reports_a_clear_error_when_the_server_is_down(local_only, monkeypatch):
    """
    ``_get_ollama_base_url`` always yields a usable default, so "not configured"
    is not reachable for Ollama. The real failure is an unreachable server, and
    it must surface as a local error rather than becoming a cloud request.
    """
    local_only.setenv("LLM_MODE", "ollama")

    def boom(**kwargs):
        raise FakeStatus("connection refused", 500)

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(boom))
    with pytest.raises(RuntimeError, match="LLM_MODE=ollama"):
        LLMService.generate(prompt="hello", task="cv_tailoring")


def test_ollama_mode_never_offers_an_online_provider(hybrid):
    hybrid.setenv("LLM_MODE", "ollama")
    for task in KNOWN_TASKS:
        candidates = _candidates(provider=None, model=None, route_mode="local", task=task)
        assert {p for p, _ in candidates} == {"ollama"}, task


def test_ollama_mode_ignores_the_context_budget(local_only):
    # A prompt far too large for the local window is still attempted locally,
    # because ollama mode forbids falling back and silently going online would
    # violate the contract. The size warning is logged instead.
    local_only.setenv("LLM_MODE", "ollama")
    candidates = _candidates(
        provider=None,
        model=None,
        route_mode="local",
        task="full_cv_generation",
        prompt="x" * 400_000,
    )
    assert candidates == [("ollama", "qwen3:8b")]


def test_ollama_mode_rejects_an_explicit_online_provider(hybrid):
    hybrid.setenv("LLM_MODE", "ollama")
    with pytest.raises(ValueError, match="cannot be combined"):
        _candidates(provider="gemini", model=None, route_mode="local")


def test_ollama_mode_does_not_fall_back_on_failure(hybrid, monkeypatch):
    """A local failure must surface, not become a cloud request."""
    hybrid.setenv("LLM_MODE", "ollama")

    def boom(**kwargs):
        raise FakeStatus("connection refused", 500)

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(boom))

    with pytest.raises(RuntimeError) as error:
        LLMService.generate(prompt="hello", task="cv_tailoring")

    message = str(error.value)
    assert "LLM_MODE=ollama" in message
    assert "forbids falling back" in message
    # The online providers are configured, so if the guard were missing they
    # would have been tried and named here.
    assert "gemini" not in message.lower()
    assert "openrouter" not in message.lower()


# ---------------------------------------------------------------------------
# LLM_MODE=online -- never touches the local machine
# ---------------------------------------------------------------------------


def test_online_mode_excludes_ollama(hybrid):
    hybrid.setenv("LLM_MODE", "online")
    candidates = _candidates(provider=None, model=None, route_mode="online", task="ats_analysis")
    providers = [p for p, _ in candidates]
    assert "ollama" not in providers
    assert providers[0] == "gemini"
    assert "openrouter" in providers


def test_online_mode_raises_when_nothing_is_configured(local_only):
    local_only.setenv("LLM_MODE", "online")
    with pytest.raises(RuntimeError, match="no online provider"):
        _candidates(provider=None, model=None, route_mode="online")


def test_online_chain_respects_the_configured_order(clean_env):
    clean_env.setenv("GROQ_API_KEY", "k")
    clean_env.setenv("OPENAI_API_KEY", "k")
    clean_env.setenv("ONLINE_PRIMARY_PROVIDER", "openai")
    clean_env.setenv("ONLINE_FALLBACK_PROVIDER", "groq")
    assert LLMService._online_provider_chain()[:2] == ["openai", "groq"]


def test_online_chain_skips_unconfigured_names(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "k")
    clean_env.setenv("ONLINE_PRIMARY_PROVIDER", "gemini,claude")
    chain = LLMService._online_provider_chain()
    assert chain[0] == "gemini"
    assert "claude" not in chain


# ---------------------------------------------------------------------------
# LLM_MODE=auto -- local first, cloud fallback
# ---------------------------------------------------------------------------


def test_auto_prefers_ollama(hybrid):
    hybrid.setenv("LLM_MODE", "auto")
    candidates = _candidates(provider=None, model=None, route_mode="automatic", task="cv_tailoring")
    assert candidates[0] == ("ollama", "qwen3:8b")
    assert {p for p, _ in candidates} == {"ollama", "gemini", "openrouter"}


def test_auto_keeps_ollama_first_and_keeps_the_online_chain(clean_env):
    """
    With only an online provider configured, auto still offers Ollama first.

    ``_get_ollama_base_url`` always resolves to a localhost default, so Ollama
    counts as configured whether or not the variable is set. That is
    deliberate: it is what makes the existing local setup work with no
    configuration at all. The cost is one fast connection-refused before the
    cloud chain is used, which is why ``LLM_MODE=online`` exists for callers
    that need the local machine never to be contacted.
    """
    clean_env.setenv("GEMINI_API_KEY", "k")
    clean_env.setenv("LLM_MODE", "auto")
    candidates = _candidates(provider=None, model=None, route_mode="automatic", task="cv_tailoring")
    assert candidates[0][0] == "ollama"
    assert ("gemini", "gemini-2.5-flash") in candidates


def test_auto_omits_ollama_when_it_is_not_configured(clean_env, monkeypatch):
    real = LLMService._provider_is_configured.__func__

    monkeypatch.setattr(
        LLMService,
        "_provider_is_configured",
        classmethod(lambda cls, provider: (False if provider == "ollama" else real(cls, provider))),
    )
    clean_env.setenv("GEMINI_API_KEY", "k")
    clean_env.setenv("LLM_MODE", "auto")
    candidates = _candidates(provider=None, model=None, route_mode="automatic", task="cv_tailoring")
    assert candidates == [("gemini", "gemini-2.5-flash")]


def test_auto_skips_ollama_for_an_online_first_task(hybrid):
    hybrid.setenv("LLM_MODE", "auto")
    candidates = _candidates(
        provider=None, model=None, route_mode="automatic", task="browser_agent"
    )
    assert candidates[0][0] != "ollama"


def test_auto_keeps_ollama_first_for_full_cv_generation(hybrid):
    hybrid.setenv("LLM_MODE", "auto")
    candidates = _candidates(
        provider=None, model=None, route_mode="automatic", task="full_cv_generation"
    )
    assert candidates[0][0] == "ollama"


def test_auto_skips_ollama_when_the_prompt_will_not_fit(hybrid):
    hybrid.setenv("LLM_MODE", "auto")
    hybrid.setenv("OLLAMA_MAX_CONTEXT_TOKENS", "4096")
    candidates = _candidates(
        provider=None,
        model=None,
        route_mode="automatic",
        task="full_cv_generation",
        prompt="x" * 200_000,
    )
    assert "ollama" not in [p for p, _ in candidates]
    assert "gemini" in [p for p, _ in candidates]


def test_auto_uses_ollama_when_the_prompt_fits(hybrid):
    hybrid.setenv("LLM_MODE", "auto")
    candidates = _candidates(
        provider=None,
        model=None,
        route_mode="automatic",
        task="bullet_optimization",
        prompt="Optimize these bullets for the role.",
    )
    assert candidates[0][0] == "ollama"


def test_explicit_provider_wins_over_the_task_preference(hybrid):
    hybrid.setenv("LLM_MODE", "auto")
    candidates = _candidates(
        provider="gemini",
        model=None,
        route_mode="automatic",
        task="browser_agent",
    )
    assert candidates[0][0] == "gemini"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def test_every_documented_task_is_known():
    for task in (
        "ats_analysis",
        "resume_translation",
        "bullet_optimization",
        "cv_tailoring",
        "full_cv_generation",
        "cv_html_generation",
        "latex_validation",
        "browser_agent",
        "generic",
    ):
        assert task in KNOWN_TASKS, task


def test_latex_generation_is_accepted_as_a_task():
    # Listed alongside the others in the routing contract, and it maps onto the
    # full-document policy because it produces a whole CV body.
    assert normalize_task("latex_generation") == "full_cv_generation"
    assert task_policy("latex_generation").max_output_tokens == 4096


def test_task_aliases_resolve():
    assert normalize_task("ats") == "ats_analysis"
    assert normalize_task("translate") == "resume_translation"
    assert normalize_task("bullets") == "bullet_optimization"
    assert normalize_task("full-cv") == "full_cv_generation"
    assert normalize_task("BULLET_OPTIMIZATION") == "bullet_optimization"


def test_unknown_task_degrades_to_generic_rather_than_failing():
    # A typo must never widen the provider set.
    assert normalize_task("cv_taloring_typo") == "generic"
    assert normalize_task(None) == "generic"
    assert normalize_task(123) == "generic"
    assert task_policy("cv_taloring_typo") is task_policy("generic")


def test_routine_tasks_are_local_first():
    for task in (
        "ats_analysis",
        "resume_translation",
        "bullet_optimization",
        "cv_tailoring",
    ):
        assert task_policy(task).local_first is True, task
        assert task_policy(task).online_first is False, task


def test_browser_agent_stays_online(clean_env):
    assert task_policy("browser_agent").online_first is True


def test_latex_generation_is_a_full_document_task():
    assert normalize_task("latex_generation") == "full_cv_generation"
    assert task_policy("latex_generation").max_output_tokens == 4096


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,status,expected",
    [
        ("Request timed out", 408, LLM_ERROR_TIMEOUT),
        ("Connection refused", None, LLM_ERROR_CONNECTION),
        ("401 Unauthorized", 401, LLM_ERROR_AUTH),
        ("429 rate limit exceeded", 429, LLM_ERROR_RATE_LIMIT),
        ("429 insufficient_credits", 429, LLM_ERROR_CREDITS),
        ("402 Payment Required", 402, LLM_ERROR_CREDITS),
        ("model not found", 404, LLM_ERROR_MODEL_UNAVAILABLE),
        ("maximum context length is 8192", 400, LLM_ERROR_CONTEXT_TOO_LARGE),
        ("returned empty content", None, LLM_ERROR_MALFORMED),
        ("503 service unavailable", 503, LLM_ERROR_SERVER),
        ("ollama model does not exist", None, LLM_ERROR_MODEL_UNAVAILABLE),
    ],
)
def test_error_classification(message, status, expected):
    assert classify_llm_error(FakeStatus(message, status)) == expected


def test_a_retired_model_id_is_a_model_problem_not_a_malformed_request():
    """
    The regression. OpenRouter reports a model id it no longer serves as a bad
    *request*: 400 "... is not a valid model ID". The generic 400 mapping read
    that as ``malformed_response``, which blamed the prompt and hid the one
    condition a different model on the same provider fixes.
    """
    retired = FakeStatus(
        "Error code: 400 - {'error': {'message': "
        "'nvidia/nemotron-3-ultra:free is not a valid model ID', 'code': 400}}",
        400,
    )

    assert classify_llm_error(retired, "openrouter") == LLM_ERROR_MODEL_UNAVAILABLE
    # And it is not a reason to re-send the identical request: the id is the
    # problem, so the same prompt fails identically every time.
    assert provider_module._is_retryable_error(retired, "openrouter") is False


def test_a_400_that_merely_mentions_a_host_is_not_a_model_problem():
    """
    The other direction, so the override above cannot run away: "no such host"
    is one of the ``model_unavailable`` markers, and on a 5xx it is a DNS
    failure. The override is therefore scoped to 400 and 422.
    """
    assert (
        classify_llm_error(FakeStatus("bad gateway: no such host", 502), "openrouter")
        == LLM_ERROR_SERVER
    )


def test_insufficient_credits_is_not_confused_with_a_rate_limit():
    # The reported 429 insufficient_credits loop: the two share a status code
    # and mean opposite things for retrying.
    credits = FakeStatus("429 insufficient_credits", 429)
    rate_limit = FakeStatus("429 rate limit exceeded", 429)
    assert classify_llm_error(credits) == LLM_ERROR_CREDITS
    assert classify_llm_error(rate_limit) == LLM_ERROR_RATE_LIMIT
    assert provider_module._is_retryable_error(credits, "experiential") is False
    assert provider_module._is_retryable_error(rate_limit, "gemini") is True


def test_classification_never_leaks_the_api_key(hybrid):
    secret = "sk-do-not-log-this-value"
    hybrid.setenv("GEMINI_API_KEY", secret)
    category = classify_llm_error(FakeStatus(f"401 Unauthorized: api_key={secret}", 401))
    assert category == LLM_ERROR_AUTH
    assert secret not in repr(category)


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


def test_direct_route_falls_back_to_the_providers_other_models(clean_env, monkeypatch):
    """
    The regression, and the cause of every failed CV generation in one run.

    ``route_mode="direct"`` is what the workflow page sends by default. It built
    a *single* candidate from the provider's default model, so when that id was
    retired -- ``nvidia/nemotron-3-ultra:free``, which OpenRouter no longer
    serves -- the request answered 400 and the router had nowhere to go, even
    though FALLBACK_MODEL_1 held a working id the whole time.

    "Direct" pins the provider, not one model, so the provider's own ordered
    list is what should be walked.
    """
    clean_env.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    clean_env.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra:free")
    clean_env.setenv("FALLBACK_MODEL_1", "openrouter/free")

    candidates = _candidates(
        provider="openrouter",
        model=None,
        route_mode="direct",
        task="full_cv_generation",
    )

    assert ("openrouter", "nvidia/nemotron-3-ultra:free") in candidates
    assert ("openrouter", "openrouter/free") in candidates
    assert all(provider == "openrouter" for provider, _ in candidates), (
        "direct mode must not widen to another provider"
    )

    # An explicit model still pins exactly one candidate.
    pinned = _candidates(
        provider="openrouter",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        route_mode="direct",
        task="full_cv_generation",
    )
    assert pinned == [("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")]


def test_direct_route_recovers_when_the_first_model_id_is_retired(
    hybrid, monkeypatch
):
    """The end-to-end shape of the fix: one bad id, then a working one."""
    hybrid.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra:free")
    hybrid.setenv("FALLBACK_MODEL_1", "openrouter/free")
    seen: list[str] = []

    def fake_execute(**kwargs):
        seen.append(kwargs["model"])
        if "nemotron-3-ultra" in kwargs["model"]:
            raise FakeStatus(
                "'nvidia/nemotron-3-ultra:free is not a valid model ID'", 400
            )
        return "generated content"

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake_execute))

    result = LLMService.generate(
        prompt="hello",
        provider="openrouter",
        route_mode="direct",
        task="full_cv_generation",
    )

    assert result == "generated content"
    assert seen == ["nvidia/nemotron-3-ultra:free", "openrouter/free"]


def test_auto_falls_over_to_the_next_provider(hybrid, monkeypatch):
    hybrid.setenv("LLM_MODE", "auto")
    seen: list[str] = []

    def fake_execute(**kwargs):
        seen.append(kwargs["provider"])
        if kwargs["provider"] == "ollama":
            raise FakeStatus("connection refused", 500)
        return "generated content"

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake_execute))

    assert LLMService.generate(prompt="hello", task="cv_tailoring") == ("generated content")
    assert seen == ["ollama", "gemini"]


def test_all_providers_failing_reports_every_attempt(hybrid, monkeypatch):
    hybrid.setenv("LLM_MODE", "auto")

    def fake_execute(**kwargs):
        raise FakeStatus(f"{kwargs['provider']} exploded", 500)

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake_execute))

    with pytest.raises(RuntimeError) as error:
        LLMService.generate(prompt="hello", task="cv_tailoring")

    message = str(error.value)
    assert "task=cv_tailoring" in message
    assert "ollama" in message
    assert "gemini" in message


def test_switching_provider_mid_chain_is_possible_without_retry_storms(hybrid, monkeypatch):
    """A dead provider is cooled down, so the next request skips it."""
    hybrid.setenv("LLM_MODE", "auto")
    monkeypatch.setattr(
        LLMService,
        "_execute_single_provider",
        staticmethod(
            lambda **kwargs: (
                (_ for _ in ()).throw(FakeStatus("connection refused", 500))
                if kwargs["provider"] == "ollama"
                else "ok"
            )
        ),
    )

    assert LLMService.generate(prompt="one", task="cv_tailoring") == "ok"
    assert LLMService._is_exhausted("ollama", "qwen3:8b") is True

    # The next request must not pay for the dead provider again. `_state_lock`
    # is a plain Lock, so the latency reset has to happen outside any other
    # holder of it -- `_record_execution_time` acquires it itself.
    LLMService._record_execution_time("ollama", "qwen3:8b", 10.0)
    candidates = LLMService._build_candidates(
        provider=None,
        model=None,
        route_mode="automatic",
        task="cv_tailoring",
    )
    ordered = LLMService._select_candidates(candidates)
    assert "ollama" not in [p for p, _ in ordered]


def test_malformed_output_is_not_retried_against_the_same_provider(hybrid, monkeypatch):
    hybrid.setenv("LLM_MODE", "auto")
    attempts: list[str] = []

    def fake_execute(**kwargs):
        attempts.append(kwargs["provider"])
        if kwargs["provider"] == "ollama":
            raise RuntimeError("ollama returned empty content.")
        return "fine"

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake_execute))
    assert LLMService.generate(prompt="hello", task="bullet_optimization") == "fine"
    assert attempts.count("ollama") == 1


def test_context_too_large_moves_on_instead_of_retrying(hybrid, monkeypatch):
    hybrid.setenv("LLM_MODE", "auto")
    attempts: list[str] = []

    def fake_execute(**kwargs):
        attempts.append(kwargs["provider"])
        if kwargs["provider"] == "ollama":
            raise FakeStatus("maximum context length is 8192 tokens", 400)
        return "fine"

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake_execute))
    assert LLMService.generate(prompt="hello", task="cv_tailoring") == "fine"
    assert attempts.count("ollama") == 1


# ---------------------------------------------------------------------------
# Interface compatibility
# ---------------------------------------------------------------------------


def test_generate_accepts_task_and_still_rejects_unknown_options(hybrid):
    with pytest.raises(TypeError, match="Unexpected LLM options"):
        LLMService.generate(prompt="x", nonsense=True)


def test_call_llm_forwards_task(hybrid, monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(LLMService, "generate", staticmethod(fake_generate))
    LLMService.call_llm("prompt", task="cv_tailoring")
    assert captured["task"] == "cv_tailoring"


def test_empty_prompt_is_rejected(hybrid):
    with pytest.raises(ValueError, match="cannot be empty"):
        LLMService.generate(prompt="   ")


def test_ollama_health_never_raises_when_unreachable(local_only, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(provider_module.requests, "get", boom)
    health = LLMService.ollama_health()
    assert health["configured"] is True
    assert health["reachable"] is False
    assert "not reachable" in health["detail"]


def test_ollama_health_names_a_missing_model(local_only, monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"models": [{"name": "llama3.2:3b"}]}

    monkeypatch.setattr(provider_module.requests, "get", lambda *a, **k: Response())
    health = LLMService.ollama_health()
    assert health["reachable"] is True
    assert health["model_available"] is False
    assert "ollama pull qwen3:8b" in health["detail"]
    assert "llama3.2:3b" in health["detail"]


def test_ollama_health_accepts_a_base_name_match(local_only, monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"models": [{"name": "qwen3:8b-instruct"}]}

    monkeypatch.setattr(provider_module.requests, "get", lambda *a, **k: Response())
    assert LLMService.ollama_health()["model_available"] is True


def test_provider_status_covers_every_supported_provider(hybrid):
    status = LLMService.provider_status()
    assert set(status) == set(provider_module.SUPPORTED_PROVIDERS)
    assert status["ollama"]["configured"] is True
    assert status["gemini"]["configured"] is True
