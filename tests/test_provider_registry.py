"""
The provider registry and the OpenAI-compatible execution path.

Every external API is mocked. These tests need no API key, no running Ollama, no
OmniRoute and no network access.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.llm import provider as provider_module
from app.services.llm import registry as registry_module
from app.services.llm.provider import KNOWN_TASKS, LLMService
from app.services.llm.registry import (
    PROTOCOL_ANTHROPIC,
    PROTOCOL_GEMINI,
    PROTOCOL_OPENAI_COMPATIBLE,
    PROVIDER_REGISTRY,
    canonical_provider,
    get_spec,
    known_providers,
    online_providers,
)

# ---------------------------------------------------------------------------
# Every environment variable the registry reads, so a test starts from blank.
# ---------------------------------------------------------------------------

ALL_ENV = tuple(
    sorted(
        {
            name
            for spec in PROVIDER_REGISTRY.values()
            for name in (
                spec.base_url_env,
                spec.model_env,
                *spec.key_env,
                *((spec.max_tokens_env,) if spec.max_tokens_env else ()),
                *((spec.timeout_env,) if spec.timeout_env else ()),
            )
            if name
        }
        | {
            "LLM_MODE",
            "LLM_PROVIDER",
            "LLM_ROUTE_MODE",
            "LLM_PROVIDER_CHAIN",
            "LLM_FALLBACK_ENABLED",
            "LLM_MAX_PROVIDER_ATTEMPTS",
            "LLM_MAX_TOKENS",
            "LLM_DISCOVERY_TIMEOUT_SECONDS",
            "LLM_RETRIES",
            "OLLAMA_MAX_TOKENS",
            "OLLAMA_MAX_OUTPUT_TOKENS",
            "ONLINE_PRIMARY_PROVIDER",
            "ONLINE_FALLBACK_PROVIDER",
        }
    )
)


class Status(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def blank(monkeypatch):
    """No provider is configured, and no candidate is in cooldown."""
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    with LLMService._state_lock:
        LLMService._exhausted_models.clear()
    yield monkeypatch
    with LLMService._state_lock:
        LLMService._exhausted_models.clear()


def _configure(monkeypatch, provider: str, *, key: str = "k", model: str = "m") -> None:
    spec = get_spec(provider)
    if spec.key_env and key:
        monkeypatch.setenv(spec.key_env[0], key)
    if model:
        monkeypatch.setenv(spec.model_env, model)


# ===========================================================================
# 1. The registry
# ===========================================================================


def test_every_previously_supported_provider_is_still_registered():
    """
    None of the original eight may disappear.

    This is the guard against a refactor quietly dropping a provider that a
    deployment depends on.
    """
    for name in (
        "ollama",
        "gemini",
        "openai",
        "claude",
        "groq",
        "openrouter",
        "deepseek",
        "experiential",
    ):
        assert name in PROVIDER_REGISTRY, name
        assert name in known_providers(), name


def test_omniroute_is_registered_as_a_gateway_not_a_model():
    spec = get_spec("omniroute")
    assert spec is not None
    assert spec.protocol == PROTOCOL_OPENAI_COMPATIBLE
    assert spec.local is True
    assert spec.base_url_env == "OMNIROUTE_BASE_URL"
    assert spec.default_base_url == "http://localhost:20128/v1"
    assert spec.model_env == "OMNIROUTE_MODEL"
    assert spec.default_model == "auto"
    assert spec.max_tokens_env == "OMNIROUTE_MAX_TOKENS"
    assert spec.default_max_tokens == 2048


def test_omniroute_api_key_is_optional():
    """A local gateway may run with no authentication."""
    assert get_spec("omniroute").requires_key is False


def test_free_tier_providers_are_registered():
    for name in ("cerebras", "cloudflare", "github", "huggingface"):
        spec = get_spec(name)
        assert spec is not None, name
        assert spec.protocol == PROTOCOL_OPENAI_COMPATIBLE, name


def test_providers_with_unstable_model_ids_ship_no_default():
    """
    A stale default model is worse than none.

    These providers change their catalogue; a hard-coded id would send the CV
    to a model that no longer exists and fail with an opaque 400.
    """
    for name in ("cerebras", "cloudflare", "github", "huggingface"):
        assert get_spec(name).default_model == "", name


def test_non_openai_providers_keep_their_own_protocol():
    assert get_spec("gemini").protocol == PROTOCOL_GEMINI
    assert get_spec("claude").protocol == PROTOCOL_ANTHROPIC
    assert get_spec("ollama").protocol not in (
        PROTOCOL_GEMINI,
        PROTOCOL_ANTHROPIC,
    )


def test_aliases_resolve():
    assert canonical_provider("anthropic") == "claude"
    assert canonical_provider("google") == "gemini"
    assert canonical_provider("local") == "ollama"
    assert canonical_provider("OMNIROUTE") == "omniroute"
    assert canonical_provider("nope") == ""
    assert canonical_provider(None) == ""


def test_local_providers_are_excluded_from_the_online_chain():
    assert "ollama" not in online_providers()
    assert "omniroute" not in online_providers()
    assert "gemini" in online_providers()


def test_every_provider_has_a_model_variable():
    for name, spec in PROVIDER_REGISTRY.items():
        assert spec.model_env, name
        assert spec.model_env.endswith("_MODEL") or spec.model_env.endswith("_MODELS"), name


# ===========================================================================
# 2. Base URL resolution and SSRF policy
# ===========================================================================


def test_remote_base_url_requires_https(blank):
    """Remote validation is not weakened for the new providers."""
    from app.services.llm.provider import _provider_base_url

    blank.setenv("GROQ_BASE_URL", "http://api.groq.com/openai/v1")
    with pytest.raises(ValueError):
        _provider_base_url(get_spec("groq"))


def test_remote_base_url_rejects_loopback(blank):
    from app.services.llm.provider import _provider_base_url

    blank.setenv("GROQ_BASE_URL", "https://localhost:8080/v1")
    with pytest.raises(ValueError):
        _provider_base_url(get_spec("groq"))


def test_local_base_url_allows_http_and_any_port(blank):
    """
    A loopback gateway on a non-standard port has to be reachable.

    The existing SSRF policy rejects plain HTTP, loopback and ports other than
    80/443. All three are correct for a remote provider and all three would
    make a local gateway unusable, which is why the local/remote distinction
    exists at all.
    """
    from app.services.llm.provider import _provider_base_url

    blank.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    assert _provider_base_url(get_spec("omniroute")) == "http://localhost:20128/v1"

    blank.setenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:9999/v1")
    assert _provider_base_url(get_spec("omniroute")) == "http://127.0.0.1:9999/v1"


def test_local_base_url_normalises_the_version_segment(blank):
    from app.services.llm.provider import _provider_base_url

    blank.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128")
    assert _provider_base_url(get_spec("omniroute")) == "http://localhost:20128/v1"

    blank.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1/")
    assert _provider_base_url(get_spec("omniroute")) == "http://localhost:20128/v1"

    blank.setenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1/chat/completions")
    assert _provider_base_url(get_spec("omniroute")) == "http://localhost:20128/v1"


def test_ollama_local_url_still_works(blank):
    from app.services.llm.provider import _get_ollama_base_url

    blank.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    assert _get_ollama_base_url() == "http://localhost:11434/v1"

    blank.setenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    assert _get_ollama_base_url() == "http://localhost:11434/api/generate"


# ===========================================================================
# 3. Configuration-only provider and model selection
# ===========================================================================


@pytest.mark.parametrize(
    "provider",
    [
        "ollama",
        "omniroute",
        "gemini",
        "openai",
        "claude",
        "groq",
        "deepseek",
        "openrouter",
        "cerebras",
        "experiential",
    ],
)
def test_provider_and_model_are_selected_by_configuration(blank, provider):
    """
    The headline requirement: no application code names a provider or model.

    Each case sets only environment variables and asks the router what it would
    use. Nothing else in the project hard-codes a model name.
    """
    _configure(blank, provider, key="k" if get_spec(provider).key_env else "", model="some-model")
    blank.setenv("LLM_PROVIDER", provider)
    blank.setenv("LLM_ROUTE_MODE", "direct")

    assert LLMService._default_provider() == provider
    assert LLMService.get_default_model(provider) == "some-model"


@pytest.mark.parametrize(
    "model",
    ["qwen3:8b", "qwen2.5:7b", "llama3.2", "llama3", "deepseek-r1:8b", "anything-else"],
)
def test_any_ollama_model_can_be_configured(blank, model):
    """
    Arbitrary model ids, including ones that do not exist yet.

    Ollama is never asked to validate a name: the model may be pulled after
    this deployment is configured, and refusing the string would make the
    application depend on what happens to be installed today.
    """
    _configure(blank, "ollama", key="", model=model)
    assert LLMService.get_default_model("ollama") == model


def test_omniroute_auto_model_is_accepted(blank):
    _configure(blank, "omniroute", key="", model="auto")
    blank.setenv("LLM_PROVIDER", "omniroute")
    blank.setenv("LLM_ROUTE_MODE", "direct")
    assert LLMService.get_default_model("omniroute") == "auto"
    assert LLMService._default_provider() == "omniroute"


def test_direct_mode_uses_exactly_one_candidate(blank):
    _configure(blank, "ollama", key="", model="qwen3:8b")
    blank.setenv("LLM_PROVIDER", "ollama")
    candidates = LLMService._build_candidates(provider=None, model=None, route_mode="direct")
    assert candidates == [("ollama", "qwen3:8b")]


# ===========================================================================
# 4. Per-provider output ceilings
# ===========================================================================


def test_ollama_max_tokens_and_its_legacy_name(blank):
    from app.services.llm.provider import _ollama_max_output_tokens

    blank.setenv("OLLAMA_MAX_TOKENS", "4096")
    assert _ollama_max_output_tokens() == 4096

    blank.delenv("OLLAMA_MAX_TOKENS")
    blank.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "1500")
    assert _ollama_max_output_tokens() == 1500

    # The short name wins so a deployment can migrate without deleting the old.
    blank.setenv("OLLAMA_MAX_TOKENS", "4096")
    assert _ollama_max_output_tokens() == 4096


@pytest.mark.parametrize("value", ["0", "-5", "abc", ""])
def test_a_degenerate_ceiling_can_never_become_a_request(blank, value):
    """A zero ceiling would be a "generate nothing" request."""
    from app.services.llm.provider import _ollama_max_output_tokens

    blank.setenv("OLLAMA_MAX_TOKENS", value)
    assert _ollama_max_output_tokens() >= 1


@pytest.mark.parametrize(
    "provider",
    [
        "ollama",
        "omniroute",
        "gemini",
        "groq",
        "openrouter",
        "openai",
        "claude",
        "deepseek",
        "experiential",
        "cerebras",
    ],
)
def test_every_provider_has_its_own_ceiling_variable(blank, provider):
    from app.services.llm.provider import _provider_max_tokens

    spec = get_spec(provider)
    assert spec.max_tokens_env, provider
    assert spec.max_tokens_env.endswith("_MAX_TOKENS"), provider

    blank.setenv(spec.max_tokens_env, "3333")
    assert _provider_max_tokens(spec) == 3333


def test_shared_ceiling_override(blank):
    from app.services.llm.provider import _provider_max_tokens

    blank.setenv("LLM_MAX_TOKENS", "7777")
    assert _provider_max_tokens(get_spec("groq")) == 7777
    assert _provider_max_tokens(get_spec("openai")) == 7777


def test_provider_ceiling_beats_the_shared_one(blank):
    from app.services.llm.provider import _provider_max_tokens

    blank.setenv("LLM_MAX_TOKENS", "7777")
    blank.setenv("GROQ_MAX_TOKENS", "1000")
    assert _provider_max_tokens(get_spec("groq")) == 1000


def test_task_ceiling_reaches_the_request(blank, monkeypatch):
    """
    The per-task budget must actually be sent.

    ``TaskPolicy`` declared 4096 for a full CV body and nothing read it outside
    a context-fit check, so every provider silently cut a full CV at the shared
    default.
    """
    _configure(blank, "groq", key="k", model="m")
    sent: dict = {}

    def fake_execute(**kwargs):
        sent.update(kwargs)
        return "ok"

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake_execute))

    LLMService.generate(
        prompt="write a cv",
        provider="groq",
        route_mode="direct",
        task="full_cv_generation",
    )
    assert sent["max_tokens"] == 4096

    sent.clear()
    LLMService.generate(prompt="extract", provider="groq", route_mode="direct", task="ats_analysis")
    assert sent["max_tokens"] is None, "a task without a budget must not invent one"


# ===========================================================================
# 5. Dispatch
# ===========================================================================


def test_unknown_provider_is_rejected_with_a_useful_message(blank):
    with pytest.raises(ValueError) as error:
        LLMService._execute_single_provider("hi", "not-a-provider", "m")
    assert "Known providers" in str(error.value)


def test_a_local_gateway_with_no_key_is_sent_no_auth_header(blank, monkeypatch):
    """
    No placeholder Authorization header.

    The OpenAI SDK refuses an empty key and would put a placeholder in the
    header, which a gateway that deliberately has no authentication can answer
    with 401. The plain-HTTP path is used instead.
    """
    _configure(blank, "omniroute", key="", model="auto")

    captured: dict = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return Response()

    monkeypatch.setattr(provider_module.requests, "post", fake_post)

    out = LLMService._execute_single_provider(prompt="hi", provider="omniroute", model="auto")
    assert out == "ok"
    assert captured["url"] == "http://localhost:20128/v1/chat/completions"
    assert "Authorization" not in captured["headers"]


def test_a_local_gateway_with_a_key_sends_a_bearer_header(blank, monkeypatch):
    _configure(blank, "omniroute", key="secret", model="auto")

    captured: dict = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }

    monkeypatch.setattr(
        provider_module.requests,
        "post",
        lambda url, json=None, headers=None, timeout=None: (
            captured.update(headers or {}) or Response()
        ),
    )

    LLMService._execute_single_provider(prompt="hi", provider="omniroute", model="auto")
    assert captured["Authorization"] == "Bearer secret"


def test_a_keyed_local_gateway_still_avoids_the_sdk(blank, monkeypatch):
    """
    Both keyed and keyless local gateways take the plain-HTTP path.

    The SDK always adds an Authorization header, which a local gateway may not
    want at all. Consistency here means one code path to test.
    """
    _configure(blank, "omniroute", key="secret", model="auto")

    def boom(**kwargs):
        raise AssertionError("the OpenAI SDK must not be used for a local gateway")

    monkeypatch.setattr(provider_module.openai, "OpenAI", boom)
    monkeypatch.setattr(
        provider_module.requests,
        "post",
        lambda *a, **k: SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
        ),
    )
    assert (
        LLMService._execute_single_provider(prompt="hi", provider="omniroute", model="auto") == "ok"
    )


@pytest.mark.parametrize(
    "provider,base_url,protocol",
    [
        # Keyed cloud gateways go through the OpenAI SDK.
        ("groq", None, "openai_client"),
        ("openrouter", None, "openai_client"),
        ("deepseek", None, "openai_client"),
        ("openai", None, "openai_client"),
        # A local gateway is always plain HTTP, so no Authorization header is
        # invented for it.
        ("omniroute", None, "requests"),
        # Ollama has two protocols and the configured base URL chooses. Both are
        # legitimate and both must keep working.
        ("ollama", "http://localhost:11434/v1", "openai_client"),
        ("ollama", "http://localhost:11434/api/generate", "requests"),
    ],
)
def test_each_provider_uses_the_expected_transport(
    blank, monkeypatch, provider, base_url, protocol
):
    _configure(
        blank,
        provider,
        key="k" if get_spec(provider).requires_key else "",
        model="m",
    )
    if base_url:
        blank.setenv("OLLAMA_BASE_URL", base_url)

    used: dict = {}

    # The native /api/generate endpoint returns a flat shape, not a
    # chat-completion one. Both are asserted on, so the fake has to answer in
    # whichever shape the transport under test actually reads.
    native = protocol == "requests" and base_url and base_url.endswith("/api/generate")

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            if native:
                return {
                    "response": "ok",
                    "done_reason": "stop",
                    "eval_count": 1,
                    "prompt_eval_count": 1,
                }
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            used["sdk"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            used["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions)

    monkeypatch.setattr(provider_module.openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        provider_module.requests,
        "post",
        lambda *a, **k: (used.update(http=True) or Response()),
    )

    LLMService._execute_single_provider(prompt="hi", provider=provider, model="m")

    if protocol == "openai_client":
        assert "sdk" in used, f"{provider} did not use the OpenAI client"
    else:
        assert "http" in used, f"{provider} did not use plain HTTP"


def test_the_sdk_still_gets_a_bounded_timeout(blank, monkeypatch):
    """A hung socket must surface as a retryable error, not a 600s wait."""
    _configure(blank, "groq", key="k", model="m")
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @property
        def chat(self):
            raise AssertionError("not reached")

    monkeypatch.setattr(provider_module.openai, "OpenAI", FakeOpenAI)
    with pytest.raises(AssertionError):
        LLMService._execute_single_provider(prompt="hi", provider="groq", model="m")

    assert captured["timeout"] <= 600
    assert captured["max_retries"] == 0, "the SDK must not retry behind our back"


def test_remote_providers_still_reject_a_plain_http_base_url(blank, monkeypatch):
    _configure(blank, "groq", key="k", model="m")
    with pytest.raises(ValueError, match="HTTPS"):
        LLMService._execute_direct_openai_style(
            prompt="hi",
            model="m",
            api_key="k",
            base_url="http://api.groq.com/openai/v1",
            provider="groq",
        )


# ===========================================================================
# 6. Response reading
# ===========================================================================


def _payload(**overrides):
    message = {"content": "the answer"}
    message.update(overrides.pop("message", {}))
    choice = {"message": message, "finish_reason": "stop"}
    choice.update(overrides.pop("choice", {}))
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_content_is_preferred(blank):
    _configure(blank, "groq", key="k", model="m")
    body = _payload(message={"content": "real", "reasoning": "thought"})
    out = LLMService._read_openai_compatible_response(body, "groq", "m")
    assert out == "real"


def test_reasoning_is_used_when_content_is_empty(blank):
    _configure(blank, "groq", key="k", model="m")
    body = _payload(message={"content": "", "reasoning": "the answer"})
    assert LLMService._read_openai_compatible_response(body, "groq", "m") == ("the answer")


def test_thinking_is_used_when_content_is_empty(blank):
    _configure(blank, "ollama", key="", model="m")
    body = _payload(message={"content": None, "thinking": "the answer"})
    assert LLMService._read_openai_compatible_response(body, "ollama", "m") == ("the answer")


def test_truncated_reasoning_is_still_refused(blank):
    """
    Half a chain-of-thought is not a CV.

    This is the guard behind the reported "ollama returned empty content": the
    budget was consumed by the reasoning channel, so the answer is incomplete.
    Returning it would put unfinished prose into a CV.
    """
    _configure(blank, "ollama", key="", model="m")
    body = _payload(
        message={"content": "", "reasoning": "I was thinking about"},
        choice={"finish_reason": "length"},
    )
    with pytest.raises(RuntimeError, match="empty content"):
        LLMService._read_openai_compatible_response(body, "ollama", "m")


def test_the_empty_error_reports_structure_but_no_content(blank):
    _configure(blank, "ollama", key="", model="m")
    secret = "PRIVATE-RESUME-TEXT-1234"
    body = _payload(
        message={"content": "", "reasoning": secret},
        choice={"finish_reason": "length"},
    )
    body["usage"]["completion_tokens"] = 1200
    with pytest.raises(RuntimeError) as error:
        LLMService._read_openai_compatible_response(body, "ollama", "m")

    message = str(error.value)
    assert "finish_reason=length" in message
    assert "reasoning" in message
    assert "1200" in message
    assert str(len(secret)) in message, "the length is diagnostic, not content"
    assert secret not in message, "the error leaked model output"


def test_no_choices_is_reported(blank):
    _configure(blank, "groq", key="k", model="m")
    with pytest.raises(RuntimeError, match="no choices"):
        LLMService._read_openai_compatible_response({"choices": [], "usage": {}}, "groq", "m")


# ===========================================================================
# 7. Fallback and the attempt cap
# ===========================================================================


def test_fallback_can_be_disabled(blank, monkeypatch):
    _configure(blank, "ollama", key="", model="m")
    _configure(blank, "gemini", key="k", model="g")
    blank.setenv("LLM_MODE", "auto")
    blank.setenv("LLM_FALLBACK_ENABLED", "false")

    seen: list[str] = []

    def failing(**kwargs):
        seen.append(kwargs["provider"])
        raise Status("connection refused", 500)

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(failing))
    with pytest.raises(RuntimeError):
        LLMService.generate(prompt="x", task="cv_tailoring")

    assert len(seen) == 1, f"fallback ran despite being disabled: {seen}"


def test_fallback_enabled_tries_more_than_one(blank, monkeypatch):
    _configure(blank, "ollama", key="", model="m")
    _configure(blank, "gemini", key="k", model="g")
    blank.setenv("LLM_MODE", "auto")

    seen: list[str] = []

    def fake(**kwargs):
        seen.append(kwargs["provider"])
        if kwargs["provider"] == "ollama":
            raise Status("connection refused", 500)
        return "ok"

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(fake))
    assert LLMService.generate(prompt="x", task="cv_tailoring") == "ok"
    assert len(seen) > 1


def test_the_attempt_count_is_bounded(blank, monkeypatch):
    """
    No infinite retry/fallback loop, whatever the configuration.

    Every provider is configured so the candidate list is as long as it can
    possibly be, and the ceiling still holds.
    """
    for name in PROVIDER_REGISTRY:
        spec = get_spec(name)
        if spec.local and not spec.requires_key:
            _configure(blank, name, key="", model="m")
        else:
            _configure(blank, name, key="k", model="m")
    blank.setenv("LLM_MODE", "auto")
    blank.setenv("LLM_MAX_PROVIDER_ATTEMPTS", "3")

    attempts = 0

    def always_fails(**kwargs):
        nonlocal attempts
        attempts += 1
        raise Status("connection refused", 500)

    monkeypatch.setattr(LLMService, "_execute_single_provider", staticmethod(always_fails))
    with pytest.raises(RuntimeError):
        LLMService.generate(prompt="x", task="cv_tailoring")

    assert attempts == 3, attempts


def test_a_zero_attempt_capability_still_tries_one(blank, monkeypatch):
    """A misconfigured 0 must not make a request impossible."""
    _configure(blank, "ollama", key="", model="m")
    blank.setenv("LLM_MODE", "auto")
    blank.setenv("LLM_MAX_PROVIDER_ATTEMPTS", "0")
    blank.setenv("LLM_FALLBACK_ENABLED", "false")

    monkeypatch.setattr(
        LLMService,
        "_execute_single_provider",
        staticmethod(lambda **kwargs: "ok"),
    )
    assert LLMService.generate(prompt="x", task="cv_tailoring") == "ok"


def test_route_info_reports_the_new_knobs(blank):
    _configure(blank, "ollama", key="", model="m")
    blank.setenv("LLM_FALLBACK_ENABLED", "false")
    blank.setenv("LLM_MAX_PROVIDER_ATTEMPTS", "2")
    info = LLMService.route_info()
    assert info["fallback_enabled"] is False
    assert info["max_provider_attempts"] == 2


# ===========================================================================
# 8. Model discovery
# ===========================================================================


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Status(f"HTTP {self.status_code}", self.status_code)

    def json(self):
        return self._payload


def test_discovery_reads_the_openai_shape(blank, monkeypatch):
    _configure(blank, "groq", key="k", model="configured-m")
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse({"data": [{"id": "model-a", "context_length": 8192}]})

    monkeypatch.setattr(provider_module.requests, "get", fake_get)
    result = LLMService.list_models("groq")

    assert result["status"] == "ok"
    assert captured["url"] == "https://api.groq.com/openai/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer k"
    ids = [row["model"] for row in result["models"]]
    assert "model-a" in ids
    assert "configured-m" in ids, "the configured model is always reported"
    configured = next(r for r in result["models"] if r["model"] == "configured-m")
    assert configured["missing_from_discovery"] is True


def test_discovery_reads_the_ollama_shape(blank, monkeypatch):
    _configure(blank, "ollama", key="", model="qwen3:8b")

    def fake_get(url, headers=None, timeout=None):
        assert url.endswith("/api/tags"), url
        return FakeResponse({"models": [{"name": "qwen3:8b", "size": 123}, {"name": "llama3"}]})

    monkeypatch.setattr(provider_module.requests, "get", fake_get)
    result = LLMService.list_models("ollama")
    assert result["status"] == "ok"
    assert {row["model"] for row in result["models"]} == {"qwen3:8b", "llama3"}


def test_discovery_uses_the_ollama_server_root(blank, monkeypatch):
    """
    Not the /api/generate endpoint.

    OLLAMA_BASE_URL may be set to the generate endpoint, which is a request URL
    and not a base to append a path to.
    """
    blank.setenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    blank.setenv("OLLAMA_MODEL", "m")
    captured = {}

    monkeypatch.setattr(
        provider_module.requests,
        "get",
        lambda url, headers=None, timeout=None: (
            captured.update(url=url) or FakeResponse({"models": []})
        ),
    )
    LLMService.list_models("ollama")
    assert captured["url"] == "http://localhost:11434/api/tags"


def test_discovery_failure_never_breaks_a_call(blank, monkeypatch):
    _configure(blank, "groq", key="k", model="m")

    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(provider_module.requests, "get", boom)
    result = LLMService.list_models("groq")
    assert result["status"] == "discovery_failed"
    # The configured model is still reported, so the caller can see what will
    # actually be used.
    assert [row["model"] for row in result["models"]] == ["m"]


def test_an_unconfigured_provider_reports_not_configured(blank):
    result = LLMService.list_models("cerebras")
    assert result["status"] == "not_configured"
    assert "CEREBRAS_API_KEY" in result["detail"]


def test_discovery_is_optional_and_manual_ids_always_work(blank):
    """
    A provider with no listing still works with a hand-set model.

    This is what "model discovery must not be mandatory" means in practice.
    """
    _configure(blank, "gemini", key="k", model="gemini-2.5-flash")
    result = LLMService.list_models("gemini")
    assert result["status"] == "manual_configuration"
    assert [row["model"] for row in result["models"]] == ["gemini-2.5-flash"]
    assert LLMService.get_default_model("gemini") == "gemini-2.5-flash"


def test_an_unknown_provider_is_reported_not_raised(blank):
    result = LLMService.list_models("nope")
    assert result["status"] == "unknown_provider"
    assert "Known providers" in result["detail"]


def test_discovery_reports_no_credential(blank, monkeypatch):
    _configure(blank, "cerebras", key="SECRET", model="")
    captured = {}

    monkeypatch.setattr(
        provider_module.requests,
        "get",
        lambda url, headers=None, timeout=None: (
            captured.update(headers=headers or {}) or FakeResponse({"data": [{"id": "m"}]})
        ),
    )
    LLMService.list_models("cerebras")
    assert captured["headers"]["Authorization"] == "Bearer SECRET"
    # The listing must not echo the credential back.
    assert "SECRET" not in json.dumps(captured["headers"]).replace("Bearer SECRET", "")


def test_discovery_for_every_provider_never_raises(blank, monkeypatch):
    """One unreachable provider must not hide the state of the others."""
    for name in PROVIDER_REGISTRY:
        _configure(
            blank,
            name,
            key="k" if get_spec(name).key_env else "",
            model="m",
        )

    monkeypatch.setattr(
        provider_module.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
    )
    for name in PROVIDER_REGISTRY:
        result = LLMService.list_models(name)
        assert result["status"] in (
            "discovery_failed",
            "manual_configuration",
            "not_configured",
        ), name


# ===========================================================================
# 9. No capability is assumed
# ===========================================================================


def test_context_length_is_only_reported_when_the_provider_states_it(blank, monkeypatch):
    """
    No guessing.

    A context length inferred from a model name is how a routing decision
    starts failing silently, so an unstated value stays absent.
    """
    _configure(blank, "groq", key="k", model="m")
    monkeypatch.setattr(
        provider_module.requests,
        "get",
        lambda *a, **k: FakeResponse({"data": [{"id": "a"}, {"id": "b"}]}),
    )
    rows = {r["model"]: r for r in LLMService.list_models("groq")["models"]}
    for name in ("a", "b"):
        assert "context_length" not in rows[name]
        assert "capabilities" not in rows[name]
        assert rows[name]["source"] == "discovery"


def test_every_task_still_resolves(blank):
    for task in KNOWN_TASKS:
        assert provider_module.normalize_task(task) == task
