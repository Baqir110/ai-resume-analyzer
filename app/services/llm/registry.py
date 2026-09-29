"""
Provider registry: the single place a provider is described.

Why this module exists
----------------------
Adding a provider previously meant editing seven separate lists in
``provider.py``: the ``if``/``elif`` dispatch chain, ``SUPPORTED_PROVIDERS``,
``_PROVIDER_KEY_ENV``, ``_PROVIDER_MODEL_ENV``, ``DEFAULT_MODELS``,
``_provider_is_configured``'s membership set, and the default provider chain.
Nothing checked that those stayed consistent, and a new provider was silently
unusable if one list was missed.

A provider is now a single declarative record. Adding OmniRoute - or any other
OpenAI-compatible gateway - is one row here and nothing else.

Protocols
---------
``openai_compatible``
    Anything that speaks ``POST {base_url}/chat/completions``. This is most
    gateways, including local ones.
``gemini``
    Google's ``google-genai`` SDK. Kept separate because the request and
    response shapes differ (no ``max_tokens`` field, different usage keys).
``anthropic``
    The ``anthropic`` SDK. Separate for the same reason, and because it has its
    own timeout and token-accounting semantics.
``ollama_native``
    Ollama's flat ``/api/generate`` response, which is not an OpenAI shape.

Local vs remote
---------------
``local=True`` providers are expected to run on the operator's own machine
(Ollama, OmniRoute by default). Local base URLs are allowed to be plain HTTP
and to use non-standard ports, because that is what a loopback gateway needs.
``local=False`` providers have their base URL run through the existing SSRF
validation, which requires HTTPS and rejects loopback and private addresses.
That distinction is deliberate: relaxing remote validation would be a security
regression, and requiring HTTPS for a loopback gateway would make OmniRoute
unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_OPENAI_COMPATIBLE = "openai_compatible"
PROTOCOL_GEMINI = "gemini"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOL_OLLAMA_NATIVE = "ollama_native"


@dataclass(frozen=True)
class ProviderSpec:
    """
    Everything the router needs to know about one provider.

    Attributes:
        name: Canonical provider name used in configuration and logs.
        protocol: Which execution path handles it.
        base_url_env: Environment variable holding the base URL.
        default_base_url: Used when the variable is unset or blank.
        key_env: Environment variables that may hold the credential, in
            priority order. Empty means the provider needs no credential,
            which is the normal case for a loopback gateway.
        model_env: Environment variable naming the model.
        default_model: Used when ``model_env`` is unset or blank.
        local: True for providers expected to run on the operator's machine.
            Controls base-URL validation only; it never changes which protocol
            is used.
        max_tokens_env: Environment variable overriding the output ceiling.
        default_max_tokens: Output ceiling when nothing is configured.
        max_tokens_floor: Lower bound, so a misconfigured 0 cannot become a
            "generate nothing" request.
        temperature: Default sampling temperature.
        timeout_env: Environment variable overriding the request timeout.
        default_timeout_seconds: Timeout when nothing is configured.
        discoverable: True when the provider exposes a model listing.
        models_path: Path appended to the base URL for discovery.
        requires_key: True when a blank credential must be rejected outright
            rather than tolerated for a local install.
        extra_headers_env: Environment variables holding extra HTTP headers,
            as ``NAME=value`` pairs. Some gateways need a routing header.
        discovery_headers: Static headers required by the listing endpoint.
            Anthropic's ``/v1/models`` rejects a request without its
            ``anthropic-version`` header, so this is not optional there.
        api_key_header: How the credential is sent. OpenAI-shaped providers use
            ``Authorization: Bearer``; Anthropic uses ``x-api-key``. Leaving
            this at the default keeps every existing provider working.
        notes: Human-readable note surfaced by the smoke test and the
            ``/model-discovery`` endpoint.
    """

    name: str
    protocol: str
    base_url_env: str
    default_base_url: str
    model_env: str
    default_model: str
    local: bool = False
    key_env: tuple[str, ...] = ()
    max_tokens_env: str = ""
    default_max_tokens: int = 2048
    max_tokens_floor: int = 1
    temperature: float = 0.2
    timeout_env: str = "LLM_REQUEST_TIMEOUT_SECONDS"
    default_timeout_seconds: int = 180
    discoverable: bool = True
    models_path: str = "/models"
    requires_key: bool = True
    extra_headers_env: tuple[str, ...] = ()
    discovery_headers: tuple[tuple[str, str], ...] = ()
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    aliases: tuple[str, ...] = ()
    notes: str = ""


def _spec(**kwargs: Any) -> ProviderSpec:
    return ProviderSpec(**kwargs)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
#
# Order matters only for display and for the default automatic chain, which is
# built separately in provider.py. Every entry is otherwise independent.

PROVIDER_REGISTRY: dict[str, ProviderSpec] = {}


def _register(spec: ProviderSpec) -> ProviderSpec:
    PROVIDER_REGISTRY[spec.name] = spec
    return spec


# -- Local inference -------------------------------------------------------

# Ollama is the only provider with two protocols: the OpenAI-compatible ``/v1``
# shim, and its own flat ``/api/generate`` endpoint. The choice is made from
# the configured base URL, exactly as before, because a user who points
# OLLAMA_BASE_URL at one or the other expects that to be honoured.
_register(
    _spec(
        name="ollama",
        protocol=PROTOCOL_OLLAMA_NATIVE,
        base_url_env="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434/v1",
        model_env="OLLAMA_MODEL",
        default_model="glm-4.7-flash",
        local=True,
        # Both spellings are accepted. OLLAMA_MAX_OUTPUT_TOKENS is the long
        # form this repository has always used; OLLAMA_MAX_TOKENS is the
        # shorter name that matches the other providers.
        max_tokens_env="OLLAMA_MAX_TOKENS",
        default_max_tokens=2048,
        # Ollama's own context ceiling; see _ollama_num_ctx.
        discoverable=True,
        models_path="/api/tags",
        requires_key=False,
        aliases=("local",),
        notes="Local inference. Model must be pulled first: ollama pull <model>.",
    )
)

_register(
    _spec(
        name="omniroute",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="OMNIROUTE_BASE_URL",
        default_base_url="http://localhost:20128/v1",
        model_env="OMNIROUTE_MODEL",
        default_model="auto",
        local=True,
        key_env=("OMNIROUTE_API_KEY",),
        max_tokens_env="OMNIROUTE_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
        # A local gateway may run without authentication, so a blank key must
        # not be treated as misconfiguration.
        requires_key=False,
        notes=(
            "Local OpenAI-compatible gateway. Leave OMNIROUTE_API_KEY blank "
            "when the gateway does not require authentication."
        ),
    )
)


# -- OpenAI-compatible gateways -------------------------------------------

_register(
    _spec(
        name="openrouter",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="OPENROUTER_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        model_env="OPENROUTER_MODEL",
        default_model="openrouter/free",
        key_env=("OPENROUTER_API_KEY",),
        max_tokens_env="OPENROUTER_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
        notes=(
            "Aggregator. Model IDs and any free-tier offering change over "
            "time; set OPENROUTER_MODEL explicitly."
        ),
    )
)

_register(
    _spec(
        name="groq",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="GROQ_BASE_URL",
        default_base_url="https://api.groq.com/openai/v1",
        model_env="GROQ_MODEL",
        default_model="openai/gpt-oss-120b",
        key_env=("GROQ_API_KEY",),
        max_tokens_env="GROQ_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
    )
)

_register(
    _spec(
        name="deepseek",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
        model_env="DEEPSEEK_MODEL",
        default_model="deepseek-chat",
        key_env=("DEEPSEEK_API_KEY",),
        max_tokens_env="DEEPSEEK_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
    )
)

_register(
    _spec(
        name="openai",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        # Historical name. OPENAI_BASE_URL belongs to the Experiential gateway
        # and must keep working, so OpenAI's own endpoint keeps the explicit
        # name it has always used here.
        base_url_env="OPENAI_DIRECT_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        model_env="OPENAI_MODEL",
        default_model="gpt-4o-mini",
        key_env=("OPENAI_API_KEY",),
        max_tokens_env="OPENAI_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
    )
)

_register(
    _spec(
        name="cerebras",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="CEREBRAS_BASE_URL",
        default_base_url="https://api.cerebras.ai/v1",
        model_env="CEREBRAS_MODEL",
        default_model="",
        key_env=("CEREBRAS_API_KEY",),
        max_tokens_env="CEREBRAS_MAX_TOKENS",
        default_max_tokens=2048,
        # No default model on purpose: Cerebras model IDs change, and shipping a
        # stale one would silently send the CV to a model that no longer
        # exists. A deployment must set CEREBRAS_MODEL; discovery lists what
        # the account can currently reach.
        discoverable=True,
    )
)

_register(
    _spec(
        name="cloudflare",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="CLOUDFLARE_BASE_URL",
        default_base_url="https://api.cloudflare.com/client/v4",
        model_env="CLOUDFLARE_MODEL",
        default_model="",
        key_env=("CLOUDFLARE_API_KEY", "CLOUDFLARE_API_TOKEN"),
        max_tokens_env="CLOUDFLARE_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
        notes=(
            "Workers AI. The account id is part of the base URL, so set "
            "CLOUDFLARE_BASE_URL to the full "
            "https://api.cloudflare.com/client/v4/ai/<account>/v1 value."
        ),
    )
)

_register(
    _spec(
        name="github",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="GITHUB_MODELS_BASE_URL",
        default_base_url="https://models.github.ai/inference",
        model_env="GITHUB_MODELS_MODEL",
        default_model="",
        key_env=("GITHUB_MODELS_API_KEY", "GITHUB_TOKEN"),
        max_tokens_env="GITHUB_MODELS_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
        notes="GitHub Models. Available models depend on repository permissions.",
    )
)

_register(
    _spec(
        name="huggingface",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        base_url_env="HUGGINGFACE_BASE_URL",
        default_base_url="https://router.huggingface.co/v1",
        model_env="HUGGINGFACE_MODEL",
        default_model="",
        key_env=("HUGGINGFACE_API_KEY", "HF_TOKEN"),
        max_tokens_env="HUGGINGFACE_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=True,
        notes="Hugging Face Inference Router.",
    )
)

_register(
    _spec(
        name="experiential",
        protocol=PROTOCOL_OPENAI_COMPATIBLE,
        # OPENAI_BASE_URL is the historical name for this gateway. It is kept.
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.experientiallabs.ai/v1",
        model_env="EXPERIENTIAL_MODEL",
        default_model="nemotron-3-ultra-550b-a55b",
        key_env=("EXPLABS_API_KEY", "EXPERIENTIAL_ORG_KEY"),
        max_tokens_env="EXPERIENTIAL_MAX_TOKENS",
        default_max_tokens=2048,
        discoverable=False,
        notes="Experiential Labs OpenAI-compatible gateway.",
    )
)


# -- Non-OpenAI-compatible providers ---------------------------------------

_register(
    _spec(
        name="gemini",
        protocol=PROTOCOL_GEMINI,
        # The google-genai client is constructed without a base URL, so this
        # is recorded for reporting only.
        base_url_env="GEMINI_BASE_URL",
        default_base_url="https://generativelanguage.googleapis.com",
        model_env="GEMINI_MODEL",
        default_model="gemini-2.5-flash",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        max_tokens_env="GEMINI_MAX_TOKENS",
        default_max_tokens=2048,
        # The google-genai response has no OpenAI-style usage block, so the
        # model list is not read from a /models endpoint here.
        discoverable=False,
        aliases=("google",),
    )
)

_register(
    _spec(
        name="claude",
        protocol=PROTOCOL_ANTHROPIC,
        base_url_env="ANTHROPIC_BASE_URL",
        default_base_url="https://api.anthropic.com",
        model_env="CLAUDE_MODEL",
        default_model="claude-sonnet-4-20250514",
        key_env=("ANTHROPIC_API_KEY",),
        max_tokens_env="CLAUDE_MAX_TOKENS",
        # Anthropic requires max_tokens, and a long CV body needs more room
        # than the shared default.
        default_max_tokens=4096,
        discoverable=True,
        # The base URL is the API root, so the listing lives under /v1.
        models_path="/v1/models",
        # Anthropic's listing endpoint requires its version header and answers
        # 404 without it. Sending it here keeps discovery working instead of
        # reporting an empty catalogue.
        discovery_headers=(("anthropic-version", "2023-06-01"),),
        # Anthropic authenticates with a bare x-api-key header, not a bearer
        # token, so the OpenAI-shaped default would be rejected.
        api_key_header="x-api-key",
        api_key_prefix="",
        # The dashboard has always offered this provider under this name.
        aliases=("anthropic",),
    )
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

#: Canonical order, used for display and the default automatic chain.
PROVIDER_ORDER: tuple[str, ...] = (
    "ollama",
    "omniroute",
    "gemini",
    "openai",
    "claude",
    "groq",
    "deepseek",
    "openrouter",
    "cerebras",
    "cloudflare",
    "github",
    "huggingface",
    "experiential",
)

_ALIAS_INDEX: dict[str, str] = {}
for _name, _spec_item in PROVIDER_REGISTRY.items():
    _ALIAS_INDEX[_name] = _name
    for _alias in _spec_item.aliases:
        _ALIAS_INDEX[_alias] = _name


def canonical_provider(name: str | None) -> str:
    """
    Resolve a provider name or alias to its canonical form.

    Returns an empty string for an unknown name so callers can raise their own
    error with better context than a bare ``KeyError``.
    """
    if not isinstance(name, str):
        return ""
    return _ALIAS_INDEX.get(name.strip().lower(), "")


def get_spec(provider: str) -> ProviderSpec | None:
    """Return the registry record for ``provider``, or None if unknown."""
    return PROVIDER_REGISTRY.get(canonical_provider(provider))


def known_providers() -> tuple[str, ...]:
    """Canonical names of every registered provider."""
    return PROVIDER_ORDER


def online_providers() -> tuple[str, ...]:
    """
    Providers that leave the machine.

    Ollama and OmniRoute are excluded because both default to a loopback
    address. An operator who points either at a remote host still gets the
    privacy behaviour they asked for by choosing the mode deliberately.
    """
    return tuple(name for name in PROVIDER_ORDER if not PROVIDER_REGISTRY[name].local)


def provider_uses_openai_protocol(provider: str) -> bool:
    """True when ``provider`` can be served by the OpenAI-compatible path."""
    spec = get_spec(provider)
    return bool(spec and spec.protocol in (PROTOCOL_OPENAI_COMPATIBLE, PROTOCOL_OLLAMA_NATIVE))


__all__ = [
    "PROVIDER_ORDER",
    "PROVIDER_REGISTRY",
    "PROTOCOL_ANTHROPIC",
    "PROTOCOL_GEMINI",
    "PROTOCOL_OLLAMA_NATIVE",
    "PROTOCOL_OPENAI_COMPATIBLE",
    "ProviderSpec",
    "canonical_provider",
    "get_spec",
    "known_providers",
    "online_providers",
    "provider_uses_openai_protocol",
]
