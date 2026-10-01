from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anthropic
import openai
import requests
from dotenv import load_dotenv

from app.core.security import UnsafeInputError, validate_public_http_url
from app.services.llm.registry import (
    PROTOCOL_ANTHROPIC,
    PROTOCOL_GEMINI,
    PROTOCOL_OPENAI_COMPATIBLE,
    PROVIDER_ORDER,
    PROVIDER_REGISTRY,
    ProviderSpec,
    canonical_provider,
    get_spec,
    online_providers,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Project / environment
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """
    Read a float setting, falling back on anything unparseable.

    A typo in a numeric setting should not take down a request, so this behaves
    like :func:`_env_int`: an unparseable value yields the default rather than
    raising at the point of use.
    """
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name).lower()

    if not value:
        return default

    return value in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


# ============================================================================
# URLs
# ============================================================================

DEFAULT_EXPERIENTIAL_BASE_URL = "https://api.experientiallabs.ai/v1"

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _clean_base_url(value: str | None, default: str = "") -> str:
    """
    Normalize a URL without turning an empty optional configuration value
    into a validation error.

    Markdown-wrapped URLs are also cleaned because older versions of the
    project sometimes stored URLs copied from Markdown.
    """
    value = (value or "").strip()

    if not value:
        return default.rstrip("/")

    markdown_match = re.search(
        r"\((https?://[^)\s]+)\)",
        value,
        flags=re.IGNORECASE,
    )

    if markdown_match:
        value = markdown_match.group(1)

    value = value.strip().strip("`").rstrip(".,;")

    return value.rstrip("/")


def _validate_remote_base_url(
    value: str,
    *,
    resolve_dns: bool = False,
) -> str:
    """
    Validate a public HTTPS URL.

    This function is intentionally NOT called during module import for
    optional providers. A disabled provider must never prevent the entire
    application from importing.
    """
    value = (value or "").strip()

    if not value:
        raise ValueError("LLM provider URL is empty")

    try:
        safe = validate_public_http_url(
            value,
            resolve_dns=resolve_dns,
        ).rstrip("/")
    except UnsafeInputError as exc:
        raise ValueError(f"Unsafe LLM provider URL: {exc}") from exc

    parsed = urlsplit(safe)

    if parsed.scheme.lower() != "https":
        raise ValueError(f"Remote LLM provider URL must use HTTPS: {safe}")

    if not parsed.netloc:
        raise ValueError(f"Remote LLM provider URL has no hostname: {safe}")

    return safe


def _get_remote_url(
    env_name: str,
    default: str,
) -> str:
    """
    Resolve and validate a remote provider URL only when that provider is
    actually needed.
    """
    raw = _env(env_name, default)

    if not raw:
        raw = default

    return _validate_remote_base_url(_clean_base_url(raw, default))


def _provider_base_url(spec: ProviderSpec) -> str:
    """
    Resolve a provider's base URL, applying the right validation policy.

    Local providers (Ollama, a loopback gateway such as OmniRoute) are allowed
    plain HTTP and a non-standard port, because that is what a gateway on the
    operator's own machine needs. Everything else goes through
    :func:`_validate_remote_base_url`, which requires HTTPS and rejects
    loopback and private addresses.

    The local branch normalises the path only -- it deliberately does not
    re-implement the private-address checks, because a local provider is
    exactly the case those checks are meant to exclude. Remote validation is
    unchanged, so no SSRF protection is weakened to make a new provider work.
    """
    raw = _env(spec.base_url_env)

    if not raw:
        raw = spec.default_base_url

    cleaned = _clean_base_url(raw, spec.default_base_url)

    if spec.local:
        return _normalise_local_base_url(cleaned)

    return _validate_remote_base_url(cleaned)


def _normalise_local_base_url(value: str) -> str:
    """
    Normalise the path of a loopback base URL.

    A gateway is configured either as a bare host or with its version segment
    already present. Appending ``/v1`` unconditionally would produce
    ``/v1/v1``; leaving a bare host alone would produce ``/chat/completions``
    without a version. Only the shape is normalised -- the scheme, host and port
    are the operator's to choose, which is what allows a non-standard port.
    """
    cleaned = _clean_base_url(value)

    if not cleaned:
        return cleaned

    lowered = cleaned.casefold()

    for suffix in ("/chat/completions", "/completions"):
        if lowered.endswith(suffix):
            return cleaned[: -len(suffix)].rstrip("/")

    if not lowered.rstrip("/").endswith("/v1"):
        cleaned = cleaned.rstrip("/") + "/v1"

    return cleaned


def _get_ollama_base_url() -> str:
    """
    Ollama is intentionally allowed to use local HTTP.

    The preferred configuration is:

        OLLAMA_BASE_URL=http://localhost:11434/v1

    The implementation also accepts:
        http://127.0.0.1:11434/v1
        http://localhost:11434
        http://localhost:11434/api/generate
    """
    raw = _clean_base_url(
        _env("OLLAMA_BASE_URL"),
        DEFAULT_OLLAMA_BASE_URL,
    )

    if not raw:
        raw = DEFAULT_OLLAMA_BASE_URL

    lower = raw.lower()

    if lower.endswith("/api/generate"):
        return raw

    if lower.endswith("/api"):
        return raw[:-4] + "/v1"

    if not lower.endswith("/v1"):
        raw = raw.rstrip("/") + "/v1"

    return raw


# Default ceiling for a single Ollama completion.
#
# This was previously hard-coded to 1200 for every Ollama call, which is low
# enough that a long CV rewrite is frequently cut off mid-sentence. Ollama
# truncates the generation rather than erroring when the ceiling is reached,
# so a too-small budget shows up as a half-written CV, not as a failure.
DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS = 2048


def _ollama_max_output_tokens() -> int:
    """
    Resolve the per-request output-token ceiling for Ollama.

    Two spellings are accepted because both are in circulation:
    ``OLLAMA_MAX_TOKENS`` (the name every other provider uses) and
    ``OLLAMA_MAX_OUTPUT_TOKENS`` (this project's long-standing name). The short
    name wins when both are set, so a deployment can migrate without deleting
    the old line first.

    Clamped to at least 1 so a misconfigured value (``0``, negative,
    non-numeric) can never turn into a "generate nothing" request that looks
    like an empty-content failure.
    """
    for name in ("OLLAMA_MAX_TOKENS", "OLLAMA_MAX_OUTPUT_TOKENS"):
        raw = _env(name)
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "%s is not a number (%r); using the %d default.",
                name,
                raw[:24],
                DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS,
            )
            break

    return max(1, DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS)


def _provider_max_tokens(spec: ProviderSpec) -> int:
    """
    Resolve one provider's output ceiling.

    The provider's own variable wins, then a shared ``LLM_MAX_TOKENS``, then the
    registry default. This is what lets a long CV body be afforded on a gateway
    that can pay for it without raising the ceiling for every provider.
    """
    for name in (spec.max_tokens_env, "LLM_MAX_TOKENS"):
        if not name:
            continue
        raw = _env(name)
        if not raw:
            continue
        try:
            return max(spec.max_tokens_floor, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "%s is not a number (%r); using the %s default of %d.",
                name,
                raw[:24],
                spec.name,
                spec.default_max_tokens,
            )
            break

    return max(spec.max_tokens_floor, spec.default_max_tokens)


def _provider_timeout_seconds(spec: ProviderSpec) -> int:
    """Resolve one provider's request timeout, in seconds."""
    raw = _env(spec.timeout_env)

    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "%s is not a number (%r); using the %s default of %ds.",
                spec.timeout_env,
                raw[:24],
                spec.name,
                spec.default_timeout_seconds,
            )

    return max(1, spec.default_timeout_seconds)


# Context-window ladder used to size each Ollama request.
#
# Ollama truncates the *middle* of an over-long prompt instead of failing, so a
# prompt that does not fit is silently corrupted: the model never sees the end
# of the resume. Sizing the window per request avoids that.
#
# The values are quantised onto a short ladder on purpose. Ollama re-allocates
# the KV cache (and can reload the weights) when num_ctx changes, so sizing
# every request exactly would thrash on back-to-back calls. Bucketing means a
# typical run reuses the same window and pays the cost at most once.
#
# VRAM for qwen3:8b (36 layers, 8 KV heads, head_dim 128, fp16) on top of
# ~5.3 GB of weights:
#   4096 -> ~5.9 GB    8192 -> ~6.4 GB    12288 -> ~7.0 GB
#   16384 -> ~7.5 GB   24576 -> ~8.8 GB  32768 -> ~10.0 GB
#
# The upper rungs are unreachable on a small card because _auto_context_ceiling()
# derives the ceiling from actual free VRAM, and are capped on every machine by
# OLLAMA_MAX_CONTEXT_TOKENS (default 12288) when the hardware cannot be
# inspected. 16384 does not fit an 8 GB card alongside anything else, so it is
# only ever selected on hardware that can hold it.
DEFAULT_OLLAMA_CONTEXT_LADDER = (4096, 8192, 12288, 16384, 24576, 32768)
DEFAULT_OLLAMA_MAX_CONTEXT_TOKENS = 12288

# Hard prompt limit of Ollama's OpenAI-compatible /v1 endpoint.
#
# Measured on ollama 0.34.4: /v1 truncates the prompt to ~2050 tokens and
# ignores num_ctx, whichever way it is sent. Every shape was tried --
# top-level num_ctx, options.num_ctx, context_length, max_tokens, an
# ollama_options passthrough and extra_num_ctx -- and all of them returned
# exactly 2050 prompt tokens. There is no request that lifts it, so the
# application has to know about it: otherwise Ollama silently drops the middle
# of a long posting and the model reasons about half a job description.
#
# The native /api/generate endpoint has no such limit; it honours num_ctx.
OPENAI_COMPAT_OLLAMA_PROMPT_TOKEN_LIMIT = 2048

# Slack left free on the card when deriving a context window from VRAM.
#
# Sized from measurement on an 8 GB laptop (RTX 4060, qwen3:8b Q4_K_M): at
# num_ctx=16384 the model fits with 753 MiB to spare, but the usable prompt
# length had already stopped growing at 8192, so the extra window bought
# nothing while leaving no room for a compositor or a browser. 1024 MiB keeps
# the derived choice at 12288 there, which measured identical prompt capacity
# for 210 MiB less VRAM. 24576 does spill (about 1.1 GB of the model onto the
# CPU, and 19.0s instead of 14.2s), which this margin is wide enough to avoid.
DEFAULT_VRAM_HEADROOM_MB = 1024

# Used only when Ollama cannot report the model's geometry. Deliberately
# pessimistic so an unknown model gets a smaller window, never a larger one.
_FALLBACK_KV_BYTES_PER_TOKEN = 160 * 1024

# Cache for the derived ceiling. Keyed by model name; the answer only changes
# when the machine or the model does, and nvidia-smi plus /api/show on every
# request would be wasteful.
_VRAM_BUDGET_CACHE: dict[str, int] = {}


#: Below this, a clamped answer is not worth the round trip: it would be a
#: fragment of a CV, which is worse than a clear "this needs the cloud" skip.
_MIN_USABLE_OUTPUT_TOKENS = 256


def _ollama_uses_compatible_shim(base_url: str) -> bool:
    """
    True when Ollama is reached through the OpenAI-compatible ``/v1`` shim.

    The shim and the native endpoint are not interchangeable: only the native
    endpoint honours ``num_ctx``. Every budget decision below depends on which
    one is in use, so it is asked once, here.
    """
    return not (base_url or "").lower().endswith("/api/generate")


def _ollama_usable_window(base_url: str) -> int:
    """
    The context window Ollama will actually honour for this configuration.

    On the ``/v1`` shim this is a fixed 2048 tokens covering prompt *and*
    completion, and no request field changes it. On the native endpoint the
    window is whatever ``num_ctx`` is set to, which the request controls, so the
    limit is not a property of the transport and is not applied here.
    """
    if _ollama_uses_compatible_shim(base_url):
        return OPENAI_COMPAT_OLLAMA_PROMPT_TOKEN_LIMIT

    return 0


def _ollama_effective_ceiling(
    prompt: str,
    requested: int,
    model: str,
    base_url: str,
) -> tuple[int, int]:
    """
    Return ``(ceiling, window)`` for one Ollama request.

    ``ceiling`` is the output budget that can actually be produced, which on the
    shim is whatever is left of the 2048-token window once the prompt is in it.
    ``window`` is the window in force, or 0 when it is not a fixed limit.

    The ceiling is never raised above what was asked for: a task that needs less
    still gets less. It is only ever reduced, because asking a server for more
    output than its window holds produces a silently truncated document, which
    is the one failure this pipeline is built to make impossible.
    """
    ceiling = max(1, int(requested))
    window = _ollama_usable_window(base_url)

    if not window:
        return ceiling, 0

    needed = _estimate_prompt_tokens(prompt)
    return max(1, min(ceiling, window - needed)), window


def _provider_default_headers(
    spec: ProviderSpec | None,
    has_key: bool,
) -> dict[str, str]:
    """
    Build the extra HTTP headers one provider needs.

    Two cases matter. A local gateway with no configured credential must not
    receive an ``Authorization: Bearer`` header at all: OpenAI's client is
    constructed with a placeholder key so it will not refuse to start, and
    sending a bogus header to a gateway that does not require one can make it
    answer 401 instead of serving the request. A gateway that does have a real
    key gets the standard header the SDK builds itself.
    """
    if not spec or spec.local or not spec.requires_key or not has_key:
        return {}

    headers: dict[str, str] = {}

    for item in spec.extra_headers_env:
        raw = _env(item)

        if not raw or "=" not in raw:
            continue

        name, _, value = raw.partition("=")
        name = name.strip()
        value = value.strip()

        if name and value:
            headers[name] = value

    return headers


#: Used when a provider name reaches the request builders without a registry
#: entry. The router rejects unknown providers before this point, so this only
#: has to be a valid, conservative record rather than a correct one.
_FALLBACK_SPEC = ProviderSpec(
    name="unknown",
    protocol=PROTOCOL_OPENAI_COMPATIBLE,
    base_url_env="",
    default_base_url="",
    model_env="",
    default_model="",
    temperature=0.2,
    default_max_tokens=2048,
    discoverable=False,
)


def _ollama_server_root() -> str:
    """
    Base URL of the Ollama server, without any API path.

    OLLAMA_BASE_URL may point at the OpenAI-compatible endpoint (.../v1) or the
    native one (.../api/generate), but the metadata endpoints used here
    (/api/show, /api/tags, /api/ps) always hang off the server root.
    """
    base = _get_ollama_base_url().rstrip("/")
    for suffix in ("/api/generate", "/v1", "/api"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _gpu_total_vram_mb() -> int | None:
    """Total VRAM in MiB, or None if it cannot be determined."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        # No nvidia-smi, no NVIDIA GPU, or it timed out. Not an error: the
        # caller falls back to a static ceiling.
        return None


def _ollama_model_kv_bytes_per_token(model: str) -> int | None:
    """
    KV-cache bytes per generated token for a model served by Ollama.

    Asked of Ollama itself via /api/show so the numbers are the real ones for
    the loaded model rather than a guess.
    """
    try:
        response = requests.post(
            f"{_ollama_server_root()}/api/show",
            json={"model": model},
            timeout=5,
        )
        if response.status_code != 200:
            return None
        info = (response.json() or {}).get("model_info") or {}
    except Exception:
        return None

    def find(suffix: str) -> int | None:
        for key, value in info.items():
            if key.endswith(suffix) and isinstance(value, int):
                return value
        return None

    layers = find("block_count")
    kv_heads = find("attention.head_count_kv")
    heads = find("attention.head_count")
    hidden = find("embedding_length")

    if not layers or not kv_heads or not heads or not hidden:
        return None

    head_dim = hidden // heads
    if head_dim <= 0:
        return None

    # K and V, fp16, per layer per token.
    return 2 * layers * kv_heads * head_dim * 2


def _ollama_weight_size_mb(model: str) -> int | None:
    """On-disk/loaded size of a model in MiB, as reported by Ollama."""
    try:
        base = _ollama_server_root()
        for endpoint in ("/api/tags", "/api/ps"):
            response = requests.get(f"{base}{endpoint}", timeout=5)
            if response.status_code != 200:
                continue
            for entry in (response.json() or {}).get("models") or []:
                name = str(entry.get("name") or "")
                if name == model or name.split(":")[0] == model.split(":")[0]:
                    size = entry.get("size")
                    if isinstance(size, (int, float)) and size > 0:
                        return int(size / (1024 * 1024))
    except Exception:
        return None
    return None


def _auto_context_ceiling(model: str) -> int | None:
    """
    Largest ladder rung this GPU can hold for ``model``.

    Returns None when the hardware or the model cannot be inspected, in which
    case the caller keeps the static default.
    """
    if model in _VRAM_BUDGET_CACHE:
        return _VRAM_BUDGET_CACHE[model]

    total_mb = _gpu_total_vram_mb()
    if not total_mb:
        return None

    weights_mb = _ollama_weight_size_mb(model)
    if not weights_mb:
        return None

    bytes_per_token = _ollama_model_kv_bytes_per_token(model) or _FALLBACK_KV_BYTES_PER_TOKEN
    headroom = max(0, _env_int("OLLAMA_VRAM_HEADROOM_MB", DEFAULT_VRAM_HEADROOM_MB))

    kv_budget_bytes = (total_mb - weights_mb - headroom) * 1024 * 1024
    if kv_budget_bytes <= 0:
        logger.warning(
            "Model %s (%.0f MiB) does not fit alongside the GPU's %.0f MiB "
            "of VRAM with a %.0f MiB headroom; falling back to the smallest "
            "context window.",
            model,
            weights_mb,
            total_mb,
            headroom,
        )
        return DEFAULT_OLLAMA_CONTEXT_LADDER[0]

    affordable = int(kv_budget_bytes // bytes_per_token)

    chosen = DEFAULT_OLLAMA_CONTEXT_LADDER[0]
    for rung in DEFAULT_OLLAMA_CONTEXT_LADDER:
        if rung <= affordable:
            chosen = rung

    logger.info(
        "Derived Ollama context ceiling for %s: %d tokens "
        "(GPU %.0f MiB, weights %.0f MiB, headroom %.0f MiB, "
        "%.1f KiB KV/token, affordable ~%d).",
        model,
        chosen,
        total_mb,
        weights_mb,
        headroom,
        bytes_per_token / 1024,
        affordable,
    )

    _VRAM_BUDGET_CACHE[model] = chosen
    return chosen


# Transport timeout for a single generation. The native Ollama path already
# used 180s; the OpenAI-compatible path used the SDK default of 600s, so the
# same generation could hang far longer depending on the endpoint.
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 180


def _ollama_request_timeout() -> int:
    """Timeout for one Ollama generation, honouring LLM_REQUEST_TIMEOUT_SECONDS."""
    return max(
        1,
        _env_int(
            "LLM_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
        ),
    )


# Slack added to the estimated prompt size before choosing a bucket: the
# chars/4 estimate undercounts for prose, and the chat template plus the
# assistant header add tokens that are not in the prompt string.
_PROMPT_TOKEN_ESTIMATE_DIVISOR = 4
_PROMPT_TOKEN_ESTIMATE_SAFETY = 1.15


def _estimate_prompt_tokens(prompt: str) -> int:
    """Rough token count for a prompt, used only for window sizing."""
    return max(
        1,
        int(len(prompt) / _PROMPT_TOKEN_ESTIMATE_DIVISOR * _PROMPT_TOKEN_ESTIMATE_SAFETY),
    )


def _ollama_num_ctx(
    prompt: str,
    max_output_tokens: int | None = None,
    model: str = "",
    base_url: str = "",
) -> int:
    """
    Choose the context window for one Ollama request.

    Returns a value from DEFAULT_OLLAMA_CONTEXT_LADDER large enough for the
    estimated prompt plus the requested output, capped by the smaller of
    ``OLLAMA_MAX_CONTEXT_TOKENS`` and whatever this GPU can actually hold for
    the model.

    The cap is what keeps a large prompt from pushing the card into CPU
    offload, which costs far more than the truncation it avoids. When
    ``base_url`` points at the OpenAI-compatible shim, its own hard prompt
    limit also applies and is treated as an additional ceiling, so the
    truncation is at least something the application reports rather than
    something Ollama does silently.
    """
    if max_output_tokens is None:
        max_output_tokens = _ollama_max_output_tokens()

    floor_rung = DEFAULT_OLLAMA_CONTEXT_LADDER[0]

    configured = max(
        floor_rung,
        _env_int(
            "OLLAMA_MAX_CONTEXT_TOKENS",
            DEFAULT_OLLAMA_MAX_CONTEXT_TOKENS,
        ),
    )

    # An explicit override wins, but never below the smallest rung.
    derived = _auto_context_ceiling(model) if model else None
    ceiling = min(configured, derived) if derived else configured

    needed = _estimate_prompt_tokens(prompt) + max_output_tokens

    chosen = floor_rung
    for bucket in DEFAULT_OLLAMA_CONTEXT_LADDER:
        if bucket >= needed:
            chosen = bucket
            break
    else:
        # Larger than every rung: use the ceiling if it allows, else the top
        # rung, and let the caller warn about the shortfall.
        chosen = ceiling

    chosen = max(1, min(chosen, ceiling))

    if base_url and not base_url.lower().endswith("/api/generate"):
        # The /v1 shim cannot carry more than its own limit whatever we ask
        # for, so never claim a window larger than it can honour.
        chosen = min(chosen, OPENAI_COMPAT_OLLAMA_PROMPT_TOKEN_LIMIT)

    return max(1, chosen)


def _warn_if_prompt_exceeds_context(
    prompt: str,
    num_ctx: int,
    *,
    provider: str,
    model: str,
    base_url: str = "",
) -> bool:
    """
    Log (and report) when a prompt cannot fit the chosen window.

    Ollama does not error in this case, it silently drops part of the prompt,
    so the only way to notice is to compare the estimate against the window.
    """
    needed = _estimate_prompt_tokens(prompt)
    fits = needed <= num_ctx

    if not fits:
        on_shim = bool(base_url) and not base_url.lower().endswith("/api/generate")
        if on_shim:
            logger.warning(
                "Ollama's OpenAI-compatible endpoint cannot carry this prompt: "
                "provider=%s model=%s estimated_prompt_tokens=%d but /v1 caps "
                "the prompt at ~%d tokens and ignores num_ctx, so part of the "
                "resume or job description will be dropped. Set "
                "OLLAMA_BASE_URL=http://localhost:11434/api/generate to use the "
                "native endpoint, which honours num_ctx.",
                provider,
                model,
                needed,
                OPENAI_COMPAT_OLLAMA_PROMPT_TOKEN_LIMIT,
            )
        else:
            logger.warning(
                "Prompt may exceed the Ollama context window and will be "
                "truncated: provider=%s model=%s estimated_prompt_tokens=%d "
                "num_ctx=%d. Ollama drops the overflow silently. Raise "
                "OLLAMA_MAX_CONTEXT_TOKENS if the GPU has VRAM headroom, or "
                "shorten the resume/job description.",
                provider,
                model,
                needed,
                num_ctx,
            )

    return fits


def _get_experiential_base_url() -> str:
    """
    Experiential is optional.

    If OPENAI_BASE_URL is commented out or empty, this function only gets
    called if Experiential is explicitly selected/configured.
    """
    raw = _env(
        "OPENAI_BASE_URL",
        DEFAULT_EXPERIENTIAL_BASE_URL,
    )

    return _get_remote_url(
        "OPENAI_BASE_URL",
        raw or DEFAULT_EXPERIENTIAL_BASE_URL,
    )


def _get_claude_base_url() -> str:
    return _get_remote_url(
        "ANTHROPIC_BASE_URL",
        DEFAULT_ANTHROPIC_BASE_URL,
    )


# ============================================================================
# Provider credentials
# ============================================================================

EXPERIENTIAL_API_KEY = _env("EXPLABS_API_KEY") or _env("EXPERIENTIAL_ORG_KEY")

GEMINI_API_KEY = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")

GROQ_API_KEY = _env("GROQ_API_KEY")

OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")

DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")

OPENAI_API_KEY = _env("OPENAI_API_KEY")

ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")


#: Environment variables that can supply each provider's credential, in
#: priority order.
#:
#: Resolved live rather than frozen at import. ``load_dotenv`` above populates
#: ``os.environ``, so a deployment that configures keys in .env sees no
#: behaviour change -- but a key rotated after start-up is now honoured, and a
#: test can vary provider configuration without reimporting the process.
#: Reading only an import-time snapshot meant routing decisions could not
#: reflect the environment the router was actually running in.
#:
#: Derived from the registry so a new provider cannot be added to the
#: dispatch path while being forgotten here.
_PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    name: spec.key_env for name, spec in PROVIDER_REGISTRY.items() if spec.key_env
}


def _live_api_key(provider: str) -> str:
    """Resolve a provider's credential from the current environment."""
    for name in _PROVIDER_KEY_ENV.get(canonical_provider(provider), ()):
        value = _env(name)
        if value:
            return value

    return ""


#: Environment variable that names each provider's model.
_PROVIDER_MODEL_ENV: dict[str, str] = {
    name: spec.model_env for name, spec in PROVIDER_REGISTRY.items()
}


# ============================================================================
# Models
# ============================================================================


def _default_model_for(name: str, spec: ProviderSpec) -> str:
    """
    Resolve a provider's model, preferring the environment.

    Read live rather than from a frozen dict so ``OLLAMA_MODEL`` and friends are
    honoured instead of whatever was configured at import time. Nothing else in
    the application hard-codes a model name, which is what lets a deployment
    switch between ``qwen3:8b``, ``qwen2.5:7b`` or anything else without a code
    change.
    """
    configured = _env(spec.model_env)
    if configured:
        return configured

    return spec.default_model


DEFAULT_MODELS: dict[str, str] = {
    name: _default_model_for(name, spec) for name, spec in PROVIDER_REGISTRY.items()
}


#: Every provider the router can dispatch to, in canonical order.
SUPPORTED_PROVIDERS: tuple[str, ...] = PROVIDER_ORDER

#: Providers that leave the machine. Ollama and OmniRoute are excluded because
#: both default to a loopback address.
ONLINE_PROVIDERS: tuple[str, ...] = online_providers()


# ============================================================================
# Routing modes
# ============================================================================
#
# LLM_MODE is the one control that answers "may this request leave the
# machine?". LLM_PROVIDER and LLM_ROUTE_MODE are the older, lower-level knobs
# and are still honoured, so no existing .env has to change.
#
#   auto    prefer Ollama, fall back to the online chain
#   ollama  local only; a failure is reported, never silently escalated
#   online  the configured online chain only
#
# The three modes are mapped onto the internal route vocabulary the candidate
# builder already understands:
#
#   auto   -> automatic
#   ollama -> local          (new; exactly one local candidate, no escalation)
#   online -> online         (new; the online chain with Ollama removed)

LLM_MODE_AUTO = "auto"
LLM_MODE_OLLAMA = "ollama"
LLM_MODE_ONLINE = "online"
LLM_MODE_LOCAL = "local"

SUPPORTED_LLM_MODES = (LLM_MODE_AUTO, LLM_MODE_OLLAMA, LLM_MODE_ONLINE)

_MODE_TO_ROUTE = {
    LLM_MODE_AUTO: "automatic",
    LLM_MODE_OLLAMA: LLM_MODE_LOCAL,
    LLM_MODE_ONLINE: LLM_MODE_ONLINE,
}

_ROUTE_TO_MODE = {
    "automatic": LLM_MODE_AUTO,
    LLM_MODE_LOCAL: LLM_MODE_OLLAMA,
    LLM_MODE_ONLINE: LLM_MODE_ONLINE,
}


# ============================================================================
# Task categories
# ============================================================================
#
# Tasks exist to make the routing decision better, not to restrict it. A caller
# that names a task gets a sensible provider order; a caller that names nothing
# gets "generic", which is today's behaviour. No task ever changes *whether* a
# fallback is allowed -- that is the mode's job -- so an unknown task name is
# simply treated as generic instead of raising.


@dataclass(frozen=True)
class TaskPolicy:
    """
    How one task category prefers to be served.

    local_first:
        Ollama is tried before any online provider. True for the routine CV
        operations: extraction, translation, bullet rewriting, tailoring.
    online_first:
        The online chain is tried first, because the task is the kind that
        benefits from stronger reasoning or a much longer context than a
        laptop GPU can hold. Reserved for full document generation and for
        the browser agent, whose prompt loop was already online-only.
    min_context_tokens:
        The rough prompt size above which this task should not be sent to
        Ollama at all in auto mode. 0 disables the check.
    max_output_tokens:
        Task-specific completion ceiling, when the task needs more (or less)
        room than OLLAMA_MAX_OUTPUT_TOKENS provides.
    """

    local_first: bool
    online_first: bool = False
    min_context_tokens: int = 0
    max_output_tokens: int | None = None
    description: str = ""


TASK_POLICIES: dict[str, TaskPolicy] = {
    # Cheap, deterministic extraction work. A local model is as good as a large
    # one here and the request stays private.
    "ats_analysis": TaskPolicy(
        local_first=True,
        description="Deterministic keyword and skill extraction.",
    ),
    "resume_translation": TaskPolicy(
        local_first=True,
        description="Bilingual resume translation.",
    ),
    "bullet_optimization": TaskPolicy(
        local_first=True,
        description="Rewriting individual achievement bullets.",
    ),
    "cv_tailoring": TaskPolicy(
        local_first=True,
        description="Targeting an existing CV to one job description.",
    ),
    "cv_format_recommendation": TaskPolicy(
        local_first=True,
        description="Recommending a layout and language.",
    ),
    # Full documents are the tasks where a stronger model actually pays off:
    # the output is long, the structure is global, and a hallucination costs a
    # whole CV. Ollama is still tried first so a private run stays possible,
    # and the online chain is the fallback.
    "full_cv_generation": TaskPolicy(
        local_first=True,
        min_context_tokens=24_000,
        max_output_tokens=4096,
        description="Long-form tailored CV or cover letter.",
    ),
    "cv_html_generation": TaskPolicy(
        local_first=True,
        max_output_tokens=4096,
        description="HTML CV payload.",
    ),
    # The browser agent's prompt loop was already online-only, and the
    # application has a dedicated failover chain for it. Keep it that way.
    "browser_agent": TaskPolicy(
        local_first=False,
        online_first=True,
        min_context_tokens=16_000,
        description="Browser automation decision loop.",
    ),
    "generic": TaskPolicy(
        local_first=True,
        description="Unclassified request; today's behaviour.",
    ),
}

#: Accepted aliases so a caller can be a little loose without breaking routing.
_TASK_ALIASES = {
    "ats": "ats_analysis",
    "analysis": "ats_analysis",
    "translation": "resume_translation",
    "translate": "resume_translation",
    "bullets": "bullet_optimization",
    "bullet_rewrite": "bullet_optimization",
    "optimize_bullets": "bullet_optimization",
    "tailoring": "cv_tailoring",
    "tailor": "cv_tailoring",
    "full_cv": "full_cv_generation",
    "latex_generation": "full_cv_generation",
    "cover_letter": "full_cv_generation",
    "html": "cv_html_generation",
    "latex_validation": "latex_validation",
    "latex": "full_cv_generation",
    "browser": "browser_agent",
    "job_agent": "browser_agent",
    "default": "generic",
    "": "generic",
}

#: ``latex_validation`` is a deterministic, local-only task: it never calls a
#: model, it only type-checks LaTeX. It is registered so an explicit caller is
#: never rejected as unknown.
TASK_POLICIES["latex_validation"] = TaskPolicy(
    local_first=True,
    description="Deterministic LaTeX checks; no model call.",
)

KNOWN_TASKS = tuple(sorted(TASK_POLICIES))


def normalize_task(task: str | None) -> str:
    """
    Map a caller-supplied task name onto a known category.

    Unknown names collapse to ``generic`` on purpose: a typo in a task label
    must never turn into a routing failure or, worse, into a wider set of
    providers than the caller asked for.
    """
    if not isinstance(task, str):
        return "generic"

    name = task.strip().lower().replace("-", "_").replace(" ", "_")

    if name in TASK_POLICIES:
        return name

    return _TASK_ALIASES.get(name, "generic")


def task_policy(task: str | None) -> TaskPolicy:
    """Return the routing policy for ``task``."""
    return TASK_POLICIES[normalize_task(task)]


# ============================================================================
# Experiential model catalog
# ============================================================================

EXPERIENTIAL_MODELS = (
    "gpt-6-astra",
    "gpt-5.6-luna",
    "deepseek-v4-flash",
    "qwen3.8-27b",
    "gemini-3.7-flash",
    "claude-fable-5",
    "gpt-4o",
    "nemotron-3-ultra-550b-a55b",
)


_MODEL_PRICING: dict[str, dict[str, float | None]] = {
    model: {
        "input": None,
        "output": None,
    }
    for model in EXPERIENTIAL_MODELS
}


# ============================================================================
# Logging
# ============================================================================

LOG_PATH = Path(
    _env(
        "LLM_PROCESSING_LOG",
        str(PROJECT_ROOT / "data" / "llm_processing.jsonl"),
    )
)

if not LOG_PATH.is_absolute():
    LOG_PATH = PROJECT_ROOT / LOG_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: Every environment variable whose value must be scrubbed from a diagnostic.
#: Derived from the provider table so a newly supported provider is redacted by
#: construction instead of by remembering to edit a second list.
_REDACTED_ENV_NAMES: tuple[str, ...] = tuple(
    sorted(
        {name for names in _PROVIDER_KEY_ENV.values() for name in names}
        | {"SERPAPI_KEY", "HF_TOKEN", "GMAIL_APP_PASSWORD", "API_KEY"}
    )
)


def _safe_error(exc: BaseException) -> str:
    """Return a short diagnostic with credentials and direct identifiers removed."""
    text = str(exc).strip() or exc.__class__.__name__
    sensitive_values = [
        value
        for name, value in ((name, os.getenv(name, "").strip()) for name in _REDACTED_ENV_NAMES)
        if value
    ]
    for value in sensitive_values:
        text = text.replace(value, "[REDACTED]")
    text = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+|api[_-]?key\s*[=:]\s*|"
        r"access[_-]?token\s*[=:]\s*|password\s*[=:]\s*|secret\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"://[^\s/@:]+:[^\s/@]+@", "://[REDACTED]@", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"(?<!\w)\+?\d[\d .()/-]{7,}\d(?!\w)", "[PHONE]", text)
    return text[:500]


def _write_log(event: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        from app.core.event_log import _write_record

        _write_record(LOG_PATH, event)

    except Exception:
        logger.exception("Unable to write LLM processing log")


# ============================================================================
# Output cleaning
# ============================================================================


def clean_llm_output(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    text = re.sub(
        r"^\s*```(?:latex|tex|html|markdown|json|text)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    conversational_prefixes = (
        r"^\s*here is the requested (?:resume|cv|cover letter|document)" r"\s*:?\s*",
        r"^\s*here is the (?:optimized|tailored|updated|generated)"
        r"(?: resume| cv| cover letter| document)?\s*:?\s*",
        r"^\s*certainly[!,]?\s*here is\s*",
        r"^\s*sure[!,]?\s*here is\s*",
    )

    for pattern in conversational_prefixes:
        text = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    return text.strip()


# ============================================================================
# Assistant text extraction
# ============================================================================

# Fields that have been observed to carry the model's chain-of-thought on
# OpenAI-compatible servers. Verified against the locally installed stack
# (ollama 0.34.4 + openai 1.109.1): Ollama returns qwen3's thinking under
# "reasoning". openai's ChatCompletionMessage does not declare that field but
# sets extra="allow", so it arrives in model_extra and is reachable with
# getattr(). Kept as an explicit allow-list so no field name is guessed at.
_REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "thinking",
)


class _NativeOllamaMessage:
    """
    Message-shaped view of Ollama's native ``/api/generate`` payload.

    The native API returns a flat dict (``response`` / ``thinking``) instead of
    an OpenAI-style message object. This adapter lets the same extraction and
    diagnostics helpers serve both Ollama paths.
    """

    __slots__ = ("content", "thinking")

    def __init__(self, content: str, thinking: str | None) -> None:
        self.content = content
        self.thinking = thinking


def _message_field(message: Any, name: str) -> Any:
    """
    Read one field from an assistant message, whichever shape it has.

    An OpenAI-compatible response reaches this code two ways: the OpenAI SDK
    hands over a model object, and the plain-HTTP path used for a local gateway
    hands over the parsed JSON body, where the message is a dict. ``getattr``
    alone silently yields ``None`` for the second shape, so a dict message would
    always read as an empty response.

    Both are supported here so the two transports cannot disagree about which
    field holds the answer.
    """
    if isinstance(message, Mapping):
        return message.get(name)

    return getattr(message, name, None)


def _observed_message_field_names(message: Any) -> list[str]:
    """Field names present on an assistant message, for diagnostics only."""
    names = [name for name in ("content", "refusal") if _message_field(message, name) is not None]
    names.extend(name for name in _REASONING_FIELDS if _message_field(message, name) is not None)

    extra: Any = (
        message.keys()
        if isinstance(message, Mapping)
        else (getattr(message, "model_extra", None) or {})
    )
    for name in extra:
        if name not in names:
            names.append(name)

    return names


def _is_truncated_finish_reason(finish_reason: Any) -> bool:
    """
    True when the provider stopped because the output ceiling was hit.

    OpenAI-compatible servers use "length" for this; some report
    "max_tokens"/"MAX_TOKENS". A truncated generation is incomplete by
    definition, so its text must not be presented as a finished answer.
    """
    if finish_reason is None:
        return False
    return str(finish_reason).strip().lower() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }


def extract_assistant_text(
    message: Any,
    finish_reason: Any = None,
) -> tuple[str, str]:
    """
    Return ``(text, source_field)`` for an assistant message.

    ``content`` is authoritative and is always preferred. It is only empty on
    OpenAI-compatible servers that route a reasoning model through a separate
    channel, which is what ollama does for qwen3.

    A reasoning channel is treated as the answer only when the provider
    reported that generation *completed* normally. When the response was
    truncated (``finish_reason="length"``) the reasoning is an unfinished
    internal monologue with no answer in it, so returning it would hand back
    chain-of-thought as if it were the model's output. In that case the caller
    gets an empty string and raises instead.

    ``finish_reason`` belongs to the choice object, not the message, so it is
    passed in by the caller.
    """
    content = _message_field(message, "content")
    if isinstance(content, str) and content.strip():
        return content, "content"

    if not _is_truncated_finish_reason(finish_reason):
        for name in _REASONING_FIELDS:
            value = _message_field(message, name)
            if isinstance(value, str) and value.strip():
                return value, name

    return "", ""


def log_response_diagnostics(
    *,
    provider: str,
    model: str,
    finish_reason: Any,
    message: Any,
    usage: dict[str, Any],
    text: str,
    source_field: str,
) -> None:
    """
    Log why a generation succeeded or produced no text.

    Records response *shape* only: field names, counts and lengths. The prompt,
    the completion and any personal data in them are never logged.
    """
    field_names = _observed_message_field_names(message)
    truncated = _is_truncated_finish_reason(finish_reason)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    logger.info(
        "LLM response diagnostics: provider=%s model=%s finish_reason=%s "
        "message_fields=%s prompt_tokens=%s completion_tokens=%s "
        "content_found=%s source_field=%s output_chars=%d truncated=%s",
        provider,
        model,
        finish_reason,
        ",".join(field_names) or "<none>",
        prompt_tokens,
        completion_tokens,
        bool(text),
        source_field or "<none>",
        len(text),
        truncated,
    )

    if not text:
        logger.warning(
            "LLM produced no usable assistant text: provider=%s model=%s "
            "finish_reason=%s message_fields=%s prompt_tokens=%s "
            "completion_tokens=%s truncated=%s. Reasoning-channel text is "
            "ignored while truncated because it is unfinished internal "
            "reasoning, not an answer.",
            provider,
            model,
            finish_reason,
            ",".join(field_names) or "<none>",
            prompt_tokens,
            completion_tokens,
            truncated,
        )


# ============================================================================
# Usage helpers
# ============================================================================


def _empty_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def _coerce_nonnegative_int(value: Any) -> int:
    try:
        number = int(value or 0)
        return max(0, number)
    except (TypeError, ValueError):
        return 0


def _coerce_nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _extract_openai_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return _empty_usage()

    prompt_tokens = _coerce_nonnegative_int(
        getattr(
            usage,
            "prompt_tokens",
            getattr(usage, "input_tokens", 0),
        )
    )

    completion_tokens = _coerce_nonnegative_int(
        getattr(
            usage,
            "completion_tokens",
            getattr(usage, "output_tokens", 0),
        )
    )

    total_tokens = _coerce_nonnegative_int(
        getattr(
            usage,
            "total_tokens",
            prompt_tokens + completion_tokens,
        )
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": 0.0,
    }


def _extract_anthropic_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return _empty_usage()

    prompt_tokens = _coerce_nonnegative_int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
    )

    completion_tokens = _coerce_nonnegative_int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": 0.0,
    }


def _estimate_cost_usd(
    model: str,
    usage: dict[str, Any],
) -> float:
    pricing = _MODEL_PRICING.get(model)

    if not pricing:
        return 0.0

    input_price = pricing.get("input")
    output_price = pricing.get("output")

    if input_price is None or output_price is None:
        return 0.0

    prompt_tokens = usage.get(
        "prompt_tokens",
        0,
    )

    completion_tokens = usage.get(
        "completion_tokens",
        0,
    )

    cost = prompt_tokens / 1_000_000 * float(input_price) + completion_tokens / 1_000_000 * float(
        output_price
    )

    return round(cost, 8)


# ============================================================================
# Error classification
# ============================================================================

#: HTTP status codes are read off the exception rather than off the message
#: text, because they are the one unambiguous signal every SDK surfaces.


def _extract_status_code(exc: BaseException) -> int | None:
    for attribute in (
        "status_code",
        "status",
        "http_status",
    ):
        value = getattr(exc, attribute, None)

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)

    if response is not None:
        value = getattr(response, "status_code", None)

        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    return None


#: Stable error categories. Every failure is bucketed before the router decides
#: what to do with it, so "retry the same provider" and "move to the next one"
#: are separate questions instead of one boolean.
LLM_ERROR_TIMEOUT = "timeout"
LLM_ERROR_CONNECTION = "connection"
LLM_ERROR_AUTH = "authentication"
LLM_ERROR_RATE_LIMIT = "rate_limit"
LLM_ERROR_CREDITS = "insufficient_credits"
LLM_ERROR_MODEL_UNAVAILABLE = "model_unavailable"
LLM_ERROR_CONTEXT_TOO_LARGE = "context_too_large"
LLM_ERROR_MALFORMED = "malformed_response"
LLM_ERROR_SERVER = "server_error"
LLM_ERROR_CONFIGURATION = "configuration"
LLM_ERROR_UNKNOWN = "unknown"


#: Order matters: the first matching pattern wins.
#:
#: "insufficient_credits" precedes "rate_limit" because a provider that says the
#: balance is exhausted should stop rather than back off -- but its markers are
#: phrases that *state* an exhausted balance, not words that merely appear near
#: a billing link. Both precede the generic status-code checks, and the status
#: code is consulted ahead of this table by ``classify_llm_error`` whenever the
#: exception carries one.
_ERROR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        LLM_ERROR_CREDITS,
        (
            # Each of these states that the balance is the problem, rather than
            # mentioning billing in passing. "billing" on its own used to be here
            # and it matched the upgrade link in Groq's rate-limit body, so a
            # retryable 429 was classified as an exhausted balance: not retried,
            # and reported with the wrong remedy.
            "insufficient_credits",
            "insufficient credits",
            "insufficient_quota",
            "insufficient quota",
            "insufficient balance",
            "insufficient funds",
            "credit balance is too low",
            "credits exhausted",
            "exceeded your current quota",
            "quota exceeded for",
            "payment required",
            "payment_required",
            # A 402 is authoritative from the status code, which is consulted
            # before this list. As a bare substring it matched request ids and
            # token counts, which is a worse source of truth than the status.
        ),
    ),
    (
        LLM_ERROR_TIMEOUT,
        (
            "timeout",
            "timed out",
            "read timed out",
            "deadline exceeded",
            "operation timed out",
        ),
    ),
    (
        LLM_ERROR_CONNECTION,
        (
            "connection refused",
            "connection reset",
            "connection aborted",
            "connection error",
            "remote protocol error",
            "name or service not known",
            "nodename nor servname",
            "failed to establish",
            "econnrefused",
            "max retries exceeded",
            "new connection error",
        ),
    ),
    (
        LLM_ERROR_AUTH,
        (
            "unauthorized",
            "authentication",
            "invalid api key",
            "invalid_api_key",
            "incorrect api key",
            "forbidden",
            "permission denied",
            "api key not valid",
            "401",
            "403",
        ),
    ),
    (
        LLM_ERROR_RATE_LIMIT,
        (
            "rate limit",
            "rate_limit",
            "rate_limit_exceeded",
            "too many requests",
            "quota exceeded",
            "quota_exceeded",
            "resource exhausted",
            "resource_exhausted",
            "try again later",
            "please try again in",
            "tokens per minute",
            "requests per minute",
            "tpm",
            "rpm",
            "429",
        ),
    ),
    (
        LLM_ERROR_MODEL_UNAVAILABLE,
        (
            "model not found",
            "pull model",
            "model does not exist",
            "no such model",
            "model not available",
            "unknown model",
            "is not a valid model",
            "no such host",
            "not_found_error",
            "model_unavailable",
        ),
    ),
    (
        LLM_ERROR_CONTEXT_TOO_LARGE,
        (
            "context length",
            "context_length",
            "context window",
            "maximum context",
            "too many tokens",
            "reduce the length",
            "prompt is too long",
            "request too large",
            "string too long",
            "input is too long",
        ),
    ),
    (
        LLM_ERROR_MALFORMED,
        (
            "empty content",
            "returned no choices",
            "expecting value",
            "json",
            "unterminated string",
            "invalid control character",
        ),
    ),
    (
        LLM_ERROR_SERVER,
        (
            "internal server error",
            "server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "temporarily unavailable",
            "overloaded",
            "upstream",
            "500",
            "502",
            "503",
            "504",
        ),
    ),
    (
        LLM_ERROR_CONFIGURATION,
        (
            "is not configured",
            "missing api key",
            "provider url",
            "url must be",
            "unsafe url",
            "unsafe llm",
            "invalid url",
            "unsupported llm provider",
            "unsupported llm route",
        ),
    ),
)

_STATUS_CATEGORIES: dict[int, str] = {
    400: LLM_ERROR_MALFORMED,
    401: LLM_ERROR_AUTH,
    402: LLM_ERROR_CREDITS,
    403: LLM_ERROR_AUTH,
    404: LLM_ERROR_MODEL_UNAVAILABLE,
    408: LLM_ERROR_TIMEOUT,
    413: LLM_ERROR_CONTEXT_TOO_LARGE,
    422: LLM_ERROR_MALFORMED,
    425: LLM_ERROR_SERVER,
    429: LLM_ERROR_RATE_LIMIT,
    500: LLM_ERROR_SERVER,
    502: LLM_ERROR_SERVER,
    503: LLM_ERROR_SERVER,
    504: LLM_ERROR_TIMEOUT,
}

#: Categories whose message may override the generic status-code mapping above,
#: because a body that states a more specific cause is trusted over the code.
_STATUS_OVERRIDE_CATEGORIES = frozenset(
    {
        LLM_ERROR_CREDITS,
        LLM_ERROR_AUTH,
        LLM_ERROR_CONTEXT_TOO_LARGE,
        LLM_ERROR_MODEL_UNAVAILABLE,
    }
)

#: Where ``model_unavailable`` is allowed to override the status code.
#:
#: A retired or misspelled model id arrives as a *bad request*, not as a missing
#: resource: OpenRouter answers ``400 ... 'nvidia/nemotron-3-ultra:free' is not
#: a valid model ID`` for an id it no longer serves, rather than a 404. Left to
#: the generic table that 400 read as ``malformed_response``, which was wrong
#: twice over: it blamed the prompt, and it hid the one condition that a
#: different model on the *same* provider fixes -- so the router reported a dead
#: end where a fallback was the whole answer.
#:
#: Scoped to these two codes because unscoped the markers would also claim a 502
#: whose body merely mentions "no such host", which is a DNS failure and not a
#: model problem at all.
_MODEL_UNAVAILABLE_STATUSES = frozenset({400, 422})


def _status_override_allowed(category: str, status: int) -> bool:
    """May ``category`` from the message body override the code's own meaning?"""
    if category not in _STATUS_OVERRIDE_CATEGORIES:
        return False

    if category == LLM_ERROR_MODEL_UNAVAILABLE:
        return status in _MODEL_UNAVAILABLE_STATUSES

    return True


#: How long to wait at most for a rate-limited provider, in seconds.
#:
#: Read live from the environment so an operator with a small token bucket can
#: shorten it and one with a generous quota need not pay for it. The default is
#: short because the common case is a per-minute bucket that refills well inside
#: ten seconds, and a longer default would turn a rate limit into a hang.
LLM_RATE_LIMIT_BACKOFF_SECONDS = 10.0

#: Phrases providers use to say when their window reopens. Deliberately narrow:
#: these must state a duration, because a bare "try again" with no number gives
#: nothing to wait for.
_RETRY_AFTER_PHRASES = (
    re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE),
    re.compile(r"retry in\s+([\d.]+)\s*s", re.IGNORECASE),
    re.compile(r"retry_after[\"']?\s*[:=]\s*[\"']?([\d.]+)", re.IGNORECASE),
    re.compile(r"try again in\s+([\d.]+)\s*ms", re.IGNORECASE),
)


def rate_limit_retry_after(exc: BaseException) -> float | None:
    """
    How long the provider asked us to wait, in seconds, or ``None``.

    Three sources, in order of authority:

    1. an explicit ``retry_after_seconds`` attribute, for a client that surfaces
       the parsed value;
    2. an HTTP ``Retry-After`` header, which is seconds or an HTTP date;
    3. a duration in the message body, which is what an OpenAI-compatible error
       envelope actually carries.

    Only the ``rate_limit`` category is worth asking. Returns ``None`` rather
    than a guess when the provider says nothing useful, so the caller can fall
    back to its own short backoff instead of inventing a delay.
    """
    if classify_llm_error(exc) != LLM_ERROR_RATE_LIMIT:
        return None

    for attribute in ("retry_after_seconds", "retry_after", "retry_delay"):
        value = getattr(exc, attribute, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if headers is not None:
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:
            raw = None
        if raw:
            text = str(raw).strip()
            if text.isdigit():
                return float(text)
            # An HTTP-date form; the caller only needs an upper bound, and the
            # wall clock is close enough for a rate limit.
            try:
                from datetime import datetime, timezone
                from email.utils import parsedate_to_datetime

                when = parsedate_to_datetime(text)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delta = (when - datetime.now(timezone.utc)).total_seconds()
                return max(0.0, delta)
            except Exception:
                pass

    message = _safe_error(exc)

    for pattern in _RETRY_AFTER_PHRASES:
        match = pattern.search(message)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        # The millisecond phrasing matched first on "in 5.895s" too, so a value
        # under one is read as seconds rather than silently scaled: providers
        # state seconds far more often, and a sub-second wait is the fallback's
        # job anyway.
        return value if value > 0 else None

    return None


def classify_llm_error(
    exc: BaseException,
    provider: str | None = None,
) -> str:
    """
    Bucket a provider failure into a stable category.

    The HTTP status is consulted first because it is unambiguous; the message
    text is the fallback, since several providers report a rate limit or an
    exhausted credit balance with a 200-shaped body and only a string to go on.
    """
    status = _extract_status_code(exc)
    message = _safe_error(exc).casefold()

    if status in _STATUS_CATEGORIES:
        # A 429 that is really about credits must not be reported as a plain
        # rate limit, because the two need different handling: one backs off,
        # the other stops immediately.
        for category, markers in _ERROR_PATTERNS:
            if any(marker in message for marker in markers):
                if _status_override_allowed(category, status):
                    return category
                break

        return _STATUS_CATEGORIES[status]

    for category, markers in _ERROR_PATTERNS:
        if any(marker in message for marker in markers):
            return category

    if status is not None and status >= 500:
        return LLM_ERROR_SERVER

    if status == 429:
        return LLM_ERROR_RATE_LIMIT

    # A missing local model is a local configuration problem, and in auto mode
    # it is exactly the case where moving to a cloud provider is the answer.
    if provider == "ollama" and any(
        marker in message
        for marker in (
            "model not found",
            "pull model",
            "model does not exist",
            "no such model",
        )
    ):
        return LLM_ERROR_MODEL_UNAVAILABLE

    return LLM_ERROR_UNKNOWN


#: Categories worth another attempt against the *same* provider. Everything else
#: is deterministic for that provider and would only multiply latency -- the
#: reason a 429 insufficient_credits from Experiential Labs used to trigger a
#: fresh request every time.
_RETRYABLE_CATEGORIES = frozenset(
    {
        LLM_ERROR_TIMEOUT,
        LLM_ERROR_CONNECTION,
        LLM_ERROR_RATE_LIMIT,
        LLM_ERROR_SERVER,
        LLM_ERROR_MODEL_UNAVAILABLE,
    }
)

#: Categories that mean "this provider cannot serve this request, ever, for
#: this configuration". They end the current attempt and move to the next
#: provider rather than raising the retry count.
_TERMINAL_CATEGORIES = frozenset(
    {
        LLM_ERROR_CREDITS,
        LLM_ERROR_AUTH,
        LLM_ERROR_CONFIGURATION,
        LLM_ERROR_MALFORMED,
        LLM_ERROR_CONTEXT_TOO_LARGE,
    }
)


#: Categories where re-sending the *identical* request could plausibly succeed.
#:
#: Deliberately narrower than ``_RETRYABLE_CATEGORIES``. The difference is
#: ``model_unavailable``: a model that is not installed does not become installed
#: because the same prompt is sent again, but it is precisely the case where a
#: *different* provider is the right answer. That belongs to the router's
#: fallback decision, not to a retry loop.
_SAME_REQUEST_RETRYABLE_CATEGORIES = frozenset(
    {"timeout", "server_error", "rate_limit", "connection"}
)


def _is_retryable_error(
    exc: BaseException,
    provider: str | None = None,
) -> bool:
    """
    Report whether retrying the *same* request could plausibly succeed.

    This is the narrower of the two questions the router asks. It drives the
    optimizer's retry loop, where every attempt is an identical prompt to an
    identical provider, so a condition that a different provider or a
    differently-sized budget would fix does not belong here -- re-sending it
    cannot help, and each attempt costs the full generation time.

    The wider question -- might another provider succeed? -- is answered by
    ``classify_llm_error`` against ``_RETRYABLE_CATEGORIES`` where the router
    chooses a fallback, and that set does include ``model_unavailable``.
    """
    status = _extract_status_code(exc)

    category = classify_llm_error(exc, provider)

    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        # 429 needs the message to tell a rate limit from an empty balance.
        return category in _SAME_REQUEST_RETRYABLE_CATEGORIES

    if status in {400, 401, 402, 403, 404, 413, 422}:
        return False

    if category in _TERMINAL_CATEGORIES:
        return False

    if status is not None and status >= 500:
        return True

    return category in _SAME_REQUEST_RETRYABLE_CATEGORIES


def _is_configuration_error(
    exc: BaseException,
) -> bool:
    message = _safe_error(exc).lower()

    terms = (
        "api key",
        "authentication",
        "unauthorized",
        "forbidden",
        "permission",
        "invalid api key",
        "invalid model",
        "unsafe llm",
        "unsafe url",
        "url must be",
        "provider url",
        "provider is not configured",
    )

    return any(term in message for term in terms)


# ============================================================================
# LLM service
# ============================================================================

#: Whether a failed provider may be replaced by the next candidate.
#:
#: Defaults to True because that is the behaviour the router has always had and
#: the reason ``LLM_MODE=auto`` exists. A deployment that wants exactly one
#: provider regardless of what goes wrong can set this to false and combine it
#: with ``LLM_MODE=direct``; that combination fails loudly on the first error
#: instead of quietly trying someone else's inference service.
LLM_FALLBACK_ENABLED = True

#: Hard ceiling on how many (provider, model) candidates one request may try.
#:
#: The candidate list is already finite, but its length depends on how many
#: providers happen to be configured and on the per-provider model lists, which
#: are environment driven. This bounds the request count independently of that.
#: One candidate is always tried even if the value is 0, so a misconfiguration
#: cannot make a request impossible.
LLM_MAX_PROVIDER_ATTEMPTS = 4


def _not_configured_detail(spec: ProviderSpec, canonical: str) -> str:
    """Explain, in one line, why a provider is not usable yet."""
    if not spec.key_env:
        return f"{canonical} needs no credential; check {spec.model_env}."
    return f"{canonical} is not configured. " f"Set {' or '.join(spec.key_env)}."


def _model_row(
    provider: str,
    model_id: str,
    *,
    source: str,
    display_name: str | None = None,
    context_length: Any = None,
    owned_by: str | None = None,
    size_bytes: Any = None,
    families: Any = None,
    missing_from_discovery: bool = False,
) -> dict[str, Any]:
    """
    One row of a model listing.

    Only fields the provider actually reported are included. No capability is
    inferred: an unstated context length stays ``None`` rather than being
    guessed from the model name, because guessing is how a routing decision
    starts failing silently.
    """
    row: dict[str, Any] = {
        "provider": provider,
        "model": model_id,
        "id": model_id,
        "source": source,
    }

    if display_name:
        row["display_name"] = str(display_name)
    if isinstance(context_length, int) and context_length > 0:
        row["context_length"] = context_length
    if owned_by:
        row["owned_by"] = str(owned_by)
    if isinstance(size_bytes, int) and size_bytes > 0:
        row["size_bytes"] = size_bytes
    if families:
        row["families"] = list(families)
    if missing_from_discovery:
        row["missing_from_discovery"] = True

    return row


class LLMService:
    """
    Central LLM routing service.

    ``LLM_MODE`` decides whether a request may leave the machine:

        auto    prefer Ollama, fall back to the online chain
        ollama  local only; a failure is reported, never escalated
        online  the online chain only

    The older ``LLM_PROVIDER`` / ``LLM_ROUTE_MODE`` pair still works and is read
    when ``LLM_MODE`` is unset, so no existing .env has to change.

    Every caller passes a ``task`` so the router can prefer the right kind of
    provider, but callers do not pick providers themselves: ``provider`` and
    ``route_mode`` stay optional and default to the environment.

    Providers never secretly call another provider. All fallback decisions
    happen here.
    """

    DEFAULT_MODELS = DEFAULT_MODELS
    SUPPORTED_PROVIDERS = SUPPORTED_PROVIDERS

    _last_usage_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
        "llm_last_usage",
        default=None,
    )

    #: Retained copy of the most recent usage. generate() consumes the
    #: contextvar, so without this a caller reporting on a finished request --
    #: the smoke test, an endpoint -- would always see zeros.
    _last_usage_retained: dict[str, Any] = {}

    _model_latencies: dict[tuple[str, str], float] = {
        ("ollama", "glm-4.7-flash"): 5000.0,
        ("gemini", "gemini-2.5-flash"): 600.0,
        ("groq", "openai/gpt-oss-120b"): 500.0,
        ("openrouter", "openrouter/free"): 3000.0,
        ("deepseek", "deepseek-chat"): 1100.0,
        ("openai", "gpt-4o-mini"): 900.0,
        ("claude", "claude-sonnet-4-20250514"): 1200.0,
    }

    _exhausted_models: dict[
        tuple[str, str],
        datetime,
    ] = {}

    _state_lock = threading.Lock()

    _cooldown_seconds = 600

    # ---------------------------------------------------------------------
    # Environment / routing
    # ---------------------------------------------------------------------

    @classmethod
    def _default_provider(cls) -> str:
        return (
            _env(
                "LLM_PROVIDER",
                "ollama",
            ).lower()
            or "ollama"
        )

    @classmethod
    def _default_mode(cls) -> str:
        """
        Resolve ``LLM_MODE``, falling back to the legacy route-mode variables.

        ``LLM_MODE`` wins when it is set, because it is the only variable that
        distinguishes "local only" from "local first". A deployment that had
        ``LLM_ROUTE_MODE=direct`` keeps exactly that behaviour, and one that had
        ``automatic`` keeps local-first-with-fallback.
        """
        configured = _env("LLM_MODE").lower()

        if configured:
            aliases = {
                "auto": LLM_MODE_AUTO,
                "automatic": LLM_MODE_AUTO,
                "dynamic": LLM_MODE_AUTO,
                "local": LLM_MODE_OLLAMA,
                "ollama": LLM_MODE_OLLAMA,
                "offline": LLM_MODE_OLLAMA,
                "private": LLM_MODE_OLLAMA,
                "online": LLM_MODE_ONLINE,
                "cloud": LLM_MODE_ONLINE,
                "remote": LLM_MODE_ONLINE,
            }
            return aliases.get(configured, LLM_MODE_AUTO)

        legacy = cls._default_route_mode()

        if legacy == "direct":
            # `direct` means exactly one provider, and the only provider that
            # is configured by default is Ollama, so the legacy setting is the
            # local-only mode spelled the old way.
            return LLM_MODE_OLLAMA if cls._default_provider() == "ollama" else LLM_MODE_AUTO

        if legacy in _ROUTE_TO_MODE:
            return _ROUTE_TO_MODE[legacy]

        return LLM_MODE_AUTO

    @classmethod
    def _mode_to_route(cls, mode: str) -> str:
        """Translate an LLM_MODE value into the internal route vocabulary."""
        return _MODE_TO_ROUTE.get(mode, "automatic")

    @classmethod
    def route_info(cls) -> dict[str, Any]:
        """
        Describe the effective routing configuration.

        Exposed through the existing status endpoints so the active mode is
        visible without reading .env. It contains no secrets: provider names,
        model names and booleans only.
        """
        mode = cls._default_mode()
        preferred = cls._default_provider()

        # Resolved live, not from the import-time snapshot. This endpoint exists
        # so an operator can confirm what a request would actually use, so it has
        # to read the same values the router reads rather than a frozen copy.
        try:
            preferred_model = cls.get_default_model(preferred)
        except Exception:
            preferred_model = ""

        return {
            "mode": mode,
            "provider": preferred,
            "model": preferred_model,
            "preferred_provider": preferred,
            "legacy_route_mode": cls._default_route_mode(),
            "ollama_model": cls.get_default_model("ollama"),
            "online_primary": _env("ONLINE_PRIMARY_PROVIDER"),
            "online_fallback": _env("ONLINE_FALLBACK_PROVIDER"),
            "allows_cloud_fallback": mode != LLM_MODE_OLLAMA,
            "fallback_enabled": _env_bool("LLM_FALLBACK_ENABLED", LLM_FALLBACK_ENABLED),
            "max_provider_attempts": max(
                1,
                _env_int(
                    "LLM_MAX_PROVIDER_ATTEMPTS",
                    LLM_MAX_PROVIDER_ATTEMPTS,
                ),
            ),
            "retries": _env_int("LLM_RETRIES", 0),
            "tasks": list(KNOWN_TASKS),
        }

    @classmethod
    def _default_route_mode(cls) -> str:
        """
        Read the legacy route-mode variables, and nothing else.

        ``LLM_MODE`` is deliberately *not* consulted here. A second mapping from
        mode to this vocabulary existed alongside :meth:`_default_mode`, and the
        two disagreed: that one folded ``online`` into ``automatic``, which
        includes Ollama in the candidate list and would have quietly sent a CV
        to a cloud provider in the one mode that promises it will not. Routing
        is resolved in exactly one place -- ``_default_mode`` ->
        ``_mode_to_route`` -- and this method reports only what the operator set.
        """
        value = _env(
            "LLM_ROUTE_MODE",
            _env(
                "LLM_ROUTING_MODE",
                "automatic",
            ),
        ).lower()

        aliases = {
            "auto": "automatic",
            "dynamic": "automatic",
            "automatic": "automatic",
            "direct": "direct",
            "experiential": "experiential",
            "gateway": "experiential",
        }

        return aliases.get(
            value,
            "automatic",
        )

    @classmethod
    def _provider_api_key(
        cls,
        provider: str,
    ) -> str:
        return _live_api_key(provider.lower())

    @classmethod
    def _provider_is_configured(
        cls,
        provider: str,
    ) -> bool:
        provider = provider.lower()

        if provider == "ollama":
            return bool(_get_ollama_base_url() and DEFAULT_MODELS.get("ollama"))

        spec = get_spec(provider)

        if spec is None:
            return False

        if not spec.requires_key:
            # A local gateway that needs no credential is usable as soon as its
            # model is named. Ollama is handled above because it also needs a
            # resolvable base URL.
            return bool(spec.default_model or _env(spec.model_env))

        if spec.key_env:
            return bool(cls._provider_api_key(provider))

        # No credential variable and no explicit requirement: usable as long as
        # a model is named.
        return bool(spec.default_model or _env(spec.model_env))

    @classmethod
    def provider_status(cls) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        for provider in SUPPORTED_PROVIDERS:
            configured = cls._provider_is_configured(provider)

            result[provider] = {
                "configured": configured,
                "model": DEFAULT_MODELS.get(provider),
            }

        return result

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @classmethod
    def list_models(
        cls,
        provider: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Discover the models a provider currently offers.

        Discovery is always optional. A provider that does not expose a listing,
        or that cannot be reached, still works: it just returns a single row
        built from its configured model so the caller can see what *will* be
        used. That is the whole point -- the application must never require
        discovery in order to run.

        Nothing about a model is assumed. Availability, free tiers and model
        identifiers all change, so the listing is read live and the configured
        model is always reported alongside it.
        """
        if provider is None:
            results: dict[str, Any] = {}
            for name in SUPPORTED_PROVIDERS:
                results[name] = cls.list_models(name, timeout=timeout)
            return {
                "status": "success",
                "providers": results,
            }

        canonical = canonical_provider(provider)

        if not canonical:
            return {
                "status": "unknown_provider",
                "provider": provider,
                "protocol": "",
                "local": False,
                "configured": False,
                "configured_model": "",
                "discoverable": False,
                "detail": (
                    f"Unknown provider {provider!r}. "
                    f"Known providers: {', '.join(SUPPORTED_PROVIDERS)}."
                ),
                "models": [],
            }

        spec = PROVIDER_REGISTRY[canonical]
        configured_model = cls.get_default_model(canonical)
        wait = timeout or _env_int("LLM_DISCOVERY_TIMEOUT_SECONDS", 10)

        base: dict[str, Any] = {
            "provider": canonical,
            "protocol": spec.protocol,
            "local": spec.local,
            "configured": cls._provider_is_configured(canonical),
            "configured_model": configured_model,
            "discoverable": spec.discoverable,
            "models": [],
            "status": "not_configured",
            "detail": "",
        }

        if not base["configured"]:
            base["detail"] = _not_configured_detail(spec, canonical)
            return base

        if not spec.discoverable:
            base["status"] = "manual_configuration"
            base["detail"] = (
                f"{canonical} does not expose a model listing here. "
                "Set "
                f"{spec.model_env} to choose a model."
            )
            base["models"] = (
                [_model_row(canonical, configured_model, source="configuration")]
                if configured_model
                else []
            )
            return base

        try:
            rows = cls._discover_models(spec, configured_model, wait)
        except Exception as exc:
            # Discovery must never be able to fail a request or a status call.
            base["status"] = "discovery_failed"
            base["detail"] = f"Could not list models: {_safe_error(exc)}"
            base["models"] = (
                [_model_row(canonical, configured_model, source="configuration")]
                if configured_model
                else []
            )
            return base

        base["models"] = rows
        base["status"] = "ok" if rows else "empty"
        if not rows:
            base["detail"] = (
                f"{canonical} returned no models. Set "
                f"{spec.model_env} to configure one manually."
            )
        return base

    @classmethod
    def _discover_models(
        cls,
        spec: ProviderSpec,
        configured_model: str,
        timeout: int,
    ) -> list[dict[str, Any]]:
        """
        Read a provider's model listing.

        Two shapes are handled: OpenAI's ``{"data": [{"id": ...}]}`` and
        Ollama's ``{"models": [{"name": ..., "size": ...}]}``. Nothing else is
        assumed, and an unrecognised shape yields no rows rather than a guess.
        """
        # OLLAMA_BASE_URL may point at .../api/generate, which is a request
        # endpoint, not a base to append to. _ollama_server_root strips the
        # API path so discovery asks the server root instead.
        base_url = _ollama_server_root() if spec.name == "ollama" else _provider_base_url(spec)
        url = f"{base_url.rstrip('/')}{spec.models_path}"
        headers = {"Accept": "application/json"}
        # A provider whose listing endpoint has its own required headers (the
        # Anthropic version header) declares them; everything else is OpenAI-shaped.
        headers.update(dict(spec.discovery_headers))
        key = _live_api_key(spec.name)
        if key:
            headers[spec.api_key_header] = f"{spec.api_key_prefix}{key}"

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        # OpenAI-compatible shape.
        for entry in payload.get("data") or []:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            rows.append(
                _model_row(
                    spec.name,
                    model_id,
                    source="discovery",
                    display_name=entry.get("name"),
                    context_length=entry.get("context_length") or entry.get("context_window"),
                    owned_by=entry.get("owned_by"),
                )
            )

        # Ollama shape.
        for entry in payload.get("models") or []:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("name") or entry.get("model") or "").strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            rows.append(
                _model_row(
                    spec.name,
                    model_id,
                    source="discovery",
                    size_bytes=entry.get("size"),
                    families=entry.get("details", {}).get("families")
                    if isinstance(entry.get("details"), dict)
                    else None,
                )
            )

        if not rows and payload.get("id"):
            # Some gateways answer a single-model request directly.
            rows.append(_model_row(spec.name, str(payload["id"]), source="discovery"))

        # The configured model is always listed, so the caller can see whether
        # discovery actually found it.
        if configured_model and configured_model not in seen:
            rows.insert(
                0,
                _model_row(
                    spec.name,
                    configured_model,
                    source="configuration",
                    missing_from_discovery=True,
                ),
            )

        return rows

    @classmethod
    def ollama_health(cls) -> dict[str, Any]:
        """
        Ask the local Ollama server what it can do.

        Kept as a separate, Ollama-specific probe because it reports more than a
        model list: whether the server is reachable at all, and -- when the
        configured model is not installed -- the exact ``ollama pull`` command
        that fixes it. A missing model otherwise surfaces much later as an
        opaque 404 after a long wait.

        Never raises: a health probe must not be able to fail a request.
        """
        spec = PROVIDER_REGISTRY["ollama"]
        model = cls.get_default_model("ollama")
        result: dict[str, Any] = {
            "configured": cls._provider_is_configured("ollama"),
            "base_url_configured": bool(_get_ollama_base_url()),
            "model": model,
            "reachable": False,
            "model_available": False,
            "available_models": [],
            "detail": "",
        }

        if not result["configured"]:
            result["detail"] = "Ollama is not configured (missing OLLAMA_BASE_URL)."
            return result

        try:
            response = requests.get(
                f"{_ollama_server_root()}{spec.models_path}",
                timeout=_env_int("LLM_DISCOVERY_TIMEOUT_SECONDS", 10),
            )
            if response.status_code != 200:
                result["detail"] = (
                    f"Ollama server returned HTTP {response.status_code} "
                    f"for {spec.models_path}."
                )
                return result

            result["reachable"] = True
            names = [
                str(entry.get("name") or "")
                for entry in (response.json() or {}).get("models") or []
            ]
            result["available_models"] = sorted(name for name in names if name)
        except Exception as exc:
            result["detail"] = f"Ollama server is not reachable: {_safe_error(exc)}"
            return result

        base_name = model.split(":")[0]
        result["model_available"] = any(
            name == model or name.split(":")[0] == base_name for name in result["available_models"]
        )

        if not result["model_available"]:
            result["detail"] = (
                f"Model {model!r} is not installed. "
                f"Run: ollama pull {model}. "
                f"Installed: {', '.join(result['available_models']) or 'none'}"
            )

        return result

    @classmethod
    def get_last_usage(cls) -> dict[str, Any]:
        """
        Token usage for the most recent generation.

        A retained copy rather than the request-scoped one: generate() consumes
        that internally, so a caller inspecting a completed request would
        otherwise always be handed an empty dict. It is scoped to the process,
        which is the right granularity for a smoke test or a status report.
        """
        with cls._state_lock:
            return dict(cls._last_usage_retained)

    @classmethod
    def classify_for_report(cls, exc: BaseException, provider: str) -> str:
        """
        Turn a failure into a short, redaction-safe line for an operator.

        Used by the smoke test and the reporting endpoints, which both need to
        explain a failure without printing the exception verbatim: an upstream
        error body can echo the request, and the request here is a prompt.
        """
        category = classify_llm_error(exc, provider)
        detail = _safe_error(exc)
        # Keep it to one line and well under any log-size concern.
        detail = " ".join(detail.split())[:180]
        return f"[{category}] {detail}"

    @classmethod
    def call_llm(
        cls,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        route_mode: str | None = None,
        **kwargs: Any,
    ) -> str:
        return cls.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            route_mode=route_mode,
            **kwargs,
        )

    @classmethod
    def get_default_model(
        cls,
        provider: str | None = None,
    ) -> str:
        """
        Resolve the configured model for a provider.

        Read live from the environment with the import-time snapshot as the
        fallback, so ``OLLAMA_MODEL`` is honoured rather than any model name
        baked into the code. Nothing else in the application hard-codes a
        model, which is what lets ``qwen3:8b``, ``glm-4.7-flash``,
        ``qwen3.6`` and ``llama3.3:70b`` all work through configuration alone.
        """
        provider = (provider or cls._default_provider()).lower()

        env_name = _PROVIDER_MODEL_ENV.get(provider)
        if env_name:
            live = _env(env_name)
            if live:
                return live

        return DEFAULT_MODELS.get(
            provider,
            DEFAULT_MODELS["ollama"],
        )

    # ---------------------------------------------------------------------
    # Provider candidates
    # ---------------------------------------------------------------------

    @classmethod
    def _online_provider_chain(cls) -> list[str]:
        """
        The configured online providers, best first.

        ``ONLINE_PRIMARY_PROVIDER`` and ``ONLINE_FALLBACK_PROVIDER`` name the
        intended order explicitly; anything else configured is appended so an
        existing deployment with only API keys set still has a chain. Ollama is
        never in this list, which is what makes ``LLM_MODE=online`` a genuine
        no-local-data guarantee rather than a preference.
        """
        ordered: list[str] = []

        for variable in (
            "ONLINE_PRIMARY_PROVIDER",
            "ONLINE_FALLBACK_PROVIDER",
        ):
            for name in _env(variable).split(","):
                provider = name.strip().lower()
                if (
                    provider in ONLINE_PROVIDERS
                    and provider not in ordered
                    and cls._provider_is_configured(provider)
                ):
                    ordered.append(provider)

        if _env("LLM_PROVIDER_CHAIN"):
            for provider in _env("LLM_PROVIDER_CHAIN").split(","):
                provider = provider.strip().lower()
                if (
                    provider in ONLINE_PROVIDERS
                    and provider not in ordered
                    and cls._provider_is_configured(provider)
                ):
                    ordered.append(provider)

        for provider in cls._configured_provider_chain():
            if provider in ONLINE_PROVIDERS and provider not in ordered:
                ordered.append(provider)

        return ordered

    @classmethod
    def _configured_provider_chain(cls) -> list[str]:
        configured_chain = _env("LLM_PROVIDER_CHAIN")

        if configured_chain:
            raw = configured_chain.split(",")
        else:
            raw = [
                "ollama",
                "gemini",
                "openrouter",
                "groq",
                "deepseek",
                "claude",
                "openai",
                "experiential",
            ]

        result: list[str] = []

        for provider in raw:
            provider = provider.strip().lower()

            if provider not in SUPPORTED_PROVIDERS:
                continue

            if provider in result:
                continue

            if not cls._provider_is_configured(provider):
                continue

            result.append(provider)

        return result

    @classmethod
    def _openrouter_models(cls) -> list[str]:
        result: list[str] = []

        values = [
            _env("OPENROUTER_MODEL"),
            _env("PRIMARY_MODEL"),
            _env("FALLBACK_MODEL_1"),
            _env("FALLBACK_MODEL_2"),
        ]

        for model in values:
            if not model:
                continue

            # Only use these values as OpenRouter models if they actually
            # look like OpenRouter model identifiers.
            if model.startswith("openrouter/") or "/" in model or model.endswith(":free"):
                if model not in result:
                    result.append(model)

        if not result:
            result.append("openrouter/free")

        return result

    @classmethod
    def _experiential_models(cls) -> list[str]:
        values: list[str] = []

        primary = _env("EXPERIENTIAL_MODEL")

        if primary:
            values.append(primary)

        fallback = _env("EXPERIENTIAL_FALLBACK_MODELS")

        if fallback:
            values.extend(item.strip() for item in fallback.split(",") if item.strip())

        if not values:
            values.append(DEFAULT_MODELS["experiential"])

        return list(dict.fromkeys(values))

    @classmethod
    def _default_models_for(cls, provider: str) -> list[str]:
        """Every model worth trying for ``provider``, best first."""
        provider = provider.lower()

        if provider == "openrouter":
            return cls._openrouter_models()

        if provider == "experiential":
            return cls._experiential_models()

        default = cls.get_default_model(provider)

        return [default] if default else []

    @classmethod
    def _ollama_fits_prompt(
        cls,
        prompt: str,
        task: str,
    ) -> tuple[bool, str]:
        """
        Can the local model actually hold this request?

        Returns ``(fits, reason)``. ``reason`` is a short loggable token such as
        ``context_limit``, which is what makes a routing decision explainable
        after the fact.
        """
        if not prompt:
            return True, ""

        policy = task_policy(task)
        needed = _estimate_prompt_tokens(prompt)

        if policy.min_context_tokens and needed >= policy.min_context_tokens:
            return False, "context_limit"

        model = cls.get_default_model("ollama")
        base_url = _get_ollama_base_url()
        requested = policy.max_output_tokens or _ollama_max_output_tokens()

        if _ollama_uses_compatible_shim(base_url):
            # The shim's window is fixed and holds prompt and completion
            # together, so the question is not "does the prompt fit" but "is
            # what is left worth sending". The two failures need different fixes
            # from the operator, so they are reported differently.
            window = _ollama_usable_window(base_url)

            if needed >= window:
                return False, "prompt_exceeds_shim_window"

            usable = window - needed

            if usable < _MIN_USABLE_OUTPUT_TOKENS:
                return False, "shim_window_leaves_no_room"

            if usable < requested:
                # Usable, but smaller than this task asked for. Worth sending:
                # the caller decides whether a shorter answer beats no answer.
                logger.info(
                    "Ollama shim window leaves %d output tokens for "
                    "task=%s (asked for %d, window %d, prompt ~%d). "
                    "Set OLLAMA_BASE_URL to the native /api/generate endpoint "
                    "for a larger window.",
                    usable,
                    task,
                    requested,
                    window,
                    needed,
                )

            return True, ""

        num_ctx = _ollama_num_ctx(prompt, requested, model, base_url)

        if needed + requested > num_ctx:
            return False, "context_limit"

        return True, ""

    @classmethod
    def _build_candidates(
        cls,
        provider: str | None,
        model: str | None,
        route_mode: str,
        task: str | None = None,
        prompt: str = "",
    ) -> list[tuple[str, str]]:
        """
        Build the ordered candidate list for one request.

        The order *is* the routing decision, so it is made here and nowhere
        else. Three things shape it:

        * the mode -- ``local`` yields exactly one local candidate and
          ``online`` never yields one, which is how the "never send the CV off
          the machine" guarantee is actually implemented;
        * the task -- an online-first task puts the cloud chain ahead of
          Ollama, and a task with a context floor skips Ollama when the prompt
          cannot fit in the local window;
        * the explicit provider -- always first when one is named, so an
          explicit request is honoured ahead of any task preference.
        """
        route_mode = route_mode.lower()
        task_name = normalize_task(task)
        policy = task_policy(task_name)

        preferred_provider = (provider or cls._default_provider()).lower()

        candidates: list[tuple[str, str]] = []

        def add(
            candidate_provider: str,
            candidate_model: str,
        ) -> None:
            pair = (
                candidate_provider.lower(),
                candidate_model.strip(),
            )

            if not pair[1]:
                return

            if pair not in candidates:
                candidates.append(pair)

        def add_provider(
            candidate_provider: str,
        ) -> None:
            # An explicit model only applies to the provider it was named for.
            requested = model if model and candidate_provider == preferred_provider else None

            if requested:
                add(candidate_provider, requested)
                return

            for candidate_model in cls._default_models_for(candidate_provider):
                add(candidate_provider, candidate_model)

        # --------------------------------------------------------------
        # Local-only mode: exactly one candidate, no escalation.
        # --------------------------------------------------------------

        if route_mode == LLM_MODE_LOCAL:
            if preferred_provider != "ollama" and provider:
                raise ValueError(
                    "LLM_MODE=ollama cannot be combined with " f"provider={preferred_provider}."
                )

            if not cls._provider_is_configured("ollama"):
                raise RuntimeError(
                    "LLM_MODE=ollama is set but Ollama is not configured. "
                    "Set OLLAMA_BASE_URL and OLLAMA_MODEL, or use "
                    "LLM_MODE=auto to allow online providers."
                )

            add(
                "ollama",
                model or cls.get_default_model("ollama"),
            )

            return candidates

        # --------------------------------------------------------------
        # Online-only mode: the configured cloud chain, never Ollama.
        # --------------------------------------------------------------

        if route_mode == LLM_MODE_ONLINE:
            online_chain = cls._online_provider_chain()

            if not online_chain:
                raise RuntimeError(
                    "LLM_MODE=online is set but no online provider is "
                    "configured. Set at least one of GEMINI_API_KEY, "
                    "OPENROUTER_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY, "
                    "OPENAI_API_KEY, ANTHROPIC_API_KEY or "
                    "EXPLABS_API_KEY, or use LLM_MODE=auto."
                )

            for candidate_provider in online_chain:
                add_provider(candidate_provider)

            return candidates

        # --------------------------------------------------------------
        # Explicit direct mode
        # --------------------------------------------------------------

        if route_mode == "direct":
            selected_provider = preferred_provider

            if selected_provider not in SUPPORTED_PROVIDERS:
                raise ValueError(f"Unsupported LLM provider: " f"{selected_provider}")

            if model:
                add(
                    selected_provider,
                    model,
                )
            else:
                # "Direct" pins the provider, not one model id. It used to build
                # a single candidate from the provider's default model, so a
                # retired id -- `nvidia/nemotron-3-ultra:free`, which OpenRouter
                # no longer serves -- answered 400 on the first request with
                # nowhere else to go, and every CV generation failed with it.
                # The provider's ordered list is what PRIMARY_MODEL and the
                # FALLBACK_MODEL_n chain are for, and the `experiential` branch
                # below already reads it this way; this route ignored it.
                for candidate_model in cls._default_models_for(selected_provider):
                    add(selected_provider, candidate_model)

            return candidates

        # --------------------------------------------------------------
        # Explicit Experiential mode
        # --------------------------------------------------------------

        if route_mode == "experiential":
            if not cls._provider_is_configured("experiential"):
                raise RuntimeError(
                    "Experiential provider is not configured. "
                    "Set EXPLABS_API_KEY or EXPERIENTIAL_ORG_KEY "
                    "and OPENAI_BASE_URL if you want to enable it."
                )

            models = [model] if model else cls._experiential_models()

            for experiential_model in models:
                if experiential_model:
                    add(
                        "experiential",
                        experiential_model,
                    )

            return candidates

        # --------------------------------------------------------------
        # Automatic mode: Ollama and the online chain, ordered by task.
        # --------------------------------------------------------------

        online_chain = cls._online_provider_chain()
        local_chain = ["ollama"] if cls._provider_is_configured("ollama") else []

        if local_chain and prompt:
            fits, reason = cls._ollama_fits_prompt(prompt, task_name)
            if not fits:
                # The local window cannot hold this request. Skipping Ollama
                # here is what stops the router from asking a model to reason
                # over a silently truncated prompt.
                logger.info(
                    "Skipping Ollama for task=%s: %s " "(estimated_prompt_tokens=%d, model=%s)",
                    task_name,
                    reason,
                    _estimate_prompt_tokens(prompt),
                    cls.get_default_model("ollama"),
                )
                local_chain = []

        if policy.online_first:
            groups = [online_chain, local_chain]
        else:
            groups = [local_chain, online_chain]

        for group in groups:
            for candidate_provider in group:
                add_provider(candidate_provider)

        return candidates

    # ---------------------------------------------------------------------
    # Exhaustion / adaptive latency
    # ---------------------------------------------------------------------

    @classmethod
    def _record_execution_time(
        cls,
        provider: str,
        model: str,
        duration_ms: float,
    ) -> None:
        key = (
            provider,
            model,
        )

        with cls._state_lock:
            old = cls._model_latencies.get(
                key,
                duration_ms,
            )

            cls._model_latencies[key] = old * 0.7 + duration_ms * 0.3

    @classmethod
    def _mark_exhausted(
        cls,
        provider: str,
        model: str,
        retry_after: float | None = None,
    ) -> None:
        """
        Take a provider out of rotation.

        ``retry_after`` is the duration the provider itself stated, in seconds,
        and is only honoured for a rate limit. A token bucket that reopens in six
        seconds must not cause a ten-minute blackout, which is what a flat
        cooldown did: the provider answered, the router declined to call it, and
        the emergency pass failed on a healthy provider.

        The value is clamped to the backoff ceiling and floored at a second, so a
        quoted duration can neither stall a request nor round to nothing. With no
        duration, the flat cooldown applies -- for a 500 or a connection reset
        there is nothing to read, and the previous default is the right one.
        """
        seconds = cls._cooldown_seconds

        if retry_after is not None and retry_after > 0:
            ceiling = _env_float("LLM_RATE_LIMIT_BACKOFF_SECONDS", LLM_RATE_LIMIT_BACKOFF_SECONDS)
            seconds = max(1.0, min(retry_after, max(ceiling, 1.0)))

        with cls._state_lock:
            cls._exhausted_models[(provider, model)] = datetime.now(timezone.utc) + timedelta(
                seconds=seconds
            )

    @classmethod
    def _is_exhausted(
        cls,
        provider: str,
        model: str,
    ) -> bool:
        key = (
            provider,
            model,
        )

        with cls._state_lock:
            until = cls._exhausted_models.get(key)

            if until is None:
                return False

            now = datetime.now(timezone.utc)

            if now >= until:
                cls._exhausted_models.pop(
                    key,
                    None,
                )
                return False

            return True

    @classmethod
    def _shortest_exhaustion_remaining(cls) -> float:
        """
        Seconds until the first cooldown expires, or ``inf`` when none is set.

        ``inf`` rather than a large number so the caller's cap decides: a caller
        that waits only when the answer is small will simply not wait here.
        """
        with cls._state_lock:
            if not cls._exhausted_models:
                return float("inf")
            soonest = min(cls._exhausted_models.values())

        remaining = (soonest - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, remaining)

    @classmethod
    def _select_candidates(
        cls,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        if not candidates:
            return []

        available = [
            candidate
            for candidate in candidates
            if not cls._is_exhausted(
                candidate[0],
                candidate[1],
            )
        ]

        # Preferred candidate remains first. The remaining candidates can
        # be ordered by observed latency.
        if len(available) <= 1:
            return available

        first = available[0]

        remaining = sorted(
            available[1:],
            key=lambda pair: cls._model_latencies.get(
                pair,
                10_000.0,
            ),
        )

        return [
            first,
            *remaining,
        ]

    # ---------------------------------------------------------------------
    # Usage state
    # ---------------------------------------------------------------------

    @classmethod
    def _set_last_usage(
        cls,
        usage: dict[str, Any],
    ) -> None:
        cls._last_usage_var.set(usage)

        with cls._state_lock:
            cls._last_usage_retained = dict(usage)

    @classmethod
    def _pop_last_usage(cls) -> dict[str, Any]:
        usage = cls._last_usage_var.get()

        cls._last_usage_var.set(None)

        return usage or _empty_usage()

    # ---------------------------------------------------------------------
    # Provider execution
    # ---------------------------------------------------------------------

    @classmethod
    def _execute_ollama(
        cls,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send one request to Ollama.

        Two endpoints are supported and the configured ``OLLAMA_BASE_URL``
        chooses between them, because both are legitimate and a user who points
        at one expects that to be honoured:

        ``/api/generate``
            Ollama's own endpoint. It is a flat response rather than an
            OpenAI shape, and it is the only one that honours ``num_ctx``, so a
            long posting is not silently truncated.
        anything else (``/v1``)
            The OpenAI-compatible shim, delegated to the shared transport.

        The native path is kept separate rather than folded into the shim
        because the response shape genuinely differs: token counts and the stop
        reason are top-level fields, and the reasoning channel is called
        ``thinking`` rather than ``reasoning``.
        """
        base_url = _get_ollama_base_url()

        if not base_url:
            raise RuntimeError("Ollama provider is not configured: missing OLLAMA_BASE_URL.")

        logger.info(
            "OLLAMA request: model=%s prompt_chars=%d estimated_prompt_tokens=%d",
            model,
            len(prompt),
            max(1, len(prompt) // 4),
        )

        if not base_url.lower().endswith("/api/generate"):
            return cls._execute_direct_openai_style(
                prompt=prompt,
                model=model,
                api_key="ollama",
                base_url=base_url,
                provider="ollama",
                max_tokens=max_tokens,
            )

        # A caller-supplied ceiling wins over the configured default: a whole CV
        # body needs more room than a keyword extraction, and the task already
        # declares that.
        max_output_tokens = max_tokens or _ollama_max_output_tokens()
        max_output_tokens, _window = _ollama_effective_ceiling(
            prompt, max_output_tokens, model, base_url
        )
        num_ctx = _ollama_num_ctx(prompt, max_output_tokens, model, base_url)
        _warn_if_prompt_exceeds_context(
            prompt, num_ctx, provider="ollama", model=model, base_url=base_url
        )

        generation_started = time.perf_counter()
        try:
            response = requests.post(
                base_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    # Without this, a reasoning model spends the entire output
                    # budget on its thinking channel and returns an empty
                    # response. Measured on qwen3:8b with a realistic CV
                    # prompt: with `think: false` the answer arrives in ~14s;
                    # without it the budget is consumed by reasoning and the
                    # call fails after ~35s.
                    "think": False,
                    "options": {
                        "temperature": 0.2,
                        # Same budget as the OpenAI-compatible path so
                        # switching OLLAMA_BASE_URL does not silently change how
                        # much output a request may produce.
                        "num_predict": max_output_tokens,
                        # Sized per request so a long resume is not silently
                        # truncated. This endpoint honours num_ctx, unlike the
                        # /v1 shim.
                        "num_ctx": num_ctx,
                    },
                },
                timeout=_ollama_request_timeout(),
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            generation_duration_ms = (time.perf_counter() - generation_started) * 1000
            logger.info(
                "OLLAMA native generation END: model=%s duration=%.0fms",
                model,
                generation_duration_ms,
            )

        content = payload.get("response", "") or ""
        prompt_tokens = _coerce_nonnegative_int(payload.get("prompt_eval_count", 0))
        completion_tokens = _coerce_nonnegative_int(payload.get("eval_count", 0))
        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": 0.0,
        }
        cls._set_last_usage(usage)

        logger.info(
            "OLLAMA native TOKENS: model=%s prompt_tokens=%d "
            "completion_tokens=%d total_tokens=%d",
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        # The native API reports the stop reason as done_reason and can carry a
        # separate thinking channel. Reuse the same diagnostics so both Ollama
        # paths are observable in the same way.
        done_reason = payload.get("done_reason")
        native_message = _NativeOllamaMessage(
            content=content,
            thinking=payload.get("thinking") or None,
        )

        text, source_field = extract_assistant_text(native_message, done_reason)

        log_response_diagnostics(
            provider="ollama",
            model=model,
            finish_reason=done_reason,
            message=native_message,
            usage=usage,
            text=text,
            source_field=source_field,
        )

        if not text:
            # Structure only: field names, lengths and counts. Never the model
            # output, which on this path is the candidate's CV text.
            raise RuntimeError(
                "Ollama returned empty content. "
                f"(finish_reason={done_reason} "
                f"message_fields="
                f"{','.join(_observed_message_field_names(native_message)) or 'none'} "
                f"thinking_chars={len(payload.get('thinking') or '')} "
                f"completion_tokens={completion_tokens} "
                f"max_tokens_requested={max_output_tokens})"
            )

        logger.info(
            "LLM generation CONTENT: provider=ollama model=%s "
            "prompt_chars=%d estimated_prompt_tokens=%d response_chars=%d "
            "completion_tokens=%d total_tokens=%d duration_ms=%.0f "
            "source_field=%s",
            model,
            len(prompt),
            max(1, len(prompt) // 4),
            len(text),
            completion_tokens,
            total_tokens,
            generation_duration_ms,
            source_field,
        )

        return clean_llm_output(text)

    @classmethod
    def _openai_compatible_body(
        cls,
        spec: ProviderSpec,
        provider: str,
        prompt: str,
        model: str,
        base_url: str,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """
        Build the JSON body for an OpenAI-compatible chat completion.

        Shared by the SDK path and the plain-HTTP path so the two cannot drift.
        Ollama's switches are applied here rather than inside the transport,
        because they are part of the request contract and not of how the bytes
        are sent.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": spec.temperature,
        }

        if provider == "ollama":
            ceiling, _window = _ollama_effective_ceiling(
                prompt,
                max_tokens or _ollama_max_output_tokens(),
                model,
                base_url,
            )
            num_ctx = _ollama_num_ctx(prompt, ceiling, model, base_url)

            body["max_tokens"] = ceiling
            body["options"] = {"num_ctx": num_ctx}
            # `think: false` is Ollama's own switch. On the /v1 shim it is
            # carried inside `options`, which the shim forwards; on the native
            # endpoint it is a top-level field. Both are sent so the request is
            # correct whichever endpoint answered.
            #
            # This is the switch that matters. Measured on qwen3:8b with a
            # realistic CV prompt and a 1200-token output budget: with the
            # switch, content is returned in 19s; without it, the model spends
            # the entire budget on its reasoning channel, returns
            # finish_reason="length" with an empty content field, and the call
            # is reported as a failure after ~35s.
            body["think"] = False
            # The OpenAI-shaped equivalent, understood by ollama >= 0.5 and
            # ignored rather than rejected by older servers.
            body["reasoning_effort"] = "none"

            _warn_if_prompt_exceeds_context(
                prompt, num_ctx, provider=provider, model=model, base_url=base_url
            )

            return body

        body["max_tokens"] = max_tokens or _provider_max_tokens(spec)
        return body

    @classmethod
    def _read_openai_compatible_response(
        cls,
        payload: Any,
        provider: str,
        model: str,
        *,
        prompt_chars: int = 0,
        estimated_prompt_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> str:
        """
        Read an OpenAI-compatible chat-completion response.

        ``payload`` may be an SDK model object or a plain dict, because the
        plain-HTTP path and the SDK path both end up here and must agree on
        which field holds the answer.

        A response whose ``content`` is empty is *not* automatically a failure.
        OpenAI-compatible servers route a reasoning model's answer to a
        separate channel, and ``extract_assistant_text`` already knows how to
        read that. It refuses a channel that was cut off mid-thought, which is
        the right call: half a chain-of-thought is not a CV.
        """
        choices = (
            payload.get("choices")
            if isinstance(payload, dict)
            else getattr(payload, "choices", None)
        )

        if not choices:
            raise RuntimeError(f"{provider} returned no choices.")

        choice = choices[0]
        message = (
            choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        )
        finish_reason = (
            choice.get("finish_reason")
            if isinstance(choice, dict)
            else getattr(choice, "finish_reason", None)
        )

        if isinstance(payload, dict):
            raw_usage = payload.get("usage") or {}
            usage = {
                "prompt_tokens": _coerce_nonnegative_int(raw_usage.get("prompt_tokens", 0)),
                "completion_tokens": _coerce_nonnegative_int(raw_usage.get("completion_tokens", 0)),
                "total_tokens": _coerce_nonnegative_int(raw_usage.get("total_tokens", 0)),
                "estimated_cost_usd": 0.0,
            }
        else:
            usage = _extract_openai_usage(payload)

        usage["estimated_cost_usd"] = _estimate_cost_usd(model, usage)
        cls._set_last_usage(usage)

        text, source_field = extract_assistant_text(message, finish_reason)

        log_response_diagnostics(
            provider=provider,
            model=model,
            finish_reason=finish_reason,
            message=message,
            usage=usage,
            text=text,
            source_field=source_field,
        )

        if not text:
            # Field availability is the only thing that makes this actionable,
            # and it is structural rather than content, so it is safe to log.
            fields = ",".join(_observed_message_field_names(message)) or "none"
            reasoning_chars = 0
            for name in _REASONING_FIELDS:
                value = _message_field(message, name)
                if isinstance(value, str) and value.strip():
                    reasoning_chars = max(reasoning_chars, len(value))

            truncated = _is_truncated_finish_reason(finish_reason)

            raise RuntimeError(
                f"{provider} returned empty content. "
                f"(finish_reason={finish_reason} "
                f"message_fields={fields} "
                f"reasoning_chars={reasoning_chars} "
                f"completion_tokens={usage.get('completion_tokens', 0)} "
                f"max_tokens_requested="
                f"{(usage.get('completion_tokens', 0) if truncated else '')})"
            )

        cleaned_content = clean_llm_output(text)

        if source_field != "content":
            logger.warning(
                "LLM answer recovered from %r because content was empty: " "provider=%s model=%s",
                source_field,
                provider,
                model,
            )

        logger.info(
            "LLM generation CONTENT: provider=%s model=%s "
            "prompt_chars=%d estimated_prompt_tokens=%d response_chars=%d "
            "completion_tokens=%d total_tokens=%d duration_ms=%.0f "
            "source_field=%s",
            provider,
            model,
            prompt_chars,
            estimated_prompt_tokens,
            len(cleaned_content),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            duration_ms,
            source_field,
        )

        return cleaned_content

    @classmethod
    def _execute_direct_openai_style(
        cls,
        prompt: str,
        model: str,
        api_key: str,
        base_url: str,
        provider: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send one OpenAI-compatible request through the OpenAI SDK.

        Used for providers that hold a credential. A local gateway that needs
        no authentication goes through :meth:`_execute_http_openai_style`
        instead, because the SDK insists on a non-empty API key and would put
        the placeholder in the Authorization header.
        """
        spec = get_spec(provider)

        if not api_key:
            raise RuntimeError(f"{provider} provider is not configured: missing API key.")

        if not base_url:
            raise RuntimeError(f"{provider} provider URL is empty.")

        if provider != "ollama" and not (spec and spec.local):
            base_url = _validate_remote_base_url(
                base_url,
                resolve_dns=True,
            )

        prompt_chars = len(prompt)
        estimated_prompt_tokens = max(1, prompt_chars // 4)

        logger.info(
            "LLM generation START: provider=%s model=%s prompt_chars=%d "
            "estimated_prompt_tokens=%d",
            provider,
            model,
            prompt_chars,
            estimated_prompt_tokens,
        )

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            # The SDK retries twice on its own by default. Those retries are
            # invisible to the caller's logging, ignore LLM_RETRIES, and fire
            # before _is_retryable_error() ever sees the error, so a single
            # doomed request could become three. Retries are handled one level
            # up, where they are logged and configurable.
            max_retries=0,
            # Also defaults to 600s, far longer than any caller waits. Cap it so
            # a hung socket surfaces as an error the retry logic can classify.
            timeout=(
                _provider_timeout_seconds(spec)
                if spec
                else _env_int(
                    "LLM_REQUEST_TIMEOUT_SECONDS",
                    DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
                )
            ),
        )

        request_kwargs = cls._openai_compatible_body(
            spec=spec or _FALLBACK_SPEC,
            provider=provider,
            prompt=prompt,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
        )

        # The SDK rejects unknown top-level keys. `reasoning_effort` is a real
        # Chat Completions field so it stays where it belongs; only Ollama's
        # own `think` and `options` need the passthrough body the server reads.
        passthrough = {}
        for key in ("think", "options"):
            if key in request_kwargs:
                passthrough[key] = request_kwargs.pop(key)
        if passthrough:
            request_kwargs["extra_body"] = passthrough

        generation_started = time.perf_counter()

        try:
            response = client.chat.completions.create(**request_kwargs)
        finally:
            generation_duration_ms = (time.perf_counter() - generation_started) * 1000
            logger.info(
                "LLM generation END: provider=%s model=%s duration=%.0fms",
                provider,
                model,
                generation_duration_ms,
            )

        logger.info(
            "LLM generation TOKENS: provider=%s model=%s prompt_tokens=%d "
            "completion_tokens=%d total_tokens=%d",
            provider,
            model,
            getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
            getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
            getattr(getattr(response, "usage", None), "total_tokens", 0) or 0,
        )

        return cls._read_openai_compatible_response(
            payload=response,
            provider=provider,
            model=model,
            prompt_chars=prompt_chars,
            estimated_prompt_tokens=estimated_prompt_tokens,
            duration_ms=generation_duration_ms,
        )

    @classmethod
    def _execute_http_openai_style(
        cls,
        prompt: str,
        model: str,
        base_url: str,
        provider: str,
        api_key: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """
        Send one OpenAI-compatible request over plain HTTP.

        This is the path for a local gateway that does not require
        authentication, such as OmniRoute started with no key configured. The
        OpenAI SDK cannot be used here: it requires a non-empty API key and
        always emits an Authorization header, so a placeholder string would be
        sent to a server that deliberately has no authentication -- which some
        gateways answer with 401.

        Request building and response reading are shared with the SDK path, so
        the two cannot disagree about which field holds the answer.
        """
        spec = get_spec(provider)

        if not base_url:
            raise RuntimeError(f"{provider} provider URL is empty.")

        if not (spec and spec.local):
            base_url = _validate_remote_base_url(
                base_url,
                resolve_dns=True,
            )

        prompt_chars = len(prompt)
        estimated_prompt_tokens = max(1, prompt_chars // 4)
        ceiling = max_tokens or (spec and _provider_max_tokens(spec)) or 2048

        logger.info(
            "LLM generation START: provider=%s model=%s prompt_chars=%d "
            "estimated_prompt_tokens=%d max_tokens=%d",
            provider,
            model,
            prompt_chars,
            estimated_prompt_tokens,
            ceiling,
        )

        body = cls._openai_compatible_body(
            spec=spec or _FALLBACK_SPEC,
            provider=provider,
            prompt=prompt,
            model=model,
            base_url=base_url,
            max_tokens=ceiling,
        )

        headers = {"Content-Type": "application/json"}

        # Only sent when a real key is configured, and in whatever form the
        # provider expects. A local gateway with no key gets no header at all.
        if api_key:
            headers[spec.api_key_header if spec else "authorization"] = (
                f"{spec.api_key_prefix if spec else 'Bearer '}{api_key}"
            )
        headers.update(_provider_default_headers(spec, bool(api_key)))

        generation_started = time.perf_counter()
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=_provider_timeout_seconds(spec or _FALLBACK_SPEC),
        )
        generation_duration_ms = (time.perf_counter() - generation_started) * 1000

        logger.info(
            "LLM generation END: provider=%s model=%s duration=%.0fms",
            provider,
            model,
            generation_duration_ms,
        )

        response.raise_for_status()
        payload = response.json()

        logger.info(
            "LLM generation TOKENS: provider=%s model=%s prompt_tokens=%d "
            "completion_tokens=%d total_tokens=%d",
            provider,
            model,
            (payload.get("usage") or {}).get("prompt_tokens", 0) or 0,
            (payload.get("usage") or {}).get("completion_tokens", 0) or 0,
            (payload.get("usage") or {}).get("total_tokens", 0) or 0,
        )

        return cls._read_openai_compatible_response(
            payload=payload,
            provider=provider,
            model=model,
            prompt_chars=prompt_chars,
            estimated_prompt_tokens=estimated_prompt_tokens,
            duration_ms=generation_duration_ms,
        )

    @classmethod
    def _execute_direct_gemini(
        cls,
        prompt: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        spec = get_spec("gemini")
        effective_api_key = (api_key or _live_api_key("gemini") or "").strip()
        if not effective_api_key:
            raise RuntimeError(
                "Gemini provider is not configured: " "missing GEMINI_API_KEY/GOOGLE_API_KEY."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Gemini provider requires the google-genai package.") from exc

        client = genai.Client(
            api_key=effective_api_key,
            # The SDK otherwise waits far longer than any caller does, and a
            # hung endpoint would not surface as a retryable error.
            http_options=types.HttpOptions(
                timeout=_provider_timeout_seconds(spec or _FALLBACK_SPEC) * 1000
            ),
        )

        # The ceiling is required here, not optional as it is elsewhere: Gemini
        # has no default this project can rely on for a long CV body, and a
        # silently truncated document is the failure mode this whole pipeline is
        # built to avoid.
        ceiling = max_tokens or (spec and _provider_max_tokens(spec)) or 2048

        request: dict[str, Any] = {
            "model": model,
            "contents": prompt,
            "config": types.GenerateContentConfig(
                max_output_tokens=ceiling,
                temperature=(spec.temperature if spec else 0.2),
            ),
        }

        response = client.models.generate_content(**request)

        content = getattr(
            response,
            "text",
            None,
        )

        if not content:
            raise RuntimeError("Gemini returned empty content.")

        usage = _empty_usage()

        response_usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if response_usage is not None:
            usage["prompt_tokens"] = _coerce_nonnegative_int(
                getattr(
                    response_usage,
                    "prompt_token_count",
                    0,
                )
            )

            usage["completion_tokens"] = _coerce_nonnegative_int(
                getattr(
                    response_usage,
                    "candidates_token_count",
                    0,
                )
            )

            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        cls._set_last_usage(usage)

        return clean_llm_output(content)

    @classmethod
    def _execute_claude(
        cls,
        prompt: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        spec = get_spec("claude")
        effective_api_key = (api_key or _live_api_key("claude") or "").strip()
        if not effective_api_key:
            raise RuntimeError("Claude provider is not configured: " "missing ANTHROPIC_API_KEY.")

        client = anthropic.Anthropic(
            api_key=effective_api_key,
            base_url=_get_claude_base_url(),
            # The SDK otherwise waits far longer than any caller does, and a
            # hung endpoint would not surface as a retryable error.
            timeout=_provider_timeout_seconds(spec or _FALLBACK_SPEC),
        )

        # Anthropic requires max_tokens. It used to be a hard-coded 4096, which
        # ignored both the configured ceiling and the task's.
        ceiling = max_tokens or (spec and _provider_max_tokens(spec)) or 4096

        response = client.messages.create(
            model=model,
            max_tokens=ceiling,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        usage = _extract_anthropic_usage(response)

        usage["estimated_cost_usd"] = _estimate_cost_usd(
            model,
            usage,
        )

        cls._set_last_usage(usage)

        content_parts = getattr(
            response,
            "content",
            [],
        )

        text_parts: list[str] = []

        for block in content_parts:
            block_text = getattr(
                block,
                "text",
                None,
            )

            if block_text:
                text_parts.append(block_text)

        content = "\n".join(text_parts).strip()

        if not content:
            raise RuntimeError("Claude returned empty content.")

        return clean_llm_output(content)

    @classmethod
    def _execute_gateway(
        cls,
        prompt: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        effective_api_key = (api_key or _live_api_key("experiential") or "").strip()
        if not effective_api_key:
            raise RuntimeError(
                "Experiential provider is not configured: "
                "missing EXPLABS_API_KEY/EXPERIENTIAL_ORG_KEY."
            )

        # IMPORTANT:
        # URL validation happens here, not at module import.
        gateway_base_url = _get_experiential_base_url()

        return cls._execute_direct_openai_style(
            prompt=prompt,
            model=model,
            api_key=effective_api_key,
            base_url=gateway_base_url,
            provider="experiential",
            max_tokens=max_tokens,
        )

    @classmethod
    def _execute_single_provider(
        cls,
        prompt: str,
        provider: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send one request to one provider.

        Dispatch is driven entirely by the registry, so a provider only has to
        be described once. This replaced an eight-branch ``if``/``elif`` chain
        that also had to be mirrored in six other lists, which is how a provider
        could end up reachable in one place and unknown in another.

        ``max_tokens`` is threaded through so a caller that knows the task needs
        a longer answer (a whole CV, not a keyword) can ask for one without
        editing a request builder.
        """
        canonical = canonical_provider(provider)

        if not canonical:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Known providers: {', '.join(SUPPORTED_PROVIDERS)}."
            )

        spec = PROVIDER_REGISTRY[canonical]
        request_key = (api_key or "").strip()
        ceiling = max_tokens or _provider_max_tokens(spec)

        if canonical == "ollama":
            return cls._execute_ollama(
                prompt,
                model,
                max_tokens=ceiling,
            )

        if spec.protocol == PROTOCOL_OPENAI_COMPATIBLE:
            base_url = _provider_base_url(spec)
            resolved_key = request_key or _live_api_key(canonical)

            if spec.local:
                # Every local provider uses the plain-HTTP path, whether or not
                # it has a credential. The OpenAI SDK requires a non-empty key
                # and always emits an Authorization header, which is an OpenAI
                # convention a gateway on the operator's own machine has no use
                # for -- and a gateway that deliberately has no authentication
                # can answer 401 rather than serve the request. Here the header
                # is sent only when a real key is configured, and nothing is
                # invented when there is not.
                return cls._execute_http_openai_style(
                    prompt=prompt,
                    model=model,
                    base_url=base_url,
                    provider=canonical,
                    api_key=resolved_key,
                    max_tokens=ceiling,
                )

            return cls._execute_direct_openai_style(
                prompt=prompt,
                model=model,
                api_key=resolved_key,
                base_url=base_url,
                provider=canonical,
                max_tokens=ceiling,
            )

        if spec.protocol == PROTOCOL_GEMINI:
            return cls._execute_direct_gemini(
                prompt,
                model,
                api_key=request_key or _live_api_key(canonical),
                max_tokens=ceiling,
            )

        if spec.protocol == PROTOCOL_ANTHROPIC:
            return cls._execute_claude(
                prompt,
                model,
                api_key=request_key or _live_api_key(canonical),
                max_tokens=ceiling,
            )

        raise ValueError(f"Unsupported LLM protocol for {canonical}: {spec.protocol}")

    # ---------------------------------------------------------------------
    # Main generation / dynamic fallback
    # ---------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        route_mode: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate one completion.

        ``task`` names the kind of work being done (``"cv_tailoring"``,
        ``"full_cv_generation"``, ...) and is the only routing hint a caller
        needs to give. ``provider``, ``model`` and ``route_mode`` stay optional;
        when they are omitted the environment decides, so switching between
        Ollama and a cloud provider never requires editing call sites.
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected LLM options: {unexpected}")
        if model is None and model_name:
            model = model_name
        request_api_key = api_key.strip() if isinstance(api_key, str) else None

        if not isinstance(prompt, str):
            raise TypeError("LLM prompt must be a string.")

        prompt = prompt.strip()

        if not prompt:
            raise ValueError("LLM prompt cannot be empty.")

        task_name = normalize_task(task)
        mode = cls._default_mode()

        effective_provider = (
            provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        )

        effective_route_mode = (
            route_mode.strip().lower()
            if isinstance(route_mode, str) and route_mode.strip()
            else cls._mode_to_route(mode)
        )

        aliases = {
            "auto": "automatic",
            "dynamic": "automatic",
            "automatic": "automatic",
            "direct": "direct",
            "gateway": "experiential",
            "experiential": "experiential",
            "local": LLM_MODE_LOCAL,
            "ollama": LLM_MODE_LOCAL,
            "online": LLM_MODE_ONLINE,
            "cloud": LLM_MODE_ONLINE,
        }

        effective_route_mode = aliases.get(
            effective_route_mode,
            effective_route_mode,
        )

        if effective_route_mode not in {
            "automatic",
            "direct",
            "experiential",
            LLM_MODE_LOCAL,
            LLM_MODE_ONLINE,
        }:
            raise ValueError(f"Unsupported LLM route mode: " f"{effective_route_mode}")

        candidates = cls._build_candidates(
            provider=effective_provider,
            model=model,
            route_mode=effective_route_mode,
            task=task_name,
            prompt=prompt,
        )

        if not candidates:
            raise RuntimeError(
                "No configured LLM providers are available for this task. "
                f"mode={mode} task={task_name}. "
                "Check LLM_MODE, LLM_PROVIDER, provider API keys, "
                "OLLAMA_BASE_URL, and the provider chain."
            )

        selected_candidates = cls._select_candidates(candidates)

        # If every candidate is temporarily exhausted, perform one
        # emergency pass instead of failing immediately.
        if not selected_candidates:
            # Every candidate is in a cooldown. If the shortest one expires soon
            # -- which it will when the cooldown came from a rate limit that
            # stated its own duration -- waiting it out is the difference between
            # a served request and a failed one. Capped, so a long cooldown means
            # "skip", not "hang".
            remaining = cls._shortest_exhaustion_remaining()
            ceiling = _env_float("LLM_RATE_LIMIT_BACKOFF_SECONDS", LLM_RATE_LIMIT_BACKOFF_SECONDS)
            wait = min(remaining, max(0.0, ceiling))

            if wait > 0:
                logger.info(
                    "Every candidate is cooling down; waiting %.1s before the "
                    "emergency pass (cap %.1fs).",
                    wait,
                    ceiling,
                )
                time.sleep(wait)

            logger.warning(
                "All LLM candidates are temporarily exhausted; " "performing emergency retry pass."
            )

            selected_candidates = candidates

        # The hard attempt ceiling. Applied after the cooldown filter so a dead
        # provider does not consume a slot, and always leaving at least one
        # candidate so the request is still attempted.
        attempt_cap = max(
            1,
            _env_int("LLM_MAX_PROVIDER_ATTEMPTS", LLM_MAX_PROVIDER_ATTEMPTS),
        )

        if len(selected_candidates) > attempt_cap:
            logger.info(
                "Limiting this request to %d of %d candidates "
                "(LLM_MAX_PROVIDER_ATTEMPTS); the rest are not tried.",
                attempt_cap,
                len(selected_candidates),
            )
            selected_candidates = selected_candidates[:attempt_cap]

        if not _env_bool("LLM_FALLBACK_ENABLED", LLM_FALLBACK_ENABLED):
            # Exactly one provider, whatever happens. The mode says what to use;
            # this says whether a failure may be handed to someone else.
            selected_candidates = selected_candidates[:1]

        # A whole CV body needs more room than a keyword extraction, and the
        # task already declares that. It was previously declared and never
        # applied, so every provider silently cut a full CV at the shared
        # default.
        policy = task_policy(task_name)
        request_max_tokens = policy.max_output_tokens or None

        request_id = uuid.uuid4().hex[:12]

        failures: list[str] = []
        last_provider = ""
        last_failure_category = "unknown"

        logger.info(
            "LLM request %s task=%s mode=%s route=%s candidates=%s",
            request_id,
            task_name,
            mode,
            effective_route_mode,
            ",".join(f"{p}/{m}" for p, m in selected_candidates),
        )

        for attempt_number, (
            candidate_provider,
            candidate_model,
        ) in enumerate(
            selected_candidates,
            start=1,
        ):
            started = time.perf_counter()

            if last_provider:
                logger.info(
                    "LLM fallback %s task=%s from=%s to=%s " "reason=%s attempt=%d/%d",
                    request_id,
                    task_name,
                    last_provider,
                    candidate_provider,
                    last_failure_category,
                    attempt_number,
                    len(selected_candidates),
                )

            logger.info(
                "LLM request %s attempt %d/%d: " "provider=%s model=%s route=%s",
                request_id,
                attempt_number,
                len(selected_candidates),
                candidate_provider,
                candidate_model,
                effective_route_mode,
            )

            _write_log(
                {
                    "timestamp": _utc_now(),
                    "event": "request_started",
                    "request_id": request_id,
                    "attempt": attempt_number,
                    "attempts_total": len(selected_candidates),
                    "task": task_name,
                    "mode": mode,
                    "provider": candidate_provider,
                    "model": candidate_model,
                    "route_mode": effective_route_mode,
                    "fallback_from": last_provider,
                    "fallback_reason": (last_failure_category if last_provider else ""),
                }
            )

            try:
                result = cls._execute_single_provider(
                    prompt=prompt,
                    provider=candidate_provider,
                    model=candidate_model,
                    api_key=request_api_key,
                    max_tokens=request_max_tokens,
                )

                duration_ms = (time.perf_counter() - started) * 1000

                cls._record_execution_time(
                    candidate_provider,
                    candidate_model,
                    duration_ms,
                )

                usage = cls._pop_last_usage()

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "event": "request_completed",
                        "request_id": request_id,
                        "attempt": attempt_number,
                        "task": task_name,
                        "mode": mode,
                        "provider": candidate_provider,
                        "model": candidate_model,
                        "route_mode": effective_route_mode,
                        "duration_ms": round(
                            duration_ms,
                            2,
                        ),
                        "prompt_tokens": usage.get(
                            "prompt_tokens",
                            0,
                        ),
                        "completion_tokens": usage.get(
                            "completion_tokens",
                            0,
                        ),
                        "total_tokens": usage.get(
                            "total_tokens",
                            0,
                        ),
                        "estimated_cost_usd": usage.get(
                            "estimated_cost_usd",
                            0.0,
                        ),
                    }
                )

                logger.info(
                    "LLM request %s completed: " "task=%s provider=%s model=%s duration=%.0fms",
                    request_id,
                    task_name,
                    candidate_provider,
                    candidate_model,
                    duration_ms,
                )

                return result

            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000

                category = classify_llm_error(
                    exc,
                    candidate_provider,
                )

                retryable = _is_retryable_error(
                    exc,
                    candidate_provider,
                )

                configuration_error = _is_configuration_error(exc)

                error_text = _safe_error(exc)

                failures.append((f"{candidate_provider}/" f"{candidate_model}: " f"{error_text}"))

                cls._record_execution_time(
                    candidate_provider,
                    candidate_model,
                    duration_ms,
                )

                # Only transient failures should cause cooldown/exhaustion.
                # A terminal category (no credits, bad key, oversized context)
                # would repeat identically on every retry, so it is recorded
                # once and the router moves on instead of hammering the
                # provider -- the 429 insufficient_credits loop this replaces.
                # A rate limit states its own duration, so the cooldown is set
                # from that rather than from the flat default.
                stated_retry_after = (
                    rate_limit_retry_after(exc) if category == LLM_ERROR_RATE_LIMIT else None
                )

                if retryable:
                    cls._mark_exhausted(
                        candidate_provider,
                        candidate_model,
                        stated_retry_after,
                    )
                elif category in _TERMINAL_CATEGORIES:
                    cls._mark_exhausted(
                        candidate_provider,
                        candidate_model,
                        stated_retry_after,
                    )

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "event": "provider_failed",
                        "request_id": request_id,
                        "attempt": attempt_number,
                        "task": task_name,
                        "mode": mode,
                        "provider": candidate_provider,
                        "model": candidate_model,
                        "route_mode": effective_route_mode,
                        "error_category": category,
                        "duration_ms": round(
                            duration_ms,
                            2,
                        ),
                        "retryable": retryable,
                        "configuration_error": (configuration_error),
                        "status_code": _extract_status_code(exc),
                        "error": error_text,
                    }
                )

                logger.warning(
                    "LLM provider failed: "
                    "task=%s provider=%s model=%s category=%s "
                    "retryable=%s config_error=%s "
                    "duration=%.0fms error=%s",
                    task_name,
                    candidate_provider,
                    candidate_model,
                    category,
                    retryable,
                    configuration_error,
                    duration_ms,
                    error_text,
                )

                last_provider = f"{candidate_provider}/{candidate_model}"
                last_failure_category = category

                # Explicit direct mode means exactly one provider -- and it
                # still does, because every direct candidate is that one
                # provider: the mode chooses the provider and the provider's
                # own ordered model list supplies the remaining candidates.
                # So continuing here cannot escalate to anyone else, and it is
                # what makes a retired model id survivable. OpenRouter answers
                # 400 "is not a valid model ID" for an id it no longer serves,
                # which used to raise on the first candidate and strand the
                # request even though FALLBACK_MODEL_1 held a working id the
                # whole time. The provider is only given up on once its models
                # are exhausted, which is what `attempt_number` here decides.
                if (
                    effective_route_mode == "direct"
                    and attempt_number >= len(selected_candidates)
                ):
                    raise

                # Local-only mode never escalates. A failure here is reported
                # as-is instead of quietly sending the candidate's CV to a
                # cloud provider, which is the whole point of LLM_MODE=ollama.
                if effective_route_mode == LLM_MODE_LOCAL:
                    raise RuntimeError(
                        f"Local Ollama request failed and LLM_MODE=ollama "
                        f"forbids falling back to an online provider. "
                        f"task={task_name} category={category} "
                        f"provider={candidate_provider} "
                        f"model={candidate_model}: {error_text}"
                    ) from exc

                # Explicit Experiential mode also means exactly the
                # Experiential chain, not direct-provider fallback.
                if effective_route_mode == "experiential":
                    if not retryable:
                        raise

                    continue

                # Automatic mode:
                # transient failures fall through to the next provider.
                #
                # Configuration/auth errors also move to the next
                # configured provider, but are not marked exhausted.
                #
                # This lets a missing/invalid cloud credential affect only
                # that provider rather than killing the whole job.
                continue

        error_summary = "; ".join(failures)

        if effective_route_mode == LLM_MODE_LOCAL:
            raise RuntimeError(
                "The local Ollama request failed. " f"task={task_name} Attempts: {error_summary}"
            )

        raise RuntimeError(
            "All configured LLM providers failed. "
            f"task={task_name} mode={mode} Attempts: {error_summary}"
        )

    # ---------------------------------------------------------------------
    # Catalog / diagnostics
    # ---------------------------------------------------------------------

    @classmethod
    def get_model_catalog(
        cls,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        selected = provider.lower() if provider else None

        providers = [selected] if selected else list(SUPPORTED_PROVIDERS)

        result: list[dict[str, Any]] = []

        for current_provider in providers:
            if current_provider not in SUPPORTED_PROVIDERS:
                continue

            configured = cls._provider_is_configured(current_provider)

            if current_provider == "openrouter":
                models = cls._openrouter_models()

            elif current_provider == "experiential":
                models = cls._experiential_models()

            else:
                models = [cls.get_default_model(current_provider)]

            for current_model in models:
                result.append(
                    {
                        "provider": current_provider,
                        "model": current_model,
                        "configured": configured,
                        "available": configured,
                    }
                )

        return result

    @classmethod
    def get_usage_summary(
        cls,
        since_hours: int | None = None,
    ) -> dict[str, Any]:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))
            if since_hours is not None
            else None
        )

        provider_stats: dict[
            str,
            dict[str, Any],
        ] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        )

        model_stats: dict[
            str,
            dict[str, Any],
        ] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        )

        total_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0

        if not LOG_PATH.exists():
            return {
                "since_hours": since_hours,
                "days": (since_hours / 24) if since_hours is not None else None,
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "providers": {},
                "models": {},
            }

        try:
            with LOG_PATH.open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("event") != "request_completed":
                        continue

                    timestamp = event.get("timestamp")

                    if not timestamp:
                        continue

                    try:
                        event_time = datetime.fromisoformat(timestamp)
                    except ValueError:
                        continue
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    else:
                        event_time = event_time.astimezone(timezone.utc)

                    if cutoff is not None and event_time < cutoff:
                        continue

                    provider_name = event.get(
                        "provider",
                        "unknown",
                    )

                    model_name = event.get(
                        "model",
                        "unknown",
                    )

                    prompt_tokens = _coerce_nonnegative_int(
                        event.get(
                            "prompt_tokens",
                            0,
                        )
                    )

                    completion_tokens = _coerce_nonnegative_int(
                        event.get(
                            "completion_tokens",
                            0,
                        )
                    )

                    total_tokens = _coerce_nonnegative_int(
                        event.get(
                            "total_tokens",
                            prompt_tokens + completion_tokens,
                        )
                    )

                    cost = _coerce_nonnegative_float(event.get("estimated_cost_usd", 0.0))

                    total_calls += 1
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_cost += cost

                    provider_entry = provider_stats[provider_name]

                    provider_entry["calls"] += 1

                    provider_entry["prompt_tokens"] += prompt_tokens

                    provider_entry["completion_tokens"] += completion_tokens

                    provider_entry["total_tokens"] += total_tokens

                    provider_entry["estimated_cost_usd"] += cost

                    model_entry = model_stats[model_name]

                    model_entry["calls"] += 1

                    model_entry["prompt_tokens"] += prompt_tokens

                    model_entry["completion_tokens"] += completion_tokens

                    model_entry["total_tokens"] += total_tokens

                    model_entry["estimated_cost_usd"] += cost

        except OSError:
            logger.exception("Unable to read LLM processing log")

        return {
            "since_hours": since_hours,
            "days": (since_hours / 24) if since_hours is not None else None,
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": (total_prompt_tokens + total_completion_tokens),
            "estimated_cost_usd": round(
                total_cost,
                8,
            ),
            "providers": dict(provider_stats),
            "models": dict(model_stats),
        }

    @classmethod
    def recent_logs(
        cls,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not LOG_PATH.exists():
            return []

        entries: list[dict[str, Any]] = []

        try:
            with LOG_PATH.open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line in handle:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        except OSError:
            logger.exception("Unable to read LLM processing log")
            return []

        return entries[-max(1, limit) :]

    @classmethod
    def clear_logs(cls) -> None:
        try:
            if LOG_PATH.exists():
                LOG_PATH.unlink()

        except OSError:
            logger.exception("Unable to clear LLM processing log")
