from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
import openai
import requests
from dotenv import load_dotenv

from app.services.llm import quota_tracker

logger = logging.getLogger(__name__)

# ============================================================================
# Paths / environment
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=True)

DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "llm_processing.jsonl"

LOG_PATH = Path(
    os.getenv(
        "LLM_PROCESSING_LOG",
        str(DEFAULT_LOG_PATH),
    )
)

if not LOG_PATH.is_absolute():
    LOG_PATH = PROJECT_ROOT / LOG_PATH


# ============================================================================
# Gateway configuration
# ============================================================================

GATEWAY_BASE_URL = (
    os.getenv(
        "OPENAI_BASE_URL",
        "https://api.experientiallabs.ai/v1",
    )
    .strip()
    .rstrip("/")
)

CLAUDE_GATEWAY_BASE_URL = GATEWAY_BASE_URL.removesuffix("/v1").rstrip("/")

EXPERIENTIAL_API_KEY = (
    os.getenv("EXPLABS_API_KEY", "").strip() or os.getenv("EXPERIENTIAL_ORG_KEY", "").strip()
)


# ============================================================================
# Default models
# ============================================================================

DEFAULT_MODELS = {
    "experiential": os.getenv(
        "EXPERIENTIAL_MODEL",
        "gpt-5.6-luna",
    ),
    "gemini": os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ),
    "groq": os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    ),
    "openrouter": os.getenv(
        "OPENROUTER_MODEL",
        "deepseek/deepseek-r1:free",
    ),
    "deepseek": os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-chat",
    ),
    "openai": os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    ),
    "claude": os.getenv(
        "CLAUDE_MODEL",
        "claude-3-5-haiku-20241022",
    ),
    "ollama": os.getenv(
        "OLLAMA_MODEL",
        "llama3.2",
    ),
}


SUPPORTED_PROVIDERS = (
    "experiential",
    "gemini",
    "groq",
    "openrouter",
    "deepseek",
    "openai",
    "claude",
    "ollama",
)


# ============================================================================
# Static fallback model catalog for Experiential Labs
# ============================================================================

_EXPERIENTIAL_FALLBACK_CATALOG: list[dict[str, Any]] = [
    {
        "model": "gpt-6-astra",
        "label": "GPT-6 Astra",
        "provider": "experiential",
        "status": "PROMOTION",
        "free": True,
        "promotion": True,
        "context_window": 128000,
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "currency": "USD",
    },
    {
        "model": "gpt-5.6-luna",
        "label": "GPT-5.6 Luna",
        "provider": "experiential",
        "status": "PROMOTION",
        "free": True,
        "promotion": True,
        "context_window": 128000,
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "currency": "USD",
    },
    {
        "model": "deepseek-v4-flash",
        "label": "DeepSeek V4 Flash",
        "provider": "experiential",
        "status": "PROMOTION",
        "free": True,
        "promotion": True,
        "context_window": 128000,
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "currency": "USD",
    },
    {
        "model": "qwen3.8-27b",
        "label": "Qwen3.8 27B",
        "provider": "experiential",
        "status": "PROMOTION",
        "free": True,
        "promotion": True,
        "context_window": 128000,
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "currency": "USD",
    },
    {
        "model": "gemini-3.7-flash",
        "label": "Gemini 3.7 Flash",
        "provider": "experiential",
        "status": "FREE",
        "free": True,
        "promotion": False,
        "context_window": 1000000,
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "currency": "USD",
    },
    {
        "model": "claude-fable-5",
        "label": "Claude Fable 5",
        "provider": "experiential",
        "status": "PAID",
        "free": False,
        "promotion": False,
        "context_window": 200000,
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
        "currency": "USD",
    },
    {
        "model": "gpt-4o",
        "label": "GPT-4o",
        "provider": "experiential",
        "status": "PAID",
        "free": False,
        "promotion": False,
        "context_window": 128000,
        "input_per_1k": 0.0025,
        "output_per_1k": 0.01,
        "currency": "USD",
    },
]


_MODEL_PRICING: dict[str, tuple[float, float]] = {
    row["model"]: (
        float(row.get("input_per_1k", 0.0) or 0.0),
        float(row.get("output_per_1k", 0.0) or 0.0),
    )
    for row in _EXPERIENTIAL_FALLBACK_CATALOG
}


# ============================================================================
# Text cleanup
# ============================================================================


def clean_llm_output(text: str) -> str:
    """Clean common LLM formatting artifacts."""

    if not text:
        return ""

    text = str(text).strip()

    # Remove opening fenced code blocks.
    text = re.sub(
        r"^\s*```(?:latex|tex|html|markdown|json|text)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove closing fenced code block.
    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    conversational_intros = [
        r"^(?:Here is|Sure[,!]?\s+Here is|Certainly[,!]?\s+Here is)\s+(?:the|an)?\s*(?:optimized|tailored|updated|generated|LaTeX|HTML|resume|CV).*?:?\s*",
        r"^Sure,?\s+I can help with that\.?\s*",
        r"^I'm ready to (?:enrich|optimize|help).*?:?\s*",
        r"^Certainly!?\s+Here is.*?:?\s*",
        r"^Below is the.*?:?\s*",
    ]

    for pattern in conversational_intros:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    return text


# ============================================================================
# Misc helpers
# ============================================================================


def _clean_base_url(url: str, default: str) -> str:
    if not url:
        return default

    url = url.strip()

    markdown_match = re.search(
        r"\]\((https?://[^)]+)\)",
        url,
    )

    if markdown_match:
        return markdown_match.group(1).rstrip("/")

    match = re.search(
        r"https?://[^\s)]+",
        url,
    )

    if match:
        return match.group(0).rstrip("/")

    if not url.startswith(("http://", "https://")):
        return f"https://{url}".rstrip("/")

    return url.rstrip("/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_log(event: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if "kind" not in event:
            event = {
                **event,
                "kind": "llm",
            }

        with LOG_PATH.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

    except Exception:
        logger.debug(
            "Failed to write LLM log entry",
            exc_info=True,
        )


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()

    sensitive_values: list[str] = []

    if EXPERIENTIAL_API_KEY:
        sensitive_values.append(EXPERIENTIAL_API_KEY)

    provider_key_names = [
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "EXPERIENTIAL_ORG_KEY",
        "EXPLABS_API_KEY",
        "ANTHROPIC_WORKSPACE_ID",
        "HF_TOKEN",
    ]

    for env_name in provider_key_names:
        value = os.getenv(
            env_name,
            "",
        ).strip()

        if value:
            sensitive_values.append(value)

    for value in sensitive_values:
        if value:
            text = text.replace(
                value,
                "***REDACTED***",
            )

    return text[:2000]


# ============================================================================
# Usage extraction
# ============================================================================


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _extract_openai_usage(response: Any) -> dict[str, int]:
    try:
        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return _empty_usage()

        prompt = int(
            getattr(
                usage,
                "prompt_tokens",
                0,
            )
            or 0
        )

        completion = int(
            getattr(
                usage,
                "completion_tokens",
                0,
            )
            or 0
        )

        total = int(
            getattr(
                usage,
                "total_tokens",
                0,
            )
            or 0
        )

        if total == 0 and (prompt or completion):
            total = prompt + completion

        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    except Exception:
        return _empty_usage()


def _extract_anthropic_usage(response: Any) -> dict[str, int]:
    try:
        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return _empty_usage()

        prompt = int(
            getattr(
                usage,
                "input_tokens",
                0,
            )
            or 0
        )

        completion = int(
            getattr(
                usage,
                "output_tokens",
                0,
            )
            or 0
        )

        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    except Exception:
        return _empty_usage()


def _estimate_cost_usd(
    model: str,
    usage: dict[str, int],
) -> float:
    input_price, output_price = _MODEL_PRICING.get(
        model,
        (0.0, 0.0),
    )

    prompt = (
        usage.get(
            "prompt_tokens",
            0,
        )
        or 0
    )

    completion = (
        usage.get(
            "completion_tokens",
            0,
        )
        or 0
    )

    return round(
        (prompt / 1000.0) * input_price + (completion / 1000.0) * output_price,
        6,
    )


# ============================================================================
# LLM Service
# ============================================================================


class LLMService:
    DEFAULT_MODELS = DEFAULT_MODELS
    SUPPORTED_PROVIDERS = SUPPORTED_PROVIDERS

    _last_usage: dict[str, int] = {}
    _usage_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Usage state
    # ------------------------------------------------------------------

    @classmethod
    def _set_last_usage(
        cls,
        usage: dict[str, int],
    ) -> None:
        with cls._usage_lock:
            cls._last_usage = usage or _empty_usage()

    @classmethod
    def _pop_last_usage(cls) -> dict[str, int]:
        with cls._usage_lock:
            usage = cls._last_usage or _empty_usage()

            cls._last_usage = {}

            return usage

    # ------------------------------------------------------------------
    # Public wrappers
    # ------------------------------------------------------------------

    @classmethod
    def call_llm(
        cls,
        prompt: str,
        provider: str = "experiential",
        model_name: str | None = None,
        route_mode: str = "experiential",
    ) -> str:
        return cls.generate(
            prompt=prompt,
            provider=provider,
            model_name=model_name,
            route_mode=route_mode,
        )

    @classmethod
    def get_default_model(
        cls,
        provider: str,
    ) -> str:
        provider = (provider or "experiential").strip().lower()

        return cls.DEFAULT_MODELS.get(
            provider,
            "",
        )

    @classmethod
    def _get_gateway_key(cls) -> str:
        key = (
            os.getenv(
                "EXPLABS_API_KEY",
                "",
            ).strip()
            or os.getenv(
                "EXPERIENTIAL_ORG_KEY",
                "",
            ).strip()
        )

        if not key:
            raise ValueError("EXPLABS_API_KEY / EXPERIENTIAL_ORG_KEY is not configured.")

        return key

    # ------------------------------------------------------------------
    # Provider status
    # ------------------------------------------------------------------

    @classmethod
    def provider_status(
        cls,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        rows.append(
            {
                "provider": "experiential",
                "model": cls.get_default_model("experiential"),
                "configured": bool(EXPERIENTIAL_API_KEY),
                "authentication": "experiential_gateway",
                "gateway": GATEWAY_BASE_URL,
            }
        )

        for provider in cls.SUPPORTED_PROVIDERS:
            if provider == "experiential":
                continue

            model = cls.get_default_model(provider)

            if provider == "claude":
                anthropic_key = os.getenv(
                    "ANTHROPIC_API_KEY",
                    "",
                ).strip()

                configured = bool(EXPERIENTIAL_API_KEY or anthropic_key)

                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "configured": configured,
                        "authentication": (
                            "direct_api" if anthropic_key else "experiential_gateway"
                        ),
                        "gateway": CLAUDE_GATEWAY_BASE_URL,
                    }
                )

                continue

            direct_key = None

            if provider == "gemini":
                direct_key = os.getenv("GEMINI_API_KEY")

            elif provider == "openai":
                direct_key = os.getenv("OPENAI_API_KEY")

            elif provider == "groq":
                direct_key = os.getenv("GROQ_API_KEY")

            elif provider == "openrouter":
                direct_key = os.getenv("OPENROUTER_API_KEY")

            elif provider == "deepseek":
                direct_key = os.getenv("DEEPSEEK_API_KEY")

            if provider == "ollama":
                base_url = os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434/api/generate",
                )

                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "configured": True,
                        "authentication": "local",
                        "endpoint": base_url,
                    }
                )

            elif direct_key and direct_key.strip():
                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "configured": True,
                        "authentication": "direct_api",
                        "endpoint": "native_provider_api",
                    }
                )

            else:
                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "configured": bool(EXPERIENTIAL_API_KEY),
                        "authentication": "experiential_gateway",
                        "gateway": GATEWAY_BASE_URL,
                    }
                )

        return rows

    # ------------------------------------------------------------------
    # Model catalog
    # ------------------------------------------------------------------

    @classmethod
    def get_model_catalog(
        cls,
        provider: str = "experiential",
    ) -> list[dict[str, Any]]:
        provider = (provider or "experiential").strip().lower()

        if provider != "experiential":
            model = cls.get_default_model(provider)

            return [
                {
                    "model": model,
                    "label": model,
                    "provider": provider,
                    "status": "PAID",
                    "free": False,
                    "promotion": False,
                    "context_window": None,
                    "input_per_1k": 0.0,
                    "output_per_1k": 0.0,
                    "currency": "USD",
                }
            ]

        static_by_id = {row["model"]: row for row in _EXPERIENTIAL_FALLBACK_CATALOG}

        try:
            key = cls._get_gateway_key()

            client = openai.OpenAI(
                api_key=key,
                base_url=GATEWAY_BASE_URL,
            )

            listing = client.models.list()

            live_ids = [
                model.id
                for model in getattr(
                    listing,
                    "data",
                    [],
                )
                if getattr(
                    model,
                    "id",
                    None,
                )
            ]

        except Exception:
            live_ids = []

        if not live_ids:
            return list(_EXPERIENTIAL_FALLBACK_CATALOG)

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for model_id in live_ids:
            if model_id in seen:
                continue

            seen.add(model_id)

            if model_id in static_by_id:
                merged.append(static_by_id[model_id])
            else:
                merged.append(
                    {
                        "model": model_id,
                        "label": model_id,
                        "provider": "experiential",
                        "status": "PAID",
                        "free": False,
                        "promotion": False,
                        "context_window": None,
                        "input_per_1k": 0.0,
                        "output_per_1k": 0.0,
                        "currency": "USD",
                    }
                )

        for model_id, row in static_by_id.items():
            if model_id not in seen:
                merged.append(row)
                seen.add(model_id)

        order = {
            "PROMOTION": 0,
            "FREE": 1,
            "PAID": 2,
        }

        merged.sort(
            key=lambda row: (
                order.get(
                    str(
                        row.get(
                            "status",
                            "PAID",
                        )
                    ).upper(),
                    3,
                ),
                str(row.get("label") or row.get("model") or "").lower(),
            )
        )

        return merged

    # ------------------------------------------------------------------
    # Usage aggregation
    # ------------------------------------------------------------------

    @classmethod
    def get_usage_summary(
        cls,
        since_hours: int | None = None,
        group_by: str = "provider",
    ) -> dict[str, Any]:
        if group_by not in (
            "provider",
            "model",
        ):
            group_by = "provider"

        cutoff_iso: str | None = None

        if since_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

            cutoff_iso = cutoff.isoformat()

        summary: dict[str, Any] = {
            "since_hours": since_hours,
            "since_iso": cutoff_iso,
            "total_calls": 0,
            "success_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "by_provider": {},
            "by_model": {},
            "by_day": [],
            "last_call": None,
        }

        if not LOG_PATH.exists():
            return summary

        per_provider: dict[
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

        per_model: dict[
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

        per_day: dict[
            str,
            dict[str, int],
        ] = defaultdict(
            lambda: {
                "tokens": 0,
                "calls": 0,
            }
        )

        last_completed: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None

        try:
            with LOG_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:
                for raw_line in file:
                    raw_line = raw_line.strip()

                    if not raw_line:
                        continue

                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    timestamp = event.get(
                        "timestamp",
                        "",
                    )

                    if cutoff_iso and timestamp and timestamp < cutoff_iso:
                        continue

                    event_type = event.get("event")

                    if event_type == "request_started":
                        summary["total_calls"] += 1

                    elif event_type == "request_completed":
                        summary["success_calls"] += 1

                        prompt = int(
                            event.get(
                                "prompt_tokens",
                                0,
                            )
                            or 0
                        )

                        completion = int(
                            event.get(
                                "completion_tokens",
                                0,
                            )
                            or 0
                        )

                        total = int(
                            event.get(
                                "total_tokens",
                                0,
                            )
                            or 0
                        )

                        if total == 0 and (prompt or completion):
                            total = prompt + completion

                        provider = str(
                            event.get(
                                "provider",
                                "unknown",
                            )
                        )

                        model = str(
                            event.get(
                                "model",
                                "unknown",
                            )
                        )

                        cost = _estimate_cost_usd(
                            model,
                            {
                                "prompt_tokens": prompt,
                                "completion_tokens": completion,
                                "total_tokens": total,
                            },
                        )

                        summary["prompt_tokens"] += prompt

                        summary["completion_tokens"] += completion

                        summary["total_tokens"] += total

                        summary["estimated_cost_usd"] = round(
                            summary["estimated_cost_usd"] + cost,
                            6,
                        )

                        provider_stats = per_provider[provider]

                        provider_stats["calls"] += 1

                        provider_stats["prompt_tokens"] += prompt

                        provider_stats["completion_tokens"] += completion

                        provider_stats["total_tokens"] += total

                        provider_stats["estimated_cost_usd"] = round(
                            provider_stats["estimated_cost_usd"] + cost,
                            6,
                        )

                        model_stats = per_model[model]

                        model_stats["calls"] += 1

                        model_stats["prompt_tokens"] += prompt

                        model_stats["completion_tokens"] += completion

                        model_stats["total_tokens"] += total

                        model_stats["estimated_cost_usd"] = round(
                            model_stats["estimated_cost_usd"] + cost,
                            6,
                        )

                        day_key = timestamp[:10] if timestamp else "unknown"

                        per_day[day_key]["tokens"] += total

                        per_day[day_key]["calls"] += 1

                        last_completed = {
                            "timestamp": timestamp,
                            "provider": provider,
                            "model": model,
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "total_tokens": total,
                            "estimated_cost_usd": cost,
                            "duration_ms": event.get("duration_ms"),
                        }

                    elif event_type == "provider_failed":
                        summary["failed_calls"] += 1

        except OSError:
            pass

        summary["by_provider"] = dict(per_provider)

        summary["by_model"] = dict(per_model)

        summary["by_day"] = sorted(
            (
                {
                    "date": day,
                    "tokens": values["tokens"],
                    "calls": values["calls"],
                }
                for day, values in per_day.items()
            ),
            key=lambda row: row["date"],
        )

        summary["last_call"] = last_completed

        return summary

    # =========================================================================
    # Direct providers
    # =========================================================================

    @classmethod
    def _execute_direct_gemini(
        cls,
        prompt: str,
        model: str,
        api_key: str,
    ) -> str:
        from google import genai

        client = genai.Client(api_key=api_key)

        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                usage_meta = getattr(
                    response,
                    "usage_metadata",
                    None,
                )

                if usage_meta is not None:
                    prompt_tokens = int(
                        getattr(
                            usage_meta,
                            "prompt_token_count",
                            0,
                        )
                        or 0
                    )

                    completion_tokens = int(
                        getattr(
                            usage_meta,
                            "candidates_token_count",
                            0,
                        )
                        or 0
                    )

                    cls._set_last_usage(
                        {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": (prompt_tokens + completion_tokens),
                        }
                    )

                else:
                    cls._set_last_usage(_empty_usage())

                result = getattr(
                    response,
                    "text",
                    None,
                )

                if not result:
                    raise RuntimeError("Gemini returned an empty response.")

                return result.strip()

            except Exception as exc:
                error_string = str(exc)

                if (
                    ("503" in error_string) or ("UNAVAILABLE" in error_string.upper())
                ) and attempt < max_retries - 1:
                    delay = (attempt + 1) * 2

                    logger.warning(
                        "Gemini 503 high-demand spike detected "
                        "on attempt %d/%d. Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        delay,
                    )

                    time.sleep(delay)
                    continue

                raise

        raise RuntimeError("Gemini execution failed.")

    @classmethod
    def _execute_direct_openai_style(
        cls,
        prompt: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> str:
        if base_url:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        else:
            client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt or "",
                }
            ],
            temperature=0.2,
        )

        cls._set_last_usage(_extract_openai_usage(response))

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Direct OpenAI-compatible API " "returned an empty response.")

        return response.choices[0].message.content.strip()

    # =========================================================================
    # Experiential Labs gateway
    # =========================================================================

    @classmethod
    def _execute_gateway(
        cls,
        prompt: str,
        model: str,
        provider: str,
        api_key: str | None = None,
    ) -> str:
        key = api_key.strip() if api_key and api_key.strip() else cls._get_gateway_key()

        client = openai.OpenAI(
            api_key=key,
            base_url=GATEWAY_BASE_URL,
        )

        system_instruction = (
            "You are an ATS optimization assistant. "
            "Respond ONLY with the requested content. "
            "Never include conversational intros, "
            "disclaimers, or meta-commentary."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": prompt or "",
                },
            ],
            temperature=0.2,
        )

        cls._set_last_usage(_extract_openai_usage(response))

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Experiential Labs gateway " "returned an empty response.")

        return response.choices[0].message.content.strip()

    # =========================================================================
    # Claude gateway
    # =========================================================================

    @classmethod
    def _execute_claude_gateway(
        cls,
        prompt: str,
        model: str,
        api_key: str | None = None,
    ) -> str:
        key = api_key.strip() if api_key and api_key.strip() else cls._get_gateway_key()

        # Important:
        # If ANTHROPIC_API_KEY is not configured, Claude is routed
        # through the Experiential Labs Anthropic-compatible endpoint.
        anthropic_key = os.getenv(
            "ANTHROPIC_API_KEY",
            "",
        ).strip()

        if anthropic_key:
            client = anthropic.Anthropic(api_key=anthropic_key)

        else:
            client = anthropic.Anthropic(
                api_key=key,
                base_url=CLAUDE_GATEWAY_BASE_URL,
            )

        system_instruction = (
            "You are an ATS optimization assistant. "
            "Respond ONLY with the requested content. "
            "Never include conversational intros, "
            "disclaimers, or meta-commentary."
        )

        # Keep this call deliberately simple.
        #
        # Do NOT use Responses API parameters here.
        # Claude's native API uses messages.create().
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_instruction,
            messages=[
                {
                    "role": "user",
                    "content": prompt or "",
                }
            ],
        )

        cls._set_last_usage(_extract_anthropic_usage(response))

        parts: list[str] = []

        for block in getattr(
            response,
            "content",
            [],
        ):
            block_type = getattr(
                block,
                "type",
                None,
            )

            if block_type == "text":
                block_text = getattr(
                    block,
                    "text",
                    "",
                )

                if block_text:
                    parts.append(block_text)

        result = "".join(parts).strip()

        if not result:
            raise RuntimeError("Anthropic / Experiential Claude " "returned an empty response.")

        return result

    # =========================================================================
    # Ollama
    # =========================================================================

    @classmethod
    def _execute_ollama(
        cls,
        prompt: str,
        model: str,
    ) -> str:
        raw_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434/api/generate",
        )

        url = _clean_base_url(
            raw_url,
            "http://localhost:11434/api/generate",
        )

        system_instruction = (
            "DO NOT OUTPUT CONVERSATIONAL INTROS, "
            "GREETINGS, FOOTERS, OR META-COMMENTARY. "
            "RETURN ONLY THE REQUESTED TEXT."
        )

        full_prompt = system_instruction + "\n\n" + (prompt or "")

        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama connection error " f"({response.status_code}): " f"{response.text[:500]}"
            )

        payload = response.json()

        prompt_tokens = int(
            payload.get(
                "prompt_eval_count",
                0,
            )
            or 0
        )

        completion_tokens = int(
            payload.get(
                "eval_count",
                0,
            )
            or 0
        )

        cls._set_last_usage(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens + completion_tokens),
            }
        )

        result = str(
            payload.get(
                "response",
                "",
            )
        ).strip()

        if not result:
            raise RuntimeError("Ollama returned an empty response.")

        return result

    # =========================================================================
    # Single provider router
    # =========================================================================

    @classmethod
    def _execute_single_provider(
        cls,
        prompt: str,
        provider: str,
        model_name: str | None = None,
        api_key: str | None = None,
        force_direct: bool = False,
    ) -> str:
        provider = (provider or "experiential").strip().lower()

        model = model_name or cls.get_default_model(provider)

        # --------------------------------------------------------------
        # Experiential Labs
        # --------------------------------------------------------------

        if provider == "experiential":
            return cls._execute_gateway(
                prompt=prompt,
                model=(
                    model
                    or os.getenv(
                        "EXPERIENTIAL_MODEL",
                        "gpt-5.6-luna",
                    )
                ),
                provider="experiential",
                api_key=api_key,
            )

        # --------------------------------------------------------------
        # Claude
        # --------------------------------------------------------------

        if provider == "claude":
            anthropic_key = (
                api_key
                or os.getenv(
                    "ANTHROPIC_API_KEY",
                    "",
                )
            ).strip()

            # Direct Anthropic API when explicitly requested
            # or when no gateway key is available.
            if anthropic_key and (force_direct or not EXPERIENTIAL_API_KEY):
                client = anthropic.Anthropic(api_key=anthropic_key)

                system_instruction = (
                    "You are an ATS optimization assistant. "
                    "Respond ONLY with the requested content. "
                    "Never include conversational intros, "
                    "disclaimers, or meta-commentary."
                )

                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_instruction,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt or "",
                        }
                    ],
                )

                cls._set_last_usage(_extract_anthropic_usage(response))

                parts: list[str] = []

                for block in getattr(
                    response,
                    "content",
                    [],
                ):
                    if (
                        getattr(
                            block,
                            "type",
                            None,
                        )
                        != "text"
                    ):
                        continue

                    block_text = getattr(
                        block,
                        "text",
                        "",
                    )

                    if block_text:
                        parts.append(block_text)

                result = "".join(parts).strip()

                if not result:
                    raise RuntimeError("Direct Anthropic API " "returned an empty response.")

                return result

            # Otherwise use the Experiential Claude gateway.
            return cls._execute_claude_gateway(
                prompt=prompt,
                model=model,
                api_key=api_key,
            )

        # --------------------------------------------------------------
        # Direct provider API keys
        # --------------------------------------------------------------

        direct_key = api_key

        if not direct_key:
            if provider == "gemini":
                direct_key = os.getenv("GEMINI_API_KEY")

            elif provider == "openai":
                direct_key = os.getenv("OPENAI_API_KEY")

            elif provider == "groq":
                direct_key = os.getenv("GROQ_API_KEY")

            elif provider == "openrouter":
                direct_key = os.getenv("OPENROUTER_API_KEY")

            elif provider == "deepseek":
                direct_key = os.getenv("DEEPSEEK_API_KEY")

        if direct_key and direct_key.strip():
            try:
                direct_key = direct_key.strip()

                if provider == "gemini":
                    return cls._execute_direct_gemini(
                        prompt,
                        model,
                        direct_key,
                    )

                if provider == "openai":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key,
                    )

                if provider == "groq":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key,
                        base_url=("https://api.groq.com/openai/v1"),
                    )

                if provider == "openrouter":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key,
                        base_url=("https://openrouter.ai/api/v1"),
                    )

                if provider == "deepseek":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key,
                        base_url=("https://api.deepseek.com"),
                    )

            except Exception as exc:
                if force_direct:
                    raise

                logger.warning(
                    "Direct execution for '%s' failed "
                    "(%s). Falling back to "
                    "Experiential Labs gateway...",
                    provider,
                    _safe_error(exc),
                )

        # --------------------------------------------------------------
        # Gateway fallback
        # --------------------------------------------------------------

        if force_direct:
            raise ValueError(
                f"Direct API Key for provider "
                f"'{provider}' is missing or invalid "
                f"in your .env file."
            )

        if provider in (
            "gemini",
            "groq",
            "openrouter",
            "deepseek",
            "openai",
        ):
            return cls._execute_gateway(
                prompt=prompt,
                model=model,
                provider=provider,
                api_key=api_key,
            )

        # --------------------------------------------------------------
        # Ollama
        # --------------------------------------------------------------

        if provider == "ollama":
            return cls._execute_ollama(
                prompt=prompt,
                model=model,
            )

        raise ValueError(f"Unsupported AI provider: {provider}")

    # =========================================================================
    # Public generate
    # =========================================================================

    @classmethod
    def generate(
        cls,
        prompt: str,
        provider: str = "experiential",
        model_name: str | None = None,
        api_key: str | None = None,
        route_mode: str = "experiential",
        **kwargs: Any,
    ) -> str:
        primary_provider = (provider or "experiential").strip().lower()

        is_direct = route_mode == "direct"

        # Experiential is always gateway-routed.
        if primary_provider == "experiential":
            is_direct = False

        # Direct means no provider fallback.
        if is_direct:
            fallback_chain = [primary_provider]

        else:
            fallback_chain = [
                primary_provider,
                "gemini",
                "openai",
                "deepseek",
                "groq",
                "openrouter",
                "claude",
                "ollama",
            ]

        providers_to_try: list[str] = []
        seen: set[str] = set()

        for current_provider in fallback_chain:
            if current_provider in seen:
                continue

            seen.add(current_provider)

            providers_to_try.append(current_provider)

        started = time.perf_counter()

        request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

        last_error: str | None = None

        for attempt_index, current_provider in enumerate(
            providers_to_try,
            start=1,
        ):
            selected_model = ""

            try:
                if current_provider == primary_provider and model_name:
                    selected_model = model_name

                else:
                    selected_model = cls.get_default_model(current_provider)

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "request_id": request_id,
                        "event": "request_started",
                        "route_mode": route_mode,
                        "provider": current_provider,
                        "model": selected_model,
                        "prompt_chars": len(prompt or ""),
                        "attempt": attempt_index,
                        "total_attempts": len(providers_to_try),
                        "status": "started",
                    }
                )

                logger.info(
                    "LLM request %s started: " "route=%s provider=%s model=%s",
                    request_id,
                    route_mode,
                    current_provider,
                    selected_model,
                )

                cls._set_last_usage(_empty_usage())

                raw_text = cls._execute_single_provider(
                    prompt=prompt,
                    provider=current_provider,
                    model_name=selected_model,
                    api_key=api_key,
                    force_direct=is_direct,
                )

                if not raw_text or not raw_text.strip():
                    raise RuntimeError("Provider returned an empty response.")

                cleaned_text = clean_llm_output(raw_text)

                if not cleaned_text:
                    raise RuntimeError("Provider returned empty " "content after cleaning.")

                usage = cls._pop_last_usage()

                cost = _estimate_cost_usd(
                    selected_model,
                    usage,
                )

                duration_ms = round(
                    (time.perf_counter() - started) * 1000,
                    1,
                )

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "request_id": request_id,
                        "event": "request_completed",
                        "route_mode": route_mode,
                        "provider": current_provider,
                        "model": selected_model,
                        "duration_ms": duration_ms,
                        "response_chars": len(cleaned_text),
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
                        "estimated_cost_usd": cost,
                        "status": "success",
                    }
                )

                logger.info(
                    "LLM request %s completed: " "%s/%s in %sms (%s tokens, $%s)",
                    request_id,
                    current_provider,
                    selected_model,
                    duration_ms,
                    usage.get(
                        "total_tokens",
                        0,
                    ),
                    cost,
                )

                return cleaned_text

            except Exception as exc:
                last_error = _safe_error(exc)

                error_lower = last_error.lower()

                if (
                    "429" in error_lower
                    or "resource_exhausted" in error_lower
                    or "rate limit" in error_lower
                    or "quota" in error_lower
                ):
                    try:
                        quota_tracker.record_rate_limit_event(
                            provider=current_provider,
                            model=selected_model,
                            error_message=last_error,
                        )

                        from app.core.event_log import log_event

                        log_event(
                            "quota",
                            "quota_warning",
                            provider=current_provider,
                            model=selected_model,
                            window="provider",
                            reason=last_error[:200],
                        )

                    except Exception:
                        pass

                logger.exception(
                    "LLM provider '%s' failed. " "Trying next fallback. " "Error: %s",
                    current_provider,
                    last_error,
                )

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "request_id": request_id,
                        "event": "provider_failed",
                        "route_mode": route_mode,
                        "provider": current_provider,
                        "model": selected_model,
                        "status": "error",
                        "error": last_error,
                    }
                )

        raise RuntimeError("All LLM execution options failed. " f"Last error: {last_error}")

    # =========================================================================
    # Log utilities
    # =========================================================================

    @classmethod
    def recent_logs(
        cls,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not LOG_PATH.exists():
            return []

        rows: list[dict[str, Any]] = []

        try:
            with LOG_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:
                lines = file.readlines()

            for line in lines[-max(1, limit) :]:
                try:
                    rows.append(json.loads(line))

                except json.JSONDecodeError:
                    continue

        except OSError:
            return []

        return rows

    @classmethod
    def clear_logs(cls) -> None:
        try:
            if LOG_PATH.exists():
                LOG_PATH.unlink()

        except OSError:
            pass


# ============================================================================
# Public exports
# ============================================================================

__all__ = [
    "LOG_PATH",
    "LLMService",
    "clean_llm_output",
]
