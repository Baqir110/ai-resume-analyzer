from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic
import openai
import requests
from dotenv import load_dotenv
from google import genai
from groq import Groq

# ============================================================
# Environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)


# ============================================================
# Processing log
# ============================================================

DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "llm_processing.jsonl"

LOG_PATH = Path(
    os.getenv(
        "LLM_PROCESSING_LOG",
        str(DEFAULT_LOG_PATH),
    )
)

if not LOG_PATH.is_absolute():
    LOG_PATH = PROJECT_ROOT / LOG_PATH


# ============================================================
# Provider configuration
# ============================================================

PROVIDER_ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


DEFAULT_MODELS = {
    "gemini": os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ),
    "groq": os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    ),
    "openrouter": os.getenv(
        "OPENROUTER_MODEL",
        "deepseek/deepseek-chat",
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
        "llama3",
    ),
}


# ============================================================
# Logging & Sanitization
# ============================================================


def clean_llm_output(text: str) -> str:
    """
    Removes Markdown code fences and AI conversational meta-text from any provider's output.
    """
    if not text:
        return ""

    text = text.strip()

    # Remove Markdown code fences (e.g., ```latex, ```html, ```markdown)
    text = re.sub(
        r"^\s*```(?:latex|tex|html|markdown|json)?\s*", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Remove standard conversational intro phrases across all LLMs
    conversational_intros = [
        r"^Here is the (?:optimized|tailored|updated|generated|LaTeX|HTML|resume|CV).*",
        r"^Sure,? I can help with that.*",
        r"^I'm ready to (?:enrich|optimize|help).*",
        r"^Certainly! Here is.*",
        r"^Below is the.*",
    ]

    for pattern in conversational_intros:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return text


def _clean_base_url(url: str, default: str) -> str:
    """Ensures base URLs are well-formed with http/https protocols."""
    if not url:
        return default
    url = url.strip()
    # Strip markdown link wrappers if present (e.g. [https://...](https://...))
    match = re.search(r"https?://[^\s\)]+", url)
    if match:
        return match.group(0)
    if not url.startswith("http://") and not url.startswith("https://"):
        return f"https://{url}"
    return url


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_log(event: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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
        pass


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    sensitive_values = []

    for env_name in PROVIDER_ENV_KEYS.values():
        value = os.getenv(env_name)
        if value:
            sensitive_values.append(value)

    workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "")
    if workspace:
        sensitive_values.append(workspace)

    for value in sensitive_values:
        if value:
            text = text.replace(value, "***REDACTED***")

    return text[:2000]


# ============================================================
# LLM Service
# ============================================================


class LLMService:

    DEFAULT_MODELS = DEFAULT_MODELS

    SUPPORTED_PROVIDERS = (
        "gemini",
        "groq",
        "openrouter",
        "deepseek",
        "openai",
        "claude",
        "ollama",
    )

    @classmethod
    def get_default_model(
        cls,
        provider: str,
    ) -> str:
        provider = (provider or "gemini").strip().lower()
        return cls.DEFAULT_MODELS.get(provider, "")

    @classmethod
    def provider_status(cls) -> list[dict[str, Any]]:
        rows = []
        for provider in cls.SUPPORTED_PROVIDERS:
            model = cls.get_default_model(provider)

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
                continue

            env_name = PROVIDER_ENV_KEYS[provider]
            configured = bool(os.getenv(env_name, "").strip())

            extra = {}
            if provider == "claude":
                extra["workspace_configured"] = bool(
                    os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
                )

            rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "configured": configured,
                    "authentication": "api_key",
                    **extra,
                }
            )

        return rows

    @classmethod
    def _get_api_key(
        cls,
        provider: str,
        custom_key: Optional[str] = None,
    ) -> str:
        if custom_key and custom_key.strip():
            return custom_key.strip()

        env_name = PROVIDER_ENV_KEYS.get(provider)
        if not env_name:
            return ""

        key = os.getenv(env_name, "").strip()
        if not key:
            raise ValueError(f"{env_name} is not configured in .env.")

        return key

    @classmethod
    def _get_claude_workspace_id(cls) -> str:
        workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        if not workspace:
            raise ValueError(
                "ANTHROPIC_WORKSPACE_ID is not configured in .env. "
                "Identity-linked Anthropic API keys require a workspace ID."
            )
        return workspace

    @classmethod
    def generate(
        cls,
        prompt: str,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> str:

        provider = (provider or "gemini").strip().lower()

        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported AI provider: {provider}")

        model = model_name or cls.get_default_model(provider)

        # Enforce strict system-level formatting instructions
        system_instruction = (
            "[SYSTEM INSTRUCTION: DO NOT OUTPUT CONVERSATIONAL INTROS, GREETINGS, "
            "FOOTERS, OR META-COMMENTARY. DO NOT ASK QUESTIONS. RETURN ONLY THE EXACT "
            "REQUESTED CODE OR TEXT FORMAT.]\n\n"
        )
        full_prompt = system_instruction + prompt

        started = time.perf_counter()
        request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

        _write_log(
            {
                "timestamp": _utc_now(),
                "request_id": request_id,
                "event": "request_started",
                "provider": provider,
                "model": model,
                "prompt_chars": len(full_prompt or ""),
                "status": "started",
            }
        )

        try:
            # =================================================
            # GEMINI
            # =================================================
            if provider == "gemini":
                key = cls._get_api_key("gemini", custom_key=api_key)
                client = genai.Client(api_key=key)
                # Recommended Chat API method to avoid automatic function calling warnings
                chat = client.chats.create(model=model)
                response = chat.send_message(full_prompt)
                text = getattr(response, "text", None) or ""

            # =================================================
            # GROQ
            # =================================================
            elif provider == "groq":
                key = cls._get_api_key("groq", custom_key=api_key)
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""

            # =================================================
            # OPENROUTER
            # =================================================
            elif provider == "openrouter":
                key = cls._get_api_key("openrouter", custom_key=api_key)
                raw_url = os.getenv("OPENROUTER_BASE_URL", "")
                base_url = _clean_base_url(raw_url, "(https://openrouter.ai/api/v1)")

                client = openai.OpenAI(
                    api_key=key,
                    base_url=base_url,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""

            # =================================================
            # DEEPSEEK
            # =================================================
            elif provider == "deepseek":
                key = cls._get_api_key("deepseek", custom_key=api_key)
                raw_url = os.getenv("DEEPSEEK_BASE_URL", "")
                base_url = _clean_base_url(
                    raw_url, "[https://api.deepseek.com](https://api.deepseek.com)"
                )

                client = openai.OpenAI(
                    api_key=key,
                    base_url=base_url,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""

            # =================================================
            # OPENAI
            # =================================================
            elif provider == "openai":
                key = cls._get_api_key("openai", custom_key=api_key)
                client = openai.OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""

            # =================================================
            # CLAUDE
            # =================================================
            elif provider == "claude":
                key = cls._get_api_key("claude", custom_key=api_key)
                workspace_id = cls._get_claude_workspace_id()
                client = anthropic.Anthropic(
                    api_key=key,
                    default_headers={"anthropic-workspace-id": workspace_id},
                )
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=0.3,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                text = "".join(
                    block.text
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                )

            # =================================================
            # OLLAMA
            # =================================================
            elif provider == "ollama":
                raw_url = os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434/api/generate",
                )
                url = _clean_base_url(raw_url, "http://localhost:11434/api/generate")

                response = requests.post(
                    url,
                    json={"model": model, "prompt": full_prompt, "stream": False},
                    timeout=120,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Ollama connection error ({response.status_code})"
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError("Ollama returned invalid JSON.") from exc

                text = payload.get("response", "") or ""

            else:
                raise ValueError(f"Unsupported AI provider: {provider}")

            if not text.strip():
                raise RuntimeError("The AI provider returned an empty response.")

            # Clean and sanitize output
            cleaned_text = clean_llm_output(text)

            duration_ms = round((time.perf_counter() - started) * 1000, 1)

            _write_log(
                {
                    "timestamp": _utc_now(),
                    "request_id": request_id,
                    "event": "request_completed",
                    "provider": provider,
                    "model": model,
                    "duration_ms": duration_ms,
                    "prompt_chars": len(full_prompt or ""),
                    "response_chars": len(cleaned_text),
                    "status": "success",
                }
            )

            return cleaned_text

        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            error_text = _safe_error(exc)

            _write_log(
                {
                    "timestamp": _utc_now(),
                    "request_id": request_id,
                    "event": "request_failed",
                    "provider": provider,
                    "model": model,
                    "duration_ms": duration_ms,
                    "prompt_chars": len(full_prompt or ""),
                    "status": "error",
                    "error": error_text,
                }
            )

            raise RuntimeError(
                f"AI Provider Error ({provider.upper()}): {error_text}"
            ) from exc

    @classmethod
    def recent_logs(cls, limit: int = 100) -> list[dict[str, Any]]:
        if not LOG_PATH.exists():
            return []

        rows = []
        try:
            with LOG_PATH.open("r", encoding="utf-8") as file:
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


__all__ = [
    "LLMService",
    "LOG_PATH",
    "clean_llm_output",
]
