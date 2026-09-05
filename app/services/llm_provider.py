from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(
    ENV_PATH,
    override=True,
)

DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "llm_processing.jsonl"
LOG_PATH = Path(
    os.getenv(
        "LLM_PROCESSING_LOG",
        str(DEFAULT_LOG_PATH),
    )
)
if not LOG_PATH.is_absolute():
    LOG_PATH = PROJECT_ROOT / LOG_PATH

# Experiential Gateway Configuration
GATEWAY_BASE_URL = (
    os.getenv(
        "OPENAI_BASE_URL",
        "https://api.experientiallabs.ai/v1",
    )
    .strip()
    .rstrip("/")
)

CLAUDE_GATEWAY_BASE_URL = (
    GATEWAY_BASE_URL[:-3] if GATEWAY_BASE_URL.endswith("/v1") else GATEWAY_BASE_URL
).rstrip("/")

EXPERIENTIAL_API_KEY = (
    os.getenv("EXPLABS_API_KEY", "").strip()
    or os.getenv("EXPERIENTIAL_ORG_KEY", "").strip()
)

DEFAULT_MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "groq": os.getenv("GROQ_MODEL", "qwen3.8-27b"),
    "openrouter": os.getenv("OPENROUTER_MODEL", "deepseek-v4-flash"),
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
    "claude": os.getenv("CLAUDE_MODEL", "claude-fable-5"),
    "ollama": os.getenv("OLLAMA_MODEL", "qwen3.8-27b"),
}

SUPPORTED_PROVIDERS = (
    "gemini",
    "groq",
    "openrouter",
    "deepseek",
    "openai",
    "claude",
    "ollama",
)


def clean_llm_output(text: str) -> str:
    """Clean common LLM formatting artifacts."""
    if not text:
        return ""

    text = text.strip()
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


def _clean_base_url(url: str, default: str) -> str:
    if not url:
        return default
    url = url.strip()

    markdown_match = re.search(r"\]\((https?://[^)]+)\)", url)
    if markdown_match:
        return markdown_match.group(1).rstrip("/")

    match = re.search(r"https?://[^\s)]+", url)
    if match:
        return match.group(0).rstrip("/")

    if not url.startswith(("http://", "https://")):
        return f"https://{url}".rstrip("/")

    return url.rstrip("/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_log(event: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    sensitive_values = []

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
    ]

    for env_name in provider_key_names:
        value = os.getenv(env_name, "").strip()
        if value:
            sensitive_values.append(value)

    for value in sensitive_values:
        if value:
            text = text.replace(value, "***REDACTED***")

    return text[:2000]


class LLMService:

    DEFAULT_MODELS = DEFAULT_MODELS
    SUPPORTED_PROVIDERS = SUPPORTED_PROVIDERS

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        provider = (provider or "gemini").strip().lower()
        return cls.DEFAULT_MODELS.get(provider, "")

    @classmethod
    def _get_gateway_key(cls) -> str:
        key = (
            os.getenv("EXPLABS_API_KEY", "").strip()
            or os.getenv("EXPERIENTIAL_ORG_KEY", "").strip()
        )
        if not key:
            raise ValueError(
                "EXPLABS_API_KEY / EXPERIENTIAL_ORG_KEY is not configured."
            )
        return key

    @classmethod
    def provider_status(cls) -> list[dict[str, Any]]:
        rows = []
        for provider in cls.SUPPORTED_PROVIDERS:
            model = cls.get_default_model(provider)

            if provider == "claude":
                configured = bool(EXPERIENTIAL_API_KEY)
                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "configured": configured,
                        "authentication": "experiential_gateway_forced",
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
                configured = bool(EXPERIENTIAL_API_KEY)
                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "configured": configured,
                        "authentication": "experiential_gateway",
                        "gateway": GATEWAY_BASE_URL,
                    }
                )

        return rows

    # ----------------------------------------------------
    # DIRECT PROVIDER EXECUTION ENGINE
    # ----------------------------------------------------

    @classmethod
    def _execute_direct_gemini(cls, prompt: str, model: str, api_key: str) -> str:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text.strip()

    @classmethod
    def _execute_direct_openai_style(
        cls, prompt: str, model: str, api_key: str, base_url: Optional[str] = None
    ) -> str:
        client = (
            openai.OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else openai.OpenAI(api_key=api_key)
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Direct OpenAI-compatible API returned empty response.")
        return response.choices[0].message.content.strip()

    # ----------------------------------------------------
    # GATEWAY EXECUTION ENGINE
    # ----------------------------------------------------

    @classmethod
    def _execute_gateway(
        cls,
        prompt: str,
        model: str,
        provider: str,
        api_key: Optional[str] = None,
    ) -> str:
        key = api_key.strip() if api_key and api_key.strip() else cls._get_gateway_key()

        client = openai.OpenAI(
            api_key=key,
            base_url=GATEWAY_BASE_URL,
        )

        system_instruction = (
            "[SYSTEM INSTRUCTION: DO NOT OUTPUT CONVERSATIONAL INTROS, GREETINGS, "
            "FOOTERS, OR META-COMMENTARY. DO NOT ASK QUESTIONS. RETURN ONLY THE EXACT REQUESTED CODE OR TEXT FORMAT.]"
        )

        full_prompt = system_instruction + "\n\n" + (prompt or "")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
        )

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Experiential Labs gateway returned an empty response.")

        return response.choices[0].message.content.strip()

    @classmethod
    def _execute_claude_gateway(
        cls,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
    ) -> str:
        key = api_key.strip() if api_key and api_key.strip() else cls._get_gateway_key()

        client = anthropic.Anthropic(
            api_key=key,
            base_url=CLAUDE_GATEWAY_BASE_URL,
        )

        system_instruction = (
            "DO NOT OUTPUT CONVERSATIONAL INTROS, GREETINGS, FOOTERS, OR META-COMMENTARY. "
            "DO NOT ASK QUESTIONS. RETURN ONLY THE EXACT REQUESTED CODE OR TEXT FORMAT."
        )

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt or ""}],
        )

        parts = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        result = "".join(parts).strip()

        if not result:
            raise RuntimeError("Experiential Labs Claude returned an empty response.")

        return result

    @classmethod
    def _execute_ollama(cls, prompt: str, model: str) -> str:
        raw_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
        url = _clean_base_url(raw_url, "http://localhost:11434/api/generate")

        system_instruction = (
            "DO NOT OUTPUT CONVERSATIONAL INTROS, GREETINGS, FOOTERS, OR META-COMMENTARY. "
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
                f"Ollama connection error ({response.status_code}): {response.text[:500]}"
            )

        payload = response.json()
        result = payload.get("response", "").strip()

        if not result:
            raise RuntimeError("Ollama returned an empty response.")

        return result

    # ----------------------------------------------------
    # SINGLE PROVIDER ROUTER
    # ----------------------------------------------------

    @classmethod
    def _execute_single_provider(
        cls,
        prompt: str,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        provider = (provider or "gemini").strip().lower()
        model = model_name or cls.get_default_model(provider)

        # Strictly route Claude requests through Experiential Gateway
        if provider == "claude":
            return cls._execute_claude_gateway(
                prompt=prompt, model=model, api_key=api_key
            )

        # 1. Check for Direct API Keys for non-Claude providers
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
                if provider == "gemini":
                    return cls._execute_direct_gemini(prompt, model, direct_key.strip())
                elif provider == "openai":
                    return cls._execute_direct_openai_style(
                        prompt, model, direct_key.strip()
                    )
                elif provider == "groq":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key.strip(),
                        base_url="https://api.groq.com/openai/v1",
                    )
                elif provider == "openrouter":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key.strip(),
                        base_url="https://openrouter.ai/api/v1",
                    )
                elif provider == "deepseek":
                    return cls._execute_direct_openai_style(
                        prompt,
                        model,
                        direct_key.strip(),
                        base_url="https://api.deepseek.com",
                    )
            except Exception as exc:
                logger.warning(
                    f"Direct execution for '{provider}' failed ({_safe_error(exc)}). "
                    f"Falling back to Experiential Labs gateway..."
                )

        # 2. Gateway Execution Fallback for remaining providers
        if provider in ("gemini", "groq", "openrouter", "deepseek", "openai"):
            return cls._execute_gateway(
                prompt=prompt, model=model, provider=provider, api_key=api_key
            )

        if provider == "ollama":
            return cls._execute_ollama(prompt=prompt, model=model)

        raise ValueError(f"Unsupported AI provider: {provider}")

    @classmethod
    def generate(
        cls,
        prompt: str,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        primary_provider = (provider or "gemini").strip().lower()

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

        providers_to_try = []
        seen = set()

        for current_provider in fallback_chain:
            if current_provider not in seen:
                seen.add(current_provider)
                providers_to_try.append(current_provider)

        started = time.perf_counter()
        request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        last_error = None

        for current_provider in providers_to_try:
            try:
                selected_model = (
                    model_name
                    if (current_provider == primary_provider)
                    else cls.get_default_model(current_provider)
                )

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "request_id": request_id,
                        "event": "request_started",
                        "provider": current_provider,
                        "model": selected_model,
                        "prompt_chars": len(prompt or ""),
                        "status": "started",
                    }
                )

                logger.info(
                    "LLM request %s started: provider=%s model=%s",
                    request_id,
                    current_provider,
                    selected_model,
                )

                raw_text = cls._execute_single_provider(
                    prompt=prompt,
                    provider=current_provider,
                    model_name=(
                        model_name if (current_provider == primary_provider) else None
                    ),
                    api_key=api_key,
                )

                if not raw_text or not raw_text.strip():
                    raise RuntimeError("Provider returned an empty response.")

                cleaned_text = clean_llm_output(raw_text)

                if not cleaned_text:
                    raise RuntimeError(
                        "Provider returned empty content after cleaning."
                    )

                duration_ms = round((time.perf_counter() - started) * 1000, 1)

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "request_id": request_id,
                        "event": "request_completed",
                        "provider": current_provider,
                        "model": selected_model,
                        "duration_ms": duration_ms,
                        "response_chars": len(cleaned_text),
                        "status": "success",
                    }
                )

                logger.info(
                    "LLM request %s completed: %s/%s in %sms",
                    request_id,
                    current_provider,
                    selected_model,
                    duration_ms,
                )

                return cleaned_text

            except Exception as exc:
                last_error = _safe_error(exc)

                logger.exception(
                    "LLM provider '%s' failed. Trying next fallback. Error: %s",
                    current_provider,
                    last_error,
                )

                _write_log(
                    {
                        "timestamp": _utc_now(),
                        "request_id": request_id,
                        "event": "provider_failed",
                        "provider": current_provider,
                        "model": selected_model,
                        "status": "error",
                        "error": last_error,
                    }
                )

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

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
