"""Interactive command-line launcher for automated job hunting."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from app.services.jobs.auto_runner import run_automated_job_hunting

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s - %(levelname)s - " "%(name)s - %(message)s"),
)

logger = logging.getLogger(__name__)


# ============================================================================
# AVAILABLE LLM CONFIGURATIONS
# ============================================================================

MODEL_OPTIONS: List[Dict[str, Any]] = [
    {
        "key": "experiential_luna",
        "label": ("Experiential Labs - GPT-5.6 Luna"),
        "provider": "experiential",
        "route_mode": "experiential",
        "model": "gpt-5.6-luna",
        "agent": "experiential",
        "description": ("Experiential hosted route; GPT-5.6 Luna"),
    },
    {
        "key": "experiential_astra",
        "label": ("Experiential Labs - GPT-6 Astra"),
        "provider": "experiential",
        "route_mode": "experiential",
        "model": "gpt-6-astra",
        "agent": "experiential",
        "description": ("Experiential hosted route; GPT-6 Astra"),
    },
    {
        "key": "experiential_deepseek",
        "label": ("Experiential Labs - DeepSeek V4 Flash"),
        "provider": "experiential",
        "route_mode": "experiential",
        "model": "deepseek-v4-flash",
        "agent": "experiential",
        "description": ("Experiential hosted DeepSeek route"),
    },
    {
        "key": "experiential_qwen",
        "label": ("Experiential Labs - Qwen3.8 27B"),
        "provider": "experiential",
        "route_mode": "experiential",
        "model": "qwen3.8-27b",
        "agent": "experiential",
        "description": ("Experiential hosted Qwen route"),
    },
    {
        "key": "gemini",
        "label": "Google Gemini - Gemini 2.5 Flash",
        "provider": "gemini",
        "route_mode": "direct",
        "model": "gemini-2.5-flash",
        "agent": "direct-gemini",
        "description": ("Direct Google Gemini API"),
    },
    {
        "key": "openai",
        "label": "OpenAI - GPT-5.6 Luna",
        "provider": "openai",
        "route_mode": "direct",
        "model": "gpt-5.6-luna",
        "agent": "direct-openai",
        "description": ("Direct OpenAI-compatible API"),
    },
    {
        "key": "claude",
        "label": "Anthropic - Claude Fable 5",
        "provider": "claude",
        "route_mode": "direct",
        "model": "claude-fable-5",
        "agent": "direct-claude",
        "description": ("Direct Anthropic-compatible API"),
    },
    {
        "key": "deepseek",
        "label": "DeepSeek - DeepSeek V4 Flash",
        "provider": "deepseek",
        "route_mode": "direct",
        "model": "deepseek-v4-flash",
        "agent": "direct-deepseek",
        "description": ("Direct DeepSeek API"),
    },
    {
        "key": "groq",
        "label": "Groq - Llama 3.3 70B",
        "provider": "groq",
        "route_mode": "direct",
        "model": "llama-3.3-70b-versatile",
        "agent": "direct-groq",
        "description": ("Direct Groq API"),
    },
    {
        "key": "openrouter",
        "label": "OpenRouter",
        "provider": "openrouter",
        "route_mode": "direct",
        "model": "deepseek-v4-flash",
        "agent": "direct-openrouter",
        "description": ("OpenRouter API"),
    },
    {
        "key": "ollama",
        "label": "Ollama - Local",
        "provider": "ollama",
        "route_mode": "direct",
        "model": "qwen3.8-27b",
        "agent": "ollama",
        "description": ("Local Ollama model"),
    },
]


def _ask_choice(
    title: str,
    options: List[Dict[str, Any]],
    default: int = 1,
) -> Dict[str, Any]:
    """Display numbered options and return the selected item."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for index, option in enumerate(
        options,
        start=1,
    ):
        print(f"{index:2d}. " f"{option['label']}")
        print(f"    {option['description']}")

    while True:
        raw = input(f"\nSelect [{default}]: ").strip()

        if not raw:
            return options[default - 1]

        try:
            number = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if 1 <= number <= len(options):
            return options[number - 1]

        print(f"Please choose 1-{len(options)}.")


def _ask_int(
    prompt: str,
    default: int,
    minimum: int = 1,
) -> int:
    """Ask for an integer."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()

        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if value >= minimum:
            return value

        print(f"Value must be at least {minimum}.")


def _ask_yes_no(
    prompt: str,
    default: bool = False,
) -> bool:
    """Ask a yes/no question."""
    default_text = "Y/n" if default else "y/N"

    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()

        if not raw:
            return default

        if raw in {
            "y",
            "yes",
        }:
            return True

        if raw in {
            "n",
            "no",
        }:
            return False

        print("Please enter y or n.")


def _print_configuration(
    config: Dict[str, Any],
    agent: Dict[str, Any],
    applications: int,
    auto_submit: bool,
) -> None:
    """Display the final selected configuration."""

    print()
    print("=" * 70)
    print("AUTOMATED JOB HUNTING ENGINE")
    print("=" * 70)

    print(f"Text Model:   {config['model']}")
    print(f"Provider:     {config['provider']}")
    print(f"Route:        {config['route_mode']}")
    print(f"Agent Model:  {agent['model']}")
    print(f"Agent Route:  {agent['route_mode']}")
    print(f"Applications: {applications}")
    print(f"Auto-submit:  {auto_submit}")

    print("=" * 70)


async def main() -> None:
    """Interactive application launcher."""

    print()
    print("=" * 70)
    print("AUTOMATED JOB HUNTING ENGINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # TEXT MODEL
    # ------------------------------------------------------------------

    selected = _ask_choice(
        "SELECT TEXT-GENERATION API / MODEL",
        MODEL_OPTIONS,
        default=1,
    )

    # ------------------------------------------------------------------
    # BROWSER USE MODEL
    # ------------------------------------------------------------------

    agent_options = [
        {
            "label": "Experiential Labs",
            "provider": "experiential",
            "route_mode": "experiential",
            "model": "experiential",
            "description": ("Use the Experiential gateway for Browser Use"),
        },
        {
            "label": "Direct Gemini",
            "provider": "gemini",
            "route_mode": "direct",
            "model": "direct-gemini",
            "description": ("Use Gemini directly for Browser Use"),
        },
        {
            "label": "Direct OpenAI",
            "provider": "openai",
            "route_mode": "direct",
            "model": "direct-openai",
            "description": ("Use OpenAI directly for Browser Use"),
        },
        {
            "label": "Direct Claude",
            "provider": "claude",
            "route_mode": "direct",
            "model": "direct-claude",
            "description": ("Use Claude directly for Browser Use"),
        },
        {
            "label": "Automatic",
            "provider": "auto",
            "route_mode": "auto",
            "model": "auto",
            "description": ("Let Browser Use choose its configured model"),
        },
    ]

    selected_agent = _ask_choice(
        "SELECT BROWSER USE AGENT",
        agent_options,
        default=1,
    )

    # ------------------------------------------------------------------
    # APPLICATION COUNT
    # ------------------------------------------------------------------

    applications = _ask_int(
        "\nMaximum applications",
        default=5,
        minimum=1,
    )

    # ------------------------------------------------------------------
    # AUTO SUBMIT
    # ------------------------------------------------------------------

    auto_submit = _ask_yes_no(
        "Allow automatic submission?",
        default=False,
    )

    _print_configuration(
        selected,
        selected_agent,
        applications,
        auto_submit,
    )

    print()
    start = _ask_yes_no(
        "Start job search?",
        default=True,
    )

    if not start:
        print("\nJob search cancelled.")
        return

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    try:
        results = await run_automated_job_hunting(
            search_term="DevOps Engineer",
            location="Germany",
            resume_path="data/resume.pdf",
            max_applications=applications,
            auto_submit=auto_submit,
            model_name=selected["model"],
            agent_llm=selected_agent["model"],
            provider=selected["provider"],
            route_mode=selected["route_mode"],
        )

    except KeyboardInterrupt:
        print("\nJob hunting interrupted by user.")
        return

    except Exception:
        logger.exception("Job hunting failed.")
        return

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("JOB HUNTING RESULTS")
    print("=" * 70)

    if not results:
        print("No jobs were processed.")
        return

    successful = 0
    submitted = 0

    for index, result in enumerate(
        results,
        start=1,
    ):
        if not isinstance(result, dict):
            print(f"\n{index}. Invalid result: " f"{result}")
            continue

        job_title = result.get(
            "job_title",
            "Unknown",
        )
        company = result.get(
            "company",
            "Unknown",
        )

        original_score = result.get("original_ats_score")

        final_score = result.get("final_ats_score")

        cv_version = result.get(
            "selected_cv_version",
            "unknown",
        )

        cover_status = result.get(
            "cover_letter_status",
            "unknown",
        )

        browser = result.get(
            "browser_application",
            {},
        )

        if not isinstance(browser, dict):
            browser = {}

        success = bool(
            result.get(
                "success",
                browser.get(
                    "success",
                    False,
                ),
            )
        )

        was_submitted = bool(
            result.get(
                "submitted",
                browser.get(
                    "submitted",
                    False,
                ),
            )
        )

        if success:
            successful += 1

        if was_submitted:
            submitted += 1

        print()
        print(f"{index}. {job_title}")
        print(f"   Company:       {company}")
        print(f"   Original ATS:  {original_score}%")
        print(f"   Final ATS:     {final_score}%")
        print(f"   Selected CV:   {cv_version}")
        print(f"   Cover Letter:  {cover_status}")
        print(f"   Browser Use:   " f"{'SUCCESS' if success else 'FAILED'}")
        print(f"   Submitted:     " f"{'YES' if was_submitted else 'NO'}")

        if result.get("cv_path"):
            print(f"   CV:            " f"{result['cv_path']}")

        if result.get("cover_letter_path"):
            print(f"   Cover Letter:  " f"{result['cover_letter_path']}")

        error = result.get("error") or browser.get("error_message")

        if error:
            print(f"   Error:         {error}")

    print()
    print("=" * 70)
    print(f"Processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Submitted: {submitted}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
