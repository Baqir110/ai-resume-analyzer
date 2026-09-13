# app/services/jobs/browser_use_applier.py
"""
AI-driven job application submission using Browser Use.

Attaches to the user's real Microsoft Edge browser via CDP
(Chrome DevTools Protocol) so all existing logins and sessions
are reused. Edge must be running with remote debugging enabled:

    scripts\\start_edge_debug.bat

Fills ATS forms (Workday, Greenhouse, Lever, Ashby) with structured
applicant data, uploads tailored PDFs, and pauses for human review
before submit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = PROJECT_ROOT / "data" / "applicant_profile.yaml"

# Where Edge exposes its DevTools protocol when launched with
# --remote-debugging-port=9222 (see scripts/start_edge_debug.bat).
EDGE_CDP_URL = os.getenv("EDGE_CDP_URL", "http://localhost:9222")


@dataclass
class ApplicationResult:
    """Outcome of one application attempt."""

    success: bool
    job_url: str
    steps_taken: int = 0
    error_message: Optional[str] = None
    captcha_encountered: bool = False
    paused_for_human: bool = False
    final_url: Optional[str] = None
    history_summary: list[str] = field(default_factory=list)


def _check_edge_cdp() -> None:
    """
    Verify Edge is running with remote debugging on EDGE_CDP_URL.

    Raises RuntimeError with a clear message if not.
    """
    version_url = f"{EDGE_CDP_URL}/json/version"
    try:
        with urllib.request.urlopen(version_url, timeout=2) as r:
            data = r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Edge on {EDGE_CDP_URL}.\n"
            "Launch Edge first with:  scripts\\start_edge_debug.bat\n"
            "Then leave Edge open and start the dashboard in another terminal."
        ) from exc

    if "Edg/" not in data:
        raise RuntimeError(
            f"Something is listening on {EDGE_CDP_URL} but it is not Microsoft Edge.\n"
            "Close any Chrome/Chromium instances, then re-run:\n"
            "  scripts\\start_edge_debug.bat"
        )


class BrowserUseApplier:
    """
    Fills job application forms with applicant data.

    Usage:
        applier = BrowserUseApplier()
        result = await applier.apply(
            job_url="https://boards.greenhouse.io/acme/jobs/12345",
            resume_path="data/outputs/resume_acme.pdf",
            cover_letter_path="data/outputs/cover_letter_acme.pdf",
            why_this_company="Acme's infrastructure challenges align with...",
        )
    """

    def __init__(self, profile_path: Path = PROFILE_PATH):
        self.profile_path = profile_path
        with open(profile_path, "r", encoding="utf-8") as f:
            self.profile = yaml.safe_load(f)
        self.personal = self.profile.get("personal", {})
        self.free_text = dict(self.profile.get("free_text", {}))
        self.demographics = self.profile.get("demographics", {})
        self.work_auth = self.profile.get("work_authorization", {})

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_task_prompt(
        self,
        job_url: str,
        resume_path: str,
        cover_letter_path: Optional[str],
        why_this_company: str,
        company_name: str = "",
        role_title: str = "",
    ) -> str:
        """
        Build the natural-language task for the agent.

        Be explicit. Browser Use reads this literally and decides
        each click/type from it. Vague prompts cause loops.
        """
        p = self.personal

        # Merge the per-job free-text answer into a local copy
        # (do not mutate self.free_text across calls).
        free_text = dict(self.free_text)
        if why_this_company:
            free_text["why_this_company"] = why_this_company

        free_text_block = "\n".join(
            f'  - If asked "{q}": answer "{a}"' for q, a in free_text.items() if a
        )

        cover_block = ""
        if cover_letter_path:
            cover_block = (
                f"\n4. Upload the cover letter PDF from: {cover_letter_path}\n"
                f"   (if the form has a separate cover letter field)"
            )

        # Guard against a profile with fewer than 2 experience entries.
        exp = self.profile.get("experience", [])
        recent_exp = exp[0] if len(exp) > 0 else {}
        prev_exp = exp[1] if len(exp) > 1 else {}

        edu = self.profile.get("education", [])
        recent_edu = edu[0] if edu else {}

        return f"""
You are filling out a job application form. Be precise and literal.
Do not invent any information. Only use the data below.

Job URL: {job_url}
Target role: {role_title or 'the advertised position'}
Company: {company_name or 'the company'}

APPLICANT DATA (use these exact values):
  First name: {p.get('first_name', '')}
  Last name: {p.get('last_name', '')}
  Full name: {p.get('full_name', '')}
  Email: {p.get('email', '')}
  Phone: {p.get('phone', '')}
  City: {p.get('city', '')}
  Country: {p.get('country', '')}
  LinkedIn: {p.get('linkedin', '')}
  GitHub: {p.get('github', '')}
  Authorized to work: {'Yes' if self.work_auth.get('authorized_to_work') else 'No'}
  Requires sponsorship: {'Yes' if self.work_auth.get('requires_sponsorship') else 'No'}

WORK HISTORY (for fields that ask for most recent employer):
  Most recent: {recent_exp.get('title', '')} at {recent_exp.get('company', '')}
  Previous: {prev_exp.get('title', '')} at {prev_exp.get('company', '')}

EDUCATION:
  Most recent: {recent_edu.get('degree', '')} — {recent_edu.get('institution', '')}

FREE-TEXT ANSWERS (only use these when the form asks the matching question):
{free_text_block}

STEPS:
1. Navigate to the job URL.
2. If there is an "Apply" button that reveals the form, click it.
3. Fill every required field using the APPLICANT DATA above.
   - For dropdowns, select the option that matches.
   - For "How did you hear about us?", pick "LinkedIn" or "Company website".
   - For work authorization / sponsorship, use the values above.
4. Upload the resume PDF from: {resume_path}{cover_block}
5. Review the form. Do NOT click Submit.
6. STOP. Report the current URL and a summary of what you filled.

CRITICAL:
- If you encounter a CAPTCHA, STOP immediately and report "CAPTCHA encountered".
- If a field is ambiguous, leave it blank rather than guessing.
- Do not click Submit under any circumstances.
""".strip()

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _make_llm(self):
        """
        Build the LLM the agent will use.

        Uses your Experiential Labs gateway via OpenAI-compatible
        base_url. Falls back to a direct provider key if set.
        """
        gateway_key = (
            os.getenv("EXPLABS_API_KEY", "").strip()
            or os.getenv("EXPERIENTIAL_ORG_KEY", "").strip()
        )
        gateway_url = os.getenv("OPENAI_BASE_URL", "https://api.experientiallabs.ai/v1").rstrip("/")

        if gateway_key:
            return ChatOpenAI(
                model=os.getenv("EXPERIENTIAL_MODEL", "gpt-5.6-luna"),
                base_url=gateway_url,
                api_key=gateway_key,
                temperature=0.1,  # low temp = more literal form filling
            )

        # Fallback: direct OpenAI key
        direct_key = os.getenv("OPENAI_API_KEY", "").strip()
        if direct_key:
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=direct_key,
                temperature=0.1,
            )

        raise RuntimeError("No LLM configured. Set EXPLABS_API_KEY or OPENAI_API_KEY.")

    async def apply(
        self,
        job_url: str,
        resume_path: str,
        cover_letter_path: Optional[str] = None,
        why_this_company: str = "",
        company_name: str = "",
        role_title: str = "",
        max_steps: int = 40,
        headless: bool = False,  # ignored when attaching via CDP
    ) -> ApplicationResult:
        """
        Fill a job application form inside the user's real Edge.

        Does NOT submit. The tab stays open in Edge so you can review
        the filled form and click Submit yourself.

        Args:
            job_url: Direct link to the application form.
            resume_path: Path to the tailored resume PDF.
            cover_letter_path: Optional path to a tailored cover letter.
            why_this_company: Per-job answer for the free-text question.
            max_steps: Hard cap on agent steps. 40 is generous for ATS.
            headless: Ignored. CDP attach uses whatever Edge window is open.

        Returns:
            ApplicationResult describing what happened.
        """
        # Fail fast with a clear message if Edge isn't reachable.
        _check_edge_cdp()

        task = self._build_task_prompt(
            job_url=job_url,
            resume_path=resume_path,
            cover_letter_path=cover_letter_path,
            why_this_company=why_this_company,
            company_name=company_name,
            role_title=role_title,
        )

        # Attach to the user's running Edge via CDP.
        # Do NOT set user_data_dir — that would launch a separate browser
        # and fight Edge's single-instance lock.
        browser = Browser(
            cdp_url=EDGE_CDP_URL,
            cross_origin_iframes=True,  # needed for embedded ATS iframes
        )

        try:
            llm = self._make_llm()

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,
            )

            history = await agent.run(max_steps=max_steps)

            # Inspect the history to figure out what happened.
            steps = []
            captcha_seen = False
            final_url = None

            for i, step in enumerate(history.history or []):
                summary = str(getattr(step, "result", ""))[:200]
                steps.append(f"[{i}] {summary}")
                if "captcha" in summary.lower():
                    captcha_seen = True

            # Last known URL is usually in the final step's state.
            try:
                final_url = history.urls()[-1] if history.urls() else job_url
            except Exception:
                final_url = job_url

            if captcha_seen:
                return ApplicationResult(
                    success=False,
                    job_url=job_url,
                    steps_taken=len(steps),
                    error_message="CAPTCHA encountered. Manual review required.",
                    captcha_encountered=True,
                    final_url=final_url,
                    history_summary=steps,
                )

            # Heuristic: if the agent ran out of steps, it probably looped.
            if len(steps) >= max_steps:
                return ApplicationResult(
                    success=False,
                    job_url=job_url,
                    steps_taken=len(steps),
                    error_message=(
                        f"Agent hit max_steps ({max_steps}). "
                        "Form may be incomplete or the page was unexpected."
                    ),
                    final_url=final_url,
                    history_summary=steps,
                )

            return ApplicationResult(
                success=True,
                job_url=job_url,
                steps_taken=len(steps),
                paused_for_human=True,
                final_url=final_url,
                history_summary=steps,
            )

        except Exception as exc:
            logger.exception("Browser Use application failed for %s", job_url)
            return ApplicationResult(
                success=False,
                job_url=job_url,
                error_message=str(exc)[:1000],
            )
        finally:
            # CRITICAL: When attached via CDP, do NOT close the browser.
            # That would kill the user's Edge. Only disconnect our
            # agent's reference to it.
            try:
                # browser_use exposes .close() which disconnects when
                # using CDP. If it kills the browser instead, remove this
                # call entirely — the CDP connection drops when the
                # Python process exits anyway.
                await browser.close()
            except Exception:
                pass


# ------------------------------------------------------------------
# Sync wrapper for FastAPI
# ------------------------------------------------------------------


def apply_to_job_sync(**kwargs) -> ApplicationResult:
    """Run the async applier from sync code (FastAPI endpoint, scripts)."""
    return asyncio.run(BrowserUseApplier().apply(**kwargs))
