"""
The workflow's shared state, in one place.

The workflow spans several Streamlit pages, so the state that crosses page
boundaries has to live somewhere that is not a widget. Before this module the
contract was implicit: ``uploaded_file_data``, ``job_desc`` and
``last_analysis`` were written by one page and read by two others, with the
relationship only discoverable by reading all three.

The keys and their shapes are unchanged -- the existing pages read them directly
and keep working. This adds the accessors, the derived readiness checks, and the
generation-result registry that the new pages need.

What is stored, and why each field earns its place:

``uploaded_file_data``
    ``(filename, bytes, mime)``. The bytes are the durable copy. Streamlit's
    uploader widget is ephemeral across a page switch, so anything that has to
    reach a later request re-sends from here.

``job_desc``
    The target posting text.

``last_analysis``
    The ATS response. Gates CV generation and the career tools.

``analysis_meta`` / ``cv_meta``
    Provider, model, timings and validation verdicts for the last run of each
    stage. Kept separate from the payloads so a 20 KB analysis dict is not
    re-copied on every access.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

#: Canonical session keys. Referenced by name everywhere so a rename is one edit.
KEY_API_BASE = "api_base"
KEY_PAGE = "current_page"
KEY_UPLOAD = "uploaded_file_data"
KEY_JOB = "job_desc"
KEY_ANALYSIS = "last_analysis"
KEY_ANALYSIS_META = "analysis_meta"
KEY_CV_META = "cv_meta"
KEY_LAYOUT = "cv_layout"
KEY_PROVIDER = "provider"
KEY_MODEL = "model_name"
KEY_ROUTE = "route_mode"
KEY_RECOMMENDED_LAYOUT = "recommended_layout"
KEY_IMPROVEMENT_RESULT = "improvement_result"

#: The formats the backend accepts. Declared once: the backend rejects anything
#: else with a 400, so a client-side list that drifts only produces a confusing
#: error at request time rather than at upload time.
ACCEPTED_RESUME_FORMATS = ("pdf", "docx", "txt")


def ensure_defaults() -> None:
    """Seed the workflow keys. Safe to call on every run."""
    import os

    st.session_state.setdefault(
        KEY_API_BASE, os.getenv("FASTAPI_API_BASE", "http://127.0.0.1:8000")
    )
    st.session_state.setdefault(KEY_PAGE, "overview")
    st.session_state.setdefault(KEY_UPLOAD, None)
    st.session_state.setdefault(KEY_JOB, "")
    st.session_state.setdefault(KEY_ANALYSIS, None)
    st.session_state.setdefault(KEY_ANALYSIS_META, None)
    st.session_state.setdefault(KEY_CV_META, None)
    st.session_state.setdefault(KEY_LAYOUT, "german_corporate")
    st.session_state.setdefault(KEY_PROVIDER, "ollama")
    st.session_state.setdefault(KEY_MODEL, "")
    st.session_state.setdefault(KEY_ROUTE, "direct")
    st.session_state.setdefault(KEY_RECOMMENDED_LAYOUT, None)
    st.session_state.setdefault(KEY_IMPROVEMENT_RESULT, None)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def set_upload(filename: str, payload: bytes, mime: str) -> None:
    st.session_state[KEY_UPLOAD] = (filename, payload, mime)


def get_upload() -> tuple[str, bytes, str] | None:
    return st.session_state.get(KEY_UPLOAD)


def has_upload() -> bool:
    return bool(st.session_state.get(KEY_UPLOAD))


def upload_size_kb() -> float:
    upload = get_upload()
    return len(upload[1]) / 1024 if upload else 0.0


def upload_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""


def file_payload() -> dict[str, Any]:
    """
    The multipart resume field, rebuilt from the stored bytes.

    Returns ``{}`` when nothing is uploaded, so a caller can merge it into a
    form body unconditionally.
    """
    upload = get_upload()
    if not upload:
        return {}
    filename, payload, mime = upload
    return {"resume_file": (filename, payload, mime)}


# ---------------------------------------------------------------------------
# Job description
# ---------------------------------------------------------------------------


def set_job(text: str) -> None:
    st.session_state[KEY_JOB] = text or ""


def get_job() -> str:
    return st.session_state.get(KEY_JOB, "") or ""


def job_stats() -> dict[str, int]:
    """
    Size of the posting, in the units a user can act on.

    Characters and words are computed here rather than estimated; a token count
    would be a guess, and a guess presented as a measurement is worse than no
    number.
    """
    text = get_job()
    stripped = text.strip()
    return {
        "chars": len(text),
        "chars_stripped": len(stripped),
        "words": len(stripped.split()),
        "lines": len([line for line in stripped.splitlines() if line.strip()]),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def set_analysis(payload: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
    st.session_state[KEY_ANALYSIS] = payload
    st.session_state[KEY_ANALYSIS_META] = meta or {}


def get_analysis() -> dict[str, Any] | None:
    return st.session_state.get(KEY_ANALYSIS)


def has_analysis() -> bool:
    return bool(st.session_state.get(KEY_ANALYSIS))


def analysis_meta() -> dict[str, Any]:
    return st.session_state.get(KEY_ANALYSIS_META) or {}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def set_cv_meta(meta: dict[str, Any]) -> None:
    st.session_state[KEY_CV_META] = meta


def cv_meta() -> dict[str, Any]:
    return st.session_state.get(KEY_CV_META) or {}


def set_generation_choice(
    provider: str,
    model_name: str = "",
    route_mode: str = "direct",
) -> None:
    """
    Remember the provider choice across pages.

    The workflow asks for it once, on the LLM settings page, and the generation
    pages read it. Previously each page rendered its own copy of the selector, so
    a user could analyse with one provider and generate with another without
    noticing.
    """
    st.session_state[KEY_PROVIDER] = provider
    st.session_state[KEY_MODEL] = model_name or ""
    st.session_state[KEY_ROUTE] = route_mode


def generation_choice() -> dict[str, str]:
    return {
        "provider": st.session_state.get(KEY_PROVIDER) or "ollama",
        "model_name": st.session_state.get(KEY_MODEL) or "",
        "route_mode": st.session_state.get(KEY_ROUTE) or "direct",
    }


def set_layout(layout: str) -> None:
    st.session_state[KEY_LAYOUT] = layout


def get_layout() -> str:
    return st.session_state.get(KEY_LAYOUT) or "german_corporate"


# ---------------------------------------------------------------------------
# Section rendering, shared by the standalone pages and the single-page workflow
# ---------------------------------------------------------------------------
#
# Each stage used to be a page of its own, so a stage that could not run yet
# rendered a full-screen empty state with a button that navigated somewhere else.
# The six stages now share one page, where that button is meaningless — the
# inputs the stage needs are further up the same scroll. These two helpers let a
# stage express the same rule in both places without each one re-deriving it:
#
#   section_header(...)  the page title when standalone, nothing when inline,
#                        because the step rail already names the section
#   blocked(...)         a one-line "still needs X" when inline, a full empty
#                        state with a way out when standalone
#
# Both return a value the caller acts on, so a stage cannot forget the check.


def section_header(
    number: int | str,
    title: str,
    subtitle: str = "",
    *,
    compact: bool = False,
) -> None:
    """
    The header for one stage.

    Suppressed when ``compact`` is set, because on the single-page workflow the
    rail at the top already says which stage this is, and six repeated titles
    stacked down the page is exactly the noise the single page is meant to remove.
    """
    if compact:
        return

    from app.dashboard import theme

    theme.step_header(number, title, subtitle)


def blocked(
    ready: bool,
    *,
    missing: str,
    action: str = "",
    page: str = "",
    icon: str = "🚧",
    compact: bool = False,
) -> bool:
    """
    Report that a stage cannot run yet. Returns True when the caller must stop.

    Standalone, this is a full empty state plus a button to go and fix it, which
    is all a user of a single page can do. Inline, the same message is one line
    and no button: the thing being asked for is visible further up the same
    page, so a button that "goes" to it would only move the scroll the wrong way.
    """
    if ready:
        return False

    from app.dashboard import theme

    if compact:
        st.caption(f"{icon} Waiting for {missing}.")
        return True

    theme.empty_state("Not ready yet", missing, action=action or None, icon=icon)
    if page:
        if st.button(f"Go to {action or page}", key=f"blocked_{page}_{missing[:12]}"):
            st.session_state[KEY_PAGE] = page
            st.rerun()
    return True


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def workflow_readiness() -> dict[str, bool]:
    """
    What each stage needs, so no page has to re-derive it.

    The ordering matters: CV generation needs the analysis because the analysis
    supplies the missing-skill list the generator is told to work from, not
    because the UI happens to run in that order.
    """
    job_ok = bool(get_job().strip())
    return {
        "resume": has_upload(),
        "job_description": job_ok,
        "analysis": has_analysis(),
        "generation": has_analysis() and has_upload() and job_ok,
    }


def next_step() -> tuple[str, str]:
    """
    The first incomplete step, as ``(page_key, human_label)``.

    Drives the overview's progress and the "continue" affordance, so the user is
    never told to generate a CV before the inputs exist.
    """
    ready = workflow_readiness()

    if not ready["resume"]:
        return "resume", "Upload a resume"
    if not ready["job_description"]:
        return "job_input", "Add a job description"
    if not ready["analysis"]:
        return "ats_analysis", "Run the ATS analysis"
    return "cv_generator", "Generate the CV"


def reset_workflow() -> None:
    """
    Clear the workflow back to its starting state.

    Also clears the stored generation results, because leaving a PDF from a
    previous resume on screen after "start over" is worse than showing nothing.
    """
    from app.dashboard.components import clear_result

    st.session_state[KEY_UPLOAD] = None
    st.session_state[KEY_JOB] = ""
    st.session_state[KEY_ANALYSIS] = None
    st.session_state[KEY_ANALYSIS_META] = None
    st.session_state[KEY_CV_META] = None

    for key in (
        "analysis",
        "docx",
        "pdf",
        "tex",
        "pdf_validation",
        "final_ats",
        "improvement_result",
    ):
        clear_result(key)


def step_is_done(name: str) -> bool:
    return bool(workflow_readiness().get(name))


__all__ = [
    "ACCEPTED_RESUME_FORMATS",
    "analysis_meta",
    "cv_meta",
    "ensure_defaults",
    "file_payload",
    "generation_choice",
    "get_analysis",
    "get_job",
    "get_layout",
    "get_upload",
    "has_analysis",
    "has_upload",
    "job_stats",
    "next_step",
    "reset_workflow",
    "set_analysis",
    "set_cv_meta",
    "set_generation_choice",
    "set_job",
    "set_layout",
    "set_upload",
    "step_is_done",
    "upload_extension",
    "upload_size_kb",
    "workflow_readiness",
]
