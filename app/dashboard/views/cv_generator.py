"""
CV generation: produce the final document.

Uses the shared workflow state for the provider, model and layout, so the choice
made on the settings page is the one used here. The previous version rendered its
own copy of the provider selector, which meant a user could set a provider on the
analysis page and silently generate with a different one.

The DOCX, PDF and LaTeX builders are kept. All three are useful: DOCX is
editable, PDF is the deliverable, LaTeX is the source for either. Each records
what produced it, so the preview page and the metrics panel can say which
provider and model were actually involved.
"""

from __future__ import annotations

import json
import time

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.components import (
    clear_result,
    get_result,
    render_error_alert,
    run_with_progress,
    store_result,
)
from app.dashboard.helpers import clear_cached_status, fetch_quota_status, get_api_base

_STEPS = [
    "Loading the resume and the analysis",
    "Calling the configured LLM provider",
    "Writing the CV body",
    "Checking that no fact was invented",
    "Assembling the LaTeX document",
    "Compiling with pdflatex",
    "Reading the PDF back and validating it",
]


def _common_data() -> dict:
    """
    The form fields every generation endpoint accepts.

    Built from the shared workflow state, so provider, model and layout cannot
    drift between what the user chose and what is sent.
    """
    choice = workflow.generation_choice()
    analysis = workflow.get_analysis() or {}

    return {
        "job_description": workflow.get_job().strip(),
        "provider": choice["provider"],
        "model_name": choice["model_name"],
        "route_mode": choice["route_mode"],
        "layout_style": workflow.get_layout(),
        "improvement_suggestions": json.dumps(analysis.get("improvement_suggestions") or []),
    }


def _store(
    key: str,
    response,
    filename: str,
    mime: str,
    started: float,
) -> None:
    """Record a successful generation, whatever shape the response took."""
    elapsed = round(time.perf_counter() - started, 2)
    content_type = response.headers.get("Content-Type", "")
    choice = workflow.generation_choice()

    payload = b""
    if "json" in content_type:
        try:
            body = response.json()
        except Exception:
            body = {}

        encoded = body.get(f"{key}_base64") or body.get("content") or body.get("document") or ""
        if isinstance(encoded, str) and encoded:
            import base64

            try:
                payload = base64.b64decode(encoded)
            except Exception:
                payload = encoded.encode("utf-8", errors="replace")
        elif isinstance(encoded, (bytes, bytearray)):
            payload = bytes(encoded)
    else:
        payload = response.content

    store_result(
        key,
        {
            "content": payload,
            "filename": filename,
            "mime": mime,
            "elapsed": elapsed,
            "size_kb": round(len(payload) / 1024, 1),
            "provider": choice["provider"],
            "model": choice["model_name"],
            "layout": workflow.get_layout(),
        },
    )

    workflow.set_cv_meta(
        {
            "provider": choice["provider"],
            "model": choice["model_name"],
            "layout": workflow.get_layout(),
            "seconds": elapsed,
            "endpoint": f"/generate-{key}",
        }
    )
    clear_cached_status(get_api_base())


def _build(api_base: str, key: str) -> None:
    """
    Run one generation and store the result.

    Shared by the three builders so the timeout, the progress steps and the
    error handling cannot differ between them.
    """
    upload = workflow.get_upload()
    if not upload:
        return

    filename, payload, mime = upload

    endpoints = {
        "pdf": (
            "/generate-german-cv",
            f"cv-{workflow.get_layout()}.pdf",
            "application/pdf",
            "📄 Generating PDF",
        ),
        "tex": (
            "/generate-tex-cv",
            f"cv-{workflow.get_layout()}.tex",
            "text/plain",
            "📝 Generating LaTeX",
        ),
        "docx": (
            "/generate-full",
            f"cv-{workflow.get_layout()}.docx",
            "application/vnd.openxmlformats-officedocument." "wordprocessingml.document",
            "📘 Generating DOCX",
        ),
    }

    path, out_name, out_mime, label = endpoints[key]

    clear_result(key)
    if key == "pdf":
        # A new PDF invalidates the previous validation report: it described a
        # different document.
        clear_result("pdf_validation")

    started = time.perf_counter()
    response, _elapsed = run_with_progress(
        label=label,
        endpoint=f"{api_base}/api/v1/resume{path}",
        data=_common_data(),
        files={"resume_file": (filename, payload, mime)},
        steps=_STEPS,
        hint="2 model calls for PDF and LaTeX. Usually 30–90 seconds locally.",
        timeout=420,
        seconds_per_step=9.0,
    )

    if response is not None and 200 <= response.status_code < 300:
        _store(key, response, out_name, out_mime, started)
        st.rerun()
    elif response is not None:
        render_error_alert(response)


def _render_selection() -> None:
    """Show and, if needed, change the generation settings."""
    choice = workflow.generation_choice()

    theme.section_header("Generation settings", "⚙️")
    c1, c2, c3 = st.columns(3)
    with c1:
        theme.value_card("Provider", choice["provider"], "")
    with c2:
        theme.value_card("Model", choice["model_name"] or "(provider default)", "")
    with c3:
        theme.value_card("Layout", workflow.get_layout(), "")

    if st.button("⚙️ Change provider, model or layout"):
        st.session_state[workflow.KEY_PAGE] = "llm_settings"
        st.rerun()

    if st.button("▤ Choose a different layout"):
        st.session_state[workflow.KEY_PAGE] = "layout_picker"
        st.rerun()


def _render_outputs() -> None:
    """Download buttons for whatever exists, plus a link to the preview."""
    theme.section_header("Generated", "📦")

    specs = (
        (
            "docx",
            "📘 DOCX",
            "application/vnd.openxmlformats-officedocument." "wordprocessingml.document",
        ),
        ("pdf", "📄 PDF", "application/pdf"),
        ("tex", "📝 LaTeX", "text/plain"),
    )

    present = [(k, label, m) for k, label, m in specs if get_result(k)]

    if not present:
        theme.empty_state(
            "Nothing generated yet",
            "Pick a format above. Each one is a separate model call.",
            icon="📦",
        )
        return

    for key, label, mime in present:
        result = get_result(key) or {}
        content = result.get("content")
        if not isinstance(content, (bytes, bytearray)) or not content:
            continue

        left, right = st.columns([3, 1])
        with left:
            theme.kv_table(
                [
                    (label, result.get("filename")),
                    ("Size", f"{result.get('size_kb', 0):,.0f} KB"),
                    ("Time", f"{result.get('elapsed', 0):.1f}s"),
                ]
            )
        with right:
            st.download_button(
                f"⬇️ {label}",
                data=content,
                file_name=result.get("filename") or f"cv.{key}",
                mime=mime,
                width="stretch",
                key=f"gen_download_{key}",
            )

    theme.rule()
    if get_result("pdf"):
        if st.button("👁️ Preview and validate the PDF", type="primary"):
            st.session_state[workflow.KEY_PAGE] = "pdf_preview"
            st.rerun()


def render_cv_generation_page() -> None:
    """Render the CV generation page."""
    theme.step_header(
        "📄",
        "CV Generation",
        "Produce the final document in the format you need.",
    )

    api_base = get_api_base()

    if not workflow.has_analysis():
        theme.empty_state(
            "No analysis yet",
            "Generation works from the gaps the ATS analysis found. Without "
            "it there is nothing to target, and the generator would have to "
            "guess.",
            action="Open ATS Analysis",
            icon="🔍",
        )
        if st.button("← Run the ATS analysis"):
            st.session_state[workflow.KEY_PAGE] = "ats_analysis"
            st.rerun()
        return

    if not workflow.has_upload() or not workflow.get_job().strip():
        theme.empty_state(
            "Missing an input",
            "Generation needs both a resume and a job description.",
            action="Open steps 1 and 2",
            icon="📎",
        )
        return

    _render_selection()

    analysis = workflow.get_analysis() or {}
    if analysis.get("ats_match_score") is not None:
        st.caption(
            f"Targeting the posting the resume currently scores "
            f"{analysis['ats_match_score']:.1f} against, addressing "
            f"{len(analysis.get('missing_skills') or [])} missing skill(s)."
        )

    theme.rule()
    theme.section_header("Output format", "📤")

    format_col, quota_col = st.columns([2, 1])

    with format_col:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📄 PDF", type="primary", width="stretch"):
                _build(api_base, "pdf")
        with c2:
            if st.button("📝 LaTeX", width="stretch"):
                _build(api_base, "tex")
        with c3:
            if st.button("📘 DOCX", width="stretch"):
                _build(api_base, "docx")

    with quota_col:
        quota = fetch_quota_status(api_base)
        if quota:
            limited = [
                (name, data) for name, data in quota.items() if name not in {"ollama", "omniroute"}
            ]
            if limited:
                name, data = limited[0]
                remaining = data.get("requests_remaining")
                if remaining is not None:
                    theme.value_card(
                        f"{name} requests left",
                        str(remaining),
                        "rolling window",
                    )
            else:
                st.caption("No rate-limited provider is in use.")

    theme.rule()
    _render_outputs()

    st.caption(
        "Each format is a separate request. Generating all three makes three " "calls, not one."
    )
