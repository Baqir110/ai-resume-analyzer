"""
CV optimisation: what the model changed, and whether to keep it.

Three things this page does that a bare "generate" button does not.

*Shows the input.* The suggestions being sent come from the ATS analysis, and
the analysis is what identified the gaps. Showing them makes the request
reviewable before it is billed.

*Offers a diff.* The backend has a ``diff-preview`` endpoint that compares two
bullet lists word by word. Using it means the before/after is computed by the
same code that would judge it, rather than by a string comparison in the UI.

*Avoids duplicate calls.* Regenerating costs a full model call, so the button is
explicit and the previous result is kept until a new one replaces it. Nothing
here re-runs the analysis to obtain data it already has.
"""

from __future__ import annotations

import json
import time

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.components import (
    clear_result,
    get_result,
    render_diff_view,
    render_error_alert,
    run_with_progress,
    store_result,
)
from app.dashboard.helpers import clear_cached_status, get_api_base, make_api_request_verbose


def _suggestions() -> list[str]:
    analysis = workflow.get_analysis() or {}
    return [str(s) for s in (analysis.get("improvement_suggestions") or [])]


def _missing_skills() -> list[str]:
    analysis = workflow.get_analysis() or {}
    return [str(s) for s in (analysis.get("missing_skills") or [])]


def _render_request_preview() -> None:
    """
    Exactly what will be sent, before it is sent.

    A model call is the expensive step in this application. Showing the request
    first means a user can see they are about to spend 20–60 seconds before they
    spend it.
    """
    suggestions = _suggestions()
    missing = _missing_skills()
    choice = workflow.generation_choice()
    analysis = workflow.get_analysis() or {}

    theme.section_header("What will be sent", "📤")

    c1, c2, c3 = st.columns(3)
    with c1:
        theme.value_card("Provider", choice["provider"], "")
    with c2:
        theme.value_card("Model", choice["model_name"] or "(provider default)", "")
    with c3:
        theme.value_card("Route mode", choice["route_mode"], "")

    theme.kv_table(
        [
            ("Resume", workflow.get_upload()[0] if workflow.get_upload() else None),
            ("Job description", f"{workflow.job_stats()['chars']:,} characters"),
            ("Layout", workflow.get_layout()),
            ("Improvement suggestions", len(suggestions)),
            ("Missing skills to integrate", len(missing)),
        ]
    )

    if suggestions:
        with st.expander(f"💡 Improvement suggestions ({len(suggestions)})", expanded=False):
            for index, item in enumerate(suggestions, 1):
                st.markdown(f"{index}. {item}")

    if missing:
        with st.expander(f"🎯 Skills to integrate ({len(missing)})", expanded=False):
            theme.chip_row(missing, kind="miss")

    if analysis.get("ats_match_score") is not None:
        st.caption(
            f"Current ATS score {analysis['ats_match_score']:.1f}. The "
            f"generator is told to work from the gaps above, not to rewrite "
            f"the whole document freely."
        )


def _render_diff() -> None:
    """Word-level before/after, computed by the backend."""
    diff_data = get_result("optimization_diff")
    if not diff_data:
        return

    theme.section_header("Before and after", "🔍")
    st.caption(
        "Computed by the backend's diff endpoint, word by word. The same "
        "comparison the application uses to judge its own output."
    )
    render_diff_view(diff_data)


def _compare(original: str, optimised: str) -> None:
    """Ask the backend to diff the two bullet lists."""
    api_base = get_api_base()
    payload = {
        "original_bullets": original,
        "optimized_bullets": optimised,
    }
    response, meta = make_api_request_verbose(
        f"{api_base}/api/v1/resume/diff-preview",
        json=payload,
        method="POST",
        timeout=30,
    )
    if response is not None and 200 <= response.status_code < 300:
        try:
            store_result("optimization_diff", response.json().get("diffs") or [])
        except Exception:
            clear_result("optimization_diff")
    else:
        st.caption(
            "The comparison endpoint did not answer; the before/after view is " "unavailable."
        )


def render_optimization_page() -> None:
    """Render the CV optimisation page."""
    theme.step_header(
        "✏",
        "CV Optimisation",
        "Review the gaps, then tailor the CV to this posting.",
    )

    api_base = get_api_base()

    if not workflow.has_analysis():
        theme.empty_state(
            "No analysis yet",
            "Optimisation works from the gaps the ATS analysis found. Run it "
            "first — the missing-skill list is the input here.",
            action="Open ATS Analysis",
            icon="🔍",
        )
        if st.button("← Run the ATS analysis"):
            st.session_state[workflow.KEY_PAGE] = "ats_analysis"
            st.rerun()
        return

    _render_request_preview()
    theme.rule()

    if not workflow.has_upload() or not workflow.get_job().strip():
        theme.pills([(theme.NOT_TESTED, "a resume and a job description are required")])
        return

    action_col, _ = st.columns([1, 3])
    with action_col:
        if st.button("✏️ Tailor the CV", type="primary", key="opt_run"):
            _run_tailoring(api_base)

    _render_results()
    _render_diff()


def _run_tailoring(api_base: str) -> None:
    """
    Generate the tailored document.

    Uses the DOCX endpoint, which returns an editable document: the point of
    this step is to look at what changed and decide whether to keep it, which
    needs a file the user can open, not a finished PDF.
    """
    upload = workflow.get_upload()
    if not upload:
        return

    filename, payload, mime = upload
    choice = workflow.generation_choice()
    analysis = workflow.get_analysis() or {}
    suggestions = analysis.get("improvement_suggestions") or []

    clear_result("docx")
    clear_result("optimization_diff")

    started = time.perf_counter()
    response, elapsed = run_with_progress(
        label="Tailoring your CV",
        endpoint=f"{api_base}/api/v1/resume/generate-full",
        data={
            "job_description": workflow.get_job().strip(),
            "provider": choice["provider"],
            "model_name": choice["model_name"],
            "route_mode": choice["route_mode"],
            "layout_style": workflow.get_layout(),
            "improvement_suggestions": json.dumps(suggestions),
        },
        files={"resume_file": (filename, payload, mime)},
        steps=[
            "Loading the resume and the gaps",
            "Calling the configured LLM provider",
            "Rewriting bullets to address the gaps",
            "Checking that facts were not invented",
            "Assembling the document",
        ],
        hint="One model call. Usually 20–60 seconds on a local model.",
        timeout=300,
        seconds_per_step=8.0,
    )

    if response is not None and 200 <= response.status_code < 300:
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            try:
                body = response.json()
            except Exception:
                body = {}
            content = body.get("docx_base64") or body.get("content") or body.get("document") or ""
            if isinstance(content, str):
                import base64

                try:
                    payload_bytes = base64.b64decode(content)
                except Exception:
                    payload_bytes = b""
            else:
                payload_bytes = b""
        else:
            payload_bytes = response.content

        store_result(
            "docx",
            {
                "content": payload_bytes,
                "filename": f"tailored-{workflow.get_layout()}.docx",
                "mime": "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document",
                "elapsed": round(time.perf_counter() - started, 2),
                "size_kb": round(len(payload_bytes) / 1024, 1),
                "provider": choice["provider"],
                "model": choice["model_name"],
            },
        )
        workflow.set_cv_meta(
            {
                "provider": choice["provider"],
                "model": choice["model_name"],
                "layout": workflow.get_layout(),
                "seconds": round(time.perf_counter() - started, 2),
                "suggestions_sent": len(suggestions),
                "endpoint": "/generate-full",
            }
        )
        clear_cached_status(api_base)
        st.rerun()
    elif response is not None:
        render_error_alert(response)


def _render_results() -> None:
    """What the tailoring produced, and the facts about how it was made."""
    result = get_result("docx")
    if not result:
        return

    theme.section_header("Result", "✅")
    theme.pills([(theme.PASSED, "a tailored document was produced")])

    theme.kv_table(
        [
            ("File", result.get("filename")),
            ("Size", f"{result.get('size_kb', 0):,.0f} KB"),
            ("Generation time", f"{result.get('elapsed', 0):.1f}s"),
            ("Provider", result.get("provider")),
            ("Model", result.get("model") or "(provider default)"),
            ("Layout", workflow.get_layout()),
        ]
    )

    content = result.get("content")
    if isinstance(content, (bytes, bytearray)) and content:
        st.download_button(
            "⬇️ Download the tailored CV",
            data=content,
            file_name=result.get("filename") or "cv.docx",
            mime=result.get("mime")
            or "application/vnd.openxmlformats-officedocument." "wordprocessingml.document",
            width="stretch",
            type="primary",
            key="opt_download",
        )

    theme.rule()
    left, right = st.columns(2)
    with left:
        if st.button("📄 Generate the final PDF", type="primary"):
            st.session_state[workflow.KEY_PAGE] = "cv_generator"
            st.rerun()
    with right:
        if st.button("🔄 Tailor again", key="opt_regenerate"):
            st.caption(
                "This makes another model call and replaces the current result. "
                "The previous document is discarded."
            )
            clear_result("docx")
            clear_result("optimization_diff")
            st.rerun()
