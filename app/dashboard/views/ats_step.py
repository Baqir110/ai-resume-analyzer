"""
Workflow step 3: ATS analysis.

Runs the analysis and presents the result as a scannable report. Every number
shown comes from the API response. Nothing is derived, rounded into a different
scale, or inferred: a metric the backend does not compute is not displayed, and
a missing field shows as absent rather than as zero.

The provider used is shown explicitly, because the answer depends on it and a
user comparing two runs needs to know whether they differed in model or in CV.
"""

from __future__ import annotations

import time

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.components import (
    clear_result,
    render_error_alert,
    run_with_progress,
    store_result,
)
from app.dashboard.helpers import clear_cached_status, get_api_base


def _score_verdict(score: float) -> str:
    """
    Band the score the way the backend's own thresholds do.

    Mirrors ``ATS_THRESHOLD_LOW`` / ``ATS_THRESHOLD_MEDIUM`` rather than
    inventing a scale, so the label agrees with the recommendation text the
    backend returns alongside it.
    """
    try:
        from app.core.config import settings

        low = settings.ATS_THRESHOLD_LOW
        medium = settings.ATS_THRESHOLD_MEDIUM
    except Exception:
        low, medium = 40.0, 65.0

    if score >= medium:
        return theme.PASSED
    if score >= low:
        return theme.NOT_TESTED
    return theme.FAILED


def _run(api_base: str) -> None:
    """POST /analyze, timing it and recording what produced the result."""
    upload = workflow.get_upload()
    if not upload:
        return

    filename, payload, mime = upload
    choice = workflow.generation_choice()
    job_desc = workflow.get_job().strip()

    workflow.set_job(job_desc)
    clear_result("analysis")

    started = time.perf_counter()
    response, elapsed = run_with_progress(
        label="ATS analysis",
        endpoint=f"{api_base}/api/v1/resume/analyze",
        data={
            "job_description": job_desc,
            "provider": choice["provider"],
            "model_name": choice["model_name"],
            "route_mode": choice["route_mode"],
        },
        files={"resume_file": (filename, payload, mime)},
        steps=[
            "Parsing the resume file",
            "Extracting skills, keywords and structure",
            "Scoring against the job description",
            "Calling the configured LLM provider",
            "Generating improvement suggestions",
            "Finalising recommendations",
        ],
        hint="LLM-driven — usually 15–45 seconds on a local model.",
        timeout=240,
        seconds_per_step=6.0,
    )

    if response is not None and 200 <= response.status_code < 300:
        payload_json = response.json()
        workflow.set_analysis(
            payload_json,
            {
                "provider": choice["provider"],
                "model": choice["model_name"],
                "route_mode": choice["route_mode"],
                "seconds": round(time.perf_counter() - started, 2),
                "endpoint": "/analyze",
            },
        )
        store_result("analysis", payload_json)
        clear_cached_status(api_base)
        st.rerun()
    elif response is not None:
        render_error_alert(response)


def _render_results(api_base: str) -> None:
    """Present the analysis. Absent fields are absent, not zero."""
    analysis = workflow.get_analysis() or {}
    meta = workflow.analysis_meta()

    theme.section_header("Scores", "📊")

    score = analysis.get("ats_match_score")
    density = analysis.get("keyword_density_score")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        theme.value_card(
            "ATS match",
            f"{score:.1f}" if isinstance(score, (int, float)) else "—",
            "out of 100",
        )
    with c2:
        theme.value_card(
            "Keyword density",
            f"{density:.1f}" if isinstance(density, (int, float)) else "—",
            "out of 100",
        )
    with c3:
        theme.value_card("Matching skills", str(len(analysis.get("matching_skills") or [])), "")
    with c4:
        theme.value_card("Missing skills", str(len(analysis.get("missing_skills") or [])), "")

    if isinstance(score, (int, float)):
        theme.pills([(_score_verdict(score), "ATS match")])
        st.progress(min(1.0, max(0.0, score / 100.0)))

    if isinstance(density, (int, float)):
        st.caption(f"Keyword density: {density:.1f} / 100")

    # -- what produced this ------------------------------------------------
    theme.kv_table(
        [
            ("Provider", meta.get("provider")),
            ("Model", meta.get("model") or "(provider default)"),
            ("Route mode", meta.get("route_mode")),
            ("Duration", f"{meta.get('seconds')}s" if meta.get("seconds") else None),
        ]
    )
    st.caption(
        "LLM call count and retries for this run are on the Overview page, read "
        "from the backend's own log."
    )

    theme.rule()
    theme.section_header("Skills", "🎯")

    matching = analysis.get("matching_skills") or []
    missing = analysis.get("missing_skills") or []

    left, right = st.columns(2)

    with left:
        st.markdown("**Matched**")
        if matching:
            theme.chip_row([str(s) for s in matching], kind="ok")
        else:
            st.caption("The backend reported no matched skills.")

    with right:
        st.markdown("**Missing from the resume**")
        if missing:
            theme.chip_row([str(s) for s in missing], kind="miss")
        else:
            theme.caption("The backend reported no missing skills.")

    # -- keywords ----------------------------------------------------------
    keywords = analysis.get("keywords") or analysis.get("top_keywords") or []
    if keywords:
        theme.section_header("Keywords extracted from the posting", "🔑")
        theme.chip_row([str(k) for k in keywords])

    # -- structure ---------------------------------------------------------
    structure = analysis.get("structure") or analysis.get("resume_structure")
    if isinstance(structure, dict):
        theme.section_header("Structure checks", "🧱")
        theme.kv_table([(k, v) for k, v in structure.items() if v not in (None, "")])

    # -- recommendations ---------------------------------------------------
    suggestions = analysis.get("improvement_suggestions") or []
    if suggestions:
        theme.section_header("Recommended changes", "💡")
        for index, item in enumerate(suggestions, 1):
            st.markdown(f"**{index}.** {item}")

    recommendation = analysis.get("recommendation")
    if isinstance(recommendation, dict):
        theme.section_header("Backend recommendation", "🧭")
        theme.kv_table([(k, v) for k, v in recommendation.items() if v not in (None, "")])

    # -- the extracted text, so the user can see what was actually read ----
    resume_text = analysis.get("resume_text")
    if isinstance(resume_text, str) and resume_text.strip():
        with st.expander("📄 Extracted resume text", expanded=False):
            st.caption(f"{len(resume_text):,} characters")
            st.text_area(
                "Extracted text",
                value=resume_text[:20_000],
                height=300,
                disabled=True,
                key="ats_extracted_text",
            )
            if len(resume_text) > 20_000:
                st.caption("Preview capped at 20,000 characters.")

    theme.rule()
    action_col, _ = st.columns([1, 3])
    with action_col:
        if st.button("Continue to CV generation →", type="primary"):
            st.session_state[workflow.KEY_PAGE] = "cv_generator"
            st.rerun()


def render_ats_step() -> None:
    """Render the ATS analysis step."""
    theme.step_header(
        "3",
        "ATS Analysis",
        "Score the resume against the posting and get concrete changes.",
    )

    api_base = get_api_base()

    if not workflow.has_upload():
        theme.empty_state(
            "No resume",
            "The analysis needs a resume file.",
            action="Open step 1 and upload one",
            icon="📎",
        )
        if st.button("← Upload a resume"):
            st.session_state[workflow.KEY_PAGE] = "resume"
            st.rerun()
        return

    if not workflow.get_job().strip():
        theme.empty_state(
            "No job description",
            "The analysis needs the posting to score against.",
            action="Open step 2 and paste it",
            icon="✍️",
        )
        if st.button("← Add a job description"):
            st.session_state[workflow.KEY_PAGE] = "job_input"
            st.rerun()
        return

    if not workflow.has_analysis():
        choice = workflow.generation_choice()

        theme.section_header("Run", "▶️")
        theme.kv_table(
            [
                ("Provider", choice["provider"]),
                ("Model", choice["model_name"] or "(provider default)"),
                ("Route mode", choice["route_mode"]),
            ]
        )

        if choice["provider"] in {"ollama", "omniroute"}:
            st.caption(
                "A local model usually needs 15–45 seconds. It is answering the "
                "same request a cloud provider would."
            )

        if st.button("🔍 Run ATS analysis", type="primary"):
            _run(api_base)

        if st.button("⚙️ Change provider or model"):
            st.session_state[workflow.KEY_PAGE] = "llm_settings"
            st.rerun()
        return

    _render_results(api_base)

    theme.rule()
    if st.button("🔄 Re-run with the current inputs"):
        workflow.set_analysis(None, None)
        st.rerun()
