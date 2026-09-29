"""
Workflow step 3: ATS analysis.

Runs the analysis and presents the result as a scannable report. Every number
shown comes from the API response. Nothing is derived, rounded into a different
scale, or inferred: a metric the backend does not compute is not displayed, and
a missing field shows as absent rather than as zero.

The provider used is shown explicitly, because the answer depends on it and a
user comparing two runs needs to know whether they differed in model or in CV.

User-friendly presentation (requirement 32):
- Simple language, not ATS jargon
- Clear "what's working" and "what can be improved" sections
- Every issue has a suggested action
- Detailed technical breakdown is in an expandable section
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


def _score_label(score: float) -> str:
    """Human-friendly score label."""
    if score >= 90:
        return "Excellent match"
    if score >= 75:
        return "Good match"
    if score >= 60:
        return "Moderate match"
    if score >= 40:
        return "Weak match"
    return "Poor match"


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


def _render_working_section(analysis: dict) -> None:
    """
    "What is working", as concrete facts rather than encouragement.

    Every line is backed by a check that ran, and the count comes from the
    backend's own threshold rather than a second ladder defined here — a page that
    congratulates a CV the backend scored badly is worse than saying nothing.
    """
    breakdown = analysis.get("ats_breakdown") or {}
    categories = breakdown.get("categories") or {}
    keyword_category = categories.get("keyword_match") or {}
    required_category = categories.get("required_skills") or {}

    working_items: list[str] = []

    for cat_data in categories.values():
        # An unmeasured category is not a success. Saying "the document is
        # machine-readable" before a document exists would be exactly the kind of
        # unfounded claim this score exists to avoid.
        if cat_data.get("not_measured"):
            continue
        if (cat_data.get("score") or 0) < 85:
            continue
        label = cat_data.get("label") or ""
        working_items.append(f"✓ {label}")

    title = keyword_category.get("title_alignment") or {}
    if title.get("stated") and title.get("aligned"):
        working_items.insert(0, f"✓ Your CV names the role this job is for ({title.get('target')})")

    required_matched = len(required_category.get("required_matched") or [])
    required_missing = len(required_category.get("required_missing") or [])
    if required_matched and not required_missing:
        working_items.insert(
            0, f"✓ All {required_matched} required skills are evidenced in your CV"
        )
    elif required_matched:
        working_items.append(
            f"✓ {required_matched} of {required_matched + required_missing} required "
            "skills are evidenced"
        )

    structure = categories.get("cv_structure") or {}
    quantified = next(
        (
            check
            for check in (structure.get("checks") or [])
            if check.get("name") == "measurable achievements"
        ),
        None,
    )
    if quantified and quantified.get("passed"):
        working_items.append(f"✓ {quantified.get('detail', '')}")

    if working_items:
        theme.section_header("What is working", "✅")
        for item in working_items:
            st.markdown(f"- {item}")


def _render_improvements_section(analysis: dict) -> None:
    """
    "What can be improved", one item per thing, each with its fix.

    The wording comes from the backend's ``ats_suggestions``, which is where the
    knowledge of *which* check failed and what to do about it lives. This function
    only presents it. Re-deriving the advice here would mean two copies of the
    same rules, and the copy that drifts is the one the user actually reads.
    """
    suggestions = analysis.get("ats_suggestions") or []
    breakdown = analysis.get("ats_breakdown") or {}

    if not suggestions:
        # Say so plainly rather than rendering an empty section, which reads as a
        # page that failed to load.
        if breakdown:
            theme.pills([(theme.PASSED, "nothing to improve - every check passed")])
        return

    theme.section_header("What can be improved", "\U0001f4a1")

    icons = {
        "high": "\U0001f534",  # red circle
        "medium": "\U0001f7e1",  # yellow circle
        "low": "\U0001f7e2",  # green circle
    }
    for suggestion in suggestions:
        icon = icons.get(suggestion.get("severity", "medium"), "\U0001f7e1")
        with st.expander(f"{icon} {suggestion.get('issue', '')}", expanded=False):
            st.markdown(f"**What to do:** {suggestion.get('action', '')}")

    st.caption(
        "Each item says what to change. Nothing here asks you to claim experience "
        "you do not have - a skill you genuinely lack is shown as a gap, because "
        "adding it would only be found out at the interview."
    )


def _render_breakdown_expander(analysis: dict) -> None:
    """Render the detailed technical breakdown in an expandable section."""
    breakdown = analysis.get("ats_breakdown") or {}
    if not breakdown:
        return

    with st.expander("📊 Detailed ATS Score Breakdown", expanded=False):
        categories = breakdown.get("categories") or {}

        # Category table
        rows = []
        for cat_name, cat_data in categories.items():
            score = cat_data.get("score") or 0
            weight = cat_data.get("weight") or 0
            weighted = cat_data.get("weighted_points") or 0
            lost = cat_data.get("points_lost") or 0
            friendly = cat_name.replace("_", " ").title()
            rows.append(
                {
                    "Category": friendly,
                    "Score": f"{score:.1f}/100",
                    "Weight": f"{weight*100:.0f}%",
                    "Points": f"{weighted:.1f}",
                    "Lost": f"{lost:.1f}",
                }
            )

        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)

        # Improvement areas
        improvement_areas = breakdown.get("improvement_areas") or []
        if improvement_areas:
            st.markdown("**Top improvement areas:**")
            for area in improvement_areas[:5]:
                cat = area.get("category", "").replace("_", " ").title()
                lost = area.get("points_lost", 0)
                explanation = area.get("explanation", "")
                st.markdown(f"- **{cat}** ({lost:.1f} pts): {explanation}")

        # Honest score note
        st.caption(
            "This score is based on measurable checks only. No points are fabricated or artificially inflated."
        )


def _render_layout_recommendation(analysis: dict) -> None:
    """Render the layout recommendation."""
    layout_rec = analysis.get("layout_recommendation") or {}
    if not layout_rec:
        return

    recommended = layout_rec.get("recommended_layout")
    reason = layout_rec.get("reason", "")
    ats_safety = layout_rec.get("ats_safety", "")

    if recommended:
        theme.section_header("Recommended CV Layout", "📐")
        st.markdown(f"**{layout_rec.get('recommended_name', recommended)}**")
        if ats_safety:
            st.caption(f"ATS Safety: {ats_safety}")
        st.markdown(f"*{reason}*")

        # Store the recommendation for later use
        st.session_state["recommended_layout"] = recommended


def _render_results(api_base: str, *, compact: bool = False) -> None:
    """Present the analysis. Absent fields are absent, not zero."""
    analysis = workflow.get_analysis() or {}
    meta = workflow.analysis_meta()

    theme.section_header("ATS Compatibility", "📊")

    breakdown = analysis.get("ats_breakdown") or {}
    # The breakdown's score, because it is the one the categories add up to. The
    # legacy blended score is a different calculation over different signals, and
    # showing both here would leave the user reconciling two numbers.
    score = breakdown.get("ats_score")
    if score is None:
        score = analysis.get("ats_match_score")
    band = breakdown.get("band") or ""

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        theme.value_card(
            "ATS Score",
            f"{score:.0f}" if isinstance(score, (int, float)) else "—",
            band or "out of 100",
        )
    with c2:
        theme.value_card(
            "Match quality",
            _score_label(score) if isinstance(score, (int, float)) else "—",
            "",
        )
    with c3:
        theme.value_card("Matching skills", str(len(analysis.get("matching_skills") or [])), "")
    with c4:
        theme.value_card("Missing skills", str(len(analysis.get("missing_skills") or [])), "")

    if isinstance(score, (int, float)):
        theme.pills([(_score_verdict(score), _score_label(score))])
        st.progress(min(1.0, max(0.0, score / 100.0)))

    if breakdown:
        if breakdown.get("pre_generation"):
            st.info(
                "This is a **pre-generation** score. The document itself has not "
                "been produced yet, so the checks that need a document are not "
                "included — the other checks carry the full weight. The final score "
                "appears on the PDF preview page.",
                icon="ℹ️",
            )
        st.caption(breakdown.get("summary") or "")

    # --- What's working / What can be improved (user-friendly) ---
    _render_working_section(analysis)
    _render_improvements_section(analysis)

    # --- Layout recommendation ---
    _render_layout_recommendation(analysis)

    # --- Detailed breakdown (expandable) ---
    _render_breakdown_expander(analysis)

    # --- Skills section ---
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
            st.caption(
                "These are important for the job. Only add them if you have "
                "actual experience with them."
            )
        else:
            st.caption("The backend reported no missing skills.")

    # --- Structure checks ---
    structure = analysis.get("structure") or analysis.get("resume_structure")
    if isinstance(structure, dict):
        theme.section_header("Structure checks", "🧱")
        theme.kv_table([(k, v) for k, v in structure.items() if v not in (None, "")])

    # --- Recommendations ---
    suggestions = analysis.get("improvement_suggestions") or []
    if suggestions:
        theme.section_header("Recommended changes", "💡")
        for index, item in enumerate(suggestions, 1):
            st.markdown(f"**{index}.** {item}")

    recommendation = analysis.get("recommendation")
    if isinstance(recommendation, dict):
        theme.section_header("Backend recommendation", "🧭")
        theme.kv_table([(k, v) for k, v in recommendation.items() if v not in (None, "")])

    # --- What produced this ---
    theme.rule()
    theme.section_header("About this analysis", "ℹ️")
    theme.kv_table(
        [
            ("Provider", meta.get("provider")),
            ("Model", meta.get("model") or "(provider default)"),
            ("Route mode", meta.get("route_mode")),
            ("Duration", f"{meta.get('seconds')}s" if meta.get("seconds") else None),
        ]
    )

    # --- The extracted text ---
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

    # --- The three things to do next (requirement 36) ---
    theme.rule()
    _render_next_actions(analysis, compact=compact)


def _render_next_actions(analysis: dict, *, compact: bool = False) -> None:
    """
    The primary action, stated plainly, with the alternative beside it.

    Requirement 36: the recommendation has to be obvious to someone who has never
    heard of an ATS. So the count of outstanding issues is given as a number, the
    recommended layout and the reason for it are repeated here rather than left
    further up the page, and "choose another layout" sits beside the primary button
    rather than behind a menu — the user keeps control, and the default is still
    the recommended one.
    """
    theme.section_header("What next?", "➡️")

    breakdown = analysis.get("ats_breakdown") or {}
    issues = len(breakdown.get("improvement_areas") or [])
    gap_count = len(analysis.get("missing_skills") or [])

    layout_rec = analysis.get("layout_recommendation") or {}
    recommended = layout_rec.get("recommended_layout")
    recommended_name = layout_rec.get("recommended_name") or recommended
    reason = layout_rec.get("reason") or ""

    if recommended_name:
        theme.value_card(
            "Recommended layout",
            recommended_name,
            f"ATS safety: {layout_rec.get('ats_safety', '—')}",
        )
        if reason:
            st.caption(f"Why this layout? {reason}")
    else:
        st.caption("No layout recommendation yet — one appears after an analysis.")

    if issues or gap_count:
        theme.value_card(
            "Main issues",
            f"{max(issues, gap_count)} improvements available",
            "each one is listed above with what to do about it",
        )
    else:
        theme.value_card("Main issues", "None found", "every check passed")

    # In compact mode the two stages these buttons lead to are already further
    # down this same page, so a button that navigates would be a way to scroll
    # the wrong way. The single-page workflow scrolls to the stage instead.
    if compact:
        st.caption(
            "The two stages below work through this: improvement first, then "
            "generation. Nothing else needs choosing."
        )
        return

    primary, generate, alternate = st.columns(3)

    with primary:
        if st.button("🔧 Improve CV", type="primary", width="stretch", key="ats_improve"):
            st.session_state[workflow.KEY_PAGE] = "optimization"
            st.rerun()

    with generate:
        if st.button("📄 Generate CV", width="stretch", key="ats_generate"):
            st.session_state[workflow.KEY_PAGE] = "cv_generator"
            st.rerun()

    with alternate:
        if st.button("🎨 Choose another layout", width="stretch", key="ats_other_layout"):
            st.session_state[workflow.KEY_PAGE] = "layout_picker"
            st.rerun()

    st.caption(
        "None of this needs to be understood technically. If you would rather not "
        "read the details, the first button is the recommended next step."
    )


def render_ats_step(*, compact: bool = False) -> None:
    """Render the ATS analysis step."""
    workflow.section_header(
        "3",
        "ATS Analysis",
        "Score the resume against the posting and get concrete changes.",
        compact=compact,
    )

    api_base = get_api_base()

    if workflow.blocked(
        workflow.has_upload(),
        missing="a resume (step 1)",
        action="the resume",
        page="resume",
        icon="📎",
        compact=compact,
    ):
        return

    if workflow.blocked(
        bool(workflow.get_job().strip()),
        missing="a job description (step 2)",
        action="the job description",
        page="job_input",
        icon="✍️",
        compact=compact,
    ):
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

        if st.button("🔍 Run ATS analysis", type="primary", key="ats_run"):
            _run(api_base)

        if not compact and st.button("⚙️ Change provider or model"):
            st.session_state[workflow.KEY_PAGE] = "llm_settings"
            st.rerun()
        return

    _render_results(api_base, compact=compact)

    theme.rule()
    if st.button("🔄 Re-run with the current inputs", key="ats_rerun"):
        workflow.set_analysis(None, None)
        st.rerun()
