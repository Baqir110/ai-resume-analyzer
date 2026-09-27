"""
Dashboard entry point.

Navigation is grouped into the order the work happens in, so the sidebar reads
as a workflow rather than a list of tools:

    Workflow   Overview → Resume → Job description → ATS analysis →
               Optimisation → CV generation → PDF preview
    Setup      LLM & model settings → CV layout
    Tools      Everything that was here before, unchanged
    System     Diagnostics & settings

Two Streamlit constraints shape the code below, and both are load-bearing.

*Navigation must not be inside a fragment.* A fragment reruns on its own; a
widget inside one cannot drive the page dispatch, because the script body that
reads it does not re-execute. The navigation block is therefore plain script.

*The provider selector uses fixed widget keys.* ``render_provider_selector``
can be called at most once per script run, or Streamlit raises on the duplicate
key. Each page that needs it calls it exactly once.

All existing pages are still reachable. The previous seven are unchanged and
appear under Tools; the new pages wrap the same API calls rather than
reimplementing them, and the session-state contract the old pages rely on
(``uploaded_file_data``, ``job_desc``, ``last_analysis``) is preserved exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

st.set_page_config(
    page_title="AI Resume & CV Optimization Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.dashboard import theme, workflow  # noqa: E402
from app.dashboard.helpers import get_api_base  # noqa: E402

workflow.ensure_defaults()
theme.inject_css()

#: Navigation, in the order the work is done. Grouped, because a flat list of
#: ten items gives no hint about which comes first or which are alternatives.
NAV_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Workflow",
        (
            ("overview", "◈  Overview"),
            ("resume", "1  Resume"),
            ("job_input", "2  Job Description"),
            ("ats_analysis", "3  ATS Analysis"),
            ("optimization", "4  Optimisation"),
            ("cv_generator", "5  CV Generation"),
            ("pdf_preview", "6  PDF Preview"),
        ),
    ),
    (
        "Setup",
        (
            ("llm_settings", "⚙  LLM & Model"),
            ("layout_picker", "▤  CV Layout"),
        ),
    ),
    (
        "Tools",
        (
            ("advanced_tools", "🔧 Advanced Tools"),
            ("career_suite", "💼 Career Suite"),
            ("analytics", "📈 Analytics"),
            ("auto_apply", "🎯 Auto Apply"),
            ("agent_control", "🤖 Agent Control"),
        ),
    ),
    (
        "System",
        (("diagnostics", "🩺 Diagnostics"),),
    ),
)

#: Flat key -> label map, for the selectbox.
_LABELS: dict[str, str] = {key: label for _group, entries in NAV_GROUPS for key, label in entries}

#: The first page, and the page reached by "start over".
DEFAULT_PAGE = "overview"


# ============================================================
# Header
# ============================================================

st.title("🎯 AI Resume & CV Hub")
st.caption(
    "ATS analysis, targeted CV generation, LaTeX and PDF output — with the "
    "provider and model chosen in configuration, not in code."
)


# ============================================================
# Sidebar — navigation
# IMPORTANT: This block is NOT a fragment. Navigation must rerun the whole
# script so the routed page changes.
# ============================================================

with st.sidebar:
    st.markdown("### Navigation")

    # The selectbox is what drives dispatch, so its options are the flat key
    # list in group order. The group headings above it are presentation only:
    # duplicating the navigation as clickable items would give two widgets
    # bound to the same state, and only one of them could be the source of
    # truth.
    for group, entries in NAV_GROUPS:
        st.markdown(f'<div class="nb-nav-group">{group}</div>', unsafe_allow_html=True)
        for key, label in entries:
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if st.session_state.get(workflow.KEY_PAGE) == key else "secondary",
            ):
                st.session_state[workflow.KEY_PAGE] = key
                st.rerun()

    st.divider()

    with st.expander("⚙️ Backend", expanded=False):
        api_base_input = st.text_input(
            "FastAPI URL",
            value=st.session_state.get(workflow.KEY_API_BASE, ""),
            key="sidebar_api_base",
        )
        if api_base_input.strip():
            st.session_state[workflow.KEY_API_BASE] = api_base_input.strip().rstrip("/")
        st.caption(
            "API keys are configured in the backend environment and are never " "displayed here."
        )

    st.divider()

    # The current step, so the sidebar says what to do next without the user
    # having to remember.
    _ready = workflow.workflow_readiness()
    _done = sum(1 for value in _ready.values() if value)
    st.markdown(
        f'<div class="nb-card"><div class="nb-card-title">Workflow</div>'
        f'<div class="nb-card-value">{_done} of 4 steps done</div></div>',
        unsafe_allow_html=True,
    )
    if not all(_ready.values()):
        _page, _action = workflow.next_step()
        st.caption(f"Next: {_action}")
        if st.button("Go there →", width="stretch"):
            st.session_state[workflow.KEY_PAGE] = _page
            st.rerun()

    if st.button("🔄 Start over", width="stretch"):
        workflow.reset_workflow()
        st.session_state[workflow.KEY_PAGE] = DEFAULT_PAGE
        st.rerun()


# ============================================================
# Sidebar — usage + quota
# Fragment, so it does not refetch on every interaction.
# ============================================================


@st.fragment
def _sidebar_usage_fragment() -> None:
    from app.dashboard.components import render_usage_and_quota_panel

    with st.sidebar:
        render_usage_and_quota_panel(get_api_base())


_sidebar_usage_fragment()


# ============================================================
# Route to the selected page
# ============================================================

selected_page = st.session_state.get(workflow.KEY_PAGE, DEFAULT_PAGE)


def _render(page: str) -> None:
    """
    Dispatch to one page.

    Imports are deferred so a page with a heavy dependency is only loaded when
    it is actually opened, and so a failure in one page's imports cannot stop
    the others from rendering.
    """
    if page == "overview":
        from app.dashboard.views.overview import render_overview_page

        render_overview_page()
    elif page == "resume":
        from app.dashboard.views.inputs import render_resume_step

        render_resume_step()
    elif page == "job_input":
        from app.dashboard.views.inputs import render_job_input_step

        render_job_input_step()
    elif page == "ats_analysis":
        from app.dashboard.views.ats_step import render_ats_step

        render_ats_step()
    elif page == "optimization":
        from app.dashboard.views.optimization import render_optimization_page

        render_optimization_page()
    elif page == "cv_generator":
        from app.dashboard.views.cv_generator import render_cv_generation_page

        render_cv_generation_page()
    elif page == "pdf_preview":
        from app.dashboard.views.pdf_preview import render_pdf_preview

        render_pdf_preview()
    elif page == "llm_settings":
        from app.dashboard.views.llm_settings import render_llm_settings_page

        render_llm_settings_page()
    elif page == "layout_picker":
        from app.dashboard.views.layout_picker import render_layout_picker

        render_layout_picker()
    elif page == "diagnostics":
        from app.dashboard.views.diagnostics import render_diagnostics_page

        render_diagnostics_page()

    # -- the pages that existed before this redesign, unchanged ------------
    elif page == "advanced_tools":
        from app.dashboard.views.advanced_tools import render_advanced_tools

        render_advanced_tools()
    elif page == "career_suite":
        from app.dashboard.views.career_suite import render_career_suite

        render_career_suite()
    elif page == "analytics":
        from app.dashboard.views.analytics import render_analytics_page

        render_analytics_page()
    elif page == "auto_apply":
        from app.dashboard.views.auto_apply import render

        render()
    elif page == "agent_control":
        from app.dashboard.views.agent_control import render as render_agent

        render_agent()
    else:
        # An unknown key means session state and the navigation disagree, which
        # can only happen if a page was renamed. Falling back is better than a
        # blank screen.
        st.session_state[workflow.KEY_PAGE] = DEFAULT_PAGE
        st.rerun()


_render(selected_page)


# ============================================================
# Bottom — backend log panel
# Fragment, like the sidebar panel above.
# ============================================================


@st.fragment
def _bottom_log_fragment() -> None:
    from app.dashboard.components import render_backend_log_panel

    st.divider()
    render_backend_log_panel(get_api_base())


_bottom_log_fragment()
