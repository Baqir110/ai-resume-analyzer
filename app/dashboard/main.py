"""Main Streamlit dashboard entry point with modular page routing."""

import os
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

# Initialize session state
st.session_state.setdefault("api_base", os.getenv("FASTAPI_API_BASE", "http://localhost:8000"))
st.session_state.setdefault("last_analysis", None)
st.session_state.setdefault("uploaded_file_data", None)
st.session_state.setdefault("job_desc", "")
st.session_state.setdefault("current_page", "analyzer")

# Styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1450px; }
    h1 { letter-spacing: -0.035em; }
    h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Header
# ============================================================

st.title("🎯 AI Resume & CV Matcher")
st.caption(
    "One workspace for ATS analysis, targeted CV generation, "
    "interview preparation, and application tracking."
)

# ============================================================
# Sidebar — navigation and config
# IMPORTANT: This block is NOT a fragment. Navigation must rerun the
# whole script so the routed page changes.
# ============================================================

with st.sidebar:
    st.markdown("## Navigation")

    pages = {
        "analyzer": "📊 Analyze Resume",
        "cv_generator": "📄 Generate CV",
        "advanced_tools": "🔧 Advanced Tools",
        "career_suite": "💼 Career Suite",
        "analytics": "📈 Analytics",
    }

    st.selectbox(
        "Choose a tool",
        list(pages.keys()),
        format_func=lambda x: pages[x],
        label_visibility="collapsed",
        key="current_page",
    )

    st.divider()

    with st.expander("⚙️ Backend Configuration", expanded=False):
        api_base_input = st.text_input(
            "FastAPI Backend URL",
            value=st.session_state.get("api_base", "http://localhost:8000"),
        )
        if api_base_input.strip():
            st.session_state["api_base"] = api_base_input.strip().rstrip("/")
        st.caption("API keys are configured in the backend environment.")

    st.divider()

    if st.session_state.get("last_analysis"):
        if st.button("🔄 Start Over", width="stretch"):
            st.session_state["last_analysis"] = None
            st.session_state["uploaded_file_data"] = None
            st.session_state["job_desc"] = ""
            st.rerun()

# ============================================================
# Sidebar — usage + quota panel
# This one IS a fragment so it doesn't refetch on every interaction.
# ============================================================


@st.fragment
def _sidebar_usage_fragment() -> None:
    from app.dashboard.components import render_usage_and_quota_panel

    with st.sidebar:
        render_usage_and_quota_panel(st.session_state.get("api_base", "http://localhost:8000"))


_sidebar_usage_fragment()

# ============================================================
# Route to selected page
# ============================================================

selected_page = st.session_state.get("current_page", "analyzer")

if selected_page == "analyzer":
    from app.dashboard.views.analyzer import render_analyzer_page

    render_analyzer_page()
elif selected_page == "cv_generator":
    from app.dashboard.views.cv_generator import render_cv_generation_page

    render_cv_generation_page()
elif selected_page == "advanced_tools":
    from app.dashboard.views.advanced_tools import render_advanced_tools

    render_advanced_tools()
elif selected_page == "career_suite":
    from app.dashboard.views.career_suite import render_career_suite

    render_career_suite()
elif selected_page == "analytics":
    from app.dashboard.views.analytics import render_analytics_page

    render_analytics_page()

# ============================================================
# Bottom — backend log panel
# This one IS a fragment too.
# ============================================================


@st.fragment
def _bottom_log_fragment() -> None:
    from app.dashboard.components import render_backend_log_panel

    st.divider()
    render_backend_log_panel(st.session_state.get("api_base", "http://localhost:8000"))


_bottom_log_fragment()
