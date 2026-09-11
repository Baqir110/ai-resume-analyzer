"""Main Streamlit dashboard entry point with modular page routing."""
import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="AI Resume & CV Optimization Hub",
    page_icon="🎯",
    layout="wide",
)

# Initialize session state
st.session_state.setdefault('api_base', os.getenv('FASTAPI_API_BASE', 'http://localhost:8000'))
st.session_state.setdefault('last_analysis', None)
st.session_state.setdefault('uploaded_file_data', None)
st.session_state.setdefault('job_desc', '')
st.session_state.setdefault('current_page', 'analyzer')

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

# Header
st.title("🎯 AI Resume & CV Matcher")
st.caption(
    "One workspace for ATS analysis, targeted CV generation, interview preparation, and application tracking."
)

# Sidebar navigation
with st.sidebar:
    st.markdown("## Navigation")
    
    pages = {
        'analyzer': ('📊 Analyze Resume', 'Analyzer'),
        'cv_generator': ('📄 Generate CV', 'CVGenerator'),
        'advanced_tools': ('🔧 Advanced Tools', 'AdvancedTools'),
        'career_suite': ('💼 Career Suite', 'CareerSuite'),
        'analytics': ('📈 Analytics', 'Analytics'),
    }
    
    selected_page = st.selectbox(
        "Choose a tool",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.get('current_page', 'analyzer')),
        format_func=lambda x: pages[x][0],
        label_visibility="collapsed",
    )
    
    st.session_state['current_page'] = selected_page
    
    st.divider()
    
    # Backend config
    with st.expander("⚙️ Backend Configuration", expanded=False):
        api_base = st.text_input(
            "FastAPI Backend URL",
            value=st.session_state.get('api_base', 'http://localhost:8000'),
        )
        if api_base.strip():
            st.session_state['api_base'] = api_base.strip().rstrip('/')
        st.caption("API keys are configured in the backend environment.")
    
    st.divider()
    
    if st.session_state.get('last_analysis'):
        if st.button("🔄 Start Over", width="stretch"):
            st.session_state['last_analysis'] = None
            st.session_state['uploaded_file_data'] = None
            st.session_state['job_desc'] = ''
            st.rerun()

# Route to selected page
if selected_page == 'analyzer':
    from app.dashboard.pages.analyzer import render_analyzer_page
    render_analyzer_page()
elif selected_page == 'cv_generator':
    from app.dashboard.pages.cv_generator import render_cv_generation_page
    render_cv_generation_page()
elif selected_page == 'advanced_tools':
    from app.dashboard.pages.advanced_tools import render_advanced_tools
    render_advanced_tools()
elif selected_page == 'career_suite':
    from app.dashboard.pages.career_suite import render_career_suite
    render_career_suite()
elif selected_page == 'analytics':
    from app.dashboard.pages.analytics import render_analytics_page
    render_analytics_page()
