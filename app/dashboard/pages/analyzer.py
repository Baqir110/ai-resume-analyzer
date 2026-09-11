"""Resume Analysis Page - Step 1 of workflow."""
import streamlit as st
import requests
from app.dashboard.helpers import make_api_request, get_api_base, build_file_payload
from app.dashboard.components import render_error_alert, render_results_summary, render_improvements


def render_analyzer_page():
    """Render resume analysis input and results."""
    st.subheader("1. Analyze your resume against a target job")
    st.caption(
        "Provide your resume and job description. The analysis powers CV generation and career tools."
    )
    
    # Input section
    input_col, file_col = st.columns([1.45, 1], gap="large")
    
    with input_col:
        job_desc = st.text_area(
            "Target job description",
            height=235,
            placeholder="Paste the complete job description here…",
            value=st.session_state.get('job_desc', ''),
        )
        st.caption("Include responsibilities, required skills, technologies, and qualifications.")
    
    with file_col:
        uploaded_file = st.file_uploader(
            "Current resume",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT.",
        )
        if uploaded_file:
            size_kb = len(uploaded_file.getvalue()) / 1024
            st.success(f"Ready: {uploaded_file.name}")
            st.caption(f"{size_kb:.0f} KB · {uploaded_file.type or 'document'}")
        else:
            st.info("Upload one resume to begin.")
    
    # Check if ready
    ready = bool(uploaded_file and job_desc.strip())
    if not ready:
        missing = []
        if not uploaded_file:
            missing.append("resume")
        if not job_desc.strip():
            missing.append("job description")
        st.caption(f"Still needed: {' and '.join(missing)}.")
    
    # Analyze button
    if st.button(
        "🔍 Analyze ATS match & skill gaps",
        width="stretch",
        type="primary",
        disabled=not ready,
    ):
        api_base = get_api_base()
        
        # Store for later use
        st.session_state['uploaded_file_data'] = (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
        st.session_state['job_desc'] = job_desc
        
        # Make API call
        with st.status("Analyzing resume...", expanded=True) as status:
            st.write("Sending to backend...")
            
            response = make_api_request(
                f"{api_base}/api/v1/resume/analyze",
                data={'job_description': job_desc},
                files={
                    'resume_file': (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=120,
            )
            
            if response and response.status_code == 200:
                st.session_state['last_analysis'] = response.json()
                status.update(label="Analysis complete!", state="complete")
                st.rerun()
            else:
                status.update(label="Analysis failed", state="error")
                render_error_alert(response)
    
    # Display results if available
    if st.session_state.get('last_analysis'):
        st.divider()
        st.subheader("2. Analysis results")
        result = st.session_state['last_analysis']
        render_results_summary(result)
        
        with st.container(border=True):
            st.subheader("Priority improvements")
            render_improvements(result.get('improvement_suggestions', []))
