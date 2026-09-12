"""Resume Analysis Page - Step 1 of workflow."""

import streamlit as st

from app.dashboard.components import (
    render_error_alert,
    render_improvements,
    render_provider_selector,
    render_quota_card,
    render_results_summary,
)
from app.dashboard.helpers import fetch_quota_status, get_api_base, make_api_request

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "experiential": "Experiential Cloud",
}


def _render_provider_panel(api_base: str, key_prefix: str = "analyzer") -> tuple[str, str, str]:
    """Render the AI provider / model selector + live quota status.

    Returns:
        (route_mode, provider, model_name)
    """
    with st.expander("⚙️ AI Provider & Model", expanded=False):
        route_mode, provider_or_model = render_provider_selector()

        if route_mode == "experiential":
            provider = "experiential"
            model_name = provider_or_model
        else:
            provider = provider_or_model
            model_name = ""

        st.caption(
            f"Routing: **{route_mode}** · Provider: **{provider}** · "
            f"Model: **{model_name or '(default)'}**"
        )

        # Live quota status (best-effort, silent on failure)
        quota_data = fetch_quota_status(api_base)
        if quota_data:
            st.divider()
            st.markdown("**Live quota status**")
            for prov, data in quota_data.items():
                render_quota_card(prov, data, PROVIDER_LABELS)

    return route_mode, provider, model_name


def render_analyzer_page():
    """Render resume analysis input and results."""
    st.subheader("1. Analyze your resume against a target job")
    st.caption(
        "Provide your resume and job description. The analysis powers CV generation and career tools."
    )

    api_base = get_api_base()

    # Input section
    input_col, file_col = st.columns([1.45, 1], gap="large")

    with input_col:
        job_desc = st.text_area(
            "Target job description",
            height=235,
            placeholder="Paste the complete job description here…",
            value=st.session_state.get("job_desc", ""),
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

    # Provider / Model / Quota panel
    route_mode, provider, model_name = _render_provider_panel(api_base, key_prefix="analyzer")

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
        # Store for later use
        st.session_state["uploaded_file_data"] = (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
        st.session_state["job_desc"] = job_desc
        st.session_state["route_mode"] = route_mode
        st.session_state["provider"] = provider
        st.session_state["model_name"] = model_name

        # Make API call
        with st.status("Analyzing resume...", expanded=True) as status:
            st.write(f"Sending to backend ({provider} · {model_name or 'default'})...")

            response = make_api_request(
                f"{api_base}/api/v1/resume/analyze",
                data={
                    "job_description": job_desc,
                    "provider": provider,
                    "model_name": model_name or "",
                    "route_mode": route_mode,
                },
                files={
                    "resume_file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=120,
            )

            if response and response.status_code == 200:
                st.session_state["last_analysis"] = response.json()
                status.update(label="Analysis complete!", state="complete")
                st.rerun()
            else:
                status.update(label="Analysis failed", state="error")
                render_error_alert(response)

    # Display results if available
    if st.session_state.get("last_analysis"):
        st.divider()
        st.subheader("2. Analysis results")
        result = st.session_state["last_analysis"]
        render_results_summary(result)

        with st.container(border=True):
            st.subheader("Priority improvements")
            render_improvements(result.get("improvement_suggestions", []))
