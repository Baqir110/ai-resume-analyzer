"""Career Suite Page - Comprehensive career tools."""

import pandas as pd
import requests
import streamlit as st

from app.dashboard.components import render_error_alert, render_provider_selector, render_quota_card
from app.dashboard.helpers import (
    build_file_payload,
    fetch_quota_status,
    get_api_base,
    make_api_request,
)

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "experiential": "Experiential Cloud",
}


def _render_provider_panel(api_base: str) -> tuple[str, str]:
    """Render provider selector for career tools.

    Career endpoints only accept ``provider`` (no model_name / route_mode).
    Returns:
        (provider, model_name) — model_name kept for consistency but unused.
    """
    with st.expander("⚙️ AI Provider", expanded=False):
        route_mode, provider_or_model = render_provider_selector()

        if route_mode == "experiential":
            provider = "experiential"
            model_name = provider_or_model
        else:
            provider = provider_or_model
            model_name = ""

        st.caption(f"Provider: **{provider}** · Model: **{model_name or '(default)'}**")

        quota_data = fetch_quota_status(api_base)
        if quota_data:
            st.divider()
            st.markdown("**Live quota status**")
            for prov, data in quota_data.items():
                render_quota_card(prov, data, PROVIDER_LABELS)

    return provider, model_name


def render_career_suite():
    """Render career tools (cover letters, interview prep, etc.)."""
    if not st.session_state.get("last_analysis"):
        st.info("Complete Step 1 (Analyze) first")
        return

    st.header("Career suite")

    api_base = get_api_base()
    provider, model_name = _render_provider_panel(api_base)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Audit Matrix",
            "✉️ Cover Letter",
            "🎯 Interview Prep",
            "💼 LinkedIn Optimizer",
            "📌 Application Pipeline",
        ]
    )

    # --------------------------------------------------------------
    # Audit Matrix (no LLM — provider not needed)
    # --------------------------------------------------------------
    with tab1:
        if st.button("Run Detailed ATS Audit", key="run_audit_btn", width="stretch"):
            response = make_api_request(
                f"{api_base}/api/v1/resume/audit-matrix",
                data={"job_description": st.session_state.get("job_desc", "")},
                files=build_file_payload(),
            )

            if response and response.status_code == 200:
                audit_data = response.json().get("data", {})
                st.json(audit_data)
            else:
                render_error_alert(response)

    # --------------------------------------------------------------
    # Cover Letter
    # --------------------------------------------------------------
    with tab2:
        company_name = st.text_input("Company Name", value="Target Company", key="cover_company")
        cov_template = st.selectbox(
            "Template",
            ["classic_professional", "modern_concise", "story_driven", "value_first"],
            key="cover_template",
        )

        if st.button(
            "Generate Cover Letter",
            key="gen_cover_btn",
            width="stretch",
            type="primary",
        ):
            response = make_api_request(
                f"{api_base}/api/v1/resume/generate-cover-letter",
                data={
                    "job_description": st.session_state.get("job_desc", ""),
                    "company_name": company_name,
                    "template": cov_template,
                    "provider": provider,
                },
                files=build_file_payload(),
            )

            if response and response.status_code == 200:
                data = response.json().get("data", {})
                st.subheader("Cover Letter")
                st.info(data.get("cover_letter", "N/A"))
                st.subheader("Cold Outreach Message")
                st.success(data.get("cold_outreach", "N/A"))
            else:
                render_error_alert(response)

    # --------------------------------------------------------------
    # Interview Prep
    # --------------------------------------------------------------
    with tab3:
        family = st.selectbox(
            "Focus Area",
            ["technical", "behavioral", "product", "leadership", "mlops_devops"],
            key="interview_family",
        )

        if st.button(
            "Generate Interview Strategy",
            key="gen_interview_btn",
            width="stretch",
            type="primary",
        ):
            response = make_api_request(
                f"{api_base}/api/v1/resume/interview-prep",
                data={
                    "job_description": st.session_state.get("job_desc", ""),
                    "family": family,
                    "provider": provider,
                },
                files=build_file_payload(),
            )

            if response and response.status_code == 200:
                prep_data = response.json().get("data", {})
                st.json(prep_data)
            else:
                render_error_alert(response)

    # --------------------------------------------------------------
    # LinkedIn Optimizer
    # --------------------------------------------------------------
    with tab4:
        target_role = st.text_input(
            "Target Role Title",
            value="Software Engineer",
            key="linkedin_role",
        )

        if st.button(
            "Optimize LinkedIn Profile",
            key="opt_linkedin_btn",
            width="stretch",
            type="primary",
        ):
            response = make_api_request(
                f"{api_base}/api/v1/resume/linkedin-optimize",
                data={
                    "target_role": target_role,
                    "provider": provider,
                },
                files=build_file_payload(),
            )

            if response and response.status_code == 200:
                linkedin_data = response.json().get("data", {})
                st.json(linkedin_data)
            else:
                render_error_alert(response)

    # --------------------------------------------------------------
    # Application Tracker
    # --------------------------------------------------------------
    with tab5:
        st.markdown("**Tracked Applications**")
        try:
            r = requests.get(
                f"{api_base}/api/v1/resume/tracker/applications",
                timeout=10,
            )
            if r.status_code == 200:
                apps = r.json().get("applications", [])
                if apps:
                    st.dataframe(
                        pd.DataFrame(apps),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No applications tracked yet")
            else:
                st.warning(f"Backend returned {r.status_code}")
        except Exception as e:
            st.warning(f"Could not load tracker: {e}")
