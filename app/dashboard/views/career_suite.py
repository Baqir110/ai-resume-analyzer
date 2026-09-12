"""Career Suite Page - Comprehensive career tools."""

import re

import streamlit as st

from app.dashboard.components import (
    clear_result,
    get_result,
    render_error_alert,
    render_provider_selector,
    render_quota_card,
    run_with_progress,
    store_result,
)
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


# ============================================================
# Provider panel
# ============================================================


def _render_provider_panel(api_base: str) -> tuple[str, str]:
    """Render provider selector for career tools."""
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


# ============================================================
# Page entry point
# ============================================================


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
    # Tab 1 — Audit Matrix (deterministic, no LLM)
    # --------------------------------------------------------------
    with tab1:
        if st.button(
            "🔍 Run Detailed ATS Audit",
            key="run_audit_btn",
            width="stretch",
            type="primary",
            help="Deterministic ATS analysis. Usually < 5 seconds.",
        ):
            clear_result("audit")
            response, _ = run_with_progress(
                label="ATS audit",
                endpoint=f"{api_base}/api/v1/resume/audit-matrix",
                data={"job_description": st.session_state.get("job_desc", "")},
                files=build_file_payload(),
                steps=[
                    "Parsing resume",
                    "Scoring sections against the job description",
                    "Detecting keyword gaps and formatting issues",
                ],
                hint="Deterministic — usually < 5 seconds.",
                timeout=60,
                seconds_per_step=2.0,
            )
            if response and response.status_code == 200:
                store_result("audit", response.json().get("data", {}))
                st.rerun()
            elif response is not None:
                render_error_alert(response)

        audit = get_result("audit")
        if audit:
            st.success("✅ Audit ready")
            st.json(audit)

    # --------------------------------------------------------------
    # Tab 2 — Cover Letter (text + PDF, language-aware)
    # --------------------------------------------------------------
    with tab2:
        st.caption(
            "Generates a tailored cover letter. Choose text preview (fast) "
            "or an HR-grade PDF. Language auto-detects from the job description."
        )

        company_name = st.text_input("Company Name", value="Target Company", key="cover_company")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            cov_template = st.selectbox(
                "Template Style",
                [
                    "classic_professional",
                    "modern_concise",
                    "story_driven",
                    "value_first",
                ],
                key="cover_template",
            )
        with col_b:
            cov_tone = st.selectbox(
                "Tone",
                ["formal", "startup", "technical"],
                key="cover_tone",
            )
        with col_c:
            cov_language = st.selectbox(
                "Language",
                ["auto", "en", "de"],
                format_func=lambda x: {
                    "auto": "🌐 Auto-detect (from JD)",
                    "en": "🇬🇧 English",
                    "de": "🇩🇪 Deutsch",
                }[x],
                key="cover_language",
                help=(
                    "Auto-detect picks German for German JDs, "
                    "English otherwise. Override if needed."
                ),
            )

        # ---- Build buttons ----
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button(
                "✍️ Preview Cover Letter (text)",
                key="gen_cover_btn",
                width="stretch",
                help="Fast preview. ~15–30 seconds.",
            ):
                clear_result("cover_letter")
                response, _ = run_with_progress(
                    label="Cover Letter (text)",
                    endpoint=f"{api_base}/api/v1/resume/generate-cover-letter",
                    data={
                        "job_description": st.session_state.get("job_desc", ""),
                        "company_name": company_name,
                        "template": cov_template,
                        "tone": cov_tone,
                        "provider": provider,
                        "language": cov_language,
                    },
                    files=build_file_payload(),
                    steps=[
                        "Analyzing the job description and your resume",
                        "Detecting the target language",
                        "Drafting the cover letter body",
                        "Writing a matching cold outreach message",
                    ],
                    hint="LLM-generated — usually 15–30 seconds.",
                    seconds_per_step=6.0,
                )
                if response and response.status_code == 200:
                    store_result("cover_letter", response.json().get("data", {}))
                    st.rerun()
                elif response is not None:
                    render_error_alert(response)

        with btn_col2:
            if st.button(
                "📄 Build Cover Letter PDF",
                key="gen_cover_pdf_btn",
                width="stretch",
                type="primary",
                help=(
                    "HR-grade business letter format. "
                    "Auto-detects language from the JD. ~25–45s."
                ),
            ):
                clear_result("cover_letter_pdf")
                response, _ = run_with_progress(
                    label="Cover Letter PDF",
                    endpoint=f"{api_base}/api/v1/resume/generate-cover-letter-pdf",
                    data={
                        "job_description": st.session_state.get("job_desc", ""),
                        "company_name": company_name,
                        "template": cov_template,
                        "tone": cov_tone,
                        "provider": provider,
                        "language": cov_language,
                    },
                    files=build_file_payload(),
                    steps=[
                        "Detecting the language (German or English)",
                        "Drafting the cover letter body in that language",
                        "Assembling the business-letter LaTeX template",
                        "Compiling to PDF with pdflatex",
                    ],
                    hint="LLM + LaTeX — usually 25–45 seconds.",
                    timeout=180,
                    seconds_per_step=6.0,
                )
                if response and response.status_code == 200:
                    store_result("cover_letter_pdf", response.content)
                    st.rerun()
                elif response is not None:
                    render_error_alert(response)

        # ---- Text result ----
        cover = get_result("cover_letter")
        if cover:
            st.success("✅ Cover letter text ready")
            with st.expander("📝 Cover Letter text", expanded=True):
                st.markdown(cover.get("cover_letter", "—"))
            with st.expander("✉️ Cold outreach message", expanded=False):
                st.info(cover.get("cold_outreach", "—"))

        # ---- PDF result ----
        cover_pdf = get_result("cover_letter_pdf")
        if cover_pdf:
            safe_company = re.sub(r"[^A-Za-z0-9_-]+", "_", company_name or "company")
            st.success(
                f"✅ Cover Letter PDF ready · "
                f"{len(cover_pdf) / 1024:.1f} KB · language: **{cov_language}**"
            )
            st.download_button(
                "📥 Download Cover Letter PDF",
                cover_pdf,
                f"Cover_Letter_{safe_company}.pdf",
                "application/pdf",
                width="stretch",
                type="primary",
                key="download_cover_letter_pdf",
            )

    # --------------------------------------------------------------
    # Tab 3 — Interview Prep (LLM)
    # --------------------------------------------------------------
    with tab3:
        family = st.selectbox(
            "Focus Area",
            ["technical", "behavioral", "product", "leadership", "mlops_devops"],
            key="interview_family",
        )

        if st.button(
            "🎯 Generate Interview Strategy",
            key="gen_interview_btn",
            width="stretch",
            type="primary",
            help="Generates tailored questions, strong answers, and gap defenses.",
        ):
            clear_result("interview")
            response, _ = run_with_progress(
                label="Interview strategy",
                endpoint=f"{api_base}/api/v1/resume/interview-prep",
                data={
                    "job_description": st.session_state.get("job_desc", ""),
                    "family": family,
                    "provider": provider,
                },
                files=build_file_payload(),
                steps=[
                    "Analyzing your resume and the job description",
                    "Generating role-specific questions",
                    "Drafting STAR-style answer frameworks",
                    "Preparing defenses for the skills you're missing",
                ],
                hint="LLM-generated — usually 20–45 seconds.",
                seconds_per_step=6.0,
            )
            if response and response.status_code == 200:
                store_result("interview", response.json().get("data", {}))
                st.rerun()
            elif response is not None:
                render_error_alert(response)

        interview = get_result("interview")
        if interview:
            st.success("✅ Interview strategy ready")
            st.json(interview)

    # --------------------------------------------------------------
    # Tab 4 — LinkedIn Optimizer (LLM)
    # --------------------------------------------------------------
    with tab4:
        target_role = st.text_input(
            "Target Role Title",
            value="Software Engineer",
            key="linkedin_role",
        )

        if st.button(
            "💼 Optimize LinkedIn Profile",
            key="opt_linkedin_btn",
            width="stretch",
            type="primary",
            help="Generates headlines, About section, and skill endorsements.",
        ):
            clear_result("linkedin")
            response, _ = run_with_progress(
                label="LinkedIn profile",
                endpoint=f"{api_base}/api/v1/resume/linkedin-optimize",
                data={"target_role": target_role, "provider": provider},
                files=build_file_payload(),
                steps=[
                    "Reading your resume",
                    "Extracting headline-worthy achievements",
                    "Drafting optimized headlines + About section",
                ],
                hint="LLM-generated — usually 15–30 seconds.",
                seconds_per_step=6.0,
            )
            if response and response.status_code == 200:
                store_result("linkedin", response.json().get("data", {}))
                st.rerun()
            elif response is not None:
                render_error_alert(response)

        linkedin = get_result("linkedin")
        if linkedin:
            st.success("✅ LinkedIn content ready")
            st.json(linkedin)

    # --------------------------------------------------------------
    # Tab 5 — Application Tracker (DB only)
    # --------------------------------------------------------------
    with tab5:
        st.markdown("**Tracked Applications**")
        try:
            r = make_api_request(
                f"{api_base}/api/v1/resume/tracker/applications",
                method="GET",
                timeout=10,
            )
            if r and r.status_code == 200:
                apps = r.json().get("applications", [])
                if apps:
                    import pandas as pd

                    st.dataframe(
                        pd.DataFrame(apps),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No applications tracked yet.")
            else:
                st.warning("Backend did not return the tracker list.")
        except Exception as e:
            st.warning(f"Could not load tracker: {e}")
