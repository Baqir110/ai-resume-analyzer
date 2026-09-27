"""Auto Apply — Browser Use form filling from the dashboard."""

import requests
import streamlit as st

from app.dashboard.components import render_provider_selector
from app.dashboard.helpers import get_api_base, get_api_headers

# Curated agent LLM options. Keep in sync with _AGENT_LLM_PRESETS
# in app/services/jobs/browser_use_applier.py.
_AGENT_LLM_LABELS = {
    "auto": "Auto — same as CV generation",
    "gateway-gpt-4o": "Gateway · gpt-4o (recommended)",
    "gateway-gpt-4o-mini": "Gateway · gpt-4o-mini",
    "gateway-deepseek-v4-flash": "Gateway · deepseek-v4-flash (free)",
    "gateway-gpt-5.6-luna": "Gateway · gpt-5.6-luna (fast, less reliable)",
    "direct-openai": "Direct OpenAI · gpt-4o-mini",
    "direct-anthropic": "Direct Anthropic · claude-3-5-haiku",
    "direct-groq": "Direct Groq · llama-3.3-70b",
}


def _render_cv_llm_selector() -> tuple[str, str, str]:
    """
    Render the CV-generation LLM selector.

    IMPORTANT: render_provider_selector() uses a fixed widget key.
    Call this AT MOST ONCE per page render. Do not call from inside
    multiple tabs or containers.
    """
    st.markdown("**CV generation LLM**")
    st.caption("Used for ATS analysis and tailored CV generation.")

    route_mode, provider_or_model = render_provider_selector()

    if route_mode == "experiential":
        provider = "experiential"
        model_name = provider_or_model
    else:
        provider = provider_or_model
        model_name = ""

    return route_mode, provider, model_name


def _render_agent_llm_selector(key_prefix: str) -> str:
    """
    Render the Browser Use agent LLM selector.

    Accepts a key_prefix so it can be rendered in multiple tabs
    without colliding on widget keys.
    """
    st.markdown("**Agent LLM (Browser Use)**")
    st.caption(
        "Used to drive the browser and fill forms. Requires reliable "
        "structured output — gpt-4o is the safest choice."
    )
    return st.selectbox(
        "Agent model",
        list(_AGENT_LLM_LABELS.keys()),
        format_func=lambda k: _AGENT_LLM_LABELS[k],
        key=f"{key_prefix}_agent_llm",
        label_visibility="collapsed",
    )


# ============================================================
# Tailored CV + Apply tab
# ============================================================


def _render_tailored_tab() -> None:
    api_base = get_api_base()
    st.caption(
        "Paste the job URL. The pipeline fetches the JD, company, and "
        "role, analyzes your resume, generates a tailored CV, and fills "
        "the application form."
    )

    job_url = st.text_input(
        "Job URL",
        key="auto_apply_job_url",
        placeholder="https://job-boards.greenhouse.io/liveperson/jobs/8067576",
    )

    last_url_key = "auto_apply_last_attempted_url"
    meta_key = "auto_apply_meta"
    jd_widget_key = "auto_apply_jd_widget"
    company_widget_key = "auto_apply_company_widget"
    role_widget_key = "auto_apply_role_widget"
    status_key = "auto_apply_fetch_status"

    new_url = bool(job_url) and job_url != st.session_state.get(last_url_key)

    if new_url:
        with st.spinner("Detecting ATS and fetching job details..."):
            try:
                resp = requests.post(
                    f"{api_base}/api/v1/jobs/fetch-jd",
                    json={"job_url": job_url},
                    headers=get_api_headers(),
                    timeout=30,
                )
                resp.raise_for_status()
                payload = resp.json()

                st.session_state[meta_key] = payload
                st.session_state[jd_widget_key] = payload["jd_text"]
                st.session_state[company_widget_key] = payload.get("company", "")
                st.session_state[role_widget_key] = payload.get("role", "")
                st.session_state[last_url_key] = job_url

                ats = payload.get("source_ats") or "unknown"
                st.session_state[status_key] = (
                    "ok",
                    f"Detected {ats}. "
                    f"Company: {payload.get('company') or '?'} · "
                    f"Role: {payload.get('role') or '?'} · "
                    f"{payload.get('chars', 0)} chars of JD.",
                )
            except requests.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text[:300]
                st.session_state[last_url_key] = job_url
                st.session_state[status_key] = (
                    "error",
                    f"Could not fetch job details: {detail}",
                )
            except requests.RequestException as e:
                st.session_state[last_url_key] = job_url
                st.session_state[status_key] = (
                    "error",
                    f"Request failed: {e}",
                )

    status = st.session_state.get(status_key)
    if status:
        kind, msg = status
        if kind == "ok":
            st.success(msg)
        else:
            st.warning(msg + "  Fill in the fields manually below.")
            if st.button("🔄 Retry fetch"):
                st.session_state[last_url_key] = None
                st.session_state[status_key] = None
                st.rerun()

    meta = st.session_state.get(meta_key) or {}
    if meta.get("source_ats") and meta["source_ats"] != "html_fallback":
        loc = meta.get("location") or "—"
        st.caption(f"📍 Location: {loc}  ·  🌐 Source: {meta.get('source_ats')}")

    # ---- LLM selectors (rendered exactly once per page) ----
    with st.expander("⚙️ AI Provider & Model", expanded=False):
        route_mode, provider, model_name = _render_cv_llm_selector()
        st.divider()
        agent_llm = _render_agent_llm_selector(key_prefix="tailored")

    with st.form("tailored_form"):
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input(
                "Company name",
                key=company_widget_key,
                help="Auto-filled. Edit if wrong.",
            )
        with col2:
            role = st.text_input(
                "Role title",
                key=role_widget_key,
                help="Auto-filled. Edit if wrong.",
            )

        job_description = st.text_area(
            "Job description",
            height=220,
            key=jd_widget_key,
            help="Auto-filled. Edit if the fetch missed anything.",
        )

        why = st.text_area(
            "Why this company? (optional)",
            height=80,
        )

        uploaded = st.file_uploader(
            "Your current resume",
            type=["pdf", "docx", "txt"],
            help="PDF, DOCX, or TXT. Used for both ATS analysis and tailoring.",
        )

        col3, col4 = st.columns(2)
        with col3:
            layout_style = st.selectbox(
                "CV layout",
                [
                    "auto",
                    "german_corporate",
                    "german_modern",
                    "german_classic",
                    "international_ats",
                    "standard",
                    "technical_lead",
                ],
                index=0,
            )
        with col4:
            max_steps = st.slider("Max agent steps", 10, 60, 40)
            auto_submit = st.checkbox(
                "Submit automatically (dangerous)",
                value=False,
                help="Leave disabled to fill the form and review it manually.",
            )

        submitted = st.form_submit_button("Generate tailored CV + Fill application")

    if not submitted:
        return

    missing = []
    if not job_url:
        missing.append("Job URL")
    if not (job_description or "").strip():
        missing.append("Job description")
    if not uploaded:
        missing.append("Resume file")
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
        return

    assert uploaded is not None

    files = {
        "resume_file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/octet-stream",
        ),
    }
    data = {
        "job_url": job_url,
        "job_description": job_description,
        "company_name": company,
        "role_title": role,
        "why_this_company": why,
        "layout_style": layout_style,
        "max_steps": str(max_steps),
        "provider": provider,
        "model_name": model_name or "",
        "route_mode": route_mode,
        "agent_llm": agent_llm,
        "auto_submit": str(auto_submit).lower(),
        "handle_email_verification": "false",
    }

    with st.spinner(
        "Running pipeline: analyze → tailor CV → compile PDF → fill form. "
        "This typically takes 2–4 minutes."
    ):
        try:
            resp = requests.post(
                f"{api_base}/api/v1/jobs/apply-with-tailored-cv",
                files=files,
                data=data,
                headers=get_api_headers(),
                timeout=900,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.HTTPError as e:
            st.error(f"API error {e.response.status_code}: " f"{e.response.text[:500]}")
            return
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            return

    if payload.get("cv_path"):
        st.info(f"Tailored CV saved: `{payload['cv_path']}`")

    if payload.get("ok"):
        app = payload.get("application") or {}
        st.success(
            f"Form filled in {app.get('steps_taken', '?')} steps. "
            "Review the tab in your browser and click Submit yourself."
        )
    else:
        stage = payload.get("stage_failed", "unknown")
        err = payload.get("error", "Unknown failure.")
        st.error(f"Pipeline failed at stage `{stage}`: {err}")

    app = payload.get("application")
    if app and app.get("history_summary"):
        with st.expander("Agent step log", expanded=False):
            for line in app["history_summary"]:
                st.text(line)


# ============================================================
# Quick Apply tab
# ============================================================


def _render_quick_tab() -> None:
    api_base = get_api_base()
    st.caption(
        "Fill a form using a resume PDF that already exists on disk. "
        "Use this when you've already generated a CV elsewhere."
    )

    # Quick Apply does NOT generate a CV, so only the agent LLM
    # matters here. No CV-generation selector.
    with st.expander("⚙️ Agent Model", expanded=False):
        agent_llm = _render_agent_llm_selector(key_prefix="quick")

    with st.form("quick_form"):
        job_url = st.text_input(
            "Job URL",
            placeholder="https://job-boards.greenhouse.io/company/jobs/12345",
        )
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company")
            resume_path = st.text_input(
                "Resume PDF path",
                value="data/outputs/resume_latest.pdf",
            )
        with col2:
            role = st.text_input("Role title")
            cover_path = st.text_input("Cover letter PDF path (optional)", value="")

        why = st.text_area("Why this company? (optional)", height=80)
        max_steps = st.slider("Max agent steps", 10, 60, 40)
        auto_submit = st.checkbox(
            "Submit automatically (dangerous)",
            value=False,
            help="Leave disabled to review the form before submission.",
        )

        submitted = st.form_submit_button("Fill application")

    if not submitted or not job_url:
        return

    with st.spinner("Agent is filling the form... 60–120 seconds."):
        try:
            resp = requests.post(
                f"{api_base}/api/v1/jobs/apply",
                json={
                    "job_url": job_url,
                    "resume_path": resume_path,
                    "cover_letter_path": cover_path or None,
                    "why_this_company": why,
                    "company_name": company,
                    "role_title": role,
                    "max_steps": max_steps,
                    "agent_llm": agent_llm,
                    "auto_submit": auto_submit,
                    "handle_email_verification": False,
                },
                headers=get_api_headers(),
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("captcha_encountered"):
                st.error("CAPTCHA encountered. Solve it in the browser, then re-run.")
            elif data.get("success"):
                st.success(
                    f"Form filled in {data['steps_taken']} steps. "
                    "Review the tab and click Submit yourself."
                )
            else:
                st.warning(data.get("error_message", "Unknown failure."))

            with st.expander("Agent step log", expanded=False):
                for line in data.get("history_summary", []):
                    st.text(line)
        except requests.HTTPError as e:
            st.error(f"API error {e.response.status_code}: " f"{e.response.text[:500]}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")


# ============================================================
# Entry point
# ============================================================


def render() -> None:
    st.title("🎯 Auto Apply")
    st.caption("AI fills the form. You review. You click Submit.")

    tab_tailored, tab_quick = st.tabs(["🎨 Tailored CV + Apply", "⚡ Quick Apply (existing PDF)"])
    with tab_tailored:
        _render_tailored_tab()
    with tab_quick:
        _render_quick_tab()

    st.divider()
    st.subheader("Recent applications")
    try:
        resp = requests.get(
            f"{get_api_base()}/api/v1/jobs/applications?limit=20",
            headers=get_api_headers(),
            timeout=10,
        )
        if resp.ok:
            rows = resp.json().get("applications", [])
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No applications yet.")
    except requests.RequestException:
        st.info("Could not load applications log.")
