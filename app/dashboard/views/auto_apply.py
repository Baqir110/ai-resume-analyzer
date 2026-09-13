"""Auto Apply — trigger Browser Use form filling from the dashboard."""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api/v1"


def render() -> None:
    st.title("🎯 Auto Apply")
    st.caption("AI fills the form. You review. You click Submit.")

    with st.form("apply_form"):
        job_url = st.text_input(
            "Job URL",
            placeholder="https://boards.greenhouse.io/acme/jobs/12345",
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

        why = st.text_area(
            "Why this company? (optional)",
            height=100,
            placeholder="Leave blank to let the agent skip this question.",
        )

        max_steps = st.slider("Max agent steps", 10, 60, 40)
        headless = st.checkbox("Headless (hide browser)", value=False)

        submitted = st.form_submit_button("Fill application")

    if submitted and job_url:
        with st.spinner("Agent is filling the form... this can take 60–120 seconds."):
            try:
                resp = requests.post(
                    f"{API_BASE}/jobs/apply",
                    json={
                        "job_url": job_url,
                        "resume_path": resume_path,
                        "cover_letter_path": cover_path or None,
                        "why_this_company": why,
                        "company_name": company,
                        "role_title": role,
                        "max_steps": max_steps,
                        "headless": headless,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("captcha_encountered"):
                    st.error("CAPTCHA encountered. Solve it manually in the browser, then re-run.")
                elif data.get("success"):
                    st.success(
                        f"Form filled in {data['steps_taken']} steps. "
                        "Review the browser window and click Submit yourself."
                    )
                else:
                    st.warning(data.get("error_message", "Unknown failure."))

                with st.expander("Agent step log", expanded=False):
                    for line in data.get("history_summary", []):
                        st.text(line)

            except requests.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text[:500]}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Recent applications")

    try:
        resp = requests.get(f"{API_BASE}/jobs/applications?limit=20", timeout=10)
        if resp.ok:
            rows = resp.json().get("applications", [])
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No applications yet.")
    except requests.RequestException:
        st.info("Could not load applications log.")
