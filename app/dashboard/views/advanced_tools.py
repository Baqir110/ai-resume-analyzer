"""Advanced Tools Page - Bulk screening and diff preview."""

import pandas as pd
import streamlit as st

from app.dashboard.components import render_diff_view, render_error_alert
from app.dashboard.helpers import get_api_base, make_api_request


def render_advanced_tools():
    """Render advanced analysis tools."""
    st.header("Advanced tools")

    tab1, tab2 = st.tabs(["🔍 Visual Bullet Diff", "👥 Bulk CV Screening"])

    api_base = get_api_base()

    # Bullet Diff
    with tab1:
        orig_text = st.text_area(
            "Original Bullets (one per line)",
            height=100,
            key="diff_orig",
        )
        opt_text = st.text_area(
            "Optimized Bullets (one per line)",
            height=100,
            key="diff_opt",
        )

        if st.button("Compare Bullets", width="stretch", type="primary", key="diff_compare_btn"):
            payload = {
                "original_bullets": [x.strip() for x in orig_text.splitlines() if x.strip()],
                "optimized_bullets": [x.strip() for x in opt_text.splitlines() if x.strip()],
            }

            response = make_api_request(
                f"{api_base}/api/v1/resume/diff-preview",
                data=payload,
                method="POST",
            )

            if response and response.status_code == 200:
                diffs = response.json().get("diffs", [])
                render_diff_view(diffs)
            else:
                render_error_alert(response)

    # Bulk Screening
    with tab2:
        bulk_job_desc = st.text_area(
            "Target Job Description for Bulk Screening",
            height=120,
            key="bulk_jd",
        )
        bulk_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="bulk_files_uploader",
        )

        if st.button(
            "🚀 Run Bulk Analysis",
            type="primary",
            width="stretch",
            key="bulk_run_btn",
        ):
            if not bulk_job_desc.strip() or not bulk_files:
                st.warning("Please provide both a job description and at least one resume file.")
            else:
                files_payload = [
                    (
                        "resume_files",
                        (f.name, f.getvalue(), f.type or "application/octet-stream"),
                    )
                    for f in bulk_files
                ]

                response = make_api_request(
                    f"{api_base}/api/v1/resume/analyze-bulk",
                    data={"job_description": bulk_job_desc},
                    files=files_payload,
                )

                if response and response.status_code == 200:
                    results = response.json().get("rankings", [])
                    st.success(f"Successfully processed {len(results)} candidate resumes.")

                    df = pd.DataFrame(results)
                    if not df.empty:
                        cols = [
                            c
                            for c in ["filename", "ats_score", "keyword_density"]
                            if c in df.columns
                        ]
                        st.dataframe(df[cols], width="stretch", hide_index=True)
                else:
                    render_error_alert(response)
