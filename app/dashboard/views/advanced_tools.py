"""Advanced Tools Page - Bulk screening and diff preview."""

import pandas as pd
import streamlit as st

from app.dashboard.components import (
    clear_result,
    get_result,
    render_diff_view,
    render_error_alert,
    run_with_progress,
    store_result,
)
from app.dashboard.helpers import get_api_base, make_api_request


def render_advanced_tools():
    """Render advanced analysis tools."""
    st.header("Advanced tools")

    tab1, tab2 = st.tabs(["🔍 Visual Bullet Diff", "👥 Bulk CV Screening"])

    api_base = get_api_base()

    with tab1:
        st.caption("Compares two sets of bullet points word-by-word. " "Runs locally — instant.")

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

        if st.button(
            "🔍 Compare Bullets",
            width="stretch",
            type="primary",
            key="diff_compare_btn",
        ):
            payload = {
                "original_bullets": [x.strip() for x in orig_text.splitlines() if x.strip()],
                "optimized_bullets": [x.strip() for x in opt_text.splitlines() if x.strip()],
            }

            with st.status("⏳ Comparing bullets…", expanded=False) as status:
                response = make_api_request(
                    f"{api_base}/api/v1/resume/diff-preview",
                    data=payload,
                    method="POST",
                    timeout=30,
                )
                if response and response.status_code == 200:
                    status.update(
                        label=f"✅ Comparison ready · HTTP {response.status_code}",
                        state="complete",
                    )
                    clear_result("diff")
                    store_result("diff", response.json().get("diffs", []))
                else:
                    status.update(label="❌ Comparison failed", state="error")
                    render_error_alert(response)

        diffs = get_result("diff")
        if diffs:
            render_diff_view(diffs)

    with tab2:
        st.caption(
            "Screens multiple resumes against one job description. "
            "Time scales with the number of files."
        )

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

        if bulk_files:
            st.caption(f"📎 {len(bulk_files)} file(s) queued")

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
                        (
                            f.name,
                            f.getvalue(),
                            f.type or "application/octet-stream",
                        ),
                    )
                    for f in bulk_files
                ]

                clear_result("bulk")
                response, _ = run_with_progress(
                    label=f"Bulk screening ({len(bulk_files)} file(s))",
                    endpoint=f"{api_base}/api/v1/resume/analyze-bulk",
                    data={"job_description": bulk_job_desc},
                    files=files_payload,
                    steps=[
                        f"Parsing {len(bulk_files)} resume(s)",
                        "Running ATS scoring on each",
                        "Ranking candidates by match score",
                    ],
                    hint=(
                        f"Estimated {max(5, len(bulk_files) * 3)}–"
                        f"{max(15, len(bulk_files) * 8)} seconds."
                    ),
                    timeout=300,
                    seconds_per_step=5.0,
                )

                if response and response.status_code == 200:
                    results = response.json().get("rankings", [])
                    store_result("bulk", results)
                    st.rerun()
                elif response is not None:
                    render_error_alert(response)

        bulk_results = get_result("bulk")
        if bulk_results:
            st.success(f"✅ Processed {len(bulk_results)} candidate resume(s)")
            df = pd.DataFrame(bulk_results)
            if not df.empty:
                cols = [c for c in ["filename", "ats_score", "keyword_density"] if c in df.columns]
                st.dataframe(df[cols], width="stretch", hide_index=True)
            else:
                st.info("No results returned.")
