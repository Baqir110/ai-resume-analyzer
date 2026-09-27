"""Read-only analytics dashboard."""

from __future__ import annotations

import requests
import streamlit as st

from app.dashboard.helpers import get_api_base, get_api_headers


def render_analytics_page() -> None:
    """Render application, pipeline, and LLM usage analytics."""

    st.title("📈 Analytics")
    st.caption("Aggregated, read-only metrics from the local tracking store.")

    api_base = get_api_base()
    period = st.selectbox("Time period", ["7d", "30d", "90d", "all"], index=1)
    try:
        response = requests.get(
            f"{api_base}/api/v1/resume/analytics/summary",
            params={"period": period},
            headers=get_api_headers(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json().get("data", {})
    except requests.RequestException as exc:
        st.warning(f"Could not load analytics: {exc}")
        return

    applications = payload.get("applications", {}) or {}
    llm = payload.get("llm", {}) or {}
    pipeline = payload.get("pipeline", {}) or {}
    skills = payload.get("skills", {}) or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", applications.get("total", 0))
    c2.metric("LLM calls", llm.get("total_calls", 0))
    c3.metric("Tokens", f"{int(llm.get('total_tokens', 0)):,}")
    c4.metric("Estimated cost", f"${float(llm.get('total_cost_usd', 0)):.4f}")

    st.subheader("Applications by status")
    by_status = applications.get("by_status", {}) or {}
    if by_status:
        st.bar_chart(dict(sorted(by_status.items())))
    else:
        st.info("No application records in this period.")

    left, right = st.columns(2)
    with left:
        st.subheader("Top missing skills")
        missing = skills.get("top_missing", []) or []
        st.dataframe(missing, use_container_width=True) if missing else st.info(
            "No skill gaps recorded."
        )
    with right:
        st.subheader("Top matched skills")
        matched = skills.get("top_matched", []) or []
        st.dataframe(matched, use_container_width=True) if matched else st.info(
            "No matches recorded."
        )

    st.subheader("Pipeline operations")
    if pipeline:
        st.json(pipeline)
    else:
        st.info("No pipeline events in this period.")

    errors = payload.get("recent_errors", []) or []
    if errors:
        st.subheader("Recent errors")
        st.dataframe(errors, use_container_width=True)


# Keep the generic name available for callers that used the old view module.
def render() -> None:
    render_analytics_page()
