"""Analytics Page - Usage, costs, and trends."""

import pandas as pd
import streamlit as st

from app.dashboard.helpers import format_tokens, get_api_base, make_api_request


def render_analytics_page():
    """Render analytics dashboard with usage trends."""
    st.header("Analytics")

    api_base = get_api_base()

    period_labels = {
        "today": "Today",
        "7d": "Last 7 days",
        "30d": "Last 30 days",
        "90d": "Last 90 days",
        "all": "All time",
    }

    period_choice = st.selectbox(
        "Time window",
        list(period_labels.keys()),
        index=2,
        format_func=lambda x: period_labels[x],
        key="analytics_period",
    )

    # Fetch analytics data
    response = make_api_request(
        f"{api_base}/api/v1/resume/analytics/summary",
        data={"period": period_choice},
        method="GET",
    )

    if not response or response.status_code != 200:
        st.info("No analytics data available yet")
        return

    data = response.json().get("data", {})
    llm = data.get("llm", {})
    apps = data.get("applications", {})

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LLM calls", llm.get("total_calls", 0))
    m2.metric("Tokens", format_tokens(llm.get("total_tokens", 0)))
    m3.metric("Est. cost", f"${llm.get('total_cost_usd', 0.0):.4f}")
    m4.metric("Failure rate", f"{llm.get('failure_rate', 0.0) * 100:.1f}%")

    st.divider()

    # Charts
    left, right = st.columns(2)

    with left:
        st.markdown("**Token usage over time**")
        series = llm.get("tokens_by_day", [])
        if series:
            df = pd.DataFrame(series).set_index("date")
            st.bar_chart(df, height=220, color="#2563eb")
        else:
            st.caption("No data for this period")

    with right:
        st.markdown("**Cost over time**")
        series = llm.get("cost_by_day", [])
        if series:
            df = pd.DataFrame(series).set_index("date")
            st.line_chart(df, height=220, color="#16a34a")
        else:
            st.caption("No data for this period")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("**Applications by status**")
        status_counts = apps.get("by_status", {})
        if status_counts:
            df = pd.DataFrame(
                [{"status": k, "count": v} for k, v in status_counts.items()]
            ).set_index("status")
            st.bar_chart(df, height=220, color="#6366f1")

    with right:
        st.markdown("**ATS score distribution**")
        buckets = apps.get("by_score_bucket", {})
        if buckets:
            df = pd.DataFrame(
                [{"score_range": k, "count": v} for k, v in buckets.items()]
            ).set_index("score_range")
            st.bar_chart(df, height=220, color="#f59e0b")
