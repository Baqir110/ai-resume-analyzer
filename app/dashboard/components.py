"""Modular dashboard components for Streamlit UI."""

from typing import Any

import streamlit as st


def render_quota_card(
    provider: str,
    quota_data: dict[str, Any],
    provider_labels: dict[str, str],
) -> None:
    """Render a provider quota status card."""
    if not quota_data:
        st.warning(f"No quota data for {provider}")
        return

    limits = quota_data.get("limits", {}) or {}
    minute_window = (quota_data.get("windows", {}) or {}).get("minute", {}) or {}
    day_window = (quota_data.get("windows", {}) or {}).get("day", {}) or {}

    with st.container(border=True):
        st.markdown(f"**{provider_labels.get(provider, provider.title())}**")

        # RPM
        if limits.get("rpm", 0):
            rpm_pct = float(minute_window.get("percent_used_requests", 0.0) or 0.0)
            rpm_used = minute_window.get("used_requests", 0)
            rpm_limit = minute_window.get("limit_requests", 0)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(min(rpm_pct / 100, 1.0))
            with col2:
                st.metric("RPM", f"{rpm_used}/{rpm_limit}")

        # Daily
        if limits.get("rpd", 0):
            rpd_pct = float(day_window.get("percent_used_requests", 0.0) or 0.0)
            rpd_used = day_window.get("used_requests", 0)
            rpd_limit = day_window.get("limit_requests", 0)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(min(rpd_pct / 100, 1.0))
            with col2:
                st.metric("RPD", f"{rpd_used}/{rpd_limit}")


def render_results_summary(result: dict[str, Any]) -> None:
    """Render analysis results summary with metrics."""
    if not result:
        st.info("No analysis results available")
        return

    m1, m2, m3 = st.columns(3)

    with m1:
        ats_score = result.get("ats_match_score", 0)
        st.metric("ATS Match Score", f"{ats_score}%")
        st.progress(ats_score / 100)

    with m2:
        keyword_score = result.get("keyword_density_score", 0)
        st.metric("Keyword Density", f"{keyword_score}%")
        st.progress(keyword_score / 100)

    with m3:
        missing_count = len(result.get("missing_skills", []))
        st.metric("Missing Skills", missing_count)

    st.divider()

    # Skills chips
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Matching Skills**")
        matching = result.get("matching_skills", [])
        if matching:
            st.markdown(
                " ".join(
                    f'<span style="background:#dcfce7;padding:2px 6px;'
                    f'border-radius:4px;margin:2px;">{s}</span>'
                    for s in matching
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("None found")

    with col2:
        st.markdown("**Missing Skills**")
        missing = result.get("missing_skills", [])
        if missing:
            st.markdown(
                " ".join(
                    f'<span style="background:#fee2e2;padding:2px 6px;'
                    f'border-radius:4px;margin:2px;">{s}</span>'
                    for s in missing
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("None found")


def render_improvements(suggestions: list[str]) -> None:
    """Render improvement suggestions."""
    if not suggestions:
        st.info("No improvement suggestions available")
        return

    for i, suggestion in enumerate(suggestions, 1):
        with st.expander(f"💡 Suggestion {i}", expanded=False):
            st.write(suggestion)


def render_provider_selector() -> tuple[str, str]:
    """Render LLM provider selector.

    Returns:
        Tuple of (route_mode, provider_or_model)
    """
    route_options = ["experiential", "direct"]
    route_labels = {"experiential": "Experiential Cloud", "direct": "Direct API"}

    selected_route = st.selectbox(
        "Routing Mode",
        route_options,
        format_func=lambda x: route_labels[x],
        key="provider_selector_route",
        help=(
            "Experiential Cloud routes through the managed gateway. "
            "Direct API calls the provider endpoint directly using your key."
        ),
    )

    if selected_route == "experiential":
        model = st.selectbox(
            "Experiential Model",
            [
                "gpt-6-astra",
                "gpt-5.6-luna",
                "deepseek-v4-flash",
                "qwen3.8-27b",
                "gemini-3.7-flash",
            ],
            key="provider_selector_experiential_model",
        )
        return selected_route, model
    else:
        provider = st.selectbox(
            "Provider",
            ["gemini", "openai", "anthropic", "groq", "deepseek"],
            format_func=lambda x: x.title(),
            key="provider_selector_direct_provider",
        )
        return selected_route, provider


def render_diff_view(diff_data: list[dict[str, Any]]) -> None:
    """Render word-level diff comparison."""
    for idx, item in enumerate(diff_data, 1):
        with st.expander(
            f"Bullet Point #{idx} (Similarity: {item.get('similarity_ratio', 0)}%)",
            expanded=False,
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Original")
                st.info(item.get("original", "N/A"))

            with col2:
                st.caption("Optimized")
                st.success(item.get("optimized", "N/A"))

            st.caption("Changes:")
            st.markdown(
                item.get("diff_html", ""),
                unsafe_allow_html=True,
            )


def render_error_alert(response) -> None:
    """Render error alert from failed API response."""
    if response is None:
        st.error("Could not connect to backend service")
        return

    detail: str
    try:
        data = response.json()
        if isinstance(data, dict):
            detail = str(data.get("detail", response.text))
        else:
            detail = response.text
    except Exception:
        detail = getattr(response, "text", "Unknown error")

    st.error(f"Request failed: {detail[:300]}")
