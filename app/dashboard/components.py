"""Modular dashboard components for Streamlit UI."""
import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd


def render_quota_card(
    provider: str,
    quota_data: Dict[str, Any],
    provider_labels: Dict[str, str],
) -> None:
    """Render a provider quota status card."""
    if not quota_data:
        st.warning(f"No quota data for {provider}")
        return
    
    limits = quota_data.get('limits', {}) or {}
    minute_window = (quota_data.get('windows', {}) or {}).get('minute', {}) or {}
    day_window = (quota_data.get('windows', {}) or {}).get('day', {}) or {}
    
    with st.container(border=True):
        st.markdown(
            f"**{provider_labels.get(provider, provider.title())}**"
        )
        
        # RPM
        if limits.get('rpm', 0):
            rpm_pct = float(minute_window.get('percent_used_requests', 0.0) or 0.0)
            rpm_used = minute_window.get('used_requests', 0)
            rpm_limit = minute_window.get('limit_requests', 0)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(min(rpm_pct / 100, 1.0))
            with col2:
                st.metric("RPM", f"{rpm_used}/{rpm_limit}")
        
        # Daily
        if limits.get('rpd', 0):
            rpd_pct = float(day_window.get('percent_used_requests', 0.0) or 0.0)
            rpd_used = day_window.get('used_requests', 0)
            rpd_limit = day_window.get('limit_requests', 0)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(min(rpd_pct / 100, 1.0))
            with col2:
                st.metric("RPD", f"{rpd_used}/{rpd_limit}")


def render_results_summary(result: Dict[str, Any]) -> None:
    """Render analysis results summary with metrics."""
    if not result:
        st.info("No analysis results available")
        return
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        ats_score = result.get('ats_match_score', 0)
        st.metric("ATS Match Score", f"{ats_score}%")
        st.progress(ats_score / 100)
    
    with m2:
        keyword_score = result.get('keyword_density_score', 0)
        st.metric("Keyword Density", f"{keyword_score}%")
        st.progress(keyword_score / 100)
    
    with m3:
        missing_count = len(result.get('missing_skills', []))
        st.metric("Missing Skills", missing_count)
    
    st.divider()
    
    # Skills chips
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Matching Skills**")
        matching = result.get('matching_skills', [])
        if matching:
            st.markdown(
                " ".join([f'<span style="background:#dcfce7;padding:2px 6px;border-radius:4px;margin:2px;">{s}</span>' for s in matching]),
                unsafe_allow_html=True
            )
        else:
            st.caption("None found")
    
    with col2:
        st.markdown("**Missing Skills**")
        missing = result.get('missing_skills', [])
        if missing:
            st.markdown(
                " ".join([f'<span style="background:#fee2e2;padding:2px 6px;border-radius:4px;margin:2px;">{s}</span>' for s in missing]),
                unsafe_allow_html=True
            )
        else:
            st.caption("None found")


def render_improvements(suggestions: List[str]) -> None:
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
        )
        return selected_route, model
    else:
        provider = st.selectbox(
            "Provider",
            ["gemini", "openai", "anthropic", "groq", "deepseek"],
            format_func=lambda x: x.title(),
        )
        return selected_route, provider


def render_diff_view(diff_data: List[Dict[str, Any]]) -> None:
    """Render word-level diff comparison."""
    for idx, item in enumerate(diff_data, 1):
        with st.expander(
            f"Bullet Point #{idx} (Similarity: {item.get('similarity_ratio', 0)}%)",
            expanded=False,
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Original")
                st.info(item.get('original', 'N/A'))
            
            with col2:
                st.caption("Optimized")
                st.success(item.get('optimized', 'N/A'))
            
            st.caption("Changes:")
            st.markdown(
                item.get('diff_html', ''),
                unsafe_allow_html=True,
            )


def render_error_alert(response) -> None:
    """Render error alert from failed API response."""
    if response is None:
        st.error("Could not connect to backend service")
    else:
        try:
            data = response.json()
            detail = data.get('detail', response.text) if isinstance(data, dict) else response.text
        except:
            detail = response.text
        
        st.error(f"Request failed: {detail[:300]}")
