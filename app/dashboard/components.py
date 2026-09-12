"""Modular dashboard components for Streamlit UI."""

import queue
import threading
import time
from typing import Any

import streamlit as st

# ============================================================
# Quota card
# ============================================================


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

        if limits.get("rpm", 0):
            rpm_pct = float(minute_window.get("percent_used_requests", 0.0) or 0.0)
            rpm_used = minute_window.get("used_requests", 0)
            rpm_limit = minute_window.get("limit_requests", 0)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(min(rpm_pct / 100, 1.0))
            with col2:
                st.metric("RPM", f"{rpm_used}/{rpm_limit}")

        if limits.get("rpd", 0):
            rpd_pct = float(day_window.get("percent_used_requests", 0.0) or 0.0)
            rpd_used = day_window.get("used_requests", 0)
            rpd_limit = day_window.get("limit_requests", 0)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(min(rpd_pct / 100, 1.0))
            with col2:
                st.metric("RPD", f"{rpd_used}/{rpd_limit}")


# ============================================================
# Analysis results
# ============================================================


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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Matching Skills**")
        matching = result.get("matching_skills", [])
        if matching:
            st.markdown(
                " ".join(
                    f'<span style="background:#dcfce7;color:#166534;'
                    f"padding:3px 8px;border-radius:6px;margin:3px 4px 3px 0;"
                    f'display:inline-block;font-weight:600;font-size:0.85rem;">'
                    f"{s}</span>"
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
                    f'<span style="background:#fee2e2;color:#991b1b;'
                    f"padding:3px 8px;border-radius:6px;margin:3px 4px 3px 0;"
                    f'display:inline-block;font-weight:600;font-size:0.85rem;">'
                    f"{s}</span>"
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


def render_recommendation_card(result: dict) -> None:
    """Render the recommended CV layout card."""
    recommendation = (result or {}).get("recommendation") or {}
    if not recommendation:
        return

    label = recommendation.get("label", "Standard")
    reason = recommendation.get("reason", "")

    with st.container(border=True):
        st.subheader("Recommended CV format")
        if recommendation.get("language_mismatch", False):
            st.warning(f"**{label}** — {reason}")
        else:
            st.success(f"**{label}** — {reason}")


# ============================================================
# Diff view
# ============================================================


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


# ============================================================
# Error alert + status badge
# ============================================================


def render_error_alert(response) -> None:
    """Render error alert from failed API response."""
    from app.dashboard.helpers import error_detail

    if response is None:
        st.error("Could not connect to backend service")
        return

    st.error(f"Request failed: {error_detail(response)[:400]}")


def render_status_badge(meta: dict) -> None:
    """Render an HTTP status badge with timing."""
    code = meta.get("status_code")
    elapsed = meta.get("elapsed", 0.0)
    error = meta.get("error")
    retried = meta.get("retried", False)

    suffix = " · retried" if retried else ""

    if code is None:
        st.error(f"❌ No response · {elapsed:.1f}s · {error or 'unknown error'}")
    elif 200 <= code < 300:
        st.success(f"✅ HTTP {code} · {elapsed:.1f}s{suffix}")
    elif 400 <= code < 500:
        st.warning(f"⚠️ HTTP {code} · {elapsed:.1f}s{suffix} · {error or ''}")
    else:
        st.error(f"❌ HTTP {code} · {elapsed:.1f}s{suffix} · {error or ''}")


# ============================================================
# Provider selector
# ============================================================


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
                "gpt-5.6-luna",
                "gpt-6-astra",
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


# ============================================================
# Progress helpers — status widget
# ============================================================


def run_with_status(
    label: str,
    endpoint: str,
    data: dict,
    files: dict | None = None,
    steps: list[str] | None = None,
    hint: str = "This usually takes 15–45 seconds.",
    timeout: int = 180,
):
    """Simple status widget (non-animated)."""
    from app.dashboard.helpers import make_api_request_verbose

    steps = steps or []
    steps_md = "\n".join(f"- {s}" for s in steps)

    with st.status(f"⏳ Building {label}…", expanded=True) as status:
        st.markdown(f"📤 **POST** `{endpoint}`")
        if steps_md:
            st.markdown("**Behind the scenes:**")
            st.markdown(steps_md)
        st.markdown("---")
        st.markdown(f"⏱️ *{hint}*")

        response, meta = make_api_request_verbose(endpoint, data=data, files=files, timeout=timeout)
        elapsed = meta.get("elapsed", 0.0)
        code = meta.get("status_code")

        if code == 200:
            status.update(
                label=f"✅ {label} ready · HTTP {code} · {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label=f"❌ {label} failed · HTTP {code or '—'}",
                state="error",
                expanded=False,
            )

    return response, elapsed


def run_with_progress(
    label: str,
    endpoint: str,
    data: dict,
    files: dict | None = None,
    steps: list[str] | None = None,
    hint: str = "This usually takes 15–45 seconds.",
    timeout: int = 180,
    seconds_per_step: float = 6.0,
):
    """Animated step-by-step progress with HTTP status code at the end."""
    from app.dashboard.helpers import make_api_request_verbose

    if not steps:
        steps = [
            "Sending request to backend",
            "Processing on server",
            "Waiting for response",
        ]

    with st.status(f"⏳ Building {label}…", expanded=True) as status:
        st.markdown(f"📤 **POST** `{endpoint}`")
        st.divider()

        progress_bar = st.progress(0.0)
        step_log = st.empty()

        q: queue.Queue = queue.Queue()

        def worker():
            try:
                resp, meta = make_api_request_verbose(
                    endpoint, data=data, files=files, timeout=timeout
                )
                q.put(("done", resp, meta))
            except Exception as exc:  # noqa: BLE001
                q.put(("error", None, {"error": str(exc), "status_code": None}))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        completed: list[str] = []
        t0 = time.perf_counter()
        step_idx = 0
        estimated_total = max(20.0, seconds_per_step * len(steps))

        while thread.is_alive():
            elapsed = time.perf_counter() - t0
            target_idx = min(len(steps) - 1, int(elapsed / seconds_per_step))
            while step_idx < target_idx:
                completed.append(f"✅ {steps[step_idx]}")
                step_idx += 1

            lines = list(completed)
            lines.append(f"⏳ **{steps[step_idx]}** &nbsp;·&nbsp; *{elapsed:.1f}s*")
            step_log.markdown("\n\n".join(lines))
            progress_bar.progress(min(0.95, elapsed / estimated_total))
            time.sleep(0.25)

        while step_idx < len(steps):
            completed.append(f"✅ {steps[step_idx]}")
            step_idx += 1

        try:
            kind, response, meta = q.get_nowait()
        except queue.Empty:
            kind, response, meta = (
                "error",
                None,
                {"error": "No response from worker", "status_code": None},
            )

        elapsed = time.perf_counter() - t0
        code = meta.get("status_code")
        err = meta.get("error")

        if kind == "error":
            step_log.markdown("\n\n".join(completed + [f"❌ **Failed**: {err}"]))
            progress_bar.progress(1.0)
            status.update(
                label=f"❌ {label} failed after {elapsed:.1f}s",
                state="error",
                expanded=False,
            )
            st.error(err or "Unknown error")
            return None, elapsed

        if code == 200:
            step_log.markdown("\n\n".join(completed + [f"✅ **HTTP {code}** in {elapsed:.1f}s"]))
            progress_bar.progress(1.0)
            status.update(
                label=f"✅ {label} ready · HTTP {code} · {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )
        else:
            step_log.markdown("\n\n".join(completed + [f"❌ **HTTP {code}** — {err or 'failed'}"]))
            progress_bar.progress(1.0)
            status.update(
                label=f"❌ {label} failed · HTTP {code}",
                state="error",
                expanded=False,
            )
            if err:
                st.error(err)

    return response, elapsed


# ============================================================
# Result persistence
# ============================================================


def store_result(key: str, value) -> None:
    """Persist a result across Streamlit reruns."""
    st.session_state[f"_result_{key}"] = value


def get_result(key: str):
    """Retrieve a previously stored result, or None."""
    return st.session_state.get(f"_result_{key}")


def clear_result(key: str) -> None:
    """Clear a stored result."""
    st.session_state.pop(f"_result_{key}", None)


# ============================================================
# Backend processing log panel
# ============================================================


@st.fragment
def render_backend_log_panel(api_base: str) -> None:
    """Render the backend processing log viewer.

    All column values are forced to str() so pyarrow can serialize the
    dataframe without ArrowInvalid errors on mixed-type columns.
    """
    import pandas as pd
    import requests

    from app.dashboard.helpers import fetch_processing_log

    with st.expander("🛠️ Backend processing log", expanded=False):
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear logs", key="clear_logs_btn", width="stretch"):
                try:
                    r = requests.delete(
                        f"{api_base.rstrip('/')}/api/v1/resume/processing-log",
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success("Log cleared.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Clear failed: HTTP {r.status_code}")
                except Exception as exc:
                    st.error(f"Clear failed: {exc}")

        logs = fetch_processing_log(api_base, 50)

        if not logs:
            st.info("No events yet. Run an analysis to see logs here.")
            return

        all_kinds = sorted({str(e.get("kind") or "llm") for e in logs})
        kind_filter = st.selectbox(
            "Filter by kind",
            ["All"] + all_kinds,
            key="log_kind_filter",
        )

        filtered = (
            logs
            if kind_filter == "All"
            else [e for e in logs if str(e.get("kind") or "llm") == kind_filter]
        )

        rows = []
        for e in reversed(filtered):
            dur = e.get("duration_ms")
            rows.append(
                {
                    "Time (UTC)": str((e.get("timestamp") or "")[:19].replace("T", " ")),
                    "Kind": str(e.get("kind") or "llm"),
                    "Event": str(e.get("event") or ""),
                    "Provider": str(e.get("provider") or ""),
                    "Model": str(e.get("model") or ""),
                    "Status": str(e.get("status") or ""),
                    "Duration ms": "" if dur is None else str(dur),
                    "Error": str(e.get("error") or "")[:120],
                }
            )

        df = pd.DataFrame(rows, dtype=str)
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(f"{len(filtered)}/{len(logs)} events shown")


# ============================================================
# Usage + quota sidebar panel
# ============================================================


@st.fragment
def render_usage_and_quota_panel(api_base: str) -> None:
    """Render the usage + quota expander."""
    from app.dashboard.helpers import fetch_quota_status, fetch_usage_summary, format_tokens

    with st.expander("📊 Usage, limits & provider health", expanded=False):
        left, right = st.columns(2)

        with left:
            st.markdown("**Token usage**")
            period_labels = {
                "today": "Today",
                "7d": "Last 7 days",
                "30d": "Last 30 days",
                "all": "All time",
            }
            period_choice = st.selectbox(
                "Time window",
                list(period_labels.keys()),
                index=3,
                format_func=lambda x: period_labels[x],
                key="usage_period_panel",
            )
            summary = fetch_usage_summary(api_base, period_choice)
            if not summary:
                st.info("No usage data yet.")
            else:
                u1, u2, u3 = st.columns(3)
                u1.metric("Tokens", format_tokens(summary.get("total_tokens", 0)))
                u2.metric("Calls", summary.get("success_calls", 0))
                u3.metric(
                    "Est. cost",
                    f"${summary.get('estimated_cost_usd', 0.0) or 0.0:.4f}",
                )

        with right:
            st.markdown("**Rate limits**")
            quota_snapshot = fetch_quota_status(api_base)
            if not quota_snapshot:
                st.info("No quota data yet.")
            else:
                for prov, data in quota_snapshot.items():
                    if prov == "ollama":
                        continue
                    limits = data.get("limits", {}) or {}
                    minute = (data.get("windows", {}) or {}).get("minute", {}) or {}
                    day = (data.get("windows", {}) or {}).get("day", {}) or {}
                    hits = data.get("rate_limit_hits_24h", 0)

                    st.markdown(f"**{prov.title()}**")
                    if hits:
                        st.caption(f"429 hits in 24h: {hits}")

                    for label, window, key in [
                        ("RPM", minute, "rpm"),
                        ("RPD", day, "rpd"),
                    ]:
                        if limits.get(key, 0):
                            pct = float(window.get("percent_used_requests", 0.0) or 0.0)
                            used = window.get("used_requests", 0)
                            lim = window.get("limit_requests", 0)
                            st.progress(min(pct / 100, 1.0))
                            st.caption(f"{label}: {used}/{lim} ({pct:.0f}%)")
