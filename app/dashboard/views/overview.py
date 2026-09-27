"""
Overview: what is configured, what works, and what to do next.

Answers three questions before anything is generated:

*Is the system ready at all?* Backend up, a provider configured, LaTeX present.
*What is actually working?* Every status is a verdict the caller passed in. A
panel that has not called a provider says NOT TESTED rather than showing a green
dot, because a user who sees green stops checking.
*What do I do next?* The first incomplete workflow step, as a link.

No credential reaches this page. Everything rendered here comes from
``backend-status``, which reports provider names, model names and booleans.
"""

from __future__ import annotations

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.helpers import (
    fetch_backend_status,
    fetch_model_catalog,
    fetch_pipeline_metrics,
    get_api_base,
)


def _environment_status(api_base: str) -> None:
    """
    Backend, provider and document toolchain.

    Each row is a verdict derived from what the API reported, not from what the
    dashboard assumes. An unreachable backend is shown as such rather than
    silently rendering an empty panel.
    """
    status = fetch_backend_status(api_base)

    if status is None:
        theme.empty_state(
            "Backend not reachable",
            f"Nothing answered at {api_base}. The API must be running before "
            f"any panel on this page can report anything.",
            action="Start it: python -m uvicorn app.main:app --port 8000",
            icon="🔌",
        )
        return

    routing = status.get("routing") or {}
    ollama = status.get("ollama") or {}
    toolchain = status.get("document_toolchain") or {}
    providers = status.get("providers") or {}

    # -- readiness ---------------------------------------------------------
    provider_name = routing.get("provider") or "(unset)"
    model_name = routing.get("model") or "(unset)"

    configured = [
        name
        for name, info in providers.items()
        if isinstance(info, dict) and info.get("configured")
    ]

    mode = routing.get("mode") or "unknown"
    is_local_only = str(mode).casefold() in {"ollama", "local"}

    if not configured:
        backend_verdict = theme.FAILED
        backend_note = "no provider is configured"
    elif is_local_only:
        backend_verdict = theme.CONFIGURED
        backend_note = "local only — a CV never leaves this machine"
    else:
        backend_verdict = theme.CONFIGURED
        backend_note = "local first, cloud fallback available"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        theme.value_card(
            "Backend",
            "ONLINE" if status.get("status") == "online" else "UNKNOWN",
            status.get("status") or "",
        )
    with c2:
        theme.value_card(
            "Routing mode",
            str(mode),
            "LLM_MODE controls whether a CV may leave the machine",
        )
    with c3:
        theme.value_card(
            "Active provider",
            provider_name,
            f"model: {model_name}",
        )
    with c4:
        theme.value_card(
            "Local or API",
            "LOCAL ONLY" if is_local_only else "LOCAL FIRST + API",
            f"{len(configured)} provider(s) configured",
        )

    st.markdown(theme.status_pill(backend_verdict, backend_note), unsafe_allow_html=True)
    st.caption(
        f"Routing detail: fallback "
        f"{'on' if routing.get('fallback_enabled') else 'off'} · "
        f"max attempts {routing.get('max_provider_attempts')} · "
        f"retries {routing.get('retries')}"
    )

    rule_col, tool_col = st.columns(2)

    # -- local inference ---------------------------------------------------
    with rule_col:
        theme.section_header("Local inference", "🖥️")

        if not ollama.get("configured"):
            theme.pills([(theme.NOT_CONFIGURED, "set OLLAMA_BASE_URL and OLLAMA_MODEL")])
        elif ollama.get("reachable"):
            if ollama.get("model_available"):
                theme.pills(
                    [
                        (
                            theme.PASSED,
                            f"{ollama.get('model')} is installed and reachable",
                        )
                    ]
                )
            else:
                theme.pills([(theme.FAILED, ollama.get("detail") or "model not installed")])
        else:
            theme.pills([(theme.NOT_INSTALLED, ollama.get("detail") or "not reachable")])

        theme.kv_table(
            [
                ("Configured model", ollama.get("model")),
                ("Installed models", len(ollama.get("available_models") or []) or None),
            ]
        )

    # -- document toolchain ------------------------------------------------
    with tool_col:
        theme.section_header("PDF generation", "📄")

        if toolchain.get("pdflatex_available"):
            theme.pills([(theme.AVAILABLE, "pdflatex found")])
            theme.kv_table([("Compiler", toolchain.get("pdflatex_path"))])
            theme.kv_table(
                [
                    ("Layouts available", len(status.get("layouts") or []) or None),
                ]
            )
        else:
            theme.pills(
                [
                    (
                        theme.NOT_INSTALLED,
                        toolchain.get("note") or "pdflatex is not on PATH",
                    )
                ]
            )

    theme.rule()


def _recent_generation(api_base: str) -> None:
    """
    What the last generation actually cost.

    Read from the pipeline event log the backend writes during every request.
    Nothing here is estimated: provider, model, per-request LLM duration, call
    count, retries and PDF compilation time were all recorded at the time.
    """
    metrics = fetch_pipeline_metrics(api_base)

    if metrics is None:
        theme.empty_state(
            "No metrics available",
            "The backend did not return generation metrics.",
            icon="📊",
        )
        return

    summary = metrics.get("summary") or {}

    if not summary.get("generations"):
        theme.empty_state(
            "Nothing generated yet",
            metrics.get("detail") or "Metrics appear here after the first CV or analysis request.",
            action="Run the ATS analysis to start",
            icon="📊",
        )
        return

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        theme.value_card(
            "LLM calls",
            str(summary.get("llm_calls", 0)),
            f"across {summary.get('generations', 0)} request(s)",
        )
    with m2:
        theme.value_card(
            "Retries",
            str(summary.get("retries", 0)),
            "additional attempts on the same provider",
        )
    with m3:
        seconds = (summary.get("llm_duration_ms") or 0) / 1000
        theme.value_card("LLM time", f"{seconds:.1f}s", "total across requests")
    with m4:
        theme.value_card(
            "Tokens",
            f"{summary.get('prompt_tokens', 0):,} / " f"{summary.get('completion_tokens', 0):,}",
            "prompt / completion",
        )
    with m5:
        pdf = summary.get("pdf") or {}
        if pdf.get("valid"):
            theme.value_card(
                "Last PDF",
                f"{pdf.get('pages', '?')} page(s)",
                f"{pdf.get('bytes', 0):,} bytes · " f"{(pdf.get('duration_ms') or 0) / 1000:.1f}s",
            )
        elif pdf:
            theme.value_card("Last PDF", "REJECTED", pdf.get("error") or "")
        else:
            theme.value_card("Last PDF", "—", "no PDF in the recent log")

    requests = metrics.get("requests") or []
    if requests:
        theme.section_header("Recent requests", "🕘")
        rows = [
            {
                "task": row.get("task") or "—",
                "provider": row.get("provider") or "—",
                "model": row.get("model") or "—",
                "attempts": row.get("attempts", 0),
                "retries": row.get("retries", 0),
                "llm_s": round((row.get("llm_duration_ms") or 0) / 1000, 2),
                "outcome": row.get("outcome") or "—",
            }
            for row in requests[:8]
        ]
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )


def _workflow_progress() -> None:
    """
    The four inputs, and where to go next.

    Derived from the shared readiness map rather than re-checked here, so the
    overview cannot disagree with the pages about what is missing.
    """
    theme.section_header("Workflow", "🧭")

    ready = workflow.workflow_readiness()
    page, action = workflow.next_step()

    steps = [
        ("1", "Resume", ready["resume"], "resume"),
        ("2", "Job description", ready["job_description"], "job_input"),
        ("3", "ATS analysis", ready["analysis"], "ats_analysis"),
        ("4", "CV generation", ready["generation"], "cv_generator"),
    ]

    cols = st.columns(4)
    for column, (number, title, done, key) in zip(cols, steps):
        with column:
            theme.value_card(
                f"{number}. {title}",
                "DONE" if done else "PENDING",
                "" if done else "not started",
            )
            if st.button(
                "Open" if not done else "Reopen",
                key=f"overview_open_{key}",
                width="stretch",
                disabled=key == "cv_generator" and not ready["generation"],
            ):
                st.session_state[workflow.KEY_PAGE] = key
                st.rerun()

    theme.rule()
    st.markdown(
        theme.status_pill(
            theme.PASSED if ready["generation"] else theme.NOT_TESTED,
            "ready to generate" if ready["generation"] else f"next: {action}",
        ),
        unsafe_allow_html=True,
    )

    if workflow.has_analysis():
        analysis = workflow.get_analysis() or {}
        score = analysis.get("ats_match_score")
        if score is not None:
            a1, a2, a3 = st.columns(3)
            with a1:
                theme.value_card("ATS score", f"{score:.1f}", "against the posting")
            with a2:
                theme.value_card(
                    "Matching skills",
                    str(len(analysis.get("matching_skills") or [])),
                    "",
                )
            with a3:
                theme.value_card(
                    "Missing skills",
                    str(len(analysis.get("missing_skills") or [])),
                    "",
                )


def render_overview_page() -> None:
    """Render the overview page."""
    theme.step_header(
        "◈",
        "Overview",
        "System status, what is configured, and what to do next.",
    )

    api_base = get_api_base()

    _environment_status(api_base)
    _workflow_progress()
    theme.rule()
    _recent_generation(api_base)

    with st.expander("📚 Configured providers", expanded=False):
        catalog = fetch_model_catalog(api_base)
        if not catalog:
            theme.empty_state("No catalogue available", icon="📚")
        else:
            rows = catalog.get("models") or []
            if not rows:
                theme.empty_state("No providers configured", icon="📚")
            else:
                st.dataframe(rows, width="stretch", hide_index=True)
