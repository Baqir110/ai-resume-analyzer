"""
Diagnostics and settings.

Where a user goes when something is not working, and where they go to change
where the dashboard points. Three groups:

*System* -- backend reachability, the document toolchain, and the routing the
backend reports. The compiler is reported by the API process, not looked up in
the browser, because those can be different machines.

*Providers* -- configured, available and tested, kept apart. A row is only ever
marked PASSED after something actually called it.

*Configuration* -- the backend URL, and a reset that clears the workflow.

Nothing here can reveal a credential. The API key is read by the dashboard
process from its own environment to attach to requests; it is never rendered,
and no endpoint returns it.
"""

from __future__ import annotations

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.helpers import (
    clear_cached_status,
    fetch_backend_status,
    fetch_model_discovery,
    fetch_processing_log,
    get_api_base,
)

_LABELS = {
    "ollama": "Ollama (local)",
    "omniroute": "OmniRoute (local gateway)",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "claude": "Anthropic Claude",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
    "cerebras": "Cerebras",
    "cloudflare": "Cloudflare Workers AI",
    "github": "GitHub Models",
    "huggingface": "Hugging Face",
    "experiential": "Experiential Labs",
}


def _system(api_base: str) -> None:
    theme.section_header("System", "🖥️")

    status = fetch_backend_status(api_base)

    if status is None:
        theme.pills([(theme.FAILED, f"no response from {api_base}")])
        st.markdown(
            "The dashboard cannot report on the backend because the backend is "
            "not answering. Start it with:"
        )
        st.code("python -m uvicorn app.main:app --host 127.0.0.1 --port 8000", language="bash")
        return

    theme.pills([(theme.PASSED, f"backend online at {api_base}")])

    routing = status.get("routing") or {}
    toolchain = status.get("document_toolchain") or {}
    ollama = status.get("ollama") or {}

    theme.rule()
    st.markdown("**Routing**")
    theme.kv_table(
        [
            ("Mode", routing.get("mode")),
            ("Provider", routing.get("provider")),
            ("Model", routing.get("model")),
            (
                "Fallback",
                "enabled" if routing.get("fallback_enabled") else "disabled",
            ),
            ("Max provider attempts", routing.get("max_provider_attempts")),
            ("Retries", routing.get("retries")),
            ("Cloud fallback allowed", "yes" if routing.get("allows_cloud_fallback") else "no"),
        ]
    )

    if not routing.get("allows_cloud_fallback"):
        theme.pills([(theme.PASSED, "local only — a CV cannot leave this machine")])

    theme.rule()
    st.markdown("**Document toolchain**")
    if toolchain.get("pdflatex_available"):
        theme.pills([(theme.PASSED, "pdflatex available")])
        theme.kv_table(
            [
                ("Compiler", toolchain.get("pdflatex_path")),
                ("PDF generation", "available"),
                ("Layouts", len(status.get("layouts") or [])),
            ]
        )
    else:
        theme.pills([(theme.FAILED, toolchain.get("note") or "pdflatex is not on PATH")])
        st.markdown(
            "Install MiKTeX or TeX Live and make sure `pdflatex` is on PATH "
            "for the process running the API. LaTeX generation and PDF output "
            "are unavailable until then; ATS analysis still works."
        )

    theme.rule()
    st.markdown("**Local inference**")
    if not ollama.get("configured"):
        theme.pills([(theme.NOT_CONFIGURED, "Ollama is not configured")])
    elif ollama.get("reachable") and ollama.get("model_available"):
        theme.pills([(theme.PASSED, f"{ollama.get('model')} is installed")])
        with st.expander("Installed models", expanded=False):
            theme.chip_row(ollama.get("available_models") or [])
    else:
        theme.pills([(theme.FAILED, ollama.get("detail") or "the local model is not ready")])


def _providers(api_base: str) -> None:
    theme.section_header("Providers", "🧠")

    discovery = fetch_model_discovery(api_base)
    providers = (discovery or {}).get("providers") or {}

    if not providers:
        theme.empty_state(
            "No provider information",
            "The backend did not return a provider list.",
            icon="🧠",
        )
        return

    rows = []
    for name, info in providers.items():
        if not isinstance(info, dict):
            continue

        configured = bool(info.get("configured"))
        listing = info.get("status")

        if not configured:
            available = theme.NOT_CONFIGURED
        elif listing == "ok":
            available = theme.AVAILABLE
        elif listing in {"manual_configuration", "discovery_failed"}:
            available = theme.CONFIGURED
        else:
            available = theme.FAILED

        rows.append(
            {
                "provider": _LABELS.get(name, name),
                "id": name,
                "side": "local" if name in {"ollama", "omniroute"} else "API",
                "configured": "yes" if configured else "no",
                "available": available,
                "smoke tested": theme.NOT_TESTED,
                "model": info.get("configured_model") or "—",
                "listed": len(info.get("models") or []),
                "note": (info.get("detail") or "")[:90],
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)

    theme.rule()
    st.markdown("**How to test a provider for real**")
    st.code("python -m scripts.llm_smoke", language="bash")
    st.code("python -m scripts.llm_smoke ollama groq", language="bash")
    st.markdown(
        "That sends a fixed harmless prompt and reports latency, token usage "
        "and whether usable text came back. It sends no CV, no job description "
        "and no credential. Until it has been run for a provider, this table "
        "says `NOT TESTED` — which is the honest answer, not a green tick."
    )

    st.markdown("**How to test the full CV pipeline for a local model**")
    st.code("python -m scripts.model_matrix --all", language="bash")
    st.markdown(
        "A model can answer a smoke test and still be unable to produce a "
        "complete CV. That command runs the real path end to end on a synthetic "
        "fixture and reports which models finished it."
    )


def _configuration(api_base: str) -> None:
    theme.section_header("Configuration", "🔧")

    new_base = st.text_input(
        "Backend URL",
        value=api_base,
        key="diagnostics_api_base",
        help="The FastAPI service the dashboard talks to.",
    )
    if new_base.strip():
        st.session_state[workflow.KEY_API_BASE] = new_base.strip().rstrip("/")
        if st.session_state[workflow.KEY_API_BASE] != api_base:
            clear_cached_status(api_base)
            st.caption("Saved. Refreshing status…")
            st.rerun()

    st.caption(
        "The API key is read from the dashboard process's own environment to "
        "attach to requests. It is never displayed here and no endpoint returns "
        "it."
    )

    theme.rule()
    st.markdown("**Session**")

    upload = workflow.get_upload()
    theme.kv_table(
        [
            ("Resume", upload[0] if upload else None),
            (
                "Job description",
                f"{workflow.job_stats()['chars']:,} characters"
                if workflow.get_job().strip()
                else None,
            ),
            ("ATS score", _score_text()),
            ("Selected layout", workflow.get_layout()),
            (
                "Provider",
                workflow.generation_choice()["provider"],
            ),
        ]
    )

    left, right = st.columns(2)
    with left:
        if st.button("🔄 Refresh all status", width="stretch"):
            clear_cached_status(api_base)
            st.rerun()
    with right:
        if st.button("🗑️ Clear this session", width="stretch"):
            workflow.reset_workflow()
            st.rerun()

    st.caption(
        "Clearing removes the resume, the job description, the analysis and "
        "every generated file from this browser session. Nothing on the server "
        "is touched."
    )


def _score_text() -> str | None:
    analysis = workflow.get_analysis() or {}
    score = analysis.get("ats_match_score")
    return f"{score:.1f}" if isinstance(score, (int, float)) else None


def _recent_events(api_base: str) -> None:
    theme.section_header("Recent pipeline events", "📜")

    logs = fetch_processing_log(api_base, limit=40)

    if not logs:
        theme.empty_state(
            "No events",
            "The backend writes an event per request. Nothing has been logged "
            "yet, or the log could not be read.",
            icon="📜",
        )
        return

    rows = [
        {
            "time": row.get("timestamp", "")[11:19],
            "event": row.get("event"),
            "provider": row.get("provider") or row.get("route_mode") or "",
            "duration_ms": row.get("duration_ms"),
            "detail": row.get("error") or row.get("layout") or "",
        }
        for row in logs[-25:]
    ]

    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Read from the backend's pipeline log. No prompt or response content is "
        "recorded there, which is why there is nothing sensitive to show."
    )


def render_diagnostics_page() -> None:
    """Render the diagnostics and settings page."""
    theme.step_header(
        "🔧",
        "Diagnostics & Settings",
        "What the system reports about itself, and where it points.",
    )

    api_base = get_api_base()

    _system(api_base)
    theme.rule()
    _providers(api_base)
    theme.rule()
    _configuration(api_base)
    theme.rule()
    _recent_events(api_base)
