"""CV Generation Page - Step 2 of workflow."""

import streamlit as st

from app.dashboard.components import (
    clear_result,
    get_result,
    render_error_alert,
    render_provider_selector,
    render_quota_card,
    run_with_progress,
    store_result,
)
from app.dashboard.helpers import build_file_payload, fetch_quota_status, get_api_base

TEMPLATE_LABELS = {
    "auto": "🎯 Auto-detect (match JD language)",
    "international_ats": "International English ATS (Single-Page)",
    "academic": "Academic / Research (Serif, Education-first)",
    "technical_lead": "Technical Lead (Open Source + Speaking)",
    "hr_executive_gold": "HR Gold Standard (Executive)",
    "german_corporate": "Corporate Slate Navy",
    "german_minimal_ats": "German Minimal ATS (Single-Column)",
    "german_modern": "Modern Two-Column",
    "german_classic": "German Classic Single-Column PDF",
    "standard": "Standard ATS Single-Column",
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "experiential": "Experiential Cloud",
}


def _render_provider_panel(api_base: str) -> tuple[str, str, str]:
    with st.expander("⚙️ AI Provider & Model", expanded=False):
        route_mode, provider_or_model = render_provider_selector()

        if route_mode == "experiential":
            provider = "experiential"
            model_name = provider_or_model
        else:
            provider = provider_or_model
            model_name = ""

        st.caption(
            f"Routing: **{route_mode}** · Provider: **{provider}** · "
            f"Model: **{model_name or '(default)'}**"
        )

        quota_data = fetch_quota_status(api_base)
        if quota_data:
            st.divider()
            st.markdown("**Live quota status**")
            for prov, data in quota_data.items():
                render_quota_card(prov, data, PROVIDER_LABELS)

    return route_mode, provider, model_name


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


def _run_build(
    kind: str,
    label: str,
    endpoint: str,
    payload: dict,
    filename: str,
    mime: str,
) -> None:
    """Run a build with animated progress; store the result."""
    clear_result(kind)

    steps = [
        "Uploading resume + job description",
        f"Sending request to {endpoint.split('/')[-1]}",
        "LLM is tailoring the CV content",
        "Assembling the document from the template",
    ]
    if kind == "pdf":
        steps.append("Compiling LaTeX to PDF with pdflatex")
    elif kind == "docx":
        steps.append("Building Word document")
    steps.append("Finalizing and returning the file")

    response, elapsed = run_with_progress(
        label=label,
        endpoint=endpoint,
        data=payload,
        files=build_file_payload(),
        steps=steps,
        hint="Usually 20–45 seconds.",
        timeout=180,
        seconds_per_step=5.0,
    )

    if response is None:
        return

    if response.status_code == 200:
        store_result(
            kind,
            {
                "content": response.content,
                "filename": filename,
                "mime": mime,
                "elapsed": elapsed,
                "size_kb": len(response.content) / 1024,
            },
        )
        st.rerun()
    else:
        render_error_alert(response)


def _render_download(kind: str, label: str) -> None:
    build = get_result(kind)
    if not build:
        return

    st.success(
        f"✅ **{label} ready** · `{build['filename']}` · "
        f"{build['size_kb']:.1f} KB · built in **{build['elapsed']:.1f}s**"
    )
    st.download_button(
        f"📥 Download {label}",
        build["content"],
        build["filename"],
        build["mime"],
        width="stretch",
        type="primary",
        key=f"download_{kind}",
    )


def render_cv_generation_page():
    if not st.session_state.get("last_analysis"):
        st.info("Complete Step 1 (Analyze) first")
        return

    st.header("Generate your tailored CV")

    api_base = get_api_base()
    route_mode, provider, model_name = _render_provider_panel(api_base)

    common_data = {
        "job_description": st.session_state.get("job_desc", ""),
        "provider": provider,
        "model_name": model_name or "",
        "route_mode": route_mode,
    }

    tab1, tab2 = st.tabs(["📄 Standard ATS Resume", "🇩🇪 German Lebenslauf / PDF Options"])

    with tab1:
        st.caption(
            "Layout is auto-selected from the job description's language "
            "(English JD → compact single-page English layout)."
        )

        if st.button(
            "🪄 Build Tailored Resume (.docx)",
            width="stretch",
            type="primary",
            key="build_docx_btn",
            help="Generates a Word document. Usually 15–30 seconds.",
        ):
            _run_build(
                kind="docx",
                label="DOCX resume",
                endpoint=f"{api_base}/api/v1/resume/generate-full",
                payload={**common_data, "layout_style": "auto"},
                filename="Optimized_Tailored_Resume.docx",
                mime=("application/vnd.openxmlformats-officedocument" ".wordprocessingml.document"),
            )

        _render_download("docx", "DOCX resume")

    with tab2:
        template_options = list(TEMPLATE_LABELS.keys())
        selected_layout = st.selectbox(
            "Layout Style",
            template_options,
            index=0,
            format_func=lambda x: TEMPLATE_LABELS[x],
            key="cv_layout_select",
        )

        st.caption("⏱️ PDF builds typically take 20–45 seconds (LLM + LaTeX).")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📄 Build PDF",
                width="stretch",
                type="primary",
                key="build_pdf_btn",
                help="Runs LLM tailoring, then compiles LaTeX to PDF.",
            ):
                _run_build(
                    kind="pdf",
                    label="PDF",
                    endpoint=f"{api_base}/api/v1/resume/generate-german-cv",
                    payload={**common_data, "layout_style": selected_layout},
                    filename=f"CV_{selected_layout}.pdf",
                    mime="application/pdf",
                )

        with col2:
            if st.button(
                "🛠️ Build LaTeX Source (.tex)",
                width="stretch",
                key="build_tex_btn",
                help="Raw LaTeX source — faster than a PDF build.",
            ):
                _run_build(
                    kind="tex",
                    label="LaTeX source",
                    endpoint=f"{api_base}/api/v1/resume/generate-tex-cv",
                    payload={**common_data, "layout_style": selected_layout},
                    filename=f"CV_{selected_layout}.tex",
                    mime="text/plain",
                )

        _render_download("pdf", "PDF")
        _render_download("tex", "LaTeX source")
