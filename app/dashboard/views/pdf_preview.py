"""
PDF preview and validation.

Shows a generated PDF together with what was actually verified about it.

The ordering matters and is deliberate. The generation path already validates a
finished PDF before returning it -- it reads the bytes back, extracts the text
and compares it against the source -- so a ``200`` with a PDF body means the
document passed. This page does not re-derive that verdict; it asks the backend
to re-check the same file and displays the report, which makes the checks
visible to the user instead of invisible.

A rejected document is never presented as a success. If validation reports a
problem, the download is still offered -- the user may want to look at what went
wrong -- but it is labelled as rejected and the problems are listed above it.
"""

from __future__ import annotations

import base64

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.components import clear_result, get_result, store_result
from app.dashboard.helpers import get_api_base, validate_pdf

#: Result keys this page understands, in the order they are offered.
_OUTPUTS = (
    ("pdf", "📄 PDF", "application/pdf"),
    ("tex", "📝 LaTeX", "text/plain"),
    ("docx", "📘 DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
)


def _validation_panel(api_base: str, pdf_bytes: bytes) -> dict | None:
    """
    Validate a PDF and render the report.

    Returns the report so the caller can gate the success message on it rather
    than on the HTTP status of the request that produced the file.
    """
    theme.section_header("PDF validation", "🔍")
    st.caption(
        "The generated file is sent back to the backend and checked with the "
        "same code that validated it during generation. Nothing here is "
        "re-implemented in the browser."
    )

    if st.button("🔍 Validate this PDF", type="primary", key="pdf_validate_btn"):
        with st.spinner("Validating…"):
            response, meta = validate_pdf(api_base, pdf_bytes, filename="generated-cv.pdf")
            if response is not None and 200 <= response.status_code < 300:
                report = response.json()
                store_result("pdf_validation", report)
                st.rerun()
            else:
                detail = (
                    meta.get("error") if isinstance(meta, dict) else "validation request failed"
                )
                st.error(f"Could not validate: {detail}")

    report = get_result("pdf_validation")
    if not report:
        theme.empty_state(
            "Not validated in this session",
            "Validation also runs automatically during generation. Run it again "
            "here to see the individual checks and their results.",
            icon="🔍",
        )
        return None

    _render_report(report)
    return report


def _render_report(report: dict) -> None:
    """Render a validation report. Absent fields are absent, not zero."""
    problems = report.get("problems") or []
    valid = bool(report.get("valid")) and not problems

    if valid:
        theme.pills([(theme.PASSED, report.get("detail") or "all checks passed")])
    else:
        theme.pills([(theme.FAILED, report.get("detail") or "validation did not pass")])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        theme.value_card(
            "Exists and parses",
            "YES" if report.get("status") in {"ok", "rejected"} else "NO",
            report.get("status") or "",
        )
    with c2:
        pages = report.get("pages")
        theme.value_card(
            "Pages",
            str(pages) if pages is not None else "—",
            "one page is required by the generator",
        )
    with c3:
        chars = report.get("text_chars")
        theme.value_card(
            "Text extracted",
            f"{chars:,}" if isinstance(chars, int) else "—",
            "characters",
        )
    with c4:
        retention = report.get("content_retention")
        theme.value_card(
            "Content retained",
            f"{retention:.0%}" if isinstance(retention, (int, float)) else "—",
            "of the words in the source",
        )

    sections = report.get("sections_found") or []
    missing = report.get("sections_missing") or []
    theme.section_header("Sections found", "🧩")
    if sections:
        theme.chip_row([str(s) for s in sections], kind="ok")
    else:
        st.caption("The backend detected no section headings in the text.")

    if missing:
        theme.pills([(theme.FAILED, "missing: " + ", ".join(missing))])

    artifacts = report.get("latex_artifacts") or []
    theme.section_header("Typesetting", "🔤")
    if artifacts:
        theme.pills(
            [
                (
                    theme.FAILED,
                    "raw LaTeX reached the page: " + ", ".join(artifacts[:6]),
                )
            ]
        )
    else:
        theme.pills([(theme.PASSED, "no raw LaTeX in the rendered text")])

    replacement = report.get("replacement_chars") or 0
    if replacement:
        theme.pills(
            [
                (
                    theme.NOT_CONFIGURED,
                    f"{replacement} character(s) a font could not render",
                )
            ]
        )

    if problems:
        theme.section_header("Problems", "⚠️")
        for problem in problems:
            st.error(problem.replace("_", " "))


def _pdf_viewer(pdf_bytes: bytes) -> None:
    """
    Embed the PDF, falling back to a download when the browser cannot.

    An embedded viewer is the closest thing to seeing the document, and it needs
    no renderer on the server. It is not visual validation: a browser renders
    with its own PDF engine, which is not pdftoppm and not a screenshot.
    """
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" '
        f'width="100%" height="820" style="border:1px solid {theme.BORDER}; '
        f'border-radius:8px;" type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Rendered by your browser. This is not a substitute for the rendered "
        "page checks run by the layout regression test."
    )


def _outputs_panel() -> None:
    """Offer whatever was generated, labelled with its validation state."""
    theme.section_header("Generated files", "📦")

    available = [(key, label, mime) for key, label, mime in _OUTPUTS if get_result(key)]

    if not available:
        theme.empty_state(
            "Nothing generated yet",
            "Run the CV generation step to produce a document.",
            icon="📦",
        )
        return

    labels = [f"{label}  ·  {key}" for key, label, _ in available]
    chosen_index = 0

    if len(available) > 1:
        chosen_index = st.radio(
            "Format",
            range(len(available)),
            format_func=lambda i: labels[i],
            key="preview_format",
            horizontal=True,
        )

    key, label, mime = available[chosen_index]
    result = get_result(key) or {}
    payload = result.get("content")

    if not isinstance(payload, (bytes, bytearray)) or not payload:
        st.caption("This result has no downloadable content.")
        return

    filename = result.get("filename") or f"cv.{key}"
    size_kb = len(payload) / 1024
    elapsed = result.get("elapsed")

    theme.kv_table(
        [
            ("Format", label),
            ("Filename", filename),
            ("Size", f"{size_kb:,.0f} KB"),
            ("Generation time", f"{elapsed:.1f}s" if elapsed else None),
        ]
    )

    is_pdf = key == "pdf"
    report = get_result("pdf_validation") if is_pdf else None
    rejected = bool(report) and not report.get("valid")

    if rejected:
        theme.pills([(theme.FAILED, "this document did not pass validation — see above")])

    if is_pdf and not rejected:
        _pdf_viewer(bytes(payload))

    st.download_button(
        "⬇️ Download" + (" (rejected document)" if rejected else ""),
        data=payload,
        file_name=filename,
        mime=mime,
        width="stretch",
        type="secondary" if rejected else "primary",
        key="preview_download",
    )


def render_pdf_preview() -> None:
    """Render the PDF preview and validation page."""
    theme.step_header(
        "👁",
        "PDF Preview & Validation",
        "What was generated, and what was actually checked about it.",
    )

    api_base = get_api_base()

    _outputs_panel()
    theme.rule()

    pdf_result = get_result("pdf")
    if pdf_result:
        payload = pdf_result.get("content")
        if isinstance(payload, (bytes, bytearray)) and payload:
            _validation_panel(api_base, bytes(payload))
        else:
            theme.empty_state("No PDF content", "The stored PDF result has no bytes.", icon="📄")
    else:
        theme.empty_state(
            "No PDF yet",
            "Generate a CV and its PDF will appear here, with the validation "
            "report for the same file.",
            action="Open CV Generation",
            icon="📄",
        )
        if st.button("← Go to CV generation"):
            st.session_state[workflow.KEY_PAGE] = "cv_generator"
            st.rerun()

    theme.rule()
    theme.section_header("Performance", "⏱️")
    _performance_panel(api_base)


def _performance_panel(api_base: str) -> None:
    """
    What the last generation cost, from the backend's own log.

    Read from ``pipeline-metrics``, which reports values recorded during the
    request. No prompt or response content is shown, and none is stored there.
    """
    from app.dashboard.helpers import fetch_pipeline_metrics

    metrics = fetch_pipeline_metrics(api_base)
    summary = (metrics or {}).get("summary") or {}

    if not summary:
        theme.empty_state("No metrics recorded yet", icon="⏱️")
        return

    rows = [
        ("LLM calls", str(summary.get("llm_calls", 0))),
        ("Retries", str(summary.get("retries", 0))),
        ("LLM time", f"{(summary.get('llm_duration_ms') or 0) / 1000:.1f}s"),
        ("Prompt tokens", f"{summary.get('prompt_tokens', 0):,}"),
        ("Completion tokens", f"{summary.get('completion_tokens', 0):,}"),
    ]

    pdf = summary.get("pdf") or {}
    if pdf.get("duration_ms") is not None:
        rows.append(("PDF compilation", f"{pdf['duration_ms'] / 1000:.1f}s"))
    if pdf.get("bytes"):
        rows.append(("PDF size", f"{pdf['bytes']:,} bytes"))

    theme.kv_table(rows)

    requests = (metrics or {}).get("requests") or []
    if requests:
        st.caption(f"{len(requests)} recent request(s) in the pipeline log.")


def clear_pdf_state() -> None:
    """Drop the generated files and the validation report."""
    for key in ("pdf", "tex", "docx", "pdf_validation"):
        clear_result(key)
