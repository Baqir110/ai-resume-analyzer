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

New: Shows the final ATS score after PDF generation and validation.
"""

from __future__ import annotations

import base64

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.components import clear_result, get_result, store_result
from app.dashboard.helpers import get_api_base, validate_ats, validate_pdf

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


def _run_final_ats(api_base: str, pdf_bytes: bytes) -> None:
    """
    Score the finished document.

    This is the step the first score could not take. Before generation there is no
    PDF, so the PDF Parsing check is reported as unmeasured and its weight is
    redistributed — the score is honest, but it is not the same measurement.
    Running it against the real bytes is what makes the number a final one, so the
    two results are stored under different keys and never conflated.
    """
    clear_result("final_ats")
    job_desc = workflow.get_job().strip()
    if not job_desc:
        st.warning("Add a job description before scoring the generated document.")
        return

    analysis = workflow.get_analysis() or {}
    expected_text = analysis.get("resume_text") or ""

    with st.spinner("Checking the generated document…"):
        response, meta = validate_ats(
            api_base,
            pdf_bytes,
            job_desc,
            expected_text=expected_text,
            layout=workflow.get_layout(),
        )

    if response is None or not (200 <= response.status_code < 300):
        detail = meta.get("error") if isinstance(meta, dict) else "the request failed"
        st.error(f"Could not score the generated document: {detail}")
        return

    store_result("final_ats", response.json())
    # The pre-generation score is deliberately kept: the two are different
    # measurements, and requirement 33 asks for both to be visible.
    st.rerun()


def _render_final_ats(
    api_base: str,
    pdf_bytes: bytes | None,
    *,
    compact: bool = False,
) -> None:
    """
    Show the before / final comparison and the document's verdict.

    Requirement 31's output: the layout used, the final score, whether the PDF is
    valid, the page count, and how many issues remain — followed by the three
    things the user can actually do about any of it.
    """
    analysis = workflow.get_analysis() or {}
    breakdown = analysis.get("ats_breakdown") or {}
    first_score = breakdown.get("ats_score")
    if first_score is None:
        first_score = analysis.get("ats_match_score")

    final = get_result("final_ats")
    theme.rule()
    theme.section_header("Final check", "🏁")

    if not final:
        st.caption(
            "The first score could not check the document itself, because there was "
            "no document yet. Scoring the generated PDF switches those checks on."
        )
        if pdf_bytes and st.button(
            "🏁 Check the generated CV", type="primary", key="final_ats_run"
        ):
            _run_final_ats(api_base, pdf_bytes)
        return

    final_score = final.get("ats_score")
    document = final.get("document") or {}
    layout = final.get("layout") or {}
    pdf_check = final.get("pdf_parsing") or {}

    # Requirement 33 asks for the whole progression — before, after optimisation,
    # final — to be visible at once. The "after" figure is produced on the
    # optimisation page, so without this it is a number the user has already
    # navigated away from. The three are shown together here, and the "after" is
    # only claimed when the loop actually ran.
    optimised_score = (get_result("improvement_result") or {}).get("final_score")

    def _card(label: str, value, note: str) -> None:
        theme.value_card(
            label,
            f"{value:.0f}/100" if isinstance(value, (int, float)) else "—",
            note,
        )

    columns = st.columns(4 if isinstance(optimised_score, (int, float)) else 3)

    with columns[0]:
        _card("Before", first_score, "your CV as written")
    with columns[1]:
        _card(
            "After optimisation",
            optimised_score,
            "safe changes only" if isinstance(optimised_score, (int, float)) else "not run yet",
        )
    with columns[2]:
        _card("Final", final_score, final.get("band") or "")

    with columns[-1] if len(columns) > 3 else columns[2]:
        if isinstance(final_score, (int, float)) and isinstance(first_score, (int, float)):
            delta = final_score - first_score
            theme.value_card(
                "Total change",
                f"{delta:+.1f}",
                "the document is measured too now" if delta else "no change",
            )
        else:
            theme.value_card("Total change", "—", "")

    if isinstance(optimised_score, (int, float)) and optimised_score == first_score:
        st.caption(
            "The optimisation step made no change. The remaining gap needs skills or "
            "qualifications the CV does not contain, and nothing was invented to close it."
        )

    # -- the document's verdict -----------------------------------------
    v1, v2, v3 = st.columns(3)
    with v1:
        valid = bool(document.get("valid"))
        theme.value_card("PDF", "Valid" if valid else "Needs work", "" if valid else "see below")
    with v2:
        pages = document.get("pages")
        theme.value_card("Pages", str(pages) if pages is not None else "—", "")
    with v3:
        theme.value_card("ATS issues", str(final.get("ats_issues", 0)), "things to look at")

    if layout.get("name"):
        theme.kv_table(
            [
                ("Layout used", layout.get("name")),
                (
                    "ATS safety",
                    f"{layout.get('ats_safety')} ({layout.get('ats_safety_score')}/100)",
                ),
                ("Structure", layout.get("structure")),
                ("Parsing risk", layout.get("parsing_risk")),
            ]
        )
    elif layout.get("requested") and not layout.get("known"):
        st.caption(layout.get("detail") or "That layout is not one the generator has.")

    # -- the readability checks, in plain language ----------------------
    if pdf_check.get("failed_checks"):
        theme.section_header("What the document needs", "🔧")
        for check in pdf_check.get("checks") or []:
            if check.get("passed"):
                continue
            st.markdown(f"- **{check.get('name', '')}** — {check.get('detail', '')}")
    elif pdf_check.get("measured"):
        theme.pills([(theme.PASSED, "every readability check passed on the generated document")])

    for problem in document.get("problems") or []:
        st.error(str(problem).replace("_", " "))

    if final.get("summary"):
        st.caption(final["summary"])

    # -- what to do next -------------------------------------------------
    # "Improve ATS score" leads to a stage further up this same page when inline,
    # so offering it as a navigation button would be a way to scroll backwards.
    # "Use this CV" stays either way -- it is an action, not a destination.
    theme.rule()
    theme.section_header("What next?", "➡️")

    if compact:
        b1, b2 = st.columns(2)
        with b1:
            st.success("This is the document you generated. Download it above.")
        with b2:
            if st.button("🎨 Change layout", key="final_relayout_inline", width="stretch"):
                st.session_state[workflow.KEY_PAGE] = "layout_picker"
                st.rerun()
    else:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🔧 Improve ATS score", key="final_improve", width="stretch"):
                st.session_state[workflow.KEY_PAGE] = "optimization"
                st.rerun()
        with b2:
            if st.button("✅ Use this CV", key="final_accept", width="stretch"):
                st.success("This is the document you generated. Download it below.")
        with b3:
            if st.button("🎨 Change layout", key="final_relayout", width="stretch"):
                st.session_state[workflow.KEY_PAGE] = "layout_picker"
                st.rerun()

    with st.expander("🔧 Technical breakdown", expanded=False):
        rows = []
        for name, data in (final.get("breakdown", {}).get("categories") or {}).items():
            label = data.get("label") or name
            if data.get("not_measured"):
                rows.append(
                    {
                        "Category": label,
                        "Score": "not measured",
                        "Weight": f"{data.get('weight', 0) * 100:.0f}%",
                        "Points": "—",
                        "Lost": "—",
                    }
                )
                continue
            rows.append(
                {
                    "Category": label,
                    "Score": f"{data.get('score', 0):.1f}/100",
                    "Weight": f"{data.get('weight', 0) * 100:.0f}%",
                    "Points": f"{data.get('weighted_points', 0):.1f}",
                    "Lost": f"{data.get('points_lost', 0):.1f}",
                }
            )
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "Every number here comes from a check that ran. Nothing is estimated, "
            "and nothing is adjusted to make the total look better."
        )


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


def render_pdf_preview(*, compact: bool = False) -> None:
    """Render the PDF preview and validation page."""
    workflow.section_header(
        "👁",
        "PDF Preview & Validation",
        "What was generated, and what was actually checked about it.",
        compact=compact,
    )

    api_base = get_api_base()

    _outputs_panel()
    theme.rule()

    pdf_result = get_result("pdf")
    pdf_bytes: bytes | None = None
    if pdf_result:
        payload = pdf_result.get("content")
        if isinstance(payload, (bytes, bytearray)) and payload:
            pdf_bytes = bytes(payload)
            _validation_panel(api_base, pdf_bytes)
        else:
            theme.empty_state("No PDF content", "The stored PDF result has no bytes.", icon="📄")
    elif compact:
        st.caption("📄 No PDF yet — generate one above and it appears here.")
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

    _render_final_ats(api_base, pdf_bytes, compact=compact)

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
    """Drop the generated files and everything derived from them."""
    # "final_ats" has to go with them: it scored *these* bytes, so leaving it
    # beside a newly generated document would show a verdict for a CV the user
    # is no longer looking at.
    for key in ("pdf", "tex", "docx", "pdf_validation", "final_ats"):
        clear_result(key)
