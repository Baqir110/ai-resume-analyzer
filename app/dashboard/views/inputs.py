"""
Workflow steps 1 and 2: the resume and the job description.

Split out of the single "analyze" page so each input can be checked on its own.
Nothing here re-implements backend logic: the file is stored and its size and
extension reported, and the text is extracted by the API on the request that
needs it. A second extractor in the UI would be a second implementation that
disagrees with the first.

Empty states are explicit. "No resume yet" says what to upload and what formats
work, because a blank drop zone is a worse starting point than a sentence.
"""

from __future__ import annotations

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.helpers import get_api_base

#: Mirrors the backend's own ceiling for a job description, so the counter
#: warns before a 400 rather than after it. Read from the service rather than
#: duplicated, so the two cannot drift.
_JD_LIMIT_NOTE = "the backend accepts up to 200,000 characters"


def render_resume_step(*, compact: bool = False) -> None:
    """Render the resume input step."""
    workflow.section_header(
        "1",
        "Resume",
        "Upload the CV you want to analyse. PDF, DOCX or TXT.",
        compact=compact,
    )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        uploaded = st.file_uploader(
            "📎 Upload resume",
            type=list(workflow.ACCEPTED_RESUME_FORMATS),
            help="Supported formats: PDF, DOCX, TXT. Parsing happens on the "
            "backend, so nothing is extracted in the browser.",
            key="resume_step_uploader",
        )

        if uploaded is not None:
            payload = uploaded.getvalue()
            extension = workflow.upload_extension(uploaded.name)
            workflow.set_upload(
                uploaded.name,
                payload,
                uploaded.type or "application/octet-stream",
            )
            theme.pills([(theme.PASSED, f"{uploaded.name} ready")])
            theme.kv_table(
                [
                    ("Size", f"{len(payload) / 1024:.0f} KB"),
                    ("Format", extension.upper() or "unknown"),
                    ("MIME", uploaded.type or "unknown"),
                ]
            )
        else:
            if compact:
                st.caption("📎 No resume uploaded yet.")
            else:
                theme.empty_state(
                    "No resume uploaded",
                    "The analysis and every generation step need one. Nothing is "
                    "sent anywhere until you upload a file.",
                    action="Drop a PDF, DOCX or TXT above",
                    icon="📎",
                )

    with right:
        # The return is unconditional; only the explanation inside it is
        # standalone-only. Tying the return to `compact` as well let the unpack
        # below run against a missing upload on the single-page workflow.
        if not workflow.has_upload():
            if not compact:
                theme.section_header("What happens next", "➡️")
                st.markdown(
                    "1. The file is stored for this session only.\n\n"
                    "2. The backend extracts the text when a request needs it.\n\n"
                    "3. The extracted text is scored against the job description."
                )
            return

        filename, payload, _mime = workflow.get_upload()

        theme.section_header("Text preview", "👀")
        st.caption(
            "A plain-text resume can be previewed here. PDF and DOCX are read "
            "on the backend, so their text is not shown until the first request "
            "returns it."
        )

        if workflow.upload_extension(filename) == "txt":
            try:
                text = payload.decode("utf-8", errors="replace")
            except Exception:
                text = ""

            st.text_area(
                "Extracted text",
                value=text[:20_000],
                height=340,
                disabled=True,
                key="resume_preview_text",
            )
            st.caption(
                f"{len(text):,} characters · "
                f"{len(text.split()):,} words · "
                f"preview capped at 20,000 characters"
            )
        else:
            theme.empty_state(
                "Preview available after the first request",
                f"{filename.upper()} text is extracted server-side so the "
                f"browser never holds a second copy of your CV. Run the ATS "
                f"analysis to see the extracted text.",
                icon="🔒",
            )

    theme.rule()
    # Only offered when there is somewhere to go. On the single-page workflow the
    # next stage is already below, so a "continue" button would just be a way to
    # scroll, which the scrollbar already does.
    if not compact and workflow.has_upload() and not workflow.has_analysis():
        if st.button("Continue to job description →", type="primary"):
            st.session_state[workflow.KEY_PAGE] = "job_input"
            st.rerun()


def render_job_input_step(*, compact: bool = False) -> None:
    """Render the job description input step."""
    workflow.section_header(
        "2",
        "Job description",
        "Paste the posting, or upload it. The whole text is used — nothing is "
        "summarised before scoring.",
        compact=compact,
    )

    tab_paste, tab_upload = st.tabs(["✍️ Paste text", "📄 Upload a file"])

    with tab_paste:
        text = st.text_area(
            "Job description",
            height=320,
            placeholder=(
                "Paste the complete posting here — responsibilities, required "
                "skills, technologies and qualifications."
            ),
            key="job_input_text",
            value=workflow.get_job(),
            on_change=_on_job_change,
        )

    with tab_upload:
        st.caption(
            "Uploading copies the file text into the box on the left, where you "
            "can check it before it is used. A job description is read as plain "
            "text; a PDF upload is extracted by the backend on request."
        )
        jd_file = st.file_uploader(
            "Upload a posting (optional)",
            type=["txt", "md", "pdf", "docx"],
            key="job_input_file",
        )
        if jd_file is not None:
            payload = jd_file.getvalue()
            if workflow.upload_extension(jd_file.name) == "pdf":
                st.info(
                    "A PDF posting is extracted on the backend. Paste the text "
                    "above if you want to review it first."
                )
            else:
                try:
                    extracted = payload.decode("utf-8", errors="replace")
                except Exception:
                    extracted = ""
                if extracted:
                    st.session_state["job_input_text"] = extracted
                    workflow.set_job(extracted)
                    st.success(f"Loaded {len(extracted):,} characters.")
                    st.rerun()

    # The text_area owns its own key, so the value has to be pushed into the
    # workflow explicitly rather than read back from the widget.
    if text is not None and text != workflow.get_job():
        workflow.set_job(text)

    stats = workflow.job_stats()
    trimmed = stats["chars"] - stats["chars_stripped"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.value_card("Characters", f"{stats['chars']:,}", "")
    with c2:
        theme.value_card("Words", f"{stats['words']:,}", "")
    with c3:
        theme.value_card("Lines", f"{stats['lines']:,}", "")
    with c4:
        theme.value_card(
            "Whitespace",
            f"{trimmed:,}" if trimmed else "0",
            "leading and trailing, ignored",
        )

    if stats["chars"] == 0:
        theme.pills([(theme.NOT_TESTED, "no job description yet")])
    elif stats["chars"] < 200:
        theme.pills(
            [
                (
                    theme.FAILED,
                    f"only {stats['chars']:,} characters — the ATS score needs the "
                    f"full posting to be meaningful",
                )
            ]
        )
    else:
        theme.pills([(theme.PASSED, f"{stats['chars']:,} characters of posting text")])

    st.caption(f"Size limit: {_JD_LIMIT_NOTE}.")

    theme.rule()

    ready = bool(workflow.get_job().strip())
    if not workflow.has_upload():
        theme.pills([(theme.NOT_TESTED, "a resume is required as well")])

    if compact:
        return

    if st.button(
        "Continue to ATS analysis →",
        type="primary",
        disabled=not (ready and workflow.has_upload()),
    ):
        st.session_state[workflow.KEY_PAGE] = "ats_analysis"
        st.rerun()

    if st.button("← Back to resume"):
        st.session_state[workflow.KEY_PAGE] = "resume"
        st.rerun()


def _on_job_change() -> None:
    """Keep the workflow copy in step with the widget."""
    workflow.set_job(st.session_state.get("job_input_text", ""))


def get_api_base_or_default() -> str:
    """Backend URL, used by the pages that talk to the API."""
    return get_api_base()
