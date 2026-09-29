"""
The whole CV workflow on one page.

Why this exists
---------------
The six stages — resume, posting, analysis, improvement, generation, preview —
were six pages, so producing a CV meant six sidebar clicks and six "Continue to
the next step" buttons, most of which existed only to change which page was
showing. That is navigation with no work in it, and it made a linear task look
like a menu.

They are now six sections of one page. Every stage's state was already shared
through ``workflow.py``, so nothing needed to change about how the stages
communicate — only how they are laid out. The stage modules are unchanged in what
they do; they take a ``compact`` flag that suppresses their own page title and
any button whose only effect would be to move the reader somewhere else on the
same page.

The stages are still individually reachable. The keys that used to be separate
nav entries still dispatch, so an old bookmark or a link from another page lands
on the right stage — and a stage opened on its own shows its full page treatment,
because the rail above is what tells a reader where they are in the flow.
"""

from __future__ import annotations

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.views.ats_step import render_ats_step
from app.dashboard.views.cv_generator import render_cv_generation_page
from app.dashboard.views.inputs import render_job_input_step, render_resume_step
from app.dashboard.views.optimization import render_optimization_page
from app.dashboard.views.pdf_preview import render_pdf_preview

#: (number, label, readiness key, glyph). The order is the order the work happens
#: in, which is also the order the sections are rendered in.
STAGES: tuple[tuple[str, str, str, str], ...] = (
    ("1", "Resume", "resume", "📎"),
    ("2", "Job description", "job_description", "✍️"),
    ("3", "ATS analysis", "analysis", "🔍"),
    ("4", "Improvement", "analysis", "✏️"),
    ("5", "Generation", "generation", "📄"),
    ("6", "Preview", "generation", "👁"),
)


#: Stage state -> the theme verdict it is shown with. Reuses the existing
#: verdicts rather than inventing a colour: done is the same green as a passing
#: check, pending the same grey as something not yet configured.
_STATE_VERDICT = {
    "done": "passed",
    "current": "not_tested",
    "pending": "not_installed",
}


def stage_state(index: int, ready: dict[str, bool]) -> str:
    """
    One of ``done``, ``current`` or ``pending``, for a stage.

    "Current" is the first stage that cannot run yet, or — once everything is
    ready — the last one, so the rail always carries exactly one marker rather
    than none or several. Public because the overview page summarises the same
    states, and two copies of this rule would eventually disagree.
    """
    done_count = sum(1 for _, _, key, _ in STAGES[:index] if ready.get(key))
    if ready.get(STAGES[index][2], False):
        return "current" if index == len(STAGES) - 1 else "done"
    return "current" if done_count == index else "pending"


def _render_rail() -> None:
    """A one-line progress rail, so a long page still shows where you are."""
    ready = workflow.workflow_readiness()
    columns = st.columns(len(STAGES))

    verdicts = {
        "passed": theme.PASSED,
        "not_tested": theme.NOT_TESTED,
        "not_installed": theme.NOT_INSTALLED,
    }

    for index, (column, (number, label, key, glyph)) in enumerate(zip(columns, STAGES)):
        state = stage_state(index, ready)
        with column:
            theme.pills([(verdicts[_STATE_VERDICT[state]], f"{glyph} {number}. {label}")])


def render_workflow_page() -> None:
    """Render every stage of the CV workflow, in order, on one page."""
    theme.step_header(
        "1–6",
        "CV workflow",
        "Upload, describe, analyse, improve, generate, check — on one page, top "
        "to bottom. Nothing here needs choosing between pages.",
    )

    _render_rail()
    theme.rule()

    render_resume_step(compact=True)
    theme.rule()

    render_job_input_step(compact=True)
    theme.rule()

    render_ats_step(compact=True)
    theme.rule()

    render_optimization_page(compact=True)
    theme.rule()

    render_cv_generation_page(compact=True)
    theme.rule()

    render_pdf_preview(compact=True)

    st.divider()
    st.caption(
        "Layout and model settings live under Setup in the sidebar. Everything "
        "needed to produce a CV is on this page."
    )


__all__ = ["STAGES", "render_workflow_page", "stage_state"]
