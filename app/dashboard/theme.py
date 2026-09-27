"""
Design system for the dashboard.

The goal is a tool that reads as a product rather than a Streamlit default.
That is mostly a matter of consistency, so it lives in one module rather than
being re-declared in every view: one colour scale, one type scale, one set of
card and status primitives, and one place where the CSS is injected.

Two rules that shaped the choices here.

*Status must be earned.* A component that says "connected" or "passed" takes the
verdict as an argument; it never infers one. A panel that cannot prove a
provider works says NOT TESTED rather than showing a green dot, because a user
who sees green stops checking.

*No secret reaches the browser.* Everything here renders from values the backend
already returns to the dashboard. Nothing reads an environment variable to put
a credential on screen, and the status helpers take a pre-sanitised summary.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: One accent, used for the primary action and active state. Deliberately a
#: single hue: a dashboard with five accent colours has no visual hierarchy, it
#: just has noise.
ACCENT = "#2563eb"

#: Semantic colours. Each is paired with a label in :func:`status_pill` so the
#: meaning never depends on colour alone -- a colour-blind user, a greyscale
#: print, and a screen reader all get the same information.
OK = "#16a34a"
WARN = "#d97706"
BAD = "#dc2626"
INFO = "#0891b2"
MUTED = "#64748b"
NEUTRAL_BG = "#f8fafc"
BORDER = "#e2e8f0"
INK = "#0f172a"

#: Verdict vocabulary. Deliberately explicit, and shared with
#: ``scripts/model_matrix.py`` so a status means the same thing in the UI, in the
#: smoke test and in the README table.
PASSED = "PASSED"
FAILED = "FAILED"
CONFIGURED = "CONFIGURED"
NOT_CONFIGURED = "NOT CONFIGURED"
NOT_INSTALLED = "NOT INSTALLED"
NOT_TESTED = "NOT TESTED"
AVAILABLE = "AVAILABLE"
IMPLEMENTED = "IMPLEMENTED"

_VERDICT_COLOURS = {
    PASSED: OK,
    AVAILABLE: OK,
    IMPLEMENTED: OK,
    CONFIGURED: INFO,
    FAILED: BAD,
    NOT_CONFIGURED: WARN,
    NOT_INSTALLED: MUTED,
    NOT_TESTED: MUTED,
}

_VERDICT_ICONS = {
    PASSED: "✅",
    AVAILABLE: "✅",
    IMPLEMENTED: "✅",
    CONFIGURED: "🔧",
    FAILED: "❌",
    NOT_CONFIGURED: "⚠️",
    NOT_INSTALLED: "📭",
    NOT_TESTED: "❔",
}

_CSS = f"""
<style>
  /* ---- Layout ------------------------------------------------------- */
  .block-container {{
    padding-top: 1.75rem;
    padding-bottom: 4rem;
    max-width: 1500px;
  }}

  /* ---- Type scale --------------------------------------------------- */
  h1 {{ letter-spacing: -0.03em; font-weight: 650; color: {INK}; }}
  h2 {{ letter-spacing: -0.02em; font-weight: 620; color: {INK}; }}
  h3 {{ letter-spacing: -0.015em; font-weight: 600; color: {INK}; }}
  h1 {{ font-size: 1.75rem !important; }}
  h2 {{ font-size: 1.30rem !important; }}
  h3 {{ font-size: 1.08rem !important; }}

  /* ---- Cards -------------------------------------------------------- */
  .nb-card {{
    background: {NEUTRAL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.9rem 1.05rem;
    margin-bottom: 0.55rem;
  }}
  .nb-card-title {{
    font-size: 0.74rem;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 0.35rem;
  }}
  .nb-card-value {{
    font-size: 1.02rem;
    font-weight: 600;
    color: {INK};
    line-height: 1.35;
  }}
  .nb-card-note {{
    font-size: 0.80rem;
    color: {MUTED};
    margin-top: 0.2rem;
  }}

  /* ---- Status pill -------------------------------------------------- */
  .nb-pill {{
    display: inline-block;
    padding: 0.10rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.03em;
    border: 1px solid transparent;
    margin: 0 0.25rem 0.25rem 0;
    white-space: nowrap;
  }}

  /* ---- Step header -------------------------------------------------- */
  .nb-step {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.08rem;
    font-weight: 620;
    color: {INK};
    padding-bottom: 0.35rem;
    border-bottom: 2px solid {BORDER};
    margin-bottom: 0.9rem;
  }}
  .nb-step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    background: {ACCENT};
    color: #fff;
    font-size: 0.82rem;
    font-weight: 700;
    flex: 0 0 auto;
  }}

  /* ---- Definition rows ----------------------------------------------- */
  .nb-kv {{ display: flex; justify-content: space-between; gap: 1rem;
            font-size: 0.86rem; padding: 0.22rem 0; }}
  .nb-kv-k {{ color: {MUTED}; }}
  .nb-kv-v {{ color: {INK}; font-weight: 550; text-align: right;
             word-break: break-word; }}

  /* ---- Sidebar navigation -------------------------------------------- */
  .nb-nav-group {{
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
    margin: 0.9rem 0 0.25rem 0;
  }}
  .nb-nav-group:first-child {{ margin-top: 0; }}

  /* ---- Chips ---------------------------------------------------------- */
  .nb-chip {{
    display: inline-block;
    padding: 0.10rem 0.45rem;
    border-radius: 5px;
    font-size: 0.75rem;
    font-weight: 550;
    margin: 0 0.22rem 0.22rem 0;
    border: 1px solid {BORDER};
    background: #fff;
    color: {INK};
  }}
  .nb-chip-ok {{ border-color: {OK}; color: {OK}; background: #f0fdf4; }}
  .nb-chip-miss {{ border-color: {BAD}; color: {BAD}; background: #fef2f2; }}

  /* ---- Section rule --------------------------------------------------- */
  .nb-rule {{ height: 1px; background: {BORDER}; margin: 1.1rem 0; }}

  /* ---- Reduce Streamlit's default noise -------------------------------- */
  [data-testid="stExpander"] details {{ border-radius: 8px; }}
  [data-testid="stMetricValue"] {{ font-weight: 650; }}
  footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    """Install the stylesheet. Idempotent within a single script run."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def step_header(number: int | str, title: str, subtitle: str = "") -> None:
    """Numbered heading for a step in the workflow."""
    subtitle_html = (
        f'<div class="nb-card-note" style="margin-top:0.15rem">{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f'<div class="nb-step"><span class="nb-step-num">{number}</span>'
        f"<span>{title}</span></div>{subtitle_html}",
        unsafe_allow_html=True,
    )


def section_header(title: str, icon: str = "") -> None:
    """A lighter heading inside a page."""
    label = f"{icon} {title}".strip()
    st.markdown(f"### {label}")


def rule() -> None:
    st.markdown('<div class="nb-rule"></div>', unsafe_allow_html=True)


def status_pill(verdict: str, note: str = "") -> str:
    """
    A colour-coded, labelled status chip.

    The verdict text is always present, so the meaning survives greyscale,
    colour blindness and a screen reader. ``note`` is shown beside it in muted
    text for the specific reason -- "missing CEREBRAS_API_KEY" is actionable in a
    way a coloured dot is not.
    """
    colour = _VERDICT_COLOURS.get(verdict, MUTED)
    icon = _VERDICT_ICONS.get(verdict, "•")
    note_html = (
        f' <span style="color:{MUTED};font-weight:450">{_escape(note)}</span>' if note else ""
    )
    return (
        f'<span class="nb-pill" style="color:{colour};'
        f'border-color:{colour};background:#fff">{icon} {verdict}</span>'
        f"{note_html}"
    )


def pills(items: list[tuple[str, str]]) -> None:
    """Render a row of status pills from ``(verdict, note)`` pairs."""
    st.markdown(
        "".join(status_pill(verdict, note) for verdict, note in items),
        unsafe_allow_html=True,
    )


def value_card(title: str, value: str, note: str = "") -> None:
    """A single labelled value. Used in columns for status readouts."""
    note_html = f'<div class="nb-card-note">{_escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="nb-card"><div class="nb-card-title">{_escape(title)}</div>'
        f'<div class="nb-card-value">{_escape(value)}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def kv_table(rows: list[tuple[str, str]]) -> None:
    """A compact label/value list, for detail that does not deserve a card."""
    rendered = "".join(
        f'<div class="nb-kv"><span class="nb-kv-k">{_escape(str(k))}</span>'
        f'<span class="nb-kv-v">{_escape(str(v))}</span></div>'
        for k, v in rows
        if v not in (None, "")
    )
    if rendered:
        st.markdown(rendered, unsafe_allow_html=True)


def chip_row(values: list[str], kind: str = "plain", limit: int = 40) -> None:
    """
    A wrap of small tags.

    ``kind`` is ``"ok"``, ``"miss"`` or ``"plain"``. Escaped, because these come
    from a model or from a parsed document and either can contain a quote or an
    angle bracket.
    """
    if not values:
        return

    shown = values[:limit]
    suffix = (
        f'<span class="nb-chip" style="color:{MUTED}">+{len(values) - limit} more</span>'
        if len(values) > limit
        else ""
    )
    css = {"ok": "nb-chip-ok", "miss": "nb-chip-miss"}.get(kind, "")
    body = "".join(f'<span class="nb-chip {css}">{_escape(str(v))}</span>' for v in shown)
    st.markdown(body + suffix, unsafe_allow_html=True)


def empty_state(
    title: str,
    detail: str = "",
    action: str = "",
    icon: str = "📭",
) -> None:
    """
    A deliberate empty state.

    Used instead of a blank region so a page that has nothing to show says why,
    rather than looking broken.
    """
    detail_html = (
        f'<div class="nb-card-note" style="margin-top:0.3rem">{_escape(detail)}</div>'
        if detail
        else ""
    )
    action_html = (
        f'<div class="nb-card-note" style="margin-top:0.5rem">' f"👉 {action}</div>"
        if action
        else ""
    )
    st.markdown(
        f'<div class="nb-card"><div class="nb-card-title">{icon} {_escape(title)}'
        f"</div>{detail_html}{action_html}</div>",
        unsafe_allow_html=True,
    )


def readiness_line(ready: bool, checks: list[tuple[str, bool]]) -> None:
    """
    A checklist of preconditions, shown above a disabled action.

    Saying *which* input is missing is the difference between a button that
    cannot be pressed and a page the user can act on.
    """
    if ready:
        st.markdown(status_pill(PASSED, "all inputs ready"), unsafe_allow_html=True)
        return

    missing = [name for name, ok in checks if not ok]
    st.markdown(
        status_pill(NOT_TESTED, "needs " + ", ".join(missing)),
        unsafe_allow_html=True,
    )


def _escape(value: Any) -> str:
    """HTML-escape anything that reaches the markup helpers."""
    import html

    return html.escape(str(value))


__all__ = [
    "ACCENT",
    "AVAILABLE",
    "CONFIGURED",
    "FAILED",
    "IMPLEMENTED",
    "NOT_CONFIGURED",
    "NOT_INSTALLED",
    "NOT_TESTED",
    "PASSED",
    "chip_row",
    "empty_state",
    "inject_css",
    "kv_table",
    "pills",
    "readiness_line",
    "rule",
    "section_header",
    "status_pill",
    "step_header",
    "value_card",
]
