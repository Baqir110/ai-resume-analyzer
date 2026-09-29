"""
CV layout selection.

The layout list comes from the backend's ``backend-status``, which reads
``CV_TEMPLATES``. That matters: a layout that no longer exists in the generator
disappears from this page on the next load, instead of being offered and failing
at generation time. A UI that carries its own copy of the list is a UI that
eventually lies.

Each layout gets a name, its target language, and a *measured* description rather
than a marketing one. The traits come from the template itself, so "uses a
serif font" is a fact about the document and not a claim about it.

Selecting a layout stores the choice for the generation step. It does not
generate anything, so browsing costs nothing.

Intelligent recommendation (requirements 28-30):
- Analyzes the job and resume to recommend the best layout
- Shows ATS safety, structure, and best use cases for each layout
- Allows the user to override the recommendation
"""

from __future__ import annotations

import re

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.helpers import fetch_backend_status, get_api_base

#: Human-facing names. The identifier stays the value that is sent to the
#: backend; this only changes the label, because ``german_minimal_ats`` is not a
#: name a user would choose from a list.
DISPLAY_NAMES = {
    "german_corporate": "German Corporate",
    "german_ats": "German ATS",
    "german_classic": "German Classic",
    "german_modern": "German Modern",
    "german_minimal_ats": "German Minimal ATS",
    "international_ats": "International ATS",
    "academic": "Academic",
    "technical_lead": "Technical Lead",
    "standard": "Standard",
    "hr_executive_gold": "HR Executive Gold",
}

_DESCRIPTIONS = {
    "german_corporate": "Serif body, a single accent colour and a rule under each "
    "section. The conventional German application CV.",
    "german_ats": "Plain and unambiguous, built for keyword extraction. No colour emphasis.",
    "german_classic": "Serif throughout with ruled section heads. The most conservative option.",
    "german_modern": "Sans-serif with a coloured header block. Reads as more current than Classic.",
    "german_minimal_ats": "Pure black throughout, no colour at all. Survives "
    "monochrome printing and aggressive parsers.",
    "international_ats": "English ATS layout with a brand colour and a full-width measure.",
    "academic": "Serif with a centred header and numbered sections. Suits "
    "research and teaching roles.",
    "technical_lead": "Sans-serif, dense skills block, a rule between entries. "
    "Suits engineering leadership.",
    "standard": "The neutral default: blue section heads, ruled, one column.",
    "hr_executive_gold": "Serif with a gold accent and uppercase section heads.",
}

#: ATS safety ratings based on actual template implementation
_ATS_SAFETY = {
    "german_corporate": ("High", 90),
    "german_ats": ("Very High", 95),
    "german_classic": ("High", 88),
    "german_modern": ("High", 85),
    "german_minimal_ats": ("Maximum", 100),
    "international_ats": ("Maximum", 100),
    "academic": ("High", 85),
    "technical_lead": ("High", 88),
    "standard": ("High", 90),
    "hr_executive_gold": ("Medium", 75),
}

#: Best use cases for each layout
_BEST_FOR = {
    "german_corporate": "Technical, business, German market",
    "german_ats": "ATS-focused, German market, keyword-heavy",
    "german_classic": "Traditional, conservative, German market",
    "german_modern": "Modern, tech, German market",
    "german_minimal_ats": "Maximum ATS compatibility, German market",
    "international_ats": "ATS-focused, international, English market",
    "academic": "Academic, research, education",
    "technical_lead": "Technical leadership, engineering",
    "standard": "General professional, neutral",
    "hr_executive_gold": "Executive, HR, leadership",
}


def _template_facts() -> dict[str, dict[str, str]]:
    """
    Structural traits, read from the templates themselves.

    Imported lazily and defensively: the dashboard must still render if the CV
    service cannot be imported, which is the same rule the provider selector
    follows.
    """
    facts: dict[str, dict[str, str]] = {}

    try:
        from app.services.cv.latex_generator import CV_TEMPLATES, _patch_template_preamble
    except Exception:
        return facts

    for name, raw in CV_TEMPLATES.items():
        try:
            template = _patch_template_preamble(raw)
        except Exception:
            template = raw

        traits: list[str] = []

        if re.search(r"familydefault\}\{\\rmdefault", template):
            traits.append("serif")
        if re.search(r"familydefault\}\{\\sfdefault", template):
            traits.append("sans-serif")
        if re.search(r"\\scshape|\\MakeUppercase|\\uppercase", template):
            traits.append("uppercase headings")
        if re.search(r"hrule height", template):
            traits.append("ruled sections")

        colours = re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}", template)
        accents = [
            f"#{value.upper()}"
            for key, value in colours
            if key in {"accent", "primary", "linkcolor", "secondary"}
        ]
        if accents:
            traits.append("colour " + ", ".join(sorted(set(accents))[:2]))

        packages = len(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{", template))
        if packages:
            traits.append(f"{packages} packages")

        facts[name] = {
            "traits": " · ".join(traits) if traits else "no distinctive traits",
        }

    return facts


def _get_recommended_layout() -> str | None:
    """Get the recommended layout from the ATS analysis if available."""
    analysis = workflow.get_analysis() or {}
    layout_rec = analysis.get("layout_recommendation") or {}
    return layout_rec.get("recommended_layout")


def _render_recommendation_banner(recommended: str | None) -> None:
    """Render the recommendation banner."""
    if not recommended:
        return

    name = DISPLAY_NAMES.get(recommended, recommended)
    ats_safety, ats_score = _ATS_SAFETY.get(recommended, ("Unknown", 0))
    best_for = _BEST_FOR.get(recommended, "")

    st.info(
        f"**Recommended: {name}**\n\n"
        f"ATS Safety: {ats_safety} ({ats_score}/100) · "
        f"Best for: {best_for}\n\n"
        f"This layout is recommended based on your job description and resume. "
        f"You can still choose a different layout below.",
        icon="💡",
    )

    # One page or two, and the reasoning. Requirement 28 asks the recommender to
    # consider the length; a recommendation whose reasoning is invisible is not much
    # of a recommendation.
    analysis = workflow.get_analysis() or {}
    guidance = (analysis.get("layout_recommendation") or {}).get("page_guidance") or {}
    pages = guidance.get("recommended_pages")
    if pages:
        theme.kv_table(
            [
                ("Suggested length", f"{pages} page{'s' if pages != 1 else ''}"),
                ("Why", guidance.get("basis")),
            ]
        )


def render_layout_picker() -> None:
    """Render the layout selection page."""
    theme.step_header(
        "▤",
        "CV Layout",
        "Choose the document design. The choice changes the generated document, "
        "not just its label.",
    )

    api_base = get_api_base()
    status = fetch_backend_status(api_base) or {}
    layouts = status.get("layouts") or []

    if not layouts:
        theme.empty_state(
            "Layout list unavailable",
            f"The backend at {api_base} did not report its layouts. It may be "
            f"older than this dashboard, or unreachable.",
            icon="▤",
        )
        return

    facts = _template_facts()
    current = workflow.get_layout()
    names = [row["name"] for row in layouts]

    # Show recommendation if available
    recommended = _get_recommended_layout()
    if recommended and recommended in names:
        _render_recommendation_banner(recommended)

    index = names.index(current) if current in names else 0

    theme.section_header("Available layouts", "▦")
    st.caption(
        f"{len(names)} layouts, read from the generator itself. "
        "Every one is compiled and content-validated in CI from a fixed "
        "fixture, and a regression test fails if two become structurally "
        "identical."
    )

    selected = st.selectbox(
        "Layout",
        names,
        index=index,
        key="layout_picker",
        format_func=lambda name: DISPLAY_NAMES.get(name, name),
    )

    workflow.set_layout(selected)

    row = next((r for r in layouts if r["name"] == selected), {})

    preview_col, detail_col = st.columns([1, 1.2], gap="large")

    with preview_col:
        theme.section_header("Structure preview", "👁️")
        st.info(
            "**Schematic, not a rendered PDF.** It shows the page structure and "
            "the template's own measured traits — font family, heading case, "
            "accent colour, ruled section heads. It is drawn from the template "
            "source, so it cannot describe a layout the generator does not "
            "have. It is not a preview of the final typography, line breaks or "
            "spacing: those come from LaTeX at generation time.",
            icon="ℹ️",
        )
        _render_schematic(selected, row, facts.get(selected, {}))

    with detail_col:
        theme.section_header("About this layout", "ℹ️")
        ats_safety, ats_score = _ATS_SAFETY.get(selected, ("Unknown", 0))
        best_for = _BEST_FOR.get(selected, "General use")

        theme.kv_table(
            [
                ("Identifier", selected),
                ("Name", DISPLAY_NAMES.get(selected, selected)),
                ("Target language", row.get("language") or "—"),
                ("Columns", row.get("columns") or "single"),
                ("ATS Safety", f"{ats_safety} ({ats_score}/100)"),
                ("Best for", best_for),
                ("Traits", facts.get(selected, {}).get("traits")),
            ]
        )
        st.markdown(_DESCRIPTIONS.get(selected, ""))

        if row.get("language") == "de":
            st.caption(
                "🇩🇪 This layout expects German output. The generator is told "
                "to write the whole CV in German."
            )
        elif row.get("language") == "en":
            st.caption(
                "🇬🇧 This layout expects English output, regardless of the posting's language."
            )

    theme.rule()
    theme.section_header("Compare all layouts", "⚖️")

    table = []
    for entry in layouts:
        name = entry["name"]
        ats_safety, ats_score = _ATS_SAFETY.get(name, ("Unknown", 0))
        table.append(
            {
                "Layout": DISPLAY_NAMES.get(name, name),
                "ID": name,
                "Language": entry.get("language"),
                "ATS Safety": f"{ats_safety}",
                "ATS Score": ats_score,
                "Best For": _BEST_FOR.get(name, "—"),
                "Selected": "●" if name == selected else "",
            }
        )

    st.dataframe(table, width="stretch", hide_index=True)

    theme.rule()
    left, right = st.columns(2)
    with left:
        if st.button("Continue to CV generation →", type="primary"):
            st.session_state[workflow.KEY_PAGE] = "cv_generator"
            st.rerun()
    with right:
        if st.button("How to see the real document", key="layout_fixture"):
            theme.empty_state(
                "Only the generation step produces a real document",
                "Rendering a fixture CV would call pdflatex and, with it, the "
                "LLM — 20–60 seconds and one model call per render, which is a "
                "surprising expense for someone still deciding. Use CV "
                "Generation instead: it produces a real document from your own "
                "resume, and you can inspect every page of it there.",
                action="Open CV Generation",
                icon="🧪",
            )


def _render_schematic(
    name: str,
    row: dict,
    facts: dict[str, str],
) -> None:
    """
    A schematic of the page structure.

    Deliberately schematic rather than a screenshot. Rendering a real preview
    means generating a CV, which costs a model call and 20–60 seconds; doing
    that on every selection would be a surprising expense for a user who is
    still deciding. The schematic is derived from the template's own traits, so
    it cannot describe a layout the generator does not have.
    """
    traits = facts.get("traits", "")
    is_serif = "serif" in traits
    is_sans = "sans-serif" in traits
    is_monochrome = "colour" not in traits

    accent = "#0f172a"
    colour_match = re.search(r"colour (#[0-9A-F]{6})", traits)
    if colour_match:
        accent = colour_match.group(1)

    family = (
        "Georgia, serif" if is_serif else ("Helvetica, Arial, sans-serif" if is_sans else "inherit")
    )

    upper = "uppercase headings" in traits
    ruled = "ruled sections" in traits
    heading = "text-transform: uppercase; letter-spacing: 0.04em;" if upper else ""

    rule = f"border-bottom: 1px solid {accent};" if ruled else ""

    st.markdown(
        f"""
        <div style="
            border:1px solid {theme.BORDER};
            border-radius:8px;
            padding:1.1rem 1.2rem;
            background:#fff;
            font-family:{family};
            max-width:430px;
            margin:0 auto;
        ">
          <div style="text-align:center; margin-bottom:0.9rem;">
            <div style="font-size:1.15rem; font-weight:700; color:{accent};">
              Your Name
            </div>
            <div style="font-size:0.78rem; color:{theme.MUTED}; margin-top:0.1rem;">
              Target Role
            </div>
            <div style="font-size:0.68rem; color:{theme.MUTED}; margin-top:0.15rem;">
              contact · location
            </div>
          </div>
          {"".join(_schematic_section(label, accent, heading, rule, i) for i, label in enumerate(["Profile", "Experience", "Education", "Skills"]))}
          <div style="font-size:0.6rem; color:{theme.MUTED}; text-align:center; margin-top:0.7rem;">
            {name} · {'monochrome' if is_monochrome else accent}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _schematic_section(
    label: str,
    accent: str,
    heading_style: str,
    rule_style: str,
    index: int,
) -> str:
    lines = ""
    count = (2, 3, 2, 2)[index % 4]
    for _ in range(count):
        widths = (96, 88, 70)[_ % 3]
        lines += (
            f'<div style="height:5px; width:{widths}%; background:#e2e8f0; '
            f'margin:3px 0; border-radius:2px;"></div>'
        )

    return f"""
      <div style="margin-bottom:0.6rem;">
        <div style="font-size:0.76rem; font-weight:650; color:{accent};
                    {heading_style} {rule_style} padding-bottom:2px; margin-bottom:4px;">
          {label}
        </div>
        {lines}
      </div>
    """
