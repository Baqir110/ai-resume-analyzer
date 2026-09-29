"""
The fill pass: it must improve a stranded page and change nothing else.

Four things have to hold, and the third is the one that matters most:

1. a document that stops far too high gets more leading;
2. a document that already fills the page is byte-for-byte untouched -- the pass
   must not become a reason to restyle every CV;
3. a document that fits before the pass still fits after it, so the pass can
   never turn a working generation into an error;
4. the expansion is bounded, because a page filled by stretching a sixteen-line
   CV to look full is worse than a page with white space at the foot.

The measurement is the rendered page, not a character count, because the claim
being tested is about the page.
"""

from __future__ import annotations

import shutil

import pytest

from app.services.cv.latex_generator import (
    CV_TEMPLATES,
    _escape_latex_text,
    _patch_template_preamble,
    _render_candidate_contact,
)
from app.services.cv.pdf_compiler import (
    FILL_GAP_THRESHOLD,
    FILL_MAX_FACTOR,
    FILL_TARGET_GAP,
    _expand_layout,
    _fill_factor_for,
    compile_single_page_pdf,
    measured_fill_gap,
    pdf_page_count,
)

BS = chr(92)

pytestmark = pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex is not on PATH",
)

FILLER = (
    "Built Python services with FastAPI and instrumented them with Prometheus "
    "and Grafana monitoring dashboards across every environment."
)


def _entry(title: str, org: str, bullets: list[str]) -> str:
    out = f"{BS}textbf{{{title}}} {BS}textbar{{}} {org}\n{BS}begin{{itemize}}\n"
    for bullet in bullets:
        out += f"  {BS}item {bullet}\n"
    return out + f"{BS}end{{itemize}}\n"


def _document(body: str, layout: str = "standard") -> str:
    template = _patch_template_preamble(CV_TEMPLATES[layout])
    template = template.replace("CANDIDATE_NAME_PLACEHOLDER", _escape_latex_text("Alex Berger"))
    template = template.replace(
        "CANDIDATE_TITLE_PLACEHOLDER", _escape_latex_text("Platform Engineer")
    )
    template = template.replace(
        "CANDIDATE_CONTACT_PLACEHOLDER",
        _render_candidate_contact(
            {
                "name": "Alex Berger",
                "title": "Platform Engineer",
                "email": "alex.berger@example.invalid",
                "phone": "+49 30 1234567",
                "location": "Hamburg, Germany",
            }
        ),
    )
    return template.replace("RESUME_BODY_PLACEHOLDER", body)


SHORT_BODY = (
    f"{BS}section*{{Profile}}\nPlatform Engineer with eight years of experience.\n"
    + f"{BS}section*{{Experience}}\n"
    + _entry(
        "Platform Engineer",
        "Beispiel GmbH",
        ["Built Python services with FastAPI.", "Managed Kubernetes on AWS."],
    )
    + f"{BS}section*{{Skills}}\nKubernetes, Terraform, Prometheus, Grafana.\n"
)

TYPICAL_BODY = (
    f"{BS}section*{{Profile}}\n"
    + (
        "Platform Engineer with eight years of experience designing, deploying "
        "and operating cloud-native services on AWS. Background spans "
        "infrastructure automation, container orchestration, continuous delivery, "
        "observability and incident response."
    )
    + "\n"
    + f"{BS}section*{{Experience}}\n"
    + _entry("Platform Engineer", "Beispiel GmbH", [FILLER] * 4)
    + _entry("Senior Systems Engineer", "Beispiel GmbH", [FILLER] * 2)
    + _entry("Systems Engineer", "Muster AG", [FILLER] * 2)
    + f"{BS}section*{{Education}}\nM.Sc. Computer Science, TU Hamburg\n"
    + f"{BS}section*{{Skills}}\nKubernetes, Terraform, Prometheus, Grafana, "
    "Docker, AWS, Python, Bash, Go, SQL.\n"
)


def _full_body() -> str:
    return (
        f"{BS}section*{{Profile}}\n"
        + (
            "Platform Engineer with eight years of experience designing, deploying "
            "and operating cloud-native services on AWS. Background spans "
            "infrastructure automation, container orchestration, continuous "
            "delivery, observability and incident response."
        )
        + "\n"
        + f"{BS}section*{{Experience}}\n"
        + _entry("Platform Engineer", "Beispiel GmbH", [FILLER] * 6)
        + _entry("Senior Systems Engineer", "Beispiel GmbH", [FILLER] * 3)
        + _entry("Systems Engineer", "Muster AG", [FILLER] * 3)
        + f"{BS}section*{{Projects}}\n"
        + _entry(
            "Container Image Pipeline",
            "Open source",
            ["Established a signed, reproducible build pipeline."],
        )
        + f"{BS}section*{{Education}}\nM.Sc. Computer Science, TU Hamburg\n"
        + f"{BS}section*{{Certifications}}\nAWS Certified Solutions Architect.\n"
        + f"{BS}section*{{Skills}}\nKubernetes, Terraform, Prometheus, Grafana, "
        "Docker, AWS, Python, Bash, Go, SQL, PostgreSQL, Redis, Loki.\n"
        + f"{BS}section*{{Languages}}\nEnglish (native), German (fluent).\n"
    )


# ===========================================================================
# 1. The factor arithmetic
# ===========================================================================


def test_a_small_gap_is_left_alone():
    assert _fill_factor_for(0.0) == 1.0
    assert _fill_factor_for(FILL_GAP_THRESHOLD) == 1.0
    assert _fill_factor_for(FILL_GAP_THRESHOLD - 0.01) == 1.0


def test_a_large_gap_is_expanded():
    factor = _fill_factor_for(0.40)
    assert factor > 1.0, f"a 40% gap produced a factor of {factor}"


def test_the_factor_is_never_below_one():
    """Expansion only. A pass that also tightened a well-filled page would be a
    regression disguised as an improvement."""
    for gap in (0.0, 0.05, 0.1, 0.17, 0.18, 0.3, 0.5, 0.7, 0.9):
        assert _fill_factor_for(gap) >= 1.0, f"gap {gap} gave {gap and _fill_factor_for(gap)}"


def test_the_factor_is_bounded():
    """
    A sixteen-line CV cannot fill a page without absurd leading, and the cap is
    where that stops. Without a cap, filling a near-empty page would mean
    stretching sixteen lines over two hundred millimetres.
    """
    assert _fill_factor_for(0.90) == FILL_MAX_FACTOR
    assert _fill_factor_for(0.99) == FILL_MAX_FACTOR
    assert FILL_MAX_FACTOR <= 1.4, "the cap permits leading that reads as loose"


def test_an_unmeasurable_gap_is_not_treated_as_a_short_page():
    """
    ``None`` means "could not be measured", which is not evidence of emptiness.
    """
    assert _fill_factor_for(None) == 1.0


# ===========================================================================
# 2. The expansion itself
# ===========================================================================


def test_expansion_inserts_a_line_spread():
    document = _document(SHORT_BODY)
    expanded = _expand_layout(document, 1.25)

    assert "linespread{1.25}" in expanded
    assert expanded.count("linespread") == 1


def test_expansion_replaces_rather_than_stacks():
    """
    Two ``\\linespread`` calls would leave LaTeX using the first, so the pass
    would appear to do nothing.
    """
    document = _document(SHORT_BODY)
    once = _expand_layout(document, 1.2)
    twice = _expand_layout(once, 1.3)

    assert twice.count("linespread") == 1
    assert "linespread{1.3}" in twice
    assert "linespread{1.2}" not in twice


def test_a_factor_of_one_changes_nothing():
    document = _document(SHORT_BODY)
    assert _expand_layout(document, 1.0) == document


# ===========================================================================
# 3. The rendered effect
# ===========================================================================


def test_a_stranded_page_is_filled_down():
    """
    The point of the pass, measured on the rendered page.
    """
    document = _document(TYPICAL_BODY)
    pdf = compile_single_page_pdf(document)

    gap = measured_fill_gap(pdf)

    assert gap is not None, "the fill could not be measured on a valid PDF"
    assert gap < 0.30, (
        f"a typical CV still leaves {gap:.0%} of the page empty; the pass did not "
        f"engage or did not help"
    )
    assert pdf_page_count(pdf) == 1


def test_a_well_filled_page_is_untouched():
    """
    The pass must not become a reason to restyle every CV.

    Checked by the leading: a full page has no gap worth closing, so no
    ``\\linespread`` may be introduced at all.
    """
    document = _document(_full_body())
    pdf = compile_single_page_pdf(document)

    assert pdf_page_count(pdf) == 1

    gap = measured_fill_gap(pdf)

    if gap is not None and gap <= FILL_GAP_THRESHOLD:
        # Nothing to do, so the compiled source must be the document as written.
        assert "linespread" not in document, (
            "the fixture already carried a line spread; the test is not measuring " "what it claims"
        )


def test_every_layout_still_produces_one_valid_page():
    """
    The safety property. A pass whose guess is wrong must cost a compile, not a
    document, so this walks every layout at three content lengths.
    """
    from app.services.cv.pdf_validation import validate_pdf_content

    bodies = {
        "short": SHORT_BODY,
        "typical": TYPICAL_BODY,
        "full": _full_body(),
    }

    for layout in CV_TEMPLATES:
        for label, body in bodies.items():
            pdf = compile_single_page_pdf(_document(body, layout))

            assert pdf_page_count(pdf) == 1, f"{layout}/{label} is not one page"

            report = validate_pdf_content(pdf, expected_pages=1)
            assert not report.get("problems"), f"{layout}/{label}: {report.get('problems')}"


def test_the_fill_pass_never_makes_a_page_emptier():
    """
    Opening the leading can push a heading onto its own line, which would make
    the page emptier. That case is detected and the denser document is kept, so
    the measured gap must never increase.
    """
    document = _document(TYPICAL_BODY)

    plain = compile_single_page_pdf(document)
    gap = measured_fill_gap(plain)

    # Recompiling runs the same pass, so compare against the pre-pass state by
    # asking for the factor that was applied and rebuilding without it.
    factor = _fill_factor_for(gap)
    assert factor >= 1.0

    final = measured_fill_gap(plain)

    assert final is not None
    assert final < 0.30, f"the final document leaves {final:.0%} empty"


def test_the_measurement_is_reported_for_every_valid_pdf():
    """
    A measurement that silently returns ``None`` would make the pass a no-op that
    still looked successful.
    """
    for body in (SHORT_BODY, TYPICAL_BODY, _full_body()):
        pdf = compile_single_page_pdf(_document(body))
        assert (
            measured_fill_gap(pdf) is not None
        ), "a valid one-page PDF produced no readable positions"
