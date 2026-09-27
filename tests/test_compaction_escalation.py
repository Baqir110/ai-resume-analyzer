"""
One-page compaction escalates, and stops as soon as the document fits.

The behaviour this pins is a real quality property, not an implementation
detail. Compaction trades legibility for fit: it tightens list spacing, then
narrows the margins, then reduces the body text from 11pt to 10pt. Applying the
most aggressive level to a document that only needed tighter list spacing
delivers a visibly worse CV for no reason.

The previous implementation looped over ``range(2)`` and called
``_compact_layout_level(source, 3)`` on its second iteration, so levels 1 and 2
were unreachable and *every* compacted document received 10pt text with 0.5cm
margins. Measured on a CV that overflowed to two pages, level 1 was already
sufficient.

The assertions read the font size back out of the compiled PDF rather than
inferring it from which branch executed, so a change to the escalation order
that produced the same visual result would not fail -- and one that produced a
worse document would.
"""

from __future__ import annotations

import re
import shutil

import pytest

from app.services.cv.pdf_compiler import (
    _MIN_REMAINING_BUDGET_SECONDS,
    PDFL_COMPILER_TOTAL_TIMEOUT_SECONDS,
    PDFLayoutError,
    _compact_layout_level,
    _compaction_time_remains,
    compile_single_page_pdf,
    pdf_page_count,
)

BS = chr(92)

pytestmark = pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex is not on PATH",
)


# ---------------------------------------------------------------------------
# A minimal document, built the way the generator builds one.
# ---------------------------------------------------------------------------

HEADER = (
    f"Alex Berger\n{BS}Platform Engineer\n"
    f"alex.berger@example.invalid {BS}textbar{{}} +49 30 1234567\n"
)


def _entry(title: str, org: str, bullets: list[str]) -> str:
    out = f"{BS}textbf{{{title}}} {BS}textbar{{}} {org}\n{BS}begin{{itemize}}\n"
    for bullet in bullets:
        out += f"  {BS}item {bullet}\n"
    return out + f"{BS}end{{itemize}}\n"


def _document(bullets_per_role: int, roles: int) -> str:
    """A CV of a controlled length, so the page count is controllable."""
    from app.services.cv.latex_generator import (
        CV_TEMPLATES,
        _escape_latex_text,
        _patch_template_preamble,
        _render_candidate_contact,
    )

    template = _patch_template_preamble(CV_TEMPLATES["standard"])
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

    body = f"{BS}section*{{Profile}}\n"
    body += "Platform Engineer with eight years of experience.\n"
    body += f"{BS}section*{{Experience}}\n"

    filler = (
        "Built Python services with FastAPI and instrumented them with "
        "Prometheus and Grafana dashboards across every environment."
    )
    for role in range(roles):
        body += _entry(
            f"Role {role}",
            "Beispiel GmbH",
            [filler] * bullets_per_role,
        )

    body += f"{BS}section*{{Education}}\nM.Sc. Computer Science, TU Hamburg\n"
    body += f"{BS}section*{{Skills}}\nKubernetes, Terraform, Prometheus.\n"

    return template.replace("RESUME_BODY_PLACEHOLDER", body)


# ---------------------------------------------------------------------------
# The escalation levels themselves
# ---------------------------------------------------------------------------


def test_each_level_is_distinct_and_cumulative():
    """
    Levels must differ, or "escalating" means applying the same thing repeatedly.
    """
    document = _document(2, 2)

    level_1 = _compact_layout_level(document, 1)
    level_2 = _compact_layout_level(document, 2)
    level_3 = _compact_layout_level(document, 3)

    assert level_1 != document, "level 1 changed nothing"
    assert level_2 != level_1, "level 2 is identical to level 1"
    assert level_3 != level_2, "level 3 is identical to level 2"

    # Cumulative: each level keeps the previous level's changes.
    assert level_2.count("linespread{0.94}") == 1
    assert level_3.count("linespread{0.94}") == 1
    assert "10pt,a4paper" in level_3
    assert "10pt,a4paper" not in level_2, "level 2 must not reduce the font size"
    assert "10pt,a4paper" not in level_1


def test_level_one_keeps_the_font_size_and_margins():
    """
    Level 1 is list spacing only.

    This is the level that is supposed to be enough for most overflowing
    documents, so it must not touch anything a reader would notice.
    """
    document = _document(2, 2)
    level_1 = _compact_layout_level(document, 1)

    assert "11pt,a4paper" in level_1
    assert "10pt,a4paper" not in level_1
    assert "linespread" not in level_1
    assert "top=0.5cm" not in level_1


def test_level_two_narrows_margins_but_keeps_the_font_size():
    document = _document(2, 2)
    level_2 = _compact_layout_level(document, 2)

    assert "11pt,a4paper" in level_2
    assert "top=0.5cm" in level_2
    assert "linespread{0.94}" in level_2


def test_level_three_is_the_only_one_that_reduces_the_font_size():
    document = _document(2, 2)
    for level in (0, 1, 2):
        assert "10pt,a4paper" not in _compact_layout_level(document, level)
    assert "10pt,a4paper" in _compact_layout_level(document, 3)


def test_level_zero_is_the_identity():
    document = _document(2, 2)
    assert _compact_layout_level(document, 0) == document


# ---------------------------------------------------------------------------
# The escalation in use
# ---------------------------------------------------------------------------


def _body_font_size(pdf_bytes: bytes) -> float | None:
    """
    Read the dominant body font size back out of the PDF.

    pypdf does not expose font size directly, so the content stream is scanned
    for the text-showing operator that sets it. Returns the most common size,
    which is the body text rather than a heading.
    """
    import io
    from collections import Counter

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    sizes: Counter[float] = Counter()

    for page in reader.pages:
        try:
            data = page.get_contents().get_data()
        except Exception:
            continue
        text = data.decode("latin-1", errors="ignore")
        # "Tf" selects a font at a size: /F12 10 Tf
        for match in re.finditer(r"/[A-Za-z0-9]+\s+([\d.]+)\s+Tf", text):
            try:
                sizes[float(match.group(1))] += 1
            except ValueError:
                continue

    if not sizes:
        return None

    return sizes.most_common(1)[0][0]


def _uncompacted_pages(document: str) -> int:
    """Page count with no compaction at all."""
    import tempfile
    import time
    from pathlib import Path

    from app.services.cv.pdf_compiler import _compile_attempt

    with tempfile.TemporaryDirectory() as tmp:
        try:
            return pdf_page_count(
                _compile_attempt(
                    document,
                    pdflatex=shutil.which("pdflatex"),
                    workdir=Path(tmp).resolve(),
                    deadline=time.monotonic() + 60,
                )
            )
        except PDFLayoutError as exc:
            return exc.pages


def test_a_document_that_fits_is_not_compacted():
    """
    The common case: one pass, and the document is delivered as generated.
    """
    document = _document(2, 2)
    pdf = compile_single_page_pdf(document)

    assert pdf_page_count(pdf) == 1
    size = _body_font_size(pdf)
    if size is not None:
        assert size == pytest.approx(11.0, abs=0.6), (
            f"an uncompacted document came out at {size}pt; compaction ran when "
            f"it was not needed"
        )


def _level_that_fits(document: str) -> int:
    """
    The lowest compaction level that brings ``document`` to one page.

    0 means it already fitted; 99 means nothing did.
    """
    from app.services.cv.pdf_compiler import _compact_layout_level

    if _uncompacted_pages(document) == 1:
        return 0

    for level in (1, 2, 3):
        try:
            if _uncompacted_pages(_compact_layout_level(document, level)) == 1:
                return level
        except Exception:
            continue

    return 99


#: Sizes searched when a test needs a document of a particular "difficulty".
#:
#: How much text fits an A4 page at 11pt depends on the installed font metrics,
#: not on this code, so a fixed bullet count would be a test that passes or
#: fails depending on the machine. Each test searches for the band it needs and
#: fails loudly if the range does not contain one.
_SIZES = (
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (8, 2),
    (10, 2),
    (12, 2),
    (5, 3),
    (7, 3),
    (9, 3),
    (11, 3),
    (13, 3),
    (16, 3),
    (20, 3),
    (24, 3),
    (12, 4),
    (16, 4),
    (20, 4),
    (26, 4),
    (32, 4),
    (40, 4),
    (50, 4),
)


def _search(accept) -> tuple[str, int]:
    """First ``(document, level)`` in the search range that ``accept`` accepts."""
    for bullets, roles in _SIZES:
        document = _document(bullets, roles)
        level = _level_that_fits(document)

        if accept(document, level):
            return document, level

    raise AssertionError(
        "no document in the search range satisfied the condition. The range "
        "needs widening; the assertion should not be relaxed, because a "
        "relaxed assertion here would pass without testing anything."
    )


def test_a_mildly_overflowing_document_keeps_its_font_size():
    """
    The regression this test exists for.

    A CV that overflows to two pages and is brought back to one by tighter list
    spacing, or by narrower margins, should keep 11pt type. The previous code
    looped over ``range(2)`` and called ``_compact_layout_level(source, 3)`` on
    its second iteration, so levels 1 and 2 were unreachable and *every*
    compacted document was delivered at 10pt with 0.5cm margins -- even when
    level 1 was enough.
    """
    document, level = _search(lambda _doc, lvl: 1 <= lvl <= 2)

    assert (
        _uncompacted_pages(document) > 1
    ), "the document fits uncompacted, so nothing is being tested"

    pdf = compile_single_page_pdf(document)
    assert pdf_page_count(pdf) == 1, "compaction did not bring it to one page"

    size = _body_font_size(pdf)
    if size is not None:
        assert size == pytest.approx(11.0, abs=0.6), (
            f"a document needing only level {level} came out at {size}pt. The "
            f"escalation is skipping levels again."
        )


def test_a_document_that_needs_aggressive_compaction_still_fits():
    """
    The escalation must still reach level 3.

    A fix that only ever tried level 1 would satisfy the previous test and break
    this one, so between them they pin both ends of the range.
    """
    document, level = _search(lambda _doc, lvl: lvl == 3)

    assert level == 3, "the search did not find a level-3 document"

    pdf = compile_single_page_pdf(document)
    assert pdf_page_count(pdf) == 1, "a document that needed level 3 was not brought to one page"

    size = _body_font_size(pdf)
    if size is not None:
        assert size == pytest.approx(10.0, abs=0.6), (
            f"a document that needed level 3 came out at {size}pt, so the "
            f"escalation stopped short"
        )


def test_a_document_that_cannot_fit_is_refused_rather_than_truncated():
    """
    The escalation must not turn "too long" into "content silently dropped".
    """
    import tempfile
    import time
    from pathlib import Path

    from app.services.cv.pdf_compiler import _compile_attempt

    document = _document(40, 8)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            _compile_attempt(
                document,
                pdflatex=shutil.which("pdflatex"),
                workdir=Path(tmp).resolve(),
                deadline=time.monotonic() + 60,
            )
        except PDFLayoutError:
            pass  # expected: it does not fit even compacted

    with pytest.raises(PDFLayoutError):
        compile_single_page_pdf(document)


# ---------------------------------------------------------------------------
# The time budget
# ---------------------------------------------------------------------------


def test_escalation_stops_when_the_budget_is_gone():
    """
    A clear layout error must not become a timeout.

    The extra levels could otherwise consume the whole budget and report a
    deadline instead of the page count, which is the actionable fact.
    """
    import time as _time

    assert _compaction_time_remains(_time.monotonic() + 60) is True
    assert _compaction_time_remains(_time.monotonic() - 1) is False

    assert 0 < _MIN_REMAINING_BUDGET_SECONDS < PDFL_COMPILER_TOTAL_TIMEOUT_SECONDS / 2, (
        "the remaining-budget floor must be short enough to leave room for at "
        "least one more pass"
    )
