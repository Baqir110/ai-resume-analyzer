"""
Formatting and parsing checks for a CV, in both its source text and its PDF.

Why this exists
---------------
Requirement 24 lists what an ATS actually has to cope with, and most of it is not
covered by "does the text contain the right words". A CV can have a perfect
keyword score and still be unreadable to a parser: text trapped in a two-column
sidebar is extracted in the wrong order, a heading set in a dingbat font comes
back as mojibake, a text box overlapping a heading merges two lines into one, and
a phone number clipped at the page margin comes back truncated. Each of those
silently loses content that the score otherwise claims is present.

So this module answers two questions separately:

* :func:`analyze_text_formatting` — is the *source* CV conventionally laid out?
  Consistent dates, consistent job titles, real bullet points, a heading
  hierarchy, and no tables or graphics where a parser will not follow them.
* :func:`analyze_pdf_layout` — does the *rendered* document survive extraction?
  Readable text, correct reading order, locatable sections, no duplication, no
  clipped or overlapping runs, nothing invisible, nothing in a second column,
  and the individual facts (dates, employers, roles, skills, contact) still
  parseable.

A note on false positives
-------------------------
Every check here can be wrong in the same direction: flagging a good CV. That is
the more expensive error, because the candidate is told their CV is broken when
it is not, and the correct response — regenerate — produces the same document. So
the thresholds are deliberately loose, the checks report *what was seen* rather
than a bare pass/fail, and nothing here raises. A finding is a prompt to look, not
a verdict.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from collections import Counter
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

#: A CV page carries well over 500 characters. 200 leaves room for a sparse but
#: genuine layout without accepting an effectively blank page.
MIN_PDF_TEXT_CHARS = 200

#: Fraction of the candidate's content words that must survive typesetting.
MIN_CONTENT_RETENTION = 0.55

#: How many detectable sections make a document locatable.
MIN_SECTIONS_EXPECTED = 2

#: Two text runs whose baselines are within this many points are on the same
#: visual line, and are candidates for an overlap.
SAME_LINE_TOLERANCE = 2.5

#: Horizontal slack, in points, when testing whether two runs on the same line
#: overlap. Absorbs glyph bearing so kerned text is not reported as collision.
OVERLAP_SLACK = 1.0

#: A run reaching within this many points of the page edge may be clipped.
PAGE_EDGE_SLACK = 6.0

#: Columns narrower than this are indentation, not a second column.
MIN_COLUMN_WIDTH = 120

#: Words too short to be evidence of anything when checking what survived.
MIN_TOKEN_CHARS = 4

#: Vocabulary present in every CV by definition; its presence proves nothing.
_GENERIC_TOKENS = frozenset(
    {
        "and",
        "the",
        "with",
        "for",
        "from",
        "that",
        "this",
        "your",
        "you",
        "our",
        "have",
        "has",
        "was",
        "were",
        "will",
        "are",
        "not",
        "all",
        "can",
        "may",
        "und",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "ein",
        "eine",
        "mit",
        "von",
        "oder",
        "als",
        "im",
        "in",
        "zu",
        "zur",
        "auf",
        "für",
        "sie",
        "wir",
        "ist",
        "sind",
        "werden",
        "durch",
        "ohne",
        "seit",
        "bis",
        "auch",
        "des",
        "dem",
        "einen",
        "beim",
        "gegen",
        "nicht",
        "oder",
        "oder",
        "experience",
        "education",
        "skills",
        "languages",
        "profile",
        "summary",
        "projects",
        "certifications",
        "contact",
        "work",
        "professional",
        "berufserfahrung",
        "ausbildung",
        "fähigkeiten",
        "kenntnisse",
        "sprachen",
        "profil",
        "zusammenfassung",
        "kontakt",
        "projekte",
        "zertifikate",
        "beruf",
        "qualification",
        "competencies",
        "employment",
        "academic",
        "page",
        "seite",
        "lebenslauf",
        "resume",
        "curriculum",
        "cv",
    }
)

#: Section heading -> indicators, in either supported output language. A CV with
#: no Projects section is not asked for one; these detect what is *there*.
SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "contact": (r"\bcontact\b", r"\bkontakt\b", r"\bemail\b", r"\be-mail\b"),
    "summary": (
        r"\bsummary\b",
        r"\bprofile\b",
        r"\babout\b",
        r"\bobjective\b",
        r"\bzusammenfassung\b",
        r"\bprofil\b",
        r"\bkurzprofil\b",
    ),
    "experience": (
        r"\bexperience\b",
        r"\bemployment\b",
        r"\bwork history\b",
        r"\bcareer\b",
        r"\bberufserfahrung\b",
        r"\barbeitserfahrung\b",
        r"\btätigkeit\b",
        r"\banstellung\b",
    ),
    "education": (
        r"\beducation\b",
        r"\bacademic\b",
        r"\bdegree\b",
        r"\bausbildung\b",
        r"\bstudium\b",
        r"\bbildung\b",
        r"\babschluss\b",
    ),
    "skills": (
        r"\bskills\b",
        r"\btechnical skills\b",
        r"\bcompetenc",
        r"\btools\b",
        r"\btechnolog",
        r"\bfähigkeiten\b",
        r"\bfertigkeiten\b",
        r"\bkenntnisse\b",
        r"\bkompetenzen\b",
    ),
    "certifications": (
        r"\bcertificat",
        r"\blicen[cs]e",
        r"\bzertifikat",
        r"\bzertifizierung",
        r"\bnachweise\b",
    ),
    "projects": (r"\bprojects\b", r"\bportfolio\b", r"\bprojekt"),
    "languages": (r"\blanguages\b", r"\blanguage skills\b", r"\bsprachen\b"),
}

_SECTION_RES: dict[str, re.Pattern[str]] = {
    name: re.compile("|".join(patterns), re.IGNORECASE)
    for name, patterns in SECTION_PATTERNS.items()
}

#: Raw LaTeX surviving into the rendered text. These are the operators that mean
#: a template or escaping step failed, rather than a document that legitimately
#: contains a brace.
LATEX_ARTIFACT_RE = re.compile(
    r"\\(?:begin|end|item|section|subsection|subsubsection|textbf|textit|emph|"
    r"texttt|textsc|underline|newline|linebreak|hspace|vspace|href|url|color|"
    r"colorbox|fcolorbox|parbox|minipage|noindent|centering|raggedright|"
    r"raggedleft|tabular|tabularx|longtable|includegraphics|multirow|"
    r"newcommand|renewcommand|newenvironment|documentclass|usepackage|"
    r"setlength|definecolor|pagestyle|thispagestyle)\b"
    r"|\{\{|\}\}|\\\\(?:\s|$)"
)

#: Unicode replacement characters, which is what an unmapped font produces.
REPLACEMENT_RE = re.compile(r"[�]")

#: Bullets a CV legitimately uses. Anything outside this set that sits at the
#: start of a line is a decorative glyph a parser may not recognise.
_KNOWN_BULLETS = frozenset(
    {"•", "‣", "⁃", "·", "◦", "▪", "▫", "●", "○", "■", "□", "–", "—", "-", "*", "→", "»"}
)

#: Private Use Area and dingbat blocks. A letter rendered from one of these comes
#: back as mojibake, which is a font that maps a character to a decorative glyph.
_SUSPECT_GLYPH_RE = re.compile(r"[--\U0001f000-\U0001faff]")


# ---------------------------------------------------------------------------
# Result helper
# ---------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """One check outcome. ``detail`` always says what was observed."""
    return {"name": name, "passed": bool(passed), "detail": detail}


def _normalise(text: str) -> str:
    """
    Strip accents and case so ``Müller`` and ``Muller`` compare equal.

    Without this a correctly rendered umlaut reads as a *missing* character and a
    good CV is reported as damaged.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.casefold()


def checkable_tokens(text: str) -> set[str]:
    """Content words worth checking for survival into the PDF."""
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+#./-]*", _normalise(text))
        if len(token) >= MIN_TOKEN_CHARS and token not in _GENERIC_TOKENS
    }


def detect_sections(text: str) -> list[str]:
    """Names of the CV sections present in the text, in document order."""
    return [name for name, pattern in _SECTION_RES.items() if pattern.search(text or "")]


def find_latex_artifacts(text: str) -> list[str]:
    """
    Raw LaTeX that reached the rendered page.

    Returned as the matched operators only, never the surrounding text, so a
    report can name the problem without reproducing the CV.
    """
    return sorted(set(LATEX_ARTIFACT_RE.findall(text or "")))


# ---------------------------------------------------------------------------
# Source-text formatting
# ---------------------------------------------------------------------------

#: Date shapes a CV uses, ordered most-specific first. The *shape* is what has
#: to be consistent, not the value: "Jan 2020", "01/2020" and "2020-01" all
#: parse, but a CV mixing all three reads as sloppy and can confuse a parser
#: matching on separators.
_DATE_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("YYYY-MM", re.compile(r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])\b")),
    ("MM/YYYY", re.compile(r"\b(?:0?[1-9]|1[0-2])/(?:19|20)?\d{2}\b")),
    (
        "Month YYYY",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+(?:19|20)\d{2}\b",
            re.IGNORECASE,
        ),
    ),
    ("YYYY", re.compile(r"\b(?:19|20)\d{2}\b")),
)

#: A line that is mostly punctuation with letter runs between them is a table row.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|.*$|^\s*\S+\t\S+")

#: Markers of a document that hides content where a parser cannot follow it.
_GRAPHIC_MARKERS = (
    "\U0001f4bb",  # laptop emoji, often used as a section icon
    "\U0001f5c2",  # disk
    "\U0001f4f7",  # page
    "\U0001f517",  # link
    "\U0001f50d",  # magnifier
)


def analyze_text_formatting(text: str) -> dict[str, Any]:
    """
    Is the source CV laid out in a way a parser can follow?

    Only checks that are decidable from plain text are attempted. Anything that
    needs a renderer — overlap, clipping, reading order — is left to
    :func:`analyze_pdf_layout`, because guessing from text would be guessing.
    """
    text = text or ""
    lines = [line for line in text.splitlines() if line.strip()]
    checks: list[dict[str, Any]] = []

    # -- dates ------------------------------------------------------------
    # Counted per shape, most specific first. "2015-03" is counted by the
    # YYYY-MM shape and also by the bare-year shape, so the specific shapes are
    # measured by how much of the text they explain and the fallback year count is
    # whatever the specific ones did not claim. That is what makes "one CV uses
    # Jan 2015 and another line says 2016" readable as inconsistent without
    # every year counting as a second format.
    year_only = _DATE_SHAPES[-1][1]
    shaped = 0
    for label, pattern in _DATE_SHAPES[:-1]:
        shaped += len(pattern.findall(text))
    total_years = len(year_only.findall(text))
    specific = {label: len(pattern.findall(text)) for label, pattern in _DATE_SHAPES[:-1]}

    if total_years == 0:
        checks.append(_check("dated entries", False, "no dates found at all"))
    else:
        bare = max(0, total_years - shaped)
        counts = {**specific, "YYYY": bare}
        # The dominant shape is the one that explains the most dates.
        dominant, best = max(counts.items(), key=lambda item: item[1])
        # Any other shape occurring at all is a second convention. A single
        # "03/2020" among five "Jan 20xx" is a real inconsistency: a reader
        # notices it, and a parser matching on separators may not.
        runners = [name for name, count in counts.items() if count >= 1 and name != dominant]
        if best == 0:
            checks.append(_check("dated entries", False, "no dates found at all"))
        elif not runners:
            checks.append(
                _check(
                    "consistent dates",
                    True,
                    f"{best} date(s) all in one format: {dominant}",
                )
            )
        else:
            checks.append(
                _check(
                    "consistent dates",
                    False,
                    f"dates use more than one format — {dominant} ({best}) and "
                    f"{', '.join(runners)} — so pick one and apply it throughout",
                )
            )

    # -- job titles -------------------------------------------------------
    # Compared on *shape* rather than content: a CV that writes "Sr. Engineer" in
    # one place and "Senior Engineer" in another reads inconsistently, and a
    # parser matching the posting's exact title string can miss the abbreviation.
    title_tokens = re.findall(
        r"\b(senior|sr\.?|junior|jr\.?|lead|principal|staff|head of|chief|"
        r"director|manager|engineer|developer|analyst|consultant|specialist|"
        r"technician|administrator|architect|scientist|designer)\b",
        text,
        re.IGNORECASE,
    )
    if title_tokens:
        styles = {_title_style(token) for token in title_tokens}
        if len(styles) == 1:
            checks.append(
                _check("consistent job titles", True, f"titles use one style: {styles.pop()}")
            )
        else:
            checks.append(
                _check(
                    "consistent job titles",
                    False,
                    f"titles mix {len(styles)} styles "
                    f"({', '.join(sorted(styles))}) — pick one, e.g. 'Senior' not 'Sr.'",
                )
            )
    else:
        checks.append(_check("consistent job titles", True, "no job titles found to compare"))

    # -- company names ----------------------------------------------------
    # A company appearing under several spellings breaks the employer trail a
    # parser builds; "Acme Corp", "Acme Corporation" and "ACME" are one employer.
    company_lines = re.findall(
        r"^\s*[\w&.,'’\- ]{2,40}?\s(?:Inc|Ltd|LLC|GmbH|AG|S\.A\.|B\.V\.|PLC|"
        r"Corp|Corporation|Company|Group|KG|OHG)\b[.,]?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if company_lines:
        bases = {_company_base(name) for name in company_lines}
        if len(bases) == len(company_lines):
            checks.append(_check("consistent company names", True, "each employer named one way"))
        else:
            checks.append(
                _check(
                    "consistent company names",
                    False,
                    f"{len(bases)} distinct employers across {len(company_lines)} "
                    "mentions — at least one employer is named more than one way",
                )
            )
    else:
        checks.append(
            _check("consistent company names", True, "no legal-form company names to compare")
        )

    # -- bullets ----------------------------------------------------------
    bullet_lines = [line for line in lines if re.match(r"^\s*\S", line)]
    bulleted = [line for line in bullet_lines if _leading_bullet(line)]
    if not lines:
        checks.append(_check("standard bullet points", False, "no lines to inspect"))
    elif len(bulleted) < max(2, len(lines) // 8):
        checks.append(
            _check(
                "standard bullet points",
                False,
                f"only {len(bulleted)} of {len(lines)} lines start with a bullet; "
                "ATS readers expect accomplishments as a list",
            )
        )
    else:
        glyphs = Counter(_leading_bullet(line) for line in bulleted)
        most_common, count = glyphs.most_common(1)[0]
        if len(glyphs) == 1:
            checks.append(
                _check(
                    "standard bullet points", True, f"one bullet style throughout: {most_common!r}"
                )
            )
        elif count / len(bulleted) >= 0.8:
            checks.append(
                _check(
                    "standard bullet points",
                    True,
                    f"{count} of {len(bulleted)} bullets use {most_common!r}",
                )
            )
        else:
            checks.append(
                _check(
                    "standard bullet points",
                    False,
                    f"bullets use {len(glyphs)} different styles: "
                    f"{', '.join(sorted(repr(g) for g in glyphs))}",
                )
            )

    # -- heading hierarchy ------------------------------------------------
    sections = detect_sections(text)
    if len(sections) >= MIN_SECTIONS_EXPECTED:
        checks.append(_check("heading hierarchy", True, f"{len(sections)} standard headings found"))
    else:
        checks.append(
            _check(
                "heading hierarchy",
                False,
                f"only {len(sections)} standard heading(s) recognised; use plain "
                "headings like 'Experience', 'Education', 'Skills'",
            )
        )

    # -- quantified achievements -----------------------------------------
    quantified = re.findall(
        r"\b\d+(?:\.\d+)?\s?(?:%|percent|x|×|k|m|bn|million|billion|hours?|days?|"
        r"weeks?|months?|years?|users?|customers?|requests?|tickets?)\b",
        text,
        re.IGNORECASE,
    )
    if quantified:
        checks.append(
            _check(
                "measurable achievements",
                True,
                f"{len(quantified)} quantified result(s) stated",
            )
        )
    else:
        checks.append(
            _check(
                "measurable achievements",
                False,
                "no results are quantified; numbers are what let a reader judge impact",
            )
        )

    # -- tables and graphics ---------------------------------------------
    table_rows = [line for line in lines if _TABLE_ROW_RE.match(line)]
    if table_rows:
        checks.append(
            _check(
                "no parsing-hostile tables",
                False,
                f"{len(table_rows)} line(s) look like table rows; a table's reading "
                "order is ambiguous, so list the content instead",
            )
        )
    else:
        checks.append(_check("no parsing-hostile tables", True, "no table rows detected"))

    graphic_hits = [marker for marker in _GRAPHIC_MARKERS if marker in text]
    if graphic_hits:
        checks.append(
            _check(
                "no decorative icon headers",
                False,
                f"{len(graphic_hits)} pictographic section marker(s); a pictograph "
                "adds nothing a parser can read",
            )
        )
    else:
        checks.append(_check("no decorative icon headers", True, "no pictographic markers"))

    # -- contact ----------------------------------------------------------
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))
    has_phone = bool(re.search(r"(?:\+\d[\d\s().-]{7,}\d)|\b0\d{2,4}[\s/-]\d{3,}\b", text))
    if has_email and has_phone:
        checks.append(_check("contact details", True, "email and phone both present"))
    elif has_email:
        checks.append(_check("contact details", True, "email present; no phone number found"))
    else:
        checks.append(
            _check("contact details", False, "no email address found — a recruiter cannot reply")
        )

    passed = sum(1 for check in checks if check["passed"])
    return {
        "checks": checks,
        "score": round(100.0 * passed / len(checks), 1) if checks else 0.0,
        "passed": passed,
        "total": len(checks),
        "failed": [check["name"] for check in checks if not check["passed"]],
        "sections_found": sections,
    }


def _leading_bullet(line: str) -> str:
    """The bullet glyph a line starts with, or "" when it starts with text."""
    match = re.match(r"^\s*([•‣⁃·◦▪▫●○■□–—\-\*•o])\s+", line)
    if match:
        return match.group(1)
    match = re.match(r"^\s*(?:[-*•‣⁃·◦▪▫●○■□–—]|\d+[.)])\s+", line)
    return match.group(0).strip()[0] if match else ""


def _title_style(token: str) -> str:
    """Bucket a job-title modifier as abbreviated or spelled out."""
    lowered = token.casefold().rstrip(".")
    if lowered in {"sr", "jr"}:
        return "abbreviated"
    return "spelled out"


def _company_base(name: str) -> str:
    """
    The employer's identity, with the legal form removed.

    ``Acme Corp``, ``Acme Corporation`` and ``Acme`` are one employer; treating
    them as three makes a CV look like it lists three jobs.
    """
    base = re.sub(
        r"\b(inc|ltd|llc|gmbh|ag|s\.?a\.?|b\.?v\.?|plc|corp|corporation|company|"
        r"group|kg|ohg|co)\b[.,]?",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^a-z0-9]", "", _normalise(base)) or _normalise(name)


# ---------------------------------------------------------------------------
# PDF layout and parsing
# ---------------------------------------------------------------------------


#: Below this point size, text stops being comfortably readable on screen or in
#: print. 8pt is the usual floor for a printed CV; anything under it is a
#: space-saving measure that costs legibility.
MIN_READABLE_FONT_SIZE = 8.0

#: Fraction of a page's text that must be at or above the readable size. Not
#: 1.0, because small print in a footer is legitimate and should not condemn the
#: document; the point is a CV that is *mostly* set too small to read.
MIN_READABLE_FRACTION = 0.9

#: How many filled rectangles a page may draw before it counts as a layout built
#: from boxes. A single rule under a heading is a rectangle; a panel per section
#: is a grid, and a grid is a text box by another name.
MAX_LAYOUT_RECTANGLES = 6


def _positioned_runs(page: Any) -> list[tuple[float, float, str]]:
    """
    Every text run on a page, as ``(x, y, text)``.

    Uses pypdf's visitor hook, which is the only way to get glyph positions
    without rendering. Without positions, reading order, column structure and
    overlap are all undecidable, and those are exactly the checks a CV most often
    fails silently.
    """
    runs: list[tuple[float, float, str]] = []

    def visitor(text, cm, tm, font_dict, font_size):  # noqa: ANN001 - pypdf signature
        if not text or not text.strip():
            return
        try:
            x, y = float(tm[4]), float(tm[5])
        except (TypeError, IndexError, ValueError):
            return
        runs.append((x, y, text))

    try:
        page.extract_text(visitor_text=visitor)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Positioned text extraction failed: %s", exc)
    return runs


def _sized_runs(page: Any) -> list[tuple[float, str]]:
    """
    Every text run on a page, as ``(font_size, text)``.

    The same visitor hook as :func:`_positioned_runs`, keeping the size, because
    legibility is not decidable from the extracted text — a paragraph and a
    footnote produce identical characters at different sizes.
    """
    runs: list[tuple[float, str]] = []

    def visitor(text, cm, tm, font_dict, font_size):  # noqa: ANN001 - pypdf signature
        if not text or not text.strip():
            return
        try:
            size = float(font_size)
        except (TypeError, ValueError):
            return
        runs.append((size, text))

    try:
        page.extract_text(visitor_text=visitor)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sized text extraction failed: %s", exc)
    return runs


def _page_layout_rectangles(page: Any) -> int:
    """
    How many filled rectangles the page draws.

    The signal for a document built from panels rather than from flowing text. A
    rule under a heading draws one; a grid of tinted panels behind each section
    draws one per section, and that is what a parser reads as a text box.
    """
    try:
        content = page.get_contents()
        if content is None:
            return 0
        data = content.get_data()
    except Exception:  # noqa: BLE001
        return 0
    # A rectangle path closed with "f" (or "f*") is a filled shape. Counting
    # occurrences rather than parsing paths keeps this cheap and forgiving.
    return len(re.findall(rb"\bre\b[^A-Za-z]*?\b(?:f|f\*)\b", data))


def _page_is_invisible(page: Any) -> bool:
    """
    Does this page draw text in invisible mode (Tr 3)?

    Invisible text is still extracted by a parser, so a CV can look empty to a
    human and full of content to a machine — or the reverse. Either way the two
    disagree, which is the failure this check exists to catch.
    """
    try:
        content = page.get_contents()
        if content is None:
            return False
        data = content.get_data()
    except Exception:  # noqa: BLE001
        return False
    return bool(re.search(rb"\b3\s+Tr\b", data))


def _page_has_images(page: Any) -> int:
    """How many image XObjects the page references."""
    try:
        resources = page.get("/Resources")
        if not resources:
            return 0
        xobjects = resources.get("/XObject")
        if not xobjects:
            return 0
        xobjects = xobjects.get_object()
        count = 0
        for ref in xobjects.values():
            try:
                if ref.get_object().get("/Subtype") == "/Image":
                    count += 1
            except Exception:  # noqa: BLE001
                continue
        return count
    except Exception:  # noqa: BLE001
        return 0


def _detect_columns(runs: list[tuple[float, float, str]], page_width: float) -> dict[str, Any]:
    """
    Decide whether the body text is set in two columns.

    Decided on *interleaving*, not on x-position. A sidebar, a skills column, or
    a header rule all produce runs on the left and the right; only a genuine
    two-column body produces lines where a left-band and a right-band run appear
    at the same height. Measuring that way, a single wide column with a
    full-width rule is not mistaken for two columns.
    """
    body = [(x, y, t) for x, y, t in runs if t.strip() and len(t.strip()) > 2]
    if len(body) < 12:
        return {"is_two_column": False, "detail": "too few positioned runs to judge"}

    xs = sorted({round(x) for x, _, _ in body})
    if len(xs) < 4:
        return {"is_two_column": False, "detail": "all text starts at one x position"}

    # Group by rounded y, then look for rows that hold text in two separated
    # x bands. A two-column body shows this on most of its lines; a header rule
    # or a centred title shows it on a handful.
    by_y: dict[int, list[float]] = {}
    for x, y, _ in body:
        by_y.setdefault(round(y / SAME_LINE_TOLERANCE), []).append(x)

    split_rows = 0
    considered_rows = 0
    for positions in by_y.values():
        if len(positions) < 2:
            continue
        considered_rows += 1
        positions.sort()
        gap = max(b - a for a, b in zip(positions, positions[1:]))
        if gap >= MIN_COLUMN_WIDTH:
            split_rows += 1

    if considered_rows == 0:
        return {"is_two_column": False, "detail": "no multi-run lines to analyse"}

    ratio = split_rows / considered_rows
    if ratio >= 0.5:
        return {
            "is_two_column": True,
            "detail": (
                f"{split_rows} of {considered_rows} lines carry text in two separate "
                "bands; a two-column body is read in the wrong order by most parsers"
            ),
        }
    if split_rows:
        return {
            "is_two_column": False,
            "detail": (
                f"{split_rows} of {considered_rows} lines are split, below the "
                "threshold for a two-column body — most likely a rule or sidebar"
            ),
        }
    return {"is_two_column": False, "detail": "text forms a single continuous block"}


def _check_reading_order(runs: list[tuple[float, float, str]]) -> dict[str, Any]:
    """
    Is the extracted order the visual order?

    A parser reads in the order the content stream emits, which is *not* the
    order a human sees. When it disagrees, a CV reads top-to-bottom correctly but
    extracts with a bullet from the end of one job landing before the heading of
    the next.
    """
    if len(runs) < 4:
        return {"passed": True, "detail": "too few runs to judge reading order"}

    inversions = 0
    for (x1, y1, t1), (x2, y2, t2) in zip(runs, runs[1:]):
        if not (t1.strip() and t2.strip()):
            continue
        # Moving up the page is correct. Moving down is a wrap only if the new
        # run starts at or left of where the previous one began.
        if y2 < y1 - SAME_LINE_TOLERANCE:
            continue
        if y2 > y1 + SAME_LINE_TOLERANCE and x2 > x1 + OVERLAP_SLACK:
            inversions += 1

    total = max(1, len(runs) - 1)
    if inversions == 0:
        return {"passed": True, "detail": "text extracts in visual order"}
    ratio = inversions / total
    if ratio <= 0.05:
        return {
            "passed": True,
            "detail": f"{inversions} of {total} transitions look out of order — within tolerance",
        }
    return {
        "passed": False,
        "detail": (
            f"{inversions} of {total} text runs extract out of visual order; a parser "
            "will interleave lines from different parts of the CV"
        ),
    }


def _check_overlap(runs: list[tuple[float, float, str]]) -> dict[str, Any]:
    """
    Do two text runs sit on top of each other?

    Overlapping runs usually mean a text box drawn over a heading. Extraction
    merges them into one unreadable line, so the affected content is lost even
    though it is plainly visible to a reader.
    """
    text_runs = [(x, y, t) for x, y, t in runs if t.strip()]
    if len(text_runs) < 2:
        return {"passed": True, "detail": "too few runs to check", "count": 0}

    by_y: dict[int, list[tuple[float, float, str]]] = {}
    for x, y, t in text_runs:
        by_y.setdefault(round(y / SAME_LINE_TOLERANCE), []).append((x, y, t))

    collisions = 0
    for group in by_y.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda item: item[0])
        for left, right in zip(group, group[1:]):
            if right[0] < left[0] + 1.0:
                collisions += 1

    if collisions == 0:
        return {"passed": True, "detail": "no overlapping text detected", "count": 0}
    return {
        "passed": False,
        "count": collisions,
        "detail": (
            f"{collisions} pair(s) of text runs share a line and overlap; overlapping "
            "text merges into one unreadable line when extracted"
        ),
    }


def _check_clipping(
    runs: list[tuple[float, float, str]],
    page_width: float,
) -> dict[str, Any]:
    """Does any text run reach the page edge, where it may be cut off?"""
    if not runs or page_width <= 0:
        return {"passed": True, "detail": "no page geometry available", "count": 0}

    clipped = [(x, y, t) for x, y, t in runs if t.strip() and x > page_width - PAGE_EDGE_SLACK]
    if not clipped:
        return {
            "passed": True,
            "detail": f"no text within {PAGE_EDGE_SLACK:.0f}pt of the page edge",
            "count": 0,
        }
    return {
        "passed": False,
        "count": len(clipped),
        "detail": (
            f"{len(clipped)} text run(s) reach the page edge; content that wide is "
            "at risk of being cut off in the rendered document"
        ),
    }


def _field_checks(
    pdf_text: str,
    source_text: str,
    sections: list[str],
) -> list[dict[str, Any]]:
    """
    Are the individual facts still parseable after typesetting?

    A CV can retain 90% of its words and still lose the one line that says who
    the candidate is, because that line sat in a header. These checks name each
    kind of fact the way a parser needs it, rather than counting words.
    """
    normalised_pdf = _normalise(pdf_text)
    normalised_source = _normalise(source_text)
    checks: list[dict[str, Any]] = []

    def added(label: str, name: str, pattern: str) -> None:
        # Only ever checked when the source actually contains it. Demanding an
        # email the candidate never wrote would report a perfect CV as broken.
        if re.search(pattern, normalised_source):
            checks.append(
                _check(
                    name,
                    bool(re.search(pattern, normalised_pdf)),
                    "present in the generated document"
                    if re.search(pattern, normalised_pdf)
                    else "in your CV but not recoverable from the PDF",
                )
            )

    added("email", "email parseable", r"[\w.+-]+@[\w-]+\.[\w.-]+")
    added("phone", "phone parseable", r"(?:\+\d[\d\s().-]{7,}\d)|\b0\d{2,4}[\s/-]\d{3,}\b")
    added("dates", "dates parseable", r"(?:19|20)\d{2}")
    added("employers", "employers parseable", r"\b(?:inc|ltd|llc|gmbh|ag|plc|corp|group)\b")

    if "skills" in sections:
        checks.append(_check("skills section parseable", True, "skills heading located"))
    if "education" in sections:
        checks.append(_check("education parseable", True, "education heading located"))
    if "experience" in sections:
        checks.append(_check("experience parseable", True, "experience heading located"))
    if "contact" in sections:
        checks.append(_check("contact details parseable", True, "contact block located"))

    return checks


def _check_duplicated_sections(pdf_text: str) -> dict[str, Any]:
    """
    Does any section heading appear more than once?

    A duplicated block usually means a template emitted its header twice, or the
    body was appended rather than replacing a previous section. The duplicate is
    then extracted twice, which reads to a parser as two identical jobs.
    """
    counts: Counter[str] = Counter()
    for name, pattern in _SECTION_RES.items():
        counts[name] = len(pattern.findall(pdf_text or ""))
    repeats = [name for name, count in counts.items() if count > 1]
    if not repeats:
        return {"passed": True, "detail": "no section appears twice"}
    return {
        "passed": False,
        "detail": (
            f"repeated section(s): {', '.join(sorted(repeats))} — a duplicated block "
            "is extracted twice and reads as repeated content"
        ),
    }


def _check_header_footer(pages: list[Any]) -> dict[str, Any]:
    """
    Does any page carry a running header or footer holding unique content?

    Content that lives in a page margin is the first thing an extractor drops or
    misplaces, so a name or phone number there is effectively lost. A repeated
    header carrying only the candidate's own name is tolerable; unique content —
    a role, a skill, a date — is not.
    """
    if len(pages) < 2:
        return {
            "passed": True,
            "detail": "single-page document: no repeated margin content possible",
        }

    margins: list[str] = []
    for page in pages:
        runs = _positioned_runs(page)
        if not runs:
            continue
        try:
            height = float(page.mediabox.height)
        except Exception:  # noqa: BLE001
            continue
        band = height * 0.07
        margins.extend(t for _, y, t in runs if y > height - band or y < band)

    if not margins:
        return {"passed": True, "detail": "no text in the page margins"}

    normalised = [_normalise(text) for text in margins if text.strip()]
    if not normalised:
        return {"passed": True, "detail": "margin text is blank"}

    unique = [text for text in normalised if len(text) > 12]
    if not unique:
        return {
            "passed": True,
            "detail": "only short strings in the margins — running header, not content",
        }

    return {
        "passed": False,
        "detail": (
            f"{len(unique)} content-bearing string(s) sit in a page margin "
            f"(e.g. {unique[0][:48]!r}); margin content is the first thing an "
            "extractor loses or reorders"
        ),
    }


def analyze_pdf_layout(
    pdf_bytes: bytes,
    expected_text: str = "",
) -> dict[str, Any]:
    """
    Can a parser actually read this document?

    Answers the ten questions requirement 24 ends on, using glyph positions
    rather than a rendered image wherever the question is decidable that way.
    Nothing is inferred about the CV's *quality* here — only about whether the
    document survives extraction.
    """
    if not pdf_bytes:
        return {
            "checks": [],
            "score": 0.0,
            "passed": 0,
            "total": 0,
            "failed": ["no PDF supplied"],
            "unreadable": True,
            "text_chars": 0,
            "pages": 0,
            "sections_found": [],
            "content_retention": None,
            "latex_artifacts": [],
            "replacement_chars": 0,
            "image_count": 0,
        }

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001
        return {
            "checks": [
                _check("document opens", False, f"could not be read ({type(exc).__name__})")
            ],
            "score": 0.0,
            "passed": 0,
            "total": 1,
            "failed": ["document opens"],
            "unreadable": True,
            "text_chars": 0,
            "pages": 0,
            "sections_found": [],
            "content_retention": None,
            "latex_artifacts": [],
            "replacement_chars": 0,
            "image_count": 0,
        }

    checks: list[dict[str, Any]] = [_check("document opens", True, f"{len(pages)} page(s)")]

    per_page_runs: list[list[tuple[float, float, str]]] = []
    page_widths: list[float] = []
    for page in pages:
        per_page_runs.append(_positioned_runs(page))
        try:
            page_widths.append(float(page.mediabox.width))
        except Exception:  # noqa: BLE001
            page_widths.append(0.0)

    pdf_text = "\n".join((page.extract_text() or "") for page in pages) if pages else ""
    text_chars = len(pdf_text.strip())
    all_runs = [run for runs in per_page_runs for run in runs]

    # -- 1. text is extractable at all ------------------------------------
    checks.append(
        _check(
            "text extractable",
            text_chars >= MIN_PDF_TEXT_CHARS,
            f"{text_chars:,} characters recovered from {len(pages)} page(s)",
        )
    )

    # -- 2. not an image-only document ------------------------------------
    image_counts = [_page_has_images(page) for page in pages]
    total_images = sum(image_counts)
    if total_images and text_chars < MIN_PDF_TEXT_CHARS:
        checks.append(
            _check(
                "no image-only content",
                False,
                f"{total_images} image(s) and almost no text; a parser sees a blank page",
            )
        )
    else:
        checks.append(
            _check(
                "no image-only content",
                True,
                "no images"
                if not total_images
                else f"{total_images} image(s), text present alongside",
            )
        )

    # -- 3. no raw markup survived ---------------------------------------
    artifacts = find_latex_artifacts(pdf_text)
    checks.append(
        _check(
            "no raw markup",
            not artifacts,
            "clean"
            if not artifacts
            else f"{len(artifacts)} markup operator(s) visible in the text",
        )
    )

    # -- 4. every character rendered --------------------------------------
    replacements = len(REPLACEMENT_RE.findall(pdf_text))
    checks.append(
        _check(
            "all characters rendered",
            replacements == 0,
            "clean" if not replacements else f"{replacements} unrenderable character(s)",
        )
    )

    # -- 5. no decorative glyph standing in for a letter -----------------
    suspect = len(_SUSPECT_GLYPH_RE.findall(pdf_text))
    checks.append(
        _check(
            "no symbol-substituted text",
            suspect == 0,
            "clean" if not suspect else f"{suspect} private-use/dingbat character(s) in the text",
        )
    )

    # -- 6. nothing drawn invisibly --------------------------------------
    invisible = [index for index, page in enumerate(pages, 1) if _page_is_invisible(page)]
    checks.append(
        _check(
            "no invisible text",
            not invisible,
            "clean" if not invisible else f"page(s) {invisible} draw text in invisible mode",
        )
    )

    # -- 7. single column -------------------------------------------------
    if all_runs:
        column_results = [
            _detect_columns(runs, width) for runs, width in zip(per_page_runs, page_widths) if runs
        ]
        two_column = [result for result in column_results if result["is_two_column"]]
        if two_column:
            checks.append(
                _check(
                    "single column",
                    False,
                    two_column[0]["detail"],
                )
            )
        else:
            detail = column_results[0]["detail"] if column_results else "no positioned text"
            checks.append(_check("single column", True, detail))
    else:
        checks.append(_check("single column", True, "no positioned text to analyse"))

    # -- 8. correct reading order -----------------------------------------
    order = _check_reading_order(all_runs)
    checks.append(_check("reading order", order["passed"], order["detail"]))

    # -- 9. no overlapping text ------------------------------------------
    overlaps = _check_overlap(all_runs)
    checks.append(_check("no overlapping text", overlaps["passed"], overlaps["detail"]))

    # -- 10. nothing clipped at the page edge ---------------------------
    clipping = _check_clipping(all_runs, max(page_widths) if page_widths else 0.0)
    checks.append(_check("no clipped text", clipping["passed"], clipping["detail"]))

    # -- 11. no running header/footer holding content -------------------
    margin = _check_header_footer(pages)
    checks.append(_check("no content in page margins", margin["passed"], margin["detail"]))

    # -- 12. no panel/box layout ----------------------------------------
    rectangle_counts = [_page_layout_rectangles(page) for page in pages]
    total_rectangles = sum(rectangle_counts)
    if total_rectangles > MAX_LAYOUT_RECTANGLES:
        checks.append(
            _check(
                "no text boxes or panels",
                False,
                f"the page draws {total_rectangles} filled shapes; a CV built from "
                "panels is read as a series of boxes rather than as text",
            )
        )
    else:
        checks.append(
            _check(
                "no text boxes or panels",
                True,
                "no box-based layout detected"
                if not total_rectangles
                else f"{total_rectangles} decorative shape(s), within a normal layout",
            )
        )

    # -- 13. readable typography -----------------------------------------
    sized: list[tuple[float, str]] = []
    for page in pages:
        sized.extend(_sized_runs(page))
    if not sized:
        checks.append(_check("readable type size", True, "no font sizes were reported"))
    else:
        small = [(size, text) for size, text in sized if size < MIN_READABLE_FONT_SIZE]
        fraction = 1.0 - (len(small) / len(sized))
        if fraction < MIN_READABLE_FRACTION:
            smallest = min((size for size, _ in small), default=0.0)
            checks.append(
                _check(
                    "readable type size",
                    False,
                    f"{len(small)} of {len(sized)} text runs are below "
                    f"{MIN_READABLE_FONT_SIZE:.0f}pt (smallest {smallest:.1f}pt); "
                    "most of the CV is too small to read comfortably",
                )
            )
        else:
            smallest_all = min((size for size, _ in sized), default=0.0)
            checks.append(
                _check(
                    "readable type size",
                    True,
                    f"smallest type is {smallest_all:.1f}pt"
                    + (f", with {len(small)} run(s) in small print" if small else ""),
                )
            )

    # -- 14. headings locatable ------------------------------------------
    sections = detect_sections(pdf_text)
    checks.append(
        _check(
            "headings locatable",
            len(sections) >= MIN_SECTIONS_EXPECTED,
            f"{len(sections)} section heading(s) found: {', '.join(sections) or 'none'}",
        )
    )

    # -- 15. no duplicated sections --------------------------------------
    duplicates = _check_duplicated_sections(pdf_text)
    checks.append(_check("no duplicated sections", duplicates["passed"], duplicates["detail"]))

    # -- 16. each kind of fact still parseable --------------------------
    checks.extend(_field_checks(pdf_text, expected_text or "", sections))

    # -- 17. content survived generation --------------------------------
    retention: float | None = None
    source_tokens = checkable_tokens(expected_text) if expected_text else set()
    if source_tokens:
        present = checkable_tokens(pdf_text)
        missing = source_tokens - present
        retention = 1.0 - len(missing) / len(source_tokens)
        checks.append(
            _check(
                "content retained",
                retention >= MIN_CONTENT_RETENTION,
                f"{retention:.0%} of your CV's content words are present "
                f"({MIN_CONTENT_RETENTION:.0%} required)",
            )
        )

    passed = sum(1 for check in checks if check["passed"])
    return {
        "checks": checks,
        "score": round(100.0 * passed / len(checks), 1) if checks else 0.0,
        "passed": passed,
        "total": len(checks),
        "failed": [check["name"] for check in checks if not check["passed"]],
        "text_chars": text_chars,
        "pages": len(pages),
        "sections_found": sections,
        "content_retention": round(retention, 4) if retention is not None else None,
        "latex_artifacts": artifacts,
        "replacement_chars": replacements,
        "image_count": total_images,
        "unreadable": False,
    }


__all__ = [
    "LATEX_ARTIFACT_RE",
    "MIN_CONTENT_RETENTION",
    "MIN_PDF_TEXT_CHARS",
    "SECTION_PATTERNS",
    "analyze_pdf_layout",
    "analyze_text_formatting",
    "checkable_tokens",
    "detect_sections",
    "find_latex_artifacts",
]
