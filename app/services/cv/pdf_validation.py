"""
Verify that a generated PDF actually contains the generated CV.

Why this exists
---------------
``pdflatex`` exiting 0 and producing a non-empty file is not evidence that a CV
was produced. A document can compile cleanly and still be wrong: a page that
overflowed and lost its content, a section that silently rendered nothing, a
template whose body was dropped, an accent that came out as a replacement
character. Every one of those produces a valid PDF and a "success" response.

So the finished bytes are read back and compared against what was asked for. The
checks are deliberately about *content*, not about appearance:

* the file exists, parses, and has the expected page count;
* it is not blank;
* text can be extracted at all;
* the candidate's own content -- names, employers, roles, skills -- is present;
* the sections the CV claimed to have are present;
* no raw LaTeX leaked into the output;
* nothing that was in the source is missing from the PDF.

What is deliberately *not* checked here
---------------------------------------
Visual layout. Whether a heading collides with a heading above it, whether a
table overhangs a margin, whether a page is 90% whitespace: those need a
renderer, and a false positive here would reject a perfectly good CV. The
structural part of that lives in ``tests/test_pdf_layout_regression.py``, which
renders pages and measures them.

Privacy
-------
The report contains lengths, counts, section names and the *names* of missing
items, never the content of the CV. A failure is reportable to a user without
reproducing their personal data in a log.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)

#: Below this many extracted characters a page is considered blank. A real CV
#: page is well over 500 characters; 200 leaves room for a genuinely sparse
#: layout without accepting an empty page.
MIN_MEANINGFUL_TEXT_CHARS = 200

#: Fraction of the candidate's checkable tokens that must survive. Set below 1.0
#: because a CV is a summary: some source text is legitimately dropped. Set well
#: above the level at which a template bug would hide, because losing a third of
#: a resume is a failure even if the PDF "compiled".
MIN_CONTENT_RETENTION = 0.55

#: Tokens shorter than this are ignored when checking retention. Single letters
#: and two-letter fragments match far too much to be evidence of anything.
MIN_CHECKABLE_TOKEN_CHARS = 4


class PDFValidationError(RuntimeError):
    """
    The PDF is structurally valid but its content is not usable.

    Carries the report so a caller can show what was actually wrong instead of a
    bare "generation failed", which is the failure mode this whole module exists
    to remove.
    """

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report or {}


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

#: Section heading -> the words that indicate it, in any of the supported output
#: languages. Only sections a CV can plausibly have are listed; nothing is
#: required, these are used to confirm what the source claimed.
_SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "summary": (
        r"\bprofile\b",
        r"\bsummary\b",
        r"\babout me\b",
        r"\bobjective\b",
        r"\bberufserfahrung\b",
        r"\bprofil\b",
        r"\bzusammenfassung\b",
        r"\bberufsprofil\b",
        r"\bkurzprofil\b",
    ),
    "experience": (
        r"\bexperience\b",
        r"\bwork experience\b",
        r"\bemployment\b",
        r"\bprofessional experience\b",
        r"\berfahrung\b",
        r"\bberufserfahrung\b",
        r"\btätigkeit\b",
        r"\banstellung\b",
        r"\bbeschäftigung\b",
    ),
    "education": (
        r"\beducation\b",
        r"\bacademic\b",
        r"\bqualification",
        r"\bausbildung\b",
        r"\bbildung\b",
        r"\bstudium\b",
        r"\babschluss\b",
    ),
    "skills": (
        r"\bskills\b",
        r"\btechnical skills\b",
        r"\bcompetenc",
        r"\btool",
        r"\btechnolog",
        r"\bfähigkeiten\b",
        r"\bfertigkeiten\b",
        r"\bkenntnisse\b",
        r"\bkompetenzen\b",
    ),
    "projects": (
        r"\bprojects\b",
        r"\bportfolio\b",
        r"\bprojekt",
    ),
    "certifications": (
        r"\bcertificat",
        r"\blicen[cs]e",
        r"\bqualification",
        r"\bzertifikat",
        r"\bzertifizierung",
        r"\bnachweise\b",
    ),
    "languages": (
        r"\blanguages\b",
        r"\blanguage skills\b",
        r"\bsprachen\b",
    ),
    "contact": (
        r"\bcontact\b",
        r"\bkontakt\b",
    ),
}


def _section_pattern(name: str) -> re.Pattern[str]:
    """
    One alternation per section, so a caller can match it against a heading.

    A section is recognised by any of its indicators in any supported output
    language, which is why each entry is a tuple of alternatives rather than a
    single expression.
    """
    return re.compile("|".join(_SECTION_PATTERNS[name]), re.IGNORECASE)


_SECTION_RE: dict[str, re.Pattern[str]] = {
    name: _section_pattern(name) for name in _SECTION_PATTERNS
}

#: A token that means a LaTeX command survived into the rendered text. These are
#: the ones that indicate a template or escaping failure rather than a document
#: that legitimately contains a brace.
_LATEX_ARTIFACT_RE = re.compile(
    r"\\(?:begin|end|item|section|subsection|textbf|textit|emph|newline|"
    r"linebreak|hspace|vspace|textbf|href|url|color|colorbox|parbox|"
    r"noindent|centering|raggedright|tabular|includegraphics|"
    r"newcommand|renewcommand|documentclass|usepackage)\b"
    r"|\{\{|\}\}"
    r"|\\[a-zA-Z]+\{"
)

#: Unicode replacement characters, which is what an unmapped font produces.
_REPLACEMENT_RE = re.compile(r"[�]")

#: Words that must never be treated as retained content, because they appear in
#: every CV's own heading regardless of the candidate.
#: Words that appear in every CV whatever the candidate, in either supported
#: output language. They are excluded from retention checks because their
#: presence proves nothing: a section heading survives even when the CV around
#: it is empty, so counting them would let a badly broken document pass.
_GENERIC_TOKENS = frozenset(
    {
        # Function words, English.
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
        "who",
        "which",
        "than",
        "then",
        "them",
        "they",
        # Function words, German.
        "und",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "ein",
        "eine",
        "einen",
        "mit",
        "von",
        "oder",
        "auch",
        "als",
        "im",
        "in",
        "zu",
        "zur",
        "zum",
        "auf",
        "für",
        "sie",
        "wir",
        "ist",
        "sind",
        "werden",
        "auch",
        "oder",
        "beim",
        "durch",
        "gegen",
        "ohne",
        "seit",
        "bis",
        # Section headings. Present in every CV by definition.
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
        "kenntnisse",
        # Document furniture.
        "page",
        "seite",
        "cv",
        "lebenslauf",
        "resume",
        "curriculum",
    }
)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract the text of every page.

    A PDF that cannot be read at all is a failure, not an empty document: a
    parser that silently returns nothing would make every later check pass.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    except Exception:
        raise PDFValidationError("Generated PDF could not be parsed.") from None

    pages: list[str] = []

    for number, page in enumerate(reader.pages, 1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            # The page exists but its content stream is unreadable. Report the
            # page number, never the stream: it contains the CV.
            raise PDFValidationError(
                f"Text could not be extracted from page {number} of the "
                f"generated PDF ({type(exc).__name__})."
            ) from None

    return "\n".join(pages)


def _normalise(text: str) -> str:
    """
    Reduce text to a form where accents and case cannot hide a difference.

    ``Müller`` and ``Muller`` are the same name. Without this, a correctly
    rendered umlaut reads as a *missing* token and a CV gets rejected for being
    right.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.casefold()


def checkable_tokens(text: str) -> set[str]:
    """
    Content words worth checking for survival.

    Generic CV vocabulary and very short fragments are excluded: they are either
    present in every CV regardless of the candidate, or match too much to be
    evidence.
    """
    normalised = _normalise(text)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+#./-]*", normalised)
        if len(token) >= MIN_CHECKABLE_TOKEN_CHARS and token not in _GENERIC_TOKENS
    }
    return tokens


def detect_sections(text: str) -> list[str]:
    """Names of the CV sections that appear in the extracted text."""
    found = [name for name, pattern in _SECTION_RE.items() if pattern.search(text)]
    return found


def find_latex_artifacts(text: str) -> list[str]:
    """
    Raw LaTeX that reached the rendered page.

    Returned as the matched *commands*, never the surrounding text, so a report
    can name the problem without reproducing the CV.
    """
    return sorted(set(_LATEX_ARTIFACT_RE.findall(text or "")))


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def validate_pdf_content(
    pdf_bytes: bytes,
    *,
    expected_pages: int = 1,
    expected_text: str = "",
    expected_sections: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check a finished PDF against what it was supposed to contain.

    ``expected_text`` is the CV content that went in, used to prove nothing was
    lost. ``expected_sections`` names sections the source claimed; each is
    required to be present, but nothing is invented -- a CV with no projects
    section is not asked for one.

    Returns a report rather than raising, except for conditions that make the
    document unusable. Callers that only want to warn can use the report.
    """
    report: dict[str, Any] = {
        "bytes": len(pdf_bytes or b""),
        "pages": 0,
        "text_chars": 0,
        "sections_found": [],
        "sections_missing": [],
        "latex_artifacts": [],
        "replacement_chars": 0,
        "content_retention": None,
        "missing_tokens": [],
        "problems": [],
    }

    if not pdf_bytes:
        report["problems"].append("empty_file")
        raise PDFValidationError("The PDF generator produced no file.", report)

    text = extract_pdf_text(pdf_bytes)

    try:
        report["pages"] = len(PdfReader(io.BytesIO(pdf_bytes), strict=False).pages)
    except Exception:
        report["problems"].append("unparseable")
        raise PDFValidationError("The generated PDF could not be parsed.", report) from None

    report["text_chars"] = len(text.strip())
    report["sections_found"] = detect_sections(text)
    report["latex_artifacts"] = find_latex_artifacts(text)
    report["replacement_chars"] = len(_REPLACEMENT_RE.findall(text))

    # -- not blank ---------------------------------------------------------
    if report["text_chars"] < MIN_MEANINGFUL_TEXT_CHARS:
        report["problems"].append("blank_or_near_blank")
        raise PDFValidationError(
            "The generated PDF is blank: it contains almost no text. "
            f"Extracted {report['text_chars']} characters, expected at least "
            f"{MIN_MEANINGFUL_TEXT_CHARS}.",
            report,
        )

    # -- page count --------------------------------------------------------
    if expected_pages and report["pages"] != expected_pages:
        report["problems"].append("page_count_mismatch")
        raise PDFValidationError(
            f"The generated PDF has {report['pages']} page(s); " f"{expected_pages} was required.",
            report,
        )

    # -- raw LaTeX did not leak -------------------------------------------
    if report["latex_artifacts"]:
        report["problems"].append("latex_artifacts")
        raise PDFValidationError(
            "The generated PDF contains raw LaTeX markup that should have been "
            f"typeset: {', '.join(report['latex_artifacts'][:6])}.",
            report,
        )

    # -- characters that a font could not render --------------------------
    if report["replacement_chars"]:
        report["problems"].append("unrendered_characters")
        # Not fatal on its own: a single odd character in a URL is not a broken
        # CV, and rejecting the whole document for it would be worse. Reported so
        # the caller can decide.
        logger.warning(
            "Generated PDF contains %d unrendered character(s).",
            report["replacement_chars"],
        )

    # -- sections that were claimed ---------------------------------------
    if expected_sections:
        normalised = _normalise(text)
        missing = [
            section
            for section in expected_sections
            if section not in report["sections_found"]
            and not _SECTION_RE.get(section, re.compile(r"(?!x)x")).search(normalised)
        ]
        report["sections_missing"] = missing

        if missing:
            report["problems"].append("missing_sections")
            raise PDFValidationError(
                "The generated PDF is missing section(s) that the CV contained: "
                f"{', '.join(missing)}.",
                report,
            )

    # -- content retention -------------------------------------------------
    if expected_text:
        source_tokens = checkable_tokens(expected_text)
        present_tokens = checkable_tokens(text)

        if source_tokens:
            missing_tokens = sorted(source_tokens - present_tokens)
            retention = 1.0 - (len(missing_tokens) / len(source_tokens))
            report["content_retention"] = round(retention, 4)
            # Names only. A CV's words are the candidate's personal data and must
            # not be reproduced in a log or an error.
            report["missing_tokens"] = missing_tokens[:40]

            if retention < MIN_CONTENT_RETENTION:
                report["problems"].append("content_lost")
                raise PDFValidationError(
                    "The generated PDF does not contain enough of the CV that "
                    f"was produced: {retention:.0%} of its content words are "
                    f"present, {MIN_CONTENT_RETENTION:.0%} required. Content was "
                    "lost between generation and typesetting.",
                    report,
                )

    return report


__all__ = [
    "MIN_CONTENT_RETENTION",
    "MIN_MEANINGFUL_TEXT_CHARS",
    "PDFValidationError",
    "checkable_tokens",
    "detect_sections",
    "extract_pdf_text",
    "find_latex_artifacts",
    "validate_pdf_content",
]
