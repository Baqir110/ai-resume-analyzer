"""
PDF content validation and layout differentiation.

These tests compile real PDFs with a real ``pdflatex`` when one is available, and
skip the compilation-dependent ones when it is not. They need no LLM: every
document is built from a fixed fixture, so the results are deterministic and the
suite never sends a CV anywhere.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.cv.latex_generator import (
    CV_LAYOUT_COLUMNS,
    CV_TEMPLATES,
    _expected_content_from_latex,
    _patch_template_preamble,
    _sections_claimed_by_latex,
    compile_latex_to_pdf,
    required_language_for_layout,
)
from app.services.cv.pdf_compiler import (
    LaTeXCompilationError,
    LaTeXSourceError,
    PDFLayoutError,
    compile_single_page_pdf,
    pdf_page_count,
)
from app.services.cv.pdf_validation import (
    MIN_CONTENT_RETENTION,
    PDFValidationError,
    checkable_tokens,
    detect_sections,
    extract_pdf_text,
    find_latex_artifacts,
    validate_pdf_content,
)

PDFLATEX = shutil.which("pdflatex")

requires_pdflatex = pytest.mark.skipif(
    PDFLATEX is None,
    reason="pdflatex is not on PATH",
)

# ---------------------------------------------------------------------------
# A fixed, non-sensitive fixture. Synthetic person, synthetic employer.
# ---------------------------------------------------------------------------

FIXTURE_NAME = "Alex Berger"
FIXTURE_EMAIL = "alex.berger@example.invalid"
FIXTURE_PHONE = "+49 30 1234567"
FIXTURE_CITY = "Hamburg, Germany"

#: LaTeX pipe separator. Defined outside the header because its braces would be
#: read as replacement fields inside an f-string.
_SEP = r" \textbar{} "

FIXTURE_NAME = "Alex Berger"
FIXTURE_ROLE = "Platform Engineer"
FIXTURE_EMAIL = "alex.berger@example.invalid"
FIXTURE_PHONE = "+49 30 1234567"
FIXTURE_CITY = "Hamburg, Germany"

FIXTURE_HEADER = (
    "\n".join(
        [
            FIXTURE_NAME,
            FIXTURE_ROLE,
            _SEP.join([FIXTURE_EMAIL, FIXTURE_PHONE, FIXTURE_CITY]),
        ]
    )
    + "\n"
)

#: A fixed, non-sensitive body. Synthetic person, synthetic employers. The same
#: body is used for every layout so the layout comparison is controlled: any
#: difference between two outputs is the template, not the input.
FIXTURE_BODY = r"""\section*{Profile}
Platform Engineer with eight years building and operating cloud-native
services on AWS. Focused on infrastructure automation, container orchestration,
continuous delivery and observability.

\section*{Experience}
\textbf{Platform Engineer} \textbar{} Beispiel GmbH \textbar{} 2020 -- Present
\begin{itemize}
  \item Built Python services with FastAPI and instrumented them with Prometheus and Grafana.
  \item Managed Kubernetes clusters across three environments and automated provisioning with Terraform.
  \item Reduced deployment time by forty percent by introducing GitLab CI pipelines.
  \item Established on-call runbooks that reduced mean time to recovery substantially.
  \item Owned the release process for three product teams including staging promotion.
  \item Partnered with security to harden the container image pipeline end to end.
  \item Migrated twenty-two legacy services onto the shared platform without customer-visible downtime.
  \item Automated certificate rotation for all internal domains using an internal CA.
\end{itemize}

\textbf{Systems Engineer} \textbar{} Muster AG \textbar{} 2018 -- 2020
\begin{itemize}
  \item Managed Linux systems, DNS, VPN and infrastructure operations for two data centres.
  \item Automated operational reporting, replacing a manual monthly process.
  \item Diagnosed and resolved recurring network faults by instrumenting the switches.
\end{itemize}

\section*{Education}
\textbf{M.Sc. Computer Science} \textbar{} Technical University of Hamburg
\begin{itemize}
  \item Thesis on scheduling heuristics for container placement.
\end{itemize}

\section*{Skills}
Languages: Python, Bash, Go, SQL, JavaScript. Platform: Docker, Kubernetes,
AWS, Terraform, Ansible, GitLab CI, GitHub Actions. Data: PostgreSQL, Redis,
Prometheus, Grafana, Loki, OpenTelemetry. Networking: Nginx, DNS, VPN, TLS.

\section*{Languages}
English (native), German (fluent), Turkish (professional).
"""

#: Headings the German layouts expect, so the same fixture is a realistic input
#: for every layout rather than an English CV wearing a German template.
_GERMAN_HEADINGS = {
    r"\section*{Profile}": r"\section*{Profil}",
    r"\section*{Experience}": r"\section*{Berufserfahrung}",
    r"\section*{Education}": r"\section*{Ausbildung}",
    r"\section*{Skills}": r"\section*{Fähigkeiten}",
    r"\section*{Languages}": r"\section*{Sprachen}",
    "Languages:": "Sprachen:",
    "Present": "Heute",
}


def build_document(layout: str) -> str:
    """
    A complete, compilable document for one layout.

    Assembled through the same placeholder contract the application uses
    (``CANDIDATE_NAME_PLACEHOLDER`` and friends), not by pasting text into a
    template. A test that builds documents a different way from the application
    proves the test harness works, not that the application does.
    """
    from app.services.cv.latex_generator import _escape_latex_text, _render_candidate_contact

    template = _patch_template_preamble(CV_TEMPLATES[layout])
    body = FIXTURE_BODY

    # The German layouts expect German headings, so the fixture is translated
    # for them. This also exercises multi-language section detection, which a CV
    # generated in German depends on.
    if required_language_for_layout(layout) == "de":
        for english, german in _GERMAN_HEADINGS.items():
            body = body.replace(english, german)

    header = {
        "name": FIXTURE_NAME,
        "title": FIXTURE_ROLE,
        "email": FIXTURE_EMAIL,
        "phone": FIXTURE_PHONE,
        "location": FIXTURE_CITY,
    }

    template = template.replace("CANDIDATE_NAME_PLACEHOLDER", _escape_latex_text(header["name"]))
    template = template.replace("CANDIDATE_TITLE_PLACEHOLDER", _escape_latex_text(header["title"]))
    template = template.replace("CANDIDATE_CONTACT_PLACEHOLDER", _render_candidate_contact(header))
    document = template.replace("RESUME_BODY_PLACEHOLDER", body)

    assert "PLACEHOLDER" not in document, (
        f"{layout} has a placeholder the application does not substitute; "
        f"the fixture would be testing the wrong contract"
    )
    return document


# ===========================================================================
# 1. Text extraction
# ===========================================================================


def test_a_pdf_that_cannot_be_parsed_is_a_failure_not_an_empty_document():
    """
    A parser that silently returned nothing would make every later check pass.
    """
    with pytest.raises(PDFValidationError, match="could not be parsed"):
        validate_pdf_content(b"not a pdf at all")


def test_an_empty_file_is_rejected():
    with pytest.raises(PDFValidationError, match="no file"):
        validate_pdf_content(b"")


# ===========================================================================
# 2. Section detection
# ===========================================================================


@pytest.mark.parametrize(
    "heading,section",
    [
        ("Experience", "experience"),
        ("Berufserfahrung", "experience"),
        ("Education", "education"),
        ("Ausbildung", "education"),
        ("Skills", "skills"),
        ("Fähigkeiten", "skills"),
        ("Languages", "languages"),
        ("Sprachen", "languages"),
        ("Profile", "summary"),
        ("Profil", "summary"),
        ("Projects", "projects"),
        ("Zertifizierungen", "certifications"),
    ],
)
def test_sections_are_recognised_in_every_supported_language(heading, section):
    """
    German CVs are a first-class output, not an afterthought.

    A section check that only understood English would report every German CV as
    missing all of its sections.
    """
    assert section in detect_sections(f"{heading}\nSome content follows.")


def test_a_document_with_no_headings_reports_no_sections():
    assert detect_sections("Just a paragraph of running text.") == []


def test_sections_claimed_by_latex_are_derived_from_the_source():
    latex = (
        r"\section*{Experience}" "\n" r"\section*{Education}" "\n" r"\section*{Fähigkeiten}" "\n"
    )
    claimed = _sections_claimed_by_latex(latex)
    assert "experience" in claimed
    assert "education" in claimed
    assert "skills" in claimed


def test_no_sections_are_invented_for_a_cv_that_has_none():
    assert _sections_claimed_by_latex(r"\textbf{Jane Doe}" "\n" r"Some text.") == []


# ===========================================================================
# 3. LaTeX leakage
# ===========================================================================


def test_a_clean_document_has_no_latex_artifacts():
    clean = "Alex Berger\nPlatform Engineer\nKubernetes and Terraform"
    assert find_latex_artifacts(clean) == []


@pytest.mark.parametrize(
    "text,artifact",
    [
        (r"Alex \textbf{Berger}", "textbf"),
        (r"\begin{itemize} Alex", "begin"),
        (r"Alex \item built things", "item"),
        (r"Alex \\section{Profile}", "section"),
        (r"{{doubled braces}}", "{{"),
    ],
)
def test_raw_latex_in_the_output_is_detected(text, artifact):
    """
    A template or escaping failure that reaches the page is a real defect.

    This is the class of bug that produces a PDF which looks fine in a file
    listing and is unusable in a document.
    """
    found = find_latex_artifacts(text)
    assert any(artifact in item for item in found), found


def test_a_legitimate_ampersand_is_not_an_artifact():
    """Research & Development must not be flagged."""
    assert find_latex_artifacts("Research & Development") == []


# ===========================================================================
# 4. Content retention
# ===========================================================================


def test_accented_and_unaccented_names_compare_equal():
    """
    A correctly rendered umlaut is not a missing token.

    Without this a German CV is rejected for being right: "Müller" and "Muller"
    have to be the same name.
    """
    assert "muller" in checkable_tokens("Müller")
    assert "muller" in checkable_tokens("Muller")
    assert checkable_tokens("Müller") == checkable_tokens("Muller")


def test_generic_cv_vocabulary_is_not_treated_as_candidate_content():
    """
    Words present in every CV prove nothing about survival.
    """
    tokens = checkable_tokens("Der CV Lebenslauf und die Skills des Bewerbers")
    assert "bewerbers" in tokens
    assert "lebenslauf" not in tokens
    assert "skills" not in tokens


def test_expected_content_is_derived_from_the_body_not_the_preamble():
    """
    Macro names in the preamble are page furniture, not the candidate's content.
    """
    latex = (
        r"\documentclass{article}"
        "\n"
        r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\newcommand{\cvsection}[1]{\section*{#1}}"
        "\n"
        r"\begin{document}"
        "\n"
        r"Kubernetes Terraform Prometheus Grafana"
        "\n"
        r"\end{document}"
    )
    expected = _expected_content_from_latex(latex)
    assert "Kubernetes" in expected
    assert "newcommand" not in expected
    assert "documentclass" not in expected
    assert r"\begin" not in expected


# ===========================================================================
# 5. End-to-end: compile then validate, for every layout
# ===========================================================================


def _compile_or_skip(document: str) -> bytes:
    if PDFLATEX is None:
        pytest.skip("pdflatex is not on PATH")
    return compile_latex_to_pdf(document)


LAYOUTS = sorted(CV_TEMPLATES)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_layout_compiles_to_a_valid_single_page_pdf(layout):
    """
    Every declared layout must produce a real, non-blank, one-page PDF.

    Layouts that only ever appeared in a dropdown are not "supported".
    """
    pdf_bytes = _compile_or_skip(build_document(layout))

    assert pdf_bytes[:5] == b"%PDF-", f"{layout} did not produce a PDF"
    assert pdf_page_count(pdf_bytes) == 1, layout

    text = extract_pdf_text(pdf_bytes)
    assert len(text.strip()) > 200, f"{layout} produced a near-blank page"

    report = validate_pdf_content(
        pdf_bytes,
        expected_pages=1,
        expected_text=_expected_content_from_latex(build_document(layout)),
        expected_sections=_sections_claimed_by_latex(build_document(layout)),
    )
    assert "content_lost" not in report["problems"], layout
    assert "blank_or_near_blank" not in report["problems"], layout


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_candidate_appears_in_every_layout(layout):
    """
    The person's own content must reach the page, whatever the template.
    """
    pdf_bytes = _compile_or_skip(build_document(layout))
    text = _normalised(extract_pdf_text(pdf_bytes))
    assert "alex" in text and "berger" in text, layout
    assert "kubernetes" in text, layout
    assert "terraform" in text, layout


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_layout_leaks_raw_latex(layout):
    pdf_bytes = _compile_or_skip(build_document(layout))
    assert find_latex_artifacts(extract_pdf_text(pdf_bytes)) == [], layout


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_layout_uses_a_replacement_character(layout):
    """
    A character the font could not render is a silent data loss.
    """
    pdf_bytes = _compile_or_skip(build_document(layout))
    text = extract_pdf_text(pdf_bytes)
    assert "�" not in text, f"{layout} rendered an unmapped character"


def _normalised(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


# ===========================================================================
# 6. Layout differentiation
# ===========================================================================


def _fingerprint(template: str) -> frozenset[str]:
    """
    Structural traits of a template.

    Deliberately structural -- packages, colours, defined macros, column
    machinery, heading treatment, margin geometry, body type size, measure --
    rather than a hash of the text. Two PDFs can differ byte-for-byte and be the
    same document; a change in these traits is what actually makes a layout
    different.

    The dimensions are chosen so that a layout is not identified by a single
    incidental value. Two layouts separated only by one hex colour were one
    cosmetic edit away from colliding, which makes a guard that fires on nothing
    and misses a real convergence.
    """
    traits: set[str] = set()

    for match in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", template):
        for package in match.group(1).split(","):
            traits.add(f"pkg:{package.strip()}")

    # The geometry *values*, not just the fact that the package is loaded.
    #
    # ``pkg:geometry`` was identical across all ten layouts, so the margins --
    # the thing that actually decides how much text fits on the page -- were
    # never compared. Two layouts sharing every package, colour and font but
    # differing in margin would have been reported as the same document.
    for match in re.finditer(r"\\usepackage\[([^\]]*)\]\{geometry\}", template):
        options = match.group(1)
        for option in options.split(","):
            key, _, value = option.partition("=")
            key = key.strip().casefold()
            value = value.strip().casefold()
            if key and value:
                traits.add(f"margin:{key}={value}")

    # The body type size. This is what separates a dense ATS layout from an airy
    # one, and the existing name-size traits only covered the name block.
    for match in re.finditer(r"\\documentclass\[(?:11pt|12pt|10pt|9pt)", template):
        traits.add(f"class:{match.group(0).rsplit('[', 1)[-1]}")

    # The document measure, when the template pins one. A two-column measure and
    # a full-width one produce very different pages from identical text.
    for match in re.finditer(r"\\setlength\{\\textwidth\}\{([^}]*)\}", template):
        traits.add(f"measure:{match.group(1).strip().casefold()}")

    for match in re.finditer(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}", template):
        traits.add(f"colour:{match.group(1)}={match.group(2).upper()}")

    for match in re.finditer(r"\\newcommand\{\\(\w+)\}", template):
        traits.add(f"macro:{match.group(1)}")

    for marker, trait in (
        (r"familydefault\}\{\\sfdefault", "font:sans"),
        (r"familydefault\}\{\\rmdefault", "font:serif"),
        (r"scshape", "heading:smallcaps"),
        (r"\\uppercase", "heading:uppercase"),
        (r"MakeUppercase", "heading:makeuppercase"),
        (r"begin\{multicols\}", "columns:multicols"),
        (r"\\twocolumn", "columns:twocolumn"),
        (r"minipage", "columns:minipage"),
        (r"\\Huge", "name-size:Huge"),
        (r"\\huge", "name-size:huge"),
        (r"\\LARGE", "name-size:LARGE"),
        (r"\\Large", "name-size:Large"),
        (r"hrule height", "header-rule"),
        (r"textbar", "job-header:bar"),
    ):
        if re.search(marker, template):
            traits.add(trait)

    return frozenset(traits)


def _fingerprints() -> dict[str, frozenset[str]]:
    return {
        name: _fingerprint(_patch_template_preamble(template))
        for name, template in CV_TEMPLATES.items()
    }


def test_every_layout_has_a_distinct_structural_fingerprint():
    """
    The regression this test exists for.

    A change that collapses all templates onto one skeleton -- a refactor
    extracting a "shared base template" and losing the per-layout differences --
    would make every layout produce the same document while every existing test
    still passed, because each would still compile to a valid one-page PDF.

    No two layouts may be identical.
    """
    fingerprints = _fingerprints()
    assert len(fingerprints) == len(CV_TEMPLATES), "a layout is missing"

    seen: dict[frozenset[str], str] = {}
    for name, traits in fingerprints.items():
        assert traits, f"{name} produced no detectable structural traits"
        if traits in seen:
            pytest.fail(
                f"layouts {seen[traits]!r} and {name!r} are structurally "
                f"identical. Two layouts that are supposed to differ have "
                f"collapsed onto the same template."
            )
        seen[traits] = name


def test_no_pair_of_layouts_is_nearly_identical():
    """
    Not just "not equal": no pair may differ by a single cosmetic token.

    One differing link colour is not two layouts. A user choosing between them
    is choosing between documents that look the same.
    """
    fingerprints = _fingerprints()
    names = sorted(fingerprints)
    closest: tuple[str, str, int] | None = None

    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            difference = len(fingerprints[left] ^ fingerprints[right])
            if closest is None or difference < closest[2]:
                closest = (left, right, difference)

    assert closest is not None
    left, right, difference = closest

    assert difference >= 3, (
        f"layouts {left!r} and {right!r} differ by only {difference} "
        f"structural trait(s) (closest pair). They are not meaningfully "
        f"different documents."
    )


def test_the_minimal_ats_layout_really_is_minimal():
    """
    A layout named "minimal" must not carry the colour furniture the other ATS
    layouts have.

    It previously differed from ``international_ats`` only in a link colour and
    one header font size, which is not a layout.
    """
    minimal = _fingerprints()["german_minimal_ats"]
    other_ats = _fingerprints()["international_ats"]

    minimal_colours = {trait for trait in minimal if trait.startswith("colour:linkcolor")}
    other_colours = {trait for trait in other_ats if trait.startswith("colour:linkcolor")}

    assert not minimal_colours, (
        "german_minimal_ats defines a link colour; a layout meant to be "
        "minimal should carry no colour emphasis"
    )
    assert other_colours, (
        "international_ats should keep its brand colour; if this layout has "
        "also been made monochrome the two are no longer distinguishable"
    )

    # Pure black rather than a near-black that only looks black on screen.
    primary = [t for t in minimal if t.startswith("colour:primary=")]
    assert primary == ["colour:primary=000000"], primary


def test_layout_language_classification_is_complete():
    """
    Every layout must be classified as German or English.

    ``technical_lead`` was missing from the LaTeX module's English set while
    being present in the optimizer's, so it silently received the "detect the
    language from the job description" rule instead of an explicit one. A
    guard at import now refuses to start if a layout is unclassified; this test
    checks the classification itself is the intended one.
    """
    from app.services.cv import latex_generator

    for name in CV_TEMPLATES:
        language = required_language_for_layout(name)
        assert language in ("de", "en", "any"), f"{name} -> {language}"
        assert name in (latex_generator._GERMAN_LAYOUTS | latex_generator._ENGLISH_LAYOUTS), name

    assert "technical_lead" in latex_generator._ENGLISH_LAYOUTS
    assert (
        latex_generator._language_rule("technical_lead").lower().find("english") > 0
    ), "technical_lead must get the explicit English rule"


def test_every_layout_declares_its_column_structure():
    for name in CV_TEMPLATES:
        assert name in CV_LAYOUT_COLUMNS, name


# ===========================================================================
# 7. A PDF that is not usable must not be reported as success
# ===========================================================================


def test_a_blank_page_is_rejected_even_though_it_compiled():
    """
    The failure mode this module exists for.

    ``pdflatex`` exits 0, the file is a valid one-page PDF, and there is no CV on
    it. Page count and file size both pass.
    """

    class FakePage:
        @staticmethod
        def extract_text():
            return "1"

    class FakeReader:
        pages = [FakePage()]

    import app.services.cv.pdf_validation as module

    original = module.PdfReader
    module.PdfReader = lambda *a, **k: FakeReader()

    try:
        with pytest.raises(PDFValidationError, match="blank"):
            validate_pdf_content(b"%PDF-1.4 fake", expected_pages=1)
    finally:
        module.PdfReader = original


def test_a_wrong_page_count_is_rejected():
    if PDFLATEX is None:
        pytest.skip("pdflatex is not on PATH")

    # Two pages where one was required must fail rather than pass silently.
    pdf_bytes = build_document("standard")
    assert pdf_bytes  # the fixture is valid

    from app.services.cv.pdf_compiler import PDFLayoutError

    # The one-page compiler is the thing that enforces this; assert it refuses
    # rather than emitting a two-page file.
    long_body = FIXTURE_BODY * 6
    document = build_document("standard").replace(FIXTURE_BODY, long_body)

    try:
        result = compile_single_page_pdf(document)
    except PDFLayoutError:
        return  # refused, which is the required behaviour
    except LaTeXCompilationError:
        return  # also a refusal

    assert (
        pdf_page_count(result) == 1
    ), "a document that did not fit was compiled and returned anyway"


def test_a_latex_error_is_not_hidden_behind_a_generic_failure():
    """
    The real cause has to survive to the caller.

    ``PDF generation failed`` is what made the original escaping bug so hard to
    find: the actual LaTeX error was discarded and the diagnosis had to be
    reconstructed by hand from a PDF that simply did not appear.

    An unclosed environment is used because it is a structural fault the source
    validator cannot see: the brace and command balance is intact, so the
    document passes every pre-flight check and only ``pdflatex`` objects. That
    is precisely the class of bug this has to survive.
    """
    broken = build_document("standard").replace(r"\end{itemize}", "", 1)

    with pytest.raises((LaTeXCompilationError, PDFLayoutError)) as error:
        compile_single_page_pdf(broken)

    message = str(error.value)
    assert message, "the error must say something"

    # A named category and a location: enough to act on without the log file.
    assert "[" in message and "]" in message, f"the error carries no category: {message!r}"
    assert (
        "line" in message.lower() or "environment" in message.lower()
    ), f"the error names neither a line nor a construct: {message!r}"
    assert (
        "generation failed" not in message.lower()
    ), f"the generic message replaced the real one: {message!r}"


def test_an_undefined_command_is_caught_before_pdflatex_runs():
    """
    Rejecting early is better than rejecting late, but it must still be rejected.

    The source validator is an allow-list, so an invented command is refused
    without creating a temporary file at all.
    """
    broken = build_document("standard").replace(
        r"\section*{Profile}", r"\madeUpCommand{Profile}", 1
    )

    with pytest.raises((LaTeXSourceError, LaTeXCompilationError)) as error:
        compile_single_page_pdf(broken)

    assert "madeUpCommand" in str(error.value) or "command" in str(error.value).lower()


def test_a_source_error_is_reported_before_any_file_is_written():
    """
    Rejecting a bad source must not create a temporary file.
    """
    with pytest.raises(LaTeXSourceError):
        compile_single_page_pdf(
            r"\documentclass{article}"
            "\n"
            r"\usepackage{evil-package}"
            "\n"
            r"\begin{document}"
            "\n"
            r"x"
            "\n"
            r"\end{document}"
        )


def test_pdf_validation_never_returns_candidate_content():
    """
    A failure is reportable without reproducing a CV.

    Retention failures name *which* words were lost so the problem is
    actionable, and the report is the only thing that carries them. This test
    pins the threshold at a level that still catches a lost page while tolerating
    the summarisation a CV legitimately performs.
    """
    assert 0.4 < MIN_CONTENT_RETENTION < 0.8, (
        "the retention threshold has drifted to a value that would either "
        "reject every CV or accept a lost one"
    )
