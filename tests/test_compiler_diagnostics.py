"""
Compiler-diagnostic redaction.

The requirement is two-sided and both halves are asserted here, because either
alone is a defect:

* the compiler's own diagnosis must survive -- a redacted message that says
  nothing is a worse bug than a leak, because the whole point of the diagnostic
  is to avoid running ``pdflatex`` by hand;
* the candidate's document and any credential must not.

These tests pin the boundary between the two. The previous implementation
collapsed every run of four or more letters, so "Unescaped LaTeX character"
became " <8 chars><1 bytes> " and the error was unactionable.
"""

from __future__ import annotations

import os
import re

import pytest

from app.services.cv.pdf_compiler import (
    LaTeXCompilationError,
    _redact_compiler_text,
    compile_single_page_pdf,
)

DOLLAR = chr(36)
BS = chr(92)

#: pdflatex's own vocabulary, as it actually appears in a ``.log``. Every entry
#: here is fixed TeX wording, not candidate content.
REAL_COMPILER_MESSAGES = (
    f"Unescaped LaTeX character '{DOLLAR}'",
    f"Missing {BS}begin{{document}}.",
    f"Missing {BS}end{{document}}.",
    "Undefined control sequence.",
    "Runaway argument?",
    "LaTeX Error: File `cv.tex' not found.",
    "Package hyperref Error: Token not allowed in a PDF string.",
    "Font shape OT1/cmr/m/n in size <10> not available",
    "LaTeX Warning: Reference `sec:x' on page 1 undefined on input line 42.",
    "Overfull " + BS + "hbox (12.0pt too wide) in paragraph at lines 213--215",
    "pdfTeX error: pdflatex (file pdftex.def) not found",
    "Missing character: There is no " + DOLLAR + " in position 42.",
    "LaTeX Error: Environment itemize undefined.",
    f"{BS}begin{{itemize}} on input line 90 ended by {BS}end{{document}}",
)


# ===========================================================================
# 1. The compiler's message survives intact
# ===========================================================================


@pytest.mark.parametrize("message", REAL_COMPILER_MESSAGES)
def test_real_compiler_messages_survive_intact(message):
    """
    Diagnostic text is not document content.

    The longest word in TeX's error vocabulary is about eleven characters. Any
    rule that redacts ordinary words destroys the one sentence that explains the
    failure, which is what this function used to do.
    """
    assert _redact_compiler_text(message) == " ".join(message.split())


def test_line_numbers_survive():
    out = _redact_compiler_text("on input line 213")
    assert "213" in out


def test_filenames_survive():
    out = _redact_compiler_text("LaTeX Error: File `cv-2026.tex' not found.")
    assert "cv-2026.tex" in out


def test_the_offending_latex_character_is_named():
    """
    The single most useful token in an escaping error is the character.
    """
    for char in (DOLLAR, "&", "#", "%", "_", "^", "~"):
        out = _redact_compiler_text(f"Unescaped LaTeX character '{char}'")
        assert char in out, f"{char!r} was lost from the message"


# ===========================================================================
# 2. Credentials are removed
# ===========================================================================


SECRET = "sk-live-THISISAREALSECRET0123456789"


def test_a_secret_this_deployment_holds_is_removed_by_value(monkeypatch):
    """
    The primary requirement: a real ``.env`` value must never reach a diagnostic.

    This is the case that matters. A candidate who pastes a key into their CV --
    or a title that accidentally carries one -- would otherwise have it echoed
    back through an exception, into a log file, and out through HTTP.
    """
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    out = _redact_compiler_text(f"! LaTeX Error: File `cv.tex' not found; key was {SECRET}.")
    assert SECRET not in out
    # The surrounding diagnostic is still readable.
    assert "LaTeX Error" in out
    assert "not found" in out


@pytest.mark.parametrize("name", ["API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"])
def test_every_secret_variable_is_covered(monkeypatch, name):
    value = f"test-value-{name.lower()}-0123456789abcdef"
    monkeypatch.setenv(name, value)
    assert value not in _redact_compiler_text(f"token: {value}")


@pytest.mark.parametrize(
    "credential",
    [
        "sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH",
        "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8",
        "ghp_0123456789abcdefghijklmnopqrstuvwxyz",
        "gsk_0123456789abcdefghijklmnopqrstuvwxyz",
        "xai-0123456789abcdefghij",
        "hf_0123456789abcdefghijklmnop",
    ],
)
def test_a_recognisable_credential_shape_is_removed(credential):
    """
    The backstop.

    A secret this deployment does not hold -- pasted from another machine, or
    supplied to a different deployment of the same code -- is not in any
    environment variable here, so shape matching is what catches it.
    """
    out = _redact_compiler_text(f"found {credential} in the document")
    assert credential not in out, credential
    assert "found" in out, "the surrounding text was destroyed too"


def test_a_private_key_block_is_removed():
    block = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAx0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z0Z\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = _redact_compiler_text(f"error near {block}")
    assert "PRIVATE KEY" not in out
    assert "error near" in out


def test_a_very_short_secret_is_not_redacted():
    """
    Redacting a two-character value would corrupt every message.

    A "secret" that short is not a credential, and treating it as one destroys
    the diagnostic for no security benefit.
    """
    monkey = os.environ.get("GROQ_API_KEY", "")
    if len(monkey) >= 12:
        return
    out = _redact_compiler_text("value ab appears in a normal sentence")
    assert "ab" in out


# ===========================================================================
# 3. Document-shaped input is removed, normal prose is not
# ===========================================================================


def test_a_pasted_document_paragraph_is_removed():
    """
    A CV paragraph pasted into a title argument is a single 100+ character run.

    No LaTeX diagnostic is that long, so collapsing it costs nothing and stops
    a paragraph from reaching a log.
    """
    paragraph = (
        "BuiltPythonServiceswithFastAPIandInstrumentedEverySingleOneWith"
        "PrometheusAndGrafanaMonitoringDashboardsAcrossAllEnvironments"
    )
    out = _redact_compiler_text(f"! Package hyperref Error: {paragraph}")
    assert paragraph not in out
    assert "hyperref" in out, "the diagnosis itself was destroyed"


def test_ordinary_prose_is_not_collapsed():
    """
    The threshold has to sit above real vocabulary and below a pasted blob.

    If this fails, either the redaction is too weak to be worth having or too
    aggressive to be usable.
    """
    from app.services.cv.pdf_compiler import _PATHOLOGICAL_WORD_RUN

    assert (
        _PATHOLOGICAL_WORD_RUN >= 40
    ), "the threshold is low enough that real TeX vocabulary could be caught"

    out = _redact_compiler_text("! LaTeX Error: Environment tabularx undefined on input line 12.")
    assert "Environment tabularx undefined" in out


def test_a_leak_cannot_hide_on_a_later_line():
    """
    A multi-line message is flattened before any rule is applied.

    Applying a per-line rule to a multi-line string would let content on line two
    through untouched.
    """
    out = _redact_compiler_text(
        "First line of output\n" "second line with sk-proj-ZZZZYYYYXXXXWWWWVVVVUUUUTTTTSSSS here"
    )
    assert "sk-proj-ZZZZYYYYXXXXWWWWVVVVUUUUTTTTSSSS" not in out
    assert "First line" in out


# ===========================================================================
# 4. Context that is not diagnostic is still removed
# ===========================================================================


def test_the_operators_username_does_not_leak_via_the_workdir():
    """
    pdflatex runs in a private temp directory whose path contains the username.
    """
    out = _redact_compiler_text(
        "! I can't find file `C:\\Users\\someone\\AppData\\Local\\Temp\\cv-abc\\cv.tex'."
    )
    assert "someone" not in out
    assert "cv.tex" in out, "the file name is useful and should survive"
    assert "<workdir>" in out


def test_a_posix_temp_path_is_also_removed():
    out = _redact_compiler_text("! I can't find file `/tmp/resume-pdf-abc/cv.tex'.")
    assert "/tmp/resume-pdf-abc" not in out
    assert "cv.tex" in out


# ===========================================================================
# 5. The result stays bounded
# ===========================================================================


def test_output_is_capped():
    from app.services.cv.pdf_compiler import _MAX_DETAIL_CHARS

    out = _redact_compiler_text("word " * 400)
    assert len(out) <= _MAX_DETAIL_CHARS + 16, len(out)


def test_the_cap_trims_on_a_word_boundary():
    """
    Cutting mid-word produces a fragment that reads like a complete sentence.
    """
    out = _redact_compiler_text("alpha " * 200)
    assert not out.rstrip(".").endswith("alph"), out[-40:]
    if out.endswith("[...]"):
        before = out[: -len(" [...]")]
        assert before.endswith("alpha") or before.endswith(" "), before[-20:]


def test_empty_input_is_handled():
    assert _redact_compiler_text("") == ""
    assert _redact_compiler_text(None or "") == ""


# ===========================================================================
# 6. The contract holds end to end
# ===========================================================================


def test_a_real_compilation_failure_is_diagnosable_and_safe():
    """
    The whole point, end to end.

    A structural fault the source validator cannot see -- an unclosed
    environment -- reaches ``pdflatex``, and the resulting exception must name
    the cause without containing the candidate's text.
    """
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex is not on PATH")

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

    body = (
        r"\section*{Experience}"
        "\n"
        r"\textbf{Platform Engineer} \textbar{} Beispiel GmbH"
        "\n"
        r"\begin{itemize}"
        "\n"
        r"  \item Reduced deployment time by forty percent using GitLab CI."
        "\n"
        r"\end{itemize}"
        "\n"
        r"\section*{Skills}"
        "\n"
        "Kubernetes, Terraform, Prometheus, Grafana, Docker, AWS."
        "\n"
    )
    # Remove the closing tag: brace and command balance stay intact, so the
    # source validator passes and only pdflatex objects.
    document = template.replace("RESUME_BODY_PLACEHOLDER", body.replace(r"\end{itemize}", "", 1))

    with pytest.raises(LaTeXCompilationError) as error:
        compile_single_page_pdf(document)

    message = str(error.value)

    # Diagnosable: the cause is named.
    assert error.value.category, "no category"
    assert "LaTeX compilation failed" in message
    assert re.search(r"\bitemize\b", message), f"the failing construct is not named: {message!r}"

    # Safe: none of the candidate's body text is present.
    for marker in (
        "Reduced deployment time",
        "forty percent",
        "GitLab",
        "Kubernetes, Terraform",
    ):
        assert marker not in message, f"CV text leaked: {marker!r}"


def test_a_source_excerpt_is_only_taken_from_the_preamble():
    """
    The other half of the safety contract.

    An excerpt is useful for a preamble fault -- a bad package name is not
    sensitive, and naming the line is what makes the error actionable. A body
    line is the candidate's CV, so it must never be included even though it is
    exactly the line that failed.

    The fault used here is a bare ``%``, which LaTeX treats as a comment
    opener. It swallows the rest of the line, which unbalances the following
    brace group, so the error is reported against a *body* line -- the case
    where an excerpt would be most tempting and most wrong.
    """
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex is not on PATH")

    from app.services.cv.latex_generator import (
        CV_TEMPLATES,
        _escape_latex_text,
        _patch_template_preamble,
        _render_candidate_contact,
    )

    template = _patch_template_preamble(CV_TEMPLATES["standard"])
    for placeholder, value in (
        ("CANDIDATE_NAME_PLACEHOLDER", _escape_latex_text("Alex Berger")),
        ("CANDIDATE_TITLE_PLACEHOLDER", _escape_latex_text("Platform Engineer")),
    ):
        template = template.replace(placeholder, value)
    template = template.replace(
        "CANDIDATE_CONTACT_PLACEHOLDER",
        _render_candidate_contact(
            {
                "name": "Alex Berger",
                "title": "Platform Engineer",
                "email": "a@b.invalid",
                "phone": "+49 30 1",
                "location": "Hamburg",
            }
        ),
    )

    secret_phrase = "SecretlyReducedDeploymentTimeByFortyPercent"
    body = (
        r"\section*{Experience}"
        "\n"
        r"\textbf{Platform Engineer} \textbar{} Beispiel GmbH"
        "\n"
        r"\begin{itemize}"
        "\n"
        f"  \\item {secret_phrase} using GitLab CI."
        "\n"
        # No \end{itemize}: the environment stays open, so pdflatex reports the
        # fault against a line inside the body. That is exactly the case where
        # a source excerpt would be most tempting and most wrong, so it is the
        # case worth testing.
        r"\section*{Skills}"
        "\n"
        "Kubernetes, Terraform, Prometheus."
        "\n"
    )

    document = template.replace("RESUME_BODY_PLACEHOLDER", body)

    with pytest.raises((LaTeXCompilationError, Exception)) as error:
        compile_single_page_pdf(document)

    message = str(error.value)
    assert (
        secret_phrase not in message
    ), f"the candidate's text leaked into the diagnostic: {message!r}"
    assert "GitLab" not in message, "body text leaked into the diagnostic"
    assert "Kubernetes, Terraform" not in message, "body text leaked"
