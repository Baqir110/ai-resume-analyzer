"""Page-tree validation and bounded formatting retries, without a TeX dependency."""

import inspect
import io
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from app.services.career import cover_letter_pdf
from app.services.cv import latex_generator, pdf_compiler
from app.services.cv.latex_generator import (
    FactualValidationError,
    _escape_latex_text,
    clean_body_for_latex,
    generate_german_latex_content,
)

SOURCE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\begin{document}
{\Huge Jane Doe}\\[4pt]
\vspace{1em}
All original experience remains here.
\end{document}
"""


def _pdf(pages):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595.28, height=841.89)
    writer.add_metadata({"/Subject": "/Type /Page is not a page object"})
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("pages", [0, 1, 2, 4])
def test_counts_actual_page_tree(pages):
    assert pdf_compiler.pdf_page_count(_pdf(pages)) == pages


def test_invalid_pdf_is_not_assumed_to_be_one_page():
    with pytest.raises(RuntimeError, match="unreadable"):
        pdf_compiler.pdf_page_count(b"not a PDF")


def _mock_compiler(monkeypatch, page_counts, log=""):
    calls = []
    monkeypatch.setattr(pdf_compiler.shutil, "which", lambda _: "/usr/bin/pdflatex")

    def run(command, **kwargs):
        index = min(len(calls) // 2, len(page_counts) - 1)
        calls.append(command)
        path = Path(kwargs["cwd"])
        (path / "document.pdf").write_bytes(_pdf(page_counts[index]))
        (path / "document.log").write_text(log)
        assert "-no-shell-escape" in command
        assert kwargs["shell"] is False
        assert 0 < kwargs["timeout"] <= pdf_compiler.PDFL_COMPILER_PASS_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pdf_compiler.subprocess, "run", run)
    return calls


def test_compact_retry_fits_without_discarding_text(monkeypatch):
    # Two pages uncompacted, one page after the first compaction level. The
    # escalation must stop there rather than pressing on to the most
    # aggressive level.
    calls = _mock_compiler(monkeypatch, [2, 1, 1, 1])
    assert pdf_compiler.pdf_page_count(pdf_compiler.compile_single_page_pdf(SOURCE)) == 1

    # One attempt refused, one attempt accepted: two LaTeX passes each.
    assert len(calls) == 4, (
        f"expected two attempts before the document fitted, got " f"{len(calls) // 2}"
    )

    compact = pdf_compiler._compact_layout(SOURCE)
    assert "All original experience remains here." in compact
    assert r"\documentclass[11pt,a4paper]" in compact


def test_rejects_remaining_extra_pages(monkeypatch):
    # Never fits, at any level. Every level must be tried before giving up.
    calls = _mock_compiler(monkeypatch, [2, 2, 2, 2])
    with pytest.raises(pdf_compiler.PDFLayoutError) as error:
        pdf_compiler.compile_single_page_pdf(SOURCE)
    assert error.value.pages == 2

    # Four levels, two LaTeX passes each: the full escalation was attempted.
    assert len(calls) == 8, (
        f"the full escalation should be four attempts, got {len(calls) // 2}. "
        f"A document that never fits must be given every level before it is "
        f"refused."
    )


def test_compaction_escalates_rather_than_skipping_to_the_hardest_level(
    monkeypatch,
):
    """
    The regression this suite exists to prevent.

    The loop used to run over ``range(2)`` and call
    ``_compact_layout_level(source, 3)`` on its second iteration, so levels 1
    and 2 were unreachable and every compacted document was delivered at 10pt
    with 0.5cm margins -- even when level 1 was sufficient.
    """
    seen: list[str] = []

    real = pdf_compiler._compact_layout_level

    def recording(source, level):
        seen.append(level)
        return real(source, level)

    monkeypatch.setattr(pdf_compiler, "_compact_layout_level", recording)

    # Fits at the first compaction level.
    _mock_compiler(monkeypatch, [2, 1, 1, 1])
    pdf_compiler.compile_single_page_pdf(SOURCE)

    assert seen == [1], (
        f"expected compaction level 1 only, got {seen}. The escalation is "
        f"still applying its most aggressive level first."
    )


def test_rejects_overflow_even_on_one_page(monkeypatch):
    _mock_compiler(monkeypatch, [1, 1], r"Overfull \vbox (25.0pt too high) detected")
    with pytest.raises(pdf_compiler.PDFLayoutError, match="outside"):
        pdf_compiler.compile_single_page_pdf(SOURCE)


def test_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(pdf_compiler.shutil, "which", lambda _: "/usr/bin/pdflatex")

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(pdf_compiler.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        pdf_compiler.compile_single_page_pdf(SOURCE)


def test_escaping_does_not_reescape_generated_braces():
    assert _escape_latex_text("\\{x} & 20% ~ ^") == (
        r"\textbackslash{}\{x\} \& 20\% \textasciitilde{} \textasciicircum{}"
    )


@pytest.mark.parametrize(
    "body",
    [
        r"\input{private.txt}",
        r"\immediate\write18{command}",
        r"\openout\stream=private.txt",
        "\\in% split\nput{private.txt}",
        r"\begin{filecontents*}{private.txt}secret\end{filecontents}",
        r"\str_new:n \l_tmpa {private.txt}",
        r"\href{javascript:alert(1)}{unsafe link}",
    ],
)
def test_body_validation_rejects_file_and_execution_primitives(body):
    with pytest.raises(pdf_compiler.LaTeXSourceError):
        pdf_compiler.validate_latex_body(body)


def test_document_validation_uses_a_package_allowlist():
    source = SOURCE.replace(
        r"\begin{document}",
        r"\usepackage{shellescape}" "\n" r"\begin{document}",
    )
    with pytest.raises(pdf_compiler.LaTeXSourceError, match="package"):
        pdf_compiler.validate_latex_document(source)


def test_compile_validates_before_looking_for_an_executable(monkeypatch):
    monkeypatch.setattr(
        pdf_compiler.shutil,
        "which",
        lambda _name: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    with pytest.raises(pdf_compiler.LaTeXSourceError):
        pdf_compiler.compile_single_page_pdf(SOURCE.replace("All original", r"\input{secret}"))
    with pytest.raises(pdf_compiler.LaTeXSourceError):
        pdf_compiler.compile_single_page_pdf("x" * (pdf_compiler.MAX_LATEX_SOURCE_CHARS + 1))


def test_compiler_uses_private_temp_and_minimal_environment(monkeypatch):
    monkeypatch.setenv("PDF_TEST_PRIVATE_TOKEN", "must-not-be-inherited")
    monkeypatch.setattr(pdf_compiler.shutil, "which", lambda _name: "/usr/bin/pdflatex")

    captured = {}

    # The fake must still create a valid PDF for structural validation.
    def run_with_pdf(command, **kwargs):
        captured["cwd"] = Path(kwargs["cwd"])
        captured["env"] = kwargs["env"]
        (captured["cwd"] / "document.pdf").write_bytes(_pdf(1))
        (captured["cwd"] / "document.log").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pdf_compiler.subprocess, "run", run_with_pdf)
    assert pdf_compiler.compile_single_page_pdf(SOURCE).startswith(b"%PDF")

    assert captured["cwd"].name.startswith("resume-pdf-")
    assert captured["cwd"].parent == Path(tempfile.gettempdir()).resolve()
    assert "PDF_TEST_PRIVATE_TOKEN" not in captured["env"]
    assert captured["env"]["shell_escape"] == "f"
    assert captured["env"]["openin_any"] == "p"
    assert captured["env"]["openout_any"] == "p"


def test_compiler_failure_does_not_return_logs_or_source(monkeypatch):
    sentinel = "PRIVATE_COMPILER_DIAGNOSTIC"
    monkeypatch.setattr(pdf_compiler.shutil, "which", lambda _name: "/usr/bin/pdflatex")

    def fail(command, **kwargs):
        (Path(kwargs["cwd"]) / "document.log").write_text(sentinel)
        kwargs["stdout"].write(sentinel)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pdf_compiler.subprocess, "run", fail)
    with pytest.raises(pdf_compiler.LaTeXCompilationError) as error:
        pdf_compiler.compile_single_page_pdf(SOURCE.replace("All original", sentinel))

    # The candidate's CV text must not survive into the error. This log holds
    # nothing but the sentinel, so there is no real LaTeX error to name and the
    # compiler-unavailable path is taken instead.
    assert sentinel not in str(error.value)
    assert error.value.category == "no_diagnostic"
    assert "produced no diagnostic" in str(error.value)
    assert str(error.value) != "LaTeX compilation failed."


def test_compiler_failure_reports_the_real_latex_error(monkeypatch, tmp_path):
    """
    A pdflatex failure must name the category, line and offending character.

    This is the regression guard for the original bug: an unescaped "&" in
    generated CV text reached pdflatex and surfaced only as
    "pdflatex failed safely (exit_code=1)" plus "LaTeX compilation failed.".
    """
    monkeypatch.setattr(pdf_compiler.shutil, "which", lambda _name: "/usr/bin/pdflatex")
    source = SOURCE.replace(
        "All original content is preserved.",
        "Sales & Marketing grew 40% in 2024.",
    )
    log = (
        "This is pdfTeX\n"
        "(./document.tex:14: Misplaced alignment tab character &.\n"
        "l.14 Sales &\n"
        "        Marketing grew 40% in 2024.\n"
        "! Emergency stop.\n"
    )

    def fail(command, **kwargs):
        (Path(kwargs["cwd"]) / "document.log").write_text(log, encoding="utf-8")
        kwargs["stdout"].write(log)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pdf_compiler.subprocess, "run", fail)
    with pytest.raises(pdf_compiler.LaTeXCompilationError) as error:
        pdf_compiler.compile_single_page_pdf(source)

    exc = error.value
    assert exc.category == "unescaped_alignment"
    assert exc.line == 14
    assert exc.token == "&"
    assert "unescaped_alignment" in str(exc)
    assert "line 14" in str(exc)
    # The body line itself is the candidate's CV and must not be echoed back.
    assert "Marketing" not in str(exc)
    assert "&" in str(exc)


def test_compiler_output_is_bounded(monkeypatch):
    monkeypatch.setattr(pdf_compiler, "MAX_TEX_ARTIFACT_BYTES", 16)
    monkeypatch.setattr(pdf_compiler.shutil, "which", lambda _name: "/usr/bin/pdflatex")

    def noisy(command, **kwargs):
        kwargs["stdout"].write("x" * 32)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pdf_compiler.subprocess, "run", noisy)
    with pytest.raises(pdf_compiler.LaTeXCompilationError, match="size limit"):
        pdf_compiler.compile_single_page_pdf(SOURCE)


def test_generated_url_arguments_are_not_mangled():
    body = clean_body_for_latex(
        r"\projheader{Repository}{Python}{https://example.test/a_b?x=1&y=2}"
    )
    assert r"https://example.test/a_b?x=1&y=2" in body
    assert r"a\_b" not in body
    assert pdf_compiler.sanitize_latex_url("https://example.test/a%20b#section") == (
        "https://example.test/a%2520b%23section"
    )


def test_generation_has_no_candidate_specific_url_defaults():
    signature = inspect.signature(generate_german_latex_content)
    assert signature.parameters["linkedin_url"].default is None
    assert signature.parameters["github_url"].default is None


def test_generation_does_not_invent_a_missing_candidate_header(monkeypatch):
    monkeypatch.setattr(
        latex_generator.LLMService,
        "generate",
        lambda *_args, **_kwargs: pytest.fail("LLM must not be called"),
    )
    with pytest.raises(FactualValidationError):
        generate_german_latex_content("12345\n67890", "Platform role", [])


def test_cover_letter_does_not_invent_a_missing_candidate_header(monkeypatch):
    monkeypatch.setattr(
        "app.services.llm.provider.LLMService.generate",
        lambda *_args, **_kwargs: pytest.fail("LLM must not be called"),
    )
    with pytest.raises(FactualValidationError):
        cover_letter_pdf.generate_cover_letter_latex("12345\n67890", "Platform role")


def test_cover_letter_compiler_uses_shared_validation(monkeypatch):
    monkeypatch.setattr(
        cover_letter_pdf,
        "compile_single_page_pdf",
        lambda source: source.encode("utf-8"),
    )
    assert cover_letter_pdf.compile_cover_letter_pdf(SOURCE).startswith(b"\\documentclass")
    with pytest.raises(pdf_compiler.LaTeXSourceError):
        cover_letter_pdf.compile_cover_letter_pdf(SOURCE.replace("All original", r"\write18{x}"))
