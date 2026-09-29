"""
A model-supplied document wrapper is discarded wherever it sits.

The application's template owns the preamble and the shell policy -- ``pdflatex``
runs with ``-no-shell-escape``, and nothing a model writes may change that. So a
model that returns a complete document has its wrapper stripped and only its body
kept.

The defect this exists for: the strip only worked when the wrapper was the very
first thing in the response. A model that answered "Here is your CV:" before its
document defeated both the anchored prefix match and the anchored preamble regex,
so ``\\documentclass`` survived into the body, was escaped into literal text, and
appeared in the middle of the compiled PDF as the line
``\\documentclass[11pt,a4paper]{article}``.

``validate_pdf_content`` caught it, so nothing broken shipped -- the request
failed instead. But it failed 2 runs in 8 on ``llama3.2``, where 5 would
otherwise have passed, and the reported reason pointed at the typesetting rather
than at a response-cleaning gap.

The suffix case matters for the same reason: ``_END_DOC_RE`` is anchored to the
end of the string, so a model that added "Let me know if you would like changes."
after its document left a literal ``\\end{document}`` in the body.
"""

from __future__ import annotations

import pytest

from app.services.cv.latex_generator import clean_llm_response_to_latex

BS = chr(92)
FENCE = chr(96) * 3

BODY = (
    BS
    + "section*{Experience}\n"
    + BS
    + "textbf{Platform Engineer} "
    + BS
    + "textbar{} Beispiel GmbH\n"
    + "- Reduced deployment time by forty percent using GitLab CI.\n"
)

#: Any of these surviving into the body means a wrapper leaked.
WRAPPER_TOKENS = ("documentclass", "begin{document}", "end{document}", "usepackage")

#: Substrings that must never appear in a body, whatever the model wrote.
PREAMBLE_COMMANDS = ("usepackage", "documentclass", "geometry", "fontenc", "inputenc")


def _assert_body_only(label: str, response: str) -> str:
    cleaned = clean_llm_response_to_latex(response)

    for token in WRAPPER_TOKENS:
        assert token not in cleaned, f"{label}: {token!r} survived into the body: {cleaned[:120]!r}"

    assert (
        "Platform Engineer" in cleaned
    ), f"{label}: the body was discarded along with the wrapper: {cleaned[:120]!r}"

    return cleaned


def test_a_bare_wrapper_is_stripped():
    _assert_body_only(
        "bare wrapper",
        BS
        + "documentclass[11pt,a4paper]{article}\n"
        + BS
        + "begin{document}\n"
        + BODY
        + BS
        + "end{document}",
    )


def test_a_conversational_prefix_does_not_defeat_the_strip():
    """
    The regression. "Here is your CV:" before the document was enough to leave
    \\documentclass in the body.
    """
    _assert_body_only(
        "conversational prefix",
        "Here is your CV:\n\n"
        + BS
        + "documentclass[11pt,a4paper]{article}\n"
        + BS
        + "begin{document}\n"
        + BODY
        + BS
        + "end{document}",
    )


def test_trailing_prose_does_not_leave_a_closing_tag():
    """
    ``_END_DOC_RE`` is anchored to the end of the string, so anything after
    \\end{document} left the tag itself in the body.
    """
    _assert_body_only(
        "trailing prose",
        BS
        + "documentclass[11pt,a4paper]{article}\n"
        + BS
        + "begin{document}\n"
        + BODY
        + BS
        + "end{document}\n\nLet me know if you would like any changes.",
    )


def test_a_preamble_without_an_opening_tag_is_stripped_in_place():
    """
    An incomplete wrapper: the commands are removed wherever they are, so a
    partial answer cannot smuggle \\usepackage or \\geometry into the body.
    """
    cleaned = _assert_body_only(
        "preamble without begin",
        BS
        + "documentclass[11pt]{article}\n"
        + BS
        + "usepackage{helvet}\n"
        + BS
        + "usepackage[T1]{fontenc}\n"
        + BS
        + "usepackage[margin=1cm]{geometry}\n"
        + BODY,
    )

    for command in PREAMBLE_COMMANDS:
        assert command not in cleaned, f"{command!r} survived"


def test_a_fenced_document_is_stripped():
    _assert_body_only(
        "fenced",
        FENCE
        + "latex\n"
        + BS
        + "documentclass[11pt]{article}\n"
        + BS
        + "begin{document}\n"
        + BODY
        + BS
        + "end{document}\n"
        + FENCE,
    )


def test_a_body_with_no_wrapper_is_untouched():
    """
    The common case must not be affected. A model that answers with a body and no
    preamble is doing the right thing, and the strip must leave it alone.
    """
    assert clean_llm_response_to_latex(BODY).strip() == BODY.strip()


def test_a_conversational_refusal_still_produces_nothing():
    """
    Fail-closed. Inventing placeholder prose for a CV the model declined to
    write would be worse than an error.
    """
    for refusal in (
        "I'm ready to help, but I need your resume first.",
        "Sure, please provide the job description.",
    ):
        assert clean_llm_response_to_latex(refusal) == ""


def test_an_empty_response_stays_empty():
    assert clean_llm_response_to_latex("") == ""
    assert clean_llm_response_to_latex("   \n  ") == ""
    assert clean_llm_response_to_latex(None) == ""


def test_a_wrapper_is_stripped_even_when_repeated():
    """
    A model that loops the document must not smuggle a second preamble through.
    """
    document = (
        BS
        + "documentclass[11pt]{article}\n"
        + BS
        + "begin{document}\n"
        + BODY
        + BS
        + "end{document}\n"
    )

    cleaned = _assert_body_only("repeated wrapper", document * 3)
    assert cleaned.count(BS + "section") == 1, "the body was duplicated"


def test_the_result_is_still_validated_as_latex():
    """
    The strip is a convenience; the real guarantee is that whatever comes out is
    then escaped and validated, so a wrapper cannot smuggle a shell escape
    through even if this function missed it.
    """
    from app.services.cv.latex_generator import clean_body_for_latex
    from app.services.cv.pdf_compiler import LaTeXSourceError, validate_latex_body

    response = (
        "Here is your CV:\n"
        + BS
        + "documentclass[11pt]{article}\n"
        + BS
        + "usepackage[margin=0cm]{geometry}\n"
        + BS
        + "begin{document}\n"
        + BS
        + "section*{Experience}\n- wrote \\write18{evil} in a shell\n"
        + BS
        + "end{document}\n"
    )

    cleaned = clean_body_for_latex(clean_llm_response_to_latex(response))

    assert "write18" in cleaned, "the shell-escape attempt should survive as text"
    assert "evil" in cleaned

    # And the body is still validated.
    validate_latex_body(cleaned)

    # A body that is unsafe for a reason the strip cannot address is still
    # refused, so the strip is not a bypass.
    with pytest.raises((LaTeXSourceError, ValueError)):
        validate_latex_body(BS + "input\\begin{document}" + chr(31) + "x")
