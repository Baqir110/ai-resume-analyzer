"""
A malformed URL costs a link, not a CV.

The defect this exists for: ``latex_escape_url`` calls ``sanitize_latex_url`` and
lets it raise, and it is called from inside a ``re.sub`` callback in
``clean_body_for_latex`` -- before the protection token exists. The exception
propagates out, ``generate_german_latex_content`` catches it with a blanket
``except Exception``, and the caller is told "CV generation failed; no CV was
produced". That message points at the model. The actual cause was one bad
character in one optional project URL: ``qwen2.5:7b`` failed four of four
full-pipeline attempts on it.

The security question is whether dropping a link instead of refusing a document
weakens anything. It does not, and the tests below are the argument:

* the sanitiser still refuses an unsafe URL for every direct caller;
* the body validator still raises for a body that genuinely contains one;
* the application never dereferences a ``\\href``, so no fetch policy is in play;
* what is substituted is ``\\relax``, which carries no URL and no injection.

What is *not* relaxed is stated just as firmly: unbalanced braces remain a hard
structural error, and a document whose body is unsafe in any other way is still
refused.
"""

from __future__ import annotations

import pytest

from app.services.cv.latex_generator import clean_body_for_latex
from app.services.cv.pdf_compiler import (
    LaTeXSourceError,
    sanitize_latex_url,
    validate_latex_body,
    validate_latex_document,
)

BS = chr(92)

#: URLs the sanitiser refuses, verified against its real behaviour.
#:
#: Cases it *normalises* rather than refuses are deliberately absent. It rewrites
#: ``file:///etc/passwd`` to ``https://file:///etc/passwd`` and ``%zz`` to
#: ``%25zz``, both of which are dead or corrected links rather than injection
#: vectors -- a bare ``file://`` string is a sign the model meant to write a link,
#: not an attempt to make the application read the file. Nothing here is
#: dereferenced by the application either way.
UNSAFE_URLS = (
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "https://example.com/a}b",
    "https://example.com/<script>",
    "https://example.com/a b",
    "https://user:password@example.com/",
    "https://example.com/" + BS + "x",
)


# ===========================================================================
# 1. The sanitiser is unchanged
# ===========================================================================


@pytest.mark.parametrize("url", UNSAFE_URLS)
def test_the_sanitiser_still_refuses_unsafe_urls(url):
    """
    The direct contract, unchanged. Dropping a link is a decision made by the
    *caller*; the sanitiser itself is no less strict than before.
    """
    with pytest.raises(LaTeXSourceError):
        sanitize_latex_url(url)


def test_the_body_validator_still_raises_for_an_unsafe_link():
    """
    Also unchanged. Nothing about the validation path was relaxed.
    """
    body = f"{BS}href{{javascript:alert(1)}}{{click here}}"

    with pytest.raises(LaTeXSourceError):
        validate_latex_body(body)


# ===========================================================================
# 2. The generation path drops the link instead
# ===========================================================================


def test_a_bad_hrlink_costs_the_link_not_the_document():
    """
    The regression. This raised out of clean_body_for_latex before.
    """
    body = (
        f"{BS}section*{{Experience}}\n"
        f"{BS}textbf{{Platform Engineer}} {BS}textbar{{}} Beispiel GmbH\n"
        f"{BS}hrlink{{javascript:alert(1)}}{{Repository}}\n"
        f"- Reduced deployment time by forty percent using GitLab CI.\n"
    )

    cleaned = clean_body_for_latex(body)

    assert "javascript" not in cleaned, "the unsafe URL survived"
    assert "Repository" in cleaned, "the visible link text was lost"
    assert BS + "section" in cleaned, "the document structure was lost"


def test_a_bad_project_url_costs_the_link_not_the_document():
    """
    The exact shape that broke qwen2.5:7b: three arguments, the third a URL.
    """
    body = (
        f"{BS}projheader{{Container Pipeline}}{{2024}}{{javascript:alert(1)}}\n"
        f"- Established a signed, reproducible build pipeline.\n"
    )

    cleaned = clean_body_for_latex(body)

    assert "javascript" not in cleaned
    assert "Container Pipeline" in cleaned, "the project title was lost"
    assert "2024" in cleaned, "the project date was lost"


def test_a_file_url_is_normalised_rather_than_dropped():
    """
    A URL the sanitiser can repair is repaired, not discarded.

    ``file:///etc/passwd`` becomes ``https://file:///etc/passwd`` -- a dead link
    to a host named "file". The application never dereferences a ``\\href``, so
    there is nothing to read, and the alternative is to drop a link the candidate
    may genuinely have meant as a web address.
    """
    body = f"{BS}projheader{{Container Pipeline}}{{2024}}{{file:///etc/passwd}}\n"

    cleaned = clean_body_for_latex(body)

    # Normalised, not dropped: the scheme the sanitiser added is present, and the
    # original bare scheme is no longer the whole URL.
    assert "https://file" in cleaned
    assert "Container Pipeline" in cleaned


def test_a_good_url_is_kept():
    """
    The other direction: the fix must not degrade working documents.
    """
    body = (
        f"{BS}section*{{Projects}}\n"
        f"{BS}projheader{{Container Pipeline}}{{2024}}"
        f"{{https://github.com/example/pipeline}}\n"
    )

    cleaned = clean_body_for_latex(body)

    assert "github.com/example/pipeline" in cleaned, "a valid link was dropped"


def test_relax_is_not_itself_rejected():
    """
    \\relax is the substitute. If the validator refused it, the document it
    rescued would be rejected by the validator that rescued it -- which is
    exactly the failure mode this change is fixing, one layer down.
    """
    body = f"{BS}href{{\\relax}}{{Repository}}"

    # Must not raise.
    validate_latex_body(body)


def test_several_bad_links_all_cost_only_links():
    body = (
        f"{BS}section*{{Projects}}\n"
        + f"{BS}projheader{{One}}{{2023}}{{javascript:alert(1)}}\n"
        + f"{BS}projheader{{Two}}{{2024}}{{data:text/html,x}}\n"
        + f"{BS}projheader{{Three}}{{2025}}{{https://ok.example}}\n"
    )

    cleaned = clean_body_for_latex(body)

    assert "javascript" not in cleaned
    assert "data:text/html" not in cleaned
    assert "ok.example" in cleaned, "the one good link was dropped too"
    for title in ("One", "Two", "Three"):
        assert title in cleaned, f"the title {title!r} was lost"


def test_a_document_with_no_links_is_unaffected():
    body = (
        f"{BS}section*{{Experience}}\n"
        f"{BS}textbf{{Platform Engineer}} {BS}textbar{{}} Beispiel GmbH\n"
        f"- Reduced deployment time by forty percent using GitLab CI.\n"
    )

    cleaned = clean_body_for_latex(body)

    assert "Platform Engineer" in cleaned
    assert BS + "hrlink" not in cleaned


# ===========================================================================
# 3. What is still refused
# ===========================================================================


def test_an_injected_brace_is_escaped_rather_than_executed():
    """
    The generation path does not reject an unbalanced body -- it *escapes* it.

    That is the stronger property: a stray brace from the model becomes text, so
    it cannot open a group, close one early, or escape the argument of a
    command. Escaping also means a malformed body is repaired rather than
    discarded, which is the same philosophy as dropping an unusable link.
    """
    body = f"{BS}section{{Experience\n- an item\n"

    cleaned = clean_body_for_latex(body)

    # The command is now literal text, not a command.
    assert BS + "textbackslash{}section" in cleaned
    assert f"{BS}section{{Experience" not in cleaned


def test_the_validators_still_refuse_unbalanced_braces_directly():
    """
    Checked at both levels, so the property does not rest on which function is
    called. Handed unbalanced source, they refuse it.
    """
    body = f"{BS}section{{Experience\n- an item\n"

    with pytest.raises(LaTeXSourceError):
        validate_latex_body(body)

    document = (
        f"{BS}documentclass[11pt,a4paper]{{article}}\n"
        f"{BS}begin{{document}}\n"
        f"{BS}section{{Broken\n"
        f"{BS}end{{document}}\n"
    )

    with pytest.raises(LaTeXSourceError):
        validate_latex_document(document)


def test_a_non_string_body_is_still_refused():
    with pytest.raises((LaTeXSourceError, ValueError, TypeError)):
        clean_body_for_latex(None)


def test_an_oversized_body_is_still_refused():
    from app.services.cv.latex_generator import MAX_LLM_LATEX_CHARS

    with pytest.raises((LaTeXSourceError, ValueError)):
        clean_body_for_latex("x" * (MAX_LLM_LATEX_CHARS + 1))
