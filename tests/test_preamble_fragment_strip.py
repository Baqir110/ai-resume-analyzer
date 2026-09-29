"""
A model's preamble fragments are discarded; its content is not.

The application's template owns the preamble and the shell policy -- ``pdflatex``
runs with ``-no-shell-escape`` and nothing a model writes may change that. So
preamble commands in a generated body are always discarded.

The defect: models frequently emit a preamble *fragment* with no
``\\documentclass`` and no ``\\begin{document}`` for the document-wrapper strip to
key off. Measured on ``llama3.2``, **8 of 8 attempts** carried at least one --
``\\usepackage``, ``\\definecolor``, ``\\hypersetup``, ``\\titleformat``,
``\\titlespacing``, ``\\pagestyle``, ``\\setlength``, ``\\emergencystretch``,
``\\renewcommand{\\familydefault}``, and the template's own ``\\jobheader`` and
``\\degreeheader`` macros. Every one then failed the source validator, so the
pipeline failed on every run.

Three things had to be true at once, and each was got wrong first:

1. **A body command must survive untouched.** The first version replaced the
   matched text -- backslash included -- with a placeholder, which silently turned
   ``\\section`` into ``\\`` and ``\\textbf{Role}`` into ``\\{Role}``.
2. **A removed command must take its arguments with it.** Leaving ``{pdfauthor=x}``
   behind unbalances the document, so the body is then refused for a structural
   reason that names the wrong cause.
3. **One unparseable command must not revert the rest.** The version that
   returned the input unchanged on an unbalanced group threw away every removal it
   had already made, so a preamble that was 95% removable came back 0% stripped.
"""

from __future__ import annotations

import pytest

from app.services.cv.latex_generator import (
    _PREAMBLE_COMMANDS,
    _PREAMBLE_COMMANDS_OPEN_ENDED,
    _read_balanced_group,
    _strip_preamble_fragments,
    clean_llm_response_to_latex,
)

BS = chr(92)

BODY = (
    BS
    + "section*{Experience}\n"
    + BS
    + "textbf{Platform Engineer} "
    + BS
    + "textbar{} Beispiel GmbH\n"
    + "- Reduced deployment time by forty percent using GitLab CI.\n"
)

#: Body commands the strip must never touch. These carry the content.
BODY_COMMANDS = (
    "section",
    "textbf",
    "textit",
    "item",
    "itemize",
    "enumerate",
    "begin",
    "end",
    "textbar",
    "detokenize",
    "href",
    "vspace",
    "quad",
    "newline",
    "centering",
)

#: Preamble commands that must be removed.
PREAMBLE_COMMANDS = (
    "usepackage",
    "definecolor",
    "hypersetup",
    "titleformat",
    "titlespacing",
    "pagestyle",
    "setlength",
    "addtolength",
    "setlist",
    "graphicspath",
)


# ===========================================================================
# 1. Body commands survive
# ===========================================================================


@pytest.mark.parametrize("command", BODY_COMMANDS)
def test_a_body_command_is_never_removed(command):
    """
    The regression. The first version replaced the whole match with a
    placeholder, which ate the command name and left the backslash.
    """
    body = f"{BS}{command}{{value}}\n"

    assert (
        _strip_preamble_fragments(body) == body
    ), f"\\{command} is body content and must survive untouched"


def test_a_realistic_body_is_unchanged():
    assert _strip_preamble_fragments(BODY) == BODY


# ===========================================================================
# 2. Preamble commands are removed, arguments and all
# ===========================================================================


@pytest.mark.parametrize("command", PREAMBLE_COMMANDS)
def test_a_preamble_command_is_removed(command):
    stripped = _strip_preamble_fragments(f"{BS}{command}{{a}}{{b}}\n{BODY}")

    assert f"{BS}{command}" not in stripped
    assert "Platform Engineer" in stripped, "the body was discarded"


def test_nothing_is_left_behind():
    """
    A bare ``{...}`` left where a command was is worse than the command: it
    unbalances the document.
    """
    stripped = _strip_preamble_fragments(
        f"{BS}usepackage[T1]{{fontenc}}\n"
        f"{BS}hypersetup{{pdfauthor=Candidate}}\n"
        f"{BS}renewcommand{{{BS}familydefault}}{{{BS}sfdefault}}\n"
        f"{BS}newcommand{{{BS}jobheader}}[2]{{#1 {BS}textbar{{}} #2}}\n"
        f"{BS}titleformat{{{BS}section}}{{}}{{}}{{}}{{}}\n" + BODY
    )

    # Braces must balance, or the body is refused for the wrong reason.
    assert stripped.count("{") == stripped.count(
        "}"
    ), f"unbalanced braces remain: {stripped[:120]!r}"

    # And the leftover text is the body alone.
    assert stripped.strip() == BODY.strip()


@pytest.mark.parametrize(
    "fragment",
    [
        BS + "usepackage[T1]{fontenc}",
        BS + "usepackage{helvet}",
        BS + "definecolor{primary}{HTML}{1F4E79}",
        BS + "hypersetup{pdfauthor=X}",
        BS + "titleformat{" + BS + "section}{}{}{}{}",
        BS + "titlespacing{" + BS + "section}{0pt}{1em}{0pt}",
        BS + "pagestyle{empty}",
        BS + "setlength{" + BS + "parindent}{0pt}",
        BS + "renewcommand{" + BS + "familydefault}{" + BS + "sfdefault}",
        BS + "newcommand{" + BS + "jobheader}[2]{#1 " + BS + "textbar{} #2}",
        BS + "emergencystretch3em",
        BS + "tolerance=2000",
        BS + "hbadness=10000",
    ],
)
def test_each_measured_fragment_is_removed_completely(fragment):
    """
    Every fragment observed on llama3.2, checked one at a time so a failure names
    the command rather than the document.
    """
    stripped = _strip_preamble_fragments(fragment + "\n" + BODY)

    residue = stripped.replace(BODY, "").strip()
    assert residue == "", f"{fragment!r} left {residue!r}"


# ===========================================================================
# 3. Robustness
# ===========================================================================


def test_an_unbalanced_group_does_not_revert_the_whole_strip():
    """
    The second regression. Returning the input unchanged on an unbalanced group
    discarded every removal already made, so one bad command at the end of a
    preamble left the whole preamble in place.
    """
    text = f"{BS}usepackage{{helvet}}\n" f"{BS}hypersetup{{pdfauthor=X\n" + BODY

    stripped = _strip_preamble_fragments(text)

    assert f"{BS}usepackage" not in stripped, "the removable command was kept"


def test_a_bare_brace_is_left_alone():
    """
    The reader is only consulted for a command's own arguments, so text that is
    not a command argument is never interpreted.
    """
    text = "a brace { on its own }\n"

    assert _strip_preamble_fragments(text) == text


def test_a_command_with_no_arguments_is_removed():
    for fragment in (
        f"{BS}raggedbottom",
        f"{BS}flushend",
        f"{BS}sloppy",
        f"{BS}fussy",
    ):
        stripped = _strip_preamble_fragments(fragment + "\n" + BODY)
        assert fragment not in stripped, f"{fragment!r} survived"


def test_the_escaped_brace_does_not_open_a_group():
    """
    ``\\{`` is a literal brace. Counting it as a group opener makes every group
    after it appear unbalanced.
    """
    # "a{b}c": the group spans indices 1..3, so the reader returns 4.
    assert _read_balanced_group("a{b}c", 1) == 4, "a simple group was misread"

    # "a\{b}c": the brace at index 2 is escaped, so the group opened at index 1
    # never closes and the reader must say so rather than guess.
    assert (
        _read_balanced_group("a" + BS + "{b}c", 1) is None
    ), "an escaped brace was treated as a group opener"


def test_a_commented_brace_is_ignored():
    """
    ``%`` comments out the rest of the line, so a brace after one is text. This
    is the most common reason a hand-rolled reader disagrees with pdflatex.

    "{a % }\\nb}" -- the ``}`` on the comment line is text, so the group that
    opened at index 0 closes at the final ``}``, index 8, and the reader returns 9.
    """
    text = "{a % }" + "\n" + "b}"
    assert _read_balanced_group(text, 0) == 9


def test_an_escaped_percent_is_not_a_comment():
    group = _read_balanced_group("{a\\%b}", 0)
    assert group == 6, "an escaped percent was treated as a comment"


def test_the_fragment_strip_reaches_the_generation_path():
    """
    Checked on the real entry point, not only on the helper. A helper that works
    and is not called is not a fix.
    """
    response = (
        BS
        + "documentclass[11pt,a4paper]{article}\n"
        + BS
        + "usepackage[T1]{fontenc}\n"
        + BS
        + "begin{document}\n"
        + BS
        + "usepackage{helvet}\n"
        + BS
        + "definecolor{primary}{HTML}{1F4E79}\n"
        + BS
        + "hypersetup{pdfauthor=Candidate}\n"
        + BODY
        + BS
        + "end{document}\n"
    )

    cleaned = clean_llm_response_to_latex(response)

    for token in ("documentclass", "usepackage", "definecolor", "hypersetup", "begin"):
        assert token not in cleaned, f"{token!r} survived the generation path"

    assert "Platform Engineer" in cleaned


def test_the_command_tables_do_not_overlap_with_body_commands():
    """
    A single overlap would make the strip delete content, which is the failure
    mode the whole exercise is about.
    """
    declared = set(_PREAMBLE_COMMANDS) | set(_PREAMBLE_COMMANDS_OPEN_ENDED)

    assert not declared & set(BODY_COMMANDS), (
        f"declared as preamble but also body content: " f"{sorted(declared & set(BODY_COMMANDS))}"
    )
