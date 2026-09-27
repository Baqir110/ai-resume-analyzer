r"""The single LaTeX escaping layer for all dynamic CV text.

Why this module exists
----------------------
``pdflatex`` aborts on the first unescaped category/parameter character.  In a
CV body the characters that actually break compilation are:

    &   "Misplaced alignment tab character &."
    #   "You can't use `macro parameter character #' in horizontal mode."
    ^   "Missing $ inserted."          (^ is a math superscript)
    \   "Undefined control sequence."   (C:\Users -> \Users)
    $   "Missing $ inserted."          (an odd count opens math mode)
    { } "Missing } inserted." / "Extra }, or forgotten }."

The model is told about ``&``, ``%`` and ``_`` in the prompt, but not about
``$``, ``#``, ``^`` or ``\\``, and models get them wrong constantly.  Relying on
the prompt is therefore not a correctness mechanism -- escaping is.  This module
is that mechanism and it is the *only* one: ``clean_body_for_latex``,
``_escape_latex_text`` and the pre-flight self-check all route through here, so
dynamic text is escaped exactly once.

Two guarantees, both covered by tests
------------------------------------
**Idempotent / never double-escapes.**  ``escape(escape(x)) == escape(x)``.
Already-correct input such as ``\\&``, ``\textless{}`` or ``\textbackslash{}``
is recognised and passed through untouched, so re-running the layer over text
that was escaped by a previous version of the pipeline cannot turn ``&`` into
``\\&``.

**Structure preserving.**  The body is LaTeX, not plain prose: ``\\section*{}``,
``\\item``, ``\\jobheader{}{}{}`` and ``\\begin{itemize}`` must survive intact
while only the *text* between them is escaped.  A plain character-wise
substitution cannot do that, so this module is a small scanner that understands
brace groups.

Safety model
------------
Control sequences are passed through only if they are on an explicit
allow-list.  Anything else -- ``\\Users``, ``\\(``, ``\\'e`` -- is emitted as
literal text (``\\textbackslash{}``).  That means model output can never
introduce an undefined macro, a math delimiter, or an accent command, which
makes the pre-flight self-check exact rather than heuristic.  ``pdf_compiler``
remains the independent security boundary; this layer is a correctness layer.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ALLOWED_CONTROL_WORDS",
    "LIST_ENVIRONMENTS",
    "MACRO_ARITY",
    "LaTeXEscapeError",
    "balance_latex_environments",
    "escape_latex_body",
    "escape_latex_text",
    "is_fully_escaped",
    "repair_latex_lists",
    "unescaped_specials",
]


class LaTeXEscapeError(ValueError):
    """Raised when LaTeX text cannot be escaped into a valid document."""


# ---------------------------------------------------------------------------
# Character level mapping
# ---------------------------------------------------------------------------

#: Literal characters that are unsafe in LaTeX text, and their replacements.
#:
#: ``{`` and ``}`` appear here for the *text* case only.  A brace that delimits a
#: macro argument is consumed by the scanner and kept, so ``\textbf{A & B}``
#: still has exactly one argument.
#:
#: ``^`` and ``~`` are not strictly fatal in LaTeX, but they are rendered as
#: accents and non-breaking spaces, which silently corrupts CV text such as
#: ``C^`` or ``N~``.  They are escaped for the same reason as ``&``.
TEXT_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}

#: Single-character control sequences that may pass through verbatim.
#:
#: Everything here is inert typography: spacing, italic correction, an empty
#: box, a hyphenation point.  ``[``/``]``/``(``/``)`` are deliberately absent --
#: they open and close math mode, and the scanner has already escaped ``$`` so
#: an unbalanced ``\[`` would be the only way back into math.
SAFE_CONTROL_CHARS: frozenset[str] = frozenset("&%#$_{}\\ /,:;|!*-")


#: Control words that may pass through verbatim.
#:
#: Grouped by origin.  The lists are intentionally generous: a legitimate
#: macro that is missing here would be rendered as visible ``\macro`` text,
#: which is a cosmetic regression, whereas a macro that *should* be missing
#: (``\Users``, ``\write18``) would be a correctness and safety hole.
ALLOWED_CONTROL_WORDS: frozenset[str] = frozenset(
    {
        # Sectioning and paragraph structure.
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
        "item",
        "par",
        "noindent",
        "indent",
        "indentfirst",
        "centering",
        "raggedright",
        "raggedleft",
        "small",
        "footnotesize",
        "scriptsize",
        "tiny",
        "normalsize",
        "large",
        "Large",
        "LARGE",
        "huge",
        "Huge",
        "normalsize",
        "sloppy",
        "fussy",
        # Emphasis and text decoration.
        "textbf",
        "textit",
        "textrm",
        "textsf",
        "texttt",
        "textsc",
        "textsl",
        "textup",
        "textmd",
        "textnormal",
        "emph",
        "underline",
        "uline",
        "sout",
        "so",
        "st",
        "text",
        "mbox",
        "hbox",
        "fbox",
        "framebox",
        "makebox",
        "phantom",
        "hphantom",
        "vphantom",
        "vbox",
        "hbox to",
        # Spacing, rules and page furniture.
        "vspace",
        "hspace",
        "hspace*",
        "smallskip",
        "medskip",
        "bigskip",
        "quad",
        "qquad",
        "enspace",
        "thinspace",
        "medspace",
        "thickspace",
        "negthinspace",
        "negmedspace",
        "negthickspace",
        "hrule",
        "hrulefill",
        "rule",
        "raisebox",
        "strut",
        "vspace*",
        "hfill",
        "vfill",
        "dotfill",
        "leaders",
        "hfill*",
        "break",
        "nobreak",
        "protect",
        "relax",
        "newline",
        "linebreak",
        "pagebreak",
        "newpage",
        "clearpage",
        # Colours.
        "color",
        "textcolor",
        "colorbox",
        "normalcolor",
        "definecolor",
        "pagecolor",
        # Lists and verbatim-free layout.
        "begin",
        "end",
        "caption",
        "centering*",
        "tabularnewline",
        # Links.
        "href",
        "url",
        "urldef",
        "nolinkurl",
        "detokenize",
        "urlstyle",
        "hypersetup",
        "uline",
        # Application template macros (see pdf_compiler._ALLOWED_DEFINITIONS).
        "jobheader",
        "degreeheader",
        "projheader",
        "hrlink",
        # Symbols and units. inputenc/textcomp map the UTF-8 characters used in
        # German CVs to these, so an already-unescaped model response is
        # recognised rather than turned into literal backslash words.
        "textless",
        "textgreater",
        "textasciitilde",
        "textasciicircum",
        "textbackslash",
        "textunderscore",
        "textbar",
        "textbraceleft",
        "textbraceright",
        "textbullet",
        "textperiodcentered",
        "textellipsis",
        "ldots",
        "dots",
        "textemdash",
        "textendash",
        "textquotedblleft",
        "textquotedblright",
        "textquoteleft",
        "textquoteright",
        "textquotedbl",
        "textquotesingle",
        "textcopyright",
        "textregistered",
        "texttrademark",
        "textdegree",
        "textcelsius",
        "textmu",
        "textohm",
        "texteuro",
        "textcent",
        "textsterling",
        "textcurrency",
        "textyen",
        "textwon",
        "textbaht",
        "textnumero",
        "textparagraph",
        "textsection",
        "texttimes",
        "textdagger",
        "textdaggerdbl",
        "textperthousand",
        "textborn",
        "textdied",
        "textmarried",
        "textdiv",
        "textohm",
        "mathsterling",
        "textvisiblespace",
        "textsuperscript",
        "textsubscript",
        # Mathematical accents used for combining marks.
        "ss",
        "aa",
        "o",
        "O",
        "l",
        "L",
        "ae",
        "AE",
        "oe",
        "OE",
        "i",
        "j",
        # Misc typography the templates and models use.
        "small",
        "textsuperscript",
        "linebreak",
        "hspace",
        "kern",
        "mkern",
        "char",
        "symbol",
        "index",
        "glossary",
        "footnote",
        "footnotemark",
        "footnotetext",
        "marginpar",
        "textfloat",
        "fancybox",
        "framebox",
        "scalebox",
        "resizebox",
        "rotatebox",
        "adjustbox",
        "raggedbottom",
        "flushbottom",
        "onecolumn",
        "twocolumn",
        "columnbreak",
        "multicolumn",
        "tabular",
        "array",
        "toprule",
        "midrule",
        "bottomrule",
        "cmidrule",
        "rowcolor",
        "cellcolor",
        "addlinespace",
        "lstinline",
        "mint",
        "pycode",
        "verbatim",
        "verb",
        "sloppypar",
        "microtype",
        "selectlanguage",
        "foreignlanguage",
        "setmainfont",
        "setsansfont",
        "setmonofont",
        "XeTeX",
        "pdfTeX",
        "hyphenation",
        "pattern",
        "nohyphens",
        "sloppy",
    }
)


# ---------------------------------------------------------------------------
# Plain-text escaping
# ---------------------------------------------------------------------------


def escape_latex_text(value: object) -> str:
    """
    Escape one plain-text value for use in LaTeX.

    This is the entry point for text with **no LaTeX structure** in it: the
    candidate name, the professional title, a contact line, a colour value.  It
    is a straight per-character substitution with no notion of an existing
    escape sequence, because its input is raw resume text and a backslash in
    "raw resume text" is a literal backslash, not a macro.

    Use :func:`escape_latex_body` instead for anything the model produced that
    already contains LaTeX structure, and use each exactly once.

    >>> escape_latex_text("Müller & Söhne")
    'Müller \\\\& Söhne'
    """
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))

    return "".join(TEXT_ESCAPES.get(char, char) for char in text)


# ---------------------------------------------------------------------------
# Structure-preserving body escaping
# ---------------------------------------------------------------------------


def _read_control_sequence(text: str, index: int) -> tuple[str, bool, int]:
    r"""
    Read the control sequence starting at the backslash ``index``.

    Returns ``(payload, is_word, index_after)`` where ``payload`` excludes the
    leading backslash: ``"section"``, ``"section*"``, ``"&"`` or ``""`` for a
    lone trailing backslash.
    """
    length = len(text)
    cursor = index + 1

    if cursor >= length:
        return "", False, cursor

    if not text[cursor].isalpha():
        return text[cursor], False, cursor + 1

    word_start = cursor
    while cursor < length and (text[cursor].isalpha() or text[cursor] == "*"):
        cursor += 1

    return text[word_start:cursor], True, cursor


def _read_group(text: str, index: int) -> tuple[str, int] | None:
    r"""
    Read one balanced brace group starting at ``index``.

    Returns ``(inner, index_after_closing_brace)`` or ``None`` when the group
    is unterminated.  A backslash escapes the next character, so ``\{`` does not
    open a group -- which is what makes this correct for already-escaped input.
    """
    length = len(text)

    if index >= length or text[index] != "{":
        return None

    depth = 0
    cursor = index

    while cursor < length:
        char = text[cursor]

        if char == "\\":
            cursor += 2
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : cursor], cursor + 1

        cursor += 1

    return None


def _emit_literal_word(word: str) -> str:
    r"""Render an unapproved control sequence as visible literal text."""
    return "\\textbackslash{}" + word


def _is_allowed_word(word: str) -> bool:
    r"""
    Report whether a control word may pass through verbatim.

    A trailing star is the LaTeX "unnumbered" marker (``\section*``,
    ``\textbf*``), so it is checked separately from the base name.
    """
    if word in ALLOWED_CONTROL_WORDS:
        return True

    if word.endswith("*") and word[:-1] in ALLOWED_CONTROL_WORDS:
        return True

    return False


#: How many balanced brace arguments an allowed macro consumes.
#:
#: One is the overwhelmingly common case (``\section*{}``, ``\textit{}``,
#: ``\begin{itemize}``) and is therefore the default.  Only the macros that
#: genuinely take more are listed, so the scanner stays a small allow-list
#: rather than a LaTeX parser.
MACRO_ARITY: dict[str, int] = {
    # Application template macros.
    "jobheader": 3,
    "degreeheader": 3,
    "projheader": 3,
    "hrlink": 2,
    # Links.
    "href": 2,
    # Framing, colouring and vertical lifting.
    "raisebox": 2,
    "colorbox": 2,
    "fcolorbox": 3,
    "textcolor": 2,
    "fbox": 1,
    "framebox": 1,
    "makebox": 1,
    "parbox": 2,
    "rule": 2,
    # Sizes and rules.
    "resizebox": 3,
    "scalebox": 2,
    "rotatebox": 2,
    "setlength": 2,
    "addtolength": 2,
    "settowidth": 2,
    # Spacing commands accept a starred form and a length argument.
    "vspace": 1,
    "hspace": 1,
}


def _macro_arity(word: str) -> int:
    r"""Return how many brace arguments ``word`` consumes."""
    if word.endswith("*"):
        word = word[:-1]

    return MACRO_ARITY.get(word, 1)


def _escape_segment(text: str) -> str:
    r"""
    Escape one brace-balanced segment of a LaTeX body.

    Structure (``\\section*{...}``, ``\\item``, ``\\begin{itemize}``) is kept;
    the text inside every macro argument is escaped recursively.
    """
    output: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        # ---- control sequences ------------------------------------------
        if char == "\\":
            bare, is_word, cursor = _read_control_sequence(text, index)

            if not is_word and bare == "\\":
                # \\ is a line break: inert, and must survive untouched.
                output.append("\\\\")
                index = cursor
                continue

            if not bare:
                # A lone trailing backslash is a literal character.
                output.append(TEXT_ESCAPES["\\"])
                index = cursor
                continue

            if is_word:
                if not _is_allowed_word(bare):
                    output.append(_emit_literal_word(bare))
                    index = cursor
                    continue
            elif bare not in SAFE_CONTROL_CHARS:
                # \[ \] \( \) \' \" and friends: never a macro, always either a
                # math delimiter or an accent. Render as a literal character so
                # model output can never re-enter math mode.
                output.append("\\textbackslash{}")
                output.append(TEXT_ESCAPES.get(bare, bare))
                index = cursor
                continue

            # Locate the macro's first argument before emitting anything, so an
            # unterminated group can be handled as literal text instead of
            # producing a macro whose argument is missing. `cursor` is the
            # position just past the control word, which is where an argument
            # may begin.
            probe = cursor
            while probe < length and text[probe] in " \t\r\n":
                probe += 1

            opens_group = probe < length and text[probe] == "{"

            if opens_group and _read_group(text, probe) is None:
                # The model wrote \textbf{ with no closing brace. Emitting
                # \textbf here would leave a macro with no argument, which
                # pdflatex resolves differently depending on the macro's
                # definition. Rendering the control word as visible text is
                # unambiguous and always valid.
                output.append(_emit_literal_word(bare))
                index = probe
                continue

            output.append("\\" + bare)
            index = cursor

            # Consume up to the macro's arity in balanced brace groups. Each
            # group is recursed into, so its *text* is escaped while the braces
            # themselves survive. LaTeX allows several arguments per macro and
            # the CV templates rely on it (\jobheader{role}{company}{dates}),
            # so consuming only the first group would escape the rest.
            for _ in range(_macro_arity(bare)):
                probe = index
                while probe < length and text[probe] in " \t\r\n":
                    probe += 1

                if probe >= length or text[probe] != "{":
                    break

                group = _read_group(text, probe)
                if group is None:
                    # A later argument is unterminated. Stop here; the brace is
                    # escaped as text by the ordinary path.
                    index = probe
                    break

                inner, after = group
                output.append("{")
                output.append(_escape_segment(inner))
                output.append("}")
                index = after

            continue

        # ---- stray braces ------------------------------------------------
        if char == "{":
            output.append(TEXT_ESCAPES["{"])
            index += 1
            continue

        if char == "}":
            output.append(TEXT_ESCAPES["}"])
            index += 1
            continue

        # ---- ordinary characters ----------------------------------------
        replacement = TEXT_ESCAPES.get(char)
        if replacement is not None:
            output.append(replacement)
            index += 1
            continue

        output.append(char)
        index += 1

    return "".join(output)


def escape_latex_body(text: str) -> str:
    """
    Escape a LaTeX body, keeping its structure and neutralising its text.

    ``text`` is a complete ``\\begin{itemize} ... \\end{itemize}``-style body as
    produced by the model or by :mod:`app.services.cv.latex_generator`.  Calls
    are idempotent, so a body that was already escaped is returned unchanged.

    Raises:
        LaTeXEscapeError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise LaTeXEscapeError("LaTeX text must be a string.")

    if not text:
        return ""

    # NFC first.  A model that emits decomposed umlauts (a + U+0308) produces a
    # body that renders as garbage under T1 even though it is valid UTF-8.
    # Normalising once here also keeps this function idempotent, because NFC
    # is idempotent.
    return _escape_segment(unicodedata.normalize("NFC", text))


def is_fully_escaped(text: str) -> bool:
    """
    Report whether ``text`` contains no unescaped LaTeX specials.

    Because :func:`escape_latex_body` is idempotent, ``escape_latex_body(x) == x``
    is exactly the condition "nothing is left to escape".  This is used as a
    pre-flight assertion instead of a second, independent set of regular
    expressions, so the check and the fix can never drift apart.
    """
    if not isinstance(text, str):
        return False

    return escape_latex_body(text) == unicodedata.normalize("NFC", text)


def unescaped_specials(text: str) -> list[tuple[int, str]]:
    r"""
    Return ``(line_number, character)`` for every unescaped special found.

    Line numbers are 1-based.  Used to build a precise pre-flight error instead
    of a generic "LaTeX compilation failed".
    """
    if not isinstance(text, str) or not text:
        return []

    problems: list[tuple[int, str]] = []
    line = 1

    for char in text:
        if char == "\n":
            line += 1
        elif char in TEXT_ESCAPES and char != "\\":
            problems.append((line, char))
        elif char == "\\":
            problems.append((line, "\\"))

    return problems


# ============================================================================
# Environment balancing
# ============================================================================
#
# A model that writes \begin{itemize} and forgets \end{itemize} produces
# "LaTeX Error: \begin{itemize} on input line 21 ended by \end{document}", which
# is fatal and unfixable by escaping. Closing the missing environments is
# deterministic: it adds no content, changes no wording, and only repairs
# structure that LaTeX requires to be balanced.

_ENVIRONMENT_RE = re.compile(r"\\(begin|end)\s*\{\s*([A-Za-z*]+)\s*\}")


def balance_latex_environments(text: str) -> str:
    r"""
    Close unterminated ``\begin`` environments and drop unmatched ``\end``s.

    Escaping cannot fix this class of error, and it is one of the most common
    things a model gets wrong, so it is repaired here rather than turned into a
    pdflatex failure.

    Nesting is preserved: a missing inner ``\end`` is emitted at the position
    of the ``\end`` that closed the outer environment, and environments still
    open at the end of the body are closed in reverse order.
    """
    if not isinstance(text, str):
        raise LaTeXEscapeError("LaTeX text must be a string.")

    if not text or ("\\begin" not in text and "\\end" not in text):
        return text

    matches = list(_ENVIRONMENT_RE.finditer(text))
    if not matches:
        return text

    stack: list[tuple[str, int]] = []
    drop: set[int] = set()
    # match index -> environment names to close immediately before that match
    close_before: dict[int, list[str]] = {}
    # match index -> environment names to close immediately after that match
    close_after: dict[int, list[str]] = {}

    for index, match in enumerate(matches):
        kind = match.group(1)
        name = match.group(2)

        if kind == "begin":
            stack.append((name, index))
            continue

        if stack and stack[-1][0] == name:
            stack.pop()
        elif any(entry[0] == name for entry in stack):
            # \end{outer} with an inner environment still open. Close the
            # inner ones, and the outer one, immediately before this point.
            # The \end that triggered the repair is then redundant -- keeping
            # it would emit a second, unmatched \end{outer} -- so it is
            # dropped.
            while stack:
                open_name, _open_index = stack.pop()
                close_before.setdefault(index, []).append(open_name)
                if open_name == name:
                    break

            drop.add(index)
        else:
            # An \end with no matching \begin. "Extra \end{itemize}" is an
            # error, so it is removed.
            drop.add(index)

    if stack:
        close_after.setdefault(len(matches), []).extend(name for name, _ in reversed(stack))

    if not close_before and not close_after and not drop:
        return text

    pieces: list[str] = []
    cursor = 0

    for index, match in enumerate(matches):
        pieces.append(text[cursor : match.start()])

        # Closers go immediately before the \end that triggered them, not
        # before the text preceding it, otherwise the repaired nesting is
        # itself malformed.
        for name in close_before.get(index, ()):
            pieces.append(f"\\end{{{name}}}")

        if index not in drop:
            pieces.append(match.group(0))
            for name in close_after.get(index, ()):
                pieces.append(f"\\end{{{name}}}")

        cursor = match.end()

    pieces.append(text[cursor:])

    for name in close_after.get(len(matches), ()):
        pieces.append(f"\n\\end{{{name}}}")

    return "".join(pieces)


#: Environments in which ``\item`` is legal.
LIST_ENVIRONMENTS: frozenset[str] = frozenset({"itemize", "enumerate", "description", "list"})

_ITEM_RE = re.compile(r"^\\item(?![A-Za-z@])")


def repair_latex_lists(text: str) -> str:
    r"""
    Wrap runs of ``\item`` that sit outside any list environment.

    "LaTeX Error: Lonely \item--perhaps a missing list environment." is fatal
    and is not an escaping problem, so no amount of character handling fixes it.
    It happens constantly: a model writes ``\jobheader{role}{company}{dates}``
    and then a run of ``\item`` bullets, having omitted the
    ``\begin{itemize}`` the prompt asked for. A local model does it more often
    than a large one.

    The repair is deterministic and content-preserving. Each orphaned run is
    wrapped in the environment the model clearly intended, so the bullets keep
    their intended appearance and the document compiles. Runs that are already
    inside a list are left exactly as they are.
    """
    if not isinstance(text, str):
        raise LaTeXEscapeError("LaTeX text must be a string.")

    if "\\item" not in text:
        return text

    lines = text.split("\n")
    output: list[str] = []
    orphaned: list[str] = []
    depth = 0

    def flush() -> None:
        if not orphaned:
            return
        output.append("\\begin{itemize}")
        output.extend(orphaned)
        output.append("\\end{itemize}")
        orphaned.clear()

    for line in lines:
        for match in _ENVIRONMENT_RE.finditer(line):
            name = match.group(2)
            if name in LIST_ENVIRONMENTS:
                if match.group(1) == "begin":
                    depth += 1
                else:
                    depth = max(0, depth - 1)

        if depth == 0 and _ITEM_RE.match(line.strip()):
            orphaned.append(line)
            continue

        flush()
        output.append(line)

    flush()

    return "\n".join(output)
