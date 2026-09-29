"""Bounded, fail-closed LaTeX-to-PDF compilation.

Only application-generated documents are accepted.  The source validator uses
an allow-list for preamble packages and rejects TeX file/process primitives
before a temporary file is created.  Compilation runs with shell escape
explicitly disabled, a minimal environment, bounded time, and bounded output.
Compiler logs are never returned to callers.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_LATEX_SOURCE_CHARS = 250_000
MAX_LATEX_SOURCE_BYTES = 512_000
MAX_LATEX_LINE_CHARS = 20_000
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_TEX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_TEX_WORKDIR_BYTES = 32 * 1024 * 1024
PDFL_COMPILER_PASS_TIMEOUT_SECONDS = 45
PDFL_COMPILER_TOTAL_TIMEOUT_SECONDS = 100


class PDFLayoutError(RuntimeError):
    """Raised when generated PDF content does not satisfy the page limit."""

    def __init__(
        self,
        pages: int,
        message: str | None = None,
        code: str = "pdf_page_limit_exceeded",
    ) -> None:
        self.pages = pages
        self.code = code
        if message is None:
            message = (
                "Generated PDF must fit on exactly one page; "
                f"the document produced {pages} pages."
            )
        super().__init__(message)


class LaTeXSourceError(ValueError):
    """Raised when LaTeX source is invalid or contains unsafe primitives."""


class LaTeXCompilationError(RuntimeError):
    """
    A compilation failure that names the actual cause.

    The previous version raised a bare "LaTeX compilation failed.", which made
    an unescaped ``&`` in generated CV text indistinguishable from a missing
    font package, a corrupt template or a broken TeX installation.  The compiler
    log already contained the answer; it was being discarded.

    Attributes:
        category: Stable machine-readable bucket, e.g. ``unescaped_alignment``.
        line: 1-based line number in the generated ``.tex``, when known.
        detail: The compiler's own message, truncated and redacted.
        token: The single offending LaTeX character, when one can be named.
        log_excerpt: Preamble-relative context only; never CV body text.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "compiler_error",
        line: int | None = None,
        detail: str = "",
        token: str = "",
        log_excerpt: str = "",
    ) -> None:
        self.category = category
        self.line = line
        self.detail = detail
        self.token = token
        self.log_excerpt = log_excerpt
        super().__init__(message)


# ============================================================================
# Compiler diagnostics
# ============================================================================
#
# A pdflatex run started with -file-line-error reports errors as
# ``./document.tex:213: Misplaced alignment tab character &.`` and repeats the
# offending line as ``l.213 ...``.  These patterns turn that into a category, a
# line number and the single character that caused it.

_FILE_LINE_ERROR_RE = re.compile(r"^(?P<file>\S*?\.tex):(?P<line>\d+):\s*(?P<message>.+?)\s*$")
_BANG_ERROR_RE = re.compile(r"^!\s*(?P<message>.+?)\s*$")
_CONTEXT_LINE_RE = re.compile(r"^l\.(?P<line>\d+)\s?(?P<source>.*?)\s*$")

# Ordered most specific first: the first pattern whose text appears in the
# compiler message decides the category.  Ordered by how actionable the fix is.
_ERROR_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "unescaped_alignment",
        (
            "misplaced alignment tab",
            "extra alignment tab",
            "alignment tab",
        ),
    ),
    ("unescaped_parameter", ("macro parameter character", "macro parameter")),
    (
        "unescaped_math",
        (
            "missing $ inserted",
            "unbalanced math",
            "math shift",
            "illegal math",
        ),
    ),
    (
        "unbalanced_braces",
        (
            "file ended while scanning",
            "missing } inserted",
            "extra }",
            "extra endcsname",
            "paragraph ended before",
        ),
    ),
    ("undefined_macro", ("undefined control sequence",)),
    ("encoding", ("unicode character", "invalid utf-8", "invalid input")),
    ("missing_package", ("not found", "no file", "cannot load")),
    ("font_error", ("font shape", "font not", "no glyph")),
    ("timeout", ("time limit",)),
)

#: The single character that each category is almost always caused by.  Naming
#: the token is what makes the error actionable, and it is safe to report: it is
#: a punctuation character, not CV content.
_CATEGORY_TOKEN = {
    "unescaped_alignment": "&",
    "unescaped_parameter": "#",
    "unescaped_math": "$",
    "unbalanced_braces": "{",
}

#: A compiler message is only echoed into the exception, never into a log line,
#: so it is bounded.  200 characters is enough for every message in
#: _ERROR_CATEGORIES plus the surrounding context.
_MAX_DETAIL_CHARS = 200

#: How much of a failing line may be shown.  Only preamble lines qualify (see
#: _build_latex_diagnostic), because preamble text is application-controlled
#: while body text is the candidate's personal data.
_MAX_EXCERPT_CHARS = 120


_ALLOWED_PACKAGES = {
    "enumitem",
    "fontenc",
    "geometry",
    "hyperref",
    "inputenc",
    "lmodern",
    "mathptmx",
    "textcomp",
    "titlesec",
    "ulem",
    "xcolor",
}

_ALLOWED_DEFINITIONS = {
    "degreeheader": 3,
    "hrlink": 2,
    "jobheader": 3,
    "projheader": 3,
}
_ALLOWED_RENEW_COMMANDS = {"familydefault"}

# File access, process execution, output-stream manipulation, catcode changes,
# and driver-specific primitives.  The CLI also disables shell escape; this
# source policy provides defense in depth and blocks shell-escape-like packages
# even on engines/distributions with different defaults.
_FORBIDDEN_COMMANDS = {
    "addcontentsline",
    "addtohook",
    "addtoshipoutpicture",
    "appendtohook",
    "advance",
    "atbegindocument",
    "atenddocument",
    "batchmode",
    "catcode",
    "commandchars",
    "csname",
    "declareenvironment",
    "declarerobustcommand",
    "def",
    "directlua",
    "divide",
    "edef",
    "endcsname",
    "endlinechar",
    "errorstopmode",
    "explsyntaxoff",
    "explsyntaxon",
    "expandafter",
    "font",
    "gdef",
    "immediate",
    "includegraphics",
    "include",
    "includeonly",
    "input",
    "inputmidi",
    "iterate",
    "latelua",
    "let",
    "loop",
    "luatexlua",
    "makeatletter",
    "makeletter",
    "multiply",
    "newenvironment",
    "newwrite",
    "nonstopmode",
    "obeylines",
    "obeyspaces",
    "openin",
    "openinany",
    "openout",
    "openoutany",
    "passoptionstopackage",
    "providecommand",
    "pdfliteral",
    "pipe",
    "pipes",
    "prependtohook",
    "read",
    "renewenvironment",
    "repeat",
    "requirepackage",
    "scantokens",
    "scrollmode",
    "setbox",
    "setfont",
    "shellescape",
    "shell_escape",
    "show",
    "showbox",
    "special",
    "write",
    "write18",
    "write-18",
    "xdef",
}
_FORBIDDEN_ENVIRONMENTS = {
    "filecontents",
    "filecontents*",
    "shellescape",
    "verbatimwrite",
    "write18",
}
_FORBIDDEN_COMMAND_RE = re.compile(
    r"(?<!\\)\\[ \t\r\n]*@?(?:"
    + "|".join(sorted((re.escape(name) for name in _FORBIDDEN_COMMANDS), key=len, reverse=True))
    + r")(?![A-Za-z@])",
    re.IGNORECASE,
)
_DRIVER_PRIMITIVE_RE = re.compile(
    r"(?<!\\)\\[ \t\r\n]*@?(?:direct|lua|pdf|shell|xe)[A-Za-z@]+",
    re.IGNORECASE,
)
# expl3/L3 programming names can construct or rescan control sequences.  The
# application's trusted templates do not use colon/underscore control words.
#
# At least one letter is required before the underscore or colon, because that
# is what separates a real L3 name (\l_tmpa, \str_new:n) from an escaped literal.
# Without that requirement the pattern also matched ``\_var`` -- the *correct*
# LaTeX output for an underscore in CV text -- and rejected every body the
# escaping layer had correctly escaped.
_EXPL3_COMMAND_RE = re.compile(
    r"(?<!\\)\\[ \t\r\n]*@?[A-Za-z@]+[_:][A-Za-z@_:]+",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069]")
_DOCUMENT_CLASS_COMMAND_RE = re.compile(r"\\documentclass(?![A-Za-z@])", re.IGNORECASE)
_DOCUMENT_CLASS_RE = re.compile(
    r"\\documentclass\[(?:10|11)pt,a4paper\]\{article\}",
    re.IGNORECASE,
)
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{\s*document\s*\}", re.IGNORECASE)
_END_DOCUMENT_RE = re.compile(r"\\end\s*\{\s*document\s*\}", re.IGNORECASE)
_ENVIRONMENT_RE = re.compile(r"\\begin\s*\{\s*([A-Za-z*]+)\s*\}", re.IGNORECASE)
_NEW_COMMAND_RE = re.compile(r"\\newcommand(?![A-Za-z@])", re.IGNORECASE)
_RENEW_COMMAND_RE = re.compile(r"\\renewcommand(?![A-Za-z@])", re.IGNORECASE)
_USEPACKAGE_RE = re.compile(r"\\usepackage(?![A-Za-z@])", re.IGNORECASE)
_USEPACKAGE_START_RE = re.compile(
    r"\\usepackage(?![A-Za-z@])[ \t\r\n]*(?:\[[^\]]*\][ \t\r\n]*)?\{",
    re.IGNORECASE,
)
_SAFE_ENVIRONMENT_KEYS = {
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
}
_URL_SAFE_CHARS = ":/?[]@!$&'()*+,;=._~-"


# ============================================================================
# Safe diagnostics
# ============================================================================
#
# Compiler output quotes the offending source line back at you, and that line is
# usually the candidate's CV.  The error therefore reports the *category*, the
# *line number*, the *offending character* and the compiler's own message -- all
# of which are needed to fix the bug -- and deliberately withholds the source
# line itself.  Nothing here needs the candidate's data to be useful.


#: Environment variables whose *values* must never appear in a diagnostic.
#:
#: Read at call time, never cached, and never logged -- the whole point is that
#: the value is only ever compared against, not printed. A candidate who pastes
#: a key into their CV, or a title that accidentally carries one, would
#: otherwise have it echoed back through an exception and into an HTTP response.
_SECRET_ENV_NAMES = (
    "API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "CEREBRAS_API_KEY",
    "CLOUDFLARE_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "GITHUB_MODELS_API_KEY",
    "GITHUB_TOKEN",
    "HUGGINGFACE_API_KEY",
    "HF_TOKEN",
    "OMNIROUTE_API_KEY",
    "EXPLABS_API_KEY",
    "EXPERIENTIAL_ORG_KEY",
    "HF_HUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "GITHUB_PAT",
)

#: A secret shorter than this is not redacted by value: replacing a two-character
#: value would corrupt every message that happened to contain those characters.
#: Short "secrets" are not credentials.
_MIN_REDACTABLE_SECRET_CHARS = 12

#: Recognisable credential shapes, as a backstop for a secret that reached the
#: text without being in this process's environment -- for example one supplied
#: to a different deployment of the same code, or pasted from elsewhere.
#:
#: Each pattern keeps its leading marker so the message still says *which kind*
#: of credential appeared, which is more useful than a uniform placeholder.
_CREDENTIAL_SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"), "AIza"),
    (re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}"), "gh"),
    (re.compile(r"\bgsk_[0-9A-Za-z]{20,}"), "gsk"),
    (re.compile(r"\bxai-[0-9A-Za-z]{16,}"), "xai-"),
    (re.compile(r"\bhf_[0-9A-Za-z]{20,}"), "hf_"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.\.[A-Za-z0-9_\-]{10,}"), "ey"),
    # A PEM private key header, up to the first few lines of it.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
            r"(?:[A-Za-z0-9+/=\s]{0,4096}?-----END [A-Z ]*PRIVATE KEY-----)?"
        ),
        "-----BEGIN PRIVATE KEY-----",
    ),
)

#: A run of letters this long is not a LaTeX diagnostic.
#:
#: The longest word in TeX's own error vocabulary is well under this --
#: "Missing" is 7, "Unavailable" is 11, "environment" is 11. Forty is a wide
#: margin, so ordinary vocabulary is never at risk, while a CV paragraph or a
#: pasted credential collapsed into one token is caught.
_PATHOLOGICAL_WORD_RUN = 40

#: Redaction marker. States the length so a reader can tell a short token from a
#: pasted block without learning either.
_REDACTED = "<redacted:{n} chars>"


def _redact_secret_values(text: str) -> str:
    """
    Remove any known credential from ``text``.

    Two independent mechanisms, because either alone has a gap:

    * every current value of every secret-named environment variable, so a
      credential this deployment actually holds cannot survive into a log even
      if it has an unrecognisable shape;
    * recognisable credential shapes, so one this deployment does not hold -- a
      different machine's key pasted into a CV, a CI token -- is still caught.

    Comparison is plain substring matching on the value. The value is never
    echoed, not even in a length: the marker reports the length of the
    *replaced run*, which is the same number either way.
    """
    if not text:
        return text

    import os

    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if len(value) < _MIN_REDACTABLE_SECRET_CHARS:
            continue
        if value in text:
            text = text.replace(value, _REDACTED.format(n=len(value)))

    for pattern, _label in _CREDENTIAL_SHAPES:
        text = pattern.sub(lambda match: _REDACTED.format(n=len(match.group(0))), text)

    return text


def _collapse_pathological_runs(text: str) -> str:
    """
    Collapse letter runs far too long to be a TeX diagnostic.

    This is the only place document-shaped text is removed, and it is targeted:
    the threshold is set above the longest real TeX message, so a compiler
    message passes through untouched and only a pasted blob is caught.
    """
    return re.sub(
        r"[A-Za-z]{%d,}" % _PATHOLOGICAL_WORD_RUN,
        lambda match: _REDACTED.format(n=len(match.group(0))),
        text,
    )


def _redact_compiler_text(text: str) -> str:
    """
    Make a compiler message safe to show, without making it useless.

    Applied to the compiler's own diagnostic line, which is fixed vocabulary and
    carries the only explanation of the failure. It is preserved.

    Three things are removed, and only these three:

    * credentials, by value and by shape (:func:`_redact_secret_values`);
    * letter runs too long to be a TeX message
      (:func:`_collapse_pathological_runs`), which is how a pasted CV paragraph
      or a stray key is caught without touching real diagnostics;
    * the absolute path of the temporary working directory, which contains the
      operator's username.

    What is kept: the error category, the message in full, LaTeX special
    characters, file names, line numbers and any token the message quotes. A
    user reading this learns what went wrong and where, which is the entire
    purpose of a diagnostic.

    The length cap is applied last and trims on a word boundary, so a long
    message is shortened rather than cut mid-word into something misleading.
    """
    if not text:
        return ""

    # Collapse the whole log into one line first: a multi-line message would
    # otherwise smuggle content past a per-line rule.
    flattened = " ".join(text.split())

    flattened = _redact_secret_values(flattened)
    flattened = _collapse_pathological_runs(flattened)

    # The compiler runs in a private temp directory whose path contains the
    # operator's username. The directory is removed; the file name is kept,
    # because "cv.tex not found" and "cv-2026.tex not found" point at
    # different problems and losing it makes the message less useful.
    #
    # The trailing component is captured and re-emitted rather than matched to
    # the end of the token: a pattern that runs to the next space consumes the
    # file name along with the directory. A real directory always ends in a
    # separator, so the trailing separator is what bounds the match.
    #
    # The replacements are raw strings. re.sub parses a replacement as a
    # template, and a non-raw trailing backslash is an incomplete escape --
    # which raises only when a Windows path is actually present, so it survives
    # any test that does not happen to include one.
    # The directory part is matched greedily and the *final* path component is
    # captured, so the file name is the one that is kept. A lazy match stops at
    # the first separator and leaves most of the path -- including the username
    # -- in the message.
    flattened = re.sub(
        r"[A-Za-z]:(?:[^\\/\s'\"]*\\)+(?P<name>[^\\/\s'\"]+)",
        r"<workdir>\\\g<name>",
        flattened,
    )
    flattened = re.sub(
        r"(?<![A-Za-z0-9])/(?:tmp|var/folders|private/var)/"
        r"(?:[^/\s'\"]*/)*(?P<name>[^/\s'\"]+)",
        r"<workdir>/\g<name>",
        flattened,
    )

    flattened = " ".join(flattened.split())

    if len(flattened) <= _MAX_DETAIL_CHARS:
        return flattened

    # Trim on a word boundary. Cutting mid-word would produce a fragment that
    # reads like a complete sentence and misleads.
    cut = flattened[:_MAX_DETAIL_CHARS]
    space = cut.rfind(" ")

    if space > _MAX_DETAIL_CHARS // 2:
        cut = cut[:space]

    return f"{cut.rstrip()} [...]"


def _categorize_compiler_message(message: str) -> str:
    """Map a pdflatex message onto a stable error category."""
    lowered = message.casefold()

    for category, markers in _ERROR_CATEGORIES:
        if any(marker in lowered for marker in markers):
            return category

    return "compiler_error"


def _extract_escaped_token(message: str) -> str:
    """Name the offending LaTeX character if the message points at one."""
    match = re.search(r"character\s+(\S)", message)
    if match:
        candidate = match.group(1)
        if candidate in "&#$_{}\\%^~<>":
            return candidate

    lowered = message.casefold()
    if "misplaced alignment tab" in lowered or "alignment tab" in lowered:
        return "&"
    if "macro parameter character" in lowered:
        return "#"
    if "math shift" in lowered or "missing $ inserted" in lowered:
        return "$"

    return ""


def _read_text_file(path: Path | None) -> str:
    """Read a compiler artifact, treating any failure as empty."""
    if path is None:
        return ""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _build_latex_diagnostic(
    artifacts: tuple[Path, ...],
    source_lines: list[str],
) -> LaTeXCompilationError:
    """
    Turn pdflatex artifacts into a :class:`LaTeXCompilationError`.

    ``artifacts`` is ``(log, stdout, stderr)``.  The ``.log`` is authoritative
    and searched first; stdout is only consulted when the log is unavailable,
    which happens when the engine dies before writing one.
    """
    message = ""
    line_number: int | None = None
    offending_source = ""

    for path in artifacts:
        text = _read_text_file(path)
        if not text:
            continue

        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue

            match = _BANG_ERROR_RE.match(stripped)
            if match:
                message = match.group("message")
                break

            match = _FILE_LINE_ERROR_RE.match(stripped)
            if match:
                message = match.group("message")
                line_number = int(match.group("line"))
                break

        if message:
            break

    # The "l.<n> <text>" echo always follows the error and names the offending
    # line. It is only used to refine the line number, never to echo content.
    for path in artifacts:
        for raw in _read_text_file(path).splitlines():
            match = _CONTEXT_LINE_RE.match(raw.strip())
            if not match:
                continue
            if message and not _CATEGORY_TOKEN.get(_categorize_compiler_message(message)):
                continue
            if line_number is None:
                line_number = int(match.group("line"))
            offending_source = match.group("source")
            break

    if not message:
        return LaTeXCompilationError(
            "LaTeX compilation failed and the compiler produced no "
            "diagnostic. Check that the pdflatex installation is complete.",
            category="no_diagnostic",
        )

    category = _categorize_compiler_message(message)
    token = _extract_escaped_token(message) or _CATEGORY_TOKEN.get(category, "")

    # A source excerpt is only ever produced for the trusted preamble. Body
    # lines are the candidate's CV, so they are never included.
    excerpt = ""
    if line_number and line_number <= len(source_lines):
        candidate_line = source_lines[line_number - 1]
        if "\\begin{document}" not in candidate_line:
            excerpt = candidate_line[:_MAX_EXCERPT_CHARS]

    safe_message = _redact_compiler_text(message)
    parts = [f"LaTeX compilation failed [{category}]"]
    if line_number:
        parts.append(f"at line {line_number}")
    parts.append(f": {safe_message}")
    if token:
        parts.append(f" Unescaped LaTeX character {token!r} in generated CV text.")
    if excerpt:
        parts.append(f" Source: {excerpt}")
    if offending_source and not excerpt:
        # Prove the text was withheld rather than silently truncated away.
        parts.append(f" (source line withheld: {len(offending_source)} characters)")

    return LaTeXCompilationError(
        "".join(parts),
        category=category,
        line=line_number,
        detail=safe_message,
        token=token,
        log_excerpt=excerpt,
    )


def _strip_tex_comments(text: str) -> str:
    """Remove unescaped TeX comments, including line-ending obfuscation."""
    output: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char != "%":
            output.append(char)
            index += 1
            continue

        slash_count = 0
        probe = index - 1
        while probe >= 0 and text[probe] == "\\":
            slash_count += 1
            probe -= 1
        if slash_count % 2:
            output.append(char)
            index += 1
            continue

        # TeX comments consume the newline too.  Omitting it also canonicalizes
        # command-splitting tricks such as ``\\in% comment\nput``.
        index += 1
        while index < length and text[index] not in "\r\n":
            index += 1
        if (
            index < length
            and text[index] == "\r"
            and index + 1 < length
            and text[index + 1] == "\n"
        ):
            index += 2
        elif index < length:
            index += 1

    return "".join(output)


def _read_brace_group(text: str, start: int) -> tuple[str, int] | None:
    """Read one balanced, TeX-escaped brace group beginning at ``start``."""
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
        index += 1
    return None


def _validate_balanced_braces(text: str) -> None:
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise LaTeXSourceError("LaTeX source has unbalanced braces.")
        index += 1
    if depth:
        raise LaTeXSourceError("LaTeX source has unbalanced braces.")


def _validate_definitions(text: str) -> None:
    definition_counts: dict[str, int] = {}
    for match in _NEW_COMMAND_RE.finditer(text):
        index = match.end()
        while index < len(text) and text[index].isspace():
            index += 1
        parsed = _read_brace_group(text, index)
        if parsed is None:
            raise LaTeXSourceError("LaTeX source contains an invalid definition.")
        name_match = re.fullmatch(r"\s*\\([A-Za-z@]+)\s*", parsed[0])
        tail_match = re.match(r"[ \t\r\n]*\[(\d+)\]", text[parsed[1] :])
        if not name_match or not tail_match:
            raise LaTeXSourceError("LaTeX source contains an invalid definition.")
        name = name_match.group(1)
        if _ALLOWED_DEFINITIONS.get(name) != int(tail_match.group(1)):
            raise LaTeXSourceError("LaTeX source contains a forbidden definition.")
        definition_counts[name] = definition_counts.get(name, 0) + 1

    if any(count > 1 for count in definition_counts.values()):
        raise LaTeXSourceError("LaTeX source contains a duplicate definition.")

    renew_count = 0
    for match in _RENEW_COMMAND_RE.finditer(text):
        index = match.end()
        while index < len(text) and text[index].isspace():
            index += 1
        parsed = _read_brace_group(text, index)
        if parsed is None:
            raise LaTeXSourceError("LaTeX source contains an invalid definition.")
        name_match = re.fullmatch(r"\s*\\([A-Za-z@]+)\s*", parsed[0])
        if not name_match or name_match.group(1) not in _ALLOWED_RENEW_COMMANDS:
            raise LaTeXSourceError("LaTeX source contains a forbidden definition.")
        index = parsed[1]
        while index < len(text) and text[index].isspace():
            index += 1
        if _read_brace_group(text, index) is None:
            raise LaTeXSourceError("LaTeX source contains an invalid definition.")
        renew_count += 1

    if renew_count > 1:
        raise LaTeXSourceError("LaTeX source contains a duplicate definition.")


def _validate_packages(text: str) -> None:
    command_count = len(_USEPACKAGE_RE.findall(text))
    matches = list(_USEPACKAGE_START_RE.finditer(text))
    if command_count != len(matches):
        raise LaTeXSourceError("LaTeX source contains an invalid package declaration.")

    for match in matches:
        group = _read_brace_group(text, match.end() - 1)
        if group is None:
            raise LaTeXSourceError("LaTeX source contains an invalid package declaration.")
        package_names = [name.strip().lower() for name in group[0].split(",") if name.strip()]
        if not package_names or any(name not in _ALLOWED_PACKAGES for name in package_names):
            raise LaTeXSourceError("LaTeX source contains a forbidden LaTeX package.")


def sanitize_latex_url(value: str) -> str:
    """Validate and percent-encode a URL used in generated LaTeX links."""
    if not isinstance(value, str):
        raise LaTeXSourceError("LaTeX URL must be a string.")

    url = value.strip()

    # Auto-fix links missing a scheme (e.g., github.com/...)
    if url and not url.startswith(("http://", "https://", "mailto:")):
        if "@" in url and "." in url.split("@")[1]:
            url = "mailto:" + url
        else:
            url = "https://" + url

    if not url or len(url) > 2_048:
        raise LaTeXSourceError("LaTeX URL has an invalid length.")
    if any(char.isspace() for char in url) or _CONTROL_RE.search(url):
        raise LaTeXSourceError("LaTeX URL contains unsafe characters.")
    if any(char in url for char in "\\{}<>\"'`"):
        raise LaTeXSourceError("LaTeX URL contains unsafe characters.")

    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        if scheme not in {"http", "https", "mailto"}:
            raise LaTeXSourceError("LaTeX URL uses a forbidden scheme.")
        if scheme in {"http", "https"}:
            if not parsed.netloc or not parsed.hostname:
                raise LaTeXSourceError("LaTeX URL is not absolute.")
            if parsed.username is not None or parsed.password is not None:
                raise LaTeXSourceError("LaTeX URL must not contain credentials.")
            # Accessing these properties also validates malformed ports.
            _ = parsed.port
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", parsed.path):
            raise LaTeXSourceError("Mail-to URL is invalid.")
    except (TypeError, ValueError):
        raise LaTeXSourceError("LaTeX URL is invalid.") from None

    normalized = urlunsplit(parsed)
    return quote(normalized, safe=_URL_SAFE_CHARS)


def _unwrap_detokenize(value: str) -> str:
    value = value.strip()
    if not value.startswith("\\detokenize"):
        return value
    parsed = _read_brace_group(value, value.find("{"))
    return parsed[0].strip() if parsed is not None else ""


def _validate_body_links(body: str) -> None:
    if (
        _FORBIDDEN_COMMAND_RE.search(body)
        or _DRIVER_PRIMITIVE_RE.search(body)
        or _EXPL3_COMMAND_RE.search(body)
    ):
        # Kept as one branch for readability; primitive-specific messages are
        # intentionally avoided so malformed source is not reflected back.
        raise LaTeXSourceError("LaTeX body contains a forbidden command.")

    for match in _ENVIRONMENT_RE.finditer(body):
        if match.group(1).casefold() in _FORBIDDEN_ENVIRONMENTS:
            raise LaTeXSourceError("LaTeX body contains a forbidden environment.")

    link_command = re.compile(r"\\(?:href|url|hrlink)(?![A-Za-z@])", re.IGNORECASE)
    for match in link_command.finditer(body):
        index = match.end()
        while index < len(body) and body[index].isspace():
            index += 1
        parsed = _read_brace_group(body, index)
        if parsed is None:
            raise LaTeXSourceError("LaTeX body contains an invalid link command.")
        url = _unwrap_detokenize(parsed[0])
        # \relax is LaTeX's "no link", not a URL. The project-header branch below
        # already exempts it; without the same exemption here, a document whose
        # only defect was an unusable URL would be refused by the very validator
        # that rescued it. It cannot carry an injection: it is a control sequence
        # the templates already emit.
        if url and url.casefold() != r"\relax":
            sanitize_latex_url(url)

    for match in re.finditer(r"\\projheader(?![A-Za-z@])", body, re.IGNORECASE):
        arguments: list[str] = []
        index = match.end()
        for _ in range(3):
            while index < len(body) and body[index].isspace():
                index += 1
            parsed = _read_brace_group(body, index)
            if parsed is None:
                raise LaTeXSourceError("LaTeX body contains an invalid project header.")
            arguments.append(parsed[0])
            index = parsed[1]
        project_url = _unwrap_detokenize(arguments[2])
        if project_url and project_url.casefold() != r"\relax":
            sanitize_latex_url(project_url)


def _validate_common_source(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise LaTeXSourceError("LaTeX source must be a non-empty string.")
    if len(source) > MAX_LATEX_SOURCE_CHARS:
        raise LaTeXSourceError("LaTeX source exceeds the size limit.")
    if len(source.encode("utf-8")) > MAX_LATEX_SOURCE_BYTES:
        raise LaTeXSourceError("LaTeX source exceeds the size limit.")
    if _CONTROL_RE.search(source):
        raise LaTeXSourceError("LaTeX source contains forbidden control characters.")
    if any(len(line) > MAX_LATEX_LINE_CHARS for line in source.splitlines()):
        raise LaTeXSourceError("LaTeX source contains an overlong line.")

    text = _strip_tex_comments(source)
    _validate_balanced_braces(text)
    if (
        _FORBIDDEN_COMMAND_RE.search(text)
        or _DRIVER_PRIMITIVE_RE.search(text)
        or _EXPL3_COMMAND_RE.search(text)
    ):
        raise LaTeXSourceError("LaTeX source contains a forbidden command.")
    for match in _ENVIRONMENT_RE.finditer(text):
        if match.group(1).casefold() in _FORBIDDEN_ENVIRONMENTS:
            raise LaTeXSourceError("LaTeX source contains a forbidden environment.")
    return text


def validate_latex_body(source: str) -> str:
    """Validate an LLM-produced body before it is inserted into a template."""
    text = _validate_common_source(source)
    if (
        _DOCUMENT_CLASS_COMMAND_RE.search(text)
        or _BEGIN_DOCUMENT_RE.search(text)
        or _END_DOCUMENT_RE.search(text)
    ):
        raise LaTeXSourceError("LaTeX body must not contain a document wrapper.")
    if _USEPACKAGE_RE.search(text):
        raise LaTeXSourceError("LaTeX body must not contain package declarations.")
    _validate_definitions(text)
    _validate_body_links(text)
    return source


def validate_latex_document(source: str) -> str:
    """Validate a complete application-generated LaTeX document."""
    text = _validate_common_source(source)
    if (
        len(_DOCUMENT_CLASS_COMMAND_RE.findall(text)) != 1
        or len(_DOCUMENT_CLASS_RE.findall(text)) != 1
    ):
        raise LaTeXSourceError("LaTeX document must use the supported A4 article class.")
    if len(_BEGIN_DOCUMENT_RE.findall(text)) != 1 or len(_END_DOCUMENT_RE.findall(text)) != 1:
        raise LaTeXSourceError("LaTeX document must have one document environment.")

    begin_match = _BEGIN_DOCUMENT_RE.search(text)
    end_match = _END_DOCUMENT_RE.search(text)
    if begin_match is None or end_match is None:
        raise LaTeXSourceError("LaTeX document environment is invalid.")
    if end_match.start() <= begin_match.end():
        raise LaTeXSourceError("LaTeX document environment is invalid.")
    if text[end_match.end() :].strip():
        raise LaTeXSourceError("LaTeX document contains trailing content.")

    preamble = text[: begin_match.start()]
    body = text[begin_match.end() : end_match.start()]
    _validate_packages(preamble)
    _validate_definitions(preamble)
    if (
        _USEPACKAGE_RE.search(body)
        or _NEW_COMMAND_RE.search(body)
        or _RENEW_COMMAND_RE.search(body)
    ):
        raise LaTeXSourceError("LaTeX document body contains a forbidden declaration.")
    _validate_body_links(body)
    return source


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the actual number of pages in a PDF page tree."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        return len(reader.pages)
    except Exception:
        # Parser diagnostics may contain fragments of generated source.
        raise RuntimeError("Generated PDF is unreadable.") from None


#: A pass is only started if at least this much of the total budget is left.
#:
#: Two LaTeX passes on an already-loaded document take well under a second each;
#: the first pass also pays for package loading. A larger document is slower, so
#: this is set below a single worst-case pass rather than above it -- the point
#: is to avoid starting an attempt that cannot finish, not to be clever.
_MIN_REMAINING_BUDGET_SECONDS = 8.0


def _compaction_time_remains(deadline: float) -> bool:
    """True when another compilation pass could plausibly complete."""
    return (deadline - time.monotonic()) > _MIN_REMAINING_BUDGET_SECONDS


#: A4 in PostScript points. Used to turn an absolute text position into a
#: fraction of the page.
_PAGE_HEIGHT_PT = 841.89

#: Below this much empty space at the foot of the page, the document is left
#: alone. A CV that ends 10% up the page has a normal bottom margin; one that
#: ends a third of the way up does not.
FILL_GAP_THRESHOLD = 0.18

#: Where an expanded document should end up, as a fraction of the page.
FILL_TARGET_GAP = 0.06

#: The most the leading is ever multiplied by.
#:
#: 1.35 is the point where 11pt text on 13.2pt leading starts to look loose
#: rather than airy. Past that, a better answer is a longer CV, and a CV is
#: exactly what the candidate controls.
FILL_MAX_FACTOR = 1.35

#: Where the ``\linespread`` is inserted, matching the compaction levels.
_FILL_ANCHOR = "\\begin{document}"


def measured_fill_gap(pdf_bytes: bytes) -> float | None:
    """
    How much of the page height is empty below the lowest text, as a fraction.

    ``None`` when the positions cannot be read, which is treated as "leave it
    alone" rather than as "expand": a measurement failure is not evidence that a
    document is short.

    The page origin is bottom-left, so the *lowest* baseline is the smallest y.
    Whatever sits below it -- the bottom margin, normally around 2cm -- is not
    emptiness, so the margin is subtracted before the figure is compared with the
    threshold. Without that, every document looks like it has a 7% gap and the
    pass would engage for the wrong reason.
    """
    import io

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    except Exception:
        return None

    positions: list[float] = []

    for page in reader.pages:

        def visitor(text, cm, tm, font_dict, font_size, chunk=None):
            if text and text.strip():
                positions.append(float(tm[5]) + float(cm[5]))

        try:
            page.extract_text(visitor_text=visitor)
        except Exception:
            return None

    if not positions:
        return None

    lowest = min(positions)
    if lowest < 0 or lowest > _PAGE_HEIGHT_PT:
        return None

    # A standard 2cm bottom margin, so "empty" means empty rather than "not yet
    # at the margin".
    margin_pt = 56.7
    gap_pt = max(0.0, lowest - margin_pt)

    return gap_pt / _PAGE_HEIGHT_PT


def _expand_layout(latex_code: str, factor: float) -> str:
    """
    Multiply the leading by ``factor``.

    Deliberately the same mechanism the compaction levels use, so expansion and
    compaction are two ends of one dial rather than two unrelated mechanisms that
    can fight: whichever runs last decides the leading, and the fill pass only
    ever runs on a document that already fits.
    """
    if factor <= 1.0:
        return latex_code

    rounded = round(factor, 3)
    directive = f"\\linespread{{{rounded}}}\\selectfont"

    # A document that already carries a line spread -- a compacted one -- has its
    # value replaced rather than having a second one stacked on top, which LaTeX
    # would ignore in favour of the first.
    # The replacement is a function, not a string. re.sub parses a string
    # replacement as a template, and a LaTeX directive begins with a backslash,
    # which is an invalid group reference there -- so a plain string replacement
    # raises instead of matching. A function's return value is used literally,
    # which is what a LaTeX command needs.
    if "\\linespread" in latex_code:
        return re.sub(
            r"\\linespread\{[\d.]+\}\\selectfont",
            lambda _match: directive,
            latex_code,
            count=1,
        )

    if _FILL_ANCHOR not in latex_code:
        return latex_code

    return latex_code.replace(_FILL_ANCHOR, f"{_FILL_ANCHOR}\n{directive}", 1)


def _fill_factor_for(gap: float) -> float:
    """
    The leading multiplier that would close ``gap`` down to the target.

    Returns ``1.0`` -- meaning "leave it alone" -- when the gap is already small
    or the required factor exceeds the cap.
    """
    if gap is None or gap <= FILL_GAP_THRESHOLD:
        return 1.0

    # The occupied fraction is what has to grow; the factor is applied to
    # leading, which scales the occupied height, so the two are the same ratio
    # once the fixed margins are accounted for.
    occupied = 1.0 - gap
    if occupied <= 0:
        return 1.0

    wanted = (1.0 - FILL_TARGET_GAP) / occupied
    if wanted >= FILL_MAX_FACTOR:
        # Too short to reach the target within the cap. Expand as far as is
        # allowed rather than not at all: a partial improvement is still an
        # improvement, and the cap is what keeps it from looking loose.
        return FILL_MAX_FACTOR

    return round(max(1.0, wanted), 3)


def _compact_layout(latex_code: str) -> str:
    """Apply level-one spacing compaction without discarding content."""
    code = latex_code
    code = re.sub(
        r"\\titlespacing\{\\section\}\{[^{}]*\}\{[^{}]*\}\{[^{}]*\}",
        lambda _match: r"\titlespacing{\section}{0pt}{1pt}{0pt}",
        code,
    )
    for key in ("itemsep", "topsep", "parsep", "partopsep"):
        code = re.sub(
            rf"{key}=\d+(?:\.\d+)?pt",
            f"{key}=0pt",
            code,
        )
    return code


def _compact_layout_level(latex_code: str, level: int) -> str:
    """Apply cumulative, content-preserving one-page compaction."""
    if level <= 0:
        return latex_code

    code = _compact_layout(latex_code)
    if level >= 2:
        if "\\linespread" not in code:
            code = code.replace(
                "\\begin{document}",
                "\\begin{document}\n\\linespread{0.94}\\selectfont",
                1,
            )
        code = re.sub(
            r"\\usepackage\[[^\]]*\]\{geometry\}",
            lambda _match: (
                r"\usepackage["
                r"top=0.5cm,"
                r"bottom=0.5cm,"
                r"left=0.8cm,"
                r"right=0.8cm"
                r"]{geometry}"
            ),
            code,
        )
    if level >= 3:
        code = code.replace(
            r"\documentclass[11pt,a4paper]{article}",
            r"\documentclass[10pt,a4paper]{article}",
        )
    return code


def _safe_tex_environment() -> dict[str, str]:
    """Build a minimal environment so API keys and cloud credentials are not inherited."""
    env = {key: value for key, value in os.environ.items() if key.upper() in _SAFE_ENVIRONMENT_KEYS}
    env.update(
        {
            "FORCE_SOURCE_DATE": "1",
            "MIKTEX_AUTOINSTALL": "0",
            "MIKTEX_ENABLE_INSTALLER": "0",
            "MIKTEX_GUI_MODE": "no",
            "SOURCE_DATE_EPOCH": "0",
            "openin_any": "p",
            "openout_any": "p",
            "shell_escape": "f",
        }
    )
    return env


def _artifact_size(paths: tuple[Path, ...]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _workdir_size(path: Path) -> int:
    total = 0
    try:
        entries = path.iterdir()
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _run_pdflatex(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_path: Path,
    artifact_paths: tuple[Path, ...],
    timeout_seconds: float,
    source_lines: list[str] | None = None,
) -> None:
    """
    Run one pdflatex pass with hard time and artifact-size bounds.

    On a non-zero exit the compiler's own diagnostic is read out of the ``.log``
    and attached to the raised error.  The previous version logged
    ``exit_code=1`` and raised "LaTeX compilation failed.", which threw away the
    only information that identifies the fault.
    """
    process_kwargs = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "check": False,
        "timeout": timeout_seconds,
        "shell": False,
        "close_fds": True,
    }
    if os.name == "nt":
        process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_kwargs["start_new_session"] = True

    with (
        open(artifact_paths[1], "w", encoding="utf-8", errors="replace") as stdout_file,
        open(artifact_paths[2], "w", encoding="utf-8", errors="replace") as stderr_file,
    ):
        try:
            process = subprocess.run(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                **process_kwargs,
            )
        except subprocess.TimeoutExpired:
            raise LaTeXCompilationError(
                "LaTeX compilation timed out after " f"{timeout_seconds:.0f}s.",
                category="timeout",
            ) from None
        except OSError:
            raise LaTeXCompilationError(
                "pdflatex could not be started. Verify the TeX installation.",
                category="compiler_missing",
            ) from None

    if _artifact_size(artifact_paths) > MAX_TEX_ARTIFACT_BYTES:
        raise LaTeXCompilationError("LaTeX compiler output exceeded the size limit.")
    if _workdir_size(cwd) > MAX_TEX_WORKDIR_BYTES:
        raise LaTeXCompilationError("LaTeX temporary output exceeded the size limit.")
    if process.returncode != 0:
        diagnostic = _build_latex_diagnostic(
            artifact_paths,
            source_lines or [],
        )
        logger.warning(
            "pdflatex failed (exit_code=%s category=%s line=%s): %s",
            process.returncode,
            diagnostic.category,
            diagnostic.line,
            diagnostic.detail or diagnostic,
        )
        raise diagnostic
    if not output_path.exists():
        raise LaTeXCompilationError(
            "LaTeX compilation did not produce a PDF.",
            category="no_output",
        )
    try:
        output_size = output_path.stat().st_size
    except OSError:
        raise LaTeXCompilationError(
            "Could not inspect the generated PDF.",
            category="unreadable_output",
        ) from None
    if output_size > MAX_PDF_BYTES:
        raise LaTeXCompilationError("Generated PDF exceeded the size limit.")


def _has_vertical_overflow(log_text: str) -> bool:
    return bool(
        re.search(
            r"Overfull\s+\\vbox\s+\([^)]*too high[^)]*\)",
            log_text,
            flags=re.IGNORECASE,
        )
    )


def _safe_document_name(document_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", document_name):
        raise ValueError("Invalid compiler document name.")
    return document_name


def _compile_attempt(
    latex_code: str,
    *,
    pdflatex: str,
    workdir: Path,
    document_name: str = "document",
    deadline: float | None = None,
) -> bytes:
    """Compile one source variant using two isolated pdflatex passes."""
    document_name = _safe_document_name(document_name)
    source = validate_latex_document(latex_code)

    tex_path = workdir / f"{document_name}.tex"
    pdf_path = workdir / f"{document_name}.pdf"
    log_path = workdir / f"{document_name}.log"
    stdout_path = workdir / "pdflatex.out"
    stderr_path = workdir / "pdflatex.err"

    try:
        for stale_path in (pdf_path, log_path, stdout_path, stderr_path):
            stale_path.unlink(missing_ok=True)
        tex_path.write_text(source, encoding="utf-8")
    except OSError:
        raise LaTeXCompilationError(
            "Could not prepare the LaTeX temporary file.",
            category="tempfile_unwritable",
        ) from None

    command = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        f"-jobname={document_name}",
        f"-output-directory={workdir}",
        tex_path.name,
    ]
    env = _safe_tex_environment()
    artifacts = (log_path, stdout_path, stderr_path)
    source_lines = source.splitlines()

    for _ in range(2):
        if deadline is None:
            timeout_seconds = float(PDFL_COMPILER_PASS_TIMEOUT_SECONDS)
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LaTeXCompilationError(
                    "LaTeX compilation exceeded the total time limit.",
                    category="timeout",
                )
            timeout_seconds = min(float(PDFL_COMPILER_PASS_TIMEOUT_SECONDS), remaining)
        _run_pdflatex(
            command,
            cwd=workdir,
            env=env,
            output_path=pdf_path,
            artifact_paths=artifacts,
            timeout_seconds=timeout_seconds,
            source_lines=source_lines,
        )

    try:
        if pdf_path.stat().st_size > MAX_PDF_BYTES:
            raise LaTeXCompilationError("Generated PDF exceeded the size limit.")
        pdf_bytes = pdf_path.read_bytes()
    except LaTeXCompilationError:
        raise
    except OSError:
        raise LaTeXCompilationError(
            "Could not read the generated PDF.",
            category="unreadable_output",
        ) from None
    page_count = pdf_page_count(pdf_bytes)

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""
    if _has_vertical_overflow(log_text):
        raise PDFLayoutError(
            page_count,
            "Generated PDF content extends outside the page boundary.",
        )
    return pdf_bytes


def _make_private_temp_directory(prefix: str) -> tempfile.TemporaryDirectory:
    directory = tempfile.TemporaryDirectory(prefix=prefix)
    if os.name != "nt":
        try:
            os.chmod(directory.name, 0o700)
        except OSError:
            pass
    return directory


def _fill_page(
    pdf_bytes: bytes,
    latex_code: str,
    pdflatex: str,
    workdir: Path,
    deadline: float,
) -> bytes:
    """
    Open up the leading of a document that stops far too high on the page.

    Only ever called for a document already known to be exactly one page. The
    original is returned whenever anything about the fill cannot be established
    or the expanded attempt does not fit, so this can cost a compile and cannot
    cost a document.
    """
    gap = measured_fill_gap(pdf_bytes)
    factor = _fill_factor_for(gap)

    if factor <= 1.0:
        return pdf_bytes

    # Enough budget left for the extra pass, or leave it: a timeout here would
    # replace a good PDF with an error, which is the opposite of the intent.
    if not _compaction_time_remains(deadline):
        logger.debug(
            "Document leaves %.0f%% of the page empty, but there is not enough "
            "time left to open up the leading.",
            gap * 100,
        )
        return pdf_bytes

    expanded_source = _expand_layout(latex_code, factor)
    if expanded_source == latex_code:
        return pdf_bytes

    logger.info(
        "Document ends %.0f%% up an empty page; opening up the leading by %.2fx.",
        gap * 100,
        factor,
    )

    try:
        expanded = _compile_attempt(
            expanded_source,
            pdflatex=pdflatex,
            workdir=workdir,
            deadline=deadline,
        )
    except (PDFLayoutError, LaTeXCompilationError, LaTeXSourceError) as exc:
        # The guess was wrong, or the expansion produced something invalid. The
        # unexpanded document is still correct, so it is what ships.
        logger.info(
            "Expanded leading did not compile to one page (%s); keeping the "
            "unexpanded document.",
            type(exc).__name__,
        )
        return pdf_bytes

    if pdf_page_count(expanded) != 1:
        logger.info(
            "Expanded leading produced %d pages; keeping the unexpanded document.",
            pdf_page_count(expanded),
        )
        return pdf_bytes

    new_gap = measured_fill_gap(expanded)
    if new_gap is not None and new_gap > gap:
        # Opening the leading up made the page emptier, which can happen if the
        # expansion pushed a heading onto its own line. Keep whichever is denser.
        logger.info(
            "Expanded leading left more space empty (%.0f%% vs %.0f%%); keeping "
            "the unexpanded document.",
            new_gap * 100,
            gap * 100,
        )
        return pdf_bytes

    logger.info(
        "Trailing whitespace reduced from %.0f%% to %.0f%%.",
        gap * 100,
        (new_gap or 0.0) * 100,
    )
    return expanded


def compile_single_page_pdf(latex_code: str) -> bytes:
    """Validate, compile, and require exactly one page without leaking logs."""
    validate_latex_document(latex_code)

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise LaTeXCompilationError("pdflatex was not found on PATH. Install MiKTeX or TeX Live.")

    with _make_private_temp_directory("resume-pdf-") as temporary_directory:
        workdir = Path(temporary_directory).resolve()
        deadline = time.monotonic() + PDFL_COMPILER_TOTAL_TIMEOUT_SECONDS
        last_pages = 0
        last_overflow_error: PDFLayoutError | None = None

        # Uncompacted first: the common case costs one pass and the document is
        # delivered exactly as generated.
        for level in (0, 1, 2, 3):
            if level > 0 and not _compaction_time_remains(deadline):
                # Escalating now would consume the remaining budget without
                # enough time to finish, turning a clear layout error into a
                # timeout. Stop and report what the last attempt measured.
                logger.info(
                    "Not escalating past compaction level %d: insufficient "
                    "time remains in the compilation budget.",
                    level - 1,
                )
                break

            current_source = latex_code if level == 0 else _compact_layout_level(latex_code, level)
            final_attempt = level == 3

            try:
                pdf_bytes = _compile_attempt(
                    current_source,
                    pdflatex=pdflatex,
                    workdir=workdir,
                    deadline=deadline,
                )
            except PDFLayoutError as exc:
                last_pages = exc.pages
                last_overflow_error = exc
                if final_attempt:
                    raise
                continue

            pages = pdf_page_count(pdf_bytes)
            last_pages = pages
            if pages == 1:
                return _fill_page(pdf_bytes, current_source, pdflatex, workdir, deadline)
            if final_attempt:
                raise PDFLayoutError(
                    pages,
                    "Generated document must be exactly one page after safe compaction.",
                )

        if last_overflow_error is not None:
            raise last_overflow_error
        raise PDFLayoutError(
            last_pages,
            "Generated document must be exactly one page.",
        )


__all__ = [
    "MAX_LATEX_SOURCE_CHARS",
    "MAX_LATEX_SOURCE_BYTES",
    "MAX_PDF_BYTES",
    "MAX_TEX_ARTIFACT_BYTES",
    "MAX_TEX_WORKDIR_BYTES",
    "PDFL_COMPILER_PASS_TIMEOUT_SECONDS",
    "PDFL_COMPILER_TOTAL_TIMEOUT_SECONDS",
    "LaTeXCompilationError",
    "LaTeXSourceError",
    "PDFLayoutError",
    "compile_single_page_pdf",
    "pdf_page_count",
    "sanitize_latex_url",
    "validate_latex_body",
    "validate_latex_document",
]
