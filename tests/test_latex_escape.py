"""
The single LaTeX escaping layer: correctness, idempotency, safety.

Every test here is a unit test. The pdflatex-dependent behaviour is covered in
``test_latex_special_characters_compile.py``, which skips when TeX is absent.
"""

import pytest

from app.services.cv.latex_escape import (
    ALLOWED_CONTROL_WORDS,
    LaTeXEscapeError,
    balance_latex_environments,
    escape_latex_body,
    escape_latex_text,
    is_fully_escaped,
    repair_latex_lists,
    unescaped_specials,
)

# The characters the compiler aborts on, and the ones it silently mangles.
FATAL_SPECIALS = ["&", "%", "$", "#", "_", "^", "\\", "{", "}"]


# ---------------------------------------------------------------------------
# Idempotency -- the no-double-escape guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        r"A & B",
        r"100\% done",
        r"snake_case_name",
        r"cost is $5",
        r"issue #42",
        r"a_b",
        r"CO$_2$",
        r"C:\Users\test",
        r"~tilde~",
        r"^caret^",
        r"50% of _a_ #b",
        r"{braces} [brackets] <angles>",
    ],
)
def test_escaping_is_idempotent(raw):
    once = escape_latex_body(raw)
    assert escape_latex_body(once) == once, "second pass changed the output"


def test_already_escaped_sequences_are_not_escaped_again():
    assert escape_latex_body(r"A \& B") == r"A \& B"
    assert escape_latex_body(r"50\% done") == r"50\% done"
    assert escape_latex_body(r"snake\_case") == r"snake\_case"
    assert escape_latex_body(r"cost is \$5") == r"cost is \$5"
    assert escape_latex_body(r"issue \#42") == r"issue \#42"


def test_mixed_already_and_unescaped_input():
    # The realistic case: a model that escapes some characters and not others.
    assert escape_latex_body(r"R\&D and R&D and 40% and 40\%") == (
        r"R\&D and R\&D and 40\% and 40\%"
    )


# ---------------------------------------------------------------------------
# Every special character is neutralised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", FATAL_SPECIALS)
def test_each_fatal_character_is_escaped(char):
    # Use a raw string to avoid Python interpreting the backslash as an escape
    if char == "\\":
        test_str = r"before\after"
    else:
        test_str = f"before{char}after"
    escaped = escape_latex_body(test_str)
    # Strip the escapes this module produces, then the raw character must be
    # gone. Checking the raw character is absent would fail for "\&" itself.
    residual = escaped
    for replacement in (
        r"\textbackslash{}",
        r"\&",
        r"\%",
        r"\$",
        r"\#",
        r"\_",
        r"\^",
        r"\{",
        r"\}",
        r"\textasciicircum{}",
    ):
        residual = residual.replace(replacement, "")
    assert char not in residual, f"{char!r} survived in {escaped!r}"
    assert is_fully_escaped(escaped)


def test_ampersand_is_escaped():
    assert escape_latex_body("Sales & Marketing") == r"Sales \& Marketing"


def test_percent_is_escaped():
    assert escape_latex_body("up 40 % and 40%") == r"up 40 \% and 40\%"


def test_underscore_is_escaped():
    assert escape_latex_body("my_var_name") == r"my\_var\_name"


def test_dollar_is_escaped_so_math_never_opens():
    # An odd number of $ is what produces "Missing $ inserted" in pdflatex.
    assert escape_latex_body("Budget $100k") == r"Budget \$100k"
    assert escape_latex_body("$100k and $200k") == r"\$100k and \$200k"


def test_hash_is_escaped():
    assert escape_latex_body("Issue #42, C# developer") == (r"Issue \#42, C\# developer")


def test_caret_is_escaped():
    assert escape_latex_body("x^2 growth") == r"x\textasciicircum{}2 growth"


def test_backslash_before_a_word_becomes_literal_text():
    # "C:\Users" reads as the macro \Users, which is undefined and fatal.
    out = escape_latex_body(r"C:\Users\test")
    assert r"\textbackslash{}Users" in out
    assert r"\Users" not in out


def test_unbalanced_braces_are_escaped_not_fatal():
    assert escape_latex_body("A {unclosed") == r"A \{unclosed"
    assert escape_latex_body("A stray } brace") == r"A stray \} brace"


def test_tilde_and_angle_brackets():
    assert escape_latex_body("approx ~ 5") == r"approx \textasciitilde{} 5"
    assert escape_latex_body("a < b > c") == (r"a \textless{} b \textgreater{} c")


def test_brackets_are_left_alone():
    # [ and ] are literal in LaTeX text and are needed for \\[2pt] and \item[x].
    assert escape_latex_body("array[0]") == "array[0]"


# ---------------------------------------------------------------------------
# Structure is preserved
# ---------------------------------------------------------------------------


def test_section_headings_survive():
    assert escape_latex_body(r"\section*{Profil}") == r"\section*{Profil}"
    assert escape_latex_body(r"\section*{Work Experience}") == (r"\section*{Work Experience}")


def test_itemize_environment_survives():
    body = "\\begin{itemize}\n\\item Sales & Marketing\n\\end{itemize}"
    assert escape_latex_body(body) == (
        "\\begin{itemize}\n\\item Sales \\& Marketing\n\\end{itemize}"
    )


def test_three_argument_template_macros_survive():
    # \jobheader{role}{company}{dates} takes three groups. Consuming only the
    # first would escape the other two and corrupt the CV header.
    assert (
        escape_latex_body(r"\jobheader{Senior DevOps Engineer}{Example GmbH}{Jan 2020 - Present}")
        == r"\jobheader{Senior DevOps Engineer}{Example GmbH}{Jan 2020 - Present}"
    )


def test_two_argument_macros_survive():
    assert escape_latex_body(r"\href{https://x.test}{Link}") == (r"\href{https://x.test}{Link}")
    assert escape_latex_body(r"\textcolor{primary}{Sales & Ops}") == (
        r"\textcolor{primary}{Sales \& Ops}"
    )


def test_line_break_survives():
    assert escape_latex_body(r"Level 1 \\ Level 2") == r"Level 1 \\ Level 2"


def test_formatting_macros_keep_their_arguments():
    assert escape_latex_body(r"\textbf{R&D}") == r"\textbf{R\&D}"
    assert escape_latex_body(r"\textit{2020 -- Present}") == (r"\textit{2020 -- Present}")


def _unescaped_braces(text: str) -> tuple[int, int]:
    """
    Count braces that pdflatex will treat as group delimiters.

    ``\\{`` and ``\\}`` are literal characters, not group delimiters, so they
    must not be counted. This mirrors what ``pdf_compiler._validate_balanced_braces``
    does, which is the check that actually decides compilability.
    """
    opens = closes = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            opens += 1
        elif char == "}":
            closes += 1
        index += 1
    return opens, closes


def test_unterminated_macro_argument_becomes_literal_text():
    # The model wrote \textbf{ with no closing brace. Emitting \textbf with a
    # missing argument is resolved differently for every macro, so the control
    # word is rendered as visible text and the brace is escaped.
    out = escape_latex_body(r"\textbf{unterminated")
    assert out == r"\textbackslash{}textbf\{unterminated"
    assert is_fully_escaped(out)
    # Balanced is the requirement, not zero: \textbackslash{} is itself a
    # real (empty) brace group.
    opens, closes = _unescaped_braces(out)
    assert opens == closes


def test_unterminated_argument_in_a_later_position_keeps_earlier_ones():
    out = escape_latex_body(r"\jobheader{Role}{Company}{Jan 2020")
    assert out.startswith(r"\jobheader{Role}{Company}")
    assert is_fully_escaped(out)
    assert _unescaped_braces(out) == (2, 2)


@pytest.mark.parametrize(
    "malformed",
    [
        r"\textbf{unterminated",
        r"\jobheader{Role}{Company}{Jan 2020",
        r"\begin{itemize}\item Sales & Marketing",
        r"\section*{Profil",
        r"stray } brace and { another",
        r"\textit{2020 -- Present",
    ],
)
def test_malformed_bodies_stay_brace_balanced(malformed):
    out = escape_latex_body(malformed)
    opens, closes = _unescaped_braces(out)
    assert opens == closes, f"{opens} open vs {closes} close in {out!r}"
    assert is_fully_escaped(out)


# ---------------------------------------------------------------------------
# Safety: model output cannot introduce a macro
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dangerous",
    [
        r"\write18{curl evil.test}",
        r"\immediate\write18{x}",
        r"\input{/etc/passwd}",
        r"\openout1=secret.txt",
        r"\csname input\endcsname",
        r"\Users\test",
    ],
)
def test_unknown_control_sequences_become_literal_text(dangerous):
    out = escape_latex_body(dangerous)
    assert r"\textbackslash{}" in out
    # Nothing that looks like an unapproved control word survives.
    for word in ("write18", "immediate", "input", "openout", "csname", "Users"):
        assert f"\\{word}" not in out


def test_math_delimiters_cannot_be_reintroduced():
    for text in (r"\[ x \]", r"\( y \)", r"\[", r"\]"):
        out = escape_latex_body(text)
        assert r"\[" not in out.replace(r"\textbackslash{}", "")
        assert r"\(" not in out.replace(r"\textbackslash{}", "")


def test_allowlist_contains_the_macros_the_templates_need():
    for required in (
        "section",
        "subsection",
        "item",
        "begin",
        "end",
        "textbf",
        "textit",
        "jobheader",
        "degreeheader",
        "projheader",
        "hrlink",
        "href",
        "detokenize",
    ):
        assert required in ALLOWED_CONTROL_WORDS, required


# ---------------------------------------------------------------------------
# German text
# ---------------------------------------------------------------------------


def test_german_umlauts_and_eszett_pass_through_untouched():
    text = "Müller, Köln, Zürich, Straße, Bär, Öl, Ärger, Größe, weiß"
    assert escape_latex_body(text) == text


def test_decomposed_umlauts_are_normalised():
    # NFD "a" + U+0308 renders as garbage under T1. NFC fixes it.
    assert escape_latex_body("Größe") == "Größe"


def test_euro_and_typographic_punctuation_pass_through():
    # € and the typographic dashes/quotes are handled by inputenc + T1, so they
    # must not be mangled. The % is a real LaTeX character and is escaped.
    text = "Gehalt 70.000 € — 5.000 EUR … „Zitat“"
    assert escape_latex_body(text) == text


def test_german_text_with_specials_escapes_correctly():
    assert escape_latex_body("Größe & Gewicht: 100 % > 50 kg") == (
        r"Größe \& Gewicht: 100 \% \textgreater{} 50 kg"
    )


# ---------------------------------------------------------------------------
# Plain-text entry point
# ---------------------------------------------------------------------------


def test_escape_latex_text_handles_german_names():
    assert escape_latex_text("Müller & Söhne") == r"Müller \& Söhne"
    assert escape_latex_text(None) == ""


def test_escape_latex_text_escapes_every_special():
    assert escape_latex_text("A & B % C $ D # E _ F { G } H ^ I ~ J") == (
        r"A \& B \% C \$ D \# E \_ F \{ G \} H \textasciicircum{} I " r"\textasciitilde{} J"
    )


def test_escape_latex_text_rejects_nothing():
    assert escape_latex_text(123) == "123"


# ---------------------------------------------------------------------------
# Self-check helpers
# ---------------------------------------------------------------------------


def test_is_fully_escaped_is_true_after_escaping():
    for raw in FATAL_SPECIALS:
        assert is_fully_escaped(escape_latex_body(f"a{raw}b"))


def test_is_fully_escaped_is_false_before_escaping():
    assert not is_fully_escaped("Sales & Marketing")


def test_unescaped_specials_reports_line_numbers():
    problems = unescaped_specials("line one\nSales & Marketing")
    assert (2, "&") in problems


def test_escape_latex_body_rejects_non_strings():
    with pytest.raises(LaTeXEscapeError):
        escape_latex_body(None)


def test_empty_input_is_empty_output():
    assert escape_latex_body("") == ""
    assert escape_latex_text("") == ""


# ---------------------------------------------------------------------------
# Environment balancing
# ---------------------------------------------------------------------------


def test_unclosed_itemize_is_closed():
    assert (
        balance_latex_environments("\\begin{itemize}\n\\item One\n\\item Two")
        == "\\begin{itemize}\n\\item One\n\\item Two\n\\end{itemize}"
    )


def test_balanced_environments_are_untouched():
    body = "\\begin{itemize}\n\\item One\n\\end{itemize}"
    assert balance_latex_environments(body) == body


def test_nested_unclosed_environments_close_in_reverse_order():
    out = balance_latex_environments("\\begin{itemize}\n\\begin{enumerate}\n\\item One")
    assert out.endswith("\\end{enumerate}\n\\end{itemize}")
    assert balance_latex_environments(out) == out


def test_unmatched_end_is_dropped():
    # "Extra \end{itemize}" is itself an error, so a stray one is removed.
    out = balance_latex_environments("\\item One\n\\end{itemize}")
    assert "\\end{itemize}" not in out
    assert "\\item One" in out


def test_mismatched_nesting_is_repaired():
    out = balance_latex_environments(
        "\\begin{itemize}\n\\begin{enumerate}\n\\item One\n\\end{itemize}"
    )
    assert balance_latex_environments(out) == out
    assert out.count("\\begin") == out.count("\\end")


def test_balancing_is_idempotent():
    for body in (
        "\\begin{itemize}\n\\item One",
        "\\item One\n\\end{itemize}",
        "\\begin{itemize}\n\\begin{enumerate}\n\\item One\n\\end{itemize}",
    ):
        once = balance_latex_environments(body)
        assert balance_latex_environments(once) == once


def test_balancing_does_not_change_text():
    body = "\\begin{itemize}\n\\item Sales & Marketing 40% done\n"
    out = balance_latex_environments(body)
    assert "Sales & Marketing 40% done" in out


def test_balancing_ignores_non_environments():
    body = "\\section*{Profil}\n\\jobheader{Role}{Firma}{2020 -- Heute}"
    assert balance_latex_environments(body) == body


def test_balancing_rejects_non_strings():
    with pytest.raises(LaTeXEscapeError):
        balance_latex_environments(None)


# ---------------------------------------------------------------------------
# Orphaned \item repair
# ---------------------------------------------------------------------------


def test_orphaned_items_are_wrapped():
    # This is the real failure a local model produces: \jobheader followed by
    # \item bullets with no \begin{itemize}. "Lonely \item" is fatal.
    body = (
        "\\jobheader{Platform Engineer}{Example GmbH}{Jan 2020 - Present}\n"
        "\\item Built Python services.\n"
        "\\item Managed Kubernetes clusters."
    )
    out = repair_latex_lists(body)
    assert "\\begin{itemize}" in out
    assert "\\end{itemize}" in out
    assert out.count("\\item") == 2
    # The content is untouched.
    assert "Built Python services." in out
    assert "Managed Kubernetes clusters." in out


def test_items_inside_a_list_are_left_alone():
    body = "\\begin{itemize}\n\\item a\n\\item b\n\\end{itemize}"
    assert repair_latex_lists(body) == body


def test_items_inside_enumerate_are_left_alone():
    body = "\\begin{enumerate}\n\\item a\n\\end{enumerate}"
    assert repair_latex_lists(body) == body


def test_two_orphaned_runs_are_wrapped_separately():
    body = "\\jobheader{A}{X}{2020}\n\\item one\n" "\\jobheader{B}{Y}{2021}\n\\item two\n"
    out = repair_latex_lists(body)
    assert out.count("\\begin{itemize}") == 2
    assert out.count("\\end{itemize}") == 2
    assert out.index("one") < out.index("B") < out.index("two")


def test_prose_after_orphans_closes_the_group():
    out = repair_latex_lists("\\item a\nSome prose.")
    assert out.index("\\end{itemize}") < out.index("Some prose.")


def test_wrapped_then_orphaned_is_handled():
    body = "\\begin{itemize}\n\\item a\n\\end{itemize}\n\\item b"
    out = repair_latex_lists(body)
    assert out.count("\\begin{itemize}") == 2
    assert repair_latex_lists(out) == out


def test_list_repair_is_idempotent():
    for body in (
        "\\item a",
        "\\item a\n\\item b",
        "\\item a\nSome prose.",
        "\\begin{itemize}\n\\item a\n\\end{itemize}\n\\item b",
        "\\jobheader{A}{B}{C}\n\\item x",
    ):
        once = repair_latex_lists(body)
        assert repair_latex_lists(once) == once


def test_list_repair_leaves_a_body_without_items_untouched():
    body = "\\section*{Profil}\nJust prose, no bullets."
    assert repair_latex_lists(body) == body


def test_list_repair_preserves_optional_labels():
    out = repair_latex_lists("\\item[x] a")
    assert "\\item[x] a" in out


def test_list_repair_rejects_non_strings():
    with pytest.raises(LaTeXEscapeError):
        repair_latex_lists(None)


def test_unbalanced_then_orphans_together_produce_valid_lists():
    """
    The combination a model actually produces: a closed list, then an unclosed
    one, then bare items. Both repairs have to compose.
    """
    body = "\\begin{itemize}\n\\item a\n" "\\item b\n" "\\item c\n"
    balanced = balance_latex_environments(body)
    repaired = repair_latex_lists(balanced)
    assert repaired.count("\\begin") == repaired.count("\\end")
    assert repair_latex_lists(repaired) == repaired
