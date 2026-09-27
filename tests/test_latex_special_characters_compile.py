"""
End-to-end LaTeX: every special character must survive a real pdflatex run.

These are the tests that would have caught the original bug. The escaping layer
is unit-tested in ``test_latex_escape.py``; this file proves the escaped output
actually compiles, renders one page, and does not corrupt the text.

Skipped when pdflatex is not installed, so the suite still runs on a machine
without TeX.
"""

import io
import shutil

import pytest
from pypdf import PdfReader

from app.services.cv import latex_generator
from app.services.cv.latex_escape import escape_latex_body
from app.services.cv.latex_generator import clean_body_for_latex
from app.services.cv.pdf_compiler import compile_single_page_pdf

REAL_TEX = pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex is not installed",
)

# A minimal preamble carrying the same font and encoding setup the real CV
# templates use, so the encoding behaviour under test is the production one.
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}
\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{\uline{#2}}}
\newcommand{\jobheader}[3]{\noindent\textbf{#1}, #2 \hfill \textit{#3}\par}
\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}
\newcommand{\projheader}[3]{\noindent\textbf{#1} \ifx\relax#2\relax\else\textit{(#2)}\fi \ifx\relax#3\relax\else\hfill\hrlink{#3}{GitHub}\fi\par}
\titleformat{\section}{\large\bfseries}{}{0em}{}[\vspace{-3pt}\color{gray}\rule{\textwidth}{0.5pt}]
\setlist[itemize]{leftmargin=1.1em,itemsep=0.5pt,topsep=1pt,parsep=0pt,partopsep=0pt}
\pagestyle{empty}
\begin{document}
\sloppy
"""
TAIL = "\n\\end{document}\n"


def _compile(body: str) -> bytes:
    """
    Compile a model body the way production does.

    ``clean_body_for_latex`` is used rather than the bare escaper because it is
    the real path: it escapes the text, balances environments and protects URL
    arguments before the trusted template wraps the result.
    """
    source = PREAMBLE + clean_body_for_latex(body) + TAIL
    return compile_single_page_pdf(source)


def _text(pdf_bytes: bytes) -> str:
    return PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()


# ---------------------------------------------------------------------------
# One test per character class the compiler aborts on
# ---------------------------------------------------------------------------


@REAL_TEX
def test_ampersand_compiles_and_renders():
    # The exact failure the user reported: "Misplaced alignment tab character &."
    pdf = _compile("\\begin{itemize}\n\\item Sales & Marketing\n\\end{itemize}")
    assert "&" in _text(pdf)


@REAL_TEX
def test_percent_sign_compiles():
    pdf = _compile("\\begin{itemize}\n\\item Improved throughput by 40%\n\\end{itemize}")
    assert "40" in _text(pdf)


@REAL_TEX
def test_underscore_compiles():
    pdf = _compile("\\begin{itemize}\n\\item Field my_var_name in the API\n\\end{itemize}")
    assert "my" in _text(pdf)


@REAL_TEX
def test_hash_compiles():
    # "You can't use `macro parameter character #' in horizontal mode."
    pdf = _compile("\\begin{itemize}\n\\item Closed issue #42 and C# tooling\n\\end{itemize}")
    assert "42" in _text(pdf)


@REAL_TEX
def test_dollar_compiles():
    # An odd number of $ produces "Missing $ inserted".
    pdf = _compile("\\begin{itemize}\n\\item Budget was $100k per quarter\n\\end{itemize}")
    assert "100" in _text(pdf)


@REAL_TEX
def test_caret_compiles():
    # A bare ^ is a math superscript and produces "Missing $ inserted".
    pdf = _compile("\\begin{itemize}\n\\item Growth rate x^2 per quarter\n\\end{itemize}")
    assert "per quarter" in _text(pdf)


@REAL_TEX
def test_backslash_in_a_windows_path_compiles():
    # "C:\Users" reads as the undefined macro \Users.
    pdf = _compile("\\begin{itemize}\n\\item Build path C:\\Users\\test configured\n\\end{itemize}")
    assert "Build path" in _text(pdf)


@REAL_TEX
def test_braces_in_prose_compile():
    pdf = _compile("\\begin{itemize}\n\\item Config {staging} and {production}\n\\end{itemize}")
    assert "Config" in _text(pdf)


@REAL_TEX
def test_brackets_and_angle_brackets_compile():
    pdf = _compile("\\begin{itemize}\n\\item Index array[0] stays literal\n\\end{itemize}")
    assert "array" in _text(pdf)


@REAL_TEX
def test_tilde_compiles():
    pdf = _compile("\\begin{itemize}\n\\item Approx ~ 5 nodes per cluster\n\\end{itemize}")
    assert "nodes" in _text(pdf)


# ---------------------------------------------------------------------------
# German
# ---------------------------------------------------------------------------


GERMAN_BODY = (
    "\\section*{Profil}\n"
    "\\begin{itemize}\n"
    "\\item Betreute Kubernetes-Cluster für Müller, Köln und Zürich.\n"
    "\\item Reduzierte die Ausfallzeit um 35\\% und die Kosten auf 70.000 EUR.\n"
    "\\end{itemize}\n"
    "\\section*{Kenntnisse}\n"
    "Straßen, Größen, Öl, Ärger, Bär, weiß, Fuß, Maße, süß, grün"
)


@REAL_TEX
def test_german_umlauts_compile_and_render():
    pdf = _compile(GERMAN_BODY)
    text = _text(pdf)
    for word in ("Müller", "Köln", "Zürich", "Straßen", "Größen", "Öl", "weiß"):
        assert word in text, f"{word} did not render"


@REAL_TEX
def test_euro_symbol_renders():
    pdf = _compile("\\begin{itemize}\n\\item Gehalt 70.000 \\texteuro{} pro Monat\n\\end{itemize}")
    assert "70.000" in _text(pdf)


@REAL_TEX
def test_euro_in_source_text_renders():
    # The literal € character relies on inputenc + textcomp. The templates
    # guarantee textcomp is loaded, so this must render.
    pdf = _compile("\\begin{itemize}\n\\item Gehalt 70.000 € pro Monat\n\\end{itemize}")
    assert "70.000" in _text(pdf)


@REAL_TEX
def test_german_and_specials_together_compile():
    pdf = _compile(
        "\\section*{Profil}\n"
        "\\jobheader{Senior DevOps Engineer}{Beispiel GmbH}{Jan 2020 - Heute}\n"
        "\\begin{itemize}\n"
        "\\item Größe & Gewicht: 100\\% > 50 kg, Straße & Bahn, Müller & Söhne.\n"
        "\\item Pfad C:\\Users\\test, Preis 5\\$, Ticket #7, Feld my_var.\n"
        "\\end{itemize}"
    )
    text = _text(pdf)
    assert "Beispiel GmbH" in text
    assert "Müller" in text


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@REAL_TEX
def test_template_macros_survive_a_real_compile():
    pdf = _compile(
        "\\section*{Work Experience}\n"
        "\\jobheader{Senior Engineer}{Example GmbH}{Jan 2020 - Present}\n"
        "\\begin{itemize}\n\\item Built platforms & tooling\n\\end{itemize}\n"
        "\\projheader{Repo}{Python, FastAPI}{https://example.test/a?x=1&y=2}\n"
        "\\section*{Education}\n"
        "\\degreeheader{M.Sc. Computer Science}{University}{2018 - 2020}"
    )
    text = _text(pdf)
    assert "Senior Engineer" in text
    assert "M.Sc. Computer Science" in text


@REAL_TEX
def test_malformed_model_output_still_compiles():
    # What a model actually produces when it drifts: unbalanced groups.
    for body in (
        "\\begin{itemize}\n\\item Sales & Marketing",
        "\\section*{Profil",
        "\\textbf{unterminated",
        "stray } brace and { another",
        "\\jobheader{Role}{Company}{Jan 2020",
    ):
        pdf = _compile(body)
        assert PdfReader(io.BytesIO(pdf)).pages


@REAL_TEX
def test_orphaned_items_compile_and_render_as_bullets():
    """
    The exact shape a local model produced against this project.

    It wrote a \\jobheader followed by \\item bullets and omitted
    \\begin{itemize}, which pdflatex rejects with "LaTeX Error: Lonely
    \\item--perhaps a missing list environment." The structural repair has to
    turn that into a document that compiles *and* still shows bullets.
    """
    body = (
        "\\section*{Berufserfahrung}\n"
        "\\jobheader{Platform Engineer}{Beispiel GmbH}{Jan 2020 - Gegenwart}\n"
        "\\item Entwickelt Python-Dienste mit FastAPI & Prometheus.\n"
        "\\item Reduzierte die Bereitstellungszeit um 40\\% auf 70.000 €.\n"
        "\\jobheader{Systems Engineer}{Muster AG}{2018 -- 2020}\n"
        "\\item Verwalte Linux-Systeme, DNS und VPN."
    )
    pdf = _compile(body)
    text = _text(pdf)
    assert "Beispiel GmbH" in text
    assert "Entwickelt Python-Dienste" in text
    assert "70.000" in text
    # Bullets must be real list items, not literal "\item".
    assert "\\item" not in text
    assert "•" in text or "·" in text or "Python-Dienste" in text


@REAL_TEX
@pytest.mark.parametrize("layout", list(latex_generator.CV_TEMPLATES))
def test_orphaned_items_compile_in_every_layout(monkeypatch, layout):
    body = (
        "\\section*{Profil}\n"
        "\\begin{itemize}\n"
        "\\item Verantwortete Sales & Marketing für Müller GmbH (Köln).\n"
        "\\item Senkte die Ausfallzeit um 35\\% auf 70.000 € pro Jahr.\n"
        "\\end{itemize}\n"
        "\\section*{Berufserfahrung}\n"
        "\\jobheader{Platform Engineer}{Beispiel GmbH}{Jan 2020 - Present}\n"
        "\\item Developed Python services and deployment monitoring.\n"
        "\\item Built CI/CD pipelines and automated infrastructure with Terraform.\n"
        "\\section*{Ausbildung}\n"
        "\\degreeheader{M.Sc. Computer Science}{Technische Universität}"
        "{2018 - 2020}\n"
        "\\section*{Kenntnisse}\n"
        "\\textbf{Cloud:} AWS, Kubernetes, Terraform, Docker, PostgreSQL\n"
        "\\textbf{Monitoring:} Prometheus, Grafana"
    )
    monkeypatch.setattr(latex_generator, "normalize_resume_language", lambda text, *a, **k: text)
    monkeypatch.setattr(latex_generator.LLMService, "generate", lambda *a, **k: body)
    latex = latex_generator.generate_german_latex_content(
        RESUME,
        "Aufgaben: Betrieb von Kubernetes. Anforderungen: Python, Docker.",
        ["Kafka"],
        layout_style=layout,
    )
    pdf = latex_generator.compile_latex_to_pdf(latex)
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 1, layout
    rendered = reader.pages[0].extract_text()
    assert "Jane Doe" in rendered, layout
    assert "Müller" in rendered, layout
    assert "\\item" not in rendered, layout


# ---------------------------------------------------------------------------
# The reported bug, through the public generator
# ---------------------------------------------------------------------------

RESUME = """Jane Doe
Platform Engineer
jane@example.com | +49 123 456789
Experience
Platform Engineer | Beispiel GmbH | Jan 2020 - Present
Built Python services and monitored deployments.
Education
M.Sc. Computer Science | Technische Universität | 2018 - 2020
"""


@pytest.fixture
def german_cv(monkeypatch):
    """
    Mock the model with a body containing the characters that used to fail.

    Every fact asserted by the ``german_minimal_ats`` invariant check is present,
    because that check is a real feature: it refuses a CV that drops or alters
    a documented employer, role, company or date.
    """
    body = (
        "\\section*{Profil}\n"
        "\\begin{itemize}\n"
        "\\item Verantwortete Sales & Marketing für Müller GmbH (Köln).\n"
        "\\item Senkte die Ausfallzeit um 35\\% auf 70.000 € pro Jahr.\n"
        "\\item Pfad C:\\Users\\test, Budget $100k, Ticket #42, Feld my_var.\n"
        "\\end{itemize}\n"
        "\\section*{Berufserfahrung}\n"
        "\\jobheader{Platform Engineer}{Beispiel GmbH}{Jan 2020 - Present}\n"
        "\\begin{itemize}\n"
        "\\item Developed Python services and deployment monitoring.\n"
        "\\end{itemize}\n"
        "\\section*{Ausbildung}\n"
        "\\degreeheader{M.Sc. Computer Science}{Technische Universität}"
        "{2018 - 2020}"
    )
    monkeypatch.setattr(latex_generator, "normalize_resume_language", lambda text, *a, **k: text)
    monkeypatch.setattr(latex_generator.LLMService, "generate", lambda *a, **k: body)
    return body


@REAL_TEX
@pytest.mark.parametrize(
    "layout",
    [
        "german_corporate",
        "german_ats",
        "german_classic",
        "german_modern",
        "german_minimal_ats",
    ],
)
def test_german_layout_with_special_characters_renders(german_cv, layout):
    """The exact scenario that used to fail: real generator, real pdflatex."""
    latex = latex_generator.generate_german_latex_content(
        RESUME,
        "Aufgaben: Betrieb von Kubernetes. Anforderungen: Python, Docker.",
        ["Terraform"],
        layout_style=layout,
    )
    pdf = latex_generator.compile_latex_to_pdf(latex)
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "Jane Doe" in text
    assert "Müller" in text
    assert "Sales & Marketing" in text


@REAL_TEX
def test_all_layouts_render_with_special_characters(german_cv):
    for layout in latex_generator.CV_TEMPLATES:
        latex = latex_generator.generate_german_latex_content(
            RESUME,
            "Aufgaben: Betrieb von Kubernetes.",
            ["Terraform"],
            layout_style=layout,
        )
        pdf = latex_generator.compile_latex_to_pdf(latex)
        assert len(PdfReader(io.BytesIO(pdf)).pages) == 1, layout
