import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from app.services.llm_provider import LLMService


# ============================================================
# LaTeX helpers
# ============================================================

def latex_escape_url(url: str) -> str:
    """
    Safely prepare a URL for use as the first argument of \\href.
    """
    url = url.strip()

    # Remove Markdown link wrappers:
    # [https://example.com](https://example.com)
    match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", url)
    if match:
        url = match.group(2).strip()

    # Remove accidental surrounding braces or backticks.
    if url.startswith("{") and url.endswith("}"):
        url = url[1:-1].strip()

    url = url.strip("`").strip()

    # URLs should contain literal underscores.
    url = url.replace(r"\_", "_")

    return url


def normalize_latex_links(text: str) -> str:
    """
    Convert malformed Markdown links and constructs into safe
    LaTeX \\hrlink commands.

    Handles:
        [GitHub](https://github.com/example)
        \\hrlink{https://github.com/example}{GitHub}
        malformed nested Markdown/LaTeX combinations
    """

    # --------------------------------------------------------
    # Repair already-generated \\hrlink commands containing
    # Markdown links in the URL argument.
    # --------------------------------------------------------
    malformed_hrlink = re.compile(
        r"""
        \\hrlink
        \{
            \s*
            \[([^\]]+)\]
            \(([^)]+)\)
            \s*
        \}
        \{
            ([^}]*)
        \}
        """,
        re.VERBOSE,
    )

    def repair_hrlink(match: re.Match) -> str:
        markdown_url = match.group(2).strip()
        label = match.group(3).strip() or match.group(1).strip() or "Link"

        url = latex_escape_url(markdown_url)

        return rf"\hrlink{{{url}}}{{{label}}}"

    text = malformed_hrlink.sub(repair_hrlink, text)

    # --------------------------------------------------------
    # Convert normal Markdown links:
    # [GitHub](https://github.com/foo)
    # ->
    # \\hrlink{https://github.com/foo}{GitHub}
    # --------------------------------------------------------
    markdown_link = re.compile(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)"
    )

    def convert_markdown_link(match: re.Match) -> str:
        label = match.group(1).strip()
        url = latex_escape_url(match.group(2))

        return rf"\hrlink{{{url}}}{{{label}}}"

    text = markdown_link.sub(convert_markdown_link, text)

    # --------------------------------------------------------
    # Remove Markdown links that may have been escaped in
    # unusual ways by the LLM.
    # Example:
    # [https://example.com](https://example.com)
    # --------------------------------------------------------
    bare_markdown_url = re.compile(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)"
    )

    text = bare_markdown_url.sub(
        lambda m: rf"\hrlink{{{latex_escape_url(m.group(2))}}}{{{m.group(1).strip()}}}",
        text,
    )

    return text


def clean_body_for_latex(body_text: str) -> str:
    """
    Sanitize generated body content without breaking LaTeX macros.
    """

    body_text = normalize_latex_links(body_text)

    # --------------------------------------------------------
    # Fix illegal line breaks that cause:
    #
    # "There's no line here to end"
    #
    # Remove standalone double-backslash line breaks before
    # sections/environments or excessive blank lines.
    # --------------------------------------------------------
    body_text = re.sub(
        r"\\\\(?=\s*\\section\*?\{)",
        "",
        body_text,
    )

    body_text = re.sub(
        r"\\\\(?=\s*\\begin\{)",
        "",
        body_text,
    )

    body_text = re.sub(
        r"\\\\(?=\s*\\end\{)",
        "",
        body_text,
    )

    body_text = re.sub(
        r"\\\\(?=\s*\n\s*\n)",
        "",
        body_text,
    )

    # --------------------------------------------------------
    # Normalize Unicode characters that frequently break
    # pdflatex or produce inconsistent output.
    # --------------------------------------------------------
    replacements = {
        "\u202f": " ",          # narrow no-break space
        "\u200b": "",           # zero-width space
        "\u2013": "--",         # en dash
        "\u2014": "---",        # em dash
        "\u2011": "-",          # non-breaking hyphen
        "\u2018": "'",          # left single quote
        "\u2019": "'",          # right single quote
        "\u201c": '"',          # left double quote
        "\u201d": '"',          # right double quote
        "\xa0": " ",            # non-breaking space
        " \vert{} ": r" \quad$\cdot$\quad ",
    }

    for char, repl in replacements.items():
        body_text = body_text.replace(char, repl)

    # --------------------------------------------------------
    # Remove accidental Markdown code fences.
    # --------------------------------------------------------
    body_text = strip_code_fences(body_text)

    # --------------------------------------------------------
    # Safely escape unescaped ampersands and percent signs.
    #
    # Do not modify already escaped:
    # \&
    # \%
    # --------------------------------------------------------
    body_text = re.sub(
        r"(?<!\\)&",
        r"\&",
        body_text,
    )

    body_text = re.sub(
        r"(?<!\\)%",
        r"\%",
        body_text,
    )

    return body_text.strip()


# ============================================================
# 1. GERMAN CORPORATE SLATE NAVY
# ============================================================

GERMAN_CORPORATE_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.0cm,bottom=1.0cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{0F172A}
\definecolor{linkcolor}{HTML}{1D4ED8}
\definecolor{subgray}{HTML}{475569}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{\uline{#2}}}

\titleformat{\section}
    {\large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-3pt}\color{subgray}\rule{\textwidth}{0.6pt}]

\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=1pt,
    topsep=1pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}

\newcommand{\jobheader}[3]{%
    \noindent\textbf{\color{primary}#1}, #2
    \hfill
    \textit{\color{subgray}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}
    \textit{\color{subgray}(#2)}
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[3pt]

    {\Large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \& MLOps}\\[4pt]

    {\small\color{subgray}
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{https://www.linkedin.com/in/muhammadbaqir-it}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110}{GitHub}
    }

\end{center}

RESUME_BODY_PLACEHOLDER

\vspace{6pt}

\noindent
\small\color{subgray}Bamberg, \today

\end{document}
"""


# ============================================================
# 2. GERMAN PROFESSIONAL ATS
# ============================================================

GERMAN_ATS_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.0cm,bottom=1.0cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{172033}
\definecolor{secondary}{HTML}{475569}
\definecolor{linkcolor}{HTML}{1D4ED8}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{\uline{#2}}}

\titleformat{\section}
    {\large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-3pt}\color{primary}\rule{\textwidth}{0.6pt}]

\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=1pt,
    topsep=1pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}, #2
    \hfill
    \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}
    \textit{\color{secondary}(#2)}
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[3pt]

    {\large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small\color{secondary}
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{https://www.linkedin.com/in/muhammadbaqir-it}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110}{GitHub}
    }

\end{center}

RESUME_BODY_PLACEHOLDER

\vspace{5pt}

\noindent
{\small\color{secondary}Bamberg, \today}

\end{document}
"""


# ============================================================
# 3. GERMAN CLASSIC
# ============================================================

GERMAN_CLASSIC_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.0cm,bottom=1.0cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{1F2937}
\definecolor{secondary}{HTML}{4B5563}
\definecolor{linkcolor}{HTML}{1D4ED8}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{\uline{#2}}}

\titleformat{\section}
    {\large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-3pt}\color{secondary}\rule{\textwidth}{0.5pt}]

\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=1pt,
    topsep=1pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{#1}, #2
    \hfill
    \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{#1}
    \textit{\color{secondary}(#2)}
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\LARGE\bfseries Muhammad Baqir}\\[3pt]

    {\large IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[5pt]

    {\small
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{https://www.linkedin.com/in/muhammadbaqir-it}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110}{GitHub}
    }

\end{center}

\vspace{2pt}

\hrule

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

\vspace{5pt}

\noindent
{\small\color{secondary}Bamberg, \today}

\end{document}
"""


# ============================================================
# 4. GERMAN MODERN
# ============================================================

GERMAN_MODERN_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.0cm,bottom=1.0cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{0F172A}
\definecolor{secondary}{HTML}{475569}
\definecolor{linkcolor}{HTML}{1D4ED8}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{\uline{#2}}}

\titleformat{\section}
    {\large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-3pt}\color{secondary}\rule{\textwidth}{0.5pt}]

\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=1pt,
    topsep=1pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}, #2
    \hfill
    \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}
    \textit{\color{secondary}(#2)}
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[3pt]

    {\large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small\color{secondary}
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{https://www.linkedin.com/in/muhammadbaqir-it}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110}{GitHub}
    }

\end{center}

RESUME_BODY_PLACEHOLDER

\vspace{5pt}

\noindent
{\small\color{secondary}Bamberg, \today}

\end{document}
"""


# ============================================================
# 5. INTERNATIONAL ATS
# ============================================================

INTERNATIONAL_ATS_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.0cm,bottom=1.0cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{172033}
\definecolor{secondary}{HTML}{475569}
\definecolor{linkcolor}{HTML}{1D4ED8}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{\uline{#2}}}

\titleformat{\section}
    {\large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-3pt}\color{primary}\rule{\textwidth}{0.6pt}]

\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=1pt,
    topsep=1pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}, #2
    \hfill
    \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}
    \textit{\color{secondary}(#2)}
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[3pt]

    {\large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small\color{secondary}
        Bamberg, Germany
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{https://www.linkedin.com/in/muhammadbaqir-it}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110}{GitHub}
    }

\end{center}

RESUME_BODY_PLACEHOLDER

\end{document}
"""


# ============================================================
# Template registry
# ============================================================

CV_TEMPLATES = {
    "german_corporate": GERMAN_CORPORATE_LATEX_TEMPLATE,
    "german_ats": GERMAN_ATS_LATEX_TEMPLATE,
    "german_classic": GERMAN_CLASSIC_LATEX_TEMPLATE,
    "german_modern": GERMAN_MODERN_LATEX_TEMPLATE,
    "international_ats": INTERNATIONAL_ATS_LATEX_TEMPLATE,
}


# ============================================================
# LLM output helpers
# ============================================================

def strip_code_fences(text: str) -> str:
    """
    Remove Markdown code fences from LLM output.
    """
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^\s*```(?:latex|tex)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


# ============================================================
# Generate LaTeX CV
# ============================================================

def generate_german_latex_content(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "gemini",
    api_key: Optional[str] = None,
    layout_style: str = "german_corporate",
) -> str:
    """
    Generate a one-page ATS-friendly LaTeX CV.
    """

    if layout_style not in CV_TEMPLATES:
        layout_style = "german_corporate"

    if layout_style == "international_ats":
        section_names = r"""
\section*{Professional Summary}

\section*{Professional Experience}

\section*{Projects}

\section*{Education}

\section*{Technical Skills}

\section*{Languages \& Certifications}
"""
    else:
        section_names = r"""
\section*{Profil}

\section*{Berufserfahrung}

\section*{Projekte}

\section*{Ausbildung}

\section*{Kenntnisse}

\section*{Sprachen \& Zertifikate}
"""

    prompt = f"""
You are an expert German recruiter, ATS resume specialist,
and technical CV writer.

Reformat the candidate's CV body content to fit strictly on
EXACTLY ONE (1) PAGE.

Target Job Description:

{job_description}

Critical Skills to Integrate Naturally:

{', '.join(missing_skills)}

Original Resume Text:

{resume_text}

Selected Layout Style:

{layout_style}

STRICT STRUCTURAL AND MACRO RULES:

1. DO NOT GENERATE ANY CONTACT HEADER, SIDEBAR, OR NAME BLOCK
   AT THE TOP.

   Do NOT output:
   - Muhammad Baqir
   - Kontakt
   - phone numbers
   - email addresses
   - LinkedIn header
   - GitHub header

   The preamble header already renders this.

   START DIRECTLY WITH THE FIRST SECTION:

   \\section*{{Profil}}

2. NEVER use double backslashes (\\\\) at the end of
   paragraphs or headers.

   Use normal empty lines for paragraph breaks.

3. Keep work experience bullet points concise:
   maximum 3-4 bullets per entry.

4. WORK EXPERIENCE LISTS:

   EVERY bullet point inside \\begin{{itemize}} MUST start
   strictly with \\item.

   NEVER drop bullet markers.

5. JOB HEADERS:

   Keep role, company, and dates on ONE line:

   \\jobheader{{Role | Sub-role}}{{Company, City}}{{Dates}}

6. PROJECT HEADERS:

   The third argument MUST be a plain URL string without
   LaTeX macros:

   \\projheader{{Project Name}}{{Tech Stack}}{{https://github.com/Baqir110/repo-name}}

7. PROJECT DEMO & DOCS LINKS:

   Put live links under projects in a SINGLE bullet item
   using \\hrlink:

   \\item \\hrlink{{https://demo.url}}{{Live Demo}}
   \\quad$\\cdot$\\quad
   \\hrlink{{https://docs.url}}{{API Docs}}

8. SKILLS SECTION:

   Format inline using bold category titles:

   \\textbf{{Programmierung \\& CI/CD:}} Python, SQL, Bash,
   Git, GitHub Actions

   \\textbf{{Backend, Cloud \\& MLOps:}} FastAPI, Docker,
   Kubernetes, PostgreSQL, Redis

   \\textbf{{Monitoring \\& Support:}} Prometheus, Grafana,
   Azure Monitor, Nagios, Zabbix

   \\textbf{{Netzwerk \\& Sicherheit:}} TCP/IP, DNS, DHCP,
   VPN, Palo Alto Firewalls, Cisco

9. SECTION ORDERING:

{section_names}

10. Escape special characters inside normal text:

   \\& for &
   \\% for %
   \\_ for _
   \\textgreater{{}} for >
   \\textless{{}} for <

11. Return RAW LaTeX body content ONLY.

    Do not include:
    - code fences
    - explanations
    - Markdown
    - commentary

12. Do not create a second document environment.

13. Do not include:
    \\documentclass
    \\usepackage
    \\begin{{document}}
    \\end{{document}}

14. Do not generate a contact section because the template
    already contains the candidate's contact information.

15. Keep the output ATS-friendly:
    - simple section headings
    - standard text
    - no tables
    - no columns
    - no graphics
    - no icons
    - no text boxes
"""

    raw_latex = LLMService.generate(
        prompt=prompt,
        provider=provider,
        api_key=api_key,
    )

    clean_body = strip_code_fences(raw_latex)
    clean_body = clean_body_for_latex(clean_body)

    template = CV_TEMPLATES[layout_style]

    return template.replace(
        "RESUME_BODY_PLACEHOLDER",
        clean_body,
    )


# ============================================================
# Compile LaTeX to PDF
# ============================================================

def compile_latex_to_pdf(latex_code: str) -> bytes:
    """
    Compile LaTeX to PDF using pdflatex.
    """

    pdflatex = shutil.which("pdflatex")

    if not pdflatex:
        raise RuntimeError(
            "pdflatex was not found on PATH. "
            "TinyTeX/TeX Live directory is not available."
        )

    # Normalize any malformed links in the complete document.
    latex_code = normalize_latex_links(latex_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        tex_path = tmp_path / "resume.tex"
        pdf_path = tmp_path / "resume.pdf"

        tex_path.write_text(
            latex_code,
            encoding="utf-8",
        )

        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-directory",
            str(tmp_path),
            str(tex_path),
        ]

        try:
            # ------------------------------------------------
            # First pass
            # ------------------------------------------------
            first = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=30,
            )

            if first.returncode != 0:
                output = (
                    (first.stdout or "")
                    + "\n"
                    + (first.stderr or "")
                )

                error_tail = output[-10000:]

                raise RuntimeError(
                    "LaTeX compilation failed:\n\n"
                    f"{error_tail}"
                )

            # ------------------------------------------------
            # Second pass
            #
            # Useful for hyperlinks, references and page
            # layout stabilization.
            # ------------------------------------------------
            second = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=30,
            )

            if second.returncode != 0:
                output = (
                    (second.stdout or "")
                    + "\n"
                    + (second.stderr or "")
                )

                error_tail = output[-10000:]

                raise RuntimeError(
                    "LaTeX compilation failed on the second pass:\n\n"
                    f"{error_tail}"
                )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "LaTeX compilation timed out after 30 seconds."
            ) from exc

        if not pdf_path.exists():
            raise RuntimeError(
                "pdflatex completed but no PDF was produced."
            )

        return pdf_path.read_bytes()