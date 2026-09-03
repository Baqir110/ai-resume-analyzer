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
    """Safely prepare a URL for use as the first argument of \href."""
    url = url.strip()

    match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", url)
    if match:
        url = match.group(2).strip()

    if url.startswith("{") and url.endswith("}"):
        url = url[1:-1].strip()

    url = url.strip("`").strip()
    url = url.replace(r"\_", "_")

    return url


def normalize_latex_links(text: str) -> str:
    """Convert malformed Markdown links into safe LaTeX \hrlink commands."""
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

    markdown_link = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")

    def convert_markdown_link(match: re.Match) -> str:
        label = match.group(1).strip()
        url = latex_escape_url(match.group(2))
        return rf"\hrlink{{{url}}}{{{label}}}"

    text = markdown_link.sub(convert_markdown_link, text)

    bare_markdown_url = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")

    text = bare_markdown_url.sub(
        lambda m: rf"\hrlink{{{latex_escape_url(m.group(2))}}}{{{m.group(1).strip()}}}",
        text,
    )

    return text


def clean_llm_response_to_latex(text: str) -> str:
    """Extracts valid LaTeX content and strips AI chatter/meta-language."""
    if not text:
        return ""

    text = text.strip()

    if (
        text.startswith("I'm ready")
        or text.startswith("Sure")
        or "Please provide" in text
    ):
        return r"\section*{Profil}\n\noindent Resume optimization pending input."

    return text


def strip_code_fences(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^\s*```(?:latex|tex)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def clean_body_for_latex(body_text: str) -> str:
    """Sanitize generated body content without breaking LaTeX macros."""
    body_text = normalize_latex_links(body_text)

    # Fix illegal line breaks
    body_text = re.sub(r"\\\\(?=\s*\\section\*?\{)", "", body_text)
    body_text = re.sub(r"\\\\(?=\s*\\begin\{)", "", body_text)
    body_text = re.sub(r"\\\\(?=\s*\\end\{)", "", body_text)
    body_text = re.sub(r"\\\\(?=\s*\n\s*\n)", "", body_text)

    # Normalize Unicode characters
    replacements = {
        "\u202f": " ",
        "\u200b": "",
        "\u2013": "--",
        "\u2014": "---",
        "\u2011": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\xa0": " ",
        " \vert{} ": r" \quad$\cdot$\quad ",
    }

    for char, repl in replacements.items():
        body_text = body_text.replace(char, repl)

    body_text = strip_code_fences(body_text)

    # Escape unescaped ampersands and percent signs
    body_text = re.sub(r"(?<!\\)&", r"\&", body_text)
    body_text = re.sub(r"(?<!\\)%", r"\%", body_text)

    return body_text.strip()


# ============================================================
# 1. GERMAN CORPORATE SLATE NAVY
# ============================================================

GERMAN_CORPORATE_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=0.8cm,bottom=0.8cm,left=1.2cm,right=1.2cm]{geometry}
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

\titlespacing{\section}{0pt}{4pt}{2pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=0.5pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5pt}

\newcommand{\jobheader}[3]{%
    \noindent\textbf{\color{primary}#1}, #2
    \hfill
    \textit{\color{subgray}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}
    \ifx\relax#2\relax\else\textit{\color{subgray}(#2)}\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[2pt]

    {\Large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \& MLOps}\\[3pt]

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

\vspace{4pt}

\noindent
\small\color{subgray}Bamberg, \today

\end{document}
"""


# ============================================================
# 2. GERMAN PROFESSIONAL ATS
# ============================================================

GERMAN_ATS_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=0.8cm,bottom=0.8cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{000000}
\definecolor{secondary}{HTML}{333333}
\definecolor{linkcolor}{HTML}{0000EE}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{#2}}

\titleformat{\section}
    {\large\bfseries\color{primary}\uppercase}
    {}{0em}{}
    [\vspace{-2pt}\rule{\textwidth}{0.8pt}]

\titlespacing{\section}{0pt}{4pt}{2pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=0.5pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{#1} -- #2
    \hfill
    \textbf{#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{#1} \ifx\relax#2\relax\else(#2)\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{[GitHub]}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\LARGE\bfseries Muhammad Baqir}\\[2pt]

    {\large IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[3pt]

    {\small
        Bamberg, Deutschland
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

\vspace{4pt}

\noindent
{\small Bamberg, \today}

\end{document}
"""


# ============================================================
# 3. GERMAN CLASSIC
# ============================================================

GERMAN_CLASSIC_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=0.9cm,bottom=0.9cm,left=1.3cm,right=1.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\definecolor{primary}{HTML}{111111}
\definecolor{secondary}{HTML}{444444}
\definecolor{linkcolor}{HTML}{000000}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{#2}}

\titleformat{\section}
    {\Large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-2pt}\hrule height 0.5pt]

\titlespacing{\section}{0pt}{5pt}{2pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=0.5pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{#1}, #2 \hfill \textit{#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{#1} \ifx\relax#2\relax\else\textit{(#2)}\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{Link}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\huge\bfseries Muhammad Baqir}\\[3pt]

    {\large IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}\\
        \hrlink{https://www.linkedin.com/in/muhammadbaqir-it}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110}{GitHub}
    }

\end{center}

\vspace{2pt}
\hrule height 1pt
\vspace{4pt}

RESUME_BODY_PLACEHOLDER

\vspace{6pt}

\noindent
{\small Bamberg, den \today}

\end{document}
"""


# ============================================================
# 4. GERMAN MODERN
# ============================================================

GERMAN_MODERN_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=0.8cm,bottom=0.8cm,left=1.2cm,right=1.2cm]{geometry}
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

\definecolor{primary}{HTML}{0284C7}
\definecolor{darkgray}{HTML}{1E293B}
\definecolor{secondary}{HTML}{64748B}
\definecolor{linkcolor}{HTML}{0284C7}

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
    [\vspace{-2pt}\color{primary}\rule{\textwidth}{1.2pt}]

\titlespacing{\section}{0pt}{4pt}{2pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=0.5pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{darkgray}#1}, \textcolor{secondary}{#2}
    \hfill
    \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{darkgray}#1}
    \ifx\relax#2\relax\else\textit{\color{secondary}(#2)}\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{darkgray} Muhammad Baqir}\\[2pt]

    {\large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[3pt]

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

\vspace{4pt}

\noindent
{\small\color{secondary}Bamberg, \today}

\end{document}
"""


# ============================================================
# 5. INTERNATIONAL ATS
# ============================================================

INTERNATIONAL_ATS_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=0.8cm,bottom=0.8cm,left=1.2cm,right=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\definecolor{primary}{HTML}{111827}
\definecolor{secondary}{HTML}{4B5563}
\definecolor{linkcolor}{HTML}{2563EB}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{#2}}

\titleformat{\section}
    {\large\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-3pt}\color{primary}\rule{\textwidth}{0.6pt}]

\titlespacing{\section}{0pt}{4pt}{2pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=0.5pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5pt}

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
    \ifx\relax#2\relax\else\textit{\color{secondary}(#2)}\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{GitHub}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[2pt]

    {\large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[3pt]

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
# Generate LaTeX CV
# ============================================================


def generate_german_latex_content(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    layout_style: str = "german_corporate",
    primary_color_hex: Optional[str] = None,
    secondary_color_hex: Optional[str] = None,
) -> str:
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
Optimize and enrich the candidate's CV body content to achieve the highest possible match against the target job description.

Target Job Description:
{job_description or "General IT Support / DevOps position."}

Critical Skills to Integrate Naturally:
{', '.join(missing_skills) if missing_skills else "None provided."}

Original Resume Text:
{resume_text}

Selected Layout Style:
{layout_style}

STRICT STRUCTURAL AND CONTENT RULES:

1. DO NOT GENERATE ANY CONTACT HEADER, SIDEBAR, OR NAME BLOCK AT THE TOP.
   The document preamble already renders candidate contact headers.
   Start directly with the first section:
   \section*{{Profil}}

2. PRESERVE ALL ORIGINAL DATA:
   - Do NOT delete any existing work experience entries, degrees, or projects.
   - Do NOT change degree titles or company names.
   - You MAY expand existing experience and project bullet points by adding technical context, missing keywords, and relevant details from the job description.

3. PAGE LIMITATION COMPLIANCE:
   - Ensure the content remains concise and fits comfortably within 1 page.
   - Combine education entries efficiently if necessary.

4. WORK EXPERIENCE & PROJECTS:
   - Every bullet point inside \begin{{itemize}} MUST start strictly with \item.
   - Use \jobheader{{Role | Sub-role}}{{Company, City}}{{Dates}} for jobs.
   - Use \projheader{{Project Name}}{{Tech Stack}}{{https://github.com/Baqir110/repo-name}} for projects.

5. SKILLS SECTION:
   Format inline using bold category titles:
   \textbf{{Programmierung \& CI/CD:}} Python, SQL, Bash, Git, GitHub Actions
   \textbf{{Backend, Cloud \& MLOps:}} FastAPI, Docker, Kubernetes, PostgreSQL, Redis
   \textbf{{Monitoring \& Support:}} Prometheus, Grafana, Azure Monitor, Nagios, Zabbix

6. SECTION ORDERING:
{section_names}

7. Escape special characters inside normal text:
   \& for &
   \% for %
   \_ for _
   \textgreater{{}} for >
   \textless{{}} for <

8. Return RAW LaTeX body content ONLY (No code fences, markdown, or conversational text).
"""

    raw_latex = LLMService.generate(
        prompt=prompt,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
    )

    clean_body = clean_llm_response_to_latex(raw_latex)
    clean_body = clean_body_for_latex(clean_body)

    template = CV_TEMPLATES[layout_style]

    if primary_color_hex:
        template = re.sub(
            r"\\definecolor\{primary\}\{HTML\}\{[A-Fa-f0-9]{6}\}",
            f"\\\\definecolor{{primary}}{{HTML}}{{{primary_color_hex.strip('#')}}}",
            template,
        )

    if secondary_color_hex:
        template = re.sub(
            r"\\definecolor\{secondary\}\{HTML\}\{[A-Fa-f0-9]{6}\}",
            f"\\\\definecolor{{secondary}}{{HTML}}{{{secondary_color_hex.strip('#')}}}",
            template,
        )

    return template.replace(
        "RESUME_BODY_PLACEHOLDER",
        clean_body,
    )


# ============================================================
# Compile LaTeX to PDF
# ============================================================


def compile_latex_to_pdf(latex_code: str) -> bytes:
    pdflatex = shutil.which("pdflatex")

    if not pdflatex:
        raise RuntimeError(
            "pdflatex was not found on PATH. "
            "TinyTeX/TeX Live directory is not available."
        )

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
                output = (first.stdout or "") + "\n" + (first.stderr or "")
                error_tail = output[-10000:]
                raise RuntimeError("LaTeX compilation failed:\n\n" f"{error_tail}")

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
                output = (second.stdout or "") + "\n" + (second.stderr or "")
                error_tail = output[-10000:]
                raise RuntimeError(
                    "LaTeX compilation failed on the second pass:\n\n" f"{error_tail}"
                )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("LaTeX compilation timed out after 30 seconds.") from exc

        if not pdf_path.exists():
            raise RuntimeError("pdflatex completed but no PDF was produced.")

        return pdf_path.read_bytes()
