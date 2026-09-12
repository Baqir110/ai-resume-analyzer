import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from app.services.cv.optimizer import (
    _clean_skill_list,
    _format_actionable_suggestions,
    auto_select_layout,
    normalize_resume_language,
    required_language_for_layout,
)
from app.services.llm.provider import LLMService

logger = logging.getLogger(__name__)

GERMAN_MINIMAL_ATS = "german_minimal_ats"


# ============================================================
# Language policy
# ============================================================

_GERMAN_LAYOUTS = {
    "german_corporate",
    "german_ats",
    "german_classic",
    "german_modern",
    "german_minimal_ats",
}
_ENGLISH_LAYOUTS = {
    "international_ats",
    "academic",
    "standard",
    "hr_executive_gold",
}


def _language_rule(layout_style: str) -> str:
    ls = (layout_style or "").strip().lower()
    if ls in _GERMAN_LAYOUTS:
        return (
            "OUTPUT LANGUAGE: Write the ENTIRE LaTeX body in professional "
            "German (Deutsch). Keep technical terms and proper nouns in their "
            "original form (Python, FastAPI, Kubernetes, etc.)."
        )
    if ls in _ENGLISH_LAYOUTS:
        return (
            "OUTPUT LANGUAGE: Write the ENTIRE LaTeX body in professional "
            "English. Do NOT mix German phrases into the body unless they are "
            "the official name of a company or institution."
        )
    return (
        "OUTPUT LANGUAGE: Detect the language of the provided Job Description. "
        "If the Job Description is in German, output the entire LaTeX body in professional German. "
        "If it is in English, output the entire LaTeX body in professional English."
    )


# ============================================================
# Keyword dump stripping
# ============================================================

_BAD_SECTION_PATTERNS = [
    r"Erg[äa]nzende\s+(?:technische\s+)?Terminologie",
    r"Erg[äa]nzende\s+Such",
    r"Additional\s+(?:Keywords|Skills|Terms)",
    r"Supplementary\s+(?:Terms|Keywords|Skills)",
    r"Extra\s+(?:Keywords|Terms)",
    r"Keyword\s+Dump",
    r"Technische\s+Weiterbildungsziele",
    r"Zieltechnologien",
    r"Deutschsprachige\s+Fachbegriffe",
    r"Englisches\s+\w*\s*Vokabular",
    r"Fachbegriffe",
    r"Suchvarianten",
    r"Vokabular",
    r"Dokumentations-\s*und\s*Sicherheitsvokabular",
]

_BAD_INLINE_LABELS_RE = re.compile(
    r"(Fachbegriffe|Suchvarianten|Vokabular|Terminologie|"
    r"Additional\s+Keywords|Supplementary\s+Terms|"
    r"Zieltechnologien|Weiterbildungsziele|"
    r"Erg[äa]nzende|Keyword[-\s]?Dump)",
    re.IGNORECASE,
)


def _strip_keyword_dump_sections(body: str) -> str:
    """Remove LLM-generated keyword-dump sections from the LaTeX body."""
    lines = body.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if any(re.search(p, stripped, re.IGNORECASE) for p in _BAD_SECTION_PATTERNS):
            skip = True
            continue
        if skip:
            if (
                stripped.startswith(r"\section")
                or stripped.startswith(r"\jobheader")
                or stripped.startswith(r"\projheader")
                or not stripped
            ):
                skip = False
                if stripped:
                    out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


def _strip_inline_keyword_dumps(body: str) -> str:
    r"""
    Remove inline keyword-dump blocks like:
        \textbf{Deutschsprachige Fachbegriffe:} term1, term2, ...
    """
    if not body:
        return body

    pattern = re.compile(
        r"\\textbf\{[^}]*?" + _BAD_INLINE_LABELS_RE.pattern + r"[^}]*?\}"
        r"[^\\]*?"
        r"(?=\\item|\\section|\\par|\\textbf|\\begin|\\end|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    body = pattern.sub("", body)
    body = pattern.sub("", body)
    return body


# ============================================================
# Template preamble patch + title inference
# ============================================================


def _patch_template_preamble(template: str) -> str:
    r"""
    Fix preamble issues that produce overflow or font-fallback garbage.
    """
    if not template:
        return template

    template = template.replace(r"\hyphenpenalty=10000", "")
    template = template.replace(r"\exhyphenpenalty=10000", "")

    template = template.replace(
        "\\usepackage{helvet}",
        "\\usepackage{lmodern}",
    )

    if "\\usepackage{textcomp}" not in template:
        template = template.replace(
            "\\usepackage[T1]{fontenc}",
            (
                "\\usepackage[T1]{fontenc}\n"
                "\\usepackage{textcomp}\n"
                "\\setlength{\\emergencystretch}{3em}\n"
                "\\tolerance=2000\n"
                "\\hbadness=10000"
            ),
        )

    if "\\begin{document}" in template and "\\sloppy" not in template:
        template = template.replace(
            "\\begin{document}",
            "\\begin{document}\n\\sloppy",
        )

    return template


_TITLE_KEYWORDS = (
    "engineer",
    "developer",
    "scientist",
    "analyst",
    "manager",
    "consultant",
    "architect",
    "specialist",
    "support",
    "devops",
    "mlops",
    "sre",
    "designer",
    "administrator",
    "technician",
    "lead",
)


def _infer_title(resume_text: str) -> str:
    """Extract a short professional title from the top of the resume."""
    if not resume_text:
        return ""
    for line in resume_text.splitlines()[:15]:
        line = line.strip()
        if not line or "@" in line or "http" in line.lower():
            continue
        if re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'\-|&/]{4,80}", line):
            if any(kw in line.lower() for kw in _TITLE_KEYWORDS):
                return line
    return ""


DEFAULT_CV_TITLE = "DevOps Engineer"


# ============================================================
# Angle bracket escaping
# ============================================================


def _escape_raw_angle_brackets(text: str) -> str:
    r"""Escape bare < and > that are outside \href / \url / math contexts."""
    lines = []
    for line in text.splitlines():
        if "\\href" in line or "\\url" in line or "\\detokenize" in line or "$" in line:
            lines.append(line)
            continue
        line = re.sub(r"(?<!\\textless\{\})<", r"\\textless{}", line)
        line = re.sub(r"(?<!\\textgreater\{\})>", r"\\textgreater{}", line)
        lines.append(line)
    return "\n".join(lines)


# ============================================================
# LaTeX link helper
# ============================================================


def latex_escape_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()

    match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", url)
    if match:
        url = match.group(2).strip()

    if url.startswith("{") and url.endswith("}"):
        url = url[1:-1].strip()

    url = url.strip("`").strip("'").strip('"')
    url = url.replace(r"\_", "_")

    return url


# ============================================================
# Factual validation helpers
# ============================================================


class FactualValidationError(ValueError):
    """Raised when generated CV content cannot be proven faithful to its source."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("; ".join(violations))


def _normalize_fact(value: str) -> str:
    value = value.replace(r"\&", "&").replace(r"\_", "_")
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", " ").replace("}", " ")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value.casefold())
    return " ".join(value.split())


def _escape_latex_text(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _section_lines(resume_text: str, headings: list[str]) -> list[str]:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    start = None
    heading_set = {heading.casefold() for heading in headings}
    all_headings = {
        "experience",
        "work experience",
        "professional experience",
        "berufserfahrung",
        "employment history",
        "education",
        "ausbildung",
        "academic background",
        "qualifications",
        "projects",
        "skills",
        "kenntnisse",
        "languages",
        "sprachen",
    }
    for index, line in enumerate(lines):
        if line.rstrip(":").casefold() in heading_set:
            start = index + 1
            break
    if start is None:
        return []
    result = []
    for line in lines[start:]:
        if line.rstrip(":").casefold() in all_headings:
            break
        result.append(line)
    return result


_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|januar|februar|märz|maerz|april|mai|juni|juli|august|"
    r"september|oktober|november|dezember)?\s*\d{4}\s*(?:-|–|—|to|bis)\s*"
    r"(?:present|current|heute|\d{4})\b|\b\d{4}\s*(?:-|–|—|to|bis)\s*"
    r"(?:present|current|heute|\d{4})\b",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(
    r"\b(?:b\.?\s?sc\.?|m\.?\s?sc\.?|bachelor(?:'s)?|master(?:'s)?|"
    r"ph\.?d\.?|diplom|staatsexamen)\b",
    re.IGNORECASE,
)


def extract_resume_invariants(resume_text: str) -> list[str]:
    experience = _section_lines(
        resume_text,
        [
            "Experience",
            "Work Experience",
            "Professional Experience",
            "Berufserfahrung",
            "Employment History",
        ],
    )
    education = _section_lines(
        resume_text,
        ["Education", "Ausbildung", "Academic Background", "Qualifications"],
    )
    invariants: list[str] = []

    if experience and not any(_DATE_RE.search(line) for line in experience):
        raise FactualValidationError(
            ["Could not unambiguously extract career facts from the Experience section."]
        )

    for line in experience:
        if not _DATE_RE.search(line):
            continue
        parts = [part.strip() for part in re.split(r"\s*(?:\||—|–)\s*", line) if part.strip()]
        dated_parts = [part for part in parts if _DATE_RE.search(part)]
        factual_parts = [part for part in parts if not _DATE_RE.search(part)]
        if len(factual_parts) < 2 or not dated_parts:
            raise FactualValidationError(
                [f"Could not unambiguously extract role, company, and dates from: {line}"]
            )
        invariants.extend([factual_parts[0], factual_parts[1], dated_parts[0]])

    for line in education:
        if _DEGREE_RE.search(line):
            invariants.append(line)

    if not invariants:
        raise FactualValidationError(
            [
                "Could not extract career invariants. Use labelled Experience and Education sections with role, company, and dates."
            ]
        )

    return list(dict.fromkeys(invariants))


def extract_candidate_header(resume_text: str) -> dict[str, str]:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    name = next(
        (
            line
            for line in lines[:5]
            if re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,80}", line)
            and "resume" not in line.casefold()
            and "lebenslauf" not in line.casefold()
        ),
        "",
    )
    if not name:
        raise FactualValidationError(["Could not extract a candidate name for the CV header."])

    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", resume_text)
    phone_match = re.search(r"(?:\+?\d[\d ()/-]{6,}\d)", resume_text)
    links = re.findall(r"https?://[^\s)]+", resume_text)
    return {
        "name": name,
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "linkedin": next((link for link in links if "linkedin.com" in link.casefold()), ""),
        "github": next((link for link in links if "github.com" in link.casefold()), ""),
    }


def validate_generated_invariants(latex_code: str, invariants: list[str]) -> None:
    from app.core.event_log import log_event

    normalized_output = _normalize_fact(latex_code)
    missing = [fact for fact in invariants if _normalize_fact(fact) not in normalized_output]
    if missing:
        log_event(
            "pipeline",
            "validation_failed",
            stage="factual",
            violations=[str(m)[:120] for m in missing[:10]],
            missing_count=len(missing),
        )
        raise FactualValidationError(
            [f"Generated CV is missing or alters: {fact}" for fact in missing]
        )


def _render_candidate_contact(header: dict[str, str]) -> str:
    parts = []
    if header["email"]:
        email = _escape_latex_text(header["email"])
        parts.append(rf"\href{{mailto:{header['email']}}}{{{email}}}")
    if header["phone"]:
        parts.append(_escape_latex_text(header["phone"]))
    if header["linkedin"]:
        parts.append(rf"\href{{\detokenize{{{latex_escape_url(header['linkedin'])}}}}}{{LinkedIn}}")
    if header["github"]:
        parts.append(rf"\href{{\detokenize{{{latex_escape_url(header['github'])}}}}}{{GitHub}}")
    return r" \quad$\cdot$\quad ".join(parts)


# ============================================================
# Actionable suggestions helpers
# ============================================================

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{2,}")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "into",
    "from",
    "that",
    "this",
    "your",
    "you",
    "are",
    "was",
    "were",
    "will",
    "shall",
    "have",
    "has",
    "had",
    "add",
    "use",
    "via",
    "must",
    "each",
    "every",
    "both",
    "also",
    "item",
    "list",
    "term",
    "terms",
    "skill",
    "skills",
    "resume",
    "cv",
    "experience",
    "entry",
    "bullet",
    "evidence",
    "section",
    "sections",
    "explicitly",
    "naturally",
    "integrate",
    "integrated",
    "weave",
    "weaving",
    "apply",
    "applied",
    "match",
    "gap",
    "critical",
    "moderate",
    "good",
    "alignment",
    "high",
    "target",
    "targets",
    "primary",
    "secondary",
    "tools",
    "tool",
    "frameworks",
    "framework",
    "libraries",
    "library",
    "before",
    "after",
    "vs",
    "versus",
    "entries",
    "missing",
    "formatting",
    "keyword",
    "keywords",
    "score",
    "scores",
    "rewrite",
    "rewritten",
    "directly",
    "these",
    "those",
    "experiential",
    "points",
    "point",
    "low",
    "medium",
    "poor",
    "excellent",
    "great",
    "check",
    "checks",
    "warning",
    "warnings",
    "error",
    "errors",
    "note",
    "notes",
    "info",
    "information",
    "mismatch",
    "gaps",
    "job",
    "jd",
    "position",
    "role",
    "candidate",
    "applicant",
    "user",
    "system",
    "requirement",
    "requirements",
    "qualification",
    "qualifications",
    "project",
    "projects",
    "company",
    "companies",
    "employer",
    "client",
    "customers",
    "customer",
    "team",
    "teams",
    "work",
    "working",
    "worked",
    "using",
    "used",
    "uses",
    "help",
    "helps",
    "helping",
    "support",
    "supporting",
    "provide",
    "provides",
    "providing",
    "deliver",
    "delivers",
    "delivering",
    "adds",
    "adding",
    "remove",
    "removes",
    "removing",
    "make",
    "makes",
    "making",
    "create",
    "creates",
    "creating",
    "applies",
    "applying",
    "start",
    "starts",
    "starting",
    "stop",
    "stops",
    "stopping",
    "first",
    "last",
    "next",
    "previous",
    "current",
    "new",
    "old",
    "one",
    "two",
    "three",
    "four",
    "five",
    "und",
    "oder",
    "aber",
    "nicht",
    "für",
    "mit",
    "ohne",
    "von",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "ist",
    "sind",
    "war",
    "waren",
    "sein",
    "haben",
    "hat",
    "hatte",
    "hatten",
    "wird",
    "werden",
    "wurde",
    "wurden",
    "kann",
    "können",
    "könnte",
    "könnten",
    "soll",
    "sollen",
    "sollte",
    "sollten",
    "muss",
    "müssen",
    "musste",
    "mussten",
    "bei",
    "beim",
    "durch",
    "über",
    "unter",
    "zwischen",
    "vor",
    "nach",
    "als",
    "wie",
    "wenn",
    "dass",
    "ob",
    "weil",
    "damit",
    "bereit",
    "bereits",
    "noch",
    "schon",
    "nur",
    "auch",
    "sehr",
    "alle",
    "allen",
    "aller",
    "alles",
    "arbeit",
    "arbeiten",
    "arbeite",
    "arbeitet",
    "erfahrung",
    "erfahrungen",
    "kenntnis",
    "kenntnisse",
    "fähigkeit",
    "fähigkeiten",
    "kompetenz",
    "kompetenzen",
    "bereich",
    "bereiche",
    "thema",
    "themen",
    "seite",
    "seiten",
    "zeile",
    "zeilen",
    "text",
    "texte",
    "wort",
    "wörter",
    "begriff",
    "begriffe",
    "hinweis",
    "hinweise",
    "ziel",
    "ziele",
    "zweck",
    "zwecke",
    "ablauf",
    "umsetzung",
    "berarbeitete",
    "netzwerkst",
}


def _extract_key_terms(suggestions: list[str]) -> list[str]:
    terms: set[str] = set()
    for s in suggestions or []:
        if not s:
            continue
        for raw_tok in _TOKEN_RE.findall(s):
            tok = raw_tok.rstrip(".,;:!?")
            if not tok or len(tok) < 2:
                continue
            tl = tok.lower()
            if tl in _STOPWORDS or len(tl) < 3:
                continue
            has_upper = any(c.isupper() for c in tok)
            has_digit = any(c.isdigit() for c in tok)
            has_symbol = any(c in tok for c in "+#/-")
            if not (has_upper or has_digit or has_symbol):
                continue
            terms.add(tl)
    return sorted(terms)


def _normalize_for_latex_match(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\", "")
    return text.casefold()


def _ensure_suggestions_applied_latex(
    generated_text: str,
    suggestions: list[str],
    provider: str,
    model_name: str | None = None,
    api_key: str | None = None,
    route_mode: str = "experiential",
) -> str:
    terms = _extract_key_terms(suggestions)
    if not terms:
        return generated_text

    normalized = _normalize_for_latex_match(generated_text)
    absent = [t for t in terms if t not in normalized]

    if absent:
        logger.info(
            "Suggestion terms not present in LaTeX body (non-blocking): %s",
            absent,
        )
    else:
        logger.info("All %d suggestion terms present in LaTeX body.", len(terms))

    return generated_text


# ============================================================
# LaTeX helpers
# ============================================================


def normalize_latex_links(text: str) -> str:
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

    return text


def clean_llm_response_to_latex(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if text.startswith("I'm ready") or text.startswith("Sure") or "Please provide" in text:
        return r"\section*{Profil}" + "\n" + r"\noindent Resume optimization pending input."

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

    body_text = _strip_keyword_dump_sections(body_text)
    body_text = _strip_inline_keyword_dumps(body_text)

    # Clean up inline math-mode wrapping around CI/CD
    body_text = re.sub(r"\$CI/CD\$", "CI/CD", body_text)

    body_text = re.sub(r"\\\\(?=\s*\\section\*?\{)", "", body_text)
    body_text = re.sub(r"\\\\(?=\s*\\begin\{)", "", body_text)
    body_text = re.sub(r"\\\\(?=\s*\\end\{)", "", body_text)
    body_text = re.sub(r"\\\\(?=\s*\n\s*\n)", "", body_text)

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
        r"\vert{}": r" \quad$\cdot$\quad ",
        r"\vert": r" \quad$\cdot$\quad ",
    }
    for char, repl in replacements.items():
        body_text = body_text.replace(char, repl)

    body_text = strip_code_fences(body_text)

    body_text = _escape_raw_angle_brackets(body_text)

    body_text = re.sub(r"(?<!\\)&", r"\&", body_text)
    body_text = re.sub(r"(?<!\\)%", r"\%", body_text)
    body_text = re.sub(r"(?<!\\)_", r"\_", body_text)

    return body_text.strip()


# ============================================================
# TEMPLATES DECLARATION (Tight 1-Page Layout Adjustments)
# ============================================================

GERMAN_CORPORATE_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

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
    [\vspace{-3pt}\color{subgray}\rule{\textwidth}{0.5pt}]

\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=1pt,
    parsep=0pt,
    partopsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent\textbf{\color{primary}#1}, #2
    \hfill
    \textit{\color{subgray}#3}
    \par\vspace{1pt}%
}

\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

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
    {\Large\bfseries\color{primary} DevOps Engineer}\\[3pt]
    {\normalsize\color{subgray}
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{https://www.linkedin.com/in/muhammad-baqir-it/}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{https://github.com/Baqir110?tab=repositories}{GitHub}
    }
\end{center}

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

\vspace{3pt}

\noindent
\normalsize\color{subgray}Bamberg, \today

\end{document}
"""

GERMAN_ATS_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

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
    [\vspace{-2pt}\rule{\textwidth}{0.6pt}]

\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=1pt,
    parsep=0pt,
    partopsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{#1} -- #2
    \hfill
    \textbf{#3}
    \par\vspace{1pt}%
}

\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

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
    {\large DevOps Engineer}\\[3pt]
    {\normalsize
        Bamberg, Deutschland
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{LINKEDIN_URL_PLACEHOLDER}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{GITHUB_URL_PLACEHOLDER}{GitHub}
    }
\end{center}

RESUME_BODY_PLACEHOLDER

\vspace{3pt}

\noindent
{\normalsize Bamberg, \today}

\end{document}
"""

GERMAN_CLASSIC_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
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

\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=1pt,
    parsep=0pt,
    partopsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{#1}, #2 \hfill \textit{#3}
    \par\vspace{1pt}%
}

\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

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
    {\huge\bfseries Muhammad Baqir}\\[2pt]
    {\large DevOps Engineer}\\[3pt]
    {\normalsize
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}\\
        \hrlink{LINKEDIN_URL_PLACEHOLDER}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{GITHUB_URL_PLACEHOLDER}{GitHub}
    }
\end{center}

\vspace{1pt}
\hrule height 0.5pt
\vspace{2pt}

RESUME_BODY_PLACEHOLDER

\vspace{3pt}

\noindent
{\normalsize Bamberg, den \today}

\end{document}
"""

GERMAN_MODERN_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}

\definecolor{primary}{HTML}{0284C7}
\definecolor{darkgray}{HTML}{1E293B}
\definecolor{secondary}{HTML}{475569}
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
    [\vspace{-2pt}\color{primary}\rule{\textwidth}{1.0pt}]

\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=1pt,
    parsep=0pt,
    partopsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{darkgray}#1}, \textcolor{secondary}{#2}
    \hfill
    \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

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
    {\large\bfseries\color{primary} DevOps Engineer}\\[3pt]
    {\normalsize\color{secondary}
        Bamberg, Deutschland (Umzugsbereit)
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{LINKEDIN_URL_PLACEHOLDER}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{GITHUB_URL_PLACEHOLDER}{GitHub}
    }
\end{center}

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

\vspace{3pt}

\noindent
{\normalsize\color{secondary}Bamberg, \today}

\end{document}
"""

INTERNATIONAL_ATS_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[top=0.7cm, bottom=0.7cm, left=1.0cm, right=1.0cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}
\sloppy
\setlength{\emergencystretch}{3em}
\tolerance=2000
\hbadness=10000

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
  [\vspace{-3pt}\color{subgray}\rule{\textwidth}{0.5pt}]
\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{leftmargin=1.1em, itemsep=0.5pt, topsep=1pt, parsep=0pt, partopsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
  \noindent\textbf{\color{primary}#1}, #2 \hfill \textit{\color{subgray}#3}\par\vspace{1pt}
}
\newcommand{\projheader}[3]{%
  \noindent\textbf{\color{primary}#1} \textit{\color{subgray}(#2)} \hfill \hrlink{#3}{GitHub}\par\vspace{1pt}
}
\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

\pagestyle{empty}

\begin{document}

\begin{center}
  {\Huge \bfseries \color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
  {\Large \bfseries \color{primary} DevOps Engineer}\\[3pt]
  {\normalsize \color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER

\end{document}
"""

STANDARD_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\definecolor{primary}{HTML}{1A202C}
\definecolor{secondary}{HTML}{4A5568}
\definecolor{linkcolor}{HTML}{2B6CB0}

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
    [\vspace{-2pt}\rule{\textwidth}{0.5pt}]

\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=1pt,
    parsep=0pt,
    partopsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}, #2 \hfill \textit{\color{secondary}#3}
    \par\vspace{1pt}%
}

\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1} \ifx\relax#2\relax\else\textit{\color{secondary}(#2)}\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{[GitHub]}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\LARGE\bfseries\color{primary} Muhammad Baqir}\\[2pt]

    {\large\color{secondary} DevOps Engineer}\\[3pt]

    {\normalsize\color{secondary}
        Bamberg, Germany
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{LINKEDIN_URL_PLACEHOLDER}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{GITHUB_URL_PLACEHOLDER}{GitHub}
    }

\end{center}

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

\end{document}
"""

HR_EXECUTIVE_GOLD_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\definecolor{primary}{HTML}{0B192C}
\definecolor{goldaccent}{HTML}{1E3E62}
\definecolor{secondary}{HTML}{475569}
\definecolor{linkcolor}{HTML}{1E40AF}

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
    [\vspace{-2pt}\color{goldaccent}\rule{\textwidth}{0.8pt}]

\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.5pt,
    topsep=1pt,
    parsep=0pt,
    partopsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent
    \textbf{\color{primary}#1} \textbar{} \textcolor{secondary}{#2}
    \hfill
    \textbf{\color{goldaccent}#3}
    \par\vspace{1pt}%
}

\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

\newcommand{\projheader}[3]{%
    \noindent
    \textbf{\color{primary}#1}
    \ifx\relax#2\relax\else\textit{\color{secondary}(#2)}\fi
    \ifx\relax#3\relax
    \else
        \hfill\hrlink{#3}{\textbf{[GitHub]}}
    \fi
    \par\vspace{1pt}%
}

\pagestyle{empty}

\begin{document}

\begin{center}

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[2pt]

    {\large\bfseries\color{goldaccent} DevOps Engineer}\\[3pt]

    {\normalsize\color{secondary}
        Bamberg, Germany
        \quad$\cdot$\quad
        +49 152 17975480
        \quad$\cdot$\quad
        \hrlink{mailto:hzindabad44@gmail.com}{hzindabad44@gmail.com}
        \quad$\cdot$\quad
        \hrlink{LINKEDIN_URL_PLACEHOLDER}{LinkedIn}
        \quad$\cdot$\quad
        \hrlink{GITHUB_URL_PLACEHOLDER}{GitHub}
    }

\end{center}

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

\end{document}
"""

GERMAN_MINIMAL_ATS_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=0.7cm,bottom=0.7cm,left=1.0cm,right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\definecolor{primary}{HTML}{0F172A}
\definecolor{subgray}{HTML}{475569}
\hypersetup{colorlinks=true,urlcolor=primary,pdfborder={0 0 0}}
\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{#2}}
\titleformat{\section}{\large\bfseries\color{primary}}{}{0em}{}[\vspace{-3pt}\color{subgray}\rule{\textwidth}{0.5pt}]
\titlespacing{\section}{0pt}{3pt}{1pt}
\setlist[itemize]{leftmargin=1.1em,itemsep=0.5pt,topsep=1pt,parsep=0pt,partopsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\newcommand{\jobheader}[3]{\noindent\textbf{\color{primary}#1}, #2\hfill\textit{\color{subgray}#3}\par\vspace{1pt}}
\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}
\newcommand{\projheader}[3]{\noindent\textbf{\color{primary}#1}\ifx\relax#2\relax\else\textit{\color{subgray}(#2)}\fi\ifx\relax#3\relax\else\hfill\href{\detokenize{#3}}{GitHub}\fi\par\vspace{1pt}}
\pagestyle{empty}

\begin{document}
\begin{center}
{\Huge\bfseries\color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
{\large\bfseries\color{primary} DevOps Engineer}\\[3pt]
{\normalsize\color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER
\end{document}
"""


ACADEMIC_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[top=0.7cm, bottom=0.7cm, left=1.0cm, right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}
\sloppy
\setlength{\emergencystretch}{3em}
\tolerance=2000
\hbadness=10000

\definecolor{primary}{HTML}{111827}
\definecolor{linkcolor}{HTML}{1F4E79}
\definecolor{subgray}{HTML}{4B5563}

\hypersetup{
    colorlinks=true,
    urlcolor=linkcolor,
    linkcolor=linkcolor,
    pdfborder={0 0 0}
}

\newcommand{\hrlink}[2]{\href{\detokenize{#1}}{#2}}

\titleformat{\section}
    {\large\scshape\bfseries\color{primary}}
    {}{0em}{}
    [\vspace{-2pt}\rule{\textwidth}{0.4pt}]
\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{leftmargin=1.1em, itemsep=0.5pt, topsep=1pt, parsep=0pt, partopsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent\textbf{#1}, \textit{#2} \hfill #3\par\vspace{1pt}
}
\newcommand{\projheader}[3]{%
    \noindent\textbf{#1} \textit{(#2)} \hfill \hrlink{#3}{GitHub}\par\vspace{1pt}
}
\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

\pagestyle{empty}

\begin{document}

\begin{center}
  {\LARGE \scshape \bfseries CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
  {\large \bfseries DevOps Engineer}\\[3pt]
  {\normalsize \color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER

\end{document}
"""


TECHNICAL_LEAD_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[top=0.7cm, bottom=0.7cm, left=1.0cm, right=1.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{hyperref}
\sloppy
\setlength{\emergencystretch}{3em}
\tolerance=2000
\hbadness=10000

\definecolor{primary}{HTML}{0F172A}
\definecolor{accent}{HTML}{7C3AED}
\definecolor{linkcolor}{HTML}{6D28D9}
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
    [\vspace{-3pt}\color{accent}\rule{\textwidth}{0.6pt}]
\titlespacing{\section}{0pt}{3pt}{1pt}

\setlist[itemize]{leftmargin=1.1em, itemsep=0.5pt, topsep=1pt, parsep=0pt, partopsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\newcommand{\jobheader}[3]{%
    \noindent\textbf{\color{primary}#1} \textbar\ \textcolor{subgray}{#2} \hfill \textbf{\color{accent}#3}\par\vspace{1pt}
}
\newcommand{\projheader}[3]{%
    \noindent\textbf{\color{primary}#1} \textit{\color{subgray}(#2)} \hfill \hrlink{#3}{GitHub}\par\vspace{1pt}
}
\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}

\pagestyle{empty}

\begin{document}

\begin{center}
  {\Huge \bfseries \color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
  {\large \bfseries \color{accent} DevOps Engineer}\\[3pt]
  {\normalsize \color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER

\end{document}
"""


CV_LAYOUT_COLUMNS = {
    "german_corporate": "single",
    "german_ats": "single",
    "german_classic": "single",
    "german_modern": "single",
    GERMAN_MINIMAL_ATS: "single",
    "international_ats": "single",
    "academic": "single",
    "technical_lead": "single",
    "standard": "single",
    "hr_executive_gold": "single",
}

CV_TEMPLATES = {
    "german_corporate": GERMAN_CORPORATE_LATEX_TEMPLATE,
    "german_ats": GERMAN_ATS_LATEX_TEMPLATE,
    "german_classic": GERMAN_CLASSIC_LATEX_TEMPLATE,
    "german_modern": GERMAN_MODERN_LATEX_TEMPLATE,
    GERMAN_MINIMAL_ATS: GERMAN_MINIMAL_ATS_LATEX_TEMPLATE,
    "international_ats": INTERNATIONAL_ATS_LATEX_TEMPLATE,
    "academic": ACADEMIC_LATEX_TEMPLATE,
    "technical_lead": TECHNICAL_LEAD_LATEX_TEMPLATE,
    "standard": STANDARD_LATEX_TEMPLATE,
    "hr_executive_gold": HR_EXECUTIVE_GOLD_LATEX_TEMPLATE,
}


def validate_cv_title_template(layout_style: str, latex_code: str) -> None:
    """Ensure every CV format renders the canonical professional headline."""
    if DEFAULT_CV_TITLE not in latex_code:
        raise ValueError(
            f"CV template '{layout_style}' does not contain the canonical title "
            f"'{DEFAULT_CV_TITLE}'."
        )


def validate_cv_layout_template(
    layout_style: str,
    latex_code: str,
    template: str = "",
) -> None:
    """Enforce the declared single/two-column layout for every template."""
    expected = CV_LAYOUT_COLUMNS.get(layout_style, "single")

    code_has = r"\begin{multicols}{2}" in latex_code and r"\end{multicols}" in latex_code
    tmpl_has = (
        bool(template) and r"\begin{multicols}{2}" in template and r"\end{multicols}" in template
    )
    has_multicol = code_has or tmpl_has

    if expected == "two" and not has_multicol:
        logger.warning(
            "Layout '%s' is declared two-column but no multicols markers "
            "were found. Proceeding single-column.",
            layout_style,
        )
        return

    if expected == "single" and has_multicol:
        raise ValueError(
            f"Layout '{layout_style}' is declared single-column but contains a two-column body."
        )

    if not re.search(r"\\documentclass\[11pt,a4paper\]\{article\}", latex_code):
        raise ValueError(f"Layout '{layout_style}' must use an 11pt A4 document class.")

    for forbidden in (r"\small", r"\footnotesize", r"\scriptsize", r"\tiny"):
        if forbidden in latex_code:
            raise ValueError(
                f"Layout '{layout_style}' contains a font size below 11pt: {forbidden}"
            )


# ============================================================
# Local Fallback LaTeX Generator
# ============================================================


def _fallback_german_latex_body(
    resume_text: str,
    missing_skills: list[str],
    layout_style: str = "german_corporate",
    github_url: str = "https://github.com/Baqir110",
) -> str:
    clean = _clean_skill_list(missing_skills)
    skills_formatted = ", ".join(clean) if clean else "Python, SQL, Docker, Linux, Git"

    if layout_style in ("international_ats", "standard", "hr_executive_gold"):
        return rf"""
\section*{{Professional Summary}}
DevOps Engineer experienced in cloud infrastructure, automation, observability, and data systems.

\section*{{Professional Experience}}

\jobheader{{DevOps Engineer}}{{Parkyeri}}{{Istanbul, Turkey}}
\begin{{itemize}}
    \item Implemented Python/TensorFlow forecasting solutions on AWS/Azure, reducing inventory shortages by 20\%.
    \item Supported AWS cloud migration and Nagios/Zabbix monitoring, cutting downtime by 35\% and cloud costs by 18\%.
    \item Redesigned Jira/ServiceNow support workflows, improving first-call resolution from 61\% to 84\%.
\end{{itemize}}

\jobheader{{DevOps Engineer}}{{Hexagon Helix}}{{Istanbul, Turkey}}
\begin{{itemize}}
    \item Integrated 20+ services into Prometheus/Grafana alerting, improving health visibility and reducing incidents by 20\%.
    \item Created structured RCA runbooks, cutting average troubleshooting time by 25\%.
\end{{itemize}}

\section*{{Projects}}

\projheader{{AI IT Operations Assistant}}{{FastAPI, Docker, PostgreSQL}}{{{github_url}/ai-it-ops-assistant}}
\begin{{itemize}}
    \item RAG platform for automated telemetry analysis with <200ms latency.
\end{{itemize}}

\projheader{{Infrastructure Monitoring Platform}}{{Prometheus, Grafana}}{{{github_url}/it-infrastructure-monitoring}}
\begin{{itemize}}
    \item Automated monitoring tools and CI/CD pipelines ensuring high availability.
\end{{itemize}}

\section*{{Education}}

\jobheader{{M.Sc. International Software Systems Science}}{{Univ. Bamberg}}{{Oct 2024 -- Present}}
\jobheader{{B.Sc. Computer Engineering}}{{Istanbul Okan University}}{{Graduated}}

\section*{{Technical Skills}}

\textbf{{Core Skills:}} Python, Java, SQL, FastAPI, Docker, PostgreSQL, Redis, Prometheus, Grafana, Git \\
\textbf{{Keywords:}} {skills_formatted}

\section*{{Languages}}

\textbf{{Languages:}} English (IELTS 8.0), German, Turkish, Urdu, Sindhi
"""
    else:
        return rf"""
\section*{{Profil}}
DevOps Engineer mit Erfahrung in Cloud-Infrastruktur, Automatisierung, Observability und Datenbanksystemen.

\section*{{Berufserfahrung}}

\jobheader{{DevOps Engineer}}{{Parkyeri}}{{Istanbul, Türkei}}
\begin{{itemize}}
    \item Entwicklung von Python/TensorFlow-Prognosen auf AWS/Azure, Bestandsengpässe um 20\% reduziert.
    \item Unterstützung bei AWS-Cloud-Migration und Nagios/Zabbix-Monitoring; 35\% weniger Ausfallzeiten.
    \item Neugestaltung von Jira/ServiceNow-Workflows; First-Call-Resolution von 61\% auf 84\% gesteigert.
\end{{itemize}}

\jobheader{{DevOps Engineer}}{{Hexagon Helix}}{{Istanbul, Türkei}}
\begin{{itemize}}
    \item Integration von 20+ Services in Prometheus/Grafana-Monitoring; wiederkehrende Incidents um 20\% reduziert.
    \item Erstellung von RCA-Runbooks, wodurch Fehleranalysezeit um 25\% sank.
\end{{itemize}}

\section*{{Projekte}}

\projheader{{AI IT Operations Assistant}}{{FastAPI, Docker, PostgreSQL}}{{{github_url}/ai-it-ops-assistant}}
\begin{{itemize}}
    \item RAG-Plattform zur Telemetrie-Analyse mit <200ms Latenz.
\end{{itemize}}

\projheader{{Infrastructure Monitoring Platform}}{{Prometheus, Grafana}}{{{github_url}/it-infrastructure-monitoring}}
\begin{{itemize}}
    \item Einbindung von Monitoring-Tools und CI/CD-Pipelines.
\end{{itemize}}

\section*{{Ausbildung}}

\jobheader{{M.Sc. International Software Systems Science}}{{Univ. Bamberg}}{{Seit Okt 2024}}
\jobheader{{B.Sc. Computer Engineering}}{{Istanbul Okan University}}{{Abschluss}}

\section*{{Kenntnisse}}

\textbf{{Technische Kenntnisse:}} Python, Java, SQL, FastAPI, Docker, PostgreSQL, Redis, Prometheus, Grafana, Git \\
\textbf{{Schlüsselbegriffe:}} {skills_formatted}

\section*{{Sprachen}}

\textbf{{Sprachen:}} Englisch (IELTS 8.0), Deutsch, Türkisch, Urdu, Sindhi
"""


# ============================================================
# Generate LaTeX CV
# ============================================================


def generate_german_latex_content(
    resume_text: str,
    job_description: str,
    missing_skills: list[str],
    provider: str = "experiential",
    model_name: str | None = None,
    api_key: str | None = None,
    route_mode: str = "experiential",
    layout_style: str = "german_corporate",
    primary_color_hex: str | None = None,
    secondary_color_hex: str | None = None,
    linkedin_url: str = "https://www.linkedin.com/in/muhammad-baqir-it/",
    github_url: str = "https://github.com/Baqir110?tab=repositories",
    improvement_suggestions: list[str] | None = None,
) -> str:
    if (layout_style or "").strip().lower() in ("auto", "auto_detect", ""):
        layout_style = auto_select_layout(job_description, resume_text)

    if layout_style not in CV_TEMPLATES:
        layout_style = "german_corporate"

    is_german_minimal_ats = layout_style == GERMAN_MINIMAL_ATS
    invariants = extract_resume_invariants(resume_text) if is_german_minimal_ats else []

    try:
        candidate_header = extract_candidate_header(resume_text)
    except FactualValidationError:
        candidate_header = {
            "name": "Candidate",
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
        }

    target_lang = required_language_for_layout(layout_style)

    # Fallback language detection based on job description keywords
    if not target_lang or target_lang == "any":
        jd_lower = (job_description or "").lower()
        if any(
            w in jd_lower for w in ["deutsch", "aufgaben", "profil", "anforderungen", "kenntnisse"]
        ):
            target_lang = "de"
        else:
            target_lang = "en"

    resume_text = normalize_resume_language(
        resume_text,
        target_lang,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        route_mode=route_mode,
    )

    clean_skills = _clean_skill_list(missing_skills)
    skills_text = ", ".join(clean_skills) if clean_skills else "None provided."
    language_rule = _language_rule(layout_style)
    suggestions_text = _format_actionable_suggestions(improvement_suggestions)

    if layout_style == "academic":
        section_names = (
            "\\section*{Education}\n\n"
            "\\section*{Research Experience}\n\n"
            "\\section*{Publications}\n\n"
            "\\section*{Teaching Experience}\n\n"
            "\\section*{Skills \\& Languages}\n\n"
            "\\section*{Awards \\& Grants}"
        )
    elif layout_style == "technical_lead":
        section_names = (
            "\\section*{Technical Summary}\n\n"
            "\\section*{Professional Experience}\n\n"
            "\\section*{Open Source \\& Projects}\n\n"
            "\\section*{Speaking \\& Community}\n\n"
            "\\section*{Education}\n\n"
            "\\section*{Skills}"
        )
    elif target_lang == "de":
        section_names = (
            "\\section*{Profil}\n\n"
            "\\section*{Berufserfahrung}\n\n"
            "\\section*{Projekte}\n\n"
            "\\section*{Ausbildung}\n\n"
            "\\section*{Kenntnisse}\n\n"
            "\\section*{Sprachen \\& Zertifikate}"
        )
    else:
        section_names = (
            "\\section*{Profile}\n\n"
            "\\section*{Work Experience}\n\n"
            "\\section*{Projects}\n\n"
            "\\section*{Education}\n\n"
            "\\section*{Skills}\n\n"
            "\\section*{Languages \\& Certificates}"
        )

    prompt = rf"""
Optimize and enrich the candidate's CV body content to achieve the highest possible match against the target job description.

{language_rule}

Target Job Description:
{job_description or "General DevOps / Cloud Engineering position."}

Critical Skills to Integrate Naturally:
{skills_text}

ACTIONABLE IMPROVEMENTS FROM THE CANDIDATE'S PRIOR ATS AUDIT:
{suggestions_text}

Original Resume Text:
{resume_text}

Selected Layout Style:
{layout_style}

STRICT ONE-PAGE LIMIT AND CONCISION REQUIREMENTS:
1. THE FINAL DOCUMENT MUST FIT ON EXACTLY ONE (1) A4 PAGE.
2. KEEP ALL BULLET POINTS CONCISE AND LIMITED:
   - Max 2 to 3 bullet points per work experience entry.
   - Max 1 to 2 bullet points per project entry.
   - Summary/Profil section must be EXACTLY 1 to 2 short sentences.
   - Limit skill lists to top key terms per category.
   - Do NOT output extra blank lines or redundant details.

STRICT STRUCTURAL AND CONTENT RULES:

1. DO NOT GENERATE ANY CONTACT HEADER, SIDEBAR, OR NAME BLOCK AT THE TOP.
   Start directly with the first section header:
   \section*{{Profil}} or \section*{{Profile}}

2. PRESERVE ALL ORIGINAL DATA:
   - Do NOT delete existing work experience entries, degrees, or projects.
   - Do NOT change degree titles or company names.

3. REPOSITORY LINKS RULE:
   - For project links, strictly use valid repositories under: {github_url} (e.g. {github_url}/ai-it-ops-assistant).

4. WORK EXPERIENCE, EDUCATION & PROJECTS:
   - Every bullet point inside \begin{{itemize}} MUST start strictly with \item.
   - Use \jobheader{{Role / Degree}}{{Company / University}}{{Dates}} for BOTH jobs and education.
   - Use \projheader{{Project Name}}{{Tech Stack}}{{{github_url}/repo-name}} for projects.
   - Professional positioning is DevOps / Cloud / Platform Engineering.
   - AUTHORITATIVE EXPERIENCE FACTS: Parkyeri — DevOps Engineer: Python/TensorFlow forecasting on AWS/Azure; 20% fewer shortages; AWS migration with Nagios, Zabbix, VMware; 35% downtime reduction; 18% cloud cost reduction; Jira/ServiceNow resolution from 61% to 84%. Hexagon Helix — DevOps Engineer: integrated 20+ systems into Prometheus/Grafana/Azure Monitor; created incident runbooks; 25% faster troubleshooting; managed DNS, VPNs, firewalls, TLS, Cisco/Juniper.
   - Preserve these metrics when roles are included. Do not invent metrics or employers.

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
   \textless{{}} for <
   \textgreater{{}} for >

8. Return RAW LaTeX body content ONLY (No code fences, markdown, or conversational text).
"""

    try:
        raw_latex = LLMService.generate(
            prompt=prompt,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            route_mode=route_mode,
        )

        raw_latex = _ensure_suggestions_applied_latex(
            generated_text=raw_latex or "",
            suggestions=improvement_suggestions or [],
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            route_mode=route_mode,
        )

        clean_body = clean_llm_response_to_latex(raw_latex)
        clean_body = clean_body_for_latex(clean_body)

        _legacy_it = "IT" + r"[ -]?" + "Support"
        _legacy_support_engineer = "Support" + r"[ -]?" + "Engineer"
        clean_body = re.sub(
            rf"(?i){_legacy_it}[ -]?Engineer",
            DEFAULT_CV_TITLE,
            clean_body,
        )
        clean_body = re.sub(
            rf"(?i){_legacy_support_engineer}",
            DEFAULT_CV_TITLE,
            clean_body,
        )
        clean_body = re.sub(
            rf"(?i){_legacy_it}",
            "DevOps and infrastructure engineering",
            clean_body,
        )
        from app.core.event_log import log_event as _le

        _le(
            "latex",
            "latex_generation_completed",
            layout=layout_style,
            body_chars=len(clean_body or ""),
            target_language=required_language_for_layout(layout_style),
        )
    except Exception as exc:
        if is_german_minimal_ats:
            raise FactualValidationError(
                [
                    "German Minimal ATS generation failed before factual validation; no CV was produced."
                ]
            ) from exc
        clean_body = _fallback_german_latex_body(
            resume_text=resume_text,
            missing_skills=missing_skills,
            layout_style=layout_style,
            github_url=github_url,
        )

    template = CV_TEMPLATES[layout_style]
    template = _patch_template_preamble(template)

    template = template.replace("LINKEDIN_URL_PLACEHOLDER", linkedin_url)
    template = template.replace("GITHUB_URL_PLACEHOLDER", github_url)

    if (
        "CANDIDATE_NAME_PLACEHOLDER" in template
        or "CANDIDATE_CONTACT_PLACEHOLDER" in template
        or "DevOps Engineer" in template
    ):
        _hdr = candidate_header

        template = template.replace(
            "CANDIDATE_NAME_PLACEHOLDER",
            _escape_latex_text(_hdr.get("name") or "Candidate"),
        )
        template = template.replace(
            "DevOps Engineer",
            _escape_latex_text(DEFAULT_CV_TITLE),
        )

        _ln = _hdr.get("linkedin") or linkedin_url
        _gh = _hdr.get("github") or github_url

        _contact_parts = []
        if _hdr.get("phone"):
            _contact_parts.append(_escape_latex_text(_hdr["phone"]))
        if _hdr.get("email"):
            _contact_parts.append(
                rf"\hrlink{{mailto:{_hdr['email']}}}" rf"{{{_escape_latex_text(_hdr['email'])}}}"
            )
        if _ln:
            _contact_parts.append(rf"\hrlink{{\detokenize{{{latex_escape_url(_ln)}}}}}{{LinkedIn}}")
        if _gh:
            _contact_parts.append(rf"\hrlink{{\detokenize{{{latex_escape_url(_gh)}}}}}{{GitHub}}")
        template = template.replace(
            "CANDIDATE_CONTACT_PLACEHOLDER",
            r" \quad$\cdot$\quad ".join(_contact_parts),
        )

    if primary_color_hex:
        template = re.sub(
            r"\\definecolor\{primary\}\{HTML\}\{[A-Fa-f0-9]{6}\}",
            rf"\\definecolor{{primary}}{{HTML}}{{{primary_color_hex.strip('#')}}}",
            template,
        )

    if secondary_color_hex:
        template = re.sub(
            r"\\definecolor\{secondary\}\{HTML\}\{[A-Fa-f0-9]{6}\}",
            rf"\\definecolor{{secondary}}{{HTML}}{{{secondary_color_hex.strip('#')}}}",
            template,
        )

    latex_code = template.replace("RESUME_BODY_PLACEHOLDER", clean_body)
    if is_german_minimal_ats:
        validate_generated_invariants(latex_code, invariants)
    validate_cv_layout_template(layout_style, latex_code, template=template)

    _forbidden_legacy_terms = [
        r"(?i)" + "IT" + r"[ -]?" + "Support",
        r"(?i)" + "Support" + r"[ -]?" + "Engineer",
    ]
    if any(re.search(pattern, latex_code) for pattern in _forbidden_legacy_terms):
        raise FactualValidationError(
            ["Generated CV still contains forbidden legacy support terminology."]
        )
    if DEFAULT_CV_TITLE not in latex_code:
        raise FactualValidationError(
            [f"Generated CV does not contain the canonical title '{DEFAULT_CV_TITLE}'."]
        )

    return latex_code


# ============================================================
# Compile LaTeX to PDF
# ============================================================


def compile_latex_to_pdf(latex_code: str) -> bytes:
    """
    Compile LaTeX to PDF with a hardened subprocess invocation.
    """
    import time as _time

    from app.core.event_log import log_event

    _p_start = _time.perf_counter()
    log_event("pdf", "pdf_compilation_started", latex_chars=len(latex_code or ""))

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError(
            "pdflatex was not found on PATH. TinyTeX/TeX Live directory is not available."
        )

    latex_code = normalize_latex_links(latex_code)

    if os.getenv("LATEX_DEBUG_DUMP") == "1":
        try:
            Path("debug_german_cv.tex").write_text(latex_code, encoding="utf-8")
        except OSError:
            logger.warning("Could not write debug_german_cv.tex", exc_info=True)

    env = os.environ.copy()
    env["MIKTEX_GUI_MODE"] = "no"
    env["MIKTEX_AUTOINSTALL"] = "0"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        tex_path = tmp_path / "resume.tex"
        pdf_path = tmp_path / "resume.pdf"
        log_path = tmp_path / "resume.log"
        out_path = tmp_path / "pdflatex.out"
        err_path = tmp_path / "pdflatex.err"

        tex_path.write_text(latex_code, encoding="utf-8")

        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-no-shell-escape",
            f"-output-directory={tmp_path}",
            str(tex_path),
        ]

        def _read(path: Path) -> str:
            try:
                return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            except OSError:
                return ""

        def _diagnostics() -> str:
            log_tail = _read(log_path)[-8000:] or "<no log produced>"
            out_tail = _read(out_path)[-4000:]
            return (
                "---- resume.log tail ----\n"
                f"{log_tail}\n\n"
                "---- pdflatex stdout tail ----\n"
                f"{out_tail}"
            )

        def _run_pass(label: str) -> str:
            with (
                open(out_path, "w", encoding="utf-8", errors="replace") as out_f,
                open(err_path, "w", encoding="utf-8", errors="replace") as err_f,
            ):
                proc = subprocess.run(
                    command,
                    cwd=tmp_path,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    check=False,
                    timeout=60,
                )

            stdout = _read(out_path)
            stderr = _read(err_path)

            if proc.returncode != 0:
                error_tail = (stdout + "\n" + stderr)[-6000:]
                raise RuntimeError(
                    f"LaTeX compilation failed on {label} pass "
                    f"(exit {proc.returncode}).\n\n"
                    f"STDOUT/STDERR tail:\n{error_tail}\n\n"
                    f"{_diagnostics()}"
                )
            return stdout

        try:
            _run_pass("first")
            _run_pass("second")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "LaTeX compilation timed out after 60 seconds per pass.\n\n" f"{_diagnostics()}"
            ) from exc

        if not pdf_path.exists():
            log_event(
                "pdf",
                "pdf_compilation_failed",
                duration_ms=round((_time.perf_counter() - _p_start) * 1000, 1),
                error="pdflatex did not produce a PDF",
            )
            raise RuntimeError(
                "pdflatex completed but no PDF was produced.\n\n" f"{_diagnostics()}"
            )

        _bytes = pdf_path.read_bytes()
        _pages = len(re.findall(rb"/Type\s*/Page[^s]", _bytes)) or 1
        if _pages != 1:
            raise RuntimeError(
                f"Generated CV must be exactly 1 page; LaTeX produced {_pages} pages."
            )
        log_event(
            "pdf",
            "pdf_compiled",
            pages=_pages,
            bytes=len(_bytes),
            duration_ms=round((_time.perf_counter() - _p_start) * 1000, 1),
        )
        return _bytes
