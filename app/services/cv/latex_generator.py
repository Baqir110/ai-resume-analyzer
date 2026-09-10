import logging
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from app.services.llm.provider import LLMService
from app.services.cv.optimizer import (
    _clean_skill_list,
    _format_actionable_suggestions,
    auto_select_layout,
    normalize_resume_language,
    required_language_for_layout,
)

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
    return "OUTPUT LANGUAGE: Match the language of the job description."


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
    out: List[str] = []
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

    def __init__(self, violations: List[str]):
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


def _section_lines(resume_text: str, headings: List[str]) -> List[str]:
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


def extract_resume_invariants(resume_text: str) -> List[str]:
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
    invariants: List[str] = []

    if experience and not any(_DATE_RE.search(line) for line in experience):
        raise FactualValidationError(
            [
                "Could not unambiguously extract career facts from the Experience section."
            ]
        )

    for line in experience:
        if not _DATE_RE.search(line):
            continue
        parts = [
            part.strip() for part in re.split(r"\s*(?:\||—|–)\s*", line) if part.strip()
        ]
        dated_parts = [part for part in parts if _DATE_RE.search(part)]
        factual_parts = [part for part in parts if not _DATE_RE.search(part)]
        if len(factual_parts) < 2 or not dated_parts:
            raise FactualValidationError(
                [
                    f"Could not unambiguously extract role, company, and dates from: {line}"
                ]
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


def extract_candidate_header(resume_text: str) -> Dict[str, str]:
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
        raise FactualValidationError(
            ["Could not extract a candidate name for the CV header."]
        )

    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", resume_text)
    phone_match = re.search(r"(?:\+?\d[\d ()/-]{6,}\d)", resume_text)
    links = re.findall(r"https?://[^\s)]+", resume_text)
    return {
        "name": name,
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "linkedin": next(
            (link for link in links if "linkedin.com" in link.casefold()), ""
        ),
        "github": next((link for link in links if "github.com" in link.casefold()), ""),
    }


def validate_generated_invariants(latex_code: str, invariants: List[str]) -> None:
    from app.core.event_log import log_event

    normalized_output = _normalize_fact(latex_code)
    missing = [
        fact for fact in invariants if _normalize_fact(fact) not in normalized_output
    ]
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


def _render_candidate_contact(header: Dict[str, str]) -> str:
    parts = []
    if header["email"]:
        email = _escape_latex_text(header["email"])
        parts.append(rf"\href{{mailto:{header['email']}}}{{{email}}}")
    if header["phone"]:
        parts.append(_escape_latex_text(header["phone"]))
    if header["linkedin"]:
        parts.append(
            rf"\href{{\detokenize{{{latex_escape_url(header['linkedin'])}}}}}{{LinkedIn}}"
        )
    if header["github"]:
        parts.append(
            rf"\href{{\detokenize{{{latex_escape_url(header['github'])}}}}}{{GitHub}}"
        )
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


def _extract_key_terms(suggestions: List[str]) -> List[str]:
    """
    Extract candidate must-have terms from actionable suggestions.

    Only returns tokens that look like real technical terms:
      - Contain an uppercase letter  (Python, AWS, Docker, Kubernetes, SQLAlchemy)
      - Contain a digit              (K8s, IPv4, Windows11)
      - Contain a symbol             (C#, C++, CI/CD, .NET, Palo-Alto)

    Prose words like 'identified', 'issues', 'closed', 'managed', 'reduced',
    'standardizing' are rejected because they have none of these signals.
    """
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
    suggestions: List[str],
    provider: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    route_mode: str = "experiential",
) -> str:
    """
    Log which suggestion terms are missing from the generated LaTeX body.

    Non-blocking: no re-prompt. The suggestions were already part of the
    first LLM call's prompt as hard constraints.
    """
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

    if (
        text.startswith("I'm ready")
        or text.startswith("Sure")
        or "Please provide" in text
    ):
        return (
            r"\section*{Profil}"
            + "\n"
            + r"\noindent Resume optimization pending input."
        )

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
# TEMPLATES
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

    {\Large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \& MLOps}\\[3pt]

    {\small\color{subgray}
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

RESUME_BODY_PLACEHOLDER

\vspace{4pt}

\noindent
\small\color{subgray}Bamberg, \today

\end{document}
"""

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

    {\large IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[3pt]

    {\small
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

\vspace{4pt}

\noindent
{\small Bamberg, \today}

\end{document}
"""

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

    {\huge\bfseries Muhammad Baqir}\\[3pt]

    {\large IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small
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

\vspace{2pt}
\hrule height 1pt
\vspace{4pt}

RESUME_BODY_PLACEHOLDER

\vspace{6pt}

\noindent
{\small Bamberg, den \today}

\end{document}
"""

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

    {\large\bfseries\color{primary}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[3pt]

    {\small\color{secondary}
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

RESUME_BODY_PLACEHOLDER

\vspace{4pt}

\noindent
{\small\color{secondary}Bamberg, \today}

\end{document}
"""

INTERNATIONAL_ATS_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[top=1.0cm, bottom=1.0cm, left=1.2cm, right=1.2cm]{geometry}
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
  [\vspace{-3pt}\color{subgray}\rule{\textwidth}{0.6pt}]
\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{leftmargin=1.1em, itemsep=1pt, topsep=1pt, parsep=0pt}

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
  {\Huge \bfseries \color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[3pt]
  {\Large \bfseries \color{primary} CANDIDATE_TITLE_PLACEHOLDER}\\[4pt]
  {\small \color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER

\end{document}
"""

STANDARD_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.0cm,bottom=1.0cm,left=1.5cm,right=1.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\hyphenpenalty=10000
\exhyphenpenalty=10000

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

\titlespacing{\section}{0pt}{6pt}{3pt}

\setlist[itemize]{
    leftmargin=1.2em,
    itemsep=1pt,
    topsep=1pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}

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

    {\LARGE\bfseries\color{primary} Muhammad Baqir}\\[3pt]

    {\large\color{secondary} IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small\color{secondary}
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
\documentclass[10pt,a4paper]{article}

\usepackage[top=0.75cm,bottom=0.75cm,left=1.1cm,right=1.1cm]{geometry}
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
    [\vspace{-2pt}\color{goldaccent}\rule{\textwidth}{1.0pt}]

\titlespacing{\section}{0pt}{5pt}{3pt}

\setlist[itemize]{
    leftmargin=1.1em,
    itemsep=0.8pt,
    topsep=0.8pt,
    parsep=0pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5pt}

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

    {\Huge\bfseries\color{primary} Muhammad Baqir}\\[3pt]

    {\large\bfseries\color{goldaccent}
    IT Support Engineer \textbar{} DevOps \textbar{} MLOps}\\[4pt]

    {\small\color{secondary}
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

\vspace{-2pt}

RESUME_BODY_PLACEHOLDER

\end{document}
"""

GERMAN_MINIMAL_ATS_LATEX_TEMPLATE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[top=1.25cm,bottom=1.25cm,left=1.6cm,right=1.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}

\definecolor{primary}{HTML}{0F172A}
\definecolor{subgray}{HTML}{475569}
\hypersetup{colorlinks=true,urlcolor=primary,pdfborder={0 0 0}}
\titleformat{\section}{\large\bfseries\color{primary}}{}{0em}{}[\vspace{-3pt}\color{subgray}\rule{\textwidth}{0.5pt}]
\titlespacing{\section}{0pt}{8pt}{3pt}
\setlist[itemize]{leftmargin=1.2em,itemsep=1pt,topsep=1pt,parsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}
\newcommand{\jobheader}[3]{\noindent\textbf{\color{primary}#1}, #2\hfill\textit{\color{subgray}#3}\par\vspace{1pt}}
\newcommand{\degreeheader}[3]{\jobheader{#1}{#2}{#3}}
\newcommand{\projheader}[3]{\noindent\textbf{\color{primary}#1}\ifx\relax#2\relax\else\textit{\color{subgray}(#2)}\fi\ifx\relax#3\relax\else\hfill\href{\detokenize{#3}}{GitHub}\fi\par\vspace{1pt}}
\pagestyle{empty}

\begin{document}
\begin{center}
{\Huge\bfseries\color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[3pt]
{\small\color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER
\end{document}
"""

CV_TEMPLATES = {
    "german_corporate": GERMAN_CORPORATE_LATEX_TEMPLATE,
    "german_ats": GERMAN_ATS_LATEX_TEMPLATE,
    "german_classic": GERMAN_CLASSIC_LATEX_TEMPLATE,
    "german_modern": GERMAN_MODERN_LATEX_TEMPLATE,
    GERMAN_MINIMAL_ATS: GERMAN_MINIMAL_ATS_LATEX_TEMPLATE,
    "international_ats": INTERNATIONAL_ATS_LATEX_TEMPLATE,
    "standard": STANDARD_LATEX_TEMPLATE,
    "hr_executive_gold": HR_EXECUTIVE_GOLD_LATEX_TEMPLATE,
}


# ============================================================
# Local Fallback LaTeX Generator
# ============================================================


def _fallback_german_latex_body(
    resume_text: str,
    missing_skills: List[str],
    layout_style: str = "german_corporate",
    github_url: str = "https://github.com/Baqir110",
) -> str:
    clean = _clean_skill_list(missing_skills)
    skills_formatted = ", ".join(clean) if clean else "Python, SQL, Docker, Linux, Git"

    if layout_style in ("international_ats", "standard", "hr_executive_gold"):
        return rf"""
\section*{{Professional Summary}}
Motivated IT professional with hands-on experience in software engineering, IT support, infrastructure automation, and data systems. Proven ability to apply technical know-how to deliver reliable software and optimize day-to-day operations.

\section*{{Professional Experience}}

\jobheader{{IT Support Engineer \& Data Specialist}}{{Parkyeri \& Hexagon Helix}}{{Istanbul, Turkey}}
\begin{{itemize}}
    \item Provided technical system support, troubleshooting, and database maintenance across server environments.
    \item Applied in-depth problem-solving skills to manage day-to-day system health and resolve operational issues efficiently.
    \item Collaborated in a team-oriented setting to deploy and maintain software services.
\end{{itemize}}

\section*{{Projects}}

\projheader{{AI IT Operations Assistant}}{{Python, FastAPI, Docker, PostgreSQL, Redis}}{{{github_url}/ai-it-ops-assistant}}
\begin{{itemize}}
    \item Containerized RAG platform for automated telemetry analysis and runbook search with <200ms API latency.
\end{{itemize}}

\projheader{{IT Infrastructure Monitoring}}{{Docker, Prometheus, Grafana, Python}}{{{github_url}/it-infrastructure-monitoring}}
\begin{{itemize}}
    \item Integrated automated monitoring tools and CI/CD pipelines to ensure continuous system availability.
\end{{itemize}}

\projheader{{Customer Churn Analytics Service}}{{Python, Scikit-Learn, FastAPI, Streamlit}}{{{github_url}/customer-churn-analytics}}
\begin{{itemize}}
    \item Built end-to-end ML microservice for churn prediction with feature explanation endpoints.
\end{{itemize}}

\section*{{Education}}

\jobheader{{M.Sc. in International Software Systems Science}}{{Otto-Friedrich-Universität Bamberg}}{{Oct 2024 -- Present}}
\jobheader{{B.Sc. in Computer Engineering}}{{Istanbul Okan University}}{{Graduated}}

\section*{{Technical Skills}}

\textbf{{Core Technical Skills:}} Python, Java, SQL, HTML, FastAPI, Docker, PostgreSQL, Redis, Prometheus, Grafana, Git \\
\textbf{{Integrated Job Keywords:}} {skills_formatted}

\section*{{Languages \& Certifications}}

\textbf{{Languages:}} English (IELTS 8.0), German, Turkish, Urdu, Sindhi
"""
    else:
        return rf"""
\section*{{Profil}}
Engagierter IT-Spezialist mit praktischer Erfahrung in Softwareentwicklung, IT-Support, Infrastruktur-Automatisierung und Datenbanksystemen. Erfahren in der Anwendung von fundiertem Know-how zur Optimierung alltäglicher Systemabläufe.

\section*{{Berufserfahrung}}

\jobheader{{IT Support Engineer \& Data Specialist}}{{Parkyeri \& Hexagon Helix}}{{Istanbul, Türkei}}
\begin{{itemize}}
    \item Durchführung von technischem Support, Systemwartung und Fehlerbehebung in Serverumgebungen.
    \item Anwendung von in-depth Lösungsansätzen im täglichen Betrieb zur Sicherstellung hoher Systemverfügbarkeit.
    \item Erfolgreiche Zusammenarbeit in teamorientierten Agile-Prozessen zur Bereitstellung von Softwarelösungen.
\end{{itemize}}

\section*{{Projekte}}

\projheader{{AI IT Operations Assistant}}{{Python, FastAPI, Docker, PostgreSQL, Redis}}{{{github_url}/ai-it-ops-assistant}}
\begin{{itemize}}
    \item Containerisierte RAG-Plattform zur automatisierten Telemetrie-Analyse und Runbook-Suche.
\end{{itemize}}

\projheader{{IT-Infrastruktur Monitoring}}{{Docker, Prometheus, Grafana, Python}}{{{github_url}/it-infrastructure-monitoring}}
\begin{{itemize}}
    \item Einbindung von Monitoring-Tools und CI/CD-Pipelines zur Erhöhung der Systemstabilität.
\end{{itemize}}

\section*{{Ausbildung}}

\jobheader{{M.Sc. International Software Systems Science}}{{Otto-Friedrich-Universität Bamberg}}{{Seit Okt 2024}}
\jobheader{{B.Sc. Computer Engineering}}{{Istanbul Okan University}}{{Abschluss}}

\section*{{Kenntnisse}}

\textbf{{Technische Kenntnisse:}} Python, Java, SQL, HTML, FastAPI, Docker, PostgreSQL, Redis, Prometheus, Grafana, Git \\
\textbf{{Integrierte Schlüsselbegriffe:}} {skills_formatted}

\section*{{Sprachen \& Zertifikate}}

\textbf{{Sprachen:}} Englisch (IELTS 8.0), Deutsch, Türkisch, Urdu, Sindhi
"""


# ============================================================
# Generate LaTeX CV
# ============================================================


def generate_german_latex_content(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    provider: str = "experiential",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    route_mode: str = "experiential",
    layout_style: str = "german_corporate",
    primary_color_hex: Optional[str] = None,
    secondary_color_hex: Optional[str] = None,
    linkedin_url: str = "https://www.linkedin.com/in/muhammad-baqir-it/",
    github_url: str = "https://github.com/Baqir110?tab=repositories",
    improvement_suggestions: Optional[List[str]] = None,
) -> str:
    if (layout_style or "").strip().lower() in ("auto", "auto_detect", ""):
        layout_style = auto_select_layout(job_description, resume_text)

    if layout_style not in CV_TEMPLATES:
        layout_style = "german_corporate"

    is_german_minimal_ats = layout_style == GERMAN_MINIMAL_ATS
    invariants = extract_resume_invariants(resume_text) if is_german_minimal_ats else []

    target_lang = required_language_for_layout(layout_style)
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

    if layout_style in ("international_ats", "standard", "hr_executive_gold"):
        section_names = (
            "\\section*{Profile}\n\n"
            "\\section*{Work Experience}\n\n"
            "\\section*{Projects}\n\n"
            "\\section*{Education}\n\n"
            "\\section*{Skills}\n\n"
            "\\section*{Languages \\& Certificates}"
        )
    else:
        section_names = (
            "\\section*{Profil}\n\n"
            "\\section*{Berufserfahrung}\n\n"
            "\\section*{Projekte}\n\n"
            "\\section*{Ausbildung}\n\n"
            "\\section*{Kenntnisse}\n\n"
            "\\section*{Sprachen \\& Zertifikate}"
        )

    prompt = rf"""
Optimize and enrich the candidate's CV body content to achieve the highest possible match against the target job description.

{language_rule}

Target Job Description:
{job_description or "General IT Support / DevOps position."}

Critical Skills to Integrate Naturally:
{skills_text}

ACTIONABLE IMPROVEMENTS FROM THE CANDIDATE'S PRIOR ATS AUDIT
(treat every item below as a HARD CONSTRAINT — the CV is rejected if any
item is not satisfied):
{suggestions_text}

Original Resume Text:
{resume_text}

Selected Layout Style:
{layout_style}

STRICT STRUCTURAL AND CONTENT RULES:

1. DO NOT GENERATE ANY CONTACT HEADER, SIDEBAR, OR NAME BLOCK AT THE TOP.
   The document preamble already renders candidate contact headers.
   Start directly with the first section:
   \section*{{Profil}} or \section*{{Profile}}

2. PRESERVE ALL ORIGINAL DATA:
   - Do NOT delete any existing work experience entries, degrees, or projects.
   - Do NOT change degree titles or company names.
   - You MAY expand existing experience and project bullet points by adding technical context, missing keywords, and relevant details from the job description.

3. REPOSITORY LINKS RULE:
   - For any project links, strictly use valid repositories under the base URL: {github_url} (e.g. {github_url}/ai-it-ops-assistant, {github_url}/it-infrastructure-monitoring, {github_url}/customer-churn-analytics, {github_url}/real-time-pipeline).
   - Do NOT output generic placeholders like "repo-name" or "Link".

4. WORK EXPERIENCE, EDUCATION & PROJECTS:
   - Every bullet point inside \begin{{itemize}} MUST start strictly with \item.
   - Use \jobheader{{Role / Degree}}{{Company / University}}{{Dates}} for BOTH jobs and education. DO NOT use \degreeheader.
   - Use \projheader{{Project Name}}{{Tech Stack}}{{{github_url}/repo-name}} for projects.

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

8. APPLY EVERY ACTIONABLE IMPROVEMENT listed near the top. If an
   improvement names specific terms, those exact terms MUST appear verbatim
   in the LaTeX body — either in the Technical Skills / Kenntnisse section
   or in a bullet point.

9. Return RAW LaTeX body content ONLY (No code fences, markdown, or conversational text).

ABSOLUTE PROHIBITIONS — VIOLATION WILL BE REJECTED:
- Do NOT create any section called "Ergänzende Terminologie",
  "Ergänzende Such- und Schreibvarianten", "Additional Keywords",
  "Supplementary Terms", "Technische Weiterbildungsziele", or anything
  similar. Keyword-dump sections cause ATS parsers to misread the CV.
  Every keyword MUST be integrated into an existing experience bullet or
  the Technical Skills / Kenntnisse block.
- Do NOT include English or German function words (tools, frameworks,
  before, after, these, low, bereit, netzwerkst, ablauf, umsetzung,
  experiential, etc.) as if they were skills. Only real, actionable
  technologies and competencies may appear in the skills section.
- Do NOT invent new companies, degrees, dates, or job titles.
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
        or "CANDIDATE_TITLE_PLACEHOLDER" in template
    ):
        try:
            _hdr = extract_candidate_header(resume_text)
        except FactualValidationError:
            _hdr = {
                "name": "Candidate",
                "email": "",
                "phone": "",
                "linkedin": "",
                "github": "",
            }

        template = template.replace(
            "CANDIDATE_NAME_PLACEHOLDER",
            _escape_latex_text(_hdr.get("name") or "Candidate"),
        )
        template = template.replace(
            "CANDIDATE_TITLE_PLACEHOLDER",
            _escape_latex_text(_infer_title(resume_text) or "Software Engineer"),
        )

        _ln = _hdr.get("linkedin") or linkedin_url
        _gh = _hdr.get("github") or github_url

        _contact_parts = []
        if _hdr.get("phone"):
            _contact_parts.append(_escape_latex_text(_hdr["phone"]))
        if _hdr.get("email"):
            _contact_parts.append(
                rf"\hrlink{{mailto:{_hdr['email']}}}"
                rf"{{{_escape_latex_text(_hdr['email'])}}}"
            )
        if _ln:
            _contact_parts.append(
                rf"\hrlink{{\detokenize{{{latex_escape_url(_ln)}}}}}{{LinkedIn}}"
            )
        if _gh:
            _contact_parts.append(
                rf"\hrlink{{\detokenize{{{latex_escape_url(_gh)}}}}}{{GitHub}}"
            )
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
    return latex_code


# ============================================================
# Compile LaTeX to PDF
# ============================================================


def compile_latex_to_pdf(latex_code: str) -> bytes:
    import time as _time
    from app.core.event_log import log_event

    _p_start = _time.perf_counter()
    log_event("pdf", "pdf_compilation_started", latex_chars=len(latex_code or ""))

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

        tex_path.write_text(latex_code, encoding="utf-8")

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
                raise RuntimeError(f"LaTeX compilation failed:\n\n{error_tail}")

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
                    f"LaTeX compilation failed on the second pass:\n\n{error_tail}"
                )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("LaTeX compilation timed out after 30 seconds.") from exc

        if not pdf_path.exists():
            log_event(
                "pdf",
                "pdf_compilation_failed",
                duration_ms=round((_time.perf_counter() - _p_start) * 1000, 1),
                error="pdflatex did not produce a PDF",
            )
            raise RuntimeError("pdflatex completed but no PDF was produced.")

        _bytes = pdf_path.read_bytes()
        _pages = len(re.findall(rb"/Type\s*/Page[^s]", _bytes)) or 1
        log_event(
            "pdf",
            "pdf_compiled",
            pages=_pages,
            bytes=len(_bytes),
            duration_ms=round((_time.perf_counter() - _p_start) * 1000, 1),
        )
        return _bytes
