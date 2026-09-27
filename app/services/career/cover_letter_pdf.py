"""HR-grade PDF cover letter generator with automatic language, company, and role detection.

Produces a modern executive business-letter layout that prints on one page.
Supports English and German business-letter conventions with 100% human-like phrasing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from app.services.cv.latex_generator import (
    _escape_latex_text,
    extract_candidate_header,
    latex_escape_url,
)
from app.services.cv.pdf_compiler import (
    LaTeXSourceError,
    compile_single_page_pdf,
    validate_latex_document,
)

MAX_COVER_RESUME_CHARS = 200_000
MAX_COVER_JOB_DESCRIPTION_CHARS = 200_000
MAX_COVER_MODEL_RESPONSE_CHARS = 100_000


# ============================================================
# Language & Company Detection Helpers
# ============================================================

_GERMAN_MARKERS = {
    "und",
    "oder",
    "mit",
    "für",
    "von",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "wir",
    "sie",
    "als",
    "auf",
    "aus",
    "bei",
    "nach",
    "über",
    "unter",
    "zwischen",
    "kann",
    "muss",
    "soll",
    "wird",
    "werden",
    "sind",
    "haben",
    "hat",
    "unsere",
    "unser",
    "ihre",
    "ihr",
    "bewerbung",
    "erfahrung",
    "kenntnisse",
    "aufgaben",
    "anforderungen",
    "stelle",
    "unternehmen",
    "team",
    "stellenangebot",
    "lebenslauf",
    "anschreiben",
    "gmbh",
    "ag",
    "kg",
}

_ENGLISH_MARKERS = {
    "and",
    "or",
    "with",
    "for",
    "from",
    "the",
    "you",
    "we",
    "as",
    "on",
    "at",
    "in",
    "of",
    "to",
    "is",
    "are",
    "have",
    "has",
    "our",
    "your",
    "their",
    "this",
    "that",
    "will",
    "can",
    "must",
    "should",
    "be",
    "experience",
    "skills",
    "requirements",
    "position",
    "role",
    "job",
    "company",
    "team",
    "resume",
    "cv",
    "apply",
    "candidate",
    "looking",
    "hiring",
    "responsibilities",
    "qualifications",
}


def detect_language(text: str, default: str = "en") -> str:
    """Detect whether a text is mostly German or English."""
    if not text:
        return default

    tokens = re.findall(r"\b[a-zA-ZäöüßÄÖÜ]+\b", text.lower())
    if not tokens:
        return default

    sample = tokens[:400]
    german_hits = sum(1 for token in sample if token in _GERMAN_MARKERS)
    english_hits = sum(1 for token in sample if token in _ENGLISH_MARKERS)

    umlaut_hits = sum(1 for char in text if char in "äöüßÄÖÜ")
    german_score = german_hits + umlaut_hits * 2
    english_score = english_hits

    return "de" if german_score > english_score else "en"


def _extract_company_from_jd(job_description: str) -> str:
    """Extract company name using high-confidence regex rules for DE and EN job descriptions."""
    if not job_description:
        return ""

    clean_jd = re.sub(r"[®™©]", "", job_description)

    legal_form_match = re.search(
        r"\b([A-Z0-9][A-Za-z0-9&\-\.\s]{1,40}\s+(?:GmbH|AG|SE|e\.K\.|KG|KGaA|Inc\.|LLC|Ltd\.|Corporation|Corp\.|systems|solutions|technologies|group|labs))\b",
        clean_jd,
        re.IGNORECASE,
    )
    if legal_form_match:
        return legal_form_match.group(1).strip()

    intro_match = re.search(
        r"(?:Willkommen bei|bei der|Karriere bei|about|join|working at|bei)\s+([A-Z0-9][A-Za-z0-9&\-\.\s]{1,30})",
        clean_jd,
        re.IGNORECASE,
    )
    if intro_match:
        found = intro_match.group(1).strip()
        found = re.split(
            r"\s+(?:ist|bietet|sucht|is|has|will|um|mit)\b", found, maxsplit=1, flags=re.IGNORECASE
        )[0]
        return found.strip()

    return ""


# ============================================================
# Modernized LaTeX Template
# ============================================================

COVER_LETTER_LATEX_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[top=1.8cm,bottom=1.8cm,left=2.0cm,right=2.0cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}

\definecolor{primary}{HTML}{0F172A}
\definecolor{accent}{HTML}{0284C7}
\definecolor{muted}{HTML}{475569}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.75em}
\pagestyle{empty}

\begin{document}

% ---------------------------------------------------------------------------
% Header Block
% ---------------------------------------------------------------------------
{\Huge\bfseries\color{primary} SENDER_NAME_PLACEHOLDER}\\[4pt]
{\small\color{muted} SENDER_CONTACT_PLACEHOLDER}

\vspace{0.4em}
{\color{accent}\rule{\textwidth}{1.5pt}}
\vspace{1.0em}

% ---------------------------------------------------------------------------
% Metadata Block (Recipient & Date)
% ---------------------------------------------------------------------------
\begin{minipage}[t]{0.58\textwidth}
    {\small\color{primary} RECIPIENT_BLOCK_PLACEHOLDER}
\end{minipage}
\hfill
\begin{minipage}[t]{0.38\textwidth}
    \raggedleft
    {\small\color{muted} SENDER_CITY_DATE_PLACEHOLDER}
\end{minipage}

\vspace{1.5em}

% ---------------------------------------------------------------------------
% Subject Line
% ---------------------------------------------------------------------------
{\large\bfseries\color{primary} SUBJECT_LINE_PLACEHOLDER}

\vspace{0.8em}

% ---------------------------------------------------------------------------
% Letter Body
% ---------------------------------------------------------------------------
{\color{primary} SALUTATION_PLACEHOLDER}

{\color{primary} BODY_PLACEHOLDER}

\vspace{1.2em}

% ---------------------------------------------------------------------------
% Closing & Sign-off
% ---------------------------------------------------------------------------
{\color{primary} SIGNOFF_PLACEHOLDER}\\[2.5em]

{\bfseries\color{primary} SENDER_NAME_PLACEHOLDER}

\end{document}
"""


# ============================================================
# Per-language conventions
# ============================================================

_CONVENTIONS = {
    "en": {
        "salutation_default": "Dear Hiring Manager,",
        "salutation_named": "Dear {name},",
        "signoff": "Sincerely,",
        "subject_prefix": "Application for",
        "city_date": "{city}, {date}",
    },
    "de": {
        "salutation_default": "Sehr geehrte Damen und Herren,",
        "salutation_named_male": "Sehr geehrter Herr {name},",
        "salutation_named_female": "Sehr geehrte Frau {name},",
        "signoff": "Mit freundlichen Grüßen",
        "subject_prefix": "Bewerbung als",
        "city_date": "{city}, den {date}",
    },
}


def _format_date(lang: str, city: str = "") -> str:
    now = datetime.now()

    months_en = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    months_de = (
        "Januar",
        "Februar",
        "März",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    )

    months = months_de if lang == "de" else months_en

    if lang == "de":
        date_str = f"{now.day}. {months[now.month - 1]} {now.year}"
    else:
        date_str = f"{now.day} {months[now.month - 1]} {now.year}"

    if city:
        return _CONVENTIONS[lang]["city_date"].format(city=city, date=date_str)

    return date_str


def _build_salutation(lang: str, hiring_manager: str) -> str:
    conv = _CONVENTIONS[lang]
    name = (hiring_manager or "").strip()

    if not name or name.lower() in {
        "hiring manager",
        "hiring_manager",
        "recruiter",
        "damen und herren",
    }:
        return conv["salutation_default"]

    if lang == "de":
        first_name = name.split()[0] if name.split() else ""
        female_endings = ("a", "e", "i", "ie", "ina", "ine", "ika")
        is_female = first_name.lower().endswith(female_endings) or first_name.lower() in {
            "anna",
            "julia",
            "sarah",
            "marie",
            "lea",
        }
        template = conv["salutation_named_female"] if is_female else conv["salutation_named_male"]
        return template.format(name=name)

    return conv["salutation_named"].format(name=name)


def _build_subject_line(lang: str, role: str, company: str) -> str:
    role = role.replace("$", "").strip()
    prefix = _CONVENTIONS[lang]["subject_prefix"]
    if not company:
        return f"{prefix} {role}"
    if lang == "de":
        return f"{prefix} {role} bei {company}"
    return f"{prefix} {role} position at {company}"


def _extract_role_from_job_description(job_description: str) -> str:
    """Extract an explicitly labelled role without inventing a default title."""
    text = (job_description or "")[:20_000]
    patterns = (
        r"(?:job title|position|role)\s*[:\-]\s*([^\n]{2,200})",
        r"(?:stellenbezeichnung|position|rolle)\s*[:\-]\s*([^\n]{2,200})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .;:")
    return ""


def _build_recipient_block(company: str, department: str = "") -> str:
    lines = [company]
    if department:
        lines.append(department)

    return r" \\ ".join(_escape_cover_letter_text(line) for line in lines if line)


# ============================================================
# LLM Prompts
# ============================================================

_COVER_LETTER_PROMPT_EN = """You are an experienced professional writing a direct, natural cover letter to a hiring team.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

PROVIDED COMPANY NAME:
{company_name}

TEMPLATE STYLE:
{template_style}

TONE:
{tone}

Write the cover letter in ENGLISH.

Return ONLY a valid JSON object with exactly these keys:

{{
  "company_name": "<Extract the EXACT hiring company name from the Job Description.>",
  "target_role": "<exact job title extracted from the JD without math symbols>",
  "hiring_manager_name": "<specific name if mentioned, otherwise 'Hiring Manager'>",
  "department": "<department or team if mentioned, otherwise empty>",
  "opening": "<2 short, punchy sentences>",
  "body_paragraph_1": "<3–4 conversational sentences with varied lengths>",
  "body_paragraph_2": "<3–4 conversational sentences with varied lengths>",
  "closing": "<2 short sentences>"
}}

HUMAN DETECTOR-BYPASS RULES (STRICT 100% HUMAN SCORE):
1. VARY SENTENCE LENGTH EXTREMELY: Put a short 3- to 5-word sentence immediately after a long sentence.
2. ABSOLUTELY BANNED AI WORDS: furthermore, additionally, moreover, seamlessly, testament, aligns, pivotal, passionate, dynamic, cutting-edge, leverage, robust, spearheaded, key role, exciting opportunity, target company, my passion, thrilled, perfect fit.
3. NO GENERIC OPENINGS: Never write "I am writing to apply", "I was excited to see", or "I believe I am an ideal fit".
4. DO NOT USE MATH SYMBOLS LIKE $ OR ESCAPED MATH MARKERS LIKE $(m/w/d)$.
5. START SENTENCES WITH CONTEXT: Begin sentences with triggers like "During our cloud migration...", "To cut costs...", "When working on...". Never start consecutive sentences with "I".
6. Total word count: 250–320 words max.
"""

_COVER_LETTER_PROMPT_DE = """Du bist eine erfahrene Fachkraft und schreibst ein direktes, natürliches Anschreiben an ein Recruiting-Team.

LEBENSLAUF:
{resume_text}

STELLENBESCHREIBUNG:
{job_description}

ANGEGEBENER UNTERNEHMENSNAME:
{company_name}

STIL:
{template_style}

TON:
{tone}

Schreibe das Anschreiben auf DEUTSCH.

Antworte NUR mit einem gültigen JSON-Objekt mit genau diesen Keys:

{{
  "company_name": "<Exakter Unternehmensname aus der Stellenbeschreibung.>",
  "target_role": "<exakter Jobtitel aus der Stellenbeschreibung ohne Mathesymbole>",
  "hiring_manager_name": "<Name, falls genannt, sonst 'Damen und Herren'>",
  "department": "<Abteilung oder Team, sonst leer>",
  "opening": "<2 kurze, direkte Sätze>",
  "body_paragraph_1": "<3–4 natürliche Sätze mit unterschiedlichen Längen>",
  "body_paragraph_2": "<3–4 natürliche Sätze mit unterschiedlichen Längen>",
  "closing": "<2 kurze Sätze>"
}}

REGELN FÜR 100% MENSCHLICHEN TEXT (AI-DETEKTOR PASS):
1. EXTREME SATZLÄNGEN-VARIATION: Setze bewusst einen sehr kurzen Satz (3–5 Wörter) direkt nach einen langen Erklärungssatz.
2. STRENG VERBOTENE KI-WÖRTER: "mit großer Freude", "hiermit bewerbe ich mich", "ich bin davon überzeugt", "nahtlos", "unter Beweis stellen", "ideale Besetzung", "wertvoller Beitrag", "innovativ", "dynamisch", "Target Company", "meine Leidenschaft", "begeistert".
3. KEINE KI-FLOSKELN IM EINSTIEG: Keine Standardfloskeln. Starte direkt mit der praktischen Arbeit.
4. ABSOLUT KEINE MATHESYMBOLE WIE $ ODER VERSTECKTE FORMELN WIE $(m/w/d)$. Schreibe (m/w/d) ganz normal.
5. KONTEXTUELLER SATZANFANG: Starte Sätze mit Kontext (z.B. "Beim Ausbau der AWS-Infrastruktur...", "Um Ausfallzeiten zu senken..."). Starte nicht jeden Satz mit "Ich".
6. Gesamtlänge: ca. 250–320 Wörter.
"""


# ============================================================
# Helpers
# ============================================================


def _escape_cover_letter_text(value: object) -> str:
    """Bound and escape plain text for insertion into the trusted template."""
    text = "" if value is None else str(value)
    text = text[:10_000]
    text = "".join(
        char for char in text if char in "\t\n\r" or not re.match(r"[\x00-\x1f\x7f]", char)
    )
    text = text.replace("$", "")

    text = re.sub(
        r"\b([a-zA-ZäöüßÄÖÜ])\s+([a-zA-ZäöüßÄÖÜ]{1,2})\s+([a-zA-ZäöüßÄÖÜ]{1,2})\b",
        r"\1 \2 \3",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    escaped = _escape_latex_text(text)
    return escaped.replace("®", r"\textregistered{}").replace("™", r"\texttrademark{}")


def _normalize_llm_field(value: object, max_chars: int = 2500) -> str:
    if value is None:
        return ""

    value = str(value).strip()
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value[:max_chars].strip()


def _normalize_llm_line(value: object, max_chars: int) -> str:
    return " ".join(_normalize_llm_field(value, max_chars).split())


def _humanize_generated_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"```(?:text|latex|markdown)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _clean_json_response(raw: str) -> str:
    if not isinstance(raw, str):
        raise RuntimeError("Cover letter model response must be text.")
    if len(raw) > MAX_COVER_MODEL_RESPONSE_CHARS:
        raise RuntimeError("Cover letter model response exceeded the size limit.")
    text = raw.strip()

    # Remove bounded reasoning blocks from local models.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Extract content inside markdown code blocks if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part_cleaned = part.strip()
            if part_cleaned.startswith("json"):
                part_cleaned = part_cleaned[4:].strip()
            if part_cleaned.startswith("{") and part_cleaned.endswith("}"):
                return part_cleaned
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    return text.strip()


def _build_contact_line(header: dict[str, str]) -> str:
    parts = []

    if header.get("email"):
        email_url = latex_escape_url(f"mailto:{header['email']}")
        email_text = _escape_latex_text(header["email"])
        parts.append(rf"\href{{\detokenize{{{email_url}}}}}{{{email_text}}}")

    if header.get("phone"):
        parts.append(_escape_latex_text(header["phone"]))

    if header.get("linkedin"):
        linkedin = latex_escape_url(header["linkedin"])
        parts.append(rf"\href{{\detokenize{{{linkedin}}}}}{{LinkedIn}}")

    if header.get("github"):
        github = latex_escape_url(header["github"])
        parts.append(rf"\href{{\detokenize{{{github}}}}}{{GitHub}}")

    return r" \quad$\cdot$\quad ".join(parts)


def _guess_city(resume_text: str) -> str:
    for line in (resume_text or "").splitlines()[:10]:
        line = line.strip()
        match = re.match(
            r"^([A-ZÄÖÜ][a-zäöüß\-]+),\s*" r"(Germany|Deutschland|Austria|Switzerland)",
            line,
        )
        if match:
            return match.group(1)
    return ""


# ============================================================
# Public API
# ============================================================


def generate_cover_letter_latex(
    resume_text: str,
    job_description: str,
    company_name: str = "",
    template_style: str = "classic_professional",
    tone: str = "formal",
    provider: str | None = None,
    model_name: str | None = None,
    route_mode: str | None = None,
    language: str = "auto",
) -> tuple[str, str]:
    from app.services.llm.provider import LLMService

    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Candidate resume text is empty.")
    if len(resume_text) > MAX_COVER_RESUME_CHARS:
        raise ValueError("Candidate resume text exceeds the size limit.")
    if not isinstance(job_description, str):
        raise ValueError("Job description must be a string.")
    if len(job_description) > MAX_COVER_JOB_DESCRIPTION_CHARS:
        raise ValueError("Job description exceeds the size limit.")
    if not isinstance(company_name, str):
        raise ValueError("Company name must be a string.")
    company_name = _normalize_llm_line(company_name, 160)
    template_style = str(template_style or "professional")[:100]
    allowed_tones = {"formal", "concise", "warm", "technical"}
    tone = str(tone or "formal").strip().lower()
    if tone not in allowed_tones:
        raise ValueError("Unsupported cover-letter tone")

    if language == "auto":
        language = detect_language(job_description, default="en")
    language = "de" if language == "de" else "en"

    header = extract_candidate_header(resume_text)

    prompt_template = _COVER_LETTER_PROMPT_DE if language == "de" else _COVER_LETTER_PROMPT_EN

    prompt = prompt_template.format(
        resume_text=(resume_text or "")[:6000],
        job_description=(job_description or "")[:4000],
        company_name=company_name or "(Extract automatically from job description)",
        template_style=template_style,
        tone=tone,
    )

    raw = LLMService.generate(
        prompt=prompt,
        provider=provider,
        model_name=model_name,
        route_mode=route_mode,
    )

    try:
        data = json.loads(_clean_json_response(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Cover letter model response was not valid JSON.") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Cover letter response must be a JSON object.")

    body_text = " ".join(str(value) for value in data.values())
    source_text = f"{resume_text}\n{job_description}"
    numeric_claims = re.findall(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|years?|months?)\b", body_text, flags=re.IGNORECASE
    )
    unsupported_claims = [
        claim for claim in numeric_claims if claim.casefold() not in source_text.casefold()
    ]
    if unsupported_claims:
        raise RuntimeError("Cover letter contains unsupported numeric claims.")

    regex_company = _normalize_llm_line(_extract_company_from_jd(job_description), 160)
    llm_company = _normalize_llm_line(data.get("company_name"), 160)

    generic_company_names = {
        "the company",
        "company",
        "employer",
        "target company",
        "das unternehmen",
        "unternehmen",
        "arbeitgeber",
    }
    if llm_company.casefold() in generic_company_names:
        llm_company = ""
    if company_name.strip().casefold() in generic_company_names:
        company_name = ""

    final_company_name = company_name.strip() or regex_company or llm_company

    hiring_manager = _normalize_llm_line(data.get("hiring_manager_name"), 160)
    department = _normalize_llm_line(data.get("department"), 160)
    target_role = _normalize_llm_line(data.get("target_role"), 200)
    target_role = target_role.replace("$", "").strip()
    target_role = _normalize_llm_line(
        target_role or _extract_role_from_job_description(job_description),
        200,
    )
    if not target_role:
        raise RuntimeError("Could not determine the target role from the job description.")

    hiring_manager = hiring_manager or (
        "Damen und Herren" if language == "de" else "Hiring Manager"
    )

    body_paragraphs = [
        _humanize_generated_text(_normalize_llm_field(data.get("opening"))),
        _humanize_generated_text(_normalize_llm_field(data.get("body_paragraph_1"))),
        _humanize_generated_text(_normalize_llm_field(data.get("body_paragraph_2"))),
        _humanize_generated_text(_normalize_llm_field(data.get("closing"))),
    ]

    body_text = "\n\n".join(_escape_cover_letter_text(p) for p in body_paragraphs if p)

    if not body_text.strip():
        raise RuntimeError("Cover letter body is empty.")

    salutation = _build_salutation(language, hiring_manager)
    signoff = _CONVENTIONS[language]["signoff"]
    city = _guess_city(resume_text)
    city_date = _format_date(language, city=city)
    subject_line = _build_subject_line(language, target_role, final_company_name)
    recipient_block = _build_recipient_block(final_company_name, department)
    sender_name = _escape_latex_text(header["name"])
    sender_contact = _build_contact_line(header)

    latex = COVER_LETTER_LATEX_TEMPLATE
    latex = latex.replace("SENDER_NAME_PLACEHOLDER", sender_name)
    latex = latex.replace("SENDER_CONTACT_PLACEHOLDER", sender_contact)
    latex = latex.replace("SENDER_CITY_DATE_PLACEHOLDER", _escape_cover_letter_text(city_date))
    latex = latex.replace("RECIPIENT_BLOCK_PLACEHOLDER", recipient_block)
    latex = latex.replace("SUBJECT_LINE_PLACEHOLDER", _escape_cover_letter_text(subject_line))
    latex = latex.replace("SALUTATION_PLACEHOLDER", _escape_cover_letter_text(salutation))
    latex = latex.replace("SIGNOFF_PLACEHOLDER", _escape_cover_letter_text(signoff))
    latex = latex.replace("BODY_PLACEHOLDER", body_text)
    if "PLACEHOLDER" in latex:
        raise RuntimeError("Cover letter contains an unresolved template value.")
    validate_latex_document(latex)

    return latex, language


def compile_cover_letter_pdf(latex_code: str) -> bytes:
    """Compile a validated cover letter through the shared bounded compiler."""
    if not isinstance(latex_code, str):
        raise LaTeXSourceError("LaTeX source must be a string.")
    validate_latex_document(latex_code)
    return compile_single_page_pdf(latex_code)
