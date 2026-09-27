import logging
import re
import unicodedata

from app.services.cv.latex_escape import (
    balance_latex_environments,
    escape_latex_body,
    escape_latex_text,
    is_fully_escaped,
    repair_latex_lists,
    unescaped_specials,
)
from app.services.cv.optimizer import (
    _clean_skill_list,
    _format_actionable_suggestions,
    _sandwich,
    _truncate,
    auto_select_layout,
    normalize_resume_language,
    required_language_for_layout,
)
from app.services.cv.pdf_compiler import (
    LaTeXSourceError,
    compile_single_page_pdf,
    sanitize_latex_url,
    validate_latex_body,
    validate_latex_document,
)
from app.services.cv.pdf_validation import PDFValidationError, validate_pdf_content
from app.services.llm.provider import LLMService

logger = logging.getLogger(__name__)

GERMAN_MINIMAL_ATS = "german_minimal_ats"
MAX_RESUME_TEXT_CHARS = 200_000
MAX_JOB_DESCRIPTION_CHARS = 200_000
MAX_LLM_LATEX_CHARS = 250_000
MAX_SKILL_ITEMS = 200

#: Prompt budgets for the generation call, matching the optimizer's values for
#: the same inputs. Every other generation path already applied these; this one
#: did not, so a long posting went to the model whole and was silently
#: truncated in the middle by anything with a smaller context window.
MAX_RESUME_CHARS = 25_000
MAX_JD_CHARS = 12_000


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
    # Was missing here while present in optimizer's copy, so this layout got
    # the "detect the language from the job description" rule instead of the
    # explicit English one. See _assert_layout_language_sets below.
    "technical_lead",
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
    "executive",
    "officer",
    "principal",
    "owner",
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
    """Extract a short professional title from explicit resume text."""
    if not resume_text:
        return ""

    for raw_line in resume_text.splitlines()[:30]:
        line = raw_line.strip()
        if not line or "@" in line or "http" in line.lower():
            continue

        # Experience rows commonly use ``Role | Company | Dates``.  Only the
        # role portion belongs in the CV header.
        role = re.split(r"\s*[|–—]\s*", line, maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'\-&/]{2,100}", role):
            if any(keyword in role.casefold() for keyword in _TITLE_KEYWORDS):
                return role[:120]
    return ""


# ============================================================
# LaTeX link helper
# ============================================================


def latex_escape_url(url: str) -> str:
    r"""Return a validated URL suitable for ``\detokenize``."""
    if not url:
        return ""

    value = str(url).strip()
    match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", value)
    if match:
        value = match.group(2).strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    value = value.strip("`").strip("'").strip('"')
    return sanitize_latex_url(value)


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
    """
    Escape plain text for the trusted header fields.

    Delegates to :mod:`app.services.cv.latex_escape` so the candidate name,
    the job title and the CV body are escaped by the same rules. Keeping a
    second implementation here is how the header and the body drifted apart
    in the first place.
    """
    return escape_latex_text(value)


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
    r"(?:present|current|heute|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|januar|februar|märz|maerz|april|mai|juni|juli|august|"
    r"september|oktober|november|dezember)?\s*\d{4})\b",
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
        # Split on pipe first so that en/em dashes inside date ranges
        # (e.g. "Jan 2020 – Feb 2024") are not treated as field separators.
        if "|" in line:
            parts = [part.strip() for part in line.split("|") if part.strip()]
        else:
            parts = [part.strip() for part in re.split(r"\s*(?:—|–)\s*", line) if part.strip()]
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
    """
    Extract the candidate's name and contact details from the parsed
    resume text.

    Rejects known PDF/JSON binary tokens that appear when the input
    was accidentally decoded from raw bytes (e.g. "obj", "endobj",
    "stream"). Without this guard, a mis-fed pipeline can return
    "obj" as the candidate's name.
    """
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]

    # Tokens that appear in decoded PDF binary, JSON envelopes, and
    # other non-resume sources. If the name regex matches one of
    # these, it is not a real name.
    _GARBAGE_TOKENS = {
        "obj",
        "endobj",
        "stream",
        "endstream",
        "xref",
        "trailer",
        "startxref",
        "catalog",
        "pages",
        "page",
        "font",
        "encoding",
        "producer",
        "creator",
        "mediabox",
        "parent",
        "kids",
        "count",
        "type",
        "length",
        "filter",
        "flatedecode",
        "resources",
        "contents",
        "annot",
        "acroform",
        "outlines",
        "names",
        "dest",
        "action",
        "uri",
        "base",
        "info",
        "title",
        "author",
        "subject",
    }

    def _is_plausible_name(candidate: str) -> bool:
        low = candidate.casefold().strip()

        # Exact-match rejections
        if low in _GARBAGE_TOKENS:
            return False

        # Names never contain digits
        if any(ch.isdigit() for ch in candidate):
            return False

        # Single-word candidates need at least 4 chars and a vowel.
        # Real one-word names are rare in a CV header; this filters
        # PDF noise like "trailer", "stream", "encoding".
        if " " not in candidate:
            if len(candidate) < 4:
                return False
            if not any(v in low for v in "aeiouäöüàèéìòù"):
                return False

        return True

    name = next(
        (
            line
            for line in lines[:5]
            if re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,80}", line)
            and _is_plausible_name(line)
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
    if header.get("email"):
        email_url = latex_escape_url(f"mailto:{header['email']}")
        email_text = _escape_latex_text(header["email"])
        parts.append(rf"\hrlink{{\detokenize{{{email_url}}}}}{{{email_text}}}")
    if header.get("phone"):
        parts.append(_escape_latex_text(header["phone"]))
    if header.get("linkedin"):
        linkedin = latex_escape_url(header["linkedin"])
        parts.append(rf"\hrlink{{\detokenize{{{linkedin}}}}}{{LinkedIn}}")
    if header.get("github"):
        github = latex_escape_url(header["github"])
        parts.append(rf"\hrlink{{\detokenize{{{github}}}}}{{GitHub}}")
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
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    route_mode: str | None = None,
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
        label = _escape_latex_text(match.group(3).strip() or match.group(1).strip() or "Link")
        url = latex_escape_url(markdown_url)
        return rf"\hrlink{{{url}}}{{{label}}}"

    text = malformed_hrlink.sub(repair_hrlink, text)

    markdown_link = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")

    def convert_markdown_link(match: re.Match) -> str:
        label = _escape_latex_text(match.group(1).strip())
        url = latex_escape_url(match.group(2))
        return rf"\hrlink{{{url}}}{{{label}}}"

    text = markdown_link.sub(convert_markdown_link, text)

    return text


# FIX: regexes that strip an LLM-produced full LaTeX document down to body-only,
# so it can be safely injected into a template that already has its own preamble.
_PREAMBLE_STRIP_RE = re.compile(
    r"^\s*\\documentclass.*?\\begin\{document\}",
    re.DOTALL,
)
_END_DOC_RE = re.compile(r"\\end\{document\}\s*$")


def clean_llm_response_to_latex(text: str) -> str:
    """Normalize a model response and discard any model-supplied preamble."""
    if not isinstance(text, str) or not text.strip():
        return ""
    if len(text) > MAX_LLM_LATEX_CHARS:
        raise LaTeXSourceError("The model returned oversized LaTeX content.")

    text = strip_code_fences(text)
    if text.casefold().startswith(("i'm ready", "sure,", "please provide")):
        # A conversational refusal is not CV content.  Generation fails closed
        # instead of inserting invented placeholder prose.
        return ""

    # A model may return a complete document.  Only its body is retained; the
    # trusted application template supplies the preamble and shell policy.
    if r"\documentclass" in text or r"\begin{document}" in text:
        stripped = _PREAMBLE_STRIP_RE.sub("", text, count=1)
        if stripped == text and r"\begin{document}" in text:
            stripped = text.split(r"\begin{document}", 1)[1]
        text = stripped
        text = _END_DOC_RE.sub("", text).strip()

    return text


def strip_code_fences(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^\s*```(?:latex|tex)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def clean_body_for_latex(body_text: str) -> str:
    """Sanitize generated body content without breaking trusted URL macros."""
    if not isinstance(body_text, str) or len(body_text) > MAX_LLM_LATEX_CHARS:
        raise LaTeXSourceError("LaTeX body must be a bounded string.")
    body_text = normalize_latex_links(body_text)

    protected_urls: list[tuple[str, str]] = []

    def protect_url(match: re.Match, group: int) -> str:
        url = latex_escape_url(match.group(group))
        token = f"LATEXSAFEURLTOKEN{len(protected_urls)}ENDTOKEN"
        protected_urls.append((token, url))
        return match.group(0).replace(match.group(group), token, 1)

    body_text = re.sub(
        r"\\hrlink\s*\{([^{}]+)\}",
        lambda match: protect_url(match, 1),
        body_text,
    )
    body_text = re.sub(
        r"(\\projheader\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{)([^{}]+)(\})",
        lambda match: protect_url(match, 2),
        body_text,
    )

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

    # The single escaping layer.
    #
    # The previous implementation escaped exactly three characters
    # (`&`, `%`, `_`) with three independent regular expressions and handled
    # `<`/`>` with a fourth. Everything else -- `$`, `#`, `^`, `\` and stray
    # braces -- reached pdflatex raw, and each one is fatal:
    #
    #   &   "Misplaced alignment tab character &."
    #   #   "You can't use `macro parameter character #' in horizontal mode."
    #   ^   "Missing $ inserted."
    #   $   "Missing $ inserted."           (an odd count opens math mode)
    #   \   "Undefined control sequence."    (C:\Users -> \Users)
    #   { } "Missing } inserted."
    #
    # escape_latex_body() is structure aware, so `\section*{Profil}`,
    # `\item`, and `\jobheader{role}{company}{dates}` survive intact while the
    # text between them is escaped, and it is idempotent, so a model that
    # already wrote `\&` does not end up with `\\&`.
    #
    # The URL placeholders installed above are left in place while escaping.
    # They are alphanumeric, so escaping is a no-op on them, whereas restoring
    # first would re-introduce the `&` and `_` that a query string legitimately
    # contains -- which is exactly what `\detokenize` exists to protect.
    body_text = escape_latex_body(body_text)

    # Repair unbalanced environments. \begin{itemize} without its \end is
    # "LaTeX Error: \begin{itemize} on input line 21 ended by \end{document}",
    # which is fatal and cannot be fixed by escaping. Closing the missing
    # environment adds no content and changes no wording.
    body_text = balance_latex_environments(body_text)

    # Wrap orphaned \item runs. A model that writes \jobheader{...} followed by
    # a list of \item bullets, but omits \begin{itemize}, produces "LaTeX Error:
    # Lonely \item--perhaps a missing list environment." That is fatal and is
    # not a character problem, so it needs a structural repair rather than
    # escaping. Small local models hit this often.
    body_text = repair_latex_lists(body_text)

    body_text = body_text.strip()
    if not body_text:
        raise LaTeXSourceError("The model did not return CV body content.")

    # Pre-flight. Because escape_latex_body is idempotent, `escaped == original`
    # is exactly the condition "no unescaped special remains", so this check can
    # never drift away from the code that fixes it. It runs before pdflatex so a
    # fatal character is reported with a line number instead of an exit code.
    #
    # It runs before the URLs are restored, because a query string legitimately
    # contains `&` and `_`; those are validated separately by
    # sanitize_latex_url() inside validate_latex_body() below.
    if not is_fully_escaped(body_text):
        problems = unescaped_specials(body_text)
        first_line, first_char = problems[0] if problems else (0, "?")
        raise LaTeXSourceError(
            "Generated CV body still contains unescaped LaTeX characters "
            f"(first at line {first_line}: {first_char!r})."
        )

    for token, url in protected_urls:
        body_text = body_text.replace(token, url)

    return validate_latex_body(body_text)


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
\definecolor{subgray}{HTML}{475569}
\definecolor{accent}{HTML}{D97706}
\definecolor{linkcolor}{HTML}{D97706}

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
    {\Huge\bfseries\color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
    {\Large\bfseries\color{primary} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
    {\normalsize\color{subgray} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

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
    {\LARGE\bfseries CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
    {\large CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
    {\normalsize CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

RESUME_BODY_PLACEHOLDER

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
    {\huge\bfseries CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
    {\large CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
    {\normalsize CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

\vspace{1pt}
\hrule height 0.5pt
\vspace{2pt}

RESUME_BODY_PLACEHOLDER

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
    {\Huge\bfseries\color{darkgray} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
    {\large\bfseries\color{primary} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
    {\normalsize\color{secondary} CANDIDATE_CONTACT_PLACEHOLDER}
\end{center}

\vspace{2pt}

RESUME_BODY_PLACEHOLDER

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
  {\Large \bfseries \color{primary} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
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

    {\LARGE\bfseries\color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]

    {\large\color{secondary} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]

    {\normalsize\color{secondary} CANDIDATE_CONTACT_PLACEHOLDER}

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
    {\large\bfseries\color{primary}}
    {}{0em}{\MakeUppercase}
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

    {\Huge\bfseries\color{primary} CANDIDATE_NAME_PLACEHOLDER}\\[2pt]

    {\large\bfseries\color{goldaccent} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]

    {\normalsize\color{secondary} CANDIDATE_CONTACT_PLACEHOLDER}

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

% Pure black throughout. An ATS-first layout has to survive being parsed and
% printed in monochrome, so it carries no colour emphasis at all: no accent
% colour, no coloured rule, no coloured link. That is what actually
% distinguishes it from the other ATS layouts, which keep a brand colour.
\definecolor{primary}{HTML}{000000}
\definecolor{subgray}{HTML}{333333}
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
{\large\bfseries\color{primary} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
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
  {\Huge\bfseries CANDIDATE_NAME_PLACEHOLDER}\\[2pt]
  {\large \bfseries CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
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
  {\large \bfseries \color{accent} CANDIDATE_TITLE_PLACEHOLDER}\\[3pt]
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

    # Word-boundary matching: `\small` in the code but NOT `\smallskip`,
    # `\smallsetminus`, etc. Same for the other sub-11pt size commands.
    _FONT_SIZE_RE = re.compile(r"\\(?:small|footnotesize|scriptsize|tiny)(?!\w)")
    if _FONT_SIZE_RE.search(latex_code):
        match = _FONT_SIZE_RE.search(latex_code)
        raise ValueError(
            f"Layout '{layout_style}' contains a font size below 11pt: {match.group(0)}"
        )


# ============================================================
# Generate LaTeX CV
# ============================================================


def generate_german_latex_content(
    resume_text: str,
    job_description: str,
    missing_skills: list[str],
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    route_mode: str | None = None,
    layout_style: str = "auto",
    primary_color_hex: str | None = None,
    secondary_color_hex: str | None = None,
    linkedin_url: str | None = None,
    github_url: str | None = None,
    improvement_suggestions: list[str] | None = None,
    variant_cfg: dict | None = None,
) -> str:
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise FactualValidationError(["Candidate resume text is empty."])
    if len(resume_text) > MAX_RESUME_TEXT_CHARS:
        raise ValueError("Candidate resume text exceeds the size limit.")
    if not isinstance(job_description, str):
        raise ValueError("Job description must be a string.")
    if len(job_description) > MAX_JOB_DESCRIPTION_CHARS:
        raise ValueError("Job description exceeds the size limit.")
    if not isinstance(missing_skills, list) or len(missing_skills) > MAX_SKILL_ITEMS:
        raise ValueError("Missing skills must be a bounded list.")
    if not all(isinstance(skill, str) for skill in missing_skills):
        raise ValueError("Missing skills must contain only strings.")
    if variant_cfg is not None and not isinstance(variant_cfg, dict):
        raise ValueError("CV variant configuration must be an object.")
    if improvement_suggestions is not None and (
        not isinstance(improvement_suggestions, list)
        or len(improvement_suggestions) > MAX_SKILL_ITEMS
        or not all(isinstance(item, str) for item in improvement_suggestions)
    ):
        raise ValueError("Improvement suggestions must be a bounded list of strings.")
    missing_skills = list(missing_skills)

    if (layout_style or "").strip().lower() in ("auto", "auto_detect", ""):
        layout_style = auto_select_layout(job_description, resume_text)
    if layout_style not in CV_TEMPLATES:
        raise ValueError("Unsupported CV layout style.")

    is_german_minimal_ats = layout_style == GERMAN_MINIMAL_ATS
    invariants = extract_resume_invariants(resume_text) if is_german_minimal_ats else []
    candidate_header = extract_candidate_header(resume_text)

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

    # Infer the title before normalization can rewrite resume wording.
    effective_title = _infer_title(resume_text)
    if variant_cfg:
        title_override = variant_cfg.get("cv_title_override")
        if title_override:
            if not isinstance(title_override, str):
                raise ValueError("CV title override must be a string.")
            effective_title = re.sub(r"\s+", " ", title_override).strip()

    if not effective_title:
        raise FactualValidationError(
            ["Could not extract a professional title from the candidate resume."]
        )
    if len(effective_title) > 120 or "\n" in effective_title or "\r" in effective_title:
        raise ValueError("Candidate title is invalid.")

    resume_text = normalize_resume_language(
        resume_text,
        target_lang,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        route_mode=route_mode,
    )
    if not isinstance(resume_text, str) or len(resume_text) > MAX_RESUME_TEXT_CHARS:
        raise FactualValidationError(["Normalized resume text is invalid or oversized."])

    # Variant skills may prioritize existing JD terms but never invent content.
    if variant_cfg:
        boost_skills = variant_cfg.get("top_skills") or []
        if (
            not isinstance(boost_skills, list)
            or len(boost_skills) > MAX_SKILL_ITEMS
            or not all(isinstance(skill, str) for skill in boost_skills)
        ):
            raise ValueError("Variant top skills must be a bounded list of strings.")
        if boost_skills:
            resume_lower = resume_text.casefold()
            supported_boost = [skill for skill in boost_skills if skill.casefold() in resume_lower]
            boost_keys = {skill.casefold() for skill in supported_boost}
            missing_skills = supported_boost + [
                skill for skill in missing_skills if skill.casefold() not in boost_keys
            ]
    clean_skills = _clean_skill_list(missing_skills)
    skills_text = ", ".join(clean_skills) if clean_skills else "(No additional skills requested.)"
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

    # Budgeted and fenced, exactly like every optimizer generation path.
    # Without the fence the untrusted resume and posting sit next to the
    # structural rules with nothing to mark them as data, and a CV containing
    # "ignore previous instructions" would be followed.
    safe_resume = _sandwich(
        "RESUME",
        _truncate(resume_text, MAX_RESUME_CHARS, field="resume"),
    )
    safe_job_description = _sandwich(
        "JOB_DESCRIPTION",
        _truncate(
            job_description,
            MAX_JD_CHARS,
            field="job_description",
        ),
    )

    prompt = rf"""
Optimize and enrich the candidate's CV body content to achieve the highest possible match against the target job description.

{language_rule}

Target Job Description:
{safe_job_description if job_description.strip() else "(No job description was provided.)"}

Critical Skills to Integrate Naturally:
{skills_text}

Only mention a skill, employer, date, credential, or metric when it is
supported by the candidate resume/profile. Do not turn an ATS gap or a target
skill into a claim of experience; leave unsupported items for manual review.

ACTIONABLE IMPROVEMENTS FROM THE CANDIDATE'S PRIOR ATS AUDIT:
{suggestions_text}

Original Resume Text:
{safe_resume}

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
   - If a block above is marked as truncated, write only what is present. Never
     infer, reconstruct or invent a section that was cut.

3. REPOSITORY LINKS RULE:
   - Use only repository URLs explicitly present in the original resume.
   - Never invent, alter, or infer a project URL.

4. WORK EXPERIENCE, EDUCATION & PROJECTS:
   - Every bullet point inside \begin{{itemize}} MUST start strictly with \item.
   - Use \jobheader{{Role / Degree}}{{Company / University}}{{Dates}} for BOTH jobs and education.
   - Use \projheader{{Project Name}}{{Tech Stack}}{{Original Repository URL}} for projects.
   - Describe only the candidate's documented professional positioning.
   - Never invent employers, metrics, dates, credentials, or project links.

5. SKILLS SECTION:
   - Use concise, resume-supported skill categories.
   - Format category labels in bold, for example \textbf{{Category name:}}.
   - Do not add a skill merely because it appears as a target-job keyword.

6. SECTION ORDERING:
{section_names}

7. Escape special characters inside normal text:
   \& for &
   \% for %
   \_ for _
   \textless{{}} for <
   \textgreater{{}} for >

8. Return RAW LaTeX body content ONLY (No code fences, markdown, or conversational text).

9. FIX: DO NOT EMIT \documentclass, \usepackage, \begin{{document}}, or \end{{document}}.
   The wrapper already adds them. Any preamble in your output will break compilation.
"""

    try:
        raw_latex = LLMService.generate(
            prompt=prompt,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            route_mode=route_mode,
            task="full_cv_generation",
        )
        if not isinstance(raw_latex, str) or len(raw_latex) > MAX_LLM_LATEX_CHARS:
            raise LaTeXSourceError("The model returned invalid or oversized LaTeX content.")

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
        raise FactualValidationError(["CV generation failed; no CV was produced."]) from exc

    template = CV_TEMPLATES[layout_style]
    template = _patch_template_preamble(template)

    header = dict(candidate_header)
    if not header.get("linkedin") and linkedin_url:
        header["linkedin"] = latex_escape_url(linkedin_url)
    if not header.get("github") and github_url:
        header["github"] = latex_escape_url(github_url)

    template = template.replace(
        "CANDIDATE_NAME_PLACEHOLDER",
        _escape_latex_text(header["name"]),
    )
    template = template.replace(
        "CANDIDATE_TITLE_PLACEHOLDER",
        _escape_latex_text(effective_title),
    )
    template = template.replace(
        "CANDIDATE_CONTACT_PLACEHOLDER",
        _render_candidate_contact(header),
    )

    if primary_color_hex:
        color = primary_color_hex.removeprefix("#")
        if not re.fullmatch(r"[A-Fa-f0-9]{6}", color):
            raise ValueError("Primary color must be a six-digit hexadecimal value.")
        template = re.sub(
            r"\\definecolor\{primary\}\{HTML\}\{[A-Fa-f0-9]{6}\}",
            rf"\\definecolor{{primary}}{{HTML}}{{{color}}}",
            template,
        )

    if secondary_color_hex:
        color = secondary_color_hex.removeprefix("#")
        if not re.fullmatch(r"[A-Fa-f0-9]{6}", color):
            raise ValueError("Secondary color must be a six-digit hexadecimal value.")
        template = re.sub(
            r"\\definecolor\{secondary\}\{HTML\}\{[A-Fa-f0-9]{6}\}",
            rf"\\definecolor{{secondary}}{{HTML}}{{{color}}}",
            template,
        )

    latex_code = template.replace("RESUME_BODY_PLACEHOLDER", clean_body)
    if is_german_minimal_ats:
        validate_generated_invariants(latex_code, invariants)
    validate_cv_layout_template(layout_style, latex_code, template=template)
    validate_latex_document(latex_code)

    forbidden_legacy_terms = [
        r"(?i)" + "IT" + r"[ -]?" + "Support",
        r"(?i)" + "Support" + r"[ -]?" + "Engineer",
    ]
    if any(re.search(pattern, latex_code) for pattern in forbidden_legacy_terms):
        raise FactualValidationError(
            ["Generated CV still contains forbidden legacy support terminology."]
        )
    escaped_title = _escape_latex_text(effective_title)
    if escaped_title not in latex_code:
        raise FactualValidationError(
            ["Generated CV does not contain the title supplied by the candidate resume."]
        )
    if "PLACEHOLDER" in latex_code:
        raise FactualValidationError(["Generated CV contains an unresolved template value."])

    return latex_code


# ============================================================
# Compile LaTeX to PDF
# ============================================================


def _expected_content_from_latex(latex_code: str) -> str:
    """
    The readable text a LaTeX body implies, used to check the PDF kept it.

    Derived from the source rather than passed in, so a caller cannot
    accidentally validate a PDF against the wrong document. Only the body is
    read: the preamble is page furniture and its macro names are not content.
    """
    body = latex_code

    marker = "\\begin{document}"
    if marker in body:
        body = body.split(marker, 1)[1]

    for command in ("end{document}", "end{document}"):
        body = body.replace(command, " ")

    # Drop control sequences and the markup around them, then the braces.
    body = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", body)
    body = re.sub(r"[{}&\\]", " ", body)
    body = body.replace("~", " ").replace("\\%", " ").replace("%", " ")

    return re.sub(r"\s+", " ", body).strip()


def _sections_claimed_by_latex(latex_code: str) -> list[str]:
    """
    Section headings the source declared, as validator section names.

    An empty result is normal and correct: a CV is not required to have a
    projects section, and nothing is invented to satisfy the check.
    """
    from app.services.cv.pdf_validation import _SECTION_RE

    names: list[str] = []

    for heading in re.findall(
        r"\\(?:sub)*section\*?\{([^}]{1,80})\}",
        latex_code,
    ):
        cleaned = re.sub(
            r"\\[a-zA-Z@]+\{([^}]*)\}",
            r"\1",
            heading,
        ).strip()

        for name, pattern in _SECTION_RE.items():
            if pattern.search(cleaned) and name not in names:
                names.append(name)

    return names


def compile_latex_to_pdf(latex_code: str) -> bytes:
    """
    Compile a validated CV and verify the result actually contains it.

    A PDF that compiles and is one page is not yet a CV. The rendered text is
    read back and checked against the source body, so content that vanished
    between generation and typesetting is caught here rather than by the user.
    """
    import time

    from app.core.event_log import log_event
    from app.services.cv.pdf_compiler import PDFLayoutError

    if not isinstance(latex_code, str):
        raise LaTeXSourceError("LaTeX source must be a string.")
    validate_latex_document(latex_code)

    source = normalize_latex_links(latex_code)
    validate_latex_document(source)
    started = time.perf_counter()
    log_event("pdf", "pdf_compilation_started", latex_chars=len(source))

    try:
        pdf_bytes = compile_single_page_pdf(source)
    except PDFLayoutError as exc:
        log_event(
            "pdf",
            "pdf_compilation_failed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            error="page_limit_exceeded",
            pages=exc.pages,
        )
        raise
    except Exception as exc:
        # Event data must not contain compiler output or generated source.
        log_event(
            "pdf",
            "pdf_compilation_failed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            error=type(exc).__name__,
        )
        raise

    # Content validation, not just existence. Everything here is derived from
    # the source that was just compiled, so nothing external is consulted.
    expected_text = _expected_content_from_latex(latex_code)
    expected_sections = _sections_claimed_by_latex(latex_code)

    try:
        report = validate_pdf_content(
            pdf_bytes,
            expected_pages=1,
            expected_text=expected_text,
            expected_sections=expected_sections,
        )
    except PDFValidationError as exc:
        log_event(
            "pdf",
            "pdf_content_rejected",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            # The *category* only. The detail can name a candidate's content.
            error=str(exc) if "content" in str(exc).lower() else "validation_failed",
            problems=exc.report.get("problems", []),
        )
        raise

    log_event(
        "pdf",
        "pdf_compiled",
        pages=report.get("pages", 1),
        bytes=len(pdf_bytes),
        text_chars=report.get("text_chars", 0),
        sections_found=len(report.get("sections_found", [])),
        content_retention=report.get("content_retention"),
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return pdf_bytes


def _assert_layout_language_sets() -> None:
    """
    Fail at import if a layout is not classified as German or English.

    ``_GERMAN_LAYOUTS`` / ``_ENGLISH_LAYOUTS`` here and in optimizer.py are two
    copies of one policy. They had already drifted once, which made
    ``technical_lead`` silently receive the wrong output-language rule. There
    is no clean way to derive this set here, because ``CV_TEMPLATES`` is defined
    far below, so the check is made explicit at import instead. A new layout now
    has to be classified or the application refuses to start.
    """
    unclassified = set(CV_TEMPLATES) - _GERMAN_LAYOUTS - _ENGLISH_LAYOUTS
    overlap = _GERMAN_LAYOUTS & _ENGLISH_LAYOUTS

    if unclassified or overlap:
        raise RuntimeError(
            "CV layout language sets are inconsistent. "
            f"Missing a classification for: {sorted(unclassified)}. "
            f"Listed as both German and English: {sorted(overlap)}."
        )


_assert_layout_language_sets()
