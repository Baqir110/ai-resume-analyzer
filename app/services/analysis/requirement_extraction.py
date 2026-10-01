"""
Structured requirements a posting states, which are not vocabulary.

Why this is separate from keyword matching
-------------------------------------------
The taxonomy that drives keyword matching knows technology names. It does not know
that "a Bachelor's degree" is a requirement, that "AWS Certified Solutions
Architect" is a named certification, that "German at B2" is a language
requirement, or that "HIPAA" is industry terminology. All four appear in postings
as plain text and all four are things a candidate either has or does not.

Treated as plain keywords they are missed entirely; treated as hard requirements
they are given equal weight to a mention of Kubernetes. So they are extracted
here, with their requirement strength, and folded into the existing
``Required Skills`` and ``Keyword Match`` categories rather than becoming new ones
— the score stays in the five categories the rest of the application depends on.

A requirement is only ever reported as *absent*. Nothing in this module proposes
content, because a candidate's qualifications cannot be inferred from a posting.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEGREE_ORDER",
    "extract_language_requirements",
    "extract_education_requirements",
    "extract_certification_requirements",
    "extract_industry_terminology",
    "find_keyword_stuffing",
    "highest_degree_in_text",
]


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

#: A language named in a posting. Matched as a phrase because "German" and
#: "Dutch" also mean other things, and a bare language name in a requirements list
#: is what a recruiter means by a language requirement.
_LANGUAGES = (
    "English",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Dutch",
    "Portuguese",
    "Polish",
    "Swedish",
    "Norwegian",
    "Danish",
    "Finnish",
    "Czech",
    "Slovak",
    "Hungarian",
    "Romanian",
    "Bulgarian",
    "Greek",
    "Turkish",
    "Russian",
    "Ukrainian",
    "Arabic",
    "Hebrew",
    "Hindi",
    "Bengali",
    "Urdu",
    "Chinese",
    "Mandarin",
    "Cantonese",
    "Japanese",
    "Korean",
    "Vietnamese",
    "Thai",
    "Indonesian",
    "Malay",
    "Swahili",
    "Persian",
    "Farsi",
    "Catalan",
    "Basque",
    "Galician",
    "Estonian",
    "Latvian",
    "Lithuanian",
    "Slovenian",
)

_LANGUAGE_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(_LANGUAGES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

#: German states a language requirement as one compound noun —
#: "Deutschkenntnisse", "Englischkenntnisse" — so there is no word boundary after
#: "Deutsch" and the plain language name never matches. These are the compound
#: forms a German posting actually uses, mapped to the language they name.
_GERMAN_COMPOUND_LANGUAGES: dict[str, str] = {
    "deutschkenntnisse": "German",
    "deutschkenntnis": "German",
    "englischkenntnisse": "English",
    "englischkenntnis": "English",
    "französischkenntnisse": "French",
    "spanischkenntnisse": "Spanish",
    "italienischkenntnisse": "Italian",
}

_GERMAN_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(_GERMAN_COMPOUND_LANGUAGES) + r")\b", re.IGNORECASE
)

#: The words a CV may use to *claim* a language, per language.
#:
#: A posting names the language in English or as a German compound; a CV writes
#: its own language's word for it. "German language required" was checked against
#: the CV with a bare ``"german" in resume_lower``, so a German Lebenslauf saying
#: "Sprachen: Deutsch (C1)" was reported as *missing* German -- a gap it can
#: never close, because the evidence is right there under the native name.
#: Matching the CV side in both languages is what the posting side already does.
LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "German": ("german", "deutsch"),
    "English": ("english", "englisch"),
    "French": ("french", "französisch", "franzosisch"),
    "Spanish": ("spanish", "spanisch"),
    "Italian": ("italian", "italienisch"),
    "Dutch": ("dutch", "niederländisch", "niederlandisch"),
    "Portuguese": ("portuguese", "portugiesisch"),
    "Polish": ("polish", "polnisch"),
    "Russian": ("russian", "russisch"),
    "Turkish": ("turkish", "türkisch", "turkisch"),
    "Chinese": ("chinese", "chinesisch", "mandarin"),
    "Japanese": ("japanese", "japanisch"),
    "Arabic": ("arabic", "arabisch"),
    "Czech": ("czech", "tschechisch"),
    "Danish": ("danish", "dänisch", "danisch"),
    "Finnish": ("finnish", "finnisch"),
    "Norwegian": ("norwegian", "norwegisch"),
    "Swedish": ("swedish", "schwedisch"),
    "Hungarian": ("hungarian", "ungarisch"),
    "Romanian": ("romanian", "rumänisch", "rumänisch"),
    "Ukrainian": ("ukrainian", "ukrainisch"),
    "Hindi": ("hindi",),
    "Korean": ("korean", "koreanisch"),
}


def language_aliases(language: str) -> tuple[str, ...]:
    """
    Every spelling a CV may use to evidence ``language``, lower-cased.

    Falls back to the language's own name for anything not in the table, so an
    unlisted language still matches a CV that writes it in English.
    """
    aliases = LANGUAGE_ALIASES.get(language)
    if aliases:
        return aliases
    return (language.casefold(),)

#: The CEFR level, when the posting gives one.
_CEFR_RE = re.compile(r"\b(?:CEFR\s*)?(A1|A2|B1|B2|C1|C2)\b", re.IGNORECASE)

#: Cues that a language mentioned in the line is a requirement rather than an
#: aside. "Our team speaks Portuguese" is not a requirement; " Portuguese is
#: required" is.
_LANGUAGE_REQUIRED_CUES = (
    "required",
    "requirement",
    "must",
    "essential",
    "fluent",
    "fluency",
    "proficient",
    "proficiency",
    "native",
    "bilingual",
    "language",
    "kenntnisse",
    "sprachkenntnisse",
    "verhandlungssicher",
)

#: Degrees, and the level each one represents. Ordered longest-first so
#: "Master of Science" is not read as a "Master".
_DEGREE_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "doctorate",
        (
            r"\bph\.?d\.?\b",
            r"\bdoctor(?:ate|al)?\b",
            r"\bdr\.?\b",
            r"\bdoctoral\b",
            r"\bd\.?sc\.?\b",
            r"\bpromotion\b",
        ),
    ),
    (
        "master",
        (
            r"\bmaster(?:'s| of [a-z]+)?\b",
            r"\bm\.?sc\.?\b",
            r"\bm\.?a\.?\b",
            r"\bmba\b",
            r"\bmsc\b",
            r"\bma\b",
            r"\bmagister\b",
        ),
    ),
    (
        "bachelor",
        (
            r"\bbachelor(?:'s| of [a-z]+)?\b",
            r"\bb\.?sc\.?\b",
            r"\bb\.?a\.?\b",
            r"\bbsc\b",
            r"\bba\b",
            r"\bbachelorgrad\b",
            r"\bdiplom-?ingenieur\b",
        ),
    ),
    (
        "associate",
        (
            r"\bassociate(?:'s| degree)?\b",
            r"\bhnd\b",
            r"\bfacharbeiter\b",
            # "Fachhochschulabschluss" is a single compound noun, so no word
            # boundary after "hochschule" can ever match it.
            r"\bfachhochschulabschluss\b",
            r"\bfachhochschule\b",
        ),
    ),
    (
        "high_school",
        (
            r"\bhigh school\b",
            r"\bsecondary school\b",
            r"\bgymnasium\b",
            r"\babitur\b",
            r"\ba-?levels?\b",
            r"\brealschule\b",
        ),
    ),
)

_DEGREE_RE = re.compile(
    "|".join(f"({pattern})" for _, patterns in _DEGREE_LEVELS for pattern in patterns),
    re.IGNORECASE,
)


def _degree_level(text: str) -> str | None:
    """The highest degree level named in a piece of text."""
    found: set[str] = set()
    for level, patterns in _DEGREE_LEVELS:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(level)
                break
    if not found:
        return None
    # Highest wins: a posting saying "Bachelor's, or a Master's" asks for at
    # least one of them, and the stricter reading is the safe one to report.
    order = ["high_school", "associate", "bachelor", "master", "doctorate"]
    return max(found, key=order.index)


def highest_degree_in_text(text: str) -> str | None:
    """
    The highest qualification a CV claims, as a comparable level.

    The ordering is what makes "the posting wants a Master's, the CV shows a
    Bachelor's" checkable. A CV that says "BSc Computer Science" resolves to
    ``bachelor``; one that names no qualification at all resolves to ``None``,
    which callers must treat as *unverified* rather than as *no degree* — a
    parser that could not read the qualification cannot conclude the candidate
    lacks one.
    """
    return _degree_level(text or "")


#: Exposed so the scoring layer and this module cannot disagree on the ordering.
DEGREE_ORDER: tuple[str, ...] = (
    "high_school",
    "associate",
    "bachelor",
    "master",
    "doctorate",
)


# ---------------------------------------------------------------------------
# Certifications
# ---------------------------------------------------------------------------

#: A named certification. The vendor prefix is required, because a bare "Certified
#: Kubernetes Administrator" is a real title but "Administrator" on its own is not.
_CERT_RE = re.compile(
    r"\b("
    r"AWS Certified[^\n,.;)]*"
    r"|Azure (?:Certified|Exam|Administrator|Developer)[^\n,.;)]*"
    r"|Google Cloud (?:Certified|Professional)[^\n,.;)]*"
    r"|Certified Kubernetes[^\n,.;)]*"
    r"|CKA\b|CKAD\b|CKS\b"
    r"|PMP\b|PRINCE2\b|Prince 2\b"
    r"|CISSP\b|CISM\b|CISA\b|CIAB\b"
    r"|ITIL(?: v?\d)?\b"
    r"|CompTIA [A-Z][^\n,.;)]*"
    r"|Cisco (?:CCNA|CCNP|CCIE)[^\n,.;)]*"
    r"|VMware (?:VCP|VCP-DT)[^\n,.;)]*"
    r"|Salesforce Certified[^\n,.;)]*"
    r"|TOGAF\b"
    r"|Certified Scrum Master\b|CSM\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Industry terminology
# ---------------------------------------------------------------------------

#: Regulatory frameworks, standards and domain acronyms. These are the vocabulary
#: of a sector rather than a technology, so the skill taxonomy does not carry
#: them — but a candidate working in that sector cannot demonstrate their job
#: without them.
_INDUSTRY_TERMS: tuple[str, ...] = (
    # Regulatory and privacy
    "GDPR",
    "HIPAA",
    "CCPA",
    "PPCI",
    "LGPD",
    "SOX",
    "DORA",
    "PSD2",
    "SOC 1",
    "SOC 2",
    "SOC 2 Type II",
    "ISO 27001",
    "ISO 9001",
    "ISO 13485",
    "PCI DSS",
    "PCI-DSS",
    "NIST",
    "NIST CSF",
    "FedRAMP",
    "Cyber Essentials",
    # Finance and accounting
    "IFRS",
    "GAAP",
    "IFRS 17",
    "Solvency II",
    "Basel III",
    "MiFID",
    "EMIR",
    "KYC",
    "AML",
    "Sarbanes-Oxley",
    "COSO",
    "IFRS 9",
    "VaR",
    # Healthcare and life sciences
    "ICH GCP",
    "GCP",
    "GMP",
    "FDA 21 CFR",
    "21 CFR Part 11",
    "HL7",
    "FHIR",
    "ICD-10",
    "SNOMED",
    "PDL1",
    "REACH",
    "CLP",
    # Legal and compliance
    "GDPR DPAs",
    "eDiscovery",
    "FCPA",
    "MLA",
    "CCPA CPRA",
    # Energy, utilities, transport
    "NERC CIP",
    "IEC 61850",
    "ISO 50001",
    "IATEK",
    "AS9100",
    "IATF 16949",
    "EASA",
    "EN 9100",
    # Education and research
    "IRB",
    "CITI",
    "IRB approval",
    "Grant writing",
    # General safety and quality
    "Six Sigma",
    "Lean",
    "Kaizen",
    "CAPA",
    "GMP",
    "HACCP",
)

_INDUSTRY_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    + "|".join(re.escape(term) for term in sorted(_INDUSTRY_TERMS, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _requirement_line(text: str, index: int) -> str:
    """The requirement block a match sits in, for reading the surrounding cues."""
    start = text.rfind("\n", 0, index)
    end = text.find("\n", index)
    end = len(text) if end < 0 else end
    return text[start + 1 : end]


def _is_required(line: str) -> bool:
    lowered = line.lower()
    return any(cue in lowered for cue in _LANGUAGE_REQUIRED_CUES)


def extract_language_requirements(job_description: str) -> list[dict[str, Any]]:
    """
    Languages the posting asks for, with the level where one is given.

    A language mentioned without any requirement cue is skipped: "our Berlin team
    speaks Polish" is a fact about the team, not something to be scored against a
    CV.
    """
    text = job_description or ""
    found: dict[str, dict[str, Any]] = {}

    matches: list[tuple[int, str]] = [
        (match.start(), match.group(1).title()) for match in _LANGUAGE_NAME_RE.finditer(text)
    ]
    matches += [
        (match.start(), _GERMAN_COMPOUND_LANGUAGES[match.group(1).lower()])
        for match in _GERMAN_COMPOUND_RE.finditer(text)
    ]

    for position, language in sorted(matches):
        line = _requirement_line(text, position)
        if not _is_required(line):
            continue

        entry = found.setdefault(
            language,
            {"language": language, "level": None, "required": True},
        )
        level_match = _CEFR_RE.search(line)
        if level_match and not entry["level"]:
            entry["level"] = level_match.group(1).upper()

    return sorted(found.values(), key=lambda item: item["language"])


def extract_education_requirements(job_description: str) -> dict[str, Any]:
    """
    The highest degree level the posting names.

    A posting that says "a degree in Computer Science" without naming a level is
    reported at ``bachelor`` — the conventional default, and the one most CVs are
    measured against. A posting that names no degree at all reports ``None``, so
    the check is skipped rather than failed.
    """
    text = job_description or ""
    level = _degree_level(text)
    if level is None:
        # "a degree in X" with no level named.
        if re.search(
            r"\b(?:a|an|degree in|studium|abschluss)\b", text, re.IGNORECASE
        ) and re.search(
            r"\bdegree\b|\bdiplom\b|\bbachelor\b|\bmaster\b|\bphd\b", text, re.IGNORECASE
        ):
            level = "bachelor"
        else:
            return {
                "stated": False,
                "level": None,
                "detail": "The posting names no degree requirement.",
            }

    return {
        "stated": True,
        "level": level,
        "detail": f"The posting asks for a {level}-level qualification or above.",
    }


def extract_certification_requirements(job_description: str) -> list[str]:
    """Named certifications the posting asks for."""
    seen: list[str] = []
    for match in _CERT_RE.finditer(job_description or ""):
        name = match.group(1).strip()
        if name and name.lower() not in {item.lower() for item in seen}:
            seen.append(name)
    return seen


def extract_industry_terminology(job_description: str) -> list[str]:
    """Regulatory frameworks, standards and sector acronyms in the posting."""
    seen: list[str] = []
    for match in _INDUSTRY_RE.finditer(job_description or ""):
        name = match.group(1).strip()
        if name and name.lower() not in {item.lower() for item in seen}:
            seen.append(name)
    return seen


# ---------------------------------------------------------------------------
# Keyword stuffing
# ---------------------------------------------------------------------------

#: A content word repeated more often than this in a CV of the given length.
#: 5% of distinct words, with a floor, so a short CV is not condemned for saying
#: "Docker" four times in three bullets.
_STUFFING_FLOOR = 4

#: Words whose repetition carries no information, and which therefore must not
#: trigger the check. A length filter is not enough: "with", "team" and "work" are
#: long enough to pass one and occur legitimately many times in any CV, so they
#: would otherwise be reported as padding on almost every document.
_STUFFING_STOPWORDS = frozenset(
    {
        # English function and filler words.
        "with",
        "from",
        "that",
        "this",
        "they",
        "them",
        "then",
        "than",
        "there",
        "these",
        "those",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "must",
        "also",
        "into",
        "onto",
        "over",
        "under",
        "each",
        "both",
        "many",
        "more",
        "most",
        "some",
        "such",
        "only",
        "just",
        "like",
        "work",
        "works",
        "worked",
        "working",
        "team",
        "teams",
        "role",
        "roles",
        "year",
        "years",
        "time",
        "times",
        "day",
        "days",
        "use",
        "used",
        "using",
        "make",
        "made",
        "making",
        "help",
        "helps",
        "need",
        "needs",
        "based",
        "across",
        "within",
        "through",
        "about",
        "after",
        "before",
        "between",
        "during",
        "against",
        "including",
        "support",
        "supported",
        "manage",
        "managed",
        "managing",
        "build",
        "built",
        "building",
        "developer",
        "developers",
        "engineer",
        "engineers",
        # German function and filler words.
        "und",
        "oder",
        "aber",
        "nicht",
        "auch",
        "noch",
        "schon",
        "sehr",
        "alle",
        "allen",
        "aller",
        "durch",
        "ohne",
        "seit",
        "bis",
        "mit",
        "von",
        "vor",
        "nach",
        "unter",
        "zwischen",
        "werden",
        "wurde",
        "wurden",
        "kann",
        "soll",
        "sollen",
        "muss",
        "sowie",
        "dabei",
        "dazu",
        "davon",
        "diese",
        "dieser",
        "arbeit",
        "arbeiten",
        "erfahrung",
        "kenntnisse",
        "zusammenarbeit",
        "bereich",
        "aufgaben",
        "stelle",
        "stellen",
        "jahre",
        "zeit",
        "projekt",
        "projekte",
        "neue",
        "neuer",
        "gute",
    }
)


def find_keyword_stuffing(
    resume_text: str,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Content words repeated far more often than they carry information.

    This is the one keyword problem that keyword matching cannot see: every
    repetition of a *matched* keyword looks like good coverage from the outside,
    so a CV that repeats one term to fill a section scores well and reads as spam.
    Recruiters and some parsers both treat that as a negative signal.

    Skipped words are the ones repetition is normal for — a name, a company, a
    job title, and anything inside a URL or an email address, all of which appear
    many times by design.

    Only *content* words are counted, and only ones long enough to mean something,
    so "the" and "and" cannot trigger this.
    """
    text = resume_text or ""
    if not text.strip():
        return {"stuffed": [], "detail": "no text to inspect", "distinct_words": 0}

    # Neutralise the parts where repetition is expected or meaningless.
    scrubbed = re.sub(r"[\w.+-]+@[\w.-]+", " ", text)  # email addresses
    scrubbed = re.sub(r"https?://\S+|www\.\S+", " ", scrubbed)  # URLs
    scrubbed = re.sub(r"\+?\d[\d\s().-]{6,}\d", " ", scrubbed)  # phone numbers
    scrubbed = re.sub(r"\b(?:19|20)\d{2}\b", " ", scrubbed)  # years
    scrubbed = re.sub(r"[^\w\s+#.-]", " ", scrubbed)

    words = [
        word.casefold()
        for word in re.findall(r"[A-Za-z][\w+#.-]{3,}", scrubbed)
        if word.casefold() not in _STUFFING_STOPWORDS
    ]
    if not words:
        return {"stuffed": [], "detail": "no content words to inspect", "distinct_words": 0}

    total = len(words)
    threshold = max(_STUFFING_FLOOR, int(total * 0.05))
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    stuffed = [
        {"term": term, "count": count, "share": round(count / total, 3)}
        for term, count in sorted(counts.items(), key=lambda item: -item[1])
        if count > threshold
    ][:top_n]

    if not stuffed:
        return {
            "stuffed": [],
            "detail": f"no word appears more than {threshold} times across {total} words",
            "distinct_words": len(counts),
        }

    worst = stuffed[0]
    return {
        "stuffed": stuffed,
        "threshold": threshold,
        "total_words": total,
        "distinct_words": len(counts),
        "detail": (
            f"{len(stuffed)} term(s) appear more than {threshold} times across "
            f"{total} words - most often {worst['term']!r} at {worst['count']} times. "
            "Repeating a term adds nothing and reads as padding."
        ),
    }
