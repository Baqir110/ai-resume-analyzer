"""
Transparent ATS scoring engine with a category breakdown.

Every score is derived from measurable checks. No category is invented, no
score is padded. The breakdown shows exactly what was measured, what it is
worth, and how many points were lost.

Scoring categories (all reproducible from actual checks):

  Keyword Match       — the posting's vocabulary, plus its job title
  Required Skills     — mandatory vs. optional skill coverage
  Experience Relevance — term overlap, soft skills, and years of experience
  CV Structure        — sections, contact details, and formatting conventions
  PDF Parsing         — machine-readability of the generated document

Each category is scored 0–100, then weighted. The weights sum to 1.0, so the
blended result is also 0–100. Every category reports its own ``points_lost``
so the user can see precisely why the score is what it is.

A note on PDF Parsing
---------------------
``PDF Parsing`` cannot be measured before a PDF exists. Rather than award
points that were never earned *or* charge the candidate for a step that has not
run yet, the category reports ``"not_measured": True``, its weight is excluded
from the blend, and the remaining weight is redistributed proportionally. The
overall score is then explicitly labelled as pre-generation. Once a PDF is
supplied the category is measured normally. This is stated in the summary
rather than hidden, because a score that quietly skipped a category would be
the kind of number requirement 27 rules out.

A note on what is *not* scored
-------------------------------
The score measures how completely and how legibly a CV answers a specific
posting. It does not measure whether the candidate is a good hire, and nothing in
it rewards padding: a missing required skill costs points whether or not the CV
is otherwise polished, and no check rewards a section the candidate does not
have. Every category therefore moves in one direction only — toward saying more
of what is actually there, more clearly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from app.services.analysis.ats_analyzer import (
    extract_context_requirements,
    extract_keywords_from_jd,
    format_skill_name,
    is_skill_in_text,
    normalize_skill,
)
from app.services.analysis.formatting_checks import analyze_pdf_layout, analyze_text_formatting
from app.services.analysis.requirement_extraction import (
    DEGREE_ORDER,
    extract_certification_requirements,
    extract_education_requirements,
    extract_industry_terminology,
    extract_language_requirements,
    find_keyword_stuffing,
    highest_degree_in_text,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights — must sum to 1.0
#
# Keyword coverage dominates because it is what an ATS actually matches on.
# PDF Parsing is weighted lowest because it is a property of the generator, not
# of the candidate: a strong CV rendered by a poor template should not be scored
# as a weak CV.
# ---------------------------------------------------------------------------

WEIGHT_KEYWORD_MATCH = 0.30
WEIGHT_REQUIRED_SKILLS = 0.20
WEIGHT_EXPERIENCE = 0.20
WEIGHT_STRUCTURE = 0.15
WEIGHT_PDF_PARSING = 0.15

_CATEGORY_WEIGHTS: dict[str, float] = {
    "keyword_match": WEIGHT_KEYWORD_MATCH,
    "required_skills": WEIGHT_REQUIRED_SKILLS,
    "experience_relevance": WEIGHT_EXPERIENCE,
    "cv_structure": WEIGHT_STRUCTURE,
    "pdf_parsing": WEIGHT_PDF_PARSING,
}

_CATEGORY_LABELS: dict[str, str] = {
    "keyword_match": "Keyword Match",
    "required_skills": "Required Skills",
    "experience_relevance": "Experience Relevance",
    "cv_structure": "CV Structure",
    "pdf_parsing": "PDF Parsing",
}

#: Below this a category is reported as a gap worth improving.
IMPROVEMENT_THRESHOLD = 85.0

# ---------------------------------------------------------------------------
# PDF parsing thresholds
# ---------------------------------------------------------------------------

#: Below this many extracted characters a PDF counts as effectively blank.
MIN_PDF_TEXT_CHARS = 200

#: Fraction of the candidate's content words that must survive typesetting.
MIN_CONTENT_RETENTION = 0.55

#: How many detectable sections count as a well-formed document.
MIN_SECTIONS_EXPECTED = 2

#: Years-of-experience tolerance when matching the JD's stated requirement.
#: A JD asking for 5 years and a CV showing 4 is a match, not a gap; demanding
#: the exact figure would flag a rounding difference as a missing qualification.
YEARS_TOLERANCE = 0.8

_YEARS_RE = re.compile(r"(\d+)\+?\s*(?:years?|jahre|jahren)", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Lowercase and strip accents, so ``Müller`` compares equal to ``Muller``."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.casefold()


# ---------------------------------------------------------------------------
# Required vs. optional skills
# ---------------------------------------------------------------------------

_REQUIRED_CUES = (
    "must have",
    "must be",
    "required",
    "requirement",
    "essential",
    "mandatory",
    "proficiency in",
    "proficient in",
    "experience with",
    "experience in",
    "kenntnisse in",
    "erfahrung mit",
    "kenntnisse in",
    "zwingend",
)

_OPTIONAL_CUES = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "bonus",
    "optional",
    "a plus",
    "desirable",
    "von vorteil",
    "wunschenswert",
    "plus",
)


def _segment_cue(segment: str) -> tuple[bool, bool]:
    """Whether one line carries a required-cue, an optional-cue, or both."""
    lowered = segment.lower()
    return (
        any(cue in lowered for cue in _REQUIRED_CUES),
        any(cue in lowered for cue in _OPTIONAL_CUES),
    )


def _classify_skill_requirement(job_description: str, skill: str) -> str:
    """
    Decide whether a skill is required, optional, or unspecified.

    Postings almost always express requirements as a *labelled list* rather than
    prose: a "Required:" or "Nice to have:" heading, then one bullet per skill.
    Splitting only on sentence boundaries isolates each bullet from the heading
    that gives it meaning, so every skill reads as unspecified -- the
    required/optional split silently collapses, and a missing "nice to have"
    costs exactly as much as a missing requirement. The heading is therefore
    carried down: each line is judged together with the last line that looked
    like a label, and the bullets beneath it inherit that label.

    Judged per-item, not per-document, so a skill named once as required and
    again as preferred stays ambiguous rather than being resolved by whichever
    mention happens to come last.
    """
    escaped = re.escape(normalize_skill(skill))
    active_label: tuple[bool, bool] | None = None

    for raw_line in job_description.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # A label is a short, bullet-free line ending in a colon that names
        # itself as one. "Required:" qualifies; a prose sentence with a colon
        # mid-clause does not.
        stripped = line.lstrip("-*• \t")
        if stripped.endswith(":") and len(stripped) <= 40:
            required_cue, optional_cue = _segment_cue(stripped)
            if required_cue or optional_cue:
                active_label = (required_cue, optional_cue)
                continue

        if escaped not in line.lower():
            continue

        has_required, has_optional = _segment_cue(line)

        # Inherit the enclosing label when the bullet is not explicit itself.
        if active_label and not (has_required or has_optional):
            has_required, has_optional = active_label

        if has_required and not has_optional:
            return "required"
        if has_optional and not has_required:
            return "optional"

    return "unspecified"


# ---------------------------------------------------------------------------
# Job title alignment
# ---------------------------------------------------------------------------

#: Seniority words. Removed before the titles are compared, so "Senior Platform
#: Engineer" and "Platform Engineer" are the same role at different levels
#: rather than two unrelated titles.
_SENIORITY_WORDS = frozenset(
    {
        "senior",
        "sr",
        "junior",
        "jr",
        "lead",
        "principal",
        "staff",
        "head",
        "chief",
        "director",
        "vp",
        "vice",
        "president",
        "intern",
        "trainee",
        "graduate",
        "entry",
        "associate",
        "global",
        "regional",
    }
)

#: A job title ends before one of these, which start the requirements.
_TITLE_STOP_WORDS = frozenset(
    {
        "about",
        "requirements",
        "required",
        "qualifications",
        "qualification",
        "responsibilities",
        "role",
        "we",
        "the",
        "you",
        "your",
        "our",
        "and",
        "location",
        "salary",
        "benefits",
        "apply",
        "now",
        "skills",
        "experience",
        "anforderungen",
        "aufgaben",
        "profil",
        "kenntnisse",
        "über",
        "standort",
    }
)

#: Curated title vocabulary. Matched as whole words so "engineer" is found in
#: "Platform Engineer" without a general word-list dependency.
_TITLE_WORDS = frozenset(
    {
        "engineer",
        "developer",
        "architect",
        "analyst",
        "scientist",
        "manager",
        "administrator",
        "technician",
        "consultant",
        "specialist",
        "designer",
        "researcher",
        "programmer",
        "tester",
        "operator",
        "coordinator",
        "accountant",
        "recruiter",
        "nurse",
        "teacher",
        "professor",
        "officer",
        "executive",
        "strategist",
        "planner",
        "auditor",
        "editor",
        "writer",
        # Support and entry-level roles, which are common openings and were
        # previously unread — so a posting for one looked like a posting with no
        # title at all, and the check silently did nothing.
        "assistant",
        "associate",
        "trainee",
        "intern",
        "apprentice",
        "clerk",
        "representative",
        "advisor",
        "controller",
        "supervisor",
        "attendant",
        "driver",
        "agent",
        "broker",
        "underwriter",
        "actuary",
    }
)

#: Ordered so the longest and most specific role is preferred when a posting
#: names several.
_TITLE_STOP_SECTIONS = re.compile(
    r"\b(?:about|requirements?|required|qualifications?|responsibilities|"
    r"what you.ll do|what we.re looking for|your profile|the role|role:|"
    r"location|salary|benefits|apply now|anforderungen|aufgaben|profil|"
    r"kenntnisse|standort|ihr profil)\b",
    re.IGNORECASE,
)


def _extract_job_title(text: str) -> str:
    """
    Best guess at the title a text is about.

    A title is only accepted when it actually contains a role noun. Accepting a
    short capitalised phrase instead would pick the candidate's *name* off the
    first line of a CV — "Jane Doe" looks exactly like a title to a shape-based
    heuristic — and then report a title mismatch on a CV whose headline is
    correct. A wrong reading here produces a confident, wrong finding, so the
    check reports "not stated" instead when it is unsure.

    Only the first few lines are read: that is where both a CV's headline and a
    posting's title live, and scanning the whole document would pick up a job title
    buried in the requirements section.
    """
    inspected = 0
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().strip("*#=-—– \t")
        if not line or len(line) > 80:
            continue

        inspected += 1
        if inspected > 6:
            break

        # A contact line is not a title.
        if "@" in line or re.search(r"\+\d[\d\s().-]{6,}", line):
            continue
        if re.search(r"\b(?:19|20)\d{2}\b", line):  # a dated role entry
            continue

        words = _normalise_words(line).split()
        if not words or len(words) > 8:
            continue
        if any(word in _TITLE_STOP_WORDS for word in words):
            continue
        if any(word in _TITLE_WORDS for word in words):
            return _trim_to_title(line)
    return ""


def _trim_to_title(line: str) -> str:
    """
    Cut a line down to the title it starts with.

    A posting's first line is often the title followed by the rest of the sentence
    — "Marketing Assistant. Python and Excel." Reading the whole line as the title
    makes every comparison fail and puts stray words in front of the user. The cut
    is at punctuation *followed by whitespace*, so "Node.js Developer" and "ci-cd
    engineer" survive intact.
    """
    match = re.search(r"[.,;:]\s", line)
    if match:
        line = line[: match.start()]
    return line.strip().strip("*#=-—– \t") or line.strip()


def _normalise_words(text: str) -> str:
    """
    Lowercase for comparison, keeping the internal punctuation of technical names.

    ``.`` and ``-`` are retained *inside* a token, so "node.js" and "ci-cd" stay
    whole, but are stripped from the ends. Without that, the last word of a line
    like "Marketing Assistant." never matches "assistant" — a full stop at the end
    of a sentence silently made the title unreadable.
    """
    tokens = re.findall(r"\S+", text or "")
    cleaned = [re.sub(r"^\W+|\W+$", "", token).casefold() for token in tokens]
    return " ".join(token for token in cleaned if token)


def _title_core(title: str) -> set[str]:
    """The title's meaning-carrying words: no seniority, no stop words."""
    words = set(re.findall(r"[a-z]+", _normalise_words(title)))
    return {
        word
        for word in words
        if word not in _SENIORITY_WORDS and word not in _TITLE_STOP_WORDS and len(word) > 2
    }


def _score_title_alignment(resume_text: str, job_description: str) -> dict[str, Any]:
    """
    Does the CV name the role the posting is for?

    Returns a dict with ``stated`` (a title could be read at all), ``aligned``,
    the posting's title, and the candidate's. When the posting states no title the
    check reports ``stated: False`` and is excluded from the score, because a
    posting that never names the role cannot fairly be matched against one.
    """
    target = _extract_job_title(job_description)
    candidate = _extract_job_title(resume_text)

    if not target:
        return {
            "stated": False,
            "aligned": False,
            "target": "",
            "candidate": candidate,
            "detail": "The posting does not state a job title, so no title match was attempted.",
        }

    target_core = _title_core(target)
    candidate_core = _title_core(candidate)

    if not target_core:
        return {
            "stated": False,
            "aligned": False,
            "target": target,
            "candidate": candidate,
            "detail": f"No role word could be read from the posting's heading ({target!r}).",
        }

    if not candidate:
        return {
            "stated": True,
            "aligned": False,
            "target": target,
            "candidate": "",
            "detail": (
                f"No job title could be read from the CV, so the posting's title "
                f"({target}) could not be matched."
            ),
        }

    # The role word has to appear, not merely share a letter with it.
    shared = target_core & candidate_core
    if shared:
        return {
            "stated": True,
            "aligned": True,
            "target": target,
            "candidate": candidate,
            "shared": sorted(shared),
            "detail": f"Both name the same role: {', '.join(sorted(shared))}.",
        }

    return {
        "stated": True,
        "aligned": False,
        "target": target,
        "candidate": candidate,
        "shared": [],
        "detail": (
            f"The CV is headed {candidate!r} but the posting is for {target!r}; "
            "use the posting's title if it is an accurate description of the role."
        ),
    }


# ---------------------------------------------------------------------------
# Location, work mode, and work authorisation
# ---------------------------------------------------------------------------

#: Facts a posting states about *where* and *under what terms* the role is
#: offered. These are content requirements in their own right: a CV that does not
#: say whether the candidate can work in the location, or needs sponsorship, leaves
#: a question the recruiter must ask before the application can progress.
#:
#: Matched as phrases, because the single words are far too common to be evidence —
#: "office" appears in almost any job description.
_LOCATION_PATTERNS: tuple[str, ...] = (
    r"\bremote\b",
    r"\bhybrid\b",
    r"\bon[- ]site\b",
    r"\bin[- ]office\b",
    r"\bfully remote\b",
    r"\bwork from home\b",
    r"\bwfh\b",
)

_WORK_AUTH_PATTERNS: tuple[str, ...] = (
    r"\bwork permit\b",
    r"\bwork authori[sz]ation\b",
    r"\bright to work\b",
    r"\bvisa sponsorship?\b",
    r"\bsponsor(?:ed|s)? (?:a )?visa\b",
    r"\bsecurity clearance\b",
    r"\brelocation (?:package|support|assistance)\b",
)

#: A named place, taken from the phrase that introduces it. A gazetteer would be
#: more thorough and would also need maintaining; the introducing phrase is a
#: reliable enough signal for the location line of a posting.
_LOCATION_HINT_RE = re.compile(
    r"\b(?:based in|located in|office in|site in|relocation to)\s+([A-Z][\w.-]+(?:\s[A-Z][\w.-]+)?)"
)


def _score_logistics(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    Does the CV address where and how the candidate can work?

    Reported separately from the technical keywords, because these are not
    vocabulary. "Docker" missing from a CV is a different kind of problem from
    "no statement about needing a visa", and a recruiter treats them differently.
    """
    resume_lower = (resume_text or "").lower()
    jd_lower = (job_description or "").lower()

    def find(patterns: tuple[str, ...], haystack: str) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, haystack, re.IGNORECASE):
                label = match if isinstance(match, str) else match[0]
                if label.lower() not in {item.lower() for item in found}:
                    found.append(label)
        return found

    jd_location = find(_LOCATION_PATTERNS, jd_lower)
    jd_auth = find(_WORK_AUTH_PATTERNS, jd_lower)

    for match in _LOCATION_HINT_RE.findall(job_description or ""):
        if match.lower() not in {item.lower() for item in jd_location}:
            jd_location.append(match)

    resume_location = find(_LOCATION_PATTERNS, resume_lower)
    resume_auth = find(_WORK_AUTH_PATTERNS, resume_lower)

    if not jd_location and not jd_auth:
        return {
            "stated": False,
            "covered": True,
            "location_required": [],
            "work_authorisation_required": [],
            "detail": (
                "The posting says nothing about location, work mode or work "
                "authorisation, so there was nothing to match."
            ),
        }

    # Being more specific than the posting asks is not a failure, so any work
    # mode satisfies a location requirement.
    location_ok = bool(resume_location) or not jd_location
    auth_ok = bool(resume_auth) or not jd_auth

    missing: list[str] = []
    if not location_ok:
        missing.append("where you can work (remote, hybrid, or on-site)")
    if not auth_ok:
        missing.append("whether you need visa sponsorship or hold a work permit")

    stated_terms = ", ".join(jd_location + jd_auth)
    if missing:
        detail = (
            f"The posting states {stated_terms}. Your CV does not address "
            + " and ".join(missing)
            + "."
        )
    else:
        detail = f"The posting states {stated_terms}, and your CV addresses both."

    return {
        "stated": True,
        "covered": location_ok and auth_ok,
        "location_required": jd_location,
        "work_authorisation_required": jd_auth,
        "location_stated": resume_location,
        "work_authorisation_stated": resume_auth,
        "missing": missing,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Category 1 — Keyword Match
# ---------------------------------------------------------------------------


def _score_keyword_match(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    Coverage of the posting's own vocabulary.

    The posting's job title is scored alongside its technical terms. That is
    deliberate: a title is the single string a screener reads first, and a CV
    titled "Software Engineer" applying to a "Platform Engineer" role loses a
    real, checkable point even when every technology matches. Seniority is
    separated from the rest of the title so "Senior Data Engineer" counts as a
    match for "Data Engineer" at a different seniority, rather than as a total
    miss — the wording differs but the role is the same.
    """
    jd_keywords = extract_keywords_from_jd(job_description)
    resume_lower = (resume_text or "").lower()

    matched: list[str] = []
    missing: list[str] = []
    matched_via_synonym: list[str] = []

    for keyword in sorted(jd_keywords, key=str.lower):
        label = format_skill_name(keyword)
        if is_skill_in_text(keyword, resume_lower):
            matched.append(label)
            # Surfaced so the UI can say "matched, via a different wording",
            # which is the difference between a keyword gap and a wording gap.
            if normalize_skill(keyword) not in resume_lower:
                matched_via_synonym.append(label)
        else:
            missing.append(label)

    title_alignment = _score_title_alignment(resume_text, job_description)
    logistics = _score_logistics(resume_text, job_description)

    # Industry terminology — regulatory frameworks, standards, sector acronyms.
    # Counted as ordinary keywords, because that is what they are: the vocabulary
    # of a sector rather than a technology. Someone working in a regulated
    # industry cannot demonstrate that without naming the frameworks, and none of
    # them are in the technology taxonomy.
    industry_terms = extract_industry_terminology(job_description)
    industry_matched = [t for t in industry_terms if is_skill_in_text(t, resume_lower)]
    industry_missing = [t for t in industry_terms if not is_skill_in_text(t, resume_lower)]
    matched.extend(industry_matched)
    missing.extend(industry_missing)

    # The title counts once, and so does the pair of location / work-authorisation
    # facts. Each is a single thing the posting either specified and the CV
    # covered, or did not — so it enters as one unit rather than as several,
    # otherwise a posting that says "remote" would outweigh a rare technical term.
    total = len(matched) + len(missing)
    extra = sum(1 for stated in (title_alignment["stated"], logistics["stated"]) if stated)

    if total == 0 and extra == 0:
        return {
            "score": 100.0,
            "matched": [],
            "missing": [],
            "matched_via_synonym": [],
            "title_alignment": title_alignment,
            "logistics": logistics,
            "explanation": (
                "No specific technical keywords were found in the job "
                "description, so nothing was counted against the CV."
            ),
        }

    numerator = len(matched)
    if title_alignment["stated"] and title_alignment["aligned"]:
        numerator += 1
    if logistics["stated"] and logistics["covered"]:
        numerator += 1

    denominator = total + extra
    score = 100.0 * numerator / denominator if denominator else 100.0

    explanation = f"{len(matched)} of {total} keywords from the posting appear in the CV."
    if title_alignment["stated"]:
        if title_alignment["aligned"]:
            explanation += f" The target role ({title_alignment['target']}) is named in the CV."
        else:
            explanation += (
                f" The CV does not use the posting's job title " f"({title_alignment['target']})."
            )
    if logistics["stated"] and not logistics["covered"]:
        explanation += " " + logistics["detail"]
    if missing:
        explanation += f" Not present: {', '.join(missing[:6])}."
        if len(missing) > 6:
            explanation += f" ({len(missing) - 6} more.)"
    if matched_via_synonym:
        explanation += f" {len(matched_via_synonym)} matched under a different wording."

    return {
        "score": round(score, 1),
        "matched": matched,
        "missing": missing,
        "matched_via_synonym": matched_via_synonym,
        "title_alignment": title_alignment,
        "logistics": logistics,
        "industry_terminology": {
            "matched": industry_matched,
            "missing": industry_missing,
        },
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Category 2 — Required Skills
def _score_structured_qualifications(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    Education, certifications and languages — the requirements a posting states
    that keyword matching cannot see.

    None of these are vocabulary. "Bachelor's degree", "AWS Certified Solutions
    Architect" and "German at B2" are qualifications a candidate either holds or
    does not, and a recruiter treats a gap in any of them as decisive. Folded into
    ``Required Skills`` rather than given a category of their own, so the score
    keeps the five categories the rest of the application depends on.

    A posting that states none of these reports ``stated: False`` and is excluded,
    because a CV cannot be marked down for a qualification nobody asked for.
    """
    resume = resume_text or ""
    resume_lower = _normalise(resume)

    # -- education ------------------------------------------------------
    education = extract_education_requirements(job_description)
    education_satisfied = False
    if education["stated"]:
        # The candidate's own highest qualification, so "the posting wants a
        # Master's and the CV shows a Bachelor's" is a real gap rather than a
        # wording difference.
        candidate_level = highest_degree_in_text(resume)
        education_satisfied = bool(
            candidate_level
            and candidate_level in DEGREE_ORDER
            and DEGREE_ORDER.index(candidate_level) >= DEGREE_ORDER.index(education["level"])
        )

    # -- certifications -------------------------------------------------
    required_certs = extract_certification_requirements(job_description)
    certs_matched = [c for c in required_certs if c.casefold() in resume_lower]
    certs_missing = [c for c in required_certs if c.casefold() not in resume_lower]

    # -- languages ------------------------------------------------------
    required_languages = extract_language_requirements(job_description)
    languages_matched: list[str] = []
    languages_missing: list[str] = []
    for entry in required_languages:
        name = entry["language"]
        (languages_matched if name.casefold() in resume_lower else languages_missing).append(name)

    stated = education["stated"] or bool(required_certs) or bool(required_languages)
    if not stated:
        return {
            "stated": False,
            "covered": True,
            "education": education,
            "education_satisfied": None,
            "certifications_required": [],
            "certifications_matched": [],
            "certifications_missing": [],
            "languages_required": [],
            "languages_matched": [],
            "languages_missing": [],
            "missing": [],
            "detail": (
                "The posting states no degree, certification or language "
                "requirement, so none was checked."
            ),
        }

    missing: list[str] = []
    if education["stated"] and not education_satisfied:
        missing.append(f"a {education['level']}-level qualification")
    missing.extend(f"the {name} certification" for name in certs_missing)
    missing.extend(f"{name} language ability" for name in languages_missing)

    parts = []
    if education["stated"]:
        parts.append(
            f"education: {'satisfied' if education_satisfied else 'not satisfied'} "
            f"(asks for {education['level']})"
        )
    if required_certs:
        parts.append(f"certifications: {len(certs_matched)}/{len(required_certs)} present")
    if required_languages:
        parts.append(f"languages: {len(languages_matched)}/{len(required_languages)} evidenced")

    return {
        "stated": True,
        "covered": not missing,
        "education": education,
        "education_satisfied": education_satisfied if education["stated"] else None,
        "certifications_required": required_certs,
        "certifications_matched": certs_matched,
        "certifications_missing": certs_missing,
        "languages_required": [entry["language"] for entry in required_languages],
        "languages_matched": languages_matched,
        "languages_missing": languages_missing,
        "missing": missing,
        "detail": "; ".join(parts)
        + (f". Not evidenced: {', '.join(missing)}." if missing else ". All evidenced."),
    }


# ---------------------------------------------------------------------------


def _score_required_skills(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    Coverage of what the posting actually requires.

    Three kinds of requirement, counted together because a recruiter treats them
    together: technical skills, and the structured qualifications — degree,
    named certifications, language ability — that keyword matching cannot see.
    A missing optional skill is a small deduction. A missing *required* item is
    the difference between being read and being screened out, so it costs twice as
    much.
    """
    jd_keywords = extract_keywords_from_jd(job_description)
    resume_lower = (resume_text or "").lower()
    qualifications = _score_structured_qualifications(resume_text, job_description)

    required_matched: list[str] = []
    required_missing: list[str] = []
    optional_matched: list[str] = []
    optional_missing: list[str] = []
    unspecified: list[str] = []

    for keyword in sorted(jd_keywords, key=str.lower):
        label = format_skill_name(keyword)
        present = is_skill_in_text(keyword, resume_lower)
        requirement = _classify_skill_requirement(job_description, keyword)

        if requirement == "required":
            (required_matched if present else required_missing).append(label)
        elif requirement == "optional":
            (optional_matched if present else optional_missing).append(label)
        else:
            unspecified.append(label)
            if not present:
                optional_missing.append(label)

    # The structured qualifications join the required pool as one unit each: a
    # posting's degree requirement is a single gate, not a set of separate items,
    # so it is counted once whether the candidate has it or not.
    qual_units = 0
    if qualifications["stated"]:
        if qualifications["education"]["stated"]:
            qual_units += 1
            if qualifications["education_satisfied"]:
                required_matched.append(
                    f"{qualifications['education']['level']}-level qualification"
                )
            else:
                required_missing.append(
                    f"{qualifications['education']['level']}-level qualification"
                )
        for name in qualifications["languages_matched"]:
            required_matched.append(f"{name} language ability")
        for name in qualifications["languages_missing"]:
            required_missing.append(f"{name} language ability")
        for name in qualifications["certifications_matched"]:
            required_matched.append(f"{name} certification")
        for name in qualifications["certifications_missing"]:
            required_missing.append(f"{name} certification")
        qual_units += (
            len(qualifications["languages_matched"])
            + len(qualifications["languages_missing"])
            + len(qualifications["certifications_matched"])
            + len(qualifications["certifications_missing"])
        )

    if not jd_keywords and not qual_units:
        return {
            "score": 100.0,
            "required_matched": [],
            "required_missing": [],
            "optional_matched": [],
            "optional_missing": [],
            "unclassified": [],
            "qualifications": qualifications,
            "explanation": (
                "No skills or qualifications could be extracted from the posting, "
                "so this check was skipped."
            ),
        }

    # Required items count twice as heavily as optional ones.
    earned = 2 * len(required_matched) + len(optional_matched) + len(unspecified)
    possible = (
        2 * (len(required_matched) + len(required_missing))
        + len(optional_matched)
        + len(optional_missing)
        + len(unspecified)
    )

    score = 100.0 * earned / possible if possible else 100.0

    explanation = (
        f"{len(required_matched)} of {len(required_matched) + len(required_missing)} "
        "required items are evidenced in the CV"
    )
    if len(optional_matched) + len(optional_missing):
        explanation += (
            f", and {len(optional_matched)} of "
            f"{len(optional_matched) + len(optional_missing)} preferred ones"
        )
    explanation += "."
    if qualifications["stated"]:
        explanation += " " + qualifications["detail"]
    if required_missing:
        explanation += f" Missing and required: {', '.join(required_missing[:6])}."

    return {
        "score": round(score, 1),
        "required_matched": required_matched,
        "required_missing": required_missing,
        "optional_matched": optional_matched,
        "optional_missing": optional_missing,
        "unclassified": unspecified,
        "qualifications": qualifications,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Category 3 — Experience Relevance
# ---------------------------------------------------------------------------


def _score_experience_relevance(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:
    """
    How closely the documented experience matches the role.

    Blends four independent signals so a single lexical coincidence cannot
    produce a confident number: whole-document lexical similarity, technical term
    overlap, soft-skill coverage, and years-of-experience against the posting's
    stated requirement. Two of them can miss entirely without the verdict moving
    much, which is the point of averaging rather than multiplying.
    """
    from app.services.analysis.ats_analyzer import calculate_context_similarity

    jd_keywords = extract_keywords_from_jd(job_description)
    resume_lower = (resume_text or "").lower()

    # -- signal 1: whole-document lexical similarity
    # Cosine similarity of TF-IDF vectors is near zero for genuinely unrelated
    # text and only approaches 0.4 for closely-related documents, so it is scaled
    # by 2.5 to use the 0-100 range. A CV sharing vocabulary with the posting
    # scores well here even where the specific terms differ, which is exactly the
    # signal keyword matching misses.
    raw_tfidf = calculate_context_similarity(resume_text, job_description)
    similarity_score = min(100.0, raw_tfidf * 2.5)

    # -- signal 2: technical term overlap, distinct from the categories that own it
    overlap_hits = sum(1 for kw in jd_keywords if is_skill_in_text(kw, resume_lower))
    overlap_score = 100.0 * overlap_hits / len(jd_keywords) if jd_keywords else 100.0

    # -- signal 3: context / soft-skill coverage
    context_requirements = extract_context_requirements(job_description)
    context_matched = [c for c in context_requirements if is_skill_in_text(c, resume_lower)]
    context_score = (
        100.0 * len(context_matched) / len(context_requirements) if context_requirements else 100.0
    )

    # -- signal 4: years of experience
    jd_years = _YEARS_RE.search(job_description)
    years_required = int(jd_years.group(1)) if jd_years else None
    years_claimed = [int(y) for y in _YEARS_RE.findall(resume_text or "")]
    years_max = max(years_claimed, default=None)

    if years_required is None:
        years_score: float | None = None
        years_note = "The posting does not state a number of years, so this was not scored."
    elif years_max is None:
        # The CV simply does not state a figure. Not evidence of too little
        # experience, so it is not penalised as if it were.
        years_score = None
        years_note = (
            "The CV does not state a total number of years, so the posting's "
            "years requirement could not be verified."
        )
    elif years_max >= years_required * YEARS_TOLERANCE:
        years_score = 100.0
        years_note = (
            f"The CV evidences at least {years_max} years; the posting asks for {years_required}."
        )
    else:
        # Scale the shortfall rather than scoring zero, so a CV one year short is
        # visibly close instead of indistinguishable from no experience at all.
        ratio = years_max / years_required if years_required else 0.0
        years_score = round(100.0 * ratio, 1)
        years_note = (
            f"The CV evidences about {years_max} years; the posting asks for "
            f"{years_required}. This is a real gap and cannot be closed by "
            "rewording."
        )

    signals = [similarity_score, overlap_score, context_score]
    if years_score is not None:
        signals.append(years_score)
    score = sum(signals) / len(signals)

    explanation = (
        f"The CV and the posting share {similarity_score:.0f}% of their vocabulary, "
        f"{overlap_score:.0f}% of the posting's technical terms appear in it, and "
        f"{len(context_matched)} of {len(context_requirements)} soft-skill "
        f"expectations are addressed. {years_note}"
    )

    return {
        "score": round(score, 1),
        "similarity_score": round(similarity_score, 1),
        "tfidf_similarity": round(raw_tfidf, 2),
        "overlap_score": round(overlap_score, 1),
        "context_score": round(context_score, 1),
        "context_matched": [format_skill_name(c) for c in context_matched],
        "context_missing": [
            format_skill_name(c) for c in sorted(context_requirements - set(context_matched))
        ],
        "years_required": years_required,
        "years_evidenced": years_max,
        "years_score": years_score,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Category 4 — CV Structure
# ---------------------------------------------------------------------------

#: The sections a parser needs in order to place content. Optional sections are
#: never demanded: forcing a Projects or Certifications block onto a candidate
#: who has neither is exactly the padding requirement 35 warns against.
_ESSENTIAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "contact": ("contact", "kontakt", "email", "@", "phone", "telefon"),
    "experience": (
        "experience",
        "employment",
        "work history",
        "berufserfahrung",
        "arbeitserfahrung",
    ),
    "education": (
        "education",
        "academic",
        "degree",
        "university",
        "ausbildung",
        "studium",
        "bildung",
    ),
    "skills": ("skills", "competenc", "technolog", "kenntnisse", "fähigkeiten", "fertigkeiten"),
}

_OPTIONAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "profile", "objective", "about", "profil", "zusammenfassung"),
    "projects": ("project", "portfolio", "projekt"),
    "certifications": ("certificat", "licen", "zertifikat", "zertifizierung"),
    "languages": ("language", "sprache", "sprachen"),
}

#: A CV shorter than this reads as empty to a parser as well as to a human.
MIN_STRUCTURE_CHARS = 300


def _score_cv_structure(resume_text: str) -> dict[str, Any]:
    """
    Section coverage, contactability, and formatting consistency.

    Two layers: the sections and contact facts a parser needs in order to place
    content at all, and the formatting conventions (consistent dates, consistent
    job titles, real bullet points, a heading hierarchy, no tables) that decide
    whether it can follow the content once placed. The formatting half comes from
    :mod:`formatting_checks` so the same rules apply to the generated PDF.
    """
    text = resume_text or ""
    text_lower = text.lower()

    found: list[str] = []
    missing: list[str] = []
    for section, markers in _ESSENTIAL_SECTIONS.items():
        (found if any(m in text_lower for m in markers) else missing).append(section)

    optional_found: list[str] = []
    for section, markers in _OPTIONAL_SECTIONS.items():
        if any(m in text_lower for m in markers):
            optional_found.append(section)

    checks: list[dict[str, Any]] = []
    for section in _ESSENTIAL_SECTIONS:
        checks.append(
            {
                "name": f"{section} section",
                "passed": section in found,
                "detail": (
                    "present"
                    if section in found
                    else "not detected — an ATS may not be able to place this content"
                ),
            }
        )

    long_enough = len(text) >= MIN_STRUCTURE_CHARS
    checks.append(
        {
            "name": "sufficient length",
            "passed": long_enough,
            "detail": f"{len(text):,} characters",
        }
    )

    # Formatting conventions, from the shared checker. Nothing is filtered out:
    # "contact section" and "contact details" look redundant but measure different
    # facts — whether the CV has a labelled contact block, and whether an email
    # address is actually present. A CV can have one without the other, and
    # dropping either hides a real gap.
    formatting = analyze_text_formatting(text)
    checks.extend(formatting["checks"])

    # Keyword stuffing. This is the one keyword problem keyword matching cannot
    # see: every repetition of a *matched* term looks like good coverage from the
    # outside, so a CV that repeats one term to fill a section scores well and
    # reads as padding. Requirement 24 asks for it explicitly.
    stuffing = find_keyword_stuffing(text)
    checks.append(
        {
            "name": "no keyword stuffing",
            "passed": not stuffing["stuffed"],
            "detail": stuffing["detail"],
        }
    )

    passed_count = sum(1 for check in checks if check["passed"])
    score = 100.0 * passed_count / len(checks)

    failed = [check["name"] for check in checks if not check["passed"]]
    explanation = (
        f"{passed_count} of {len(checks)} checks passed: the sections a parser needs "
        f"in order to place content, plus formatting conventions "
        f"(dates, job titles, company names, bullets, headings, measurable results)."
    )
    if failed:
        explanation += f" Not met: {', '.join(failed)}."
    if optional_found:
        explanation += f" Also present: {', '.join(optional_found)}."

    return {
        "score": round(score, 1),
        "sections_found": found,
        "missing_sections": missing,
        "optional_sections_found": optional_found,
        "checks": checks,
        "failed_checks": failed,
        "formatting": {
            "score": formatting["score"],
            "passed": formatting["passed"],
            "total": formatting["total"],
            "failed": formatting["failed"],
        },
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Category 5 — PDF Parsing
# ---------------------------------------------------------------------------


def _unmeasured_pdf_category(reason: str) -> dict[str, Any]:
    """The category as it stands before a PDF exists."""
    return {
        "score": None,
        "not_measured": True,
        "reason": reason,
        "explanation": (
            f"Not measured yet — {reason} The score below therefore covers the "
            "other four categories, weighted to fill 100%."
        ),
    }


def _score_pdf_parsing(
    pdf_bytes: bytes | None,
    expected_text: str = "",
) -> dict[str, Any]:
    """
    Whether the finished document can actually be read by a parser.

    This is the category that answers the question requirement 24 ends on: a PDF
    that compiles is not evidence that it is machine-readable. The checks live in
    :mod:`formatting_checks`, which works from glyph positions rather than a
    rendered image, so reading order, column structure, overlap and clipping are
    all decidable without a renderer.
    """
    if not pdf_bytes:
        return _unmeasured_pdf_category("no PDF has been generated yet")

    layout = analyze_pdf_layout(pdf_bytes, expected_text)

    retention = layout["content_retention"]
    failed = layout["failed"]

    explanation = (
        f"{layout['passed']} of {layout['total']} checks passed on the generated "
        f"document: {layout['text_chars']:,} characters recovered across "
        f"{layout['pages']} page(s)."
    )
    if retention is not None:
        explanation += f" {retention:.0%} of your CV's content survived typesetting."
    if layout["sections_found"]:
        explanation += f" Sections located: {', '.join(layout['sections_found'])}."
    if failed:
        explanation += f" Not met: {', '.join(failed)}."

    return {
        "score": round(layout["score"], 1),
        "not_measured": False,
        "reason": "",
        "text_chars": layout["text_chars"],
        "pages": layout["pages"],
        "content_retention": retention,
        "sections_detected": layout["sections_found"],
        "latex_artifacts": layout["latex_artifacts"],
        "replacement_chars": layout["replacement_chars"],
        "image_count": layout["image_count"],
        "checks": layout["checks"],
        "failed_checks": failed,
        "issues": [check["detail"] for check in layout["checks"] if not check["passed"]],
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _band(score: float) -> str:
    """Plain-language band. Mirrors the thresholds the dashboard uses."""
    if score >= 90:
        return "Excellent match"
    if score >= 75:
        return "Good match"
    if score >= 60:
        return "Moderate match"
    if score >= 40:
        return "Weak match"
    return "Poor match"


def compute_ats_breakdown(
    resume_text: str,
    job_description: str,
    pdf_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Score a CV against a posting and report where every point went.

    ``pdf_bytes`` is optional. Supplying it enables the PDF Parsing category;
    without it that category is reported as unmeasured and the other four carry
    the full weight. The result says which of the two happened.
    """
    resume_text = resume_text or ""
    job_description = job_description or ""

    raw_categories: dict[str, dict[str, Any]] = {
        "keyword_match": _score_keyword_match(resume_text, job_description),
        "required_skills": _score_required_skills(resume_text, job_description),
        "experience_relevance": _score_experience_relevance(resume_text, job_description),
        "cv_structure": _score_cv_structure(resume_text),
        "pdf_parsing": _score_pdf_parsing(pdf_bytes, resume_text),
    }

    measured = {k: v for k, v in raw_categories.items() if not v.get("not_measured")}
    unmeasured = {k: v for k, v in raw_categories.items() if v.get("not_measured")}

    total_weight = sum(_CATEGORY_WEIGHTS[k] for k in measured) or 1.0

    categories: dict[str, dict[str, Any]] = {}
    total = 0.0
    total_lost = 0.0

    for name, data in raw_categories.items():
        base_weight = _CATEGORY_WEIGHTS[name]

        if name in unmeasured:
            # Redistributed proportionally so the blend still reaches 100%.
            share = base_weight / total_weight if total_weight else 0.0
            categories[name] = {
                **data,
                "label": _CATEGORY_LABELS[name],
                "weight": round(share, 4),
                "weighted_points": 0.0,
                "points_lost": 0.0,
            }
            continue

        weight = base_weight / total_weight
        score = float(data["score"])
        weighted = score * weight
        lost = (100.0 - score) * weight

        total += weighted
        total_lost += lost

        categories[name] = {
            **data,
            "label": _CATEGORY_LABELS[name],
            "max_score": 100.0,
            "weight": round(weight, 4),
            "weighted_points": round(weighted, 2),
            "points_lost": round(lost, 2),
        }

    ats_score = round(min(100.0, max(0.0, total)), 1)

    # Where the points actually went, worst first.
    improvement_areas = [
        {
            "category": name,
            "label": _CATEGORY_LABELS[name],
            "score": categories[name]["score"],
            "points_lost": categories[name]["points_lost"],
            "explanation": categories[name]["explanation"],
        }
        for name in measured
        if categories[name]["points_lost"] > 0
    ]
    improvement_areas.sort(key=lambda area: area["points_lost"], reverse=True)

    # Required-skill gaps are called out on their own, because they are the one
    # class of loss the candidate usually cannot fix by editing.
    unfixable = categories.get("required_skills", {}).get("required_missing", [])
    years_gap = categories.get("experience_relevance", {}).get("years_score")

    # Aligned with _band() so the sentence and the label never disagree.
    if ats_score >= 90:
        summary = "Excellent match. Your CV is highly compatible with this posting."
    elif ats_score >= 75:
        summary = "Good match. Your CV fits this posting well, with some room to improve."
    elif ats_score >= 60:
        summary = "Moderate match. Your CV is a reasonable fit, with gaps worth closing."
    elif ats_score >= 40:
        summary = "Weak match. Your CV has significant gaps against this posting."
    else:
        summary = "Poor match. Your CV does not yet match this posting closely."

    if unfixable:
        summary += (
            f" {len(unfixable)} required skill(s) are not evidenced in the CV: "
            f"{', '.join(unfixable[:4])}."
            if len(unfixable) > 4
            else f" The following required skills are not evidenced: {', '.join(unfixable)}."
        )
    if years_gap is not None and years_gap < 100:
        summary += " The posting's years-of-experience requirement is also not yet evidenced."

    if unmeasured:
        summary += (
            " This score is pre-generation: document readability has not been "
            "measured yet, so the other four checks carry the full weight."
        )

    return {
        "ats_score": ats_score,
        "max_score": 100.0,
        "band": _band(ats_score),
        "total_points_lost": round(total_lost, 2),
        "summary": summary,
        "categories": categories,
        "improvement_areas": improvement_areas,
        "unfillable_gaps": {
            "required_skills_missing": unfixable,
            "years_of_experience_short": years_gap is not None and years_gap < 100,
        },
        "pre_generation": bool(unmeasured),
        "is_honest_score": True,
    }


# ---------------------------------------------------------------------------
# User-facing phrasing
# ---------------------------------------------------------------------------

_PHRASING: tuple[tuple[str, str, str], ...] = (
    (
        "keyword_match",
        "Some important words from the job description are missing from your CV.",
        "If you have experience with them, mention them where you actually used them. "
        "If you do not, this is a genuine gap.",
    ),
    (
        "required_skills",
        "One or more required skills are not clearly shown in your CV.",
        "Check that every requirement in the posting is either evidenced in your "
        "experience or consciously left out.",
    ),
    (
        "experience_relevance",
        "Your experience is not closely aligned with this role yet.",
        "Reorder your experience so the most relevant work appears first, and describe "
        "it in the language the posting uses.",
    ),
    (
        "cv_structure",
        "One section of your CV could not be clearly identified.",
        "Use standard headings such as 'Experience', 'Education' and 'Skills', and make "
        "sure your email address is present.",
    ),
    (
        "pdf_parsing",
        "The document contains information that an ATS may have difficulty reading.",
        "Regenerate the CV, or choose a simpler single-column layout.",
    ),
)


#: Individual formatting failures, each with the specific fix. Keyed by the check
#: name :mod:`formatting_checks` reports, so a new check there surfaces here
#: automatically rather than being silently dropped from the user's view.
_FORMATTING_ACTIONS: dict[str, str] = {
    "consistent dates": (
        "Use one date format throughout, for example 'Jan 2020 - Present' or "
        "'2020-01 - Present'."
    ),
    "consistent job titles": (
        "Write each job title the same way every time — 'Senior Engineer' or "
        "'Sr. Engineer', not both."
    ),
    "consistent company names": (
        "Name each employer one way and keep that spelling, including 'Ltd' or " "'Inc'."
    ),
    "standard bullet points": (
        "Present your accomplishments as a list, with the same bullet character " "on every line."
    ),
    "heading hierarchy": (
        "Give each section a plain heading such as 'Experience', 'Education' or "
        "'Skills' rather than a graphic or a stylised font."
    ),
    "measurable achievements": (
        "Say what the work achieved, with a number: a percentage change, a volume, "
        "a time saving."
    ),
    "no parsing-hostile tables": (
        "Replace tables with a list. A table's reading order is ambiguous, so the "
        "cells come back in the wrong sequence."
    ),
    "no decorative icon headers": (
        "Use text headings rather than icons — an icon carries nothing a reader "
        "or a parser can interpret."
    ),
    "contact details": "Add an email address so a recruiter can reply.",
}

#: Failures that are properties of the *rendered* document rather than the source.
#: Each says what to do, because "the PDF has a problem" alone is not actionable.
_PDF_ACTIONS: dict[str, str] = {
    "text extractable": (
        "The document has almost no selectable text. Try generating it again, or "
        "choose a different layout."
    ),
    "no image-only content": (
        "Some pages have no text at all. Move anything shown as a picture into " "ordinary text."
    ),
    "no raw markup": "This is a generation fault. Regenerate the document.",
    "all characters rendered": (
        "Some characters could not be displayed. Check for special symbols, and "
        "remove any that are not needed."
    ),
    "no symbol-substituted text": (
        "Some headings use a decorative font whose letters extract as unreadable "
        "symbols. Use a normal font for text."
    ),
    "no invisible text": (
        "Part of the document is hidden text. Remove it — a reader and a parser "
        "will disagree about what the CV says."
    ),
    "single column": (
        "This layout sets the text in two columns. Most parsers read across rather "
        "than down, so the lines come back interleaved. Choose a single-column "
        "layout."
    ),
    "reading order": (
        "The text does not come out in the order it appears. A two-column layout is "
        "the usual cause; a single-column one fixes it."
    ),
    "no overlapping text": (
        "Some lines sit on top of each other and merge when read. This is usually a "
        "text box over a heading; remove the box or the overlap."
    ),
    "no clipped text": (
        "Some text runs to the edge of the page and may be cut off. Shorten the "
        "line or widen the margin."
    ),
    "no content in page margins": (
        "Some important-looking text sits in a page margin. Page margins are the "
        "first thing a reader loses or reorders, so move that content into the body."
    ),
    "no text boxes or panels": (
        "This layout is built from coloured panels rather than from plain text. A "
        "parser reads a panel as a box and loses the order of what is inside it. "
        "Choose a single-column, text-only layout."
    ),
    "readable type size": (
        "Most of the text is set too small to read comfortably. This usually means "
        "a layout is cramming content onto one page — try a roomier layout, or "
        "shorten the CV."
    ),
    "headings locatable": (
        "Section headings could not be found in the document. Use plain headings "
        "such as 'Experience' and 'Skills'."
    ),
    "no duplicated sections": (
        "A section appears twice, which reads as repeated content. This is a "
        "generation fault; regenerate the document."
    ),
    "content retained": (
        "Some of your CV was lost while producing the document. Try a simpler "
        "layout, which leaves more room for the text."
    ),
}


def generate_user_friendly_suggestions(
    ats_breakdown: dict[str, Any],
    missing_skills: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Turn the breakdown into plain language, with an action for every item.

    The technical categories stay available; this is the version meant to be read
    by someone who does not know what an ATS is. Three layers, most specific
    first, so the advice names the actual cause rather than the category:

    * an exact fact the candidate can act on — a mismatched job title, a required
      skill that is absent, a specific formatting failure;
    * otherwise the category-level phrasing;
    * and never a bare failure without a next step, because "your document has a
      problem" is not something anyone can act on.
    """
    suggestions: list[dict[str, str]] = []
    categories = ats_breakdown.get("categories") or {}

    # -- 1. specific, fact-level findings --------------------------------
    title = (categories.get("keyword_match") or {}).get("title_alignment") or {}
    if title.get("stated") and not title.get("aligned"):
        suggestions.append(
            {
                "category": "job_title",
                "issue": (
                    f"Your CV is headed '{title.get('candidate') or 'untitled'}', but "
                    f"this job is for a '{title.get('target')}'."
                ),
                "action": (
                    f"If that title describes you accurately, use "
                    f"'{title.get('target')}' as your headline. If it does not, "
                    "leave it as it is — an accurate title is better than a "
                    "borrowed one."
                ),
                "severity": "medium",
            }
        )

    logistics = (categories.get("keyword_match") or {}).get("logistics") or {}
    if logistics.get("stated") and not logistics.get("covered"):
        suggestions.append(
            {
                "category": "location_and_work_authorisation",
                "issue": (
                    "This job states where and on what terms you would work, and "
                    "your CV does not say: " + "; ".join(logistics.get("missing") or []) + "."
                ),
                "action": (
                    "Add a line stating your situation honestly — for example "
                    "'Based in London, open to remote', or 'Based in Berlin, require "
                    "visa sponsorship'. A recruiter often cannot progress an "
                    "application without knowing this."
                ),
                "severity": "medium",
            }
        )

    qualifications = (categories.get("required_skills") or {}).get("qualifications") or {}
    if qualifications.get("stated") and qualifications.get("missing"):
        suggestions.append(
            {
                "category": "qualifications",
                "issue": (
                    "This job asks for something your CV does not show: "
                    + "; ".join(qualifications["missing"])
                    + "."
                ),
                "action": (
                    "If you have it, add it under the name it actually carries — "
                    "'BSc', 'German (C1)', 'CKA'. If you do not have it, this is a "
                    "qualification gap, and the honest answer is that it may not be "
                    "the right role for you yet."
                ),
                "severity": "high",
            }
        )

    industry = (categories.get("keyword_match") or {}).get("industry_terminology") or {}
    if industry.get("missing"):
        suggestions.append(
            {
                "category": "industry_terminology",
                "issue": (
                    "This field works with terms your CV does not mention: "
                    + ", ".join(industry["missing"][:6])
                    + "."
                ),
                "action": (
                    "These are the frameworks and standards the industry works within. "
                    "If you have used them, say where — in this field it often decides "
                    "whether your CV is read by someone who understands it."
                ),
                "severity": "medium",
            }
        )

    structure_checks = {
        check.get("name"): check
        for check in (categories.get("cv_structure") or {}).get("checks") or []
    }
    stuffing = structure_checks.get("no keyword stuffing")
    if stuffing and not stuffing.get("passed"):
        suggestions.append(
            {
                "category": "keyword_stuffing",
                "issue": "Some terms are repeated far more often than they add information.",
                "action": (
                    "Replace the repetition with what you actually did. Naming a tool "
                    "four times reads as padding; one specific sentence about it does not."
                ),
                "severity": "medium",
            }
        )

    required = categories.get("required_skills") or {}
    if required.get("required_missing"):
        names = required["required_missing"]
        suggestions.append(
            {
                "category": "required_skills_missing",
                "issue": (
                    f"{len(names)} skill(s) the job requires are not shown in your CV: "
                    f"{', '.join(names[:6])}"
                    + (f" and {len(names) - 6} more." if len(names) > 6 else ".")
                ),
                "action": (
                    "Check each one against what you have actually done. Add the ones "
                    "you have and left out. The ones you do not have are a real gap, "
                    "and no amount of editing will close them."
                ),
                "severity": "high",
            }
        )

    experience = categories.get("experience_relevance") or {}
    if experience.get("years_score") is not None and experience["years_score"] < 100:
        suggestions.append(
            {
                "category": "years_of_experience",
                "issue": (
                    f"The job asks for {experience.get('years_required')} years of "
                    f"experience; your CV shows about {experience.get('years_evidenced')}."
                ),
                "action": (
                    "This gap cannot be closed by rewording. Make sure the years you "
                    "have are stated clearly, then decide whether the role is a "
                    "realistic target."
                ),
                "severity": "high",
            }
        )

    # -- 2. the specific checks behind the formatting categories ---------
    for category_key, action_map in (
        ("cv_structure", _FORMATTING_ACTIONS),
        ("pdf_parsing", _PDF_ACTIONS),
    ):
        data = categories.get(category_key) or {}
        if data.get("not_measured"):
            continue
        for check in data.get("checks") or []:
            if check.get("passed"):
                continue
            name = check.get("name", "")
            if name not in action_map:
                continue
            suggestions.append(
                {
                    "category": f"{category_key}:{name}",
                    "issue": f"{name.replace('_', ' ').capitalize()}: "
                    f"{check.get('detail', '')}",
                    "action": action_map[name],
                    "severity": "high" if category_key == "pdf_parsing" else "medium",
                }
            )

    # -- 3. category-level phrasing, where no specific finding explains it
    for key, issue, action in _PHRASING:
        data = categories.get(key) or {}
        if data.get("not_measured"):
            continue
        score = data.get("score")
        if score is None or score >= IMPROVEMENT_THRESHOLD:
            continue
        # Already covered by a specific finding above.
        if any(s["category"].split(":")[0] == key for s in suggestions):
            continue
        suggestions.append(
            {
                "category": key,
                "issue": issue,
                "action": action,
                "severity": "high" if score < 60 else "medium",
            }
        )

    missing = [str(s) for s in (missing_skills or [])]
    if missing:
        suggestions.append(
            {
                "category": "missing_skills",
                "issue": (
                    f"{len(missing)} term(s) from the posting do not appear anywhere in "
                    f"your CV: {', '.join(missing[:6])}."
                    + (f" and {len(missing) - 6} more." if len(missing) > 6 else "")
                ),
                "action": (
                    "Add any of these you genuinely have experience with. Do not add "
                    "the ones you do not — an interviewer will ask."
                ),
                "severity": "high" if len(missing) > 3 else "medium",
            }
        )

    return suggestions


def get_ats_summary(ats_breakdown: dict[str, Any]) -> dict[str, Any]:
    """Compact summary for the dashboard header."""
    score = float(ats_breakdown.get("ats_score", 0.0))
    categories = ats_breakdown.get("categories") or {}

    working = [
        (data.get("label") or name)
        for name, data in categories.items()
        if not data.get("not_measured") and (data.get("score") or 0) >= IMPROVEMENT_THRESHOLD
    ]
    improve = [
        (data.get("label") or name)
        for name, data in categories.items()
        if not data.get("not_measured") and (data.get("score") or 0) < IMPROVEMENT_THRESHOLD
    ]

    if score >= 90:
        primary_action = "Your CV is ready. Generate it with the recommended layout."
    elif improve:
        primary_action = f"Work on {improve[0].lower()} — that is where the most points are."
    else:
        primary_action = "Review the details below, then generate your CV."

    return {
        "score": score,
        "band": ats_breakdown.get("band") or _band(score),
        "what_is_working": working,
        "what_can_improve": improve,
        "primary_action": primary_action,
        "total_points_lost": ats_breakdown.get("total_points_lost", 0.0),
        "pre_generation": bool(ats_breakdown.get("pre_generation")),
    }


__all__ = [
    "IMPROVEMENT_THRESHOLD",
    "MIN_CONTENT_RETENTION",
    "MIN_PDF_TEXT_CHARS",
    "compute_ats_breakdown",
    "generate_user_friendly_suggestions",
    "get_ats_summary",
    "get_ats_summary",
]
