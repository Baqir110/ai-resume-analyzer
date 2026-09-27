# app/services/jobs/description_parser.py
"""Best-effort, bounded parsing of job-description sections.

Discovery sources commonly return one HTML/text blob rather than the
structured responsibility and requirement fields on ``NormalizedJob``.  This
module recognizes a conservative set of English and German headings and
extracts the lines below them without making an LLM call.

The parser is intentionally heuristic and dependency-free.  Input, line,
item, and item-length limits keep remote job descriptions from consuming
unbounded CPU or memory during discovery.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

# Public so callers/tests can understand the fail-closed bounds.
MAX_DESCRIPTION_CHARS = 200_000
MAX_DESCRIPTION_LINES = 5_000
MAX_ITEMS_PER_SECTION = 200
MAX_ITEM_CHARS = 2_000

_RESPONSIBILITY_HEADERS = re.compile(
    r"^(?:responsibilities|what you.?ll do|your role|your tasks|the role|"
    r"aufgaben|ihre aufgaben|deine aufgaben)\s*:?[\s.]*$",
    re.IGNORECASE,
)
_REQUIREMENT_HEADERS = re.compile(
    r"^(?:requirements(?:\s*(?:&|and)\s*qualifications)?|qualifications|"
    r"what you.?ll need|must have|your profile|anforderungen(?:\s*&\s*qualifikationen)?|"
    r"ihr profil|dein profil|voraussetzungen)\s*:?[\s.]*$",
    re.IGNORECASE,
)
_PREFERRED_HEADERS = re.compile(
    r"^(?:nice to have|preferred|bonus points|good to have|pluspunkte|von vorteil|"
    r"wünschenswert)\s*:?[\s.]*$",
    re.IGNORECASE,
)
_BULLET_PREFIX = re.compile(r"^(?:[-•*▪●○\u2022]\s*|\d+[.)]\s*|[-–—]\s+)")
_MD_HEADING_PREFIX = re.compile(r"^#{1,6}\s+")
_MD_BOLD_HEADER = re.compile(r"^\*\*(.+?)\*\*:?$")


def _clean_line(line: str) -> str:
    """Remove one list marker and enforce a hard per-item length limit."""
    line = _BULLET_PREFIX.sub("", line, count=1).strip()
    return line[:MAX_ITEM_CHARS]


def _strip_html(text: str) -> str:
    """Convert the small HTML subset used by job boards into plain lines.

    ``html.unescape`` is intentionally done before tag removal so headings
    such as ``<strong>Requirements</strong>`` and entities survive the same
    way as plain text.  The caller has already enforced the input bound.
    """
    text = html.unescape(text)
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</(?:p|li|div|section|article|ul|ol|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<li\b[^>]*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<(?:p|div|section|article|ul|ol|tr|h[1-6])\b[^>]*>", "\n", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", "", text)
    return text


@dataclass
class ParsedSections:
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    preferred_requirements: list[str] = field(default_factory=list)
    matched_headers: bool = False


def parse_description_sections(description: str) -> ParsedSections:
    """Parse recognized sections, failing closed when input exceeds bounds.

    Oversized/non-string input returns an empty result rather than partially
    parsing attacker-controlled text.  Each returned section is independently
    capped, so the complete result remains bounded even for repeated headings.
    """
    if not isinstance(description, str) or not description.strip():
        return ParsedSections()
    if len(description) > MAX_DESCRIPTION_CHARS:
        return ParsedSections()

    text = _strip_html(description)
    # ``splitlines`` handles LF, CRLF, CR, and Unicode line separators.  It
    # also fixes the original bug where a literal backtick-n was split.
    lines = text.splitlines()[:MAX_DESCRIPTION_LINES]

    result = ParsedSections()
    current_bucket: list[str] | None = None
    any_header_found = False

    for raw_line in lines:
        line = _MD_HEADING_PREFIX.sub("", raw_line.strip())
        line = _MD_BOLD_HEADER.sub(r"\1", line)
        if not line:
            continue

        if _RESPONSIBILITY_HEADERS.match(line):
            current_bucket = result.responsibilities
            any_header_found = True
            continue
        if _REQUIREMENT_HEADERS.match(line):
            current_bucket = result.requirements
            any_header_found = True
            continue
        if _PREFERRED_HEADERS.match(line):
            current_bucket = result.preferred_requirements
            any_header_found = True
            continue

        if current_bucket is not None and len(current_bucket) < MAX_ITEMS_PER_SECTION:
            cleaned = _clean_line(line)
            if cleaned:
                current_bucket.append(cleaned)

    result.matched_headers = any_header_found
    return result


def enrich_job_from_description(job) -> None:
    """Fill empty source fields from recognizable description sections.

    Existing source data is never overwritten.  If no heading is recognized,
    the job remains untouched instead of receiving a guessed requirement
    bucket.
    """
    if job.requirements or job.preferred_requirements or job.responsibilities:
        return

    parsed = parse_description_sections(job.description)
    if not parsed.matched_headers:
        return

    job.responsibilities = parsed.responsibilities
    job.requirements = parsed.requirements
    job.preferred_requirements = parsed.preferred_requirements


__all__ = [
    "MAX_DESCRIPTION_CHARS",
    "MAX_DESCRIPTION_LINES",
    "MAX_ITEMS_PER_SECTION",
    "MAX_ITEM_CHARS",
    "ParsedSections",
    "parse_description_sections",
    "enrich_job_from_description",
]
