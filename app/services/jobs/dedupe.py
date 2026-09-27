# app/services/jobs/dedupe.py
"""Cross-source job deduplication.

The same posting routinely appears on Indeed, LinkedIn, Google Jobs and
the company's own Greenhouse/Lever board. This module collapses those
into a single NormalizedJob so the agent never applies twice.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.services.jobs.agent_schemas import NormalizedJob, fingerprint_job

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_COMPANY_SUFFIXES = re.compile(
    r"\b(gmbh|gmbh & co\.? kg|ag|inc\.?|incorporated|llc|ltd\.?|limited|"
    r"corp\.?|corporation|s\.?a\.?|s\.?r\.?l\.?|plc|co\.?)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# Tracking params that don't affect what page loads, so they must not
# make two identical job URLs look like different jobs.
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "referer",
    "referrer",
    "gh_src",
    "trk",
    "trkInfo",
    "src",
    "fbclid",
    "gclid",
    "from",
}

_TITLE_SYNONYMS = [
    (re.compile(r"\bsr\.?\b"), "senior"),
    (re.compile(r"\bjr\.?\b"), "junior"),
    (re.compile(r"\bswe\b"), "software engineer"),
    (re.compile(r"\(m/w/d\)|\(w/m/d\)|\(d/m/w\)"), ""),
    (re.compile(r"\bm/f/d\b"), ""),
]


def normalize_company(name: str) -> str:
    name = name or ""
    name = _COMPANY_SUFFIXES.sub("", name)
    name = _PUNCT.sub(" ", name)
    return _WS.sub(" ", name).strip().lower()


def normalize_title(title: str) -> str:
    title = (title or "").lower()
    for pattern, replacement in _TITLE_SYNONYMS:
        title = pattern.sub(replacement, title)
    title = re.sub(r"\([^)]*\)", " ", title)  # drop parenthetical notes
    title = _PUNCT.sub(" ", title)
    return _WS.sub(" ", title).strip()


def normalize_location(location: str) -> str:
    location = (location or "").lower()
    location = re.sub(r"\bgermany\b|\bdeutschland\b", "", location)
    location = _PUNCT.sub(" ", location)
    return _WS.sub(" ", location).strip()


def canonicalize_url(url: str) -> str:
    """Strip tracking params and normalize scheme/host casing so the
    same application URL from two sources hashes identically."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip().lower()

    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query.sort()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), netloc, path, "", urlencode(query), ""))


def compute_fingerprint(job: NormalizedJob) -> str:
    company = normalize_company(job.company)
    title = normalize_title(job.title)
    location = normalize_location(job.location)
    url = canonicalize_url(job.application_url or job.original_url)
    # URL is the strongest signal (same posting, same source-of-truth
    # link); fall back to company+title+location when URLs differ
    # across boards (e.g. Indeed mirror vs. company career page).
    return fingerprint_job(company, title, location, url=url)


def compute_dedupe_key(job: NormalizedJob) -> str:
    """Cross-board identity used alongside the primary URL fingerprint."""
    return fingerprint_job(
        normalize_company(job.company),
        normalize_title(job.title),
        normalize_location(job.location),
    )


def _hash(value: str) -> str:
    """Persist the same full SHA-256 URL fingerprint used by the schema."""
    return fingerprint_job("", "", "", url=value)


# Source priority: lower index = higher priority (kept when semantically duplicate)
_SOURCE_PRIORITY = [
    "greenhouse",
    "lever",
    "ashby",
    "workday",
    "bamboohr",
    "jobspy",
    "indeed",
    "linkedin",
    "web_search",
]


def source_rank(source: str) -> int:
    """Lower rank = higher priority source. Unknown sources ranked last."""
    s = (source or "").lower()
    for i, name in enumerate(_SOURCE_PRIORITY):
        if name in s:
            return i
    return len(_SOURCE_PRIORITY)


def semantic_dedup(jobs: list[NormalizedJob], threshold: float = 0.85) -> list[NormalizedJob]:
    """Second-pass semantic dedup using TF-IDF cosine similarity.

    Among jobs that survived fingerprint dedup, removes near-duplicate
    postings only when they share a stable ATS/job identifier as well as a
    sufficiently similar description.  Company/title/location similarity
    alone never merges distinct requisitions.

    If sklearn is not installed the function returns *jobs* unchanged.
    """
    if not _SKLEARN_AVAILABLE or len(jobs) < 2:
        return jobs

    descriptions = [j.description or "" for j in jobs]
    companies = [normalize_company(j.company) for j in jobs]

    # Build TF-IDF matrix once; empty descriptions get zero vectors.
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
    except ValueError:
        # All documents are empty / only stop-words -- nothing to compare.
        return jobs

    sim_matrix = cosine_similarity(tfidf_matrix)

    # Mark jobs to drop (index -> True means "drop this one").
    drop: set[int] = set()
    for i in range(len(jobs)):
        if i in drop:
            continue
        for j in range(i + 1, len(jobs)):
            if j in drop:
                continue
            if companies[i] != companies[j]:
                continue
            # Similar descriptions alone are insufficient: distinct postings
            # commonly share a title and location.  Require a shared stable
            # ATS/job identifier before semantic merging.
            left_id = (jobs[i].job_id or "").strip().casefold()
            right_id = (jobs[j].job_id or "").strip().casefold()
            if not left_id or left_id != right_id:
                continue
            if sim_matrix[i, j] >= threshold:
                # Keep the higher-priority source; ties go to the first-seen (i).
                if source_rank(jobs[j].source) < source_rank(jobs[i].source):
                    drop.add(i)
                    break  # i is dropped; no point comparing it further
                else:
                    drop.add(j)

    kept = []
    for idx, job in enumerate(jobs):
        if idx in drop:
            job.status = "DUPLICATE"
        else:
            kept.append(job)
    return kept


class Deduplicator:
    """Stateful dedup pass across one or more discovery batches.

    Usage:
        dedupe = Deduplicator(existing_fingerprints=tracker.known_fingerprints())
        unique_jobs = dedupe.dedupe_batch(all_discovered_jobs)
    """

    def __init__(
        self,
        existing_fingerprints: set[str] | None = None,
        existing_soft_fingerprints: set[str] | None = None,
    ):
        self._existing_fingerprints = set(existing_fingerprints or set())
        self._seen_urls: set[str] = set()
        self._seen_soft = set(existing_soft_fingerprints or set())

    def _soft_key(self, job: NormalizedJob) -> str:
        return compute_dedupe_key(job)

    def is_duplicate(self, job: NormalizedJob) -> bool:
        job.fingerprint = job.fingerprint or compute_fingerprint(job)
        if job.fingerprint in self._existing_fingerprints:
            return True

        url_key = canonicalize_url(job.application_url or job.original_url)
        if url_key and _hash(url_key) in self._seen_urls:
            return True
        # Company/title/location is only a candidate signal.  It must not
        # suppress a distinct requisition with a different ATS URL.
        return False

    def register(self, job: NormalizedJob) -> None:
        job.fingerprint = job.fingerprint or compute_fingerprint(job)
        url_key = canonicalize_url(job.application_url or job.original_url)
        if url_key:
            self._seen_urls.add(_hash(url_key))
        self._seen_soft.add(self._soft_key(job))

    def dedupe_batch(self, jobs: list[NormalizedJob]) -> list[NormalizedJob]:
        """Returns unique jobs, preferring the most complete record
        (longest description, i.e. richest source) when duplicates
        collide within the same batch."""
        groups: dict[str, NormalizedJob] = {}
        order: list[str] = []
        batch_urls: set[str] = set()

        for job in jobs:
            job.fingerprint = compute_fingerprint(job)
            url_key = canonicalize_url(job.application_url or job.original_url)
            identity_key = job.fingerprint or url_key
            if self.is_duplicate(job) or (url_key and url_key in batch_urls):
                job.status = "DUPLICATE"
                continue

            existing = groups.get(identity_key)
            if existing is None:
                groups[identity_key] = job
                order.append(identity_key)
            elif len(job.description) > len(existing.description):
                groups[identity_key] = job  # richer record for the same URL
            if url_key:
                batch_urls.add(url_key)

        unique = [groups[k] for k in order]
        unique = semantic_dedup(unique)
        for job in unique:
            self.register(job)
        return unique


__all__ = [
    "normalize_company",
    "normalize_title",
    "normalize_location",
    "canonicalize_url",
    "compute_fingerprint",
    "compute_dedupe_key",
    "source_rank",
    "Deduplicator",
    "semantic_dedup",
]
