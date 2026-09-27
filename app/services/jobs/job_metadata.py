"""Extract job metadata (company, role, JD) from an ATS job URL.

Instead of scraping HTML, this calls the ATS platform's own public JSON
API. Every major ATS exposes one — they use it to render the "view open
positions" widget on their customer's career sites, so the contract is
stable and doesn't require authentication.

Supported:
  Greenhouse  job-boards.greenhouse.io/{board}/jobs/{id}
  Lever       jobs.lever.co/{company}/{id}
  Ashby       jobs.ashbyhq.com/{board}/{id}
  Workable    apply.workable.com/{account}/j/{shortcode}

For unknown URLs, falls back to HTML extraction via jd_fetcher.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from app.core.network import public_async_client
from app.core.security import UnsafeInputError, validate_public_http_url

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class JobMetadata:
    company: str
    role: str
    jd_text: str
    location: str = ""
    source_ats: str = ""
    source_url: str = ""
    raw_title: str = ""


# ---------------------------------------------------------------------------
# URL pattern detection
# ---------------------------------------------------------------------------

_GREENHOUSE_RE = re.compile(
    r"(?:job-boards|boards)\.greenhouse\.io/([^/]+)/jobs/(\d+)",
    re.IGNORECASE,
)
_LEVER_RE = re.compile(
    r"jobs\.lever\.co/([^/]+)/([a-f0-9\-]+)",
    re.IGNORECASE,
)
_ASHBY_RE = re.compile(
    r"jobs\.ashbyhq\.com/([^/]+)/([a-f0-9\-]+)",
    re.IGNORECASE,
)
_WORKABLE_RE = re.compile(
    r"apply\.workable\.com/([^/]+)/j/([A-Z0-9]+)",
    re.IGNORECASE,
)


def _titlecase_token(token: str) -> str:
    """liveperson -> Liveperson, hexagon-helix -> Hexagon Helix."""
    return " ".join(part.capitalize() for part in re.split(r"[-_]", token) if part)


def _strip_html(html_text: str) -> str:
    """Convert the JD HTML into readable plain text."""
    if not html_text:
        return ""
    text = html.unescape(html_text)
    # Convert common block tags to newlines
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"(?i)</li>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "• ", text)
    text = re.sub(r"(?i)</h[1-6]>", "\n\n", text)
    # Strip the rest
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def _get_public(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_redirects: int = 4,
) -> httpx.Response:
    """GET a public URL while validating every redirect target."""

    current = url
    for _ in range(max_redirects + 1):
        try:
            validate_public_http_url(current, resolve_dns=True)
        except UnsafeInputError as exc:
            raise RuntimeError(f"URL is not allowed: {exc}") from exc
        response = await client.get(current)
        try:
            content_length = int(response.headers.get("content-length", "0"))
        except ValueError:
            content_length = 0
        if content_length > 8 * 1024 * 1024:
            raise RuntimeError("Fetched job metadata response is too large")
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise RuntimeError("Redirect response did not include a location")
            current = urljoin(current, location)
            continue
        return response
    raise RuntimeError("Too many redirects while fetching job metadata")


# ---------------------------------------------------------------------------
# ATS-specific extractors
# ---------------------------------------------------------------------------


async def _fetch_greenhouse(client: httpx.AsyncClient, board: str, job_id: str) -> JobMetadata:
    api = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
    resp = await _get_public(client, api)
    if resp.status_code != 200:
        raise RuntimeError(f"Greenhouse API returned HTTP {resp.status_code} for {api}")
    data = resp.json()

    title = (data.get("title") or "").strip()
    content_html = data.get("content") or ""
    location = ((data.get("location") or {}).get("name") or "").strip()

    # The board token is the company slug. Try to fetch a nicer display
    # name from the board endpoint; fall back to title-casing the slug.
    company = _titlecase_token(board)
    try:
        board_resp = await _get_public(
            client, f"https://boards-api.greenhouse.io/v1/boards/{board}"
        )
        if board_resp.status_code == 200:
            board_data = board_resp.json()
            # Some boards expose company_name; many don't. Best effort.
            company = (board_data.get("company_name") or board_data.get("name") or company).strip()
    except Exception:
        pass

    return JobMetadata(
        company=company,
        role=title,
        jd_text=_strip_html(content_html),
        location=location,
        source_ats="greenhouse",
        source_url=f"https://job-boards.greenhouse.io/{board}/jobs/{job_id}",
        raw_title=title,
    )


async def _fetch_lever(
    client: httpx.AsyncClient, company_slug: str, posting_id: str
) -> JobMetadata:
    api = f"https://api.lever.co/v0/postings/{company_slug}/{posting_id}"
    resp = await _get_public(client, api)
    if resp.status_code != 200:
        raise RuntimeError(f"Lever API returned HTTP {resp.status_code} for {api}")
    data = resp.json()

    title = (data.get("text") or "").strip()
    description = data.get("descriptionPlain") or data.get("description") or ""
    if data.get("descriptionPlain"):
        jd_text = description.strip()
    else:
        jd_text = _strip_html(description)

    # Append additional content (lists, closing, etc.) if present
    for key in ("lists", "additional"):
        for item in data.get(key) or []:
            header = item.get("text") or ""
            body_html = item.get("content") or ""
            body = _strip_html(body_html)
            if header:
                jd_text += f"\n\n{header}\n"
            if body:
                jd_text += body + "\n"

    location = ((data.get("categories") or {}).get("location") or "").strip()

    return JobMetadata(
        company=_titlecase_token(company_slug),
        role=title,
        jd_text=jd_text.strip(),
        location=location,
        source_ats="lever",
        source_url=f"https://jobs.lever.co/{company_slug}/{posting_id}",
        raw_title=title,
    )


async def _fetch_ashby(client: httpx.AsyncClient, board: str, posting_id: str) -> JobMetadata:
    # Ashby's public posting API returns the whole board; we filter by id.
    api = f"https://api.ashbyhq.com/posting-api/job-board/{board}"
    resp = await _get_public(client, api)
    if resp.status_code != 200:
        raise RuntimeError(f"Ashby API returned HTTP {resp.status_code} for {api}")
    data = resp.json()
    jobs = data.get("jobs") or []

    match = None
    for job in jobs:
        if str(job.get("id")) == posting_id:
            match = job
            break

    if match is None:
        raise RuntimeError(f"No job found for id {posting_id} on Ashby board {board}")

    title = (match.get("title") or "").strip()
    jd_html = match.get("descriptionHtml") or match.get("description") or ""
    jd_text = _strip_html(jd_html)
    location = (match.get("location") or "").strip()

    return JobMetadata(
        company=_titlecase_token(board),
        role=title,
        jd_text=jd_text,
        location=location,
        source_ats="ashby",
        source_url=f"https://jobs.ashbyhq.com/{board}/{posting_id}",
        raw_title=title,
    )


async def _fetch_workable(client: httpx.AsyncClient, account: str, shortcode: str) -> JobMetadata:
    api = f"https://apply.workable.com/api/v1/widget/accounts/{account}"
    resp = await _get_public(client, api)
    if resp.status_code != 200:
        raise RuntimeError(f"Workable API returned HTTP {resp.status_code} for {api}")
    data = resp.json()
    jobs = data.get("jobs") or []

    match = None
    for job in jobs:
        if str(job.get("shortcode")) == shortcode:
            match = job
            break

    if match is None:
        raise RuntimeError(f"No job found for shortcode {shortcode} on Workable account {account}")

    title = (match.get("title") or "").strip()
    jd_html = match.get("description") or ""
    jd_text = _strip_html(jd_html)
    location = match.get("city") or match.get("location") or ""
    if isinstance(location, dict):
        location = location.get("city") or location.get("name") or ""

    return JobMetadata(
        company=_titlecase_token(account),
        role=title,
        jd_text=jd_text,
        location=str(location).strip(),
        source_ats="workable",
        source_url=f"https://apply.workable.com/{account}/j/{shortcode}",
        raw_title=title,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_job_metadata(url: str, timeout: float = 20.0) -> JobMetadata:
    """
    Extract company, role, and JD from a job posting URL.

    Tries the ATS JSON API first. Falls back to HTML extraction for
    unknown URLs.
    """
    if not url:
        raise RuntimeError("URL is empty")
    try:
        url = validate_public_http_url(url, resolve_dns=True)
    except UnsafeInputError as exc:
        raise RuntimeError(f"Not a valid URL: {exc}") from exc

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    async with public_async_client(
        timeout=timeout, follow_redirects=False, headers=headers
    ) as client:
        # Greenhouse
        m = _GREENHOUSE_RE.search(url)
        if m:
            logger.info("Detected Greenhouse: board=%s job=%s", m.group(1), m.group(2))
            return await _fetch_greenhouse(client, m.group(1), m.group(2))

        # Lever
        m = _LEVER_RE.search(url)
        if m:
            logger.info("Detected Lever: company=%s posting=%s", m.group(1), m.group(2))
            return await _fetch_lever(client, m.group(1), m.group(2))

        # Ashby
        m = _ASHBY_RE.search(url)
        if m:
            logger.info("Detected Ashby: board=%s posting=%s", m.group(1), m.group(2))
            return await _fetch_ashby(client, m.group(1), m.group(2))

        # Workable
        m = _WORKABLE_RE.search(url)
        if m:
            logger.info("Detected Workable: account=%s shortcode=%s", m.group(1), m.group(2))
            return await _fetch_workable(client, m.group(1), m.group(2))

    # Fallback: HTML scraping via the existing fetcher
    logger.info("No ATS pattern matched %s; falling back to HTML extraction", url)
    from app.services.jobs.jd_fetcher import fetch_job_description

    jd_text = await fetch_job_description(url)
    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[0] if parsed.path else ""
    return JobMetadata(
        company=_titlecase_token(slug) if slug else "",
        role="",
        jd_text=jd_text,
        source_ats="html_fallback",
        source_url=url,
    )
