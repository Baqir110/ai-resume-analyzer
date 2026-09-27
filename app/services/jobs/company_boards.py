# app/services/jobs/company_boards.py
"""List *all* open jobs on a company's ATS board.

job_metadata.py already knows how to pull one job's JD from its ATS
job-detail URL. This module is the sibling capability: given a company
board handle, list every open posting so discovery isn't limited to
jobs a search engine already indexed.

All endpoints used here are the same public, unauthenticated JSON APIs
each ATS vendor documents for embedding a company's "open positions"
widget on their own career page -- e.g. Greenhouse's
`boards-api.greenhouse.io/v1/boards/{board}/jobs`. Nothing here logs
in, solves a CAPTCHA, or touches an endpoint that requires credentials.

Network access to these hosts isn't available in this sandbox, so
treat the parsing logic as best-effort: keep the try/except wide and
log-and-skip on any shape mismatch rather than raising, since ATS
vendors change response shapes without notice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.network import public_async_client
from app.core.security import validate_public_http_url
from app.services.jobs.agent_schemas import NormalizedJob

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_TIMEOUT = 20.0


def _safe_handle(handle: str) -> str:
    value = str(handle or "").strip()
    if not value or not all(char.isalnum() or char in "_-" for char in value):
        raise ValueError("Invalid ATS board handle")
    return value


@dataclass
class CompanyBoard:
    """One configured company career board to poll."""

    ats: str  # greenhouse | lever | ashby | workable
    handle: str  # board token / company slug / account name
    company_name: str = ""


async def _get_json(client: httpx.AsyncClient, url: str) -> Any | None:
    try:
        validate_public_http_url(url, resolve_dns=True)
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - vendor APIs shift shape often
        logger.warning("Board fetch failed for %s: %s", url, exc)
        return None


async def list_greenhouse_jobs(
    client: httpx.AsyncClient, board: CompanyBoard
) -> list[NormalizedJob]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{board.handle}/jobs?content=true"
    data = await _get_json(client, url)
    if not data:
        return []
    jobs: list[NormalizedJob] = []
    for item in data.get("jobs", []):
        try:
            location = (item.get("location") or {}).get("name", "")
            jobs.append(
                NormalizedJob(
                    job_id=f"greenhouse-{item.get('id')}",
                    title=item.get("title", "Unknown"),
                    company=board.company_name or board.handle,
                    location=location,
                    description=item.get("content", "") or "",
                    application_url=item.get("absolute_url", ""),
                    original_url=item.get("absolute_url", ""),
                    source="greenhouse",
                    source_job_id=str(item.get("id", "")),
                    company_url=f"https://boards.greenhouse.io/{board.handle}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Greenhouse job entry: %s", exc)
    return jobs


async def list_lever_jobs(client: httpx.AsyncClient, board: CompanyBoard) -> list[NormalizedJob]:
    url = f"https://api.lever.co/v0/postings/{board.handle}?mode=json"
    data = await _get_json(client, url)
    if not data:
        return []
    jobs: list[NormalizedJob] = []
    for item in data:
        try:
            categories = item.get("categories", {}) or {}
            jobs.append(
                NormalizedJob(
                    job_id=f"lever-{item.get('id')}",
                    title=item.get("text", "Unknown"),
                    company=board.company_name or board.handle,
                    location=categories.get("location", ""),
                    employment_type=categories.get("commitment", ""),
                    description=item.get("descriptionPlain", "") or item.get("description", ""),
                    application_url=item.get("applyUrl") or item.get("hostedUrl", ""),
                    original_url=item.get("hostedUrl", ""),
                    source="lever",
                    source_job_id=str(item.get("id", "")),
                    company_url=f"https://jobs.lever.co/{board.handle}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Lever job entry: %s", exc)
    return jobs


async def list_ashby_jobs(client: httpx.AsyncClient, board: CompanyBoard) -> list[NormalizedJob]:
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board.handle}"
    data = await _get_json(client, url)
    if not data:
        return []
    jobs: list[NormalizedJob] = []
    for item in data.get("jobs", []):
        try:
            jobs.append(
                NormalizedJob(
                    job_id=f"ashby-{item.get('id')}",
                    title=item.get("title", "Unknown"),
                    company=board.company_name or board.handle,
                    location=item.get("location", ""),
                    remote_type="remote" if item.get("isRemote") else "",
                    description=item.get("descriptionPlain", "") or "",
                    application_url=item.get("jobUrl", ""),
                    original_url=item.get("jobUrl", ""),
                    source="ashby",
                    source_job_id=str(item.get("id", "")),
                    company_url=f"https://jobs.ashbyhq.com/{board.handle}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Ashby job entry: %s", exc)
    return jobs


async def list_workable_jobs(client: httpx.AsyncClient, board: CompanyBoard) -> list[NormalizedJob]:
    try:
        handle = _safe_handle(board.handle)
        url = f"https://apply.workable.com/api/v3/accounts/{handle}/jobs"
        validate_public_http_url(url, resolve_dns=True)
        resp = await client.post(url, json={})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Board fetch failed for %s: %s", url, exc)
        return []
    jobs: list[NormalizedJob] = []
    for item in data.get("results", []):
        try:
            jobs.append(
                NormalizedJob(
                    job_id=f"workable-{item.get('shortcode')}",
                    title=item.get("title", "Unknown"),
                    company=board.company_name or board.handle,
                    location=(item.get("location") or {}).get("city", ""),
                    employment_type=item.get("employment_type", ""),
                    description=item.get("description", "") or "",
                    application_url=item.get("url", ""),
                    original_url=item.get("url", ""),
                    source="workable",
                    source_job_id=str(item.get("shortcode", "")),
                    company_url=f"https://apply.workable.com/{board.handle}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Workable job entry: %s", exc)
    return jobs


async def list_smartrecruiters_jobs(
    client: httpx.AsyncClient, board: CompanyBoard
) -> list[NormalizedJob]:
    """SmartRecruiters' public Postings API -- documented, unauthenticated,
    the same one their "current openings" embeddable widget uses."""
    url = f"https://api.smartrecruiters.com/v1/companies/{board.handle}/postings"
    data = await _get_json(client, url)
    if not data:
        return []
    jobs: list[NormalizedJob] = []
    for item in data.get("content", []):
        try:
            location = item.get("location", {}) or {}
            location_str = ", ".join(
                filter(
                    None, [location.get("city"), location.get("region"), location.get("country")]
                )
            )
            jobs.append(
                NormalizedJob(
                    job_id=f"smartrecruiters-{item.get('id')}",
                    title=item.get("name", "Unknown"),
                    company=board.company_name or board.handle,
                    location=location_str,
                    remote_type="remote" if location.get("remote") else "",
                    description="",  # Postings list endpoint omits full text; job detail call needed for that
                    application_url=(item.get("applyUrl") or item.get("ref", "")),
                    original_url=item.get("ref", ""),
                    source="smartrecruiters",
                    source_job_id=str(item.get("id", "")),
                    company_url=f"https://careers.smartrecruiters.com/{board.handle}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed SmartRecruiters job entry: %s", exc)
    return jobs


async def list_recruitee_jobs(
    client: httpx.AsyncClient, board: CompanyBoard
) -> list[NormalizedJob]:
    """Recruitee's public careers-site API, e.g. https://{handle}.recruitee.com/api/offers/"""
    url = f"https://{board.handle}.recruitee.com/api/offers/"
    data = await _get_json(client, url)
    if not data:
        return []
    jobs: list[NormalizedJob] = []
    for item in data.get("offers", []):
        try:
            jobs.append(
                NormalizedJob(
                    job_id=f"recruitee-{item.get('id')}",
                    title=item.get("title", "Unknown"),
                    company=board.company_name or board.handle,
                    location=item.get("location", "") or item.get("city", ""),
                    remote_type="remote" if item.get("remote") else "",
                    employment_type=item.get("employment_type_code", ""),
                    description=item.get("description", "") or "",
                    application_url=item.get("careers_url", ""),
                    original_url=item.get("careers_url", ""),
                    source="recruitee",
                    source_job_id=str(item.get("id", "")),
                    company_url=f"https://{board.handle}.recruitee.com",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Recruitee job entry: %s", exc)
    return jobs


async def list_personio_jobs(client: httpx.AsyncClient, board: CompanyBoard) -> list[NormalizedJob]:
    """Personio's public XML job-postings feed, used for job-board syndication:
    https://{handle}.jobs.personio.de/xml

    LOWER CONFIDENCE than the JSON APIs above -- Personio's syndication feed
    format has changed in the past and this hasn't been validated against a
    live company feed in this environment. Treat parse failures here as
    expected until confirmed against a real board.
    """
    try:
        handle = _safe_handle(board.handle)
        url = f"https://{handle}.jobs.personio.de/xml"
        validate_public_http_url(url, resolve_dns=True)
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Personio feed fetch failed for %s: %s", board.handle, exc)
        return []

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Personio feed XML parse failed for %s: %s", board.handle, exc)
        return []

    jobs: list[NormalizedJob] = []
    for position in root.findall(".//position"):
        try:

            def _text(tag: str) -> str:
                el = position.find(tag)
                return (el.text or "").strip() if el is not None else ""

            jobs.append(
                NormalizedJob(
                    job_id=f"personio-{_text('id')}",
                    title=_text("name"),
                    company=board.company_name or board.handle,
                    location=_text("office"),
                    employment_type=_text("employmentType"),
                    description=_text("jobDescriptions"),
                    application_url=_text("applyUrl"),
                    source="personio",
                    source_job_id=_text("id"),
                    company_url=f"https://{board.handle}.jobs.personio.de",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Personio job entry: %s", exc)
    return jobs


async def list_workday_jobs(
    client: httpx.AsyncClient,
    board: dict,  # {"company": ..., "tenant": ..., "board": ..., "instance": ...}
) -> list[NormalizedJob]:
    """Fetch one Workday board via the internal CXS JSON API.

    API: POST https://{tenant}.{instance}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs
    Body: {"limit": 20, "offset": 0, "searchText": "", "locations": []}
    """
    try:
        tenant = _safe_handle(board.get("tenant") or board.get("company", ""))
        board_name = _safe_handle(board.get("board", ""))
        instance = _safe_handle(board.get("instance", "wd3"))
        company_display = board.get("company", tenant)

        url = (
            f"https://{tenant}.{instance}.myworkdayjobs.com" f"/wday/cxs/{tenant}/{board_name}/jobs"
        )
        validate_public_http_url(url, resolve_dns=True)
        resp = await client.post(
            url,
            json={"limit": 20, "offset": 0, "searchText": "", "locations": []},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Workday board fetch failed for %s: %s", board, exc)
        return []

    jobs: list[NormalizedJob] = []
    for item in data.get("jobPostings", []):
        try:
            external_path = item.get("externalPath", "")
            apply_url = (
                f"https://{tenant}.{instance}.myworkdayjobs.com{external_path}"
                if external_path
                else ""
            )
            bullet = " ".join(item.get("bulletFields", []) or [])
            jobs.append(
                NormalizedJob(
                    job_id=f"workday-{tenant}-{external_path}",
                    title=item.get("title", "Unknown"),
                    company=company_display,
                    location=item.get("locationsText", ""),
                    description=bullet,
                    application_url=apply_url,
                    original_url=apply_url,
                    source="workday",
                    source_job_id=external_path,
                    company_url=(
                        f"https://{tenant}.{instance}.myworkdayjobs.com" f"/en-US/{board_name}/jobs"
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Workday job entry: %s", exc)
    return jobs


_LISTERS = {
    "greenhouse": list_greenhouse_jobs,
    "lever": list_lever_jobs,
    "ashby": list_ashby_jobs,
    "workable": list_workable_jobs,
    "smartrecruiters": list_smartrecruiters_jobs,
    "recruitee": list_recruitee_jobs,
    "personio": list_personio_jobs,
}


async def list_all_boards(boards: list[CompanyBoard]) -> list[NormalizedJob]:
    """Fetch every configured company board concurrently."""
    if not boards:
        return []
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    jobs: list[NormalizedJob] = []
    async with public_async_client(timeout=_TIMEOUT, headers=headers) as client:
        import asyncio

        tasks = []
        for board in boards:
            try:
                board.handle = _safe_handle(board.handle)
            except ValueError:
                logger.warning("Skipping board with invalid handle")
                continue
            lister = _LISTERS.get(board.ats)
            if lister is None:
                logger.warning(
                    "Unknown ATS '%s' for board '%s' -- skipping", board.ats, board.handle
                )
                continue
            tasks.append(lister(client, board))
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                logger.warning("Board listing task failed: %s", result)
                continue
            jobs.extend(result)
    return jobs


__all__ = ["CompanyBoard", "list_all_boards", "list_workday_jobs"]
