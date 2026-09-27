# app/services/jobs/discovery.py
"""Multi-source job discovery, per agent spec Section 4-5.

Combines:
  - JobSpySource     -- existing finder.py (Indeed/LinkedIn/Google), kept
                         as-is and wrapped, not replaced.
  - ATSDirectSource   -- direct company board JSON APIs (company_boards.py).
                         More reliable for actual submission than scraped
                         Indeed listings, which are frequently
                         Cloudflare-walled (see your run log).
  - WebSearchSource   -- optional; only activates if a search API key is
                         configured (SERPAPI_KEY / GOOGLE_CSE_*). Left as
                         a documented no-op otherwise rather than failing,
                         since this sandbox has no such key.

All sources feed into one NormalizedJob list, then Deduplicator collapses
cross-source repeats before the decision engine ever sees them.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.core.network import public_async_client
from app.core.security import UnsafeInputError, validate_public_http_url
from app.services.jobs.agent_schemas import NormalizedJob
from app.services.jobs.company_boards import CompanyBoard, list_all_boards
from app.services.jobs.dedupe import Deduplicator

# ---------------------------------------------------------------------------
# Aggregator domain list -- used by WebSearchSource to decide whether to try
# scraping a result URL or just keep the search snippet as-is.
# ---------------------------------------------------------------------------
_AGGREGATOR_DOMAINS: frozenset[str] = frozenset(
    {
        "indeed.com",
        "linkedin.com",
        "stepstone.de",
        "xing.com",
        "monster.com",
        "glassdoor.com",
        "jobvector.de",
        "arbeitsagentur.de",
    }
)


def _is_aggregator_url(url: str) -> bool:
    """Return True if *url* is hosted on a known job-aggregator domain."""
    try:
        host = urlparse(url).netloc.lower()
        host = host.removeprefix("www.")
        return any(host == d or host.endswith("." + d) for d in _AGGREGATOR_DOMAINS)
    except Exception:  # noqa: BLE001
        return False


logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    search_term: str = "DevOps Engineer"
    location: str = "Germany"
    results_wanted: int = 30
    hours_old: int = 168  # 7 days — wider window so quiet periods still return results
    company_boards: list[CompanyBoard] = field(default_factory=list)
    workday_boards: list[dict] = field(default_factory=list)
    enable_jobspy: bool = True
    enable_ats_direct: bool = True
    enable_web_search: bool = False
    enable_arbeitsagentur: bool = True
    enable_workday: bool = True


class JobSource(ABC):
    name: str = "unknown"

    @abstractmethod
    async def discover(self, config: DiscoveryConfig) -> list[NormalizedJob]: ...


class JobSpySource(JobSource):
    """Wraps the existing finder.find_jobs_germany (Indeed/LinkedIn/Google)."""

    name = "jobspy"

    async def discover(self, config: DiscoveryConfig) -> list[NormalizedJob]:
        if not config.enable_jobspy:
            return []
        from app.services.jobs.finder import find_jobs_germany

        raw_jobs = await asyncio.to_thread(
            find_jobs_germany,
            search_term=config.search_term,
            location=config.location,
            results_wanted=config.results_wanted,
            hours_old=config.hours_old,
        )
        jobs: list[NormalizedJob] = []
        for raw in raw_jobs:
            jobs.append(
                NormalizedJob(
                    job_id=str(raw.get("job_url", "")),
                    title=raw.get("title", "Unknown"),
                    company=raw.get("company", "Unknown"),
                    location=raw.get("location", ""),
                    description=raw.get("description", ""),
                    application_url=raw.get("job_url", ""),
                    original_url=raw.get("job_url", ""),
                    source=f"jobspy_{raw.get('site', 'unknown')}",
                    raw=raw,
                )
            )
        return jobs


class ATSDirectSource(JobSource):
    """Direct company career-board JSON APIs -- the backend-first path.

    These jobs are far more reliable to actually submit to than a
    scraped Indeed mirror URL: the application_url points straight at
    the company's own ATS-hosted form.
    """

    name = "ats_direct"

    async def discover(self, config: DiscoveryConfig) -> list[NormalizedJob]:
        if not config.enable_ats_direct or not config.company_boards:
            return []
        return await list_all_boards(config.company_boards)


class ArbeitsagenturSource(JobSource):
    """Germany's official public job board API (Section 4). No ToS
    tension here -- this API exists specifically for third-party reuse."""

    name = "arbeitsagentur"

    async def discover(self, config: DiscoveryConfig) -> list[NormalizedJob]:
        if not config.enable_arbeitsagentur:
            return []
        from app.services.jobs.arbeitsagentur import search_arbeitsagentur_jobs

        return await search_arbeitsagentur_jobs(
            query=config.search_term, location=config.location, size=config.results_wanted
        )


class WebSearchSource(JobSource):
    """Web-search-driven discovery of company career pages, per Section 5.

    Requires SERPAPI_KEY (or a Google Programmable Search key) to be
    configured. Without one this returns an empty list and logs once,
    rather than failing the whole discovery run -- consistent with
    "if a source fails, continue with other sources" (Section 19/38).

    For each SerpAPI result:
    - Aggregator URLs (indeed.com, linkedin.com, etc.) are kept as-is using
      the search snippet, since those pages are typically scraping-resistant.
    - Non-aggregator URLs (company career pages, ATS direct pages) are fetched
      synchronously with httpx and parsed with BeautifulSoup to extract a real
      title, company name, and description. Falls back to the snippet on any
      error so one slow/blocked page never stalls the whole batch.
    """

    name = "web_search"

    async def discover(self, config: DiscoveryConfig) -> list[NormalizedJob]:
        if not config.enable_web_search:
            return []
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            logger.info(
                "Web search discovery skipped: SERPAPI_KEY not configured. "
                "Set it in .env to enable this source."
            )
            return []

        queries = _build_search_queries(config)
        jobs: list[NormalizedJob] = []
        async with public_async_client(timeout=20.0) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        "https://serpapi.com/search",
                        params={"engine": "google", "q": query, "api_key": api_key, "num": 10},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Web search query failed (%s): %s", query, exc)
                    continue

                for result in data.get("organic_results", []):
                    url = result.get("link", "")
                    if not url:
                        continue
                    snippet = result.get("snippet", "")
                    serp_title = result.get("title", "Unknown")

                    if _is_aggregator_url(url):
                        # Aggregator pages are scraping-resistant; use snippet as-is.
                        jobs.append(
                            NormalizedJob(
                                job_id=url,
                                title=serp_title,
                                company="",
                                description=snippet,
                                application_url=url,
                                original_url=url,
                                source="web_search",
                            )
                        )
                    else:
                        # Company career page -- try to scrape richer data.
                        enriched = await asyncio.to_thread(
                            _scrape_job_page, url, serp_title, snippet
                        )
                        jobs.append(enriched)
        return jobs


def _scrape_job_page(url: str, fallback_title: str, fallback_snippet: str) -> NormalizedJob:
    """Synchronously fetch *url* and extract job fields using BeautifulSoup.

    Always returns a NormalizedJob; falls back to the SERP title/snippet on
    any error (network failure, parse error, BS4 not installed, etc.).
    """
    try:
        from bs4 import BeautifulSoup  # optional dependency
    except ImportError:
        return NormalizedJob(
            job_id=url,
            title=fallback_title,
            company="",
            description=fallback_snippet,
            application_url=url,
            original_url=url,
            source="web_search",
        )

    try:
        validate_public_http_url(url, resolve_dns=True)
    except UnsafeInputError as exc:
        logger.warning("Refusing unsafe web-search result URL: %s", exc)
        return NormalizedJob(
            job_id=url,
            title=fallback_title,
            company="",
            description=fallback_snippet,
            application_url="",
            original_url="",
            source="web_search",
        )

    try:
        import httpx as _httpx

        resp = _httpx.get(
            url, timeout=10, follow_redirects=False, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # --- title ---
        title = fallback_title
        title_tag = soup.find("title")
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.get_text(strip=True):
            title = h1_tag.get_text(strip=True)
        elif title_tag and title_tag.get_text(strip=True):
            title = title_tag.get_text(strip=True)

        # --- company from og:site_name or domain ---
        company = ""
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            company = og_site["content"].strip()
        else:
            from urllib.parse import urlparse as _up

            host = _up(url).netloc.lower().removeprefix("www.")
            # Take the first label before the TLD, e.g. "careers.siemens.com" -> "siemens"
            parts = host.split(".")
            company = parts[-2] if len(parts) >= 2 else host

        # --- description: first 2000 chars of visible text ---
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        description = text[:2000] if text else fallback_snippet

        return NormalizedJob(
            job_id=url,
            title=title,
            company=company,
            description=description,
            application_url=url,
            original_url=url,
            source="web_search",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Page scrape failed for %s (%s); using snippet", url, exc)
        return NormalizedJob(
            job_id=url,
            title=fallback_title,
            company="",
            description=fallback_snippet,
            application_url=url,
            original_url=url,
            source="web_search",
        )


def _build_search_queries(config: DiscoveryConfig) -> list[str]:
    """Generate targeted queries for German job portals only.
    No Greenhouse/Lever/Ashby ATS boards -- those are international noise.
    Targets: StepStone, XING, Karriere.de, Arbeitsagentur, SAP Jobs, and
    direct .de career pages."""
    role = config.search_term
    location = config.location
    return [
        f'"{role}" "{location}" site:stepstone.de',
        f'"{role}" "{location}" site:xing.com/jobs',
        f'"{role}" "{location}" site:karriere.de',
        f'"{role}" Deutschland (Stellen OR Stelle OR Jobs) site:.de',
    ]


class WorkdaySource(JobSource):
    """Polls Workday-hosted company job boards via their internal CXS JSON API.

    Each board config entry looks like:
        {"company": "siemens", "tenant": "siemens", "board": "Siemens", "instance": "wd3"}

    API endpoint pattern:
        POST https://{tenant}.{instance}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs
    """

    name = "workday"

    async def discover(self, config: DiscoveryConfig) -> list[NormalizedJob]:
        if not config.enable_workday:
            return []
        boards = config.workday_boards
        if not boards:
            # Fall back to WORKDAY_BOARDS from settings if config is empty.
            try:
                from app.core.config import settings

                boards = list(settings.WORKDAY_BOARDS)
            except Exception:  # noqa: BLE001
                return []
        if not boards:
            return []

        all_jobs: list[NormalizedJob] = []
        async with public_async_client(timeout=20.0) as client:
            tasks = [_fetch_workday_board(client, b) for b in boards]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, Exception):
                    logger.warning("Workday board fetch failed: %s", result)
                    continue
                all_jobs.extend(result)
        return all_jobs


async def _fetch_workday_board(
    client,
    board: dict,  # type: ignore[type-arg]
) -> list[NormalizedJob]:
    """Fetch one Workday board and return NormalizedJob list."""
    try:
        tenant = board.get("tenant") or board.get("company", "")
        board_name = board.get("board", "")
        instance = board.get("instance", "wd3")
        company_display = board.get("company", tenant)

        url = f"https://{tenant}.{instance}.myworkdayjobs.com/wday/cxs/{tenant}/{board_name}/jobs"
        try:
            validate_public_http_url(url, resolve_dns=True)
        except UnsafeInputError as exc:
            raise RuntimeError(f"Unsafe Workday URL: {exc}") from exc
        resp = await client.post(
            url,
            json={"limit": 20, "offset": 0, "searchText": "", "locations": []},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

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
                        company_url=f"https://{tenant}.{instance}.myworkdayjobs.com/en-US/{board_name}/jobs",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed Workday job entry: %s", exc)
        return jobs
    except Exception as exc:  # noqa: BLE001
        logger.warning("Workday board %s failed: %s", board, exc)
        return []


class DiscoveryEngine:
    def __init__(self, sources: list[JobSource] | None = None):
        self.sources = sources or [
            JobSpySource(),
            ATSDirectSource(),
            ArbeitsagenturSource(),
            WebSearchSource(),
            WorkdaySource(),
        ]

    async def discover_all(
        self, config: DiscoveryConfig, dedupe: Deduplicator | None = None
    ) -> list[NormalizedJob]:
        dedupe = dedupe or Deduplicator()
        results = await asyncio.gather(
            *(self._safe_discover(source, config) for source in self.sources)
        )
        all_jobs: list[NormalizedJob] = []
        for source, jobs in zip(self.sources, results):
            logger.info("Source '%s' returned %d jobs", source.name, len(jobs))
            all_jobs.extend(jobs)

        unique = dedupe.dedupe_batch(all_jobs)
        for job in unique:
            from app.services.jobs.description_parser import enrich_job_from_description

            enrich_job_from_description(job)
        unique = self._prioritize_by_source_performance(unique)
        logger.info(
            "Discovery complete: %d raw -> %d unique after dedupe", len(all_jobs), len(unique)
        )
        return unique

    def _prioritize_by_source_performance(self, jobs: list[NormalizedJob]) -> list[NormalizedJob]:
        """Section 26/27: sort discovered jobs so ones from
        historically-higher-interview-rate sources come first. This
        matters when MAX_DAILY_APPLICATIONS caps how many of today's
        matches actually get applied to -- it should be the jobs from
        sources that have actually produced interviews, not just
        whichever source's async task finished first.

        A source with fewer than 3 recorded submissions gets a neutral
        weight (1.0) rather than being penalized for lack of data --
        see source_priority_weights()'s own docstring for that reasoning.
        Sort is stable, so within a weight tier, original discovery
        order (and dedupe's richest-record-wins choice) is preserved.
        """
        try:
            from app.services.tracking.analytics import source_priority_weights

            weights = source_priority_weights()
        except Exception:  # noqa: BLE001
            logger.warning("Could not compute source priority weights; using discovery order as-is")
            return jobs
        if not weights:
            return jobs
        return sorted(jobs, key=lambda j: -weights.get(j.source, 1.0))

    async def _safe_discover(
        self, source: JobSource, config: DiscoveryConfig
    ) -> list[NormalizedJob]:
        # Section 19/38: one failing source must never halt discovery.
        try:
            return await source.discover(config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Discovery source '%s' failed: %s", source.name, exc)
            return []


__all__ = [
    "DiscoveryConfig",
    "JobSource",
    "JobSpySource",
    "ATSDirectSource",
    "WebSearchSource",
    "WorkdaySource",
    "DiscoveryEngine",
    "_AGGREGATOR_DOMAINS",
    "_is_aggregator_url",
]
