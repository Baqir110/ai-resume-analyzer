# app/services/jobs/arbeitsagentur.py
"""Bundesagentur für Arbeit (German Federal Employment Agency) job search.

This is a real, publicly documented, unauthenticated REST API
("Jobsuche API") that the Agency runs specifically so third parties can
build on top of Germany's official job board -- unlike Indeed/LinkedIn,
there's no ToS tension here at all; syndicating this data is what the
API is *for*. It requires an `X-API-Key` header, but the Agency
publishes a shared public client key for exactly this kind of
read-only, unauthenticated-user access (the same one the Agency's own
"Jobsuche" mobile app traffic uses) rather than per-developer
registration.

Confidence note: this has been implemented from the API's publicly
documented shape, but not exercised against the live endpoint from this
sandbox (no network path to arbeitsagentur.de here). Endpoint schemas
occasionally get versioned (pc/v4 here) -- if this starts returning
empty results, check whether the Agency has moved to a newer path
before assuming your config is wrong.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx  # noqa: F401 - retained for test/client compatibility

from app.core.network import public_async_client
from app.services.jobs.agent_schemas import NormalizedJob

logger = logging.getLogger(__name__)

BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
# Publicly shared client key for the Jobsuche API's unauthenticated tier.
PUBLIC_API_KEY = "jobboerse-jobsuche"
_TIMEOUT = 20.0


async def search_arbeitsagentur_jobs(
    query: str,
    location: str = "",
    radius_km: int = 25,
    size: int = 25,
    page: int = 1,
) -> list[NormalizedJob]:
    headers = {"X-API-Key": PUBLIC_API_KEY, "Accept": "application/json"}
    params: dict[str, Any] = {"was": query, "size": size, "page": page}
    if location:
        params["wo"] = location
        params["umkreis"] = radius_km

    try:
        async with public_async_client(timeout=_TIMEOUT, headers=headers) as client:
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bundesagentur für Arbeit search failed: %s", exc)
        return []

    jobs: list[NormalizedJob] = []
    for item in data.get("stellenangebote", []):
        try:
            arbeitsort = item.get("arbeitsort", {}) or {}
            location_str = ", ".join(
                filter(
                    None, [arbeitsort.get("ort"), arbeitsort.get("region"), arbeitsort.get("land")]
                )
            )
            ref_nr = item.get("refnr", "")
            detail_url = (
                f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{ref_nr}" if ref_nr else ""
            )
            jobs.append(
                NormalizedJob(
                    job_id=f"arbeitsagentur-{ref_nr}",
                    title=item.get("titel", "Unknown"),
                    company=item.get("arbeitgeber", "Unknown"),
                    location=location_str,
                    country="Germany",
                    description=item.get("stellenbeschreibung", "") or "",
                    application_url=detail_url,
                    original_url=detail_url,
                    source="arbeitsagentur",
                    source_job_id=ref_nr,
                    discovered_at=item.get("aktuelleVeroeffentlichungsdatum", ""),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Bundesagentur job entry: %s", exc)
    return jobs


__all__ = ["search_arbeitsagentur_jobs"]
