"""Job market insights and analytics.

Uses the LLM gateway to produce estimates based on the model's training data.
Results are cached for 1 hour per (role, location, seniority) tuple.

NOT a real-time data provider — treat numbers as informed estimates.
For authoritative data, integrate a paid API (Adzuna, Jooble, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.llm.provider import LLMService

logger = logging.getLogger(__name__)

_CACHE_TTL = timedelta(hours=1)
_CACHE: dict[str, dict[str, Any]] = {}


_SYSTEM_PROMPT = """You are a labor-market analyst. Given a job role, location,
and seniority level, return a single JSON object with ALL of the following
fields filled in.

EXACT SCHEMA (every field is required):
{
  "salary_range": {
    "currency": "EUR",
    "min": 70000,
    "max": 110000,
    "median": 90000
  },
  "top_skills": [
    {"skill": "Kubernetes", "demand_pct": 88, "trend": "rising"},
    {"skill": "Terraform", "demand_pct": 75, "trend": "rising"},
    {"skill": "AWS", "demand_pct": 85, "trend": "stable"},
    {"skill": "Python", "demand_pct": 92, "trend": "stable"},
    {"skill": "Docker", "demand_pct": 80, "trend": "stable"}
  ],
  "top_hiring_companies": ["Example Corp", "Another GmbH", "Third AG"],
  "competition_level": "high",
  "hiring_trend": "rising",
  "typical_interview_rounds": 4,
  "remote_friendliness": "hybrid",
  "summary": "Two to three sentence overview of the market.",
  "caveats": "Estimate based on model training data."
}

RULES:
- Return ONLY the raw JSON object. No markdown fences, no explanation.
- ALL fields above must be present.
- salary_range uses annual gross numbers in the local currency.
- top_skills must have at least 5 entries; demand_pct is 0-100.
- competition_level is one of: low, medium, high.
- hiring_trend is one of: rising, stable, falling.
- remote_friendliness is one of: remote-first, remote-friendly, hybrid, onsite-only.
- Never invent company names that do not exist.
"""


class MarketInsightsEngine:
    """AI-powered market intelligence for roles."""

    def __init__(self, cache_ttl: timedelta = _CACHE_TTL):
        self.cache_ttl = cache_ttl

    async def get_market_insights(
        self,
        role: str,
        location: str,
        seniority: str = "mid",
        provider: str = "experiential",
        route_mode: str = "experiential",
    ) -> dict[str, Any]:
        role = (role or "").strip()
        location = (location or "").strip()
        seniority = (seniority or "mid").strip().lower()

        if not role or not location:
            return {
                "error": "role and location are required",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

        cache_key = f"{role.lower()}|{location.lower()}|{seniority}"
        cached = _CACHE.get(cache_key)
        if cached and (datetime.now(timezone.utc) - cached["_cached_at"]) < self.cache_ttl:
            return {**cached, "cached": True}

        prompt = (
            f"Role: {role}\n"
            f"Location: {location}\n"
            f"Seniority: {seniority}\n\n"
            "Produce the JSON described in the system prompt. "
            "Include EVERY field. Do not omit any."
        )

        raw = ""
        try:
            raw = await asyncio.to_thread(
                LLMService.generate,
                prompt,
                provider=provider,
                route_mode=route_mode,
            )
            logger.info(
                "market_insights raw response (first 800): %s",
                (raw or "")[:800],
            )
            insights = self._parse_json(raw)
        except Exception as exc:
            logger.warning("market_insights LLM call failed: %s", exc)
            insights = self._fallback_insights(role, location, seniority)
            insights["_error"] = str(exc)[:200]

        # Validate required fields — fill missing pieces from fallback
        required = ("salary_range", "top_skills", "summary")
        missing = [f for f in required if not insights.get(f)]
        if missing:
            logger.warning(
                "market_insights: LLM response missing fields %s — filling from fallback",
                missing,
            )
            fallback = self._fallback_insights(role, location, seniority)
            for key, value in fallback.items():
                insights.setdefault(key, value)

        insights.setdefault("role", role)
        insights.setdefault("location", location)
        insights.setdefault("seniority", seniority)
        insights["last_updated"] = datetime.now(timezone.utc).isoformat()
        insights["_cached_at"] = datetime.now(timezone.utc)
        insights["data_sources"] = ["AI-estimated (LLM)"]

        _CACHE[cache_key] = insights
        return {**insights, "cached": False}

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("empty LLM response")

        # Strip markdown fences if present
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                text = text.removeprefix("json")
                text = text.strip()

        # If the LLM wrapped the JSON in prose, find the first '{' and last '}'
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        return json.loads(text.strip())

    def _fallback_insights(self, role: str, location: str, seniority: str) -> dict[str, Any]:
        return {
            "salary_range": {"currency": "USD", "min": 0, "max": 0, "median": 0},
            "top_skills": [],
            "top_hiring_companies": [],
            "competition_level": "unknown",
            "hiring_trend": "unknown",
            "typical_interview_rounds": 0,
            "remote_friendliness": "unknown",
            "summary": (
                f"Market data for {seniority} {role} in {location} is currently "
                "unavailable. Configure an LLM provider to populate this field."
            ),
            "caveats": "Estimate unavailable — LLM call failed.",
        }
