"""Resume authenticity and plagiarism signal detection.

Combines deterministic heuristics (AI-phrase density, buzzword overload,
quantification patterns) with an LLM deep read. Decision-support tool, not
a verdict — every signal is surfaced with its evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.services.llm.provider import LLMService

logger = logging.getLogger(__name__)


_AI_TELLS = [
    r"\bleverag(?:e|ed|ing)\b",
    r"\bspearhead(?:ed|ing)\b",
    r"\bsynerg(?:y|ies|istic)\b",
    r"\bcutting[- ]edge\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bgame[- ]chang(?:er|ing)\b",
    r"\bpassionate about\b",
    r"\bdetail[- ]oriented\b",
    r"\bresults[- ]driven\b",
    r"\bproven track record\b",
    r"\bdynamic professional\b",
    r"\binnovative solutions?\b",
    r"\bholistic approach\b",
    r"\bseamless(?:ly)?\b",
]

_VAGUE_BUZZWORDS = [
    "synergy",
    "paradigm",
    "disrupt",
    "innovate",
    "transformative",
    "revolutionize",
    "world-class",
    "best-in-class",
    "go-to-market",
    "hyper-growth",
    "thought leader",
    "value add",
    "move the needle",
]


class AuthenticityChecker:
    """Flags resumes that show signs of AI generation or low authenticity."""

    async def check_authenticity(
        self,
        resume_text: str,
        provider: str = "experiential",
        route_mode: str = "experiential",
    ) -> dict[str, Any]:
        text = (resume_text or "").strip()
        if not text:
            return {
                "error": "empty resume text",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        heuristics = self._run_heuristics(text)
        llm_analysis = await self._run_llm_analysis(text, provider=provider, route_mode=route_mode)
        overall_score = self._combine_scores(heuristics, llm_analysis)

        return {
            "authenticity_score": overall_score,
            "verdict": self._verdict_from_score(overall_score),
            "heuristics": heuristics,
            "llm_analysis": llm_analysis,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Heuristic layer
    # ------------------------------------------------------------------

    def _run_heuristics(self, text: str) -> dict[str, Any]:
        lower = text.lower()
        word_count = max(1, len(text.split()))

        ai_hits: list[str] = []
        for pattern in _AI_TELLS:
            matches = re.findall(pattern, lower, flags=re.IGNORECASE)
            if matches:
                ai_hits.append(matches[0])

        buzz_hits = [b for b in _VAGUE_BUZZWORDS if b in lower]

        number_hits = len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))
        numbers_per_100_words = (number_hits / word_count) * 100

        ai_density = len(ai_hits) / word_count * 100
        buzz_density = len(buzz_hits) / word_count * 100

        score = 100.0
        score -= min(40.0, ai_density * 200.0)
        score -= min(30.0, buzz_density * 300.0)
        if numbers_per_100_words < 1.0:
            score -= 15.0
        if word_count < 150:
            score -= 10.0

        return {
            "score": max(0, min(100, round(score))),
            "ai_phrase_hits": ai_hits,
            "vague_buzzword_hits": buzz_hits,
            "numbers_per_100_words": round(numbers_per_100_words, 2),
            "word_count": word_count,
        }

    # ------------------------------------------------------------------
    # LLM layer
    # ------------------------------------------------------------------

    async def _run_llm_analysis(self, text: str, provider: str, route_mode: str) -> dict[str, Any]:
        prompt = (
            "You are a resume authenticity analyst. Read the resume below and "
            "produce a JSON assessment.\n\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "ai_generation_likelihood": <0-100>,\n'
            '  "fabrication_risk": <0-100>,\n'
            '  "plagiarism_risk": <0-100>,\n'
            '  "specificity_score": <0-100>,\n'
            '  "red_flags": ["<short flag>", ...],\n'
            '  "strengths": ["<short strength>", ...],\n'
            '  "recommendations": ["<short action>", ...],\n'
            '  "summary": "<2-3 sentence assessment>"\n'
            "}\n\n"
            "Rules:\n"
            "- Base judgement on evidence in the text, not vibes.\n"
            "- Red flags should cite specific phrases or patterns.\n"
            "- Genuine resumes typically score 0-30 on AI/fabrication.\n\n"
            f"RESUME:\n{text[:6000]}"
        )

        try:
            raw = await asyncio.to_thread(
                LLMService.generate,
                prompt,
                provider=provider,
                route_mode=route_mode,
            )
            parsed = self._parse_json(raw)
            for key in (
                "ai_generation_likelihood",
                "fabrication_risk",
                "plagiarism_risk",
                "specificity_score",
            ):
                parsed[key] = max(0, min(100, int(parsed.get(key, 0) or 0)))
            parsed.setdefault("red_flags", [])
            parsed.setdefault("strengths", [])
            parsed.setdefault("recommendations", [])
            parsed.setdefault("summary", "")
            return parsed
        except Exception as exc:
            logger.warning("authenticity LLM call failed: %s", exc)
            return {
                "ai_generation_likelihood": 0,
                "fabrication_risk": 0,
                "plagiarism_risk": 0,
                "specificity_score": 0,
                "red_flags": [],
                "strengths": [],
                "recommendations": [],
                "summary": f"LLM analysis unavailable: {exc}"[:200],
                "_error": str(exc)[:200],
            }

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            text = text.removeprefix("json")
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _combine_scores(self, heuristics: dict[str, Any], llm: dict[str, Any]) -> int:
        heuristic_score = heuristics.get("score", 50)
        llm_authenticity = 100 - max(
            llm.get("ai_generation_likelihood", 0),
            llm.get("fabrication_risk", 0),
            llm.get("plagiarism_risk", 0),
        )
        return round(heuristic_score * 0.6 + llm_authenticity * 0.4)

    def _verdict_from_score(self, score: int) -> str:
        if score >= 80:
            return "highly_authentic"
        if score >= 60:
            return "likely_authentic"
        if score >= 40:
            return "mixed_signals"
        if score >= 20:
            return "suspicious"
        return "highly_suspicious"
