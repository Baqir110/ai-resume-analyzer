"""Skill progression tracking and AI-generated learning roadmaps.

Persists detected skills in SQLite for growth tracking.
Roadmap generation is LLM-driven to produce a real curriculum.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.llm.provider import LLMService

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "skill_progression.db"


_SYSTEM_PROMPT = """You are a career-development coach. Given a candidate's
current skills and a target role, produce a structured learning roadmap.

Return ONLY valid JSON (no markdown fences) matching this schema:
{
  "gap_summary": "<1-2 sentence summary of biggest gaps>",
  "estimated_hours_per_week": <integer 3-20>,
  "phases": [
    {
      "name": "<phase name>",
      "duration_weeks": <integer>,
      "focus": "<what this phase builds>",
      "skills": [
        {
          "name": "<skill>",
          "priority": "critical|important|nice-to-have",
          "why": "<1 sentence>",
          "resources": [
            {"type": "course|book|docs|project", "title": "<title>", "url": "<optional>"}
          ]
        }
      ],
      "milestone": "<what the candidate can demonstrate at the end>"
    }
  ],
  "final_portfolio_pieces": ["<project idea>", ...],
  "adjacent_roles": ["<role>", ...]
}

Rules:
- Total duration must fit within months_available.
- Prioritise the highest-leverage skills first.
- Use real, well-known resources (official docs, classic books).
"""


class SkillProgressionTracker:
    """Tracks skills over time and generates AI-driven learning roadmaps."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skill_progression (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    first_detected TIMESTAMP,
                    last_detected TIMESTAMP,
                    job_count INTEGER DEFAULT 1,
                    proficiency_level TEXT DEFAULT 'beginner',
                    learning_resources TEXT,
                    UNIQUE(user_id, skill)
                )
                """)
            conn.commit()

    def track_skill(
        self,
        user_id: str,
        skill: str,
        proficiency: str = "intermediate",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skill_progression
                (user_id, skill, first_detected, last_detected, proficiency_level)
                VALUES (
                    ?,
                    ?,
                    COALESCE(
                        (SELECT first_detected FROM skill_progression
                         WHERE user_id = ? AND skill = ?),
                        CURRENT_TIMESTAMP
                    ),
                    CURRENT_TIMESTAMP,
                    ?
                )
                """,
                (user_id, skill, user_id, skill, proficiency),
            )
            conn.commit()

    def get_user_skills(self, user_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM skill_progression WHERE user_id = ? " "ORDER BY last_detected DESC",
                (user_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    async def generate_learning_roadmap(
        self,
        current_skills: list[str],
        target_role: str,
        months_available: int = 6,
        provider: str = "experiential",
        route_mode: str = "experiential",
    ) -> dict[str, Any]:
        current_skills = [s.strip() for s in (current_skills or []) if s and s.strip()]
        target_role = (target_role or "").strip()
        months_available = max(1, min(int(months_available or 6), 24))

        if not target_role:
            return {"error": "target_role is required"}

        prompt = (
            f"Current skills: {', '.join(current_skills) or '(none provided)'}\n"
            f"Target role: {target_role}\n"
            f"Time budget: {months_available} months\n\n"
            "Produce the JSON described in the system prompt."
        )

        try:
            raw = await asyncio.to_thread(
                LLMService.generate,
                prompt,
                provider=provider,
                route_mode=route_mode,
            )
            roadmap = self._parse_json(raw)
        except Exception as exc:
            logger.warning("skill_roadmap LLM call failed: %s", exc)
            roadmap = self._fallback_roadmap(current_skills, target_role)
            roadmap["_error"] = str(exc)[:200]

        roadmap.setdefault("target_role", target_role)
        roadmap.setdefault("months_available", months_available)
        roadmap.setdefault("current_skills", current_skills)
        roadmap["generated_at"] = datetime.now(timezone.utc).isoformat()
        return roadmap

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            text = text.removeprefix("json")
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())

    def _fallback_roadmap(self, current_skills: list[str], target_role: str) -> dict[str, Any]:
        return {
            "gap_summary": (
                "Roadmap generation is unavailable because the LLM call failed. "
                "Check your provider configuration and try again."
            ),
            "estimated_hours_per_week": 8,
            "phases": [],
            "final_portfolio_pieces": [],
            "adjacent_roles": [],
            "caveats": "Fallback — no AI-generated roadmap available.",
        }
