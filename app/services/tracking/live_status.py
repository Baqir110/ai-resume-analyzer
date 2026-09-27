# app/services/tracking/live_status.py
"""Live agent status (Section 30 "Live Agent" panel).

The agent process (run_job_agent.py --serve) and the dashboard
(Streamlit) run as separate processes/containers per docker-compose.yml,
so "live" here means "polled from a single-row status table in the
same SQLite DB" rather than a websocket/push channel -- consistent
with the rest of this codebase's SQLite-based architecture, and good
enough for a dashboard that refreshes every few seconds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from app.services.tracking.tracker import ApplicationTrackerService

_TABLE_READY = False


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    with ApplicationTrackerService._get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_live_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                stage TEXT,
                source TEXT,
                job_title TEXT,
                company TEXT,
                retry_count INTEGER DEFAULT 0,
                current_error TEXT,
                updated_at TEXT
            )
            """)
        conn.commit()
    _TABLE_READY = True


def set_stage(
    stage: str,
    source: str = "",
    job_title: str = "",
    company: str = "",
    retry_count: int = 0,
    current_error: str = "",
) -> None:
    """Called by the orchestrator at each major step (Section 32's own
    example log line -- DISCOVERY, JOB_FOUND, MATCH_SCORE, etc. -- maps
    directly onto these `stage` values)."""
    _ensure_table()
    with ApplicationTrackerService._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_live_status (id, stage, source, job_title, company,
                retry_count, current_error, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                stage=excluded.stage, source=excluded.source,
                job_title=excluded.job_title, company=excluded.company,
                retry_count=excluded.retry_count, current_error=excluded.current_error,
                updated_at=excluded.updated_at
            """,
            (
                stage,
                source,
                job_title,
                company,
                retry_count,
                current_error,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_status() -> Optional[dict[str, Any]]:
    _ensure_table()
    with ApplicationTrackerService._get_connection() as conn:
        row = conn.execute("SELECT * FROM agent_live_status WHERE id = 1").fetchone()
    return dict(row) if row else None


def clear() -> None:
    _ensure_table()
    with ApplicationTrackerService._get_connection() as conn:
        conn.execute("DELETE FROM agent_live_status WHERE id = 1")
        conn.commit()


__all__ = ["set_stage", "get_status", "clear"]
