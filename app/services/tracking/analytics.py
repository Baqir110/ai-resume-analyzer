# app/services/tracking/analytics.py
"""Follow-up tracking and outcome learning (Sections 25-27).

Follow-ups: every SUBMITTED application gets a follow_up_date computed
from settings.FOLLOW_UP_DAYS. due_follow_ups() surfaces what's overdue
so the dashboard/CLI can remind the user -- nothing here sends email
automatically, since deciding *when* to nudge a real employer is a
judgment call the user should keep.

Learning: aggregates response rates by role/company/source/CV version
so future discovery/decision runs can prioritize what's actually
working (Section 26/27). This never rewrites candidate facts -- it
only informs *prioritization*, never CV content.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.jobs.agent_schemas import ApplicationState
from app.services.tracking.state_machine import ApplicationStateMachine
from app.services.tracking.tracker import ApplicationTrackerService


def _ensure_followup_column() -> None:
    ApplicationStateMachine._ensure_schema()
    with ApplicationTrackerService._get_connection() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(applications)")}
        if "follow_up_date" not in cols:
            conn.execute("ALTER TABLE applications ADD COLUMN follow_up_date TEXT")
        conn.commit()


def schedule_follow_up(application_id: int, follow_up_days: int) -> str:
    """Called when an application transitions to SUBMITTED."""
    _ensure_followup_column()
    follow_up_date = (datetime.now(timezone.utc) + timedelta(days=follow_up_days)).isoformat()
    with ApplicationTrackerService._get_connection() as conn:
        conn.execute(
            "UPDATE applications SET follow_up_date = ? WHERE id = ?",
            (follow_up_date, application_id),
        )
        conn.commit()
    return follow_up_date


def due_follow_ups() -> list[dict[str, Any]]:
    """Applications whose follow-up date has passed and haven't moved
    to a later state yet (interview/rejected/withdrawn)."""
    _ensure_followup_column()
    now = datetime.now(timezone.utc).isoformat()
    with ApplicationTrackerService._get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM applications
            WHERE follow_up_date IS NOT NULL
              AND follow_up_date <= ?
              AND state IN (?, ?)
            ORDER BY follow_up_date ASC
            """,
            (now, ApplicationState.SUBMITTED.value, ApplicationState.VERIFIED.value),
        ).fetchall()
    return [dict(row) for row in rows]


@dataclass
class OutcomeStats:
    dimension: str  # "role" | "company" | "source" | "cv_version"
    key: str
    submitted: int = 0
    interviews: int = 0
    offers: int = 0

    @property
    def interview_rate(self) -> float:
        return round(100.0 * self.interviews / self.submitted, 1) if self.submitted else 0.0

    @property
    def offer_rate(self) -> float:
        return round(100.0 * self.offers / self.submitted, 1) if self.submitted else 0.0


def _terminal_states_reached(application_id: int) -> set[str]:
    return {e["to_state"] for e in ApplicationStateMachine.history(application_id)}


def compute_outcome_stats(dimension: str = "source") -> list[OutcomeStats]:
    """Aggregate response rates by the requested dimension.

    dimension: "source" (jobspy_indeed/greenhouse/...), "company",
    or "role" (job_title, coarse -- exact titles rarely repeat enough
    to be statistically meaningful, but it's a starting point).
    """
    _ensure_followup_column()
    with ApplicationTrackerService._get_connection() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM applications").fetchall()]

    key_field = {"source": "source", "company": "company_name", "role": "job_title"}.get(
        dimension, "source"
    )

    buckets: dict[str, OutcomeStats] = {}
    for row in rows:
        key = row.get(key_field) or "unknown"
        stats = buckets.setdefault(key, OutcomeStats(dimension, key))
        stats.key = key
        reached = _terminal_states_reached(row["id"])
        if {"SUBMITTED", "VERIFIED"} & reached:
            stats.submitted += 1
        if "INTERVIEW" in reached:
            stats.interviews += 1
        if "OFFER" in reached:
            stats.offers += 1

    return sorted(buckets.values(), key=lambda s: -s.interview_rate)


def source_priority_weights() -> dict[str, float]:
    """Section 27: turn outcome stats into weights the discovery engine
    could use to prioritize which sources to poll first. A source with
    no submissions yet gets a neutral 1.0 rather than 0, so it isn't
    starved before it has a chance to prove itself."""
    stats = compute_outcome_stats(dimension="source")
    weights = {}
    for s in stats:
        weights[s.key] = 1.0 + (s.interview_rate / 100.0) if s.submitted >= 3 else 1.0
    return weights


__all__ = [
    "schedule_follow_up",
    "due_follow_ups",
    "OutcomeStats",
    "compute_outcome_stats",
    "source_priority_weights",
]
