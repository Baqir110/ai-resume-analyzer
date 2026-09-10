"""
Analytics aggregator.

Reads the pipeline event log (JSONL) and the tracker SQLite DB, and returns
a structured payload for the analytics dashboard. Pure read-only.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

_KNOWN_STATUSES = ("Saved", "Applied", "Interview", "Offer", "Rejected", "Withdrawn")


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _load_events(log_path: Path, since_hours: Optional[int]) -> list[dict]:
    if not log_path.exists():
        return []
    cutoff: Optional[datetime] = None
    if since_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    events: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if cutoff is not None:
                    ts = _parse_iso(ev.get("timestamp", ""))
                    if ts is None or ts < cutoff:
                        continue
                events.append(ev)
    except OSError:
        return []
    return events


def _load_applications(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT company_name, job_title, ats_score, status, created_at "
            "FROM applications ORDER BY created_at DESC"
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def compute_analytics(
    log_path: Path,
    db_path: Path,
    since_hours: Optional[int] = 24 * 30,
) -> dict[str, Any]:
    events = _load_events(log_path, since_hours)
    applications = _load_applications(db_path)

    status_counter: Counter[str] = Counter()
    score_buckets = {"0-40": 0, "40-60": 0, "60-75": 0, "75-90": 0, "90-100": 0}
    apps_by_day: dict[str, int] = defaultdict(int)

    for app in applications:
        st = app.get("status") or "Unknown"
        status_counter[st] += 1
        score = app.get("ats_score") or 0
        if score < 40:
            score_buckets["0-40"] += 1
        elif score < 60:
            score_buckets["40-60"] += 1
        elif score < 75:
            score_buckets["60-75"] += 1
        elif score < 90:
            score_buckets["75-90"] += 1
        else:
            score_buckets["90-100"] += 1
        created = (app.get("created_at") or "")[:10]
        if created:
            apps_by_day[created] += 1

    for s in _KNOWN_STATUSES:
        status_counter.setdefault(s, 0)

    tokens_by_day: dict[str, int] = defaultdict(int)
    cost_by_day: dict[str, float] = defaultdict(float)
    calls_by_provider: Counter[str] = Counter()
    calls_by_model: Counter[str] = Counter()

    total_calls = completed_calls = failed_calls = 0
    total_tokens = 0
    total_cost = 0.0
    duration_sum = 0.0
    duration_count = 0

    ops_started: Counter[str] = Counter()
    ops_completed: Counter[str] = Counter()
    ops_failed: Counter[str] = Counter()
    op_durations: dict[str, list[float]] = defaultdict(list)

    missing_skills_counter: Counter[str] = Counter()
    matched_skills_counter: Counter[str] = Counter()

    recent_errors: list[dict[str, Any]] = []

    for ev in events:
        kind = ev.get("kind") or "llm"
        event = ev.get("event") or ""
        ts = ev.get("timestamp", "")

        if kind == "llm":
            if event == "request_started":
                total_calls += 1
            elif event == "request_completed":
                completed_calls += 1
                tokens = int(ev.get("total_tokens", 0) or 0)
                cost = float(ev.get("estimated_cost_usd", 0.0) or 0.0)
                duration = float(ev.get("duration_ms", 0.0) or 0.0)
                total_tokens += tokens
                total_cost += cost
                if duration:
                    duration_sum += duration
                    duration_count += 1
                provider = str(ev.get("provider", "unknown"))
                model = str(ev.get("model", "unknown"))
                calls_by_provider[provider] += 1
                calls_by_model[model] += 1
                day = ts[:10] if ts else "unknown"
                tokens_by_day[day] += tokens
                cost_by_day[day] += cost
            elif event == "provider_failed":
                failed_calls += 1
                if len(recent_errors) < 25:
                    recent_errors.append(
                        {
                            "timestamp": ts,
                            "kind": kind,
                            "provider": ev.get("provider", ""),
                            "model": ev.get("model", ""),
                            "error": (ev.get("error") or "")[:200],
                        }
                    )

        elif kind == "pipeline":
            op = str(ev.get("operation", "unknown"))
            if event == "pipeline_started":
                ops_started[op] += 1
            elif event == "pipeline_completed":
                ops_completed[op] += 1
                d = float(ev.get("duration_ms", 0.0) or 0.0)
                if d:
                    op_durations[op].append(d)
            elif event == "pipeline_failed":
                ops_failed[op] += 1
                if len(recent_errors) < 25:
                    recent_errors.append(
                        {
                            "timestamp": ts,
                            "kind": kind,
                            "provider": ev.get("provider", ""),
                            "model": ev.get("model", ""),
                            "error": (ev.get("error") or "")[:200],
                        }
                    )

        elif kind == "analysis" and event == "analysis_completed":
            for s in ev.get("missing_skills") or []:
                if s:
                    missing_skills_counter[str(s)] += 1
            for s in ev.get("matching_skills") or []:
                if s:
                    matched_skills_counter[str(s)] += 1

    op_avg_durations = {
        op: round(sum(durs) / len(durs), 1) for op, durs in op_durations.items() if durs
    }

    days_sorted = sorted(set(list(tokens_by_day.keys()) + list(apps_by_day.keys())))
    tokens_series = [
        {"date": d, "tokens": tokens_by_day.get(d, 0)} for d in days_sorted
    ]
    cost_series = [
        {"date": d, "cost": round(cost_by_day.get(d, 0.0), 6)} for d in days_sorted
    ]
    apps_series = [{"date": d, "count": apps_by_day.get(d, 0)} for d in days_sorted]

    return {
        "meta": {
            "since_hours": since_hours,
            "events_count": len(events),
            "applications_count": len(applications),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "applications": {
            "total": len(applications),
            "by_status": dict(status_counter),
            "by_score_bucket": score_buckets,
            "by_day": apps_series,
        },
        "llm": {
            "total_calls": total_calls,
            "completed_calls": completed_calls,
            "failed_calls": failed_calls,
            "failure_rate": (
                round(failed_calls / total_calls, 4) if total_calls else 0.0
            ),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_duration_ms": (
                round(duration_sum / duration_count, 1) if duration_count else 0.0
            ),
            "calls_by_provider": dict(calls_by_provider),
            "calls_by_model": dict(calls_by_model.most_common(10)),
            "tokens_by_day": tokens_series,
            "cost_by_day": cost_series,
        },
        "pipeline": {
            "started": dict(ops_started),
            "completed": dict(ops_completed),
            "failed": dict(ops_failed),
            "avg_duration_ms_by_op": op_avg_durations,
        },
        "skills": {
            "top_missing": missing_skills_counter.most_common(15),
            "top_matched": matched_skills_counter.most_common(15),
        },
        "recent_errors": recent_errors[:20],
    }
