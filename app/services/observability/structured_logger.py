# app/services/observability/structured_logger.py
"""Structured, machine-parseable event log (Section 32).

The existing `logging.basicConfig` text format throughout this codebase
is fine for a human watching the console, but Section 32's own example
("15:31:04 MATCH_SCORE=91") is asking for something a dashboard or
external monitoring tool could parse without regex-scraping free text.

This writes one JSON object per line to a dedicated log file
(data/agent_events.jsonl by default), alongside -- not instead of --
the normal Python logging everything else in this codebase already
uses. Each event has a fixed `event` name and a `data` payload, so
"errors should be structured and actionable" (the section's own
closing line) means every error event carries `job_id`/`company`/
`stage` fields a script can filter on, not just a message string.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path(os.getenv("AGENT_EVENTS_LOG_PATH", "data/agent_events.jsonl"))


_SENSITIVE_KEY_PATTERNS = (
    "api_key",
    "token",
    "password",
    "secret",
    "cookie",
    "auth",
    "credential",
    "session_id",
    "access_key",
    "private_key",
)


def _scrub(key: str, value: Any) -> Any:
    """Return ``"[REDACTED]"`` if *key* names a sensitive field and
    *value* is a string; otherwise return *value* unchanged."""
    if isinstance(value, str):
        key_lower = key.lower()
        if any(pat in key_lower for pat in _SENSITIVE_KEY_PATTERNS):
            return "[REDACTED]"
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
        value = re.sub(r"(?<!\w)\+?\d[\d .()/-]{7,}\d(?!\w)", "[PHONE]", value)
    return value


def _scrub_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub sensitive string values from *data*."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _scrub_dict(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [_scrub(k, item) for item in v]
        else:
            result[k] = _scrub(k, v)
    return result


def log_event(event: str, **data: Any) -> None:
    """Appends one structured event. Never raises -- a logging failure
    must not take down the pipeline it's trying to observe."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **_scrub_dict(data),
    }
    try:
        from app.core.event_log import _write_record

        _DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_record(_DEFAULT_LOG_PATH, record)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write structured event %s (non-fatal)", event)


def read_events(limit: int = 200) -> list[dict[str, Any]]:
    """Reads the most recent `limit` events, newest last (for a
    dashboard tail view). Missing file -> empty list, not an error."""
    if not _DEFAULT_LOG_PATH.exists():
        return []
    try:
        with open(_DEFAULT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than fail the whole read
        return events
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read structured events (non-fatal)")
        return []


__all__ = ["log_event", "read_events"]
