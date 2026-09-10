"""
Shared pipeline event logger.

All events append to the same JSONL file used by the LLM provider, so
everything shows up in one log. Every record has a `kind` field so
consumers can filter:

    llm       — LLM provider calls
    parse     — resume file parsing
    analysis  — ATS analysis
    translate — pre-flight resume translation
    latex     — LaTeX body generation
    pdf       — PDF compilation
    docx      — DOCX generation
    pipeline  — endpoint lifecycle (started / completed / failed)
    quota     — quota warnings & rate-limit hits
    cache     — cache hit / miss
    response  — HTTP response served
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _log_path() -> Path:
    """Resolve the log path from the LLM provider (avoids circular import)."""
    try:
        from app.services.llm.provider import LOG_PATH

        return LOG_PATH
    except Exception:
        return Path("data/llm_processing.jsonl")


_current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "pipeline_request_id", default=None
)


def new_request_id() -> str:
    """Fresh pipeline request ID (matches the LLM provider's format)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def set_request_id(rid: Optional[str]) -> None:
    """Bind the current pipeline run's ID to this async context."""
    _current_request_id.set(rid)


def current_request_id() -> Optional[str]:
    """Return the ID for the current pipeline run, or None."""
    return _current_request_id.get()


def log_event(
    kind: str,
    event: str,
    request_id: Optional[str] = None,
    **fields: Any,
) -> None:
    """
    Append one event to the pipeline log.

    If `request_id` is omitted, the current context's request_id is used
    (set via `set_request_id`, which the endpoint decorator does).
    """
    try:
        rid = request_id or current_request_id()
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "event": event,
        }
        if rid:
            record["request_id"] = rid
        for k, v in fields.items():
            if k not in record:
                record[k] = v
            else:
                record[f"_{k}"] = v

        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("log_event failed: %s", exc)


__all__ = [
    "log_event",
    "new_request_id",
    "set_request_id",
    "current_request_id",
]
