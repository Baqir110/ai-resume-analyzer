"""
Shared pipeline event logger.

All events append to the same JSONL file used by the LLM provider so
everything shows up in one log. Every record has a `kind` field so
consumers can filter by pipeline stage.

This version adds thread-safe file locking (Unix fcntl / Windows msvcrt)
so concurrent writes don't corrupt the log file.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl  # Unix

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAS_FCNTL = False
    try:
        import msvcrt  # Windows

        _HAS_MSVCRT = True
    except ImportError:
        _HAS_MSVCRT = False

logger = logging.getLogger(__name__)

# A single process-wide lock serializes writes across threads in this process.
# On top of that, fcntl/msvcrt locks serialize writes across processes.
_write_lock = threading.RLock()

_current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "pipeline_request_id", default=None
)


def _log_path() -> Path:
    """Resolve the log path from the LLM provider (avoids circular import)."""
    try:
        from app.services.llm.provider import LOG_PATH

        return LOG_PATH
    except Exception:
        return Path("data/llm_processing.jsonl")


def new_request_id() -> str:
    """Fresh pipeline request ID (matches the LLM provider's format)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def set_request_id(rid: str | None) -> None:
    """Bind the current pipeline run's ID to this async context."""
    _current_request_id.set(rid)


def current_request_id() -> str | None:
    """Return the ID for the current pipeline run, or None."""
    return _current_request_id.get()


def _write_record(path: Path, record: dict) -> None:
    """Write a single JSON line to the log with file locking."""
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

    with _write_lock, path.open("a", encoding="utf-8") as fh:
        if _HAS_FCNTL:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
            finally:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        elif _HAS_MSVCRT:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                fh.write(line)
                fh.flush()
            except OSError:
                # Fallback: just write; thread lock already protects us.
                fh.write(line)
                fh.flush()
            finally:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
        else:
            fh.write(line)
            fh.flush()


def log_event(
    kind: str,
    event: str,
    request_id: str | None = None,
    **fields: Any,
) -> None:
    """Append one event to the pipeline log."""
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
        _write_record(path, record)
    except Exception as exc:
        logger.debug("log_event failed: %s", exc)


__all__ = [
    "current_request_id",
    "log_event",
    "new_request_id",
    "set_request_id",
]
