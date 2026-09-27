# tests/test_structured_logger.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.services.observability.structured_logger as sl


def test_log_event_writes_valid_json_line(tmp_path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(sl, "_DEFAULT_LOG_PATH", log_path)

    sl.log_event("DISCOVERY_STARTED")
    sl.log_event("MATCH_SCORE", application_id=5, score=91.2, decision="APPLY")

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "DISCOVERY_STARTED"
    assert "timestamp" in first
    second = json.loads(lines[1])
    assert second["event"] == "MATCH_SCORE"
    assert second["application_id"] == 5
    assert second["score"] == 91.2


def test_read_events_returns_most_recent_first_and_last(tmp_path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(sl, "_DEFAULT_LOG_PATH", log_path)

    for i in range(5):
        sl.log_event("JOB_FOUND", index=i)

    events = sl.read_events(limit=3)
    assert len(events) == 3
    assert [e["index"] for e in events] == [2, 3, 4]  # last 3, in order


def test_read_events_on_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "_DEFAULT_LOG_PATH", tmp_path / "nonexistent.jsonl")
    assert sl.read_events() == []


def test_read_events_skips_corrupted_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(sl, "_DEFAULT_LOG_PATH", log_path)
    log_path.write_text('{"event": "OK"}\nnot valid json\n{"event": "OK2"}\n')

    events = sl.read_events()
    assert len(events) == 2
    assert events[0]["event"] == "OK"
    assert events[1]["event"] == "OK2"


def test_log_event_failure_does_not_raise(tmp_path, monkeypatch):
    """A logging failure (e.g. unwritable path) must never break the
    pipeline calling it."""
    monkeypatch.setattr(sl, "_DEFAULT_LOG_PATH", Path("/nonexistent-root/cannot/write/here.jsonl"))
    sl.log_event("SOMETHING")  # must not raise


def test_log_event_scrubs_sensitive_keys(tmp_path, monkeypatch):
    """log_event must redact values whose key names match sensitive patterns
    (api_key, token, password, etc.) while leaving non-sensitive fields intact."""
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(sl, "_DEFAULT_LOG_PATH", log_path)

    sl.log_event("TEST", api_key="sk-secret123", score=91)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = __import__("json").loads(lines[0])
    assert record["api_key"] == "[REDACTED]", "api_key must be scrubbed"
    assert record["score"] == 91, "non-sensitive field must be preserved"
