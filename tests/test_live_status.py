# tests/test_live_status.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.services.tracking.live_status as live_status


def test_set_and_get_status(tmp_path, monkeypatch):
    from app.services.tracking import tracker

    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "live.db")
    live_status._TABLE_READY = False

    assert live_status.get_status() is None

    live_status.set_stage("DISCOVERY")
    status = live_status.get_status()
    assert status["stage"] == "DISCOVERY"

    live_status.set_stage(
        "ANALYZING", source="greenhouse", job_title="DevOps Engineer", company="ACME"
    )
    status = live_status.get_status()
    assert status["stage"] == "ANALYZING"
    assert status["company"] == "ACME"


def test_status_is_single_row_upserted(tmp_path, monkeypatch):
    from app.services.tracking import tracker

    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "live2.db")
    live_status._TABLE_READY = False

    for i in range(5):
        live_status.set_stage("APPLYING", retry_count=i)

    from app.services.tracking.tracker import ApplicationTrackerService

    with ApplicationTrackerService._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) as c FROM agent_live_status").fetchone()["c"]
    assert count == 1  # never accumulates rows
    assert live_status.get_status()["retry_count"] == 4


def test_clear_removes_status(tmp_path, monkeypatch):
    from app.services.tracking import tracker

    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "live3.db")
    live_status._TABLE_READY = False

    live_status.set_stage("DISCOVERY")
    assert live_status.get_status() is not None
    live_status.clear()
    assert live_status.get_status() is None
