"""Regressions found during the second project audit."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.analytics.dashboard_aggregator import compute_analytics
from app.services.career import audit_matrix, cover_letter_pdf
from app.services.llm.provider import LLMService


def test_usage_is_isolated_between_concurrent_requests():
    barrier = threading.Barrier(2)

    def request(tokens):
        LLMService._set_last_usage({"total_tokens": tokens})
        barrier.wait(timeout=5)
        return LLMService._pop_last_usage()["total_tokens"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, 10)
        second = pool.submit(request, 20)
        assert first.result(timeout=10) == 10
        assert second.result(timeout=10) == 20


def test_usage_is_consumed_only_once():
    LLMService._set_last_usage({"total_tokens": 10})
    assert LLMService._pop_last_usage()["total_tokens"] == 10
    assert LLMService._pop_last_usage()["total_tokens"] == 0


@pytest.fixture
def matrix(monkeypatch):
    monkeypatch.setattr(audit_matrix, "analyze_resume_content", lambda *args: {})
    return audit_matrix.AuditMatrixService


def test_multiword_soft_skills_are_matched(matrix):
    result = matrix.run_full_audit(
        "Stakeholder-management and cross functional collaboration.",
        "Stakeholder management, cross-functional collaboration and time management.",
    )["audit_breakdown"]["soft_skills_and_domain"]
    assert "stakeholder management" in result["matched"]
    assert "cross-functional" in result["matched"]
    assert result["missing"] == ["time management"]


def test_soft_skill_matching_uses_word_boundaries(matrix):
    result = matrix.run_full_audit("agile", "fragile")
    assert result["audit_breakdown"]["soft_skills_and_domain"]["matched"] == []


@pytest.mark.parametrize("metric", ["30%", "12.5%", "12,5 %"])
def test_percentage_metrics_are_counted(matrix, metric):
    result = matrix.run_full_audit(f"- Availability was {metric}.", "")
    assert result["audit_breakdown"]["measurable_impact"]["quantified_bullet_points"] == 1


def test_analytics_ignores_non_objects_and_invalid_numeric_fields(tmp_path):
    log = tmp_path / "events.jsonl"
    record = {
        "kind": "llm",
        "event": "request_completed",
        "timestamp": [],
        "total_tokens": "not a number",
        "estimated_cost_usd": "NaN",
        "duration_ms": {},
    }
    log.write_text("[]\nnull\n42\nbroken\n" + json.dumps(record) + "\n", encoding="utf-8")
    data = compute_analytics(log, tmp_path / "missing.db", since_hours=None)
    assert data["meta"]["events_count"] == 1
    assert data["llm"]["completed_calls"] == 1
    assert data["llm"]["total_tokens"] == 0
    assert data["llm"]["total_cost_usd"] == 0
    json.dumps(data, allow_nan=False)


def test_analytics_returns_newest_twenty_errors(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        {
            "kind": "pipeline",
            "event": "pipeline_failed",
            "error": str(index),
            "timestamp": (start + timedelta(minutes=index)).isoformat(),
        }
        for index in range(40)
    ]
    # Simulate interleaved writes; timestamps, not file order, define recency.
    records = records[::2] + records[1::2]
    log = tmp_path / "events.jsonl"
    log.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    data = compute_analytics(log, tmp_path / "missing.db", since_hours=None)
    assert [row["error"] for row in data["recent_errors"]] == [
        str(index) for index in range(39, 19, -1)
    ]


def test_cover_letter_endpoint_forwards_tone(monkeypatch):
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return "latex", "en"

    monkeypatch.setattr(cover_letter_pdf, "generate_cover_letter_latex", generate)
    monkeypatch.setattr(cover_letter_pdf, "compile_cover_letter_pdf", lambda _: b"mock PDF")
    response = TestClient(app).post(
        "/api/v1/resume/generate-cover-letter-pdf",
        data={"job_description": "Engineer", "tone": "technical"},
        files={"resume_file": ("resume.txt", b"Jane Doe", "text/plain")},
    )
    assert response.status_code == 200
    assert captured["tone"] == "technical"
