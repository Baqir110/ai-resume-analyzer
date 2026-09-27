"""Regression coverage for routing, dashboard requests, and stored statistics."""

import sqlite3
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from app.api import new_features_endpoints
from app.dashboard import helpers
from app.main import app
from app.services.analysis import ats_analyzer
from app.services.analysis.skill_roadmap import SkillProgressionTracker
from app.services.llm import provider
from app.services.tracking.collaborative_feedback import CollaborativeFeedbackManager
from app.services.tracking.version_manager import ResumeVersionManager


@pytest.fixture
def versions(tmp_path, monkeypatch):
    manager = ResumeVersionManager(tmp_path / "versions.db")
    monkeypatch.setattr(new_features_endpoints, "version_manager", manager)
    return manager


def test_compare_route_is_not_user_listing(versions):
    first = versions.save_version("alice", "old", "Build APIs", 50)
    second = versions.save_version("alice", "old", "Build reliable APIs", 70)
    response = TestClient(app).get(
        "/api/v1/resume/versions/compare",
        params={"version_id_1": first, "version_id_2": second},
    )
    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison["score_improvement"] == 20
    assert comparison["word_additions"] == 1


def test_missing_comparison_returns_404(versions):
    response = TestClient(app).get(
        "/api/v1/resume/versions/compare",
        params={"version_id_1": 100, "version_id_2": 200},
    )
    assert response.status_code == 404


def test_user_listing_still_works(versions):
    versions.save_version("alice", "old", "new", 50)
    response = TestClient(app).get("/api/v1/resume/versions/alice")
    assert response.status_code == 200
    assert len(response.json()["versions"]) == 1


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [("longword", "x", 0), ("one two", "one", -1), ("", "one\n two", 2)],
)
def test_version_word_delta(versions, before, after, expected):
    first = versions.save_version("alice", "original", before, 50)
    second = versions.save_version("alice", "original", after, 60)
    assert versions.compare_versions(first, second)["word_additions"] == expected


def test_feedback_counts_threads_not_joined_comments(tmp_path):
    manager = CollaborativeFeedbackManager(tmp_path / "feedback.db")
    resolved = manager.create_feedback_thread(1, "skills", None, "reviewer")
    manager.create_feedback_thread(1, "education", None, "reviewer")
    for _ in range(3):
        manager.add_comment(resolved, "reviewer", "Useful detail")
    manager.resolve_thread(resolved)
    summary = manager.get_feedback_summary(1)
    assert summary["total_threads"] == 2
    assert summary["resolved_threads"] == 1
    assert summary["total_comments"] == 3
    assert summary["completion_percentage"] == 50


def test_empty_feedback_summary(tmp_path):
    manager = CollaborativeFeedbackManager(tmp_path / "feedback.db")
    summary = manager.get_feedback_summary(100)
    assert summary["resolved_threads"] == 0
    assert summary["completion_percentage"] == 0


def test_tracking_preserves_skill_history(tmp_path):
    tracker = SkillProgressionTracker(tmp_path / "skills.db")
    tracker.track_skill("alice", "python")
    with sqlite3.connect(tracker.db_path) as conn:
        conn.execute(
            "UPDATE skill_progression SET first_detected = ?, learning_resources = ?",
            ("2020-01-01 00:00:00", '["Python docs"]'),
        )
    original = tracker.get_user_skills("alice")[0]
    tracker.track_skill("alice", "python", "advanced")
    tracker.track_skill("bob", "python")
    updated = tracker.get_user_skills("alice")[0]
    assert updated["id"] == original["id"]
    assert updated["first_detected"] == original["first_detected"]
    assert updated["learning_resources"] == original["learning_resources"]
    assert updated["job_count"] == 2
    assert updated["proficiency_level"] == "advanced"
    assert tracker.get_user_skills("bob")[0]["job_count"] == 1


def test_dashboard_json_request_reaches_diff_endpoint(monkeypatch):
    client = TestClient(app)

    def post(url, **kwargs):
        assert kwargs["data"] is None
        assert kwargs["files"] is None
        return client.post("/api/v1/resume/diff-preview", json=kwargs["json"])

    monkeypatch.setattr(helpers.requests, "post", post)
    response = helpers.make_api_request(
        "http://backend/api/v1/resume/diff-preview",
        json={"original_bullets": ["Build APIs"], "optimized_bullets": ["Build fast APIs"]},
    )
    assert response.status_code == 200
    assert response.json()["diffs"][0]["words_added"] == 1


def test_dashboard_keeps_multipart_payload(monkeypatch):
    payload = {"job_description": "Python"}
    files = {"resume_file": ("resume.txt", b"Python", "text/plain")}
    response = requests.Response()
    response.status_code = 200

    def post(url, **kwargs):
        assert kwargs["data"] == payload
        assert kwargs["files"] == files
        assert kwargs["json"] is None
        return response

    monkeypatch.setattr(helpers.requests, "post", post)
    assert helpers.make_api_request("http://backend/analyze", data=payload, files=files) is response


@pytest.mark.parametrize("status", [200, 201, 204])
def test_dashboard_accepts_success_statuses(monkeypatch, status):
    response = requests.Response()
    response.status_code = status
    monkeypatch.setattr(helpers.requests, "post", lambda *args, **kwargs: response)
    _, meta = helpers.make_api_request_verbose("http://backend/test")
    assert meta["error"] is None


def test_repository_paths_resolve_after_service_reorganization():
    root = Path(__file__).resolve().parents[1]
    assert provider.PROJECT_ROOT == root
    assert provider.ENV_PATH == root / ".env"
    assert ats_analyzer.BASE_DIR == root
