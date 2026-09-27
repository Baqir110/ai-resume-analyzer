import asyncio
import json

import pytest

from app.services.analysis import market_insights, skill_roadmap


@pytest.mark.parametrize("raw", ["[]", "null", "42", '"text"', "{}"])
def test_roadmap_invalid_shape_uses_fallback(tmp_path, monkeypatch, raw):
    tracker = skill_roadmap.SkillProgressionTracker(tmp_path / "skills.db")
    monkeypatch.setattr(skill_roadmap.LLMService, "generate", lambda *args, **kwargs: raw)
    result = asyncio.run(tracker.generate_learning_roadmap(["Python"], "Engineer"))
    assert "_error" in result
    assert result["phases"] == []


def test_roadmap_sends_schema(tmp_path, monkeypatch):
    tracker = skill_roadmap.SkillProgressionTracker(tmp_path / "skills.db")

    def generate(prompt, **kwargs):
        assert skill_roadmap._SYSTEM_PROMPT in prompt
        return '{"phases": [], "gap_summary": "No gaps"}'

    monkeypatch.setattr(skill_roadmap.LLMService, "generate", generate)
    assert "_error" not in asyncio.run(tracker.generate_learning_roadmap(["Python"], "Engineer"))


def test_market_failure_is_not_cached_and_schema_is_sent(monkeypatch):
    monkeypatch.setattr(market_insights, "_CACHE", {})
    calls = []

    def generate(prompt, **kwargs):
        assert market_insights._SYSTEM_PROMPT in prompt
        calls.append(kwargs.get("provider"))
        if len(calls) == 1:
            return "[]"
        return json.dumps({"salary_range": {"min": 1}, "top_skills": [], "summary": "Estimate"})

    monkeypatch.setattr(market_insights.LLMService, "generate", generate)
    engine = market_insights.MarketInsightsEngine()
    first = asyncio.run(engine.get_market_insights("Engineer", "Berlin"))
    assert "_error" in first
    assert not market_insights._CACHE
    second = asyncio.run(engine.get_market_insights("Engineer", "Berlin"))
    assert "_error" not in second
    third = asyncio.run(engine.get_market_insights("Engineer", "Berlin"))
    assert third["cached"]
    # Use different role to bypass cache and test explicit provider
    asyncio.run(
        engine.get_market_insights(
            "Data Scientist", "Berlin", provider="openai", route_mode="direct"
        )
    )
    # First two calls use default routing (None), third is cached (no call), fourth explicitly uses openai
    assert calls[0] is None
    assert calls[1] is None
    assert calls[2] == "openai"
