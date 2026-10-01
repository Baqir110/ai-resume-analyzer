"""
Regression tests for defects found in the October audit.

Each test here pins a specific defect that was live in this repository: a
scoring rule that rewarded a CV for skills it did not have, a language match
that could only be satisfied in English, a private vocabulary that disagreed
with the shared one, an error class that answered 500 for a client mistake, and
a provider that reported itself ready while being unable to answer.
"""

from __future__ import annotations

import pytest

from app.services.analysis.ats_scoring import (
    _score_required_skills,
    _score_structured_qualifications,
)
from app.services.jobs.decision_engine import _education_score
from app.services.llm.provider import LLMService, redact_secrets

# ---------------------------------------------------------------------------
# A missing skill must not be scored as a present one
# ---------------------------------------------------------------------------


def test_unclassified_skills_are_not_counted_as_both_earned_and_missing():
    """
    The regression. A posting whose keywords carry no "Required:"/"Nice to have:"
    label left every keyword in the `unclassified` bucket, and every *absent* one
    was additionally appended to `optional_missing`. It was therefore counted once
    as earned (via `unspecified`) and once as missing, so a CV containing none of
    a posting's keywords still scored 50% on a category worth 20% of the score.
    """
    jd = "We need strong Python and Kubernetes and Docker skills for this role."
    absent = "Experienced engineer, no stack listed. a@b.de"

    result = _score_required_skills(absent, jd)

    assert result["optional_matched"] == []
    assert set(result["optional_missing"]) == {"Docker", "Kubernetes", "Python"}
    assert result["score"] == 0.0, (
        "a CV with none of the required keywords must not earn half credit"
    )


def test_a_cv_holding_every_unclassified_keyword_still_scores_full():
    """The other direction: the fix must not penalise an otherwise complete CV."""
    jd = "We need strong Python and Kubernetes and Docker skills for this role."
    present = "Python, Kubernetes and Docker expert. a@b.de"

    result = _score_required_skills(present, jd)

    assert set(result["optional_matched"]) == {"Docker", "Kubernetes", "Python"}
    assert result["score"] == 100.0


# ---------------------------------------------------------------------------
# A language must be matched in the language the CV is written in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resume,expected_matched",
    [
        # The regression: a German Lebenslauf names the language "Deutsch".
        ("Hans Mueller\nSprachen: Deutsch (C1), Englisch (B2)", ["German"]),
        ("Hans Mueller\nLanguages: German (C1), English (B2)", ["German"]),
    ],
)
def test_a_german_requirement_is_satisfied_by_the_german_word(resume, expected_matched):
    """
    The posting side already understood "Deutschkenntnisse"; the CV side was a
    bare English substring test, so a German CV was told it could not satisfy a
    German requirement -- a gap it can never close, reported as unfillable.
    """
    result = _score_structured_qualifications(resume, "German language required")

    assert result["languages_matched"] == expected_matched
    assert result["languages_missing"] == []


def test_a_language_the_cv_really_lacks_is_still_missing():
    """The negative control, so the alias table cannot match everything."""
    result = _score_structured_qualifications(
        "John Doe\nLanguages: Spanish (C1)",
        "German language required",
    )

    assert result["languages_matched"] == []
    assert result["languages_missing"] == ["German"]


def test_redaction_removes_credentials_and_identifiers():
    """
    The API layer's `_safe_error` returned upstream exception text to the client
    with no redaction at all, while three sibling implementations of the same
    idea scrubbed it. A key echoed inside a provider's message went straight into
    an HTTP response body.
    """
    text = (
        "POST https://api.example.com failed: api_key=sk-or-v1-SECRET "
        "for hans.mueller@mail.de / +49 170 1234567"
    )

    scrubbed = redact_secrets(text)

    assert "SECRET" not in scrubbed
    assert "hans.mueller@mail.de" not in scrubbed
    assert "1234567" not in scrubbed
    assert "[REDACTED]" in scrubbed


# ---------------------------------------------------------------------------
# One degree vocabulary, not two
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate,required,expected",
    [
        ("bachelor", "bachelor", 100.0),
        ("bachelor", "master", 70.0),
        ("bachelor", "phd", 40.0),
        # The regression: these three matched nothing in the engine's private
        # four-name list and fell through to the neutral 75.0. A bachelor was
        # docked for a posting that only asked for a high-school diploma.
        ("bachelor", "doctorate", 40.0),
        ("bachelor", "high_school", 100.0),
        ("bachelor", "associate", 100.0),
        ("master", "bachelor", 100.0),
        ("phd", "doctorate", 100.0),
        ("bachelor", "", 100.0),
    ],
)
def test_degree_scoring_uses_the_shared_vocabulary(candidate, required, expected):
    result = _education_score(candidate, required)

    assert result == expected


def test_an_unrecognised_degree_stays_neutral_rather_than_crashing():
    """Unknown stays unknown; the fix must not turn "cannot tell" into a penalty."""
    assert _education_score("bachelor", "college") == 75.0


# ---------------------------------------------------------------------------
# A provider with a key but no model is not usable
# ---------------------------------------------------------------------------


@pytest.fixture
def key_only_provider(monkeypatch):
    for name in ("CEREBRAS_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
    monkeypatch.delenv("CEREBRAS_MODEL", raising=False)


def test_a_key_without_a_model_is_not_reported_as_configured(key_only_provider):
    """
    Four registry entries ship no default model. A credential alone was enough to
    call them configured, so they entered the online chain, produced zero
    candidates, and made every request fail with "No configured LLM providers are
    available" while the status endpoint reported them ready.
    """
    assert LLMService._provider_is_configured("cerebras") is False


def test_discovery_still_works_without_a_configured_model(key_only_provider):
    """
    The guard on the guard. Requiring a model before discovery would make it
    impossible to discover one, which is the whole reason discovery is offered.
    """
    assert LLMService._provider_has_credential("cerebras") is True


def test_a_keyless_chain_produces_no_phantom_candidates(key_only_provider):
    """Cerebras must not appear in the chain at all, rather than appear empty."""
    chain = LLMService._online_provider_chain()

    assert "cerebras" not in chain
