# tests/test_description_parser.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.jobs.agent_schemas import NormalizedJob
from app.services.jobs.description_parser import (
    MAX_DESCRIPTION_CHARS,
    MAX_DESCRIPTION_LINES,
    MAX_ITEM_CHARS,
    MAX_ITEMS_PER_SECTION,
    enrich_job_from_description,
    parse_description_sections,
)


def test_parses_plain_text_english_sections():
    description = """
    About the role: build cool stuff.

    Responsibilities:
    - Design and build backend services
    - Own the CI/CD pipeline
    - Mentor junior engineers

    Requirements:
    - 3+ years of Python experience
    - Strong Kubernetes background

    Nice to have:
    - Experience with Terraform
    """
    result = parse_description_sections(description)
    assert result.matched_headers is True
    assert "Design and build backend services" in result.responsibilities
    assert "Own the CI/CD pipeline" in result.responsibilities
    assert "3+ years of Python experience" in result.requirements
    assert "Experience with Terraform" in result.preferred_requirements


def test_parses_german_sections():
    description = """
    Ihre Aufgaben:
    - Betreuung der IT-Infrastruktur
    - Zusammenarbeit mit dem Entwicklerteam

    Ihr Profil:
    - Abgeschlossenes Studium der Informatik
    - Erfahrung mit Docker

    Von Vorteil:
    - Kenntnisse in Kubernetes
    """
    result = parse_description_sections(description)
    assert result.matched_headers is True
    assert "Betreuung der IT-Infrastruktur" in result.responsibilities
    assert "Abgeschlossenes Studium der Informatik" in result.requirements
    assert "Kenntnisse in Kubernetes" in result.preferred_requirements


def test_parses_simple_html_greenhouse_style_content():
    description = "<p>Intro text.</p><p><strong>Requirements</strong></p><ul><li>Python</li><li>Docker</li></ul>"
    result = parse_description_sections(description)
    assert result.matched_headers is True
    assert "Python" in result.requirements
    assert "Docker" in result.requirements


def test_no_recognizable_headers_returns_empty_not_error():
    result = parse_description_sections("Just a plain paragraph with no structure at all.")
    assert result.matched_headers is False
    assert result.responsibilities == []
    assert result.requirements == []


def test_empty_description_returns_empty():
    result = parse_description_sections("")
    assert result.matched_headers is False


def test_enrich_job_populates_empty_fields():
    job = NormalizedJob(
        job_id="1",
        title="DevOps Engineer",
        company="ACME",
        description="Requirements:\n- Python\n- Docker",
    )
    enrich_job_from_description(job)
    assert "Python" in job.requirements
    assert "Docker" in job.requirements


def test_enrich_job_does_not_override_existing_source_data():
    """If the source already populated requirements (e.g. a future
    source that does provide structured fields), don't clobber it with
    a heuristic guess."""
    job = NormalizedJob(
        job_id="1",
        title="DevOps Engineer",
        company="ACME",
        description="Requirements:\n- Python\n- Docker",
        requirements=["Already set by source"],
    )
    enrich_job_from_description(job)
    assert job.requirements == ["Already set by source"]


def test_actual_newlines_are_split_instead_of_a_literal_backtick_n():
    result = parse_description_sections("Requirements:\n- Python\n- Docker")
    assert result.matched_headers is True
    assert result.requirements == ["Python", "Docker"]


def test_handles_crlf_entities_and_markdown_headings():
    result = parse_description_sections(
        "Intro\r\n### Requirements &amp; Qualifications\r\n1) Python\r\n• Docker"
    )
    assert result.requirements == ["Python", "Docker"]


def test_oversized_or_non_string_descriptions_fail_closed():
    oversized = "Requirements:\n" + ("x" * MAX_DESCRIPTION_CHARS)
    assert parse_description_sections(oversized).matched_headers is False
    assert parse_description_sections(None).matched_headers is False


def test_item_count_and_item_length_are_bounded():
    result = parse_description_sections(
        "Requirements:\n"
        + "\n".join(f"- item-{index}" for index in range(MAX_ITEMS_PER_SECTION + 50))
    )
    assert len(result.requirements) == MAX_ITEMS_PER_SECTION

    result = parse_description_sections("Requirements:\n- " + ("x" * (MAX_ITEM_CHARS + 100)))
    assert len(result.requirements[0]) == MAX_ITEM_CHARS


def test_line_count_limit_stops_later_sections():
    description = "\n".join(
        ["Requirements:"]
        + [f"- item-{index}" for index in range(MAX_DESCRIPTION_LINES)]
        + ["Nice to have:", "- Terraform"]
    )
    result = parse_description_sections(description)
    assert result.matched_headers is True
    assert result.preferred_requirements == []
