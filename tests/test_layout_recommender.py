"""Tests for the layout recommendation engine."""

import pytest

from app.services.analysis.layout_recommender import (
    LAYOUT_METADATA,
    get_all_layouts_metadata,
    recommend_layout,
)

LONG_CV = (
    """
Jane Doe
Senior Platform Engineer
jane@example.com

PROFESSIONAL SUMMARY
"""
    + ("A long paragraph of professional context. " * 40)
    + """

WORK EXPERIENCE
Senior Platform Engineer | Northwind Systems | Jan 2020 - Present
- Ran Kubernetes clusters on AWS across production environments.
Platform Engineer | Cobalt Data | Jan 2017 - Dec 2019
- Automated deployment pipelines.
Platform Engineer | Delta Systems | Jan 2014 - Dec 2016
- Maintained internal tooling.

EDUCATION
BSc Computer Science

SKILLS
"""
    + (", ".join(f"Skill{i}" for i in range(40)) + "\n")
)

SHORT_CV = """
Jane Doe
jane@example.com

EXPERIENCE
Engineer | Acme | 2024 - 2025
- Built things.

SKILLS
Python
"""


class TestRecommendLayout:
    def test_recommendation_structure(self):
        """Recommendation should have required fields."""
        result = recommend_layout(
            "We need a software engineer with Python and Kubernetes.",
            "Software Engineer",
        )
        assert "recommended_layout" in result
        assert "reason" in result
        assert "ats_safety" in result
        assert "alternatives" in result

    def test_technical_role_recommends_ats_layout(self):
        """Technical roles should get ATS-safe layouts."""
        result = recommend_layout(
            "We need a software engineer with Python, Kubernetes, AWS, Docker.",
            "Software Engineer",
        )
        rec_layout = result["recommended_layout"]
        metadata = LAYOUT_METADATA.get(rec_layout, {})
        assert metadata.get("ats_safety_score", 0) >= 80

    def test_german_job_gets_german_layout(self):
        """German jobs should get German layouts."""
        result = recommend_layout(
            "Wir suchen einen Softwareentwickler mit Python und Kubernetes Erfahrung.",
            "Softwareentwickler",
        )
        rec_layout = result["recommended_layout"]
        metadata = LAYOUT_METADATA.get(rec_layout, {})
        assert metadata.get("language") == "de"

    def test_alternatives_provided(self):
        """Alternatives should be provided."""
        result = recommend_layout(
            "We need a software engineer.",
            "Software Engineer",
        )
        assert len(result.get("alternatives", [])) > 0

    def test_ats_critical_gets_high_safety(self):
        """ATS-critical jobs should get high-safety layouts."""
        result = recommend_layout(
            "Software engineer for our tech company. Apply now through our ATS.",
            "Software Engineer",
        )
        rec_layout = result["recommended_layout"]
        metadata = LAYOUT_METADATA.get(rec_layout, {})
        assert metadata.get("ats_safety_score", 0) >= 80


class TestContentVolume:
    def test_skill_count_is_measured(self):
        result = recommend_layout("Senior Platform Engineer role.", LONG_CV)
        skills = result["detected_profile"]["content_volume"]["skills"]
        assert skills > 20, skills

    def test_dated_ranges_are_counted(self):
        """Counts every date range, so education ranges count too.

        Deliberately not scoped to roles only: the signal is how much of the page
        dated entries consume, and a degree with a date range occupies about as
        much as a job with one.
        """
        result = recommend_layout("Senior Platform Engineer role.", LONG_CV)
        entries = result["detected_profile"]["content_volume"]["entries"]
        # Three roles, plus the education entry.
        assert entries >= 3, entries

    def test_a_sparse_cv_counts_nothing_invented(self):
        result = recommend_layout("Engineer role.", SHORT_CV)
        volume = result["detected_profile"]["content_volume"]
        assert volume["skills"] <= 2
        assert volume["entries"] <= 1


class TestPageGuidance:
    def test_guidance_is_always_present(self):
        result = recommend_layout("Senior Platform Engineer role.", LONG_CV)
        assert "page_guidance" in result
        assert result["page_guidance"]["basis"]

    def test_long_cv_gets_more_than_one_page(self):
        result = recommend_layout("Senior Platform Engineer role.", LONG_CV)
        assert result["page_guidance"]["recommended_pages"] >= 2

    def test_short_cv_is_told_to_stay_on_one_page(self):
        result = recommend_layout("Junior Engineer role.", SHORT_CV)
        assert result["page_guidance"]["recommended_pages"] == 1
        assert "one page" in result["page_guidance"]["basis"].lower()

    def test_senior_role_allows_a_second_page(self):
        result = recommend_layout("Senior Platform Engineer role.", SHORT_CV)
        assert result["page_guidance"]["recommended_pages"] >= 2
        assert "senior" in result["page_guidance"]["basis"].lower()

    def test_academic_cv_is_not_advised_to_cut(self):
        result = recommend_layout("PhD researcher in machine learning at the university.", LONG_CV)
        guidance = result["page_guidance"]
        assert guidance["recommended_pages"] >= 2
        assert "cannot be cut" in guidance["basis"]

    def test_no_cv_text_says_so_instead_of_guessing(self):
        result = recommend_layout("Senior Platform Engineer role.", "")
        assert result["page_guidance"]["recommended_pages"] is None
        assert "no CV text" in result["page_guidance"]["basis"]


class TestLayoutMetadata:
    def test_all_layouts_have_metadata(self):
        """All layouts should have metadata."""
        metadata = get_all_layouts_metadata()
        assert len(metadata) >= 5

    def test_metadata_fields(self):
        """Each layout should have required metadata fields."""
        metadata = get_all_layouts_metadata()
        for layout_id, meta in metadata.items():
            assert "name" in meta
            assert "ats_safety" in meta
            assert "structure" in meta
            assert "best_for" in meta
            assert "parsing_risk" in meta

    def test_ats_safety_scores(self):
        """ATS safety scores should be reasonable."""
        for layout_id, meta in LAYOUT_METADATA.items():
            score = meta.get("ats_safety_score", 0)
            assert 0 <= score <= 100
