"""Tests for structured requirement extraction and keyword-stuffing detection.

These cover the requirement-24 content items that keyword matching cannot reach:
industry terminology, degree level, named certifications, language ability, and
the one keyword problem that keyword matching structurally cannot see.
"""

import pytest

from app.services.analysis.requirement_extraction import (
    DEGREE_ORDER,
    extract_certification_requirements,
    extract_education_requirements,
    extract_industry_terminology,
    extract_language_requirements,
    find_keyword_stuffing,
    highest_degree_in_text,
)

# ---------------------------------------------------------------------------
# Industry terminology
# ---------------------------------------------------------------------------


class TestIndustryTerminology:
    def test_regulatory_frameworks_are_found(self):
        jd = "You will ensure compliance with GDPR and ISO 27001 across the estate."
        assert "GDPR" in extract_industry_terminology(jd)
        assert "ISO 27001" in extract_industry_terminology(jd)

    def test_sector_acronyms_are_found(self):
        jd = "Experience with HIPAA, PCI DSS and SOC 2 in a regulated environment."
        found = extract_industry_terminology(jd)
        assert "HIPAA" in found
        assert "PCI DSS" in found

    def test_financial_standards(self):
        jd = "Reporting under IFRS and SOX, with KYC and AML exposure."
        found = extract_industry_terminology(jd)
        assert "IFRS" in found and "SOX" in found and "KYC" in found and "AML" in found

    def test_a_plain_tech_posting_has_no_industry_terms(self):
        jd = "We need a backend engineer with Python, Docker and Kubernetes."
        assert extract_industry_terminology(jd) == []

    def test_no_duplicates(self):
        jd = "GDPR, GDPR, GDPR, ISO 27001 and ISO 27001."
        found = extract_industry_terminology(jd)
        assert found.count("GDPR") == 1
        assert found.count("ISO 27001") == 1

    def test_empty_posting(self):
        assert extract_industry_terminology("") == []

    def test_not_matched_inside_a_longer_word(self):
        """'GDPRS' is not the GDPR framework."""
        assert extract_industry_terminology("we process GDPRS documents") == []


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------


class TestEducationRequirements:
    @pytest.mark.parametrize(
        "jd,level",
        [
            ("A Bachelor's degree is required.", "bachelor"),
            ("We require a Master's in Computer Science.", "master"),
            ("A PhD or doctorate is essential for this role.", "doctorate"),
            ("Fachhochschulabschluss or equivalent required.", "associate"),
            ("A-levels or equivalent required.", "high_school"),
        ],
    )
    def test_levels_are_recognised(self, jd, level):
        result = extract_education_requirements(jd)
        assert result["stated"] is True
        assert result["level"] == level

    def test_highest_level_wins_when_several_are_named(self):
        jd = "A Bachelor's is required; a Master's is preferred."
        assert extract_education_requirements(jd)["level"] == "master"

    def test_posting_with_no_degree_is_not_stated(self):
        result = extract_education_requirements("We need someone to build things.")
        assert result["stated"] is False
        assert result["level"] is None

    def test_unnamed_degree_defaults_to_bachelor(self):
        result = extract_education_requirements("A degree in Computer Science is required.")
        assert result["stated"] is True
        assert result["level"] == "bachelor"

    def test_highest_degree_in_a_cv(self):
        assert highest_degree_in_text("BSc Computer Science, 2013-2016") == "bachelor"
        assert highest_degree_in_text("MSc Software Engineering") == "master"
        assert highest_degree_in_text("PhD in Machine Learning") == "doctorate"

    def test_a_cv_naming_no_qualification_is_unverified_not_none(self):
        assert highest_degree_in_text("Worked in IT for years.") is None

    def test_degree_order_is_ascending(self):
        assert list(DEGREE_ORDER) == [
            "high_school",
            "associate",
            "bachelor",
            "master",
            "doctorate",
        ]


# ---------------------------------------------------------------------------
# Certifications
# ---------------------------------------------------------------------------


class TestCertificationRequirements:
    def test_aws_certifications(self):
        found = extract_certification_requirements(
            "You must hold AWS Certified Solutions Architect."
        )
        assert any("AWS" in name for name in found)

    def test_kubernetes_admin(self):
        found = extract_certification_requirements("Certified Kubernetes Administrator required.")
        assert found

    def test_several_are_all_found(self):
        jd = "PMP and CISSP are required, as is Certified Kubernetes Administrator."
        found = extract_certification_requirements(jd)
        assert len(found) >= 3

    def test_plain_posting_names_none(self):
        assert extract_certification_requirements("Experience with Python and Docker.") == []

    def test_duplicates_collapse(self):
        jd = "PMP required. PMP is essential."
        assert extract_certification_requirements(jd).count("PMP") == 1


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------


class TestLanguageRequirements:
    def test_required_language_is_found(self):
        found = extract_language_requirements("German is required for this role.")
        assert "German" in [entry["language"] for entry in found]

    def test_cefr_level_is_captured(self):
        found = extract_language_requirements("Fluent German at B2 or above is required.")
        entry = next(e for e in found if e["language"] == "German")
        assert entry["level"] == "B2"

    def test_a_passing_mention_is_not_a_requirement(self):
        """'Our Berlin team speaks Polish' is a fact about the team."""
        found = extract_language_requirements("Our Berlin team speaks Polish.")
        assert found == []

    def test_several_languages(self):
        jd = "Fluent English required. German is essential for this client."
        languages = [e["language"] for e in extract_language_requirements(jd)]
        assert "English" in languages
        assert "German" in languages

    def test_german_posting(self):
        found = extract_language_requirements("Deutschkenntnisse in C1 sind erforderlich.")
        assert "German" in [e["language"] for e in found]

    def test_plain_posting_names_none(self):
        assert extract_language_requirements("Build services with Python.") == []

    def test_empty_posting(self):
        assert extract_language_requirements("") == []


# ---------------------------------------------------------------------------
# Keyword stuffing
# ---------------------------------------------------------------------------

_NORMAL_CV = """
Jane Doe
jane.doe@example.com | +44 20 7946 0000

EXPERIENCE
Platform Engineer | Northwind | Jan 2020 - Present
- Ran Kubernetes clusters on AWS across production environments.
- Cut build times from 3 days to 40 minutes using Terraform.
- Built delivery pipelines with GitHub Actions and Jenkins.
- Instrumented services with Prometheus and Grafana.

EDUCATION
BSc Computer Science | University of Manchester | Sep 2013 - Jul 2016

SKILLS
Kubernetes, Terraform, AWS, GitHub Actions, Jenkins, Prometheus, Grafana, Python
"""

_STUFFED_CV = (
    "Jane Doe\njane.doe@example.com\n\n"
    "EXPERIENCE\nPlatform Engineer | Northwind | Jan 2020 - Present\n"
    + "Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes\n"
    + "Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes\n"
    + "Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes\n"
    + "Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes Kubernetes\n"
    + "EDUCATION\nBSc Computer Science\n\n"
    + "SKILLS\nKubernetes, Terraform, AWS, Python, Docker, Jenkins\n"
)


class TestKeywordStuffing:
    def test_a_normal_cv_is_not_stuffed(self):
        result = find_keyword_stuffing(_NORMAL_CV)
        assert result["stuffed"] == []
        assert "no word appears more than" in result["detail"]

    def test_repetition_is_detected(self):
        result = find_keyword_stuffing(_STUFFED_CV)
        assert result["stuffed"]
        assert result["stuffed"][0]["term"] == "kubernetes"

    def test_the_detail_names_the_term_and_the_count(self):
        result = find_keyword_stuffing(_STUFFED_CV)
        assert "kubernetes" in result["detail"]
        assert str(result["stuffed"][0]["count"]) in result["detail"]

    def test_repeated_names_do_not_count(self):
        """A name and a company appear many times by design."""
        cv = _NORMAL_CV + "\n" + "Jane Doe\nNorthwind\nJane Doe\nNorthwind\nJane Doe\n"
        assert find_keyword_stuffing(cv)["stuffed"] == []

    def test_repeated_emails_do_not_count(self):
        cv = "jane.doe@example.com\n" * 12 + "\nEXPERIENCE\nEngineer\n"
        assert find_keyword_stuffing(cv)["stuffed"] == []

    def test_repeated_urls_do_not_count(self):
        cv = "https://github.com/jane " * 12 + "\nEXPERIENCE\nEngineer\n"
        assert find_keyword_stuffing(cv)["stuffed"] == []

    def test_repeated_years_do_not_count(self):
        cv = "2020 2021 2022 " * 12 + "\nEXPERIENCE\nEngineer at Acme\n"
        assert find_keyword_stuffing(cv)["stuffed"] == []

    def test_short_words_are_ignored(self):
        """'the' cannot be evidence of anything."""
        cv = "the and for with that this " * 20 + "\nEXPERIENCE\nEngineer\n"
        assert find_keyword_stuffing(cv)["stuffed"] == []

    def test_empty_text(self):
        result = find_keyword_stuffing("")
        assert result["stuffed"] == []
        assert "no text to inspect" in result["detail"]

    def test_result_is_capped(self):
        cv = (" ".join(f"term{i}" * 30 for i in range(20))) + "\nEXPERIENCE\nEngineer\n"
        assert len(find_keyword_stuffing(cv, top_n=3)["stuffed"]) <= 3
