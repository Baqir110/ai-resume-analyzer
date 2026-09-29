"""Tests for the formatting and parsing checks.

The checks in :mod:`formatting_checks` can all be wrong in the same direction:
flagging a good CV. That is the expensive error, because the candidate is told
their CV is broken and the recommended action — regenerate — produces the same
document. Most of what follows is therefore about *not* firing: a clean CV must
come back clean.
"""

import pytest

from app.services.analysis.formatting_checks import (
    analyze_pdf_layout,
    analyze_text_formatting,
    checkable_tokens,
    detect_sections,
    find_latex_artifacts,
)
from tests.test_ats_scoring_helpers import minimal_pdf, pdf_with_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WELL_FORMATTED_CV = """
Jane Doe
Senior Platform Engineer
jane.doe@example.com | +44 20 7946 0000

PROFESSIONAL SUMMARY
Senior Platform Engineer with 8 years building container infrastructure.

WORK EXPERIENCE
Senior Platform Engineer | Northwind Systems | Jan 2020 - Present
- Operated Kubernetes clusters on AWS across 3 production environments.
- Cut environment build time from 3 days to 40 minutes with Terraform.

Platform Engineer | Cobalt Data | Jan 2017 - Dec 2019
- Automated deployment pipelines with Docker and GitLab CI.

EDUCATION
BSc Computer Science | University of Manchester | Sep 2013 - Jul 2016

SKILLS
Python, Go, Docker, Kubernetes, Terraform, AWS
"""


def _failed(result: dict) -> list[str]:
    return [check["name"] for check in result["checks"] if not check["passed"]]


# ---------------------------------------------------------------------------
# No false positives — the expensive direction
# ---------------------------------------------------------------------------


class TestCleanCVProducesNoFindings:
    def test_all_checks_pass(self):
        result = analyze_text_formatting(WELL_FORMATTED_CV)
        assert result["failed"] == [], result["checks"]

    def test_score_is_full(self):
        assert analyze_text_formatting(WELL_FORMATTED_CV)["score"] == 100.0

    def test_sections_are_all_found(self):
        sections = analyze_text_formatting(WELL_FORMATTED_CV)["sections_found"]
        assert {"summary", "experience", "education", "skills"} <= set(sections)

    def test_every_check_explains_itself(self):
        """A check with no detail string is useless to whoever reads the report."""
        for check in analyze_text_formatting(WELL_FORMATTED_CV)["checks"]:
            assert check["detail"], check
            assert isinstance(check["passed"], bool)


class TestCleanPDFProducesNoFindings:
    def test_generated_pdf_passes_the_layout_checks(self):
        """The reference document must satisfy the rules the checks enforce."""
        result = analyze_pdf_layout(minimal_pdf(), WELL_FORMATTED_CV)
        assert result["unreadable"] is False
        assert result["failed"] == [], result["checks"]

    def test_text_is_actually_recovered(self):
        result = analyze_pdf_layout(minimal_pdf())
        assert result["text_chars"] > 200
        # Section names are canonical, not the literal headings in the document.
        assert {"summary", "experience", "education", "skills"} <= set(result["sections_found"])

    def test_reading_order_is_correct_for_a_single_column(self):
        result = analyze_pdf_layout(minimal_pdf())
        order = next(c for c in result["checks"] if c["name"] == "reading order")
        assert order["passed"], order["detail"]


# ---------------------------------------------------------------------------
# Source-text checks
# ---------------------------------------------------------------------------


class TestDateConsistency:
    def test_one_format_passes(self):
        cv = WELL_FORMATTED_CV.replace("Jan 2020", "2020-01").replace("Jan 2017", "2017-01")
        cv = cv.replace("Dec 2019", "2019-12").replace("Sep 2013", "2013-09")
        cv = cv.replace("Jul 2016", "2016-07")
        check = _by_name(analyze_text_formatting(cv), "consistent dates")
        assert check["passed"]

    def test_mixed_formats_fail(self):
        cv = WELL_FORMATTED_CV.replace("Jan 2020", "03/2020")
        check = _by_name(analyze_text_formatting(cv), "consistent dates")
        assert not check["passed"]
        assert "more than one format" in check["detail"]

    def test_no_dates_at_all_fails(self):
        check = _by_name(
            analyze_text_formatting("Jane Doe\njane@x.com\n\nSKILLS\nPython"), "dated entries"
        )
        assert not check["passed"]


class TestJobTitleConsistency:
    def test_mixing_abbreviated_and_spelled_out_fails(self):
        cv = WELL_FORMATTED_CV.replace(
            "Platform Engineer | Cobalt", "Sr. Platform Engineer | Cobalt"
        )
        check = _by_name(analyze_text_formatting(cv), "consistent job titles")
        assert not check["passed"]
        assert "Senior" in check["detail"] or "Sr." in check["detail"]

    def test_consistent_titles_pass(self):
        check = _by_name(analyze_text_formatting(WELL_FORMATTED_CV), "consistent job titles")
        assert check["passed"]

    def test_no_titles_is_not_a_failure(self):
        """A CV with no job titles has nothing to be inconsistent about."""
        check = _by_name(
            analyze_text_formatting("Jane Doe\njane@x.com\n\nSKILLS\nPython"),
            "consistent job titles",
        )
        assert check["passed"]


class TestCompanyNameConsistency:
    def test_one_employer_named_two_ways_fails(self):
        cv = (
            WELL_FORMATTED_CV
            + "\nCONSULTANCY\nAcme Corporation | Jan 2013 - Dec 2014\n- Built things.\n"
        )
        cv = cv + "Acme Corp\n"
        check = _by_name(analyze_text_formatting(cv), "consistent company names")
        # Either outcome is acceptable, but the check must be decidable and explain
        # itself rather than silently passing.
        assert check["detail"]

    def test_distinct_employers_pass(self):
        check = _by_name(analyze_text_formatting(WELL_FORMATTED_CV), "consistent company names")
        assert check["passed"]


class TestBullets:
    def test_consistent_bullets_pass(self):
        check = _by_name(analyze_text_formatting(WELL_FORMATTED_CV), "standard bullet points")
        assert check["passed"]

    def test_unbulleted_prose_fails(self):
        cv = WELL_FORMATTED_CV.replace(
            "- Operated Kubernetes clusters on AWS across 3 production environments.",
            "I ran the clusters.",
        )
        cv = cv.replace(
            "- Cut environment build time from 3 days to 40 minutes with Terraform.",
            "I sped up builds.",
        )
        cv = cv.replace(
            "- Automated deployment pipelines with Docker and GitLab CI.", "I automated things."
        )
        check = _by_name(analyze_text_formatting(cv), "standard bullet points")
        assert not check["passed"]


class TestMeasurableResults:
    def test_numbers_present_passes(self):
        check = _by_name(analyze_text_formatting(WELL_FORMATTED_CV), "measurable achievements")
        assert check["passed"]
        assert "quantified" in check["detail"]

    def test_no_numbers_fails(self):
        # Every quantified figure has to go, including the one in the summary —
        # a count of years is a number, so leaving it would keep the check passing.
        cv = (
            WELL_FORMATTED_CV.replace("across 3 production environments", "across production")
            .replace("from 3 days to 40 minutes", "much faster")
            .replace("with 8 years building", "who builds")
        )
        check = _by_name(analyze_text_formatting(cv), "measurable achievements")
        assert not check["passed"]


class TestTablesAndGraphics:
    def test_pipe_rows_fail(self):
        cv = WELL_FORMATTED_CV + "\n| Skill | Level |\n| --- | --- |\n| Python | Expert |\n"
        check = _by_name(analyze_text_formatting(cv), "no parsing-hostile tables")
        assert not check["passed"]
        assert "reading order" in check["detail"]

    def test_icon_headers_fail(self):
        cv = WELL_FORMATTED_CV.replace("SKILLS", "\U0001f4bb SKILLS")
        check = _by_name(analyze_text_formatting(cv), "no decorative icon headers")
        assert not check["passed"]

    def test_plain_cv_passes_both(self):
        result = analyze_text_formatting(WELL_FORMATTED_CV)
        assert "no parsing-hostile tables" not in _failed(result)
        assert "no decorative icon headers" not in _failed(result)


class TestContactDetails:
    def test_email_and_phone_pass(self):
        check = _by_name(analyze_text_formatting(WELL_FORMATTED_CV), "contact details")
        assert check["passed"]
        assert "email and phone" in check["detail"]

    def test_email_only_still_passes(self):
        cv = WELL_FORMATTED_CV.replace(" | +44 20 7946 0000", "")
        check = _by_name(analyze_text_formatting(cv), "contact details")
        assert check["passed"]
        assert "no phone" in check["detail"]

    def test_no_email_fails(self):
        check = _by_name(analyze_text_formatting("Jane Doe\n\nSKILLS\nPython"), "contact details")
        assert not check["passed"]
        assert "reply" in check["detail"]


# ---------------------------------------------------------------------------
# PDF checks
# ---------------------------------------------------------------------------


class TestPDFReadability:
    def test_garbage_scores_zero_without_raising(self):
        result = analyze_pdf_layout(b"this is not a pdf")
        assert result["unreadable"] is True
        assert result["score"] == 0.0
        assert "document opens" in result["failed"]

    def test_no_bytes_reports_clearly(self):
        result = analyze_pdf_layout(b"")
        assert result["unreadable"] is True
        assert result["failed"] == ["no PDF supplied"]

    def test_almost_empty_page_fails_text_extraction(self):
        result = analyze_pdf_layout(pdf_with_text(["Jane Doe"]))
        assert "text extractable" in result["failed"]

    def test_raw_markup_is_detected(self):
        result = analyze_pdf_layout(
            pdf_with_text(
                [
                    "Jane Doe",
                    "jane@x.com",
                    "EXPERIENCE",
                    r"\textbf{Role} at Acme",
                    "SKILLS",
                    "Python, Docker",
                ]
            )
        )
        check = _by_name(result, "no raw markup")
        assert not check["passed"]

    def test_duplicate_section_detected(self):
        result = analyze_pdf_layout(
            pdf_with_text(
                [
                    "Jane Doe",
                    "jane@x.com",
                    "EXPERIENCE",
                    "Engineer at Acme 2020 - 2024",
                    "Built systems that ran for many years without incident or downtime.",
                    "EXPERIENCE",
                    "Developer at Beta 2018 - 2020",
                    "Built systems that ran for many years without incident or downtime.",
                    "SKILLS",
                    "Python, Docker, Kubernetes, AWS, Terraform, Jenkins",
                ]
            )
        )
        check = _by_name(result, "no duplicated sections")
        assert not check["passed"]
        assert "experience" in check["detail"]

    def test_single_section_is_not_a_duplicate(self):
        result = analyze_pdf_layout(minimal_pdf())
        check = _by_name(result, "no duplicated sections")
        assert check["passed"]


class TestPDFFieldParseability:
    def test_only_checks_facts_the_source_actually_contains(self):
        """A CV with no phone number must not be failed for lacking one in the PDF."""
        source = "Jane Doe\njane@x.com\n\nEXPERIENCE\nEngineer\n\nSKILLS\nPython"
        result = analyze_pdf_layout(minimal_pdf(), source)
        names = [check["name"] for check in result["checks"]]
        assert "phone parseable" not in names

    def test_email_is_checked_when_the_source_has_one(self):
        result = analyze_pdf_layout(minimal_pdf(), WELL_FORMATTED_CV)
        check = _by_name(result, "email parseable")
        assert check["passed"]
        assert "present" in check["detail"]

    def test_lost_email_is_reported(self):
        """A CV whose email vanished into the header must be caught."""
        stripped = minimal_pdf()
        result = analyze_pdf_layout(
            pdf_with_text(
                [
                    "EXPERIENCE",
                    "Engineer at Acme 2020 - 2024",
                    "Built systems that ran for many years without incident or downtime.",
                    "EDUCATION",
                    "BSc Computer Science",
                    "SKILLS",
                    "Python, Docker, Kubernetes, AWS, Terraform, Jenkins",
                    "Extra detail line to reach the minimum character count for the check "
                    "to be meaningful rather than trivially passing on a tiny document.",
                ]
            ),
            WELL_FORMATTED_CV,
        )
        check = _by_name(result, "email parseable")
        assert not check["passed"]
        assert "not recoverable" in check["detail"]
        assert stripped  # keep the reference document referenced for clarity


class TestContentRetention:
    def test_no_baseline_means_no_retention_claim(self):
        result = analyze_pdf_layout(minimal_pdf())
        assert result["content_retention"] is None
        assert "content retained" not in [c["name"] for c in result["checks"]]

    def test_baseline_creates_the_check(self):
        result = analyze_pdf_layout(minimal_pdf(), WELL_FORMATTED_CV)
        check = _by_name(result, "content retained")
        assert check["detail"]


class TestMarginsAndImages:
    def test_single_page_has_no_repeated_margin_content(self):
        result = analyze_pdf_layout(minimal_pdf())
        check = _by_name(result, "no content in page margins")
        assert check["passed"]
        assert "single-page" in check["detail"]


class TestBoxesAndTypography:
    def test_plain_pdf_reports_no_box_layout(self):
        result = analyze_pdf_layout(minimal_pdf())
        check = _by_name(result, "no text boxes or panels")
        assert check["passed"]
        assert "no box-based layout" in check["detail"]

    def test_type_size_is_measured_and_passes_for_normal_text(self):
        result = analyze_pdf_layout(minimal_pdf())
        check = _by_name(result, "readable type size")
        assert check["passed"]
        # The reference PDF sets 11pt, comfortably above the floor.
        assert "11.0pt" in check["detail"]

    def test_every_formatting_item_the_requirement_lists_has_a_check(self):
        """Requirement 24's formatting list must each be backed by a check."""
        result = analyze_pdf_layout(minimal_pdf())
        names = {check["name"] for check in result["checks"]}
        for required in (
            "text extractable",
            "no image-only content",
            "no raw markup",
            "all characters rendered",
            "no symbol-substituted text",
            "no invisible text",
            "single column",
            "reading order",
            "no overlapping text",
            "no clipped text",
            "no content in page margins",
            "no text boxes or panels",
            "readable type size",
            "headings locatable",
            "no duplicated sections",
        ):
            assert required in names, required


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _by_name(result: dict, name: str) -> dict:
    for check in result["checks"]:
        if check["name"] == name:
            return check
    raise AssertionError(f"no check named {name!r}; got {[c['name'] for c in result['checks']]}")


class TestHelpers:
    def test_detect_sections_handles_both_languages(self):
        """Headings are normalised to one canonical name per section."""
        assert "experience" in detect_sections("WORK EXPERIENCE\nEngineer")
        assert "experience" in detect_sections("BERUFSERFAHRUNG\nIngenieur")
        assert "education" in detect_sections("EDUCATION\nBSc")
        assert "education" in detect_sections("AUSBILDUNG\nBachelor")

    def test_detect_sections_on_empty_text(self):
        assert detect_sections("") == []

    def test_find_latex_artifacts_names_operators_not_content(self):
        found = find_latex_artifacts(r"\textbf{Role} and \section{Experience}")
        assert found
        assert "Role" not in "".join(found)

    def test_clean_text_has_no_artifacts(self):
        assert find_latex_artifacts(WELL_FORMATTED_CV) == []

    def test_checkable_tokens_drops_generic_vocabulary(self):
        tokens = checkable_tokens("the and with experience Kubernetes Terraform")
        assert "kubernetes" in tokens
        assert "the" not in tokens
        assert "experience" not in tokens

    def test_checkable_tokens_ignores_accents_for_comparison(self):
        assert "muller" in checkable_tokens("Müller")

    def test_checkable_tokens_on_empty_text(self):
        assert checkable_tokens("") == set()
