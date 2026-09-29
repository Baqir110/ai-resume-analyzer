"""Tests for the transparent ATS scoring engine.

Covers the contract the rest of the application depends on:
``compute_ats_breakdown`` returning per-category scores, weights, weighted
points, points lost, and a top-level score reproducible from them.
"""

import pytest

from app.services.analysis.ats_scoring import (
    IMPROVEMENT_THRESHOLD,
    compute_ats_breakdown,
    generate_user_friendly_suggestions,
    get_ats_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A CV that is both a strong match for MATCHING_JD and conventionally laid out.
#: Deliberately built to pass every formatting check, so the negative tests below
#: test the absence of false positives rather than the presence of findings.
STRONG_RESUME = """
Jane Doe
Senior Platform Engineer
jane.doe@example.com | +44 20 7946 0000 | London, UK

PROFESSIONAL SUMMARY
Senior Platform Engineer with 8 years building and operating container
infrastructure for regulated industries.

WORK EXPERIENCE
Senior Platform Engineer | Northwind Systems | Jan 2020 - Present
- Operated Kubernetes clusters on AWS across 3 production environments,
  holding 99.95% availability.
- Replaced manual provisioning with Terraform and Ansible automation,
  cutting environment build time from 3 days to 40 minutes.
- Built CI/CD pipelines using GitHub Actions and Jenkins, deploying 60
  times per month.
- Instrumented services with Prometheus and Grafana, reducing paging
  volume by 40%.

Platform Engineer | Cobalt Data | Jan 2017 - Dec 2019
- Automated deployment pipelines with Docker and GitLab CI.
- Operated PostgreSQL and Redis in a high-availability topology.

EDUCATION
BSc (Hons) Computer Science | University of Manchester | Sep 2013 - Jul 2016

SKILLS
Python, Go, Docker, Kubernetes, Terraform, Ansible, AWS, Jenkins, GitHub
Actions, GitLab CI, Prometheus, Grafana, PostgreSQL, Redis

CERTIFICATIONS
Certified Kubernetes Administrator
AWS Certified Solutions Architect - Associate
"""

MATCHING_JD = """
Senior Platform Engineer

We are hiring a Senior Platform Engineer to run our container platform.

Required:
- Strong Kubernetes and Docker experience in production
- Infrastructure as code with Terraform
- CI/CD pipeline ownership (Jenkins or GitHub Actions)
- Monitoring and observability tooling (Prometheus, Grafana)

Nice to have:
- Ansible automation
- Experience with AWS

You will collaborate with development teams, mentor engineers, and own the
platform roadmap.
"""

UNRELATED_JD = """
Head of Marketing Communications

We are looking for an experienced communications lead to own our brand
narrative, media strategy and public relations programme. You will manage a
team of writers and designers, own the editorial calendar, and report to the
CMO on brand awareness metrics.

Requirements: 8+ years in marketing communications, agency experience, media
relations, and team leadership. Experience with market research and campaign
measurement strongly preferred.
"""

SPARSE_RESUME = """
Sam Smith
sam@example.com

I have worked in IT for a while.
"""


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------


class TestBreakdownShape:
    def test_top_level_keys(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        for key in (
            "ats_score",
            "max_score",
            "summary",
            "categories",
            "improvement_areas",
            "is_honest_score",
        ):
            assert key in result

    def test_all_five_categories_present(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        assert set(result["categories"]) == {
            "keyword_match",
            "required_skills",
            "experience_relevance",
            "cv_structure",
            "pdf_parsing",
        }

    def test_each_category_reports_its_arithmetic(self):
        """score, weight, weighted_points and points_lost must be consistent."""
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        for name, data in result["categories"].items():
            if data.get("not_measured"):
                continue
            assert data["max_score"] == 100.0
            assert 0.0 <= data["score"] <= 100.0
            assert data["weight"] > 0
            assert data["weighted_points"] == pytest.approx(
                data["score"] * data["weight"], abs=0.05
            )
            assert data["points_lost"] == pytest.approx(
                (100.0 - data["score"]) * data["weight"], abs=0.05
            )
            assert data["explanation"]

    def test_score_is_the_sum_of_weighted_points(self):
        """The headline number must be reconstructible, not asserted."""
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        total = sum(
            data["weighted_points"]
            for data in result["categories"].values()
            if not data.get("not_measured")
        )
        assert result["ats_score"] == pytest.approx(total, abs=0.15)

    def test_weights_sum_to_one_when_everything_measured(self):
        """With a PDF supplied, no weight is redistributed."""
        from tests.test_ats_scoring_helpers import minimal_pdf

        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD, minimal_pdf())
        weights = [d["weight"] for d in result["categories"].values()]
        assert sum(weights) == pytest.approx(1.0, abs=0.01)
        assert result["pre_generation"] is False

    def test_honest_score_flag_set(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        assert result["is_honest_score"] is True


# ---------------------------------------------------------------------------
# Pre-generation behaviour: PDF Parsing cannot be measured
# ---------------------------------------------------------------------------


class TestPreGeneration:
    def test_pdf_category_unmeasured_without_a_pdf(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        pdf = result["categories"]["pdf_parsing"]
        assert pdf["not_measured"] is True
        assert pdf["score"] is None
        assert pdf["weighted_points"] == 0.0
        assert pdf["points_lost"] == 0.0

    def test_pre_generation_flagged_and_explained(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        assert result["pre_generation"] is True
        assert "pre-generation" in result["summary"].lower()

    def test_remaining_weight_is_redistributed(self):
        """Skipping a category must not silently cap the achievable score."""
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        weights = [d["weight"] for d in result["categories"].values() if not d.get("not_measured")]
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_explanation_names_the_skipped_category(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        assert "not measured" in result["categories"]["pdf_parsing"]["explanation"].lower()


# ---------------------------------------------------------------------------
# Honesty: a weak CV must score badly
# ---------------------------------------------------------------------------


class TestScoreIsHonest:
    def test_matching_pair_scores_well(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        assert result["ats_score"] >= 70

    def test_unrelated_pair_scores_badly(self):
        """A CV with none of the posting's content must not score well."""
        result = compute_ats_breakdown(SPARSE_RESUME, UNRELATED_JD)
        assert result["ats_score"] < 55

    def test_sparse_resume_loses_structure_points(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        structure = result["categories"]["cv_structure"]
        assert structure["score"] < 60
        assert "experience" in structure["missing_sections"]
        assert "education" in structure["missing_sections"]

    def test_missing_required_skills_are_reported_as_a_gap(self):
        """A skill the CV lacks is named, never quietly counted as present."""
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        gaps = result["unfillable_gaps"]["required_skills_missing"]
        assert gaps
        assert "Kubernetes" in gaps or "kubernetes" in [g.lower() for g in gaps]

    def test_years_shortfall_is_named_as_a_gap(self):
        """A CV that states its years, but too few, must be flagged as a real gap."""
        junior = (
            "Sam Smith\nsam@example.com\n\nSUMMARY\n2 years in IT support.\n\n"
            "EXPERIENCE\nIT Support, Acme, 2023 - 2024\n\n"
            "EDUCATION\nBSc\n\nSKILLS\nExcel\n"
        )
        result = compute_ats_breakdown(junior, UNRELATED_JD)
        assert result["categories"]["experience_relevance"]["years_score"] is not None
        assert result["unfillable_gaps"]["years_of_experience_short"] is True
        assert "years" in result["summary"].lower()

    def test_empty_inputs_do_not_crash(self):
        result = compute_ats_breakdown("", "")
        assert 0.0 <= result["ats_score"] <= 100.0
        assert result["categories"]

    def test_improvement_areas_sorted_by_points_lost(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        lost = [area["points_lost"] for area in result["improvement_areas"]]
        assert lost == sorted(lost, reverse=True)

    def test_unmeasured_category_is_never_an_improvement_area(self):
        """The unmeasured PDF category must not be reported as lost points."""
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        names = {area["category"] for area in result["improvement_areas"]}
        assert "pdf_parsing" not in names


# ---------------------------------------------------------------------------
# Individual categories
# ---------------------------------------------------------------------------


class TestKeywordMatch:
    def test_matched_and_missing_are_separated(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        kw = result["categories"]["keyword_match"]
        assert kw["matched"]
        # "Ansible" is listed as nice-to-have, "Terraform" as required; both are
        # in the CV, so the gaps are whatever the posting mentions and it does not.
        assert isinstance(kw["missing"], list)

    def test_synonym_match_is_reported_as_such(self):
        result = compute_ats_breakdown(
            "Experience with k8s, docker and postgres.",
            "Kubernetes Docker PostgreSQL required.",
        )
        kw = result["categories"]["keyword_match"]
        assert kw["score"] == 100.0
        assert kw["matched_via_synonym"]

    def test_no_keywords_in_posting_scores_full(self):
        """Nothing to match is not a failure of the candidate."""
        result = compute_ats_breakdown(SPARSE_RESUME, "We need a friendly person.")
        assert result["categories"]["keyword_match"]["score"] == 100.0


class TestRequiredSkills:
    def test_required_verbs_recognised(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        skills = result["categories"]["required_skills"]
        assert skills["required_matched"]
        # A skill the CV has, named under "Required:", must land in required_matched.
        assert any("Terraform" in s for s in skills["required_matched"])

    def test_nice_to_have_is_not_required(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        skills = result["categories"]["required_skills"]
        assert "Ansible" not in skills["required_missing"]

    def test_missing_required_costs_more_than_missing_optional(self):
        """Weighted so a required gap dominates, not averages away."""
        without_required = compute_ats_breakdown(
            "Skills: Python, Docker, PostgreSQL, Git.", MATCHING_JD
        )
        without_optional = compute_ats_breakdown(
            "Skills: Terraform, Kubernetes, Jenkins, Prometheus, Grafana, AWS, GitHub Actions.",
            MATCHING_JD,
        )
        assert (
            without_required["categories"]["required_skills"]["score"]
            < without_optional["categories"]["required_skills"]["score"]
        )


class TestExperienceRelevance:
    def test_years_are_extracted_from_both_sides(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        exp = result["categories"]["experience_relevance"]
        assert exp["years_score"] is None or exp["years_score"] > 0
        assert "years_score" in exp

    def test_years_shortfall_is_scored_not_zeroed(self):
        """One year short should read as close, not as no experience."""
        result = compute_ats_breakdown(
            "3 years of experience. Python.", "10 years experience required."
        )
        exp = result["categories"]["experience_relevance"]
        assert 0 < exp["years_score"] < 100

    def test_years_absent_is_not_penalised(self):
        """A CV that never states a total is not evidence of too little."""
        # A CV that describes its history in dates rather than in a number of
        # years — which is the more common way to write one.
        undated_total = STRONG_RESUME.replace("with 8 years building", "who builds")
        result = compute_ats_breakdown(undated_total, "10 years experience required.")
        exp = result["categories"]["experience_relevance"]
        assert exp["years_score"] is None
        assert "could not be verified" in exp["explanation"]


class TestCVStructure:
    def test_complete_cv_passes_every_check(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        structure = result["categories"]["cv_structure"]
        assert structure["score"] == 100.0
        assert not structure["failed_checks"]

    def test_optional_sections_never_demanded(self):
        """A CV with no Projects section is not penalised for lacking one."""
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        structure = result["categories"]["cv_structure"]
        assert "projects" not in structure["missing_sections"]
        assert "projects" not in structure["optional_sections_found"]

    def test_missing_email_is_caught(self):
        result = compute_ats_breakdown(
            "EXPERIENCE\nA role\n\nEDUCATION\nA degree\n\nSKILLS\nPython",
            MATCHING_JD,
        )
        structure = result["categories"]["cv_structure"]
        assert "contact details" in structure["failed_checks"]


class TestPDFParsing:
    def test_measured_once_a_pdf_is_supplied(self):
        from tests.test_ats_scoring_helpers import minimal_pdf

        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD, minimal_pdf())
        pdf = result["categories"]["pdf_parsing"]
        assert pdf["not_measured"] is False
        assert pdf["score"] is not None
        assert pdf["checks"]

    def test_garbage_bytes_score_zero_rather_than_raising(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD, b"not a pdf at all")
        pdf = result["categories"]["pdf_parsing"]
        assert pdf["score"] == 0.0
        assert pdf["issues"]

    def test_retention_only_claimed_when_there_is_a_baseline(self):
        from tests.test_ats_scoring_helpers import minimal_pdf

        without_text = compute_ats_breakdown("", MATCHING_JD, minimal_pdf())
        assert without_text["categories"]["pdf_parsing"]["content_retention"] is None


# ---------------------------------------------------------------------------
# User-facing layer
# ---------------------------------------------------------------------------


class TestJobTitleAlignment:
    def test_matching_title_scores_full_marks(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        alignment = result["categories"]["keyword_match"]["title_alignment"]
        assert alignment["stated"] is True
        assert alignment["aligned"] is True
        assert "engineer" in alignment["shared"]

    def test_seniority_difference_is_still_a_match(self):
        """'Senior X' answers a posting for 'X' — the role is the same."""
        junior = STRONG_RESUME.replace("Senior Platform Engineer", "Platform Engineer")
        result = compute_ats_breakdown(junior, MATCHING_JD)
        assert result["categories"]["keyword_match"]["title_alignment"]["aligned"] is True

    def test_different_role_is_a_mismatch(self):
        other = STRONG_RESUME.replace("Senior Platform Engineer", "Head of Marketing")
        other = other.replace("Platform Engineer, Northwind", "Marketing Lead, Northwind")
        result = compute_ats_breakdown(other, MATCHING_JD)
        alignment = result["categories"]["keyword_match"]["title_alignment"]
        assert alignment["stated"] is True
        assert alignment["aligned"] is False
        assert alignment["detail"]

    def test_posting_without_a_title_is_excluded_not_failed(self):
        """A posting that never names the role cannot fairly be matched on one."""
        result = compute_ats_breakdown(STRONG_RESUME, "We need someone. Python and Docker.")
        alignment = result["categories"]["keyword_match"]["title_alignment"]
        assert alignment["stated"] is False

    def test_a_name_is_not_mistaken_for_a_title(self):
        """'Jane Doe' must not be read as a job title and reported as a mismatch."""
        result = compute_ats_breakdown(STRONG_RESUME, "Marketing Assistant. Python.")
        alignment = result["categories"]["keyword_match"]["title_alignment"]
        assert alignment["candidate"] != "Jane Doe"
        assert alignment["target"] == "Marketing Assistant"


class TestLogistics:
    def test_silent_posting_is_excluded(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        logistics = result["categories"]["keyword_match"]["logistics"]
        assert logistics["stated"] is False
        assert logistics["covered"] is True

    def test_remote_posting_and_remote_cv_is_covered(self):
        cv = STRONG_RESUME.replace("London, UK", "London, UK - open to remote")
        result = compute_ats_breakdown(cv, MATCHING_JD + "\n\nThis role is fully remote.")
        assert result["categories"]["keyword_match"]["logistics"]["covered"] is True

    def test_remote_posting_and_silent_cv_is_a_gap(self):
        result = compute_ats_breakdown(
            STRONG_RESUME, MATCHING_JD + "\n\nThis role is fully remote."
        )
        logistics = result["categories"]["keyword_match"]["logistics"]
        assert logistics["stated"] is True
        assert logistics["covered"] is False
        assert logistics["missing"]

    def test_work_authorisation_requirement_is_noticed(self):
        jd = MATCHING_JD + "\n\nApplicants must have the right to work in the UK."
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        logistics = result["categories"]["keyword_match"]["logistics"]
        assert logistics["work_authorisation_required"]
        assert logistics["covered"] is False

    def test_stating_the_situation_satisfies_the_requirement(self):
        cv = STRONG_RESUME.replace("London, UK", "London, UK - open to remote, right to work")
        jd = (
            MATCHING_JD
            + "\n\nThis role is remote. Applicants must have the right to work in the UK."
        )
        result = compute_ats_breakdown(cv, jd)
        assert result["categories"]["keyword_match"]["logistics"]["covered"] is True

    def test_a_gap_gets_a_suggestion_with_an_action(self):
        result = compute_ats_breakdown(
            STRONG_RESUME, MATCHING_JD + "\n\nThis role is fully remote."
        )
        suggestions = generate_user_friendly_suggestions(result)
        logistics = [s for s in suggestions if s["category"] == "location_and_work_authorisation"]
        assert logistics
        assert "honestly" in logistics[0]["action"]


class TestIndustryTerminology:
    def test_sector_vocabulary_counts_as_keywords(self):
        jd = (
            "Compliance Engineer. You will maintain our GDPR and ISO 27001 posture "
            "across the estate. Must have Python for the reporting tooling."
        )
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        terms = result["categories"]["keyword_match"]["industry_terminology"]
        assert "GDPR" in terms["missing"]
        assert "ISO 27001" in terms["missing"]

    def test_sector_vocabulary_present_in_the_cv_counts(self):
        jd = "Compliance Engineer. Must have Python. Familiarity with GDPR required."
        cv = STRONG_RESUME + "\nCERTIFICATIONS\nCertified in GDPR compliance practice.\n"
        result = compute_ats_breakdown(cv, jd)
        terms = result["categories"]["keyword_match"]["industry_terminology"]
        assert "GDPR" in terms["matched"]

    def test_plain_tech_posting_reports_no_industry_terms(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        terms = result["categories"]["keyword_match"]["industry_terminology"]
        assert terms["matched"] == [] and terms["missing"] == []


class TestStructuredQualifications:
    def test_degree_requirement_is_scored_as_required(self):
        jd = MATCHING_JD + "\n\nA Bachelor's degree in Computer Science is required."
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert quals["stated"] is True
        assert quals["education"]["level"] == "bachelor"
        # The CV holds a BSc, so the requirement is met.
        assert quals["education_satisfied"] is True
        assert quals["missing"] == []

    def test_degree_shortfall_is_a_gap(self):
        jd = MATCHING_JD + "\n\nA Master's degree is required for this role."
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert quals["education"]["level"] == "master"
        assert quals["education_satisfied"] is False
        assert quals["missing"]

    def test_unreadable_degree_is_not_treated_as_none(self):
        """A CV the parser could not read is unverified, not unqualified."""
        jd = MATCHING_JD + "\n\nA Master's degree is required."
        cv = STRONG_RESUME.replace("BSc (Hons) Computer Science", "Studied computing")
        result = compute_ats_breakdown(cv, jd)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert quals["education_satisfied"] is False

    def test_required_language_is_a_gap_when_absent(self):
        jd = MATCHING_JD + "\n\nFluent German at B2 is required."
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert "German" in quals["languages_missing"]

    def test_stated_language_is_credited(self):
        jd = MATCHING_JD + "\n\nFluent German at B2 is required."
        cv = STRONG_RESUME + "\nLANGUAGES\nEnglish (native), German (C1)\n"
        result = compute_ats_breakdown(cv, jd)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert "German" in quals["languages_matched"]
        assert quals["missing"] == []

    def test_named_certification_is_a_gap_when_absent(self):
        jd = MATCHING_JD + "\n\nCertified Kubernetes Administrator is required."
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert quals["certifications_missing"]

    def test_a_posting_with_none_of_these_is_not_penalised(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        quals = result["categories"]["required_skills"]["qualifications"]
        assert quals["stated"] is False
        assert quals["covered"] is True
        assert quals["missing"] == []

    def test_a_gap_appears_in_the_missing_list(self):
        jd = MATCHING_JD + "\n\nA Master's degree is required. German fluency essential."
        result = compute_ats_breakdown(STRONG_RESUME, jd)
        assert result["categories"]["required_skills"]["required_missing"]


class TestKeywordStuffing:
    _BULLET = (
        "- Operated Kubernetes clusters on AWS across 3 production environments,\n"
        "  holding 99.95% availability."
    )

    def _stuffed(self) -> str:
        return STRONG_RESUME.replace(self._BULLET, "- " + ("Kubernetes " * 40))

    def test_a_normal_cv_passes(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        assert "no keyword stuffing" not in result["categories"]["cv_structure"]["failed_checks"]

    def test_repetition_is_caught(self):
        result = compute_ats_breakdown(self._stuffed(), MATCHING_JD)
        assert "no keyword stuffing" in result["categories"]["cv_structure"]["failed_checks"]

    def test_the_failure_is_reported_in_plain_language(self):
        result = compute_ats_breakdown(self._stuffed(), MATCHING_JD)
        check = next(
            c
            for c in result["categories"]["cv_structure"]["checks"]
            if c["name"] == "no keyword stuffing"
        )
        assert not check["passed"]
        assert "kubernetes" in check["detail"]


class TestUserFriendlyLayer:
    def test_every_suggestion_has_an_action(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        suggestions = generate_user_friendly_suggestions(result, ["Kubernetes", "Terraform"])
        assert suggestions
        for suggestion in suggestions:
            assert suggestion["issue"]
            assert suggestion["action"]
            assert suggestion["severity"] in {"high", "medium", "low"}

    def test_no_jargon_in_the_issue_text(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        suggestions = generate_user_friendly_suggestions(result, ["Kubernetes"])
        blob = " ".join(s["issue"] for s in suggestions).lower()
        for term in ("tf-idf", "cosine", "retention ratio", "normalization", "weighting"):
            assert term not in blob

    def test_suggestions_never_tell_the_user_to_invent_experience(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        suggestions = generate_user_friendly_suggestions(result, ["Kubernetes", "Ansible"])
        blob = " ".join(s["action"] for s in suggestions).lower()
        assert "actually have" in blob or "genuinely" in blob

    def test_strong_cv_needs_no_suggestions(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        suggestions = generate_user_friendly_suggestions(result, [])
        assert suggestions == []

    def test_summary_shape(self):
        result = compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)
        summary = get_ats_summary(result)
        for key in (
            "score",
            "band",
            "what_is_working",
            "what_can_improve",
            "primary_action",
            "pre_generation",
        ):
            assert key in summary

    def test_summary_primary_action_names_a_real_category(self):
        result = compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)
        summary = get_ats_summary(result)
        assert summary["what_can_improve"]
        assert summary["primary_action"]

    def test_bands_are_ordered(self):
        bands = [
            compute_ats_breakdown(STRONG_RESUME, MATCHING_JD)["band"],
            compute_ats_breakdown(SPARSE_RESUME, MATCHING_JD)["band"],
        ]
        assert bands[0] != bands[1]
        assert bands[0] == "Excellent match"
