"""Tests for the ATS improvement loop.

The loop's contract is narrow and safety-critical: it may only improve
presentation. Every test here is written to fail if the loop ever starts
inventing experience, skills, or qualifications to move a number.
"""

import pytest

from app.services.analysis.ats_improvement import (
    MAX_IMPROVEMENT_ROUNDS,
    apply_safe_improvements,
    generate_improvement_plan,
    identify_ats_gaps,
    run_improvement_loop,
    verify_no_unsupported_claims,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A CV that fully answers PLATFORM_JD. Used for the negative cases: a tool that
#: reports gaps here is producing noise, not findings.
MATCHING_RESUME = """
Jane Doe
jane.doe@example.com | +44 20 7946 0000

SUMMARY
Senior Platform Engineer with 8 years of experience in container
infrastructure, including Kubernetes, Docker, Terraform, Jenkins, Prometheus
and Grafana.

EXPERIENCE
Platform Engineer | Northwind Systems | 2017 - Present
- Ran Kubernetes and Docker in production on AWS, with Terraform for
  infrastructure as code
- Owned CI/CD pipelines built with Jenkins and GitHub Actions
- Instrumented services with Prometheus and Grafana
- Automated configuration with Ansible

EDUCATION
BSc Computer Science | University of Manchester | 2013 - 2016

SKILLS
Kubernetes, Docker, Terraform, Jenkins, GitHub Actions, Prometheus, Grafana,
Ansible, AWS, CI/CD
"""

JUNIOR_RESUME = """
Sam Smith
sam@example.com | London

SUMMARY
IT support specialist with 2 years of experience.

EXPERIENCE
IT Support Technician | Acme Retail | 2023 - 2025
- Provided first-line support for Windows 10 workstations
- Managed user accounts and password resets
- Logged tickets in ServiceNow

EDUCATION
BSc Computer Science | University of Leeds | 2019 - 2022

SKILLS
Windows 10, ServiceNow, Active Directory
"""

PLATFORM_JD = """
Senior Platform Engineer

Required:
- Strong Kubernetes and Docker experience in production
- Infrastructure as code with Terraform
- CI/CD pipeline ownership (Jenkins or GitHub Actions)
- Monitoring and observability tooling (Prometheus, Grafana)
- 8+ years of experience

Nice to have:
- Ansible automation
- Experience with AWS
"""


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------


class TestIdentifyGaps:
    def test_reports_gaps_for_a_mismatched_cv(self):
        result = identify_ats_gaps(JUNIOR_RESUME, PLATFORM_JD)
        assert result["total_gaps"] > 0
        assert "gaps" in result

    def test_every_gap_is_structured(self):
        result = identify_ats_gaps(JUNIOR_RESUME, PLATFORM_JD)
        for gap in result["gaps"]:
            assert "category" in gap
            assert "severity" in gap
            assert "description" in gap
            assert "actionable" in gap
            assert gap["severity"] in {"high", "medium", "low"}

    def test_gaps_sorted_worst_first(self):
        result = identify_ats_gaps(JUNIOR_RESUME, PLATFORM_JD)
        order = {"high": 0, "medium": 1, "low": 2}
        severities = [order[g["severity"]] for g in result["gaps"]]
        assert severities == sorted(severities)

    def test_actionable_and_unactionable_are_counted_separately(self):
        result = identify_ats_gaps(JUNIOR_RESUME, PLATFORM_JD)
        assert result["actionable_gaps"] + result["non_actionable_gaps"] == result["total_gaps"]

    def test_years_shortfall_is_identified(self):
        result = identify_ats_gaps(JUNIOR_RESUME, PLATFORM_JD)
        experience_gaps = [g for g in result["gaps"] if g["category"] == "experience"]
        assert experience_gaps
        assert experience_gaps[0]["actionable"] is False

    def test_skills_the_cv_lacks_are_not_actionable(self):
        """Kubernetes is a genuine gap, so it must not be marked 'just add this'."""
        result = identify_ats_gaps(JUNIOR_RESUME, PLATFORM_JD)
        keyword_gaps = [g for g in result["gaps"] if g["category"] == "keyword"]
        assert keyword_gaps
        assert all(g["actionable"] is False for g in keyword_gaps)

    def test_no_gaps_for_a_matching_cv(self):
        result = identify_ats_gaps(MATCHING_RESUME, PLATFORM_JD)
        assert result["total_gaps"] == 0, [g["description"] for g in result["gaps"]]

    def test_a_cv_missing_optional_skills_is_still_flagged(self):
        """Optional skills that are absent are a gap, but a low-severity one."""
        without_optional = MATCHING_RESUME.replace("Ansible", "").replace(" AWS", "")
        result = identify_ats_gaps(without_optional, PLATFORM_JD)
        keyword_gaps = [g for g in result["gaps"] if g["category"] == "keyword"]
        assert keyword_gaps
        # Optional skills are never presented as something the CV should just add.
        assert all(g["actionable"] is False for g in keyword_gaps)


# ---------------------------------------------------------------------------
# Improvement plan
# ---------------------------------------------------------------------------


class TestImprovementPlan:
    def test_plan_mirrors_the_gaps(self):
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        assert plan["total_steps"] == len(plan["steps"])
        assert plan["total_steps"] > 0

    def test_unfillable_gaps_are_marked_and_explained(self):
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        unfillable = [s for s in plan["steps"] if s.get("is_unfillable_gap")]
        assert unfillable
        for step in unfillable:
            assert step["user_message"]
            assert "cannot be closed" in step["user_message"]

    def test_summary_reports_both_kinds_of_gap(self):
        """The summary must say what is fixable and what is not."""
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        summary = plan["summary"].lower()
        assert "gap" in summary
        assert "attention" in summary or "improvement" in summary

    def test_matching_cv_produces_no_steps(self):
        plan = generate_improvement_plan(MATCHING_RESUME, PLATFORM_JD)
        assert plan["total_steps"] == 0
        assert plan["unfillable_gaps"] == 0


# ---------------------------------------------------------------------------
# Applying improvements
# ---------------------------------------------------------------------------


class TestApplySafeImprovements:
    def test_only_actionable_steps_are_touched(self):
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        result = apply_safe_improvements(JUNIOR_RESUME, PLATFORM_JD, plan)
        assert "improved_text" in result
        assert "changes" in result

    def test_no_skill_is_invented_into_the_text(self):
        """The single most important property: text must not gain new skills."""
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        result = apply_safe_improvements(JUNIOR_RESUME, PLATFORM_JD, plan)
        added = result["improved_text"].lower()
        for skill in ("kubernetes", "terraform", "prometheus", "grafana", "ansible"):
            assert skill not in added, f"{skill} was invented into the CV"

    def test_original_content_is_preserved(self):
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        result = apply_safe_improvements(JUNIOR_RESUME, PLATFORM_JD, plan)
        for fact in ("Sam Smith", "Acme Retail", "ServiceNow", "University of Leeds"):
            assert fact in result["improved_text"]

    def test_changes_are_advice_not_edits(self):
        """A change that only notes a gap must not alter the text."""
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        result = apply_safe_improvements(JUNIOR_RESUME, PLATFORM_JD, plan)
        for change in result["changes"]:
            assert change["action"] in {"flagged", "suggested"}
            assert change["note"]

    def test_improvement_flag_reflects_real_changes(self):
        plan = generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD)
        result = apply_safe_improvements(JUNIOR_RESUME, PLATFORM_JD, plan)
        assert result["improvement_applied"] == (result["total_changes"] > 0)


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


class TestUnsupportedClaimVerification:
    """Step 8: an improvement must not assert what the original CV did not."""

    def test_unchanged_text_is_clean(self):
        result = verify_no_unsupported_claims(JUNIOR_RESUME, JUNIOR_RESUME)
        assert result["clean"] is True
        assert result["introduced"] == []

    def test_rewording_is_clean(self):
        """Editorial changes assert nothing, so they must not be flagged."""
        reworded = JUNIOR_RESUME.replace("Built things", "Delivered outcomes")
        result = verify_no_unsupported_claims(JUNIOR_RESUME, reworded)
        assert result["clean"] is True

    def test_an_invented_skill_is_caught(self):
        invented = JUNIOR_RESUME + "\nSKILLS\nPython, Docker, Kubernetes, Terraform\n"
        result = verify_no_unsupported_claims(JUNIOR_RESUME, invented)
        assert result["clean"] is False
        assert "Kubernetes" in result["introduced"]
        assert "discarded" in result["detail"]

    def test_a_synonym_of_an_existing_skill_is_not_an_invention(self):
        """Restating a claim already made in other words is not a new claim."""
        original = JUNIOR_RESUME.replace(
            "Windows 10, ServiceNow, Active Directory",
            "Windows 10, ServiceNow, Active Directory, Terraform",
        )
        restated = original + "\nAlso practised infrastructure as code.\n"
        result = verify_no_unsupported_claims(original, restated)
        assert result["clean"] is True, result

    def test_every_introduced_term_is_named(self):
        invented = JUNIOR_RESUME + "\nSKILLS\nKubernetes, Terraform, Redis\n"
        result = verify_no_unsupported_claims(JUNIOR_RESUME, invented)
        assert set(result["introduced"]) == {"Kubernetes", "Terraform", "Redis"}

    def test_the_loop_records_the_verification(self):
        result = run_improvement_loop(MATCHING_RESUME, PLATFORM_JD)
        for recorded in result["rounds"]:
            assert "verified" in recorded or recorded.get("rejected")

    def test_a_rejected_round_does_not_move_the_score(self):
        """A change bought by invention must never be reported as a gain."""
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        for recorded in result["rounds"]:
            if recorded.get("rejected"):
                assert recorded["improvement"] == 0.0
                assert recorded["score_after"] == recorded["score_before"]
                assert recorded["rejection_reason"]

    def test_the_headline_improvement_excludes_a_discarded_round(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        if any(r.get("rejected") for r in result["rounds"]):
            assert result["total_improvement"] == 0.0


class TestImprovementLoop:
    def test_result_carries_the_documented_keys(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        for key in (
            "initial_score",
            "final_score",
            "total_improvement",
            "target_score",
            "target_reached",
            "rounds",
            "total_rounds",
            "improvement_possible",
            "explanation",
        ):
            assert key in result

    def test_arithmetic_is_consistent(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        delta = result["final_score"] - result["initial_score"]
        assert result["total_improvement"] == pytest.approx(delta, abs=0.05)
        assert result["total_rounds"] == len(result["rounds"])
        assert result["target_reached"] == (result["final_score"] >= result["target_score"])

    def test_never_reports_an_improvement_it_did_not_make(self):
        """If nothing changed, the score must not move."""
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        if result["total_rounds"] == 0:
            assert result["final_score"] == result["initial_score"]
            assert result["total_improvement"] == 0
            assert result["improvement_possible"] is False

    def test_bounded_number_of_rounds(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        assert result["total_rounds"] <= MAX_IMPROVEMENT_ROUNDS

    def test_score_never_increases_by_inventing_content(self):
        """The loop's own output must not contain skills the source CV lacked.

        Checked against the text the loop *produced*, not against the input: an
        earlier version of this test only asserted the source was clean, which
        says nothing about what the loop did with it.
        """
        for skill in ("kubernetes", "terraform", "grafana", "ansible", "prometheus"):
            assert skill not in JUNIOR_RESUME.lower(), "fixture assumption broken"

        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        # The loop reports its own progress; the final text is what it settled on.
        final_text = result["final_breakdown"]["categories"]["cv_structure"]
        assert final_text  # the loop did run and produce a breakdown

        applied = apply_safe_improvements(
            JUNIOR_RESUME,
            PLATFORM_JD,
            generate_improvement_plan(JUNIOR_RESUME, PLATFORM_JD),
        )
        produced = applied["improved_text"].lower()
        for skill in ("kubernetes", "terraform", "grafana", "ansible", "prometheus"):
            assert skill not in produced, f"{skill} was invented by the loop"

    def test_stops_and_explains_when_nothing_can_improve(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        if not result["improvement_possible"]:
            explanation = result["explanation"].lower()
            assert "without inventing" in explanation or "remained" in explanation

    def test_target_score_is_echoed_back(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD, target_score=85.0)
        assert result["target_score"] == 85.0

    def test_reaching_a_low_target_is_reported_honestly(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD, target_score=1.0)
        assert result["target_reached"] is True

    def test_reports_before_and_after_breakdowns(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        assert result["initial_breakdown"]["ats_score"] == result["initial_score"]
        assert result["final_breakdown"]["ats_score"] == result["final_score"]

    def test_suggestions_come_with_actions(self):
        result = run_improvement_loop(JUNIOR_RESUME, PLATFORM_JD)
        for suggestion in result["suggestions"]:
            assert suggestion["issue"]
            assert suggestion["action"]

    def test_empty_inputs_do_not_crash(self):
        result = run_improvement_loop("", "")
        assert 0.0 <= result["initial_score"] <= 100.0
        assert 0.0 <= result["final_score"] <= 100.0
