"""
Structural health must not depend on the language the CV is written in.

``check_resume_structure`` required the English header for each section, so a
German Lebenslauf was charged 45 points for "missing" the three sections it
plainly had. That fed the 15% structural weight of the ATS score, and a run on a
German CV reported 8.82% against a well-matched one -- a scoring artefact, not a
finding about the candidate.
"""

from __future__ import annotations

from app.services.analysis.ats_analyzer import check_resume_structure

#: The body is padded past the 300-character brevity threshold so the only thing
#: the assertions below can be reacting to is the section headers.
_GERMAN_RESUME = (
    "HANS MUELLER\n"
    "hans.mueller@example.de | Muenchen\n"
    "\n"
    "BERUFSERFAHRUNG\n"
    "2020 - heute, Senior Netzwerk Engineer, Beispiel GmbH.\n"
    "Verantwortlich fuer Routing, BGP, OSPF, MPLS und VLAN-Konfiguration.\n"
    "\n"
    "AUSBILDUNG\n"
    "2016 - 2020, Bachelor Informatik, Universitaet Bayern.\n"
    "\n"
    "FAEHIGKEITEN\n"
    "Python, FastAPI, Kubernetes, Docker, PostgreSQL, Ansible, Terraform.\n"
    "Linuxadministration, Shell Scripting, CI/CD mit Jenkins und Git.\n"
    " Monitoring mit SNMP und Syslog, Load Balancing, Firewall-Konfiguration.\n"
)

_ENGLISH_RESUME = (
    "HANS MUELLER\n"
    "hans.mueller@example.de | Munich\n"
    "\n"
    "EXPERIENCE\n"
    "2020 - present, Senior Network Engineer, Example GmbH.\n"
    "Responsible for routing, BGP, OSPF, MPLS and VLAN configuration.\n"
    "\n"
    "EDUCATION\n"
    "2016 - 2020, Bachelor Computer Science, University of Bavaria.\n"
    "\n"
    "SKILLS\n"
    "Python, FastAPI, Kubernetes, Docker, PostgreSQL, Ansible, Terraform.\n"
    "Linux administration, shell scripting, CI/CD with Jenkins and Git.\n"
    "Monitoring with SNMP and Syslog, load balancing, firewall configuration.\n"
)


def test_a_german_lebenslauf_is_not_charged_for_english_headers():
    """The regression: 45 structural points lost to a language, not a defect."""
    result = check_resume_structure(_GERMAN_RESUME)

    assert not any(
        "section header" in warning for warning in result["warnings"]
    ), f"German headers were not recognised: {result['warnings']}"


def test_german_and_english_resumes_score_identically():
    """
    The two documents carry the same content, so the structural signal must not
    be able to tell them apart.
    """
    assert check_resume_structure(_GERMAN_RESUME)["structure_score"] == check_resume_structure(
        _ENGLISH_RESUME
    )["structure_score"]


def test_the_common_german_headings_all_count():
    """
    Each accepted spelling on its own. The snippet holds a single heading, so the
    assertion is that *that* section is no longer reported missing -- the other
    two legitimately are, and would be in any document this short.
    """
    accepted = {
        "experience": ("BERUFSERFAHRUNG", "Arbeitserfahrung", "Experience"),
        "education": ("Ausbildung", "Studium", "Education"),
        "skills": ("FÄHIGKEITEN", "FAEHIGKEITEN", "Kenntnisse", "Skills"),
    }

    for section, headings in accepted.items():
        for heading in headings:
            body = f"{heading}\nPython, Kubernetes, Docker, Ansible und Terraform.\n"
            warnings = check_resume_structure(body)["warnings"]

            assert not any(
                f"'{section.capitalize()}'" in warning for warning in warnings
            ), f"{heading} was not accepted as a '{section}' header: {warnings}"


def test_a_genuinely_broken_resume_is_still_penalised():
    """
    The guard on the guard. Accepting German must not make the check lenient: a
    document with nothing in it still loses all three headers, the missing
    contact detail and the brevity penalty.
    """
    result = check_resume_structure("")

    # 100 less three headers (15 each), the missing contact detail (10), and the
    # brevity penalty (20).
    assert result["structure_score"] == 25.0
    assert len([w for w in result["warnings"] if "section header" in w]) == 3, (
        "every section must still be reported missing on an empty document"
    )
