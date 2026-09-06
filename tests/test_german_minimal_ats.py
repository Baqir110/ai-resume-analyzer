import shutil

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.latex_generator import (
    FactualValidationError,
    compile_latex_to_pdf,
    extract_resume_invariants,
    generate_german_latex_content,
)


RESUME = """Jane Doe
jane@example.com | +49 123 456789
https://www.linkedin.com/in/janedoe

Experience
Senior Platform Engineer | Example GmbH | Jan 2020 - Present

Education
M.Sc. Computer Science | Technical University | 2018 - 2020
"""

VALID_BODY = r"""
\section*{Profil}
\section*{Berufserfahrung}
\jobheader{Senior Platform Engineer}{Example GmbH}{Jan 2020 - Present}
\section*{Ausbildung}
\jobheader{M.Sc. Computer Science}{Technical University}{2018 - 2020}
"""


def _generate_body(*_args, **_kwargs):
    return VALID_BODY


def test_extract_resume_invariants_keeps_work_and_degree_facts():
    assert extract_resume_invariants(RESUME) == [
        "Senior Platform Engineer",
        "Example GmbH",
        "Jan 2020 - Present",
        "M.Sc. Computer Science | Technical University | 2018 - 2020",
    ]


def test_extract_resume_invariants_rejects_an_unparseable_experience_section():
    resume = """Jane Doe
Experience
Senior Platform Engineer | Example GmbH

Education
M.Sc. Computer Science | Technical University | 2018 - 2020
"""

    with pytest.raises(FactualValidationError, match="Experience section"):
        extract_resume_invariants(resume)


def test_german_minimal_ats_uses_candidate_header_and_validates_facts(monkeypatch):
    monkeypatch.setattr(
        "app.services.latex_generator.LLMService.generate", _generate_body
    )

    latex = generate_german_latex_content(
        resume_text=RESUME,
        job_description="Platform engineering role",
        missing_skills=[],
        layout_style="german_minimal_ats",
    )

    assert "Jane Doe" in latex
    assert "Muhammad Baqir" not in latex
    assert "Senior Platform Engineer" in latex
    assert "Example GmbH" in latex


def test_german_minimal_ats_rejects_altered_facts(monkeypatch):
    monkeypatch.setattr(
        "app.services.latex_generator.LLMService.generate",
        lambda *_args, **_kwargs: VALID_BODY.replace("Example GmbH", "Different GmbH"),
    )

    with pytest.raises(FactualValidationError, match="Example GmbH"):
        generate_german_latex_content(
            resume_text=RESUME,
            job_description="Platform engineering role",
            missing_skills=[],
            layout_style="german_minimal_ats",
        )


def test_german_minimal_ats_does_not_fallback_after_llm_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("app.services.latex_generator.LLMService.generate", fail)

    with pytest.raises(FactualValidationError, match="no CV was produced"):
        generate_german_latex_content(
            resume_text=RESUME,
            job_description="Platform engineering role",
            missing_skills=[],
            layout_style="german_minimal_ats",
        )


@pytest.mark.parametrize(
    "endpoint",
    ["/api/v1/resume/generate-german-cv", "/api/v1/resume/generate-tex-cv"],
)
def test_generation_endpoints_return_422_for_factual_validation_failure(
    monkeypatch, endpoint
):
    monkeypatch.setattr(
        "app.api.endpoints.generate_german_latex_content",
        lambda **_kwargs: (_ for _ in ()).throw(
            FactualValidationError(["Generated CV is missing or alters: Example GmbH"])
        ),
    )

    response = TestClient(app).post(
        endpoint,
        data={"job_description": "Platform engineering role", "layout_style": "german_minimal_ats"},
        files={"resume_file": ("resume.txt", RESUME.encode(), "text/plain")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "factual_validation_failed",
        "violations": ["Generated CV is missing or alters: Example GmbH"],
    }


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_german_minimal_ats_compiles_with_local_pdflatex(monkeypatch):
    monkeypatch.setattr(
        "app.services.latex_generator.LLMService.generate", _generate_body
    )
    latex = generate_german_latex_content(
        resume_text=RESUME,
        job_description="Platform engineering role",
        missing_skills=[],
        layout_style="german_minimal_ats",
    )

    assert compile_latex_to_pdf(latex).startswith(b"%PDF")
