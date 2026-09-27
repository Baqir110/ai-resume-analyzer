"""All CV layouts and cover-letter styles, with actual TeX rendering when installed."""

import io
import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.main import app
from app.services.career import cover_letter_pdf
from app.services.career.cover_letter_templates import list_templates
from app.services.cv import latex_generator
from app.services.cv.pdf_compiler import PDFLayoutError, compile_single_page_pdf

RESUME = """Jane Doe
Platform Engineer
jane@example.com | +49 123 456789
https://www.linkedin.com/in/janedoe
https://github.com/janedoe

Experience
Platform Engineer | Example GmbH | Jan 2020 - Present
Built Python services and monitored deployments.
Education
M.Sc. Computer Science | Technical University | 2018 - 2020
"""
BODY = r"""\section*{Profile}
Platform engineer building Python services and reliable deployments.
\section*{Experience}
\jobheader{Platform Engineer}{Example GmbH}{Jan 2020 - Present}
\begin{itemize}
\item Built Python services and monitored deployments.
\end{itemize}
\section*{Education}
\degreeheader{M.Sc. Computer Science}{Technical University}{2018 - 2020}
\section*{Skills}
Python, deployment monitoring
"""
REAL_TEX = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")


@pytest.fixture
def mocked_cv(monkeypatch):
    monkeypatch.setattr(
        latex_generator, "normalize_resume_language", lambda text, *args, **kwargs: text
    )
    monkeypatch.setattr(latex_generator.LLMService, "generate", lambda *args, **kwargs: BODY)


def _save_sample(name, content):
    directory = os.getenv("PDF_TEST_OUTPUT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(content)


@pytest.mark.parametrize("layout", list(latex_generator.CV_TEMPLATES))
def test_every_template_uses_the_candidate(mocked_cv, layout):
    latex = latex_generator.generate_german_latex_content(
        RESUME, "Platform role", [], layout_style=layout
    )
    assert "Jane Doe" in latex
    assert "jane@example.com" in latex
    assert "PLACEHOLDER" not in latex
    assert "Example Candidate" not in latex
    assert "Bamberg" not in latex
    assert "DevOps Engineer" not in latex
    assert r"\documentclass[11pt,a4paper]{article}" in latex


@REAL_TEX
@pytest.mark.parametrize("layout", list(latex_generator.CV_TEMPLATES))
def test_every_cv_layout_renders_one_page(mocked_cv, layout):
    latex = latex_generator.generate_german_latex_content(
        RESUME, "Platform role", [], layout_style=layout
    )
    pdf = latex_generator.compile_latex_to_pdf(latex)
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "Jane Doe" in text
    assert "Example GmbH" in text
    assert "Technical University" in text
    _save_sample(f"cv-{layout}.pdf", pdf)


def _mock_cover(monkeypatch, language):
    if language == "de":
        paragraph = (
            "Bei meiner Arbeit entwickelte ich Python Dienste und betreute deren Betrieb. "
            "Dabei untersuchte ich Fehler und verbesserte die Zusammenarbeit im Team. "
            "Diese Erfahrung möchte ich in Ihre ausgeschriebene Position einbringen. "
            "Gern erläutere ich meine Aufgaben und meinen Beitrag im persönlichen Gespräch. "
            "Die beschriebenen Aufgaben entsprechen meinem Interesse an zuverlässiger Software."
        )
    else:
        paragraph = (
            "In my current role I build Python services and monitor deployments. "
            "I investigate operational problems and work with colleagues to improve reliability. "
            "This experience is relevant to the responsibilities described in your position. "
            "I would welcome a conversation about the team and its engineering practices. "
            "My work has taught me to communicate clearly and document decisions."
        )
    data = {
        "company_name": "Example GmbH",
        "target_role": "Platform Engineer",
        "hiring_manager_name": "",
        "department": "Engineering",
        "opening": paragraph,
        "body_paragraph_1": paragraph,
        "body_paragraph_2": paragraph,
        "closing": paragraph,
    }
    monkeypatch.setattr(
        "app.services.llm.provider.LLMService.generate", lambda *args, **kwargs: json.dumps(data)
    )


@REAL_TEX
@pytest.mark.parametrize("style", [item["id"] for item in list_templates()])
@pytest.mark.parametrize("language", ["en", "de"])
def test_cover_letter_styles_render_one_page(monkeypatch, style, language):
    _mock_cover(monkeypatch, language)
    latex, used = cover_letter_pdf.generate_cover_letter_latex(
        RESUME,
        "Platform role",
        company_name="Target Company",
        template_style=style,
        language=language,
    )
    assert used == language
    assert "Bamberg" not in latex
    assert "Target Company" not in latex
    pdf = cover_letter_pdf.compile_cover_letter_pdf(latex)
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 1
    assert "Jane Doe" in reader.pages[0].extract_text()
    _save_sample(f"cover-{style}-{language}.pdf", pdf)


@REAL_TEX
def test_overlong_real_document_is_not_cropped():
    body = "\n".join(
        r"\noindent A complete paragraph that must remain in the output.\par" for _ in range(200)
    )
    source = r"\documentclass[11pt,a4paper]{article}\begin{document}" + body + r"\end{document}"
    with pytest.raises(PDFLayoutError):
        compile_single_page_pdf(source)


@pytest.mark.parametrize(
    "endpoint,compiler",
    [
        ("generate-german-cv", "app.api.endpoints.compile_latex_to_pdf"),
        (
            "generate-cover-letter-pdf",
            "app.services.career.cover_letter_pdf.compile_cover_letter_pdf",
        ),
    ],
)
def test_pdf_endpoints_return_422_for_page_overflow(monkeypatch, endpoint, compiler):
    monkeypatch.setattr(
        "app.api.endpoints.analyze_resume_content", lambda **kwargs: {"missing_skills": []}
    )
    monkeypatch.setattr("app.api.endpoints.generate_german_latex_content", lambda **kwargs: BODY)
    monkeypatch.setattr(
        cover_letter_pdf, "generate_cover_letter_latex", lambda **kwargs: (BODY, "en")
    )

    def reject(*args, **kwargs):
        raise PDFLayoutError(2)

    monkeypatch.setattr(compiler, reject)
    response = TestClient(app).post(
        f"/api/v1/resume/{endpoint}",
        data={"job_description": "Platform role"},
        files={"resume_file": ("resume.txt", RESUME.encode(), "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "pdf_page_limit_exceeded"
    assert response.json()["detail"]["pages"] == 2


def test_generation_does_not_invent_fallback_career(monkeypatch):
    monkeypatch.setattr(
        latex_generator, "normalize_resume_language", lambda text, *args, **kwargs: text
    )

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(latex_generator.LLMService, "generate", fail)
    with pytest.raises(latex_generator.FactualValidationError, match="no CV was produced"):
        latex_generator.generate_german_latex_content(RESUME, "Role", [], layout_style="standard")


def test_months_on_both_sides_of_date_range():
    resume = RESUME.replace("Jan 2020 - Present", "Jan 2020 – Feb 2024")
    assert "Jan 2020 – Feb 2024" in latex_generator.extract_resume_invariants(resume)
