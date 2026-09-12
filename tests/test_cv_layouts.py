from pathlib import Path

from app.services.cv.latex_generator import (
    CV_TEMPLATES,
    _patch_template_preamble,
    compile_latex_to_pdf,
)

TEST_BODY = r"""
\section*{Profile}
DevOps Engineer with experience in cloud infrastructure, automation,
CI/CD, observability, Kubernetes, Docker, AWS, Azure, Python and MLOps.

\section*{Work Experience}

\textbf{DevOps Engineer} -- Example Company \hfill \textit{2024 -- Present}

\begin{itemize}
    \item Built and maintained cloud infrastructure using AWS, Docker and Kubernetes.
    \item Implemented CI/CD pipelines and infrastructure automation.
    \item Developed monitoring and alerting using Prometheus and Grafana.
\end{itemize}

\textbf{Systems Engineer} -- Example Company \hfill \textit{2022 -- 2024}

\begin{itemize}
    \item Managed Linux systems, networking, DNS, VPN and infrastructure operations.
    \item Automated operational workflows using Python and infrastructure-as-code practices.
\end{itemize}

\section*{Projects}

\textbf{AI IT Operations Assistant}

\begin{itemize}
    \item Infrastructure monitoring, anomaly detection, incident analysis and RAG-based troubleshooting.
    \item Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis, Prometheus and Grafana.
\end{itemize}

\section*{Education}

\textbf{M.Sc. International Software Systems Science} -- University of Bamberg
\hfill \textit{2024 -- Present}

\textbf{B.Sc. Computer Engineering} -- Okan University
\hfill \textit{2019 -- 2023}

\section*{Skills}

AWS, Azure, Kubernetes, Docker, Linux, Terraform, Python, FastAPI,
Prometheus, Grafana, CI/CD, PostgreSQL, Redis, Git, MLOps

\section*{Languages}

English, German, Turkish, Urdu
"""


def main():
    output_dir = Path("test_cv_output")
    output_dir.mkdir(exist_ok=True)

    failures = []

    for layout, template in CV_TEMPLATES.items():
        print(f"\nTesting: {layout}")

        try:
            latex = _patch_template_preamble(template)

            latex = latex.replace(
                "CANDIDATE_NAME_PLACEHOLDER",
                "Muhammad Baqir",
            )

            latex = latex.replace(
                "CANDIDATE_TITLE_PLACEHOLDER",
                "DevOps Engineer",
            )

            latex = latex.replace(
                "CANDIDATE_CONTACT_PLACEHOLDER",
                "Bamberg, Germany | +49 152 17975480 | baqir@example.com",
            )

            latex = latex.replace(
                "LINKEDIN_URL_PLACEHOLDER",
                "https://www.linkedin.com/in/muhammad-baqir-it/",
            )

            latex = latex.replace(
                "GITHUB_URL_PLACEHOLDER",
                "https://github.com/Baqir110",
            )

            latex = latex.replace(
                "RESUME_BODY_PLACEHOLDER",
                TEST_BODY,
            )

            pdf_bytes = compile_latex_to_pdf(latex)

            pdf_path = output_dir / f"{layout}.pdf"
            pdf_path.write_bytes(pdf_bytes)

            print(f"[PASS] {layout}")
            print(f"       PDF: {pdf_path}")
            print(f"       Size: {len(pdf_bytes):,} bytes")

        except Exception as exc:
            failures.append((layout, str(exc)))
            print(f"[FAIL] {layout}")
            print(f"       {exc}")

    print("\n" + "=" * 70)

    if failures:
        print(f"FAILED: {len(failures)} layout(s)")
        for layout, error in failures:
            print(f"\n{layout}:")
            print(error)

        raise SystemExit(1)

    print(f"PASSED: {len(CV_TEMPLATES)}/{len(CV_TEMPLATES)} layouts")
    print("=" * 70)


if __name__ == "__main__":
    main()
