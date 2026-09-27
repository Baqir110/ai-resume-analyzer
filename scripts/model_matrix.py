"""
Free / local model matrix: smoke test and full CV pipeline.

Reports only what it actually measured. A model that is not installed is
reported ``NOT INSTALLED`` and is not downloaded. A provider that is not
configured is reported ``NOT CONFIGURED``. A model whose smoke test fails is
reported ``FAIL`` with the reason, and is not carried into the pipeline stage.

Two stages, and passing the first says nothing about the second:

``smoke``
    A fixed, harmless prompt -- "Reply with exactly the word: acknowledged".
    No resume, no job description, no credential, no personal data. Establishes
    that the model is reachable and returns usable text.

``pipeline``
    The real application path, end to end, on a synthetic fixture:

        fixture resume -> ATS analysis -> bullet optimisation -> CV generation
        -> LaTeX -> pdflatex -> PDF -> PDF parse -> content validation

    A model that answers "acknowledged" in six seconds can still be unable to
    produce a complete CV. Only this stage says whether the model is usable by
    the application, and the two are reported separately on purpose.

Usage:
    python -m scripts.model_matrix --smoke
    python -m scripts.model_matrix --pipeline
    python -m scripts.model_matrix --all --json
    python -m scripts.model_matrix --model qwen3:8b --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Fixed and harmless. No user data of any kind, by construction.
SMOKE_PROMPT = "Reply with exactly the word: acknowledged"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED = "SKIPPED"
STATUS_NOT_INSTALLED = "NOT INSTALLED"
STATUS_NOT_CONFIGURED = "NOT CONFIGURED"
STATUS_NOT_TESTED = "NOT TESTED"

#: The models the task names explicitly. A bare name resolves through Ollama's
#: tag rules, so ``llama3.2`` finds ``llama3.2:latest``.
TASK_REQUIRED = ("qwen3:8b", "qwen2.5:7b", "llama3.2", "llama3", "deepseek-r1:8b")

#: Extra installed models worth trying. Deliberately not downloaded.
EXTRA_CANDIDATES = ("gemma4", "glm-4.7-flash", "qwen3.6", "llama3.3:70b")


# ---------------------------------------------------------------------------
# Synthetic fixture. No real person, no real employer, no real contact details.
# ---------------------------------------------------------------------------

FIXTURE_RESUME = """\
Alex Berger
Platform Engineer
alex.berger@example.invalid | +49 30 1234567 | Hamburg, Germany

PROFESSIONAL SUMMARY
Platform Engineer with eight years of experience designing, deploying and
operating cloud-native services on AWS. Background spans infrastructure
automation, container orchestration, continuous delivery, observability and
incident response.

EXPERIENCE

Platform Engineer | Beispiel GmbH | Jan 2020 - Present
Built Python services with FastAPI and instrumented every one with Prometheus
and Grafana monitoring dashboards.
Managed Kubernetes clusters across three environments and automated all
infrastructure provisioning with Terraform modules.
Reduced deployment time by 40 percent by introducing GitLab CI pipelines with
progressive delivery and automated rollback.
Established on-call runbooks and reduced mean time to recovery by consolidating
incident documentation into a single searchable source.
Owned the release process for three product teams including staging promotion
and release verification.
Partnered with the security team to harden the container image pipeline from
build through signing and provenance attestation.
Migrated twenty-two legacy services onto the shared platform with zero
customer-visible downtime across the migration window.
Automated certificate rotation for all internal domains using an internal CA
and HashiCorp Vault.

Systems Engineer | Muster AG | 2018 - 2020
Managed Linux systems, DNS, VPN and infrastructure operations for two data
centres across redundant sites.
Automated operational reporting with Python scripts, replacing a manual monthly
process that consumed two days of effort per cycle.
Reduced mean time to recovery by consolidating on-call documentation and
escalation paths into a single runbook.
Supported 24/7 incident response for internal business-critical services.

EDUCATION

M.Sc. Computer Science | Technical University of Hamburg | 2014 - 2018
Thesis on scheduling heuristics for container placement in orchestrator
clusters.

B.Sc. Information Systems | University of Passau | 2011 - 2014

SKILLS

Languages: Python, Bash, Go, SQL, JavaScript
Platform: Docker, Kubernetes, AWS, Terraform, Ansible, GitLab CI, GitHub Actions
Data: PostgreSQL, Redis, Prometheus, Grafana, Loki, OpenTelemetry
Networking: Nginx, DNS, VPN, BGP, TLS

LANGUAGES

English (native), German (fluent)
"""

FIXTURE_JD = """\
Senior Platform Engineer (m/w/d).

Sie betreiben und entwickeln unsere Kubernetes-Plattform auf AWS, automatisieren
die komplette Infrastruktur mit Terraform und verbessern unsere CI/CD-Pipelines
kontinuierlich. Sie überwachen Deployments mit Prometheus und Grafana, definieren
SLOs und stellen die Betriebssicherheit sicher.

Anforderungen:
- sehr gute Kenntnisse in Python, Docker, Kubernetes und AWS
- Infrastructure as Code mit Terraform
- CI/CD mit GitLab CI oder GitHub Actions
- Monitoring und Observability mit Prometheus, Grafana und Loki
- Linux, Bash, PostgreSQL, Nginx und Ansible
- Deployment-Monitoring, Incident Response und Runbook-Arbeit
- Deutschkenntnisse auf B2-Niveau erforderlich

Benefits: 70.000 EUR brutto pro Jahr, flexible Arbeitszeiten, Homeoffice und
ein Weiterbildungsbudget.
"""


# ---------------------------------------------------------------------------
# Stage 1: smoke
# ---------------------------------------------------------------------------


def installed_ollama_models() -> dict[str, dict[str, Any]]:
    """
    What this machine has, read from the local server.

    Read-only. Nothing is ever pulled: downloading a multi-gigabyte model to
    satisfy a test would be a surprising side effect of running a check.
    """
    from app.services.llm.provider import LLMService

    result = LLMService.list_models("ollama")
    rows: dict[str, dict[str, Any]] = {}

    for row in result.get("models", []):
        name = row.get("model")
        if name:
            rows[name] = row

    return rows


def resolve_installed(requested: str, installed: dict[str, dict]) -> str:
    """
    Map a requested model name to an installed one, using Ollama's tag rules.

    ``llama3.2`` is served by ``llama3.2:latest``. Reporting that as NOT INSTALLED
    would be wrong: the model is present and reachable under the name the user
    would put in ``.env``.
    """
    if requested in installed:
        return requested

    # Exact base match: "llama3.2" -> "llama3.2:latest"
    for name in installed:
        if name == requested or name.split(":", 1)[0] == requested:
            return name

    # Prefix match for the short forms: "gemma4" -> "gemma4:latest"
    for name in installed:
        if name.startswith(requested + ":") or name.split(":", 1)[0].startswith(requested):
            return name

    return ""


def smoke_test(model: str) -> dict[str, Any]:
    """
    One harmless prompt against one model, timed, with the response classified.

    Reports the response *length* and whether it is text, never the content: the
    caller is a status report, and a model that echoes its prompt must not put
    that into a log.
    """
    from app.services.llm.provider import LLMService

    row: dict[str, Any] = {
        "model": model,
        "status": STATUS_NOT_TESTED,
        "latency_s": None,
        "response_chars": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "usable_text": False,
        "detail": "",
    }

    started = time.perf_counter()
    try:
        answer = LLMService.generate(
            prompt=SMOKE_PROMPT,
            provider="ollama",
            model=model,
            # direct: a smoke test must exercise the model it was asked about,
            # not fall through to something else that happens to answer.
            route_mode="direct",
            task="generic",
        )
    except Exception as exc:
        row["latency_s"] = round(time.perf_counter() - started, 2)
        row["status"] = STATUS_FAIL
        row["detail"] = LLMService.classify_for_report(exc, "ollama")[:160]
        return row

    row["latency_s"] = round(time.perf_counter() - started, 2)
    text = (answer or "").strip()
    row["response_chars"] = len(text)
    row["usable_text"] = bool(text)

    usage = LLMService.get_last_usage()
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["completion_tokens"] = usage.get("completion_tokens")

    if not text:
        row["status"] = STATUS_FAIL
        row["detail"] = "provider returned no usable text"
        return row

    row["status"] = STATUS_PASS
    row["detail"] = "usable text returned"
    return row


# ---------------------------------------------------------------------------
# Stage 2: the real pipeline
# ---------------------------------------------------------------------------


def pipeline_test(model: str, layout: str = "german_corporate") -> dict[str, Any]:
    """
    Run the real application path on the synthetic fixture.

    Deliberately calls the same service functions the HTTP endpoints call, in
    the same order, rather than re-implementing anything. A test that used its
    own prompt builder would prove the test works, not the application.

    Each stage is timed and reported, because a model that completes all six
    stages in nine minutes is not actually usable even though it "passed".
    """
    from app.services.analysis.ats_analyzer import analyze_resume_content
    from app.services.cv.latex_generator import (
        _expected_content_from_latex,
        _sections_claimed_by_latex,
        compile_latex_to_pdf,
        generate_german_latex_content,
    )
    from app.services.cv.optimizer import optimize_resume_bullets
    from app.services.cv.pdf_validation import (
        extract_pdf_text,
        find_latex_artifacts,
        validate_pdf_content,
    )
    from app.services.llm.provider import LLMService

    row: dict[str, Any] = {
        "model": model,
        "layout": layout,
        "status": STATUS_NOT_TESTED,
        "total_s": None,
        "llm_calls": 0,
        "stages": {},
        "ats_score": None,
        "pdf_bytes": None,
        "pdf_pages": None,
        "pdf_text_chars": None,
        "content_retention": None,
        "sections_found": [],
        "validation_problems": [],
        "latex_artifacts": [],
        "detail": "",
    }

    calls = {"n": 0}
    real_generate = LLMService.generate.__func__

    def counting(cls, prompt, *args, **kwargs):
        calls["n"] += 1
        return real_generate(cls, prompt, *args, **kwargs)

    LLMService.generate = classmethod(counting)

    overall = time.perf_counter()

    try:
        # -- 1. ATS analysis (no LLM: deterministic scoring) ----------------
        stage = time.perf_counter()
        analysis = analyze_resume_content(FIXTURE_RESUME, FIXTURE_JD)
        row["stages"]["ats_analysis_s"] = round(time.perf_counter() - stage, 2)
        row["ats_score"] = analysis.get("ats_match_score")
        missing = list(analysis.get("missing_skills") or [])
        row["suggestions"] = list(analysis.get("improvement_suggestions") or [])

        if row["ats_score"] is None:
            row["status"] = STATUS_FAIL
            row["detail"] = "ATS analysis produced no score"
            return row

        # -- 2. Bullet optimisation (LLM) ----------------------------------
        stage = time.perf_counter()
        optimize_resume_bullets(
            resume_text=FIXTURE_RESUME,
            job_description=FIXTURE_JD,
            missing_skills=missing,
            provider="ollama",
            model_name=model,
            route_mode="direct",
            layout_style=layout,
        )
        row["stages"]["optimize_s"] = round(time.perf_counter() - stage, 2)

        # -- 3. CV generation to LaTeX (LLM) --------------------------------
        stage = time.perf_counter()
        latex_code = generate_german_latex_content(
            FIXTURE_RESUME,
            FIXTURE_JD,
            missing,
            provider="ollama",
            model_name=model,
            route_mode="direct",
            layout_style=layout,
            improvement_suggestions=analysis.get("improvement_suggestions") or [],
        )
        row["stages"]["generate_s"] = round(time.perf_counter() - stage, 2)
        row["latex_chars"] = len(latex_code)

        if not latex_code.strip():
            row["status"] = STATUS_FAIL
            row["detail"] = "generation produced no LaTeX"
            return row

        # -- 4. pdflatex ----------------------------------------------------
        stage = time.perf_counter()
        pdf_bytes = compile_latex_to_pdf(latex_code)
        row["stages"]["pdflatex_s"] = round(time.perf_counter() - stage, 2)
        row["pdf_bytes"] = len(pdf_bytes)

        # -- 5. PDF parse ---------------------------------------------------
        stage = time.perf_counter()
        text = extract_pdf_text(pdf_bytes)
        row["stages"]["pdf_parse_s"] = round(time.perf_counter() - stage, 2)
        row["pdf_text_chars"] = len(text.strip())

        # -- 6. Content + layout validation ---------------------------------
        stage = time.perf_counter()
        report = validate_pdf_content(
            pdf_bytes,
            expected_pages=1,
            expected_text=_expected_content_from_latex(latex_code),
            expected_sections=_sections_claimed_by_latex(latex_code),
        )
        row["stages"]["validation_s"] = round(time.perf_counter() - stage, 2)
        row["pdf_pages"] = report["pages"]
        row["content_retention"] = report["content_retention"]
        row["sections_found"] = report["sections_found"]
        row["validation_problems"] = report["problems"]
        row["latex_artifacts"] = find_latex_artifacts(text)

        row["status"] = STATUS_PASS if not report["problems"] else STATUS_FAIL
        if report["problems"]:
            row["detail"] = f"validation: {', '.join(report['problems'])}"

    except Exception as exc:
        row["status"] = STATUS_FAIL
        row["detail"] = LLMService.classify_for_report(exc, "ollama")[:200]
    finally:
        LLMService.generate = classmethod(real_generate)
        row["total_s"] = round(time.perf_counter() - overall, 2)
        row["llm_calls"] = calls["n"]

    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value}{suffix}"


def print_smoke(rows: list[dict[str, Any]], installed: dict[str, dict]) -> None:
    print("=" * 78)
    print(f"Smoke test  |  prompt: {SMOKE_PROMPT!r}")
    print("=" * 78)
    print("  Fixed and harmless. No resume, no job description, no credentials.")
    print()

    name_width = max((len(r["model"]) for r in rows), default=14)

    for row in rows:
        status = row["status"]
        latency = _fmt(row["latency_s"], "s")
        if status == STATUS_NOT_INSTALLED:
            print(f"  {row['model']:<{name_width}}  {status}")
            continue
        print(
            f"  {row['model']:<{name_width}}  {status:<6} {latency:>7}  "
            f"chars={_fmt(row['response_chars']):>6}  "
            f"completion={_fmt(row['completion_tokens']):>6}  "
            f"{row['detail']}"
        )

    print()
    print(f"  {len(rows)} model(s) tested.")


def print_pipeline(rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 78)
    print("Full CV pipeline  |  fixture resume -> ATS -> optimize -> LaTeX")
    print("                 ->  pdflatex -> PDF -> parse -> validate")
    print("=" * 78)
    print("  Synthetic fixture. No real person, employer or contact details.")
    print()

    name_width = max((len(r["model"]) for r in rows), default=14)

    for row in rows:
        status = row["status"]
        print(f"  {row['model']:<{name_width}}  {status}")

        if status == STATUS_NOT_TESTED:
            continue

        stages = row.get("stages", {})
        rendered = "  ".join(
            f"{key.replace('_s', '').replace('_', ' ')}={value}s" for key, value in stages.items()
        )
        print(f"      total={_fmt(row['total_s'], 's')}  " f"llm_calls={row['llm_calls']}")
        print(f"      {rendered}")
        print(
            f"      ats={_fmt(row['ats_score'])}  "
            f"latex={_fmt(row.get('latex_chars'), 'c')}  "
            f"pdf={_fmt(row['pdf_bytes'], 'B')}  "
            f"pages={_fmt(row['pdf_pages'])}  "
            f"text={_fmt(row['pdf_text_chars'], 'c')}  "
            f"retention={_fmt(row['content_retention'])}"
        )
        print(
            f"      sections={len(row.get('sections_found') or [])}  "
            f"artifacts={len(row.get('latex_artifacts') or [])}  "
            f"problems={row.get('validation_problems') or 'none'}"
        )
        if row.get("detail"):
            print(f"      {row['detail']}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test local/free models: harmless smoke test, then the real CV "
            "pipeline. Never downloads a model. Never sends user data."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model to test; repeatable. Default: everything installed.",
    )
    parser.add_argument("--smoke", action="store_true", help="smoke test only")
    parser.add_argument("--pipeline", action="store_true", help="full CV pipeline only")
    parser.add_argument("--all", action="store_true", help="both stages")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--layout", default="german_corporate", help="layout for the pipeline")
    args = parser.parse_args(argv)

    # Default: both stages, since that is the question being asked.
    do_smoke = args.smoke or args.all or not (args.smoke or args.pipeline)
    do_pipeline = args.pipeline or args.all or not (args.smoke or args.pipeline)

    import requests

    try:
        with requests.get("http://localhost:11434/api/tags", timeout=5):
            ollama_up = True
    except Exception:
        ollama_up = False

    installed: dict[str, dict] = {}

    if args.model:
        targets = list(dict.fromkeys(args.model))
    elif ollama_up:
        installed = installed_ollama_models()
        targets = list(installed)
        # Anything the task names explicitly but that is absent is still
        # reported, as NOT INSTALLED, rather than silently omitted.
        for name in TASK_REQUIRED:
            if name not in targets:
                targets.append(name)
    else:
        targets = list(TASK_REQUIRED)

    if not ollama_up:
        print("=" * 78)
        print("Ollama is not reachable on http://localhost:11434")
        print("=" * 78)
        print(f"  Every local model is {STATUS_NOT_INSTALLED} from here.")
        print("  Start it with `ollama serve` and re-run.")
        return 0

    if not installed:
        installed = installed_ollama_models()

    print()
    print("=" * 78)
    print("Local model inventory")
    print("=" * 78)
    print(f"  {len(installed)} model(s) installed. Nothing is downloaded.")
    print()
    for name in sorted(installed):
        row = installed[name]
        size_gb = row.get("size_bytes")
        size = f"{size_gb / 1e9:.1f} GB" if size_gb else "-"
        flag = " (required)" if name in TASK_REQUIRED else ""
        print(f"    {name:26} {size:>9}{flag}")

    smoke_rows: list[dict[str, Any]] = []
    pipeline_rows: list[dict[str, Any]] = []

    for requested in targets:
        resolved = resolve_installed(requested, installed)

        if not resolved:
            smoke_rows.append(
                {
                    "model": requested,
                    "status": STATUS_NOT_INSTALLED,
                    "latency_s": None,
                    "response_chars": None,
                    "completion_tokens": None,
                    "prompt_tokens": None,
                    "usable_text": False,
                    "detail": "not pulled; not downloaded by this script",
                }
            )
            pipeline_rows.append(
                {
                    "model": requested,
                    "layout": args.layout,
                    "status": STATUS_NOT_INSTALLED,
                    "total_s": None,
                    "llm_calls": 0,
                    "stages": {},
                    "detail": "not installed",
                }
            )
            continue

        entry = {
            "model": requested,
            "resolved": resolved,
        }

        if do_smoke:
            result = smoke_test(resolved)
            result["model"] = requested
            result["resolved"] = resolved
            smoke_rows.append(result)
            entry["smoke"] = result["status"]
        else:
            entry["smoke"] = STATUS_NOT_TESTED

        # Only a model that answered is carried into the pipeline stage.
        if do_pipeline:
            if not do_smoke or entry["smoke"] == STATUS_PASS:
                result = pipeline_test(resolved, args.layout)
                result["model"] = requested
                result["resolved"] = resolved
                pipeline_rows.append(result)
            else:
                pipeline_rows.append(
                    {
                        "model": requested,
                        "resolved": resolved,
                        "layout": args.layout,
                        "status": STATUS_SKIPPED,
                        "total_s": None,
                        "llm_calls": 0,
                        "stages": {},
                        "detail": f"smoke test was {entry['smoke']}",
                    }
                )

    if args.json:
        print(
            json.dumps(
                {
                    "smoke_prompt": SMOKE_PROMPT,
                    "installed": sorted(installed),
                    "smoke": smoke_rows,
                    "pipeline": pipeline_rows,
                },
                indent=2,
                default=str,
            )
        )
    else:
        if do_smoke:
            print_smoke(smoke_rows, installed)
        if do_pipeline:
            print_pipeline(pipeline_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
