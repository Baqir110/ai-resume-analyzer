"""Complete automated job-application pipeline.

Flow:

    Original Resume
          ↓
    Original ATS Analysis
          ↓
    Tailored CV v1
          ↓
    ATS Re-analysis
          ↓
    Tailored CV v2
          ↓
    ATS Re-analysis
          ↓
    Tailored CV v3
          ↓
    Select Highest-Scoring CV
          ↓
    Detect Cover-Letter Requirement
          ↓
    Generate Cover Letter if Required
          ↓
    Browser Use
          ↓
    Optional Submission
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pypdf import PdfReader

from app.services.analysis.ats_analyzer import analyze_resume_content as analyze_resume
from app.services.career.cover_letter_pdf import (
    compile_cover_letter_pdf,
    generate_cover_letter_latex,
)
from app.services.cv.latex_generator import compile_latex_to_pdf, generate_german_latex_content
from app.services.cv.variant_router import classify_role, get_variant_config
from app.services.jobs.agent_schemas import NormalizedJob
from app.services.jobs.backend_submitter import ApplicationPackage
from app.services.jobs.browser_use_applier import BrowserUseApplier
from app.services.jobs.package_validator import validate_package
from app.services.jobs.profile_manager import load_raw_profile

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "generated"

TARGET_ATS_SCORE = 80.0
MAX_CV_OPTIMIZATION_ROUNDS = 3
MIN_SCORE_IMPROVEMENT = 0.5


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_pdf(pdf_bytes: bytes, filename: str) -> Path:
    """Atomically save a uniquely named PDF and return its absolute path.

    A company/title-only filename lets concurrent applications overwrite one
    another.  The random suffix keeps every generated artifact immutable and
    makes the document history unambiguous.
    """
    _ensure_output_dir()
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("PDF bytes must be non-empty")
    safe_name = Path(str(filename)).name
    stem = Path(safe_name).stem[:100] or "document"
    suffix = Path(safe_name).suffix or ".pdf"
    final_path = OUTPUT_DIR / f"{stem}_{uuid4().hex[:16]}{suffix}"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=str(OUTPUT_DIR),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(pdf_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, final_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return final_path.resolve()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    _ensure_output_dir()

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix="extract-",
            dir=str(OUTPUT_DIR),
            delete=False,
        ) as temporary_file:
            temporary_file.write(pdf_bytes)
            temporary_path = Path(temporary_file.name)

        reader = PdfReader(str(temporary_path))
        pages: List[str] = []

        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "Could not extract text from PDF page: %s",
                    exc,
                )
                text = ""

            if text.strip():
                pages.append(text)

        return "\n".join(pages).strip()

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _get_analysis_value(
    analysis: Any,
    names: List[str],
    default: Any = None,
) -> Any:
    """Read a field from dicts, dataclasses, or normal objects."""
    if analysis is None:
        return default

    for name in names:
        if isinstance(analysis, dict):
            if name in analysis:
                value = analysis[name]
                if value is not None:
                    return value

        try:
            value = getattr(analysis, name)
        except AttributeError:
            continue

        if value is not None:
            return value

    return default


def _extract_score(analysis: Any) -> float:
    """
    Extract the ATS score.

    The current ATS analyzer uses `ats_match_score`.
    Older pipeline versions used `score`, so both are supported.
    """
    score = _get_analysis_value(
        analysis,
        [
            "ats_match_score",
            "final_score",
            "ats_score",
            "score",
            "blended_score",
            "overall_score",
        ],
    )

    if score is None:
        raise ValueError(
            "Could not find ATS score. " f"Analyzer result type: {type(analysis).__name__}"
        )

    try:
        score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ATS score is not numeric: {score!r}") from exc

    # Support analyzers returning 0.0-1.0 as well as 0-100.
    if 0.0 <= score <= 1.0:
        score *= 100.0

    return max(0.0, min(100.0, score))


def _extract_missing_skills(analysis: Any) -> List[str]:
    """Extract missing skills from the ATS result."""
    value = _get_analysis_value(
        analysis,
        [
            "missing_skills",
            "skills_missing",
            "missing",
        ],
        [],
    )

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    return []


def _extract_layout_style(analysis: Any) -> str:
    """Extract the analyzer's recommended layout."""
    value = _get_analysis_value(
        analysis,
        [
            "recommended_layout",
            "layout_style",
            "layout",
        ],
        "standard",
    )

    return str(value or "standard")


def _score_resume(
    resume_text: str,
    job_description: str,
) -> Dict[str, Any]:
    """Run ATS analysis and normalize its result."""
    logger.info("Beginning resume vs job description analysis.")

    analysis = analyze_resume(
        resume_text,
        job_description,
    )

    score = _extract_score(analysis)
    missing_skills = _extract_missing_skills(analysis)
    layout_style = _extract_layout_style(analysis)

    logger.info(
        "ATS score extracted successfully: %.2f%%",
        score,
    )

    return {
        "analysis": analysis,
        "score": score,
        "missing_skills": missing_skills,
        "layout_style": layout_style,
    }


def _cover_letter_required(job_description: str) -> bool:
    """
    Detect an explicit cover-letter requirement.

    This is intentionally conservative. If the job description does not
    explicitly require a cover letter, Browser Use can still encounter a
    cover-letter upload field on the actual application website.
    """
    text = (job_description or "").lower()

    required_patterns = [
        r"\bcover letter\b.{0,100}\brequired\b",
        r"\brequired\b.{0,100}\bcover letter\b",
        r"\bapplication letter\b.{0,100}\brequired\b",
        r"\bletter of motivation\b.{0,100}\brequired\b",
        r"\bmotivationsschreiben\b.{0,100}\berforderlich\b",
        r"\bmotivationsschreiben\b.{0,100}\bnotwendig\b",
        r"\bmotivationsschreiben\b.{0,100}\bbenötigt\b",
        r"\banschreiben\b.{0,100}\berforderlich\b",
        r"\banschreiben\b.{0,100}\bnotwendig\b",
        r"\banschreiben\b.{0,100}\bbenötigt\b",
        r"\banschreiben\b.{0,100}\bbeifügen\b",
        r"\banschreiben\b.{0,100}\beinreichen\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in required_patterns
    )


def _cover_letter_optional(job_description: str) -> bool:
    """Detect explicit optional cover-letter wording."""
    text = (job_description or "").lower()

    optional_patterns = [
        r"\bcover letter\b.{0,100}\boptional\b",
        r"\boptional\b.{0,100}\bcover letter\b",
        r"\banschreiben\b.{0,100}\boptional\b",
        r"\bmotivationsschreiben\b.{0,100}\boptional\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in optional_patterns
    )


def _safe_filename(value: str) -> str:
    """Create a safe filename component."""
    value = str(value or "").strip()
    value = re.sub(r"[^\w\s.-]", "", value)
    value = re.sub(r"\s+", "_", value)

    return value[:100] or "job"


def _compile_cv(latex_content: str) -> bytes:
    """Compile CV LaTeX into PDF bytes."""
    result = compile_latex_to_pdf(latex_content)

    if isinstance(result, bytes):
        return result

    if isinstance(result, bytearray):
        return bytes(result)

    if isinstance(result, Path):
        return result.read_bytes()

    if isinstance(result, str):
        path = Path(result)

        if path.exists():
            return path.read_bytes()

    raise TypeError(
        "compile_latex_to_pdf() returned an unsupported value: " f"{type(result).__name__}"
    )


def _compile_cover_letter(latex_content: str) -> bytes:
    """Compile cover-letter LaTeX into PDF bytes."""
    result = compile_cover_letter_pdf(latex_content)

    if isinstance(result, bytes):
        return result

    if isinstance(result, bytearray):
        return bytes(result)

    if isinstance(result, Path):
        return result.read_bytes()

    if isinstance(result, str):
        path = Path(result)

        if path.exists():
            return path.read_bytes()

    raise TypeError(
        "compile_cover_letter_pdf() returned an unsupported value: " f"{type(result).__name__}"
    )


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert Browser Use ApplicationResult into a serializable dict."""
    if result is None:
        return {}

    if isinstance(result, dict):
        return result

    if is_dataclass(result):
        return asdict(result)

    if hasattr(result, "model_dump"):
        try:
            dumped = result.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(result, "dict"):
        try:
            dumped = result.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(result, "__dict__"):
        try:
            return dict(result.__dict__)
        except Exception:
            pass

    return {
        "result": str(result),
    }


async def _run_browser_application(
    application_url: str,
    resume_path: Path,
    cover_letter_path: Optional[Path],
    why_this_company: str,
    company_name: str,
    role_title: str,
    agent_llm: str,
    auto_submit: bool,
    max_steps: int = 40,
    verification_timeout: int = 120,
) -> Dict[str, Any]:
    """Run Browser Use regardless of whether the applier exposes async/sync."""
    applier = BrowserUseApplier()

    kwargs = {
        "job_url": application_url,
        "resume_path": str(resume_path),
        "cover_letter_path": (str(cover_letter_path) if cover_letter_path else None),
        "why_this_company": why_this_company,
        "company_name": company_name,
        "role_title": role_title,
        "agent_llm": agent_llm,
        "auto_submit": auto_submit,
        "max_steps": max_steps,
        "headless": True,
        "handle_email_verification": False,
        "verification_timeout": verification_timeout,
    }

    result = applier.apply(**kwargs)

    if inspect.isawaitable(result):
        result = await result

    return _result_to_dict(result)


async def full_pipeline(
    job: Dict[str, Any],
    resume_text: str,
    profile: Optional[Dict[str, Any]] = None,
    model_name: str = "gpt-5.6-luna",
    agent_llm: str = os.getenv("BROWSER_AGENT_LLM", "experiential"),
    auto_submit: bool = False,
    provider: str | None = None,
    route_mode: str | None = None,
    max_steps: int = 40,
    verification_timeout: int = 120,
) -> Dict[str, Any]:
    """Execute the complete job application pipeline."""

    _ = profile

    if not isinstance(max_steps, int) or not 1 <= max_steps <= 100:
        raise ValueError("max_steps must be between 1 and 100")
    if not isinstance(verification_timeout, int) or not 0 <= verification_timeout <= 900:
        raise ValueError("verification_timeout must be between 0 and 900 seconds")

    if not isinstance(job, dict):
        raise TypeError("job must be a dictionary")

    job_id = str(job.get("job_id") or job.get("id") or "job_application")

    job_title = str(job.get("title") or job.get("job_title") or "Unknown Role")

    company_name = str(job.get("company") or job.get("company_name") or "Unknown Company")

    job_description = str(job.get("description") or job.get("job_description") or "")

    application_url = str(
        job.get("application_url")
        or job.get("apply_url")
        or job.get("job_url")
        or job.get("url")
        or ""
    )

    if not resume_text.strip():
        raise ValueError("Resume text is empty.")

    if not job_description.strip():
        raise ValueError(f"Job description is empty for '{job_title}'.")

    if not application_url:
        raise ValueError(f"No application URL found for '{job_title}' " f"at '{company_name}'.")

    logger.info(
        "Starting full application pipeline: %s at %s",
        job_title,
        company_name,
    )

    # --- CV variant classification ---
    cv_variant = classify_role(job_title, job_description)
    variant_cfg = get_variant_config(cv_variant)
    logger.info(
        "CV variant: %s for job: %s",
        cv_variant.value,
        job_title,
    )

    # ==================================================================
    # ORIGINAL RESUME
    # ==================================================================

    original_result = _score_resume(
        resume_text,
        job_description,
    )

    original_score = original_result["score"]

    logger.info(
        "Original ATS score: %.2f%%",
        original_score,
    )

    best_resume_text = resume_text
    best_pdf_bytes: Optional[bytes] = None
    best_score = original_score
    best_analysis = original_result["analysis"]
    best_version = "original"

    current_resume_text = resume_text
    current_missing_skills = original_result["missing_skills"]
    current_layout = original_result["layout_style"]

    optimization_history: List[Dict[str, Any]] = [
        {
            "version": "original",
            "score": round(original_score, 2),
            "selected": True,
            "missing_skills": current_missing_skills,
        }
    ]

    # ==================================================================
    # CV OPTIMIZATION
    # ==================================================================

    for round_number in range(
        1,
        MAX_CV_OPTIMIZATION_ROUNDS + 1,
    ):
        logger.info(
            "CV optimization round %d/%d",
            round_number,
            MAX_CV_OPTIMIZATION_ROUNDS,
        )

        try:
            latex_content = generate_german_latex_content(
                resume_text=current_resume_text,
                job_description=job_description,
                missing_skills=current_missing_skills,
                layout_style=current_layout,
                model_name=model_name,
                provider=provider,
                route_mode=route_mode,
                variant_cfg=variant_cfg,
            )

            pdf_bytes = _compile_cv(latex_content)

            generated_text = extract_pdf_text(pdf_bytes)

            if not generated_text.strip():
                raise ValueError("Generated CV PDF contains no readable text.")

            generated_result = _score_resume(
                generated_text,
                job_description,
            )

            generated_score = generated_result["score"]

            improved = generated_score > best_score + MIN_SCORE_IMPROVEMENT

            optimization_history.append(
                {
                    "version": f"tailored_v{round_number}",
                    "score": round(generated_score, 2),
                    "previous_best": round(best_score, 2),
                    "improved": improved,
                    "selected": False,
                    "missing_skills": generated_result["missing_skills"],
                }
            )

            logger.info(
                "Tailored CV v%d ATS score: %.2f%%",
                round_number,
                generated_score,
            )

            if generated_score > best_score:
                best_score = generated_score
                best_resume_text = generated_text
                best_pdf_bytes = pdf_bytes
                best_analysis = generated_result["analysis"]
                best_version = f"tailored_v{round_number}"

                current_resume_text = generated_text
                current_missing_skills = generated_result["missing_skills"]
                current_layout = generated_result["layout_style"]

                logger.info(
                    "New best CV selected: tailored_v%d " "(%.2f%%)",
                    round_number,
                    best_score,
                )

            if best_score >= TARGET_ATS_SCORE:
                logger.info(
                    "ATS target reached: %.2f%% >= %.2f%%",
                    best_score,
                    TARGET_ATS_SCORE,
                )
                break

        except Exception as exc:
            logger.exception(
                "CV optimization round %d failed: %s",
                round_number,
                exc,
            )

            optimization_history.append(
                {
                    "version": f"tailored_v{round_number}",
                    "score": None,
                    "selected": False,
                    "error": "CV optimization failed",
                }
            )

    # ==================================================================
    # SAVE BEST CV
    # ==================================================================

    safe_company = _safe_filename(company_name)
    safe_title = _safe_filename(job_title)

    if best_pdf_bytes is None:
        logger.info(
            "No tailored CV improved the original score. " "Generating one final candidate CV."
        )

        try:
            fallback_latex = generate_german_latex_content(
                resume_text=best_resume_text,
                job_description=job_description,
                missing_skills=_extract_missing_skills(best_analysis),
                layout_style=_extract_layout_style(best_analysis),
                model_name=model_name,
                provider=provider,
                route_mode=route_mode,
                variant_cfg=variant_cfg,
            )

            fallback_pdf = _compile_cv(fallback_latex)
            fallback_text = extract_pdf_text(fallback_pdf)

            if fallback_text.strip():
                fallback_result = _score_resume(
                    fallback_text,
                    job_description,
                )
                fallback_score = fallback_result["score"]
                if fallback_score > best_score:
                    best_pdf_bytes = fallback_pdf
                    best_version = "tailored_fallback"
                    best_score = fallback_score
                    best_resume_text = fallback_text
                    best_analysis = fallback_result["analysis"]
                else:
                    logger.warning(
                        "Fallback CV did not improve the original score; retaining original/manual review path."
                    )
            else:
                logger.warning(
                    "Fallback CV compiled but text extraction returned empty; "
                    "retaining original/manual review path."
                )

        except Exception as exc:
            logger.exception(
                "Fallback CV generation failed: %s",
                exc,
            )

    if best_pdf_bytes is None:
        # No tailored CV beat the original — fall back to the source resume PDF
        # rather than aborting the whole application attempt.
        _source_resume = OUTPUT_DIR.parent / "resume.pdf"
        if _source_resume.exists():
            logger.warning(
                "No tailored CV improved the score; using original resume PDF as fallback."
            )
            best_pdf_bytes = _source_resume.read_bytes()
            best_version = "original_pdf_fallback"
        else:
            raise RuntimeError("No usable CV could be generated.")

    cv_path = save_pdf(
        best_pdf_bytes,
        f"{safe_company}_{safe_title}_CV.pdf",
    )

    for item in optimization_history:
        item["selected"] = item.get("version") == best_version

    logger.info(
        "Selected CV: %s",
        best_version,
    )

    logger.info(
        "Final ATS score: %.2f%%",
        best_score,
    )

    # ==================================================================
    # COVER LETTER
    # ==================================================================

    cover_letter_required = _cover_letter_required(job_description)

    cover_letter_optional = _cover_letter_optional(job_description)

    cover_letter_status = "not_required"
    cover_letter_path: Optional[Path] = None

    if cover_letter_required:
        cover_letter_status = "required"

        logger.info("Cover letter explicitly required.")

        try:
            cover_result = generate_cover_letter_latex(
                resume_text=best_resume_text,
                job_description=job_description,
                company_name=company_name,
                model_name=model_name,
                provider=provider,
                route_mode=route_mode,
            )

            if isinstance(cover_result, tuple):
                cover_latex = cover_result[0]
            else:
                cover_latex = cover_result

            cover_pdf_bytes = _compile_cover_letter(cover_latex)

            cover_letter_path = save_pdf(
                cover_pdf_bytes,
                f"{safe_company}_{safe_title}_Cover_Letter.pdf",
            )

            cover_letter_status = "generated"

            logger.info(
                "Cover letter generated: %s",
                cover_letter_path,
            )

        except Exception as exc:
            cover_letter_status = "generation_failed"

            logger.exception(
                "Cover-letter generation failed: %s",
                exc,
            )

    elif cover_letter_optional:
        cover_letter_status = "optional"

        logger.info("Cover letter is explicitly optional.")

    else:
        logger.info("No explicit cover-letter requirement detected.")

    # ==================================================================
    # PACKAGE VALIDATION / BROWSER USE
    # ==================================================================

    # Never let a generated artifact reach a real form before the complete
    # package has passed validation.  Previously this check lived only in the
    # orchestrator *after* Browser Use had already run, which made a corrupt
    # package fail open at the most dangerous boundary.
    package_job = NormalizedJob(
        job_id=job_id,
        title=job_title,
        company=company_name,
        description=job_description,
        application_url=application_url,
        fingerprint=str(job.get("fingerprint", "")),
    )
    try:
        package = ApplicationPackage(
            job=package_job,
            cv_path=str(cv_path),
            cover_letter_path=(str(cover_letter_path) if cover_letter_path else None),
            answers={},
            profile=load_raw_profile(),
            expected_fingerprint=package_job.fingerprint,
        )
        validation = validate_package(
            package,
            require_cover_letter=cover_letter_required,
        )
    except Exception:
        logger.exception("Could not validate application package")
        validation = None

    if validation is None or not validation.valid:
        reason = "Application package validation failed"
        if validation is not None and validation.errors:
            reason += ": " + "; ".join(validation.errors[:5])
        logger.error("Refusing browser execution: %s", reason)
        return {
            "status": "package_validation_failed",
            "success": False,
            "submitted": False,
            "job_id": job_id,
            "job_title": job_title,
            "company": company_name,
            "job_url": application_url,
            "cv_path": str(cv_path),
            "cover_letter_path": (str(cover_letter_path) if cover_letter_path else None),
            "cover_letter_required": cover_letter_required,
            "error": reason,
            "browser_application": {
                "success": False,
                "submitted": False,
                "error_message": reason,
            },
        }

    why_this_company = (
        f"I am interested in the {job_title} position at {company_name}. "
        "Please use only facts from my profile and resume when completing the form."
    )

    try:
        browser_result = await _run_browser_application(
            application_url=application_url,
            resume_path=cv_path,
            cover_letter_path=cover_letter_path,
            why_this_company=why_this_company,
            company_name=company_name,
            role_title=job_title,
            agent_llm=agent_llm,
            auto_submit=auto_submit,
            max_steps=max_steps,
            verification_timeout=verification_timeout,
        )

    except Exception:
        logger.exception("Browser Use application failed.")

        browser_result = {
            "success": False,
            "submitted": False,
            "error_message": "Browser application failed",
        }

    # ==================================================================
    # FINAL RESULT
    # ==================================================================

    browser_success = bool(browser_result.get("success", False))

    result: Dict[str, Any] = {
        "status": ("completed" if browser_success else "pipeline_completed_browser_failed"),
        "success": browser_success,
        "submitted": bool(browser_result.get("submitted", False)),
        "job_id": job_id,
        "job_title": job_title,
        "company": company_name,
        "job_url": application_url,
        "model": model_name,
        "provider": provider,
        "route_mode": route_mode,
        "original_ats_score": round(
            original_score,
            2,
        ),
        "final_ats_score": round(
            best_score,
            2,
        ),
        "ats_target": TARGET_ATS_SCORE,
        "target_reached": (best_score >= TARGET_ATS_SCORE),
        "selected_cv_version": best_version,
        "cv_variant": cv_variant.value,
        "cv_path": str(cv_path),
        "cover_letter_required": cover_letter_required,
        "cover_letter_optional": cover_letter_optional,
        "cover_letter_status": cover_letter_status,
        "cover_letter_path": (str(cover_letter_path) if cover_letter_path else None),
        "optimization_rounds": optimization_history,
        "browser_application": browser_result,
    }

    logger.info(
        "Pipeline complete: %s at %s | "
        "ATS %.2f%% -> %.2f%% | CV=%s | Cover=%s | "
        "Submitted=%s",
        job_title,
        company_name,
        original_score,
        best_score,
        best_version,
        cover_letter_status,
        result["submitted"],
    )

    return result


run_full_application = full_pipeline


__all__ = [
    "full_pipeline",
    "run_full_application",
    "save_pdf",
    "extract_pdf_text",
]
