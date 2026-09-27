#!/usr/bin/env python
"""Continuous autonomous agent — LinkedIn + Indeed.

Discovers jobs from LinkedIn and Indeed only.

Applies immediately after each role×location search batch — no waiting for a
full pass to finish.

No Arbeitsagentur, Ashby, Greenhouse, Lever, or non-German SerpAPI scraping.

Configuration is primarily controlled through .env:

    AUTOMATIC_APPLY=true
    AUTOMATIC_SUBMIT=true
    MAX_MISSING_REQUIRED_SKILLS=5
    MINIMUM_MATCH_SCORE=55

Usage:
    python run_agent_continuous.py
    python run_agent_continuous.py --auto
    python run_agent_continuous.py --dry-run
    python run_agent_continuous.py --once
    python run_agent_continuous.py --delay 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

Path("data").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "data/agent_continuous.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("run_agent_continuous")


# ─────────────────────────────────────────────────────────────────────────────
# JobSpy worker — LinkedIn + Indeed ONLY
# ─────────────────────────────────────────────────────────────────────────────
#
# Runs in a clean subprocess to avoid importing problematic ML dependencies
# into the main process.
# ─────────────────────────────────────────────────────────────────────────────

_JOBSPY_WORKER = r"""
import sys
import json
import warnings

warnings.filterwarnings("ignore")

from jobspy import scrape_jobs


term = sys.argv[1]
location = sys.argv[2]
results = int(sys.argv[3])
hours = int(sys.argv[4])


try:
    df = scrape_jobs(
        site_name=["linkedin", "indeed"],
        search_term=term,
        location=location,
        results_wanted=results,
        hours_old=hours,
        country_indeed="germany",
        linkedin_fetch_description=True,
        easy_apply=True,
    )

    jobs = []

    for _, row in df.iterrows():
        url = str(
            row.get("job_url")
            or row.get("site_url")
            or ""
        ).strip()

        if not url:
            continue

        desc = row.get("description")

        if desc is None or isinstance(desc, float):
            desc = ""

        site = str(
            row.get("site")
            or "unknown"
        ).strip()

        title = str(
            row.get("title")
            or term
        ).strip()

        company = str(
            row.get("company")
            or "Unknown"
        ).strip()

        job_location = str(
            row.get("location")
            or location
        ).strip()

        easy_apply = bool(
            row.get("is_easy_apply")
            or "linkedin" in site.lower()
        )

        jobs.append(
            {
                "job_url": url,
                "title": title,
                "company": company,
                "location": job_location,
                "description": str(desc),
                "site": site,
                "easy_apply": easy_apply,
            }
        )

    # Easy Apply jobs first.
    jobs.sort(
        key=lambda job: 0 if job["easy_apply"] else 1
    )

    print(json.dumps(jobs))

except Exception:
    import traceback

    traceback.print_exc(file=sys.stderr)
    print(json.dumps([]))
"""


def _scrape_subprocess(
    term: str,
    location: str,
    results: int = 30,
    hours: int = 168,
) -> list[dict]:
    """Run JobSpy for LinkedIn + Indeed in an isolated subprocess."""

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _JOBSPY_WORKER,
                term,
                location,
                str(results),
                str(hours),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.stderr.strip():
            logger.debug(
                "jobspy stderr: %s",
                result.stderr[:500],
            )

        if result.stdout.strip():
            parsed = json.loads(result.stdout)

            if isinstance(parsed, list):
                return parsed

    except subprocess.TimeoutExpired:
        logger.warning(
            "jobspy timed out for '%s' in %s",
            term,
            location,
        )

    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not parse jobspy output for '%s' in %s: %s",
            term,
            location,
            exc,
        )

    except Exception as exc:
        logger.warning(
            "jobspy subprocess error for '%s' in %s: %s",
            term,
            location,
            exc,
        )

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Database storage
# ─────────────────────────────────────────────────────────────────────────────


async def _store_discovered(jobs: list[dict]) -> int:
    """Save new jobs to the tracker DB.

    Returns the number of jobs actually inserted.
    """

    from app.core.security import UnsafeInputError, validate_public_http_url
    from app.services.jobs.agent_schemas import NormalizedJob
    from app.services.jobs.dedupe import compute_dedupe_key, compute_fingerprint
    from app.services.tracking.state_machine import ApplicationStateMachine

    known = ApplicationStateMachine.known_fingerprints()

    stored = 0

    for job in jobs:
        job_url = str(job.get("job_url") or "").strip()

        if not job_url:
            continue
        try:
            job_url = validate_public_http_url(job_url, resolve_dns=False)
        except UnsafeInputError as exc:
            logger.warning("Skipping unsafe discovered URL: %s", exc)
            continue

        normalized = NormalizedJob(
            job_id=job_url,
            title=str(job.get("title") or "Unknown"),
            company=str(job.get("company") or "Unknown"),
            location=str(job.get("location") or ""),
            description=str(job.get("description") or ""),
            application_url=job_url,
            original_url=job_url,
            source=str(job.get("site") or job.get("source") or "unknown"),
        )
        fingerprint = compute_fingerprint(normalized)
        dedupe_key = compute_dedupe_key(normalized)
        if fingerprint in known:
            continue

        ApplicationStateMachine.create(
            company_name=normalized.company,
            job_title=normalized.title,
            job_url=job_url,
            fingerprint=fingerprint,
            dedupe_key=dedupe_key,
            source=normalized.source,
            job_description=normalized.description,
            location=normalized.location,
        )

        known.add(fingerprint)
        stored += 1

    return stored


# ─────────────────────────────────────────────────────────────────────────────
# Apply phase
# ─────────────────────────────────────────────────────────────────────────────


async def _apply_phase(
    orchestrator,
    auto_submit: bool,
    settings,
) -> dict:
    """Analyze and apply newly discovered jobs.

    Discovery happens outside this function.

    Only DISCOVERED jobs are sent through the matching/analyze pipeline.
    Jobs already in READY_TO_SUBMIT are not re-analyzed here.

    AUTOMATIC_APPLY and AUTOMATIC_SUBMIT are taken from settings unless
    explicitly enabled by the CLI --auto flag.
    """

    from pathlib import Path

    from app.services.jobs.agent_schemas import ApplicationState, NormalizedJob
    from app.services.tracking.state_machine import ApplicationStateMachine

    # ─────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────
    # --auto is an explicit CLI override.
    if auto_submit:
        settings.AUTOMATIC_SUBMIT = True
        settings.AUTOMATIC_APPLY = True

        logger.warning("CLI --auto enabled — AUTOMATIC_APPLY=true and " "AUTOMATIC_SUBMIT=true")

    # Otherwise respect .env/settings.
    automatic_apply = bool(
        getattr(
            settings,
            "AUTOMATIC_APPLY",
            False,
        )
    )

    automatic_submit = bool(
        getattr(
            settings,
            "AUTOMATIC_SUBMIT",
            False,
        )
    )

    if automatic_submit:
        logger.warning(
            "AUTOMATIC_SUBMIT=true — browser agent may submit applications " "without human review"
        )

    logger.info(
        "Apply configuration — AUTOMATIC_APPLY=%s  AUTOMATIC_SUBMIT=%s",
        automatic_apply,
        automatic_submit,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Load resume
    # ─────────────────────────────────────────────────────────────────────

    resume_file = Path(orchestrator.resume_path)

    if not resume_file.exists():
        logger.error(
            "Resume not found: %s",
            resume_file,
        )

        return {
            "applied": 0,
            "skipped": 0,
        }

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(resume_file))

        resume_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    except Exception as exc:
        logger.exception(
            "Could not read resume PDF: %s",
            exc,
        )

        return {
            "applied": 0,
            "skipped": 0,
        }

    if not resume_text:
        logger.error("Could not extract text from resume PDF.")

        return {
            "applied": 0,
            "skipped": 0,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Get newly discovered jobs
    # ─────────────────────────────────────────────────────────────────────

    pending = ApplicationStateMachine.list_automation_queue()

    ready = ApplicationStateMachine.list_by_state(ApplicationState.READY_TO_SUBMIT)

    logger.info(
        "Apply phase: processing %d queued jobs from DB...",
        len(pending),
    )

    if ready:
        logger.info(
            "DB currently contains %d READY_TO_SUBMIT job(s); "
            "these are already prepared and will not be re-analyzed.",
            len(ready),
        )

    # ─────────────────────────────────────────────────────────────────────
    # No newly discovered jobs
    # ─────────────────────────────────────────────────────────────────────

    if not pending:
        counts = {
            state: len(ApplicationStateMachine.list_by_state(state)) for state in ApplicationState
        }

        logger.info(
            "No active queue jobs to process. "
            "DB — Discovered:%d  Matched:%d  Ready:%d  "
            "Submitted:%d  Failed:%d",
            counts.get(
                ApplicationState.DISCOVERED,
                0,
            ),
            counts.get(
                ApplicationState.MATCHED,
                0,
            ),
            counts.get(
                ApplicationState.READY_TO_SUBMIT,
                0,
            ),
            counts.get(
                ApplicationState.SUBMITTED,
                0,
            ),
            counts.get(
                ApplicationState.SUBMISSION_FAILED,
                0,
            ),
        )

        return {
            "applied": 0,
            "skipped": 0,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Browser concurrency
    # ─────────────────────────────────────────────────────────────────────

    semaphore = asyncio.Semaphore(
        max(
            1,
            int(
                getattr(
                    settings,
                    "MAX_CONCURRENT_BROWSER_WORKERS",
                    1,
                )
            ),
        )
    )

    applied = 0
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────
    # Process each newly discovered job
    # ─────────────────────────────────────────────────────────────────────

    for record in pending:
        app_id = record["id"]

        job = NormalizedJob(
            job_id=str(app_id),
            title=record.get(
                "job_title",
                "Unknown",
            ),
            company=record.get(
                "company_name",
                "Unknown",
            ),
            description=record.get("job_description") or "",
            application_url=record.get(
                "job_url",
                "",
            ),
            original_url=record.get(
                "job_url",
                "",
            ),
            source=record.get(
                "source",
                "linkedin",
            ),
            fingerprint=record.get(
                "fingerprint",
                "",
            ),
        )

        logger.info(
            "Analyzing: %s at %s",
            job.title,
            job.company,
        )

        # ─────────────────────────────────────────────────────────────
        # Analyze / match
        # ─────────────────────────────────────────────────────────────

        try:
            _id, match_result = await orchestrator._analyze_and_transition(
                job,
                resume_text,
            )

        except Exception as exc:
            logger.exception(
                "Analysis failed for %s at %s: %s",
                job.title,
                job.company,
                exc,
            )
            continue

        if match_result is None:
            logger.warning(
                "Matching failed for %s at %s",
                job.title,
                job.company,
            )
            continue

        # ─────────────────────────────────────────────────────────────
        # Decision
        # ─────────────────────────────────────────────────────────────

        decision = str(
            getattr(
                match_result,
                "decision",
                "",
            )
        ).upper()

        overall_score = float(
            getattr(
                match_result,
                "overall_score",
                0.0,
            )
            or 0.0
        )

        if decision != "APPLY":
            skipped += 1

            logger.info(
                "SKIP  %s at %s (decision=%s, score=%.0f)",
                job.title,
                job.company,
                decision or "UNKNOWN",
                overall_score,
            )

            continue

        # ─────────────────────────────────────────────────────────────
        # Automatic apply disabled
        # ─────────────────────────────────────────────────────────────

        if not automatic_apply:
            logger.info(
                "MATCHED %s at %s — " "AUTOMATIC_APPLY=false, not opening browser",
                job.title,
                job.company,
            )
            continue

        # ─────────────────────────────────────────────────────────────
        # Execute browser application pipeline
        # ─────────────────────────────────────────────────────────────

        logger.info(
            "APPLY %s at %s (score=%.0f, auto_submit=%s)",
            job.title,
            job.company,
            overall_score,
            automatic_submit,
        )

        if not ApplicationStateMachine.reserve_daily_application(
            app_id,
            max(0, int(getattr(settings, "MAX_DAILY_APPLICATIONS", 0))),
        ):
            logger.info(
                "DAILY_BUDGET_EXHAUSTED %s at %s",
                job.title,
                job.company,
            )
            skipped += 1
            continue

        try:
            result = await orchestrator._execute_pipeline(
                app_id,
                job,
                resume_text,
                semaphore,
            )

        except Exception as exc:
            logger.exception(
                "Application pipeline failed for %s at %s: %s",
                job.title,
                job.company,
                exc,
            )
            continue

        if not result:
            logger.warning(
                "Application pipeline returned no result for %s at %s",
                job.title,
                job.company,
            )
            continue

        submitted = bool(result.get("submitted"))

        success = bool(result.get("success"))

        if submitted or success:
            applied += 1

            logger.info(
                "APPLICATION SUCCESS: %s at %s",
                job.title,
                job.company,
            )
        else:
            logger.info(
                "Application pipeline finished without submission: " "%s at %s",
                job.title,
                job.company,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Final DB state
    # ─────────────────────────────────────────────────────────────────────

    counts = {
        state: len(ApplicationStateMachine.list_by_state(state)) for state in ApplicationState
    }

    logger.info(
        "DB after apply — " "Discovered:%d  Matched:%d  Ready:%d  " "Submitted:%d  Failed:%d",
        counts.get(
            ApplicationState.DISCOVERED,
            0,
        ),
        counts.get(
            ApplicationState.MATCHED,
            0,
        ),
        counts.get(
            ApplicationState.READY_TO_SUBMIT,
            0,
        ),
        counts.get(
            ApplicationState.SUBMITTED,
            0,
        ),
        counts.get(
            ApplicationState.SUBMISSION_FAILED,
            0,
        ),
    )

    return {
        "applied": applied,
        "skipped": skipped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────


async def _main(args: argparse.Namespace) -> None:
    from app.core.config import settings
    from app.services.jobs.orchestrator import JobAgentOrchestrator

    # --auto explicitly overrides settings.
    if args.auto:
        settings.AUTOMATIC_APPLY = True
        settings.AUTOMATIC_SUBMIT = True

    # --dry-run must always prevent applications.
    if args.dry_run:
        settings.AUTOMATIC_APPLY = False
        settings.AUTOMATIC_SUBMIT = False

    orchestrator = JobAgentOrchestrator(
        settings,
        resume_path=args.resume,
    )

    total_disc = 0
    total_app = 0
    total_skip = 0
    cycle = 0

    # ─────────────────────────────────────────────────────────────────────
    # Locations
    # ─────────────────────────────────────────────────────────────────────
    #
    # LinkedIn/Indeed can search both physical locations and Remote.
    # There is NO separate physical-location list anymore because
    # Arbeitsagentur has been completely removed.
    # ─────────────────────────────────────────────────────────────────────

    geo_locations = [loc for loc in settings.TARGET_LOCATIONS if loc.strip()]

    if not geo_locations:
        geo_locations = ["Germany"]

    # Remove duplicate locations while preserving order.
    geo_locations = list(dict.fromkeys(geo_locations))

    # ─────────────────────────────────────────────────────────────────────
    # Roles
    # ─────────────────────────────────────────────────────────────────────

    if args.test:
        roles = settings.TARGET_ROLES[:1]
        search_locations = geo_locations[:1]
        results_per_search = 5
    else:
        roles = settings.TARGET_ROLES
        search_locations = geo_locations
        results_per_search = 30

    # ─────────────────────────────────────────────────────────────────────
    # Startup information
    # ─────────────────────────────────────────────────────────────────────

    print("=" * 72)
    print("  Continuous Agent — LinkedIn + Indeed")
    print(f"  Roles       : {', '.join(roles)}")
    print(f"  Locations   : {', '.join(search_locations)}")
    print(f"  Auto-apply  : " f"{getattr(settings, 'AUTOMATIC_APPLY', False)}")
    print(f"  Auto-submit : " f"{getattr(settings, 'AUTOMATIC_SUBMIT', False)}")
    print(f"  Dry-run     : {args.dry_run}")
    print(f"  Delay       : {args.delay}s")
    print("  Sources     : LinkedIn + Indeed")
    print("  Applies immediately after each role×location search")
    print("=" * 72)

    seen_urls: set[str] = set()

    # ─────────────────────────────────────────────────────────────────────
    # Continuous loop
    # ─────────────────────────────────────────────────────────────────────

    while True:
        cycle += 1

        print()

        print(
            f"── CYCLE {cycle}  "
            f"(applied: {total_app}  "
            f"skipped: {total_skip}  "
            f"discovered: {total_disc}) ──"
        )

        for role in roles:
            for location in search_locations:
                # ─────────────────────────────────────────────────────
                # LinkedIn + Indeed
                # ─────────────────────────────────────────────────────

                logger.info(
                    "Searching LinkedIn+Indeed: '%s' in %s ...",
                    role,
                    location,
                )

                jobs = await asyncio.to_thread(
                    _scrape_subprocess,
                    role,
                    location,
                    results_per_search,
                    168,
                )

                # ─────────────────────────────────────────────────────
                # Remove duplicate URLs
                # ─────────────────────────────────────────────────────

                all_new: list[dict] = []

                for job in jobs:
                    job_url = str(job.get("job_url") or "").strip()

                    if not job_url:
                        continue

                    if job_url in seen_urls:
                        continue

                    seen_urls.add(job_url)
                    all_new.append(job)

                # Test mode limits discovery.
                if args.test and len(all_new) > 2:
                    all_new = all_new[:2]

                # ─────────────────────────────────────────────────────
                # Store jobs
                # ─────────────────────────────────────────────────────

                stored = await _store_discovered(all_new)

                easy_apply = sum(1 for job in all_new if job.get("easy_apply"))

                normal = len(all_new) - easy_apply

                logger.info(
                    "  → %d new stored  " "(%d Easy Apply, %d normal)",
                    stored,
                    easy_apply,
                    normal,
                )

                total_disc += stored

                # ─────────────────────────────────────────────────────
                # Apply immediately after this search
                # ─────────────────────────────────────────────────────

                if not args.dry_run and stored > 0:
                    try:
                        apply_stats = await _apply_phase(
                            orchestrator,
                            args.auto,
                            settings,
                        )

                        total_app += apply_stats["applied"]

                        total_skip += apply_stats["skipped"]

                        if apply_stats["applied"] or apply_stats["skipped"]:
                            print(
                                f"    ✓ applied: "
                                f"{apply_stats['applied']}  "
                                f"skipped: "
                                f"{apply_stats['skipped']}"
                            )

                    except KeyboardInterrupt:
                        raise

                    except Exception as exc:
                        logger.exception(
                            "Apply phase crashed: %s",
                            exc,
                        )

                if args.test:
                    break

            if args.test:
                break

        # ─────────────────────────────────────────────────────────────────
        # Dry run
        # ─────────────────────────────────────────────────────────────────

        if args.dry_run:
            logger.info("DRY RUN — discovery only, no applications sent")

        # ─────────────────────────────────────────────────────────────────
        # Once
        # ─────────────────────────────────────────────────────────────────

        if args.once:
            logger.info("--once flag set — exiting.")

            print(
                f"\nDone. discovered: {total_disc}  "
                f"applied: {total_app}  "
                f"skipped: {total_skip}"
            )

            break

        # ─────────────────────────────────────────────────────────────────
        # Next cycle
        # ─────────────────────────────────────────────────────────────────

        logger.info(
            "Cycle %d complete. " "Next cycle in %ds — press Ctrl+C to stop.",
            cycle,
            args.delay,
        )

        await asyncio.sleep(args.delay)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Continuous autonomous job agent — " "LinkedIn + Indeed")
    )

    parser.add_argument(
        "--resume",
        default="data/resume.pdf",
        help="Path to base resume PDF",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help=("Enable automatic application and " "automatic submission"),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Discover + analyze only; " "never apply"),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=("Run one complete cycle then exit"),
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=300,
        help=("Seconds between cycles " "(default: 300)"),
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=("Test mode: one role, one location, " "maximum two jobs"),
    )

    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    try:
        asyncio.run(_main(args))

    except KeyboardInterrupt:
        print("\n\nAgent stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
