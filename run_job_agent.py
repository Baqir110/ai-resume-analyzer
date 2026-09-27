#!/usr/bin/env python
"""Zero-question autonomous job agent CLI (Section 36).

Unlike run_job_search.py (interactive, asks the user to pick a model
each run), this reads everything from app.core.config.settings and
.env, and never calls input(). Point it at a resume and go:

    python run_job_agent.py                 # one full autonomous pass
    python run_job_agent.py --discover       # discovery only
    python run_job_agent.py --prepare        # discover+analyze+prepare, never submit
    python run_job_agent.py --apply          # full pass, respects AUTOMATIC_SUBMIT
    python run_job_agent.py --retry-failed   # retry SUBMISSION_FAILED applications
    python run_job_agent.py --status         # print application counts by state
    python run_job_agent.py --serve          # run continuously on the configured schedule
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.core.config import settings
from app.services.jobs.agent_schemas import ApplicationState
from app.services.jobs.orchestrator import JobAgentOrchestrator
from app.services.jobs.scheduler import AgentScheduler, ScheduleConfig
from app.services.tracking.state_machine import ApplicationStateMachine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("run_job_agent")


def _print_status() -> None:
    print("=" * 70)
    print("APPLICATION STATUS")
    print("=" * 70)
    for state in ApplicationState:
        records = ApplicationStateMachine.list_by_state(state)
        if records:
            print(f"{state.value:28s} {len(records)}")
    print("=" * 70)


async def _retry_failed(orchestrator: JobAgentOrchestrator) -> None:
    failed = ApplicationStateMachine.list_by_state(ApplicationState.SUBMISSION_FAILED)
    if not failed:
        print("No failed applications to retry.")
        return
    print(
        f"Retrying {len(failed)} failed applications (max {orchestrator.MAX_RETRY_ATTEMPTS} attempts each)..."
    )
    results = await orchestrator.retry_failed()
    print(f"Retry pass complete: {len(results)} applications re-processed.")
    for r in results:
        job = r.get("job", {})
        print(
            f"  - {job.get('title')} at {job.get('company')}: "
            f"{'submitted' if r.get('submitted') else 'success' if r.get('success') else 'still failing'}"
        )


async def _run(args: argparse.Namespace) -> int:
    orchestrator = JobAgentOrchestrator(settings, resume_path=args.resume)

    if args.status:
        _print_status()
        return 0

    if args.follow_ups:
        from app.services.tracking.analytics import due_follow_ups

        due = due_follow_ups()
        if not due:
            print("No follow-ups due.")
        for app in due:
            print(
                f"Follow up: {app['job_title']} at {app['company_name']} "
                f"(submitted, follow-up was due {app['follow_up_date']})"
            )
        return 0

    if args.stats:
        from app.services.tracking.analytics import compute_outcome_stats

        print(f"{'SOURCE':20s} {'SUBMITTED':>10s} {'INTERVIEW %':>12s} {'OFFER %':>10s}")
        for s in compute_outcome_stats(dimension="source"):
            print(
                f"{s.key:20s} {s.submitted:>10d} {s.interview_rate:>11.1f}% {s.offer_rate:>9.1f}%"
            )
        return 0

    if args.retry_failed:
        await _retry_failed(orchestrator)
        return 0

    if args.dashboard:
        import subprocess

        dashboard_path = Path(__file__).parent / "app" / "dashboard" / "main.py"
        print(f"Launching dashboard: streamlit run {dashboard_path}")
        return subprocess.call(["streamlit", "run", str(dashboard_path)])

    if args.discover:
        jobs = await orchestrator.discover()
        print(f"Discovered {len(jobs)} new (non-duplicate) jobs.")
        return 0

    if args.analyze:
        summary = await orchestrator.analyze_only()
        print()
        print("=" * 70)
        print("ANALYSIS SUMMARY (no CVs generated, nothing applied)")
        print("=" * 70)
        for row in summary:
            print(
                f"{row['decision']:6s} {row['priority']:14s} score={row['score']:>5.1f}  "
                f"{row['title']} @ {row['company']}"
            )
            if row["decision"] == "SKIP" and row["reasons"]:
                print(f"       reason: {'; '.join(row['reasons'])}")
        return 0

    if args.serve:
        schedule = ScheduleConfig(
            discover_times=settings.DISCOVER_TIMES,
            apply_times=settings.APPLY_TIMES,
        )

        async def on_discover():
            await orchestrator.discover()

        async def on_apply():
            await orchestrator.run_once()

        scheduler = AgentScheduler(schedule, on_discover=on_discover, on_apply=on_apply)
        await scheduler.run_forever()
        return 0

    # --prepare/--apply explicitly enable package generation/filling, while
    # --prepare always disables the final click.  The bare default remains
    # fail-closed and respects the environment.
    if args.prepare:
        settings.AUTOMATIC_APPLY = True
        settings.AUTOMATIC_SUBMIT = False
    elif args.apply:
        settings.AUTOMATIC_APPLY = True

    results = await orchestrator.run_once()

    print()
    print("=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    by_decision: dict[str, int] = {}
    for r in results:
        key = r.get("decision", "UNKNOWN")
        by_decision[key] = by_decision.get(key, 0) + 1
    for key, count in by_decision.items():
        print(f"{key:24s} {count}")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous AI job application agent")
    parser.add_argument("--resume", default="data/resume.pdf", help="Path to base resume PDF")
    parser.add_argument("--discover", action="store_true", help="Run discovery only")
    parser.add_argument(
        "--analyze", action="store_true", help="Discover + score/decide, no CV generation or apply"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Discover+analyze+prepare, never submit"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Full autonomous pass (default if no flag given)"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry SUBMISSION_FAILED applications"
    )
    parser.add_argument("--status", action="store_true", help="Print application counts by state")
    parser.add_argument(
        "--follow-ups", action="store_true", help="List applications due for a follow-up"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print interview/offer rates by source"
    )
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit dashboard")
    parser.add_argument(
        "--serve", action="store_true", help="Run continuously on the configured schedule"
    )
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
