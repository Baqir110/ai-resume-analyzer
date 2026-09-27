# app/dashboard/views/agent_control.py
"""Agent Control — live view of the autonomous agent's state machine.

Shows counts per ApplicationState (Section 30 "Live Agent" +
"Overview") and the READY_TO_SUBMIT queue the user reviews before the
final click, per Section 18's default policy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from app.services.jobs.agent_schemas import ApplicationState
from app.services.tracking.state_machine import ApplicationStateMachine

_AGENT_SCRIPT = str(Path(__file__).resolve().parents[3] / "run_job_agent.py")


def _launch(flag: str) -> None:
    """Fire-and-forget: spawn the agent CLI in a new visible console window."""
    project_root = str(Path(_AGENT_SCRIPT).parent)
    kwargs: dict = {"cwd": project_root}
    # On Windows open a new visible console so the user can watch progress.
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    subprocess.Popen([sys.executable, _AGENT_SCRIPT, flag], **kwargs)


def render() -> None:
    st.header("🤖 Agent Control")
    controls_enabled = os.getenv("DASHBOARD_CONTROL_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not controls_enabled:
        st.info(
            "Agent controls are disabled. Set DASHBOARD_CONTROL_ENABLED=true "
            "only on a trusted local dashboard."
        )
    st.caption(
        "Live status of the autonomous job agent. By default the agent "
        "stops at READY_TO_SUBMIT and waits for you to review before the "
        "final click — see AUTOMATIC_SUBMIT in your .env to change that."
    )

    # ── Action buttons ────────────────────────────────────────────────
    st.subheader("🚀 Run Agent")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button(
        "🔍 Discover Jobs",
        use_container_width=True,
        disabled=not controls_enabled,
        help="Search all enabled sources for new jobs now",
    ):
        _launch("--discover")
        st.success("Discovery started in background — refresh in a few seconds to see results.")
    if b2.button(
        "📋 Prepare CVs",
        use_container_width=True,
        disabled=not controls_enabled,
        help="Discover + score + generate tailored CVs (no submit)",
    ):
        _launch("--prepare")
        st.success("Prepare pass started — check Recent Events below.")
    if b3.button(
        "⚡ Full Pipeline",
        use_container_width=True,
        disabled=not controls_enabled,
        help="Discover → analyze → prepare → apply (respects AUTOMATIC_SUBMIT setting)",
    ):
        _launch("--apply")
        st.success("Full pipeline started — jobs will appear in Ready to Submit queue.")
    if b4.button(
        "🔁 Retry Failed",
        use_container_width=True,
        disabled=not controls_enabled,
        help="Re-attempt all SUBMISSION_FAILED applications",
    ):
        _launch("--retry-failed")
        st.success("Retry pass started.")

    st.divider()

    from app.services.tracking.live_status import get_status

    live = get_status()
    st.subheader("📡 Live Agent")
    if live is None or live.get("stage") in (None, "", "IDLE"):
        st.caption("Agent is idle — press a button above or run `--serve` in a terminal.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stage", live.get("stage", "-"))
        c2.metric("Current job", live.get("job_title") or "-")
        c3.metric("Source", live.get("source") or "-")
        c4.metric("Retry count", live.get("retry_count", 0))
        if live.get("current_error"):
            st.warning(f"Current error: {live['current_error']}")
        st.caption(f"Last updated: {live.get('updated_at', '-')}")

    counts = {
        state: len(ApplicationStateMachine.list_by_state(state)) for state in ApplicationState
    }

    overview_states = [
        ApplicationState.DISCOVERED,
        ApplicationState.MATCHED,
        ApplicationState.SKIPPED,
        ApplicationState.READY_TO_SUBMIT,
        ApplicationState.SUBMITTED,
        ApplicationState.SUBMISSION_FAILED,
        ApplicationState.CAPTCHA_REQUIRED,
    ]
    cols = st.columns(len(overview_states))
    for col, state in zip(cols, overview_states):
        col.metric(state.value.replace("_", " ").title(), counts.get(state, 0))

    st.divider()
    st.subheader("✅ Ready to Submit")
    ready = ApplicationStateMachine.list_by_state(ApplicationState.READY_TO_SUBMIT)
    if not ready:
        st.info("Nothing waiting on you right now.")
    for app in ready:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            c1.markdown(f"**{app['job_title']}** — {app['company_name']}")
            c2.markdown(f"Match: {app.get('match_score', '—')}")
            if app.get("job_url"):
                c3.link_button("Open application", app["job_url"])
            with st.expander("History"):
                for event in ApplicationStateMachine.history(app["id"]):
                    st.text(
                        f"{event['created_at']}  {event['from_state']} → {event['to_state']}  ({event['reason']})"
                    )

    st.divider()
    st.subheader("📋 Recent Events")
    from app.services.observability.structured_logger import read_events

    events = read_events(limit=30)
    if not events:
        st.caption("No structured events logged yet.")
    else:
        for event in reversed(events):
            ts = event.get("timestamp", "")[:19].replace("T", " ")
            name = event.get("event", "?")
            extra = {k: v for k, v in event.items() if k not in ("timestamp", "event")}
            st.text(f"{ts}  {name:22s} {extra}")

    st.divider()
    st.subheader("⚠️ Needs Attention")
    for state, label in [
        (ApplicationState.CAPTCHA_REQUIRED, "CAPTCHA / human verification required"),
        (ApplicationState.SUBMISSION_FAILED, "Submission failed"),
        (ApplicationState.VERIFICATION_FAILED, "Could not verify submission"),
    ]:
        records = ApplicationStateMachine.list_by_state(state)
        if records:
            st.markdown(f"**{label}** ({len(records)})")
            for app in records:
                st.text(f"• {app['job_title']} at {app['company_name']}")
