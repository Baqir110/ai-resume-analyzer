"""
The agent page must not scale its database work with the state machine.

ApplicationState has 22 members. The Agent Control page used to call
``list_by_state`` once per member to build 22 integers, plus three more calls
for the lists it displays, so one page render opened SQLite twenty-five times.

The cost was not only time. Each extra connection is another chance to collide
with a concurrent writer's lock, and that collision surfaced as an entirely
broken page rather than a missing metric -- an intermittent failure that showed
up in a pre-push run long after the page was written.

This asserts the shape of the code's database access rather than a wall-clock
number: one bulk query covering every state, and no per-state query. That is
deterministic, and it is the property that stops the regression returning.
"""

from __future__ import annotations

import re

import pytest
from streamlit.testing.v1 import AppTest

from app.dashboard import workflow
from app.services.jobs.agent_schemas import ApplicationState
from app.services.tracking import state_machine
from tests.test_dashboard_workflow import APP

#: A rendered count: a plain integer, or an abbreviated one such as "9.3K" or
#: "1.2M". Anything else is a placeholder.
_COUNT_LIKE = re.compile(r"^\d[\d,.]*\s*[KMB]?$", re.IGNORECASE)

#: A rendered count: a plain integer, or an abbreviated one such as "9.3K" or
#: "1.2M". Anything else is a placeholder.
_COUNT_LIKE = re.compile(r"^\d[\d,.]*\s*[KMB]?$", re.IGNORECASE)

#: A rendered count: a plain integer, or an abbreviated one such as "9.3K" or
#: "1.2M". Anything else is a placeholder.
_COUNT_LIKE = re.compile(r"^\d[\d,.]*\s*[KMB]?$", re.IGNORECASE)


def test_agent_page_reads_every_state_in_one_query(monkeypatch):
    """
    One bulk read, not one read per state.

    Asserting on the call shape rather than on elapsed time means this fails
    when the pattern comes back, and does not fail on a slow machine.
    """
    bulk: list[int] = []
    single: list[ApplicationState] = []

    original_bulk = state_machine.ApplicationStateMachine.list_by_states.__func__
    original_single = state_machine.ApplicationStateMachine.list_by_state.__func__

    def counting_bulk(cls, states):
        states = list(states)
        bulk.append(len(states))
        return original_bulk(cls, states)

    def counting_single(cls, state):
        single.append(state)
        return original_single(cls, state)

    monkeypatch.setattr(
        state_machine.ApplicationStateMachine, "list_by_states", classmethod(counting_bulk)
    )
    monkeypatch.setattr(
        state_machine.ApplicationStateMachine, "list_by_state", classmethod(counting_single)
    )

    app = AppTest.from_file(APP, default_timeout=90)
    app.session_state[workflow.KEY_PAGE] = "agent_control"
    app.session_state["api_base"] = "http://127.0.0.1:9"
    app.run()

    assert not app.exception, [f"{type(e.value).__name__}: {e.value}" for e in app.exception]

    assert not single, (
        f"the page issued {len(single)} per-state queries. That is the N+1 this "
        f"replaced, and it multiplies the window in which a concurrent writer "
        f"can make the whole page fail."
    )

    assert len(bulk) == 1, f"expected one bulk query, got {len(bulk)}: {bulk}"

    total_states = len(list(ApplicationState))
    assert bulk[0] >= total_states, (
        f"the bulk query covered {bulk[0]} of {total_states} states, so the "
        f"per-state metrics would silently under-report"
    )


def test_agent_page_shows_every_overview_metric(monkeypatch):
    """
    The batching must not lose a metric.

    The counts are what the page is for, so a fix that made the query cheap by
    asking for less would be a regression wearing a fix's clothes.
    """
    app = AppTest.from_file(APP, default_timeout=90)
    app.session_state[workflow.KEY_PAGE] = "agent_control"
    app.session_state["api_base"] = "http://127.0.0.1:9"
    app.run()

    assert not app.exception, [f"{type(e.value).__name__}: {e.value}" for e in app.exception]

    labels = [str(metric.label) for metric in app.metric]
    by_label = {str(metric.label): str(metric.value).strip() for metric in app.metric}

    for state in (
        ApplicationState.DISCOVERED,
        ApplicationState.MATCHED,
        ApplicationState.SKIPPED,
        ApplicationState.READY_TO_SUBMIT,
        ApplicationState.SUBMITTED,
        ApplicationState.SUBMISSION_FAILED,
        ApplicationState.CAPTCHA_REQUIRED,
    ):
        expected = state.value.replace("_", " ").title()
        assert expected in labels, f"missing the {expected!r} metric; got {labels}"

    # Each state metric must show a count, not a placeholder -- "" or "-" or
    # "n/a" is what a lost metric looks like.
    #
    # Scoped to the state metrics deliberately. The page also shows cost and
    # duration, which are currency and time, so asserting a count on every metric
    # on the page was asserting something that was never true. Both a plain
    # integer and an abbreviated one ("9.3K") count here: whether a value is large
    # enough to abbreviate depends on the shared usage log, which every other
    # test also writes to, and depending on that made this test fail or pass
    # purely on the order it ran in.
    for state in (
        ApplicationState.DISCOVERED,
        ApplicationState.MATCHED,
        ApplicationState.SKIPPED,
        ApplicationState.READY_TO_SUBMIT,
        ApplicationState.SUBMITTED,
        ApplicationState.SUBMISSION_FAILED,
        ApplicationState.CAPTCHA_REQUIRED,
    ):
        label = state.value.replace("_", " ").title()
        text = by_label[label].lstrip("+-")
        assert _COUNT_LIKE.match(
            text
        ), f"metric {label!r} rendered {by_label[label]!r} rather than a count"

    # And no metric anywhere on the page may be blank.
    for metric in app.metric:
        assert str(metric.value).strip(), f"metric {metric.label!r} rendered nothing"
