"""
The generation prompt states a size budget, and the budget is grounded.

The one-page requirement is a hard rule the compiler enforces, and it is not
relaxed by anything here. What changed is that the prompt now says *how big* the
document may be, in units the model can count, because "must fit on one page" and
"be concise" are not targets a model can hit.

``llama3.2`` failed a third of its full-pipeline runs on exactly this: a
well-formed CV of about 4 700 characters that did not fit, with 4 633 passing in
the same session. The wall was measurable and the prompt did not mention it.

Two things are asserted, and the second is the one that would matter if someone
tried to turn this into a way of making a model pass:

* the prompt carries a real, bounded budget;
* the budget is *below* the measured wall, so aiming at it produces a document
  that fits -- and the compiler still refuses anything that does not.
"""

from __future__ import annotations

import re

from app.services.cv.latex_generator import TARGET_LATEX_CHARS

#: Measured by compiling generated documents of known length through the German
#: corporate layout at 11pt. The largest body that fitted on one page, and the
#: smallest that did not.
LARGEST_THAT_FITTED = 4_691
SMALLEST_THAT_OVERFLOWED = 4_731


def test_the_budget_is_a_plausible_length():
    assert (
        1_000 <= TARGET_LATEX_CHARS <= 20_000
    ), f"{TARGET_LATEX_CHARS} is not a plausible body length"


def test_the_budget_sits_below_the_measured_wall():
    """
    If the budget were above the overflow threshold, a model aiming at it would
    produce a document the compiler then refuses, which would make the change
    worse than not making it.
    """
    assert TARGET_LATEX_CHARS < SMALLEST_THAT_OVERFLOWED, (
        f"the budget ({TARGET_LATEX_CHARS}) is at or above the point where a "
        f"document overflows ({SMALLEST_THAT_OVERFLOWED})"
    )


def test_the_budget_leaves_headroom():
    """
    A budget exactly at the wall would be defeated by the model's own variance,
    which is the failure this change exists to stop.
    """
    headroom = SMALLEST_THAT_OVERFLOWED - TARGET_LATEX_CHARS
    assert headroom >= 400, (
        f"only {headroom} characters of headroom; a model that overshoots the "
        f"budget slightly would still overflow"
    )


def test_the_budget_fits_a_real_full_cv():
    """
    The budget must be above what a legitimately full CV needs, or the prompt is
    asking the model to pad or to cut content it should keep.
    """
    full_cv = LARGEST_THAT_FITTED
    assert TARGET_LATEX_CHARS >= full_cv * 0.85, (
        f"the budget ({TARGET_LATEX_CHARS}) is far below a full CV that fits "
        f"({full_cv}); the prompt would be asking for content to be dropped"
    )


def test_the_prompt_states_the_budget_in_countable_units():
    """
    The point of the change. The rule has to name a number, or the model is back
    to guessing.
    """
    import inspect

    from app.services.cv import latex_generator

    source = inspect.getsource(latex_generator)

    assert "MUST BE AT MOST" in source
    assert "{TARGET_LATEX_CHARS}" in source, "the prompt template does not interpolate the budget"

    # And the one-page rule survives alongside it.
    assert "MUST FIT ON EXACTLY ONE" in source


def test_the_constant_is_referenced_by_name_not_hardcoded():
    """
    A literal in the prompt would drift from the constant and nothing would
    notice.
    """
    import inspect

    from app.services.cv import latex_generator

    source = inspect.getsource(latex_generator)

    numbers = re.findall(r"MUST BE AT MOST \{[^}]+\} CHARACTERS", source)
    assert numbers, "the budget line is missing from the prompt"
    assert "4200" not in numbers[0], "the budget is hardcoded in the prompt instead of interpolated"
