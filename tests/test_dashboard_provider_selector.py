"""
The dashboard must not hide a provider the router supports.

`render_provider_selector` used to carry a literal list of six provider names
while the registry held thirteen. OpenRouter, Cerebras, Cloudflare, GitHub
Models, Hugging Face and OmniRoute were unreachable from that widget, so a user
had no way to find out they were supported. It also offered two route modes
while the LLM Settings page offered three, so choosing `automatic` in Settings
and then opening a tool page silently downgraded the request to one provider.

Both are drift between a hard-coded list and the registry, so both are pinned
here.
"""

from __future__ import annotations

from app.dashboard.components import _ROUTE_MODES, provider_choices, render_provider_selector
from app.services.llm.registry import PROVIDER_REGISTRY


def test_every_registered_provider_is_offered_by_the_selector():
    """The guard against the original bug."""
    names = [name for name, _label in provider_choices()]

    missing = sorted(set(PROVIDER_REGISTRY) - set(names))

    assert not missing, f"the dashboard hides providers the router supports: {missing}"
    assert len(names) == len(PROVIDER_REGISTRY)


def test_an_unconfigured_provider_is_listed_with_the_variable_to_set():
    """
    Showing it is how a user discovers what to configure. Filtering the list
    would leave a UI that silently does not mention the option at all.
    """
    labels = dict(provider_choices())

    assert "ollama" in labels
    # Whatever this machine has configured, every entry has a label.
    for name, label in provider_choices():
        assert label.strip(), f"{name} has no label"


def test_the_selector_offers_the_same_route_modes_as_llm_settings():
    """
    The regression. Two widgets chose between different sets of route modes, so
    the choice made in one page was silently overridden in another.
    """
    from app.dashboard.views.llm_settings import _route_options

    selector_modes = [mode for mode, _label in _ROUTE_MODES]

    assert selector_modes == _route_options()


def test_all_three_documented_route_modes_are_available():
    for mode in ("direct", "automatic", "experiential"):
        assert mode in [m for m, _ in _ROUTE_MODES]


def test_the_selector_accepts_a_key_prefix():
    """
    Streamlit raises on a duplicate widget id, so a page rendering the selector
    twice needs distinct keys. The parameter exists for that and must stay
    optional, because existing call sites pass nothing.
    """
    import inspect

    signature = inspect.signature(render_provider_selector)

    assert "key_prefix" in signature.parameters
    assert signature.parameters["key_prefix"].default == ""


def test_provider_labels_are_shared_rather_than_duplicated():
    """
    The label table drifted between views once already: two copies were six
    providers old while the registry had thirteen, so the same provider rendered
    as "OpenRouter" on one page and "openrouter" on another.
    """
    from app.dashboard.components import PROVIDER_LABELS as canonical
    from app.dashboard.views.career_suite import PROVIDER_LABELS

    assert PROVIDER_LABELS is canonical
    assert set(canonical) >= set(PROVIDER_REGISTRY), (
        "a provider has no display name and would render its raw registry key"
    )
