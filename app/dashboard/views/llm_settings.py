"""
LLM and model settings: choose the provider and model, once, for the workflow.

The point of this page is that a single choice is made here and every later
step uses it. Previously each page rendered its own copy of the provider
selector, so a user could analyse with one provider and generate with another
without noticing.

Three states are kept strictly apart, because collapsing them is how a user ends
up believing an untested provider works:

``CONFIGURED``
    The backend found a credential and a model. Says nothing about reachability.
``AVAILABLE``
    The provider's own listing answered, so the models shown are what it
    currently offers.
``PASSED`` / ``FAILED``
    A model answered a real request. Only a smoke test or a completed generation
    can produce these. Everything else is ``NOT TESTED``.

Discovery is optional. A provider with no listing still works with a hand-set
model id, and the page says so rather than showing an empty list as if it were a
failure.
"""

from __future__ import annotations

import streamlit as st

from app.dashboard import theme, workflow
from app.dashboard.components import render_quota_card
from app.dashboard.helpers import fetch_model_discovery, fetch_quota_status, get_api_base

#: Display names for the providers whose canonical name is not obvious.
_LABELS = {
    "ollama": "Ollama (local)",
    "omniroute": "OmniRoute (local gateway)",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "claude": "Anthropic Claude",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
    "cerebras": "Cerebras",
    "cloudflare": "Cloudflare Workers AI",
    "github": "GitHub Models",
    "huggingface": "Hugging Face",
    "experiential": "Experiential Labs",
}

#: Providers that run on the operator's own machine. Worth calling out because
#: they are the ones where "free" and "private" are both true.
_LOCAL = {"ollama", "omniroute"}


def _label(name: str) -> str:
    return _LABELS.get(name, name)


def _provider_table(api_base: str) -> list[dict]:
    """
    One row per registered provider, with its three states kept distinct.

    Reads the catalogue (what is configured) and the discovery listing (what is
    available) separately, because a provider can be configured and still
    unreachable, and showing only one of those facts is misleading either way.
    """
    # Discovery alone answers what this table needs: which providers are
    # configured, and what each one's listing said. backend-status is read on
    # the overview and diagnostics pages; fetching it here as well would be a
    # second request for data this function never uses.
    catalog = (fetch_model_discovery(api_base) or {}).get("providers") or {}

    rows: list[dict] = []

    for name, info in catalog.items():
        if not isinstance(info, dict):
            continue

        configured = bool(info.get("configured"))
        discovery_status = info.get("status") or "unknown"
        models = info.get("models") or []

        # Ollama has its own health probe, which is more specific than a
        # discovery listing: it knows whether the *configured* model is
        # installed, not just whether a listing answered.
        if name == "ollama":
            listing_answered = discovery_status == "ok"
        else:
            listing_answered = discovery_status == "ok"

        if not configured:
            available = theme.NOT_CONFIGURED
        elif listing_answered and models:
            # The provider's own listing answered, so the models shown are what
            # it currently offers.
            available = theme.AVAILABLE
        elif discovery_status in {"manual_configuration", "discovery_failed"}:
            # Both of these mean "configured, but the listing did not help",
            # not "broken". A provider that offers no listing is still usable
            # with a hand-set model, and one whose listing could not be read may
            # still be perfectly reachable. Reporting either as a failure tells
            # the user their working credential is broken.
            available = theme.CONFIGURED
        else:
            available = theme.FAILED

        rows.append(
            {
                "provider": _label(name),
                # The canonical name, kept alongside the label. Two tables
                # describe the same providers; they have to be joinable.
                "id": name,
                "side": "local" if name in _LOCAL else "API",
                "configured": theme.PASSED if configured else theme.NOT_CONFIGURED,
                "available": available,
                "smoke_tested": theme.NOT_TESTED,
                "model": info.get("configured_model") or "—",
                "models_listed": len(models),
                "note": info.get("detail") or "",
            }
        )

    return rows


def _model_picker(api_base: str, provider: str) -> str:
    """
    Choose a model for ``provider``.

    The configured model always appears even when discovery found nothing, so
    there is always something selectable. Discovery only ever *adds* options.
    """
    discovery = fetch_model_discovery(api_base, provider) or {}
    rows = discovery.get("models") or []

    configured = discovery.get("configured_model") or ""
    ids: list[str] = []

    for row in rows:
        name = row.get("model")
        if name and name not in ids:
            ids.append(name)

    if configured and configured not in ids:
        # Prefer the configured model at the top: it is what a request uses
        # today, so it is the safe default to leave unchanged.
        ids.insert(0, configured)

    if not ids:
        st.caption(
            "No model listing available for this provider. Set its model "
            "variable in `.env`, or type an id below."
        )
        return st.text_input(
            "Model id",
            value=configured,
            key=f"llm_manual_model_{provider}",
        )

    if discovery.get("status") == "discovery_failed":
        theme.pills(
            [
                (
                    theme.FAILED,
                    "the listing could not be read; showing the configured model",
                )
            ]
        )

    if discovery.get("status") == "manual_configuration":
        theme.pills([(theme.NOT_TESTED, "this provider exposes no model listing")])

    # A gateway that can choose for itself gets that as the first option.
    if provider == "omniroute":
        if "auto" not in ids:
            ids.insert(0, "auto")

    default_index = ids.index(configured) if configured in ids else 0

    chosen = st.selectbox(
        "Model",
        ids,
        index=default_index,
        key=f"llm_model_{provider}",
        format_func=lambda value: (f"{value}  (configured)" if value == configured else value),
    )

    # Show what the provider actually said about the chosen model, if anything.
    for row in rows:
        if row.get("model") != chosen:
            continue
        facts = []
        if row.get("context_length"):
            facts.append(f"context {row['context_length']:,} tokens")
        if row.get("size_bytes"):
            facts.append(f"{row['size_bytes'] / 1e9:.1f} GB on disk")
        if row.get("families"):
            facts.append(", ".join(row["families"][:3]))
        if row.get("missing_from_discovery"):
            facts.append("NOT in the provider's current listing")
        if facts:
            st.caption(" · ".join(facts))
        break
    else:
        if chosen and chosen != configured:
            st.caption(
                "This id is not in the provider's current listing. It may still "
                "work; it may also be gone."
            )

    return chosen


def render_llm_settings_page() -> None:
    """Render the LLM and model settings page."""
    theme.step_header(
        "⚙",
        "LLM & Model Settings",
        "Choose once here. The analysis and every generation step use this " "choice.",
    )

    api_base = get_api_base()

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        theme.section_header("Provider", "🧠")

        catalog = fetch_model_discovery(api_base) or {}
        providers = catalog.get("providers") or {}

        if not providers:
            theme.empty_state(
                "No providers reported",
                f"The backend at {api_base} did not return a provider list.",
                icon="🧠",
            )
            return

        names = sorted(providers)

        current = workflow.generation_choice()["provider"]
        index = names.index(current) if current in names else 0

        provider = st.selectbox(
            "Active provider",
            names,
            index=index,
            key="llm_provider",
            format_func=_label,
        )

        info = providers.get(provider) or {}
        configured = bool(info.get("configured"))

        if configured:
            theme.pills(
                [
                    (
                        theme.CONFIGURED,
                        f"model: {info.get('configured_model') or '(unset)'}",
                    )
                ]
            )
        else:
            detail = info.get("detail") or "not configured"
            theme.pills([(theme.NOT_CONFIGURED, detail)])

        if provider in _LOCAL:
            st.caption(
                f"🔒 {provider} runs on this machine. With `LLM_MODE` set to a "
                f"local-only value, a CV never leaves the machine."
            )

        theme.rule()
        theme.section_header("Routing", "🧭")

        route_mode = st.selectbox(
            "Route mode",
            ["direct", "automatic", "experiential"],
            key="llm_route_mode",
            help=(
                "direct: exactly this provider, no fallback.\n\n"
                "automatic: this provider first, then the configured chain.\n\n"
                "experiential: route through the Experiential Labs gateway."
            ),
        )

        st.caption(
            "The backend's own LLM_MODE still governs whether a request may "
            "reach a cloud provider. The dashboard cannot widen that."
        )

    with right:
        theme.section_header("Model", "🎯")
        model = _model_picker(api_base, provider)

        theme.rule()
        theme.section_header("Quota", "📊")
        quota = fetch_quota_status(api_base)
        if quota:
            shown = 0
            for name, data in quota.items():
                if name in _LOCAL:
                    # A local provider has no quota. Showing an empty card for
                    # it implies a limit that does not exist.
                    continue
                if shown >= 3:
                    break
                render_quota_card(name, data, _LABELS)
                shown += 1
            if not shown:
                st.caption(
                    "No rate-limited providers are configured. Local inference " "has no quota."
                )
        else:
            st.caption("Quota information is not available.")

    theme.rule()

    # -- the decision, made explicit before it is applied -------------------
    theme.section_header("This workflow will use", "✅")
    theme.kv_table(
        [
            ("Provider", _label(provider)),
            ("Model", model or "(provider default)"),
            ("Route mode", route_mode),
            ("Configured", "yes" if configured else "no"),
            (
                "Smoke tested",
                "no — run `python -m scripts.llm_smoke` to verify",
            ),
        ]
    )

    if st.button(
        "Apply to this workflow",
        type="primary",
        key="llm_apply",
    ):
        workflow.set_generation_choice(provider, model, route_mode)
        theme.pills(
            [
                (
                    theme.PASSED,
                    f"{_label(provider)} / {model or 'default'} / {route_mode}",
                )
            ]
        )

    theme.rule()
    theme.section_header("All providers", "📋")
    st.dataframe(_provider_table(api_base), width="stretch", hide_index=True)
    st.caption(
        "CONFIGURED means a credential and a model were found. AVAILABLE means "
        "the provider's own listing answered. No row says a model works until "
        "one has actually been tested."
    )
