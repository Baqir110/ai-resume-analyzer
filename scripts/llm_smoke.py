"""
Provider smoke test.

Checks, per configured provider: reachable, model available, a harmless prompt
succeeds, a usable textual response comes back, plus latency and token usage.

Safety rules this script follows:

* The prompt is a fixed, harmless string. A real CV or job description is never
  sent, whatever the provider.
* No credential is printed, and no prompt or response body is printed - only
  lengths and status.
* A provider that is not configured reports ``NOT CONFIGURED`` and does not
  fail the run, so a partially configured machine still gives useful output.
* Nothing is written to the repository.

Usage:
    python -m scripts.llm_smoke              # every configured provider
    python -m scripts.llm_smoke ollama       # one provider
    python -m scripts.llm_smoke --list       # show the model catalogue only
    python -m scripts.llm_smoke --json       # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from app.services.llm.provider import LLMService, canonical_provider, get_spec

#: Deliberately trivial and provider-agnostic. No user data of any kind.
SMOKE_PROMPT = "Reply with exactly the word: acknowledged"

STATUS_OK = "OK"
STATUS_NOT_CONFIGURED = "NOT CONFIGURED"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"


def _row(provider: str) -> dict[str, Any]:
    """One provider's result. Contains no secrets and no content."""
    return {
        "provider": provider,
        "status": STATUS_SKIPPED,
        "model": "",
        "detail": "",
        "latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "response_chars": None,
        "answer_is_text": False,
    }


def _usage() -> dict[str, Any]:
    return LLMService.get_last_usage()


def smoke_provider(provider: str, *, timeout_note: str = "") -> dict[str, Any]:
    """
    Exercise one provider end to end.

    A failure is reported, never raised: the point of a smoke test is to keep
    going and report, so one unreachable provider does not hide the state of the
    others.
    """
    canonical = canonical_provider(provider)
    row = _row(provider)

    if not canonical:
        row["status"] = STATUS_FAILED
        row["detail"] = f"Unknown provider. Known: {', '.join(LLMService.SUPPORTED_PROVIDERS)}."
        return row

    row["provider"] = canonical

    if not LLMService._provider_is_configured(canonical):
        row["status"] = STATUS_NOT_CONFIGURED
        row["detail"] = _not_configured(canonical)
        return row

    row["model"] = LLMService.get_default_model(canonical)

    if not row["model"]:
        row["status"] = STATUS_NOT_CONFIGURED
        row["detail"] = "No model configured. Set the provider's model variable to choose one."
        return row

    started = time.perf_counter()
    try:
        answer = LLMService.generate(
            prompt=SMOKE_PROMPT,
            provider=canonical,
            # direct: a smoke test must exercise the provider it was asked
            # about, not fall through to something else that happens to work.
            route_mode="direct",
            task="generic",
        )
    except Exception as exc:
        row["latency_s"] = round(time.perf_counter() - started, 2)
        row["status"] = STATUS_FAILED
        row["detail"] = LLMService.classify_for_report(exc, canonical)
        return row

    row["latency_s"] = round(time.perf_counter() - started, 2)
    text = (answer or "").strip()
    row["response_chars"] = len(text)
    row["answer_is_text"] = bool(text)

    usage = _usage()
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["completion_tokens"] = usage.get("completion_tokens")

    if not text:
        row["status"] = STATUS_FAILED
        row["detail"] = "Provider returned no usable text."
        return row

    row["status"] = STATUS_OK
    row["detail"] = "Usable text returned."
    if timeout_note:
        row["detail"] += f" {timeout_note}"
    return row


def _not_configured(provider: str) -> str:
    spec = get_spec(provider)
    if spec is None:
        return "Not a known provider."
    if not spec.key_env:
        return f"No model set. Set {spec.model_env}."
    return f"No credential. Set {' or '.join(spec.key_env)}."


def discover(provider: str) -> dict[str, Any]:
    """Model listing for one provider."""
    return LLMService.list_models(provider)


def print_table(rows: list[dict[str, Any]]) -> None:
    width = max((len(row["provider"]) for row in rows), default=10)

    print("=" * 78)
    print(f"{'PROVIDER':<{width}}  {'STATUS':<16} {'MODEL':<26} {'SECS':>7}  DETAIL")
    print("=" * 78)

    for row in rows:
        model = (row["model"] or "-")[:26]
        latency = f"{row['latency_s']:.1f}" if row["latency_s"] is not None else "-"
        print(
            f"{row['provider']:<{width}}  {row['status']:<16} {model:<26} "
            f"{latency:>7}  {row['detail']}"
        )

    print()
    print("Token usage and response size, where reported:")
    for row in rows:
        if row["status"] != STATUS_OK:
            continue
        print(
            f"  {row['provider']:<14} prompt={row['prompt_tokens']} "
            f"completion={row['completion_tokens']} "
            f"response_chars={row['response_chars']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safe LLM provider smoke test. Sends no user data.",
    )
    parser.add_argument(
        "providers",
        nargs="*",
        help="providers to test (default: every registered provider)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show the model catalogue and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a table",
    )
    args = parser.parse_args(argv)

    targets = args.providers or list(LLMService.SUPPORTED_PROVIDERS)

    if args.list:
        payload = {}
        for provider in targets:
            result = discover(provider)
            payload[provider] = {
                "status": result.get("status"),
                "configured": result.get("configured"),
                "models": [row["model"] for row in result.get("models", [])],
                "detail": result.get("detail", ""),
            }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for provider, info in payload.items():
                print(f"\n{provider}  [{info['status']}]")
                if info["detail"]:
                    print(f"  {info['detail']}")
                for model in info["models"]:
                    print(f"    - {model}")
        return 0

    rows = [smoke_provider(provider) for provider in targets]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("LLM provider smoke test")
        print(f"Prompt: {SMOKE_PROMPT!r}  (fixed, no user data is sent)")
        print()
        print_table(rows)

    failed = [r for r in rows if r["status"] == STATUS_FAILED]
    unknown = [
        r for r in rows if r["status"] not in {STATUS_OK, STATUS_NOT_CONFIGURED, STATUS_FAILED}
    ]
    if unknown:
        print(f"Unexpected statuses: {[r['provider'] for r in unknown]}")

    if failed:
        print()
        print(f"{len(failed)} provider(s) FAILED: {', '.join(r['provider'] for r in failed)}")
        return 1

    print()
    print(
        f"No failures. {sum(1 for r in rows if r['status'] == STATUS_OK)} reachable, "
        f"{sum(1 for r in rows if r['status'] == STATUS_NOT_CONFIGURED)} not configured."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
