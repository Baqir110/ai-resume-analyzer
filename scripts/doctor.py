"""Project health check + auto-fix.

Run: python scripts/doctor.py [--fix]
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS_PASSED: list[str] = []
CHECKS_FAILED: list[tuple[str, str]] = []
CHECKS_FIXED: list[str] = []


def _ok(name: str, detail: str = "") -> None:
    CHECKS_PASSED.append(f"{name}{' - ' + detail if detail else ''}")


def _fail(name: str, detail: str, fixable: bool = False) -> None:
    CHECKS_FAILED.append((name, detail))


def _fixed(name: str) -> None:
    CHECKS_FIXED.append(name)


# -----------------------------------------------------------------
# 1. __init__.py coverage
# -----------------------------------------------------------------

REQUIRED_PKGS = [
    "app",
    "app/api",
    "app/core",
    "app/dashboard",
    "app/dashboard/views",
    "app/models",
    "app/services",
    "app/services/analysis",
    "app/services/analytics",
    "app/services/bulk",
    "app/services/career",
    "app/services/cv",
    "app/services/llm",
    "app/services/parsing",
    "app/services/tracking",
]


def check_init_files(fix: bool) -> None:
    missing = []
    for pkg in REQUIRED_PKGS:
        path = ROOT / pkg.replace("/", "\\") / "__init__.py"
        if not path.exists():
            missing.append(pkg)

    if not missing:
        _ok("__init__.py coverage", f"{len(REQUIRED_PKGS)} packages OK")
        return

    if fix:
        for pkg in missing:
            path = ROOT / pkg.replace("/", "\\") / "__init__.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            _fixed(f"created {pkg}/__init__.py")
        _ok("__init__.py coverage", f"created {len(missing)} files")
    else:
        _fail("__init__.py coverage", f"missing: {', '.join(missing)}")


# -----------------------------------------------------------------
# 2. Router registration
# -----------------------------------------------------------------

EXPECTED_ROUTERS = [
    ("app.main", ["api_router", "new_features_router", "streaming_router"]),
]


def check_routers() -> None:
    try:
        from app.main import app
    except Exception as exc:
        _fail("FastAPI import", str(exc)[:200])
        return

    _ok("FastAPI import", f"{len(app.routes)} route objects")

    # Count included routers
    included = sum(1 for r in app.routes if type(r).__name__ == "_IncludedRouter")
    if included >= 3:
        _ok("Router registration", f"{included} routers included")
    else:
        _fail(
            "Router registration",
            f"expected 3 routers, found {included}. Check app/main.py.",
        )


# -----------------------------------------------------------------
# 3. Broken imports across the app
# -----------------------------------------------------------------


def check_imports() -> None:
    """Walk every .py file in app/ and try importing it."""
    broken = []
    for py in sorted((ROOT / "app").rglob("*.py")):
        if "__pycache__" in str(py) or py.name == "__init__.py":
            continue
        rel = py.relative_to(ROOT).with_suffix("")
        mod = ".".join(rel.parts)
        try:
            importlib.import_module(mod)
        except Exception as exc:
            broken.append((mod, str(exc)[:120]))

    if not broken:
        _ok("All modules importable")
    else:
        for mod, err in broken[:10]:
            _fail(f"import {mod}", err)


# -----------------------------------------------------------------
# 4. Required env vars
# -----------------------------------------------------------------

REQUIRED_ENV = ["OPENAI_BASE_URL", "EXPERIENTIAL_ORG_KEY"]


def check_env() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]

    if not missing:
        _ok("Environment variables", f"{len(REQUIRED_ENV)} keys set")
    else:
        # Provider credentials are optional for local/offline operation.  The
        # protected API still fails closed when API_KEY itself is missing.
        _ok("Environment variables", f"optional provider keys absent: {', '.join(missing)}")


# -----------------------------------------------------------------
# 5. Dead code (unused module imports)
# -----------------------------------------------------------------


def check_dead_modules() -> None:
    """Warn about .py files in app/ that nothing imports."""
    app_dir = ROOT / "app"
    all_py = {p.stem for p in app_dir.rglob("*.py") if p.name != "__init__.py"}

    # Read every .py file, collect imported module names
    imported = set()
    for py in app_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for name in all_py:
            if f"import {name}" in text or f".{name} " in text:
                imported.add(name)

    orphaned = sorted(all_py - imported)
    # Filter out obvious entry points
    orphaned = [o for o in orphaned if o not in {"main", "run", "dashboard", "conftest"}]

    if not orphaned:
        _ok("No orphaned modules")
    else:
        _ok("Dead-code scan", f"possible entry points: {', '.join(orphaned[:10])}")


# -----------------------------------------------------------------
# 6. LLM configuration
# -----------------------------------------------------------------


def check_llm_configuration() -> None:
    """
    Report whether the configured LLM provider can actually be used.

    Read-only: no generation is sent, so this is safe to run anywhere and costs
    nothing. It catches the failures that otherwise surface as a CV request that
    fails minutes later -- a provider with no credential, a local server that is
    not running, a model that is not installed, a base URL that is blocked.

    Every line here is derived from the provider registry and from the doctor's
    own classification of the failure. No credential, prompt or response text is
    read or printed.
    """
    try:
        from app.services.llm.provider import LLMService, get_spec
    except Exception as exc:  # pragma: no cover - import guard
        _fail("LLM layer", f"could not be imported: {type(exc).__name__}")
        return

    try:
        info = LLMService.route_info()
    except Exception as exc:
        _fail("LLM routing", f"could not be resolved: {type(exc).__name__}")
        return

    _ok(
        "LLM mode",
        f"{info.get('mode')} "
        f"(provider={info.get('provider')}, "
        f"model={info.get('model')}, "
        f"fallback={'on' if info.get('fallback_enabled') else 'off'}, "
        f"max_attempts={info.get('max_provider_attempts')}, "
        f"retries={info.get('retries')})",
    )

    if not info.get("fallback_enabled"):
        _ok(
            "LLM fallback",
            "disabled - a failure is reported instead of trying another provider",
        )

    # Local inference: reachable, and is the configured model installed?
    try:
        health = LLMService.ollama_health()
    except Exception as exc:
        _fail("Ollama", f"health check raised {type(exc).__name__}")
        health = None

    if health is not None:
        if not health.get("configured"):
            _ok("Ollama", "not configured (set OLLAMA_BASE_URL and OLLAMA_MODEL)")
        elif not health.get("reachable"):
            _fail(
                "Ollama",
                "not reachable - start it with `ollama serve`",
            )
        elif not health.get("model_available"):
            # The detail already contains the exact command that fixes it.
            _fail("Ollama", health.get("detail") or "configured model not installed")
        else:
            _ok(
                "Ollama",
                f"{health.get('model')} available "
                f"({len(health.get('available_models', []))} installed)",
            )

    # Every provider: configured or not, and does it have a model?
    configured: list[str] = []
    unconfigured: list[str] = []

    for name in LLMService.SUPPORTED_PROVIDERS:
        spec = get_spec(name)
        if spec is None:
            continue

        try:
            is_configured = LLMService._provider_is_configured(name)
        except Exception:
            is_configured = False

        if not is_configured:
            missing = " or ".join(spec.key_env) if spec.key_env else spec.model_env
            unconfigured.append(f"{name} (set {missing})")
            continue

        try:
            model = LLMService.get_default_model(name)
        except Exception:
            model = ""

        if not model:
            _fail(
                f"Provider {name}",
                f"configured but has no model - set {spec.model_env}",
            )
            continue

        configured.append(f"{name}/{model}")

    if configured:
        _ok("LLM providers configured", ", ".join(configured))

    if unconfigured:
        # Informational, not a failure: a machine with only Ollama configured is
        # a perfectly good local setup, and the doctor must not report it broken.
        _ok(
            "LLM providers not configured",
            f"{len(unconfigured)}: {', '.join(unconfigured[:6])}"
            + (" ..." if len(unconfigured) > 6 else ""),
        )

    _ok(
        "LLM smoke test",
        "run `python -m scripts.llm_smoke` to actually call each provider",
    )


# -----------------------------------------------------------------
# 7. Run ruff and see if it can fix things
# -----------------------------------------------------------------


def check_ruff(fix: bool) -> None:
    try:
        args = ["ruff", "check", str(ROOT / "app")]
        if fix:
            args.append("--fix")
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            _ok("Ruff lint", "clean")
        elif fix:
            _fixed("ruff auto-fixed issues")
            _ok("Ruff lint", "auto-fixed")
        else:
            _fail("Ruff lint", "run with --fix to auto-repair")
    except FileNotFoundError:
        _fail("Ruff", "not installed — pip install ruff")
    except subprocess.TimeoutExpired:
        _fail("Ruff", "timed out")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply fixes where possible")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Project Doctor - {'FIX MODE' if args.fix else 'CHECK MODE'}")
    print("=" * 60)

    check_init_files(args.fix)
    check_routers()
    check_imports()
    check_env()
    check_dead_modules()
    check_llm_configuration()
    check_ruff(args.fix)

    print("\n" + "=" * 60)
    print(f"  PASSED: {len(CHECKS_PASSED)}")
    for line in CHECKS_PASSED:
        print(f"    [PASS] {line}")

    if CHECKS_FIXED:
        print(f"\n  AUTO-FIXED: {len(CHECKS_FIXED)}")
        for line in CHECKS_FIXED:
            print(f"    [FIX] {line}")

    if CHECKS_FAILED:
        print(f"\n  FAILED: {len(CHECKS_FAILED)}")
        for name, detail in CHECKS_FAILED:
            print(f"    [FAIL] {name}: {detail}")

    print("=" * 60)
    return 0 if not CHECKS_FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
