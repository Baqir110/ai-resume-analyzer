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
    CHECKS_PASSED.append(f"{name}{' — ' + detail if detail else ''}")


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
        _fail("Environment variables", f"missing: {', '.join(missing)}")


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
        _fail(
            "Possible dead code",
            f"{len(orphaned)} unreferenced: {', '.join(orphaned[:10])}",
        )


# -----------------------------------------------------------------
# 6. Run ruff and see if it can fix things
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
    print(f"  Project Doctor — {'FIX MODE' if args.fix else 'CHECK MODE'}")
    print("=" * 60)

    check_init_files(args.fix)
    check_routers()
    check_imports()
    check_env()
    check_dead_modules()
    check_ruff(args.fix)

    print("\n" + "=" * 60)
    print(f"  PASSED: {len(CHECKS_PASSED)}")
    for line in CHECKS_PASSED:
        print(f"    ✅ {line}")

    if CHECKS_FIXED:
        print(f"\n  AUTO-FIXED: {len(CHECKS_FIXED)}")
        for line in CHECKS_FIXED:
            print(f"    🔧 {line}")

    if CHECKS_FAILED:
        print(f"\n  FAILED: {len(CHECKS_FAILED)}")
        for name, detail in CHECKS_FAILED:
            print(f"    ❌ {name}: {detail}")

    print("=" * 60)
    return 0 if not CHECKS_FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
