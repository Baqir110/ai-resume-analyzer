"""
restructure_project.py

Reorganize app/services/ into domain subpackages and move event_log.py into
app/core/ (it's shared infrastructure, not a service).

Also rewrites every import path in the codebase so nothing breaks.

Creates a full backup of app/ before touching anything.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
APP = ROOT / "app"
SERVICES = APP / "services"


# ---------------------------------------------------------------------------
# Move map: old path (relative to app/) -> new path (relative to app/)
# ---------------------------------------------------------------------------

MOVE_MAP: dict[str, str] = {
    "services/parser.py":             "services/parsing/resume_parser.py",
    "services/analyzer.py":           "services/analysis/ats_analyzer.py",
    "services/suggestions.py":        "services/analysis/suggestions.py",
    "services/llm_provider.py":       "services/llm/provider.py",
    "services/quota_tracker.py":      "services/llm/quota_tracker.py",
    "services/event_log.py":          "core/event_log.py",
    "services/optimizer.py":          "services/cv/optimizer.py",
    "services/latex_generator.py":    "services/cv/latex_generator.py",
    "services/diff_preview.py":       "services/cv/diff_preview.py",
    "services/cover_letter.py":       "services/career/cover_letter.py",
    "services/interview_prep.py":     "services/career/interview_prep.py",
    "services/linkedin_optimizer.py": "services/career/linkedin_optimizer.py",
    "services/audit_matrix.py":       "services/career/audit_matrix.py",
    "services/bulk_analyzer.py":      "services/bulk/bulk_analyzer.py",
    "services/tracker.py":            "services/tracking/tracker.py",
}


# ---------------------------------------------------------------------------
# Import map: old module path -> new module path
# ---------------------------------------------------------------------------

MODULE_MAP: dict[str, str] = {
    "app.services.parsing.resume_parser":             "app.services.parsing.resume_parser",
    "app.services.analysis.ats_analyzer":           "app.services.analysis.ats_analyzer",
    "app.services.analysis.suggestions":        "app.services.analysis.suggestions",
    "app.services.llm.provider":       "app.services.llm.provider",
    "app.services.llm.quota_tracker":      "app.services.llm.quota_tracker",
    "app.core.event_log":          "app.core.event_log",
    "app.services.cv.optimizer":          "app.services.cv.optimizer",
    "app.services.cv.latex_generator":    "app.services.cv.latex_generator",
    "app.services.cv.diff_preview":       "app.services.cv.diff_preview",
    "app.services.career.cover_letter":       "app.services.career.cover_letter",
    "app.services.career.interview_prep":     "app.services.career.interview_prep",
    "app.services.career.linkedin_optimizer": "app.services.career.linkedin_optimizer",
    "app.services.career.audit_matrix":       "app.services.career.audit_matrix",
    "app.services.bulk.bulk_analyzer":      "app.services.bulk.bulk_analyzer",
    "app.services.tracking.tracker":            "app.services.tracking.tracker",
}


# ---------------------------------------------------------------------------
# New packages that need __init__.py
# ---------------------------------------------------------------------------

NEW_PACKAGES = [
    "services/parsing",
    "services/analysis",
    "services/llm",
    "services/cv",
    "services/career",
    "services/bulk",
    "services/tracking",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _say(msg: str) -> None:
    print(f"  {msg}")


def _move(old: Path, new: Path) -> None:
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))


def _rewrite_imports(text: str) -> tuple[str, int]:
    """
    Rewrite import statements in a Python source file.

    Handles:
        from app.services.X import ...
        from app.services.X import (
        import app.services.X
        from app.services import X       (rare, but possible)
    """
    hits = 0

    for old, new in MODULE_MAP.items():
        # `from app.services.llm.provider import X` and `import app.services.llm.provider`
        # Uses negative lookaround so `app.services.llm_provider.sub` isn't touched.
        pattern = re.compile(
            rf"(?<![\w.]){re.escape(old)}(?![\w.])",
        )
        new_text, n = pattern.subn(new, text)
        if n:
            hits += n
            text = new_text

    # Special case: `from app.services import X` where X is a moved module.
    for old, new in MODULE_MAP.items():
        old_name = old.rsplit(".", 1)[-1]
        new_parent = new.rsplit(".", 1)[0]

        pattern = re.compile(
            rf"from\s+app\.services\s+import\s+{re.escape(old_name)}\b"
        )
        replacement = f"from {new_parent} import {old_name}"
        new_text, n = pattern.subn(replacement, text)
        if n:
            hits += n
            text = new_text

    return text, hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not APP.is_dir():
        print(f"[ERROR] {APP} not found — run from project root")
        return 2

    # Warn if git worktree is dirty
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0 and status.stdout.strip():
            print("[WARN] Uncommitted changes detected in git.")
            print("       Consider committing first so you can git-restore if needed.")
            print()
    except Exception:
        pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / f"app_backup_{stamp}"
    print(f"Backing up {APP} -> {backup}")
    shutil.copytree(APP, backup, dirs_exist_ok=False)
    print()

    # ---------------------------------------------------------------
    # 1. Create __init__.py in new subpackages
    # ---------------------------------------------------------------
    print("Creating package markers:")
    for pkg_rel in NEW_PACKAGES:
        pkg = APP / pkg_rel
        pkg.mkdir(parents=True, exist_ok=True)
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
        _say(f"✓ {pkg_rel}/__init__.py")
    print()

    # ---------------------------------------------------------------
    # 2. Move files
    # ---------------------------------------------------------------
    print("Moving files:")
    for old_rel, new_rel in MOVE_MAP.items():
        old = APP / old_rel
        new = APP / new_rel
        if not old.exists():
            _say(f"✗ {old_rel} (missing — skipped)")
            continue
        if new.exists():
            _say(f"≈ {new_rel} (already present — skipped)")
            continue
        _move(old, new)
        _say(f"✓ {old_rel}  →  {new_rel}")
    print()

    # ---------------------------------------------------------------
    # 3. Rewrite imports across the codebase
    # ---------------------------------------------------------------
    print("Rewriting imports:")
    targets: list[Path] = []
    for pattern in ("app/**/*.py", "tests/**/*.py", "*.py"):
        targets.extend(ROOT.glob(pattern))

    total_files = 0
    total_hits = 0
    for path in sorted(set(targets)):
        if "__pycache__" in path.parts:
            continue
        if "app_backup_" in str(path):
            continue
        if path.name in {"restructure_project.py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text, hits = _rewrite_imports(text)
        if hits:
            path.write_text(new_text, encoding="utf-8")
            total_files += 1
            total_hits += hits
            rel = path.relative_to(ROOT)
            _say(f"✓ {rel}  ({hits} import{'s' if hits > 1 else ''})")
    print()
    print(f"Rewrote {total_hits} imports across {total_files} files.")
    print()

    # ---------------------------------------------------------------
    # 4. Clear __pycache__
    # ---------------------------------------------------------------
    print("Clearing __pycache__ directories:")
    removed = 0
    for cache in APP.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    _say(f"Removed {removed} cache director{'ies' if removed != 1 else 'y'}.")
    print()

    # ---------------------------------------------------------------
    # 5. Verify
    # ---------------------------------------------------------------
    print("Verifying imports:")
    ok = True
    for check in [
        "import app.main",
        "from app.api.endpoints import router",
        "from app.services.cv.latex_generator import generate_german_latex_content",
        "from app.services.llm.provider import LLMService",
        "from app.core.event_log import log_event",
    ]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", check],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=ROOT,
            )
            if result.returncode == 0:
                _say(f"✓ {check}")
            else:
                ok = False
                _say(f"✗ {check}")
                # Show first error line
                for line in result.stderr.splitlines():
                    if "Error" in line or "error" in line:
                        _say(f"    {line}")
                        break
        except Exception as exc:
            ok = False
            _say(f"✗ {check}: {exc}")

    print()
    if ok:
        print("Restructure complete.")
        print(f"Backup retained at: {backup.name}")
        print()
        print("Restart the app:")
        print("    python run.py")
        return 0

    print("Some verifications failed.")
    print(f"To restore: rm -rf app && mv {backup.name} app")
    return 1


if __name__ == "__main__":
    sys.exit(main())