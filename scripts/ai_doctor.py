"""AI-powered code doctor.

Runs the full diagnostic pipeline (ruff, mypy, pytest, custom checks),
then asks the LLM to analyze failures and propose patches.

Usage:
    python scripts/ai_doctor.py              # analyze only
    python scripts/ai_doctor.py --fix        # apply safe patches
    python scripts/ai_doctor.py --fix --all  # apply all suggested patches
    python scripts/ai_doctor.py --json       # machine-readable output
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ===========================================================================
# Terminal colors (works on Windows 10+)
# ===========================================================================


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


if not sys.stdout.isatty():
    C.disable()


def _h(text: str) -> str:
    return f"{C.BOLD}{C.CYAN}{text}{C.RESET}"


def _ok(text: str) -> str:
    return f"{C.GREEN}[OK]{C.RESET} {text}"


def _fail(text: str) -> str:
    return f"{C.RED}[FAIL]{C.RESET} {text}"


def _warn(text: str) -> str:
    return f"{C.YELLOW}[WARN]{C.RESET} {text}"


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class Finding:
    """One issue detected by a checker."""

    checker: str
    severity: str
    rule: str
    file: str | None
    line: int | None
    message: str
    snippet: str = ""
    suggested_fix: dict | None = None
    auto_safe: bool = False


@dataclass
class Report:
    """Aggregate of all findings from a doctor run."""

    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    findings: list[Finding] = field(default_factory=list)
    files_changed: int = 0
    summary: dict = field(default_factory=dict)


# ===========================================================================
# Checkers
# ===========================================================================


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=ROOT,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def check_ruff() -> list[Finding]:
    """Run ruff and parse its JSON output."""
    code, stdout, stderr = _run(["ruff", "check", "app", "--output-format=json"])
    if code == 127:
        return []
    if not stdout.strip():
        return []

    try:
        issues = json.loads(stdout)
    except json.JSONDecodeError:
        return []

    findings = []
    for issue in issues:
        location = issue.get("location") or {}
        findings.append(
            Finding(
                checker="ruff",
                severity="error" if issue.get("fix") is None else "warning",
                rule=issue.get("code", "?"),
                file=issue.get("filename"),
                line=location.get("row"),
                message=issue.get("message", ""),
                auto_safe=issue.get("fix") is not None,
            )
        )
    return findings


def check_mypy() -> list[Finding]:
    """Run mypy and parse its text output."""
    code, stdout, stderr = _run(["mypy", "app", "--no-error-summary"])
    if code == 127 or not stdout.strip():
        return []

    pattern = re.compile(
        r"^(.+?):(\d+):(?:\d+:)?\s*(error|note|warning):\s*(.+?)(?:\s*\[([^\]]+)\])?$"
    )
    findings = []
    for line in stdout.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        file, line_no, sev, msg, code_tag = m.groups()
        findings.append(
            Finding(
                checker="mypy",
                severity=sev,
                rule=code_tag or "mypy",
                file=file,
                line=int(line_no),
                message=msg.strip(),
            )
        )
    return findings


def check_pytest() -> list[Finding]:
    """Run pytest, collect test failures."""
    code, stdout, stderr = _run(
        [sys.executable, "-m", "pytest", "-m", "not integration", "-q", "--tb=line"],
        timeout=180,
    )
    if code == 0:
        return []

    findings = []
    for line in (stdout + stderr).splitlines():
        m = re.match(r"^(.+?\.py):(\d+):\s*(.+)$", line)
        if m:
            file, line_no, msg = m.groups()
            findings.append(
                Finding(
                    checker="pytest",
                    severity="error",
                    rule="test_fail",
                    file=file,
                    line=int(line_no),
                    message=msg[:200],
                )
            )
    return findings


def check_imports() -> list[Finding]:
    """Check that every module in app/ imports cleanly.

    Runs each import in a subprocess with a timeout so modules that do
    network calls or start Streamlit at import time don't hang the doctor.
    """
    findings: list[Finding] = []
    timeout = 8  # seconds per module

    # Modules that are Streamlit entry points or scripts — not designed to
    # be imported as libraries (they run code at import time).
    skip_prefixes = (
        "app.dashboard.main",
        "app.dashboard.views",
    )

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }

    for py in sorted((ROOT / "app").rglob("*.py")):
        if "__pycache__" in str(py) or py.name == "__init__.py":
            continue

        rel = py.relative_to(ROOT).with_suffix("")
        mod = ".".join(rel.parts)

        if any(mod.startswith(p) for p in skip_prefixes):
            continue

        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {mod}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=ROOT,
                env=env,
            )
        except subprocess.TimeoutExpired:
            findings.append(
                Finding(
                    checker="import",
                    severity="warning",
                    rule="import_timeout",
                    file=str(py.relative_to(ROOT)),
                    line=None,
                    message=(
                        f"import took >{timeout}s "
                        f"(likely runs network or UI code at module level)"
                    ),
                )
            )
            continue
        except Exception as exc:
            findings.append(
                Finding(
                    checker="import",
                    severity="error",
                    rule="import_crash",
                    file=str(py.relative_to(ROOT)),
                    line=None,
                    message=f"{type(exc).__name__}: {exc}"[:200],
                )
            )
            continue

        if result.returncode != 0:
            output = (result.stderr or "") + "\n" + (result.stdout or "")
            lines = [ln for ln in output.splitlines() if ln.strip()]
            message = lines[-1] if lines else "import failed"
            findings.append(
                Finding(
                    checker="import",
                    severity="error",
                    rule="import_fail",
                    file=str(py.relative_to(ROOT)),
                    line=None,
                    message=message[:200],
                )
            )

    return findings


def check_docstrings() -> list[Finding]:
    """Find public functions/classes missing docstrings (informational)."""
    findings = []
    for py in sorted((ROOT / "app").rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if m:
                name = m.group(2)
                if name.startswith("_"):
                    continue
                if i < len(lines):
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if not next_line.startswith(('"""', "'''")):
                        findings.append(
                            Finding(
                                checker="docstring",
                                severity="info",
                                rule="missing_docstring",
                                file=str(py.relative_to(ROOT)),
                                line=i,
                                message=f"{m.group(1)} {name} has no docstring",
                            )
                        )
    return findings


# ===========================================================================
# AI analysis
# ===========================================================================

_AI_PROMPT = """You are a senior Python engineer reviewing a codebase.

The deterministic checkers produced the following findings:
{findings}

For each finding, either:
1. If it's safe to auto-fix (like a missing import, a typo in a variable name, an obvious bug), return a patch.
2. If it requires human judgment, return a comment explaining what to review.
3. If it's a false positive, say so and suggest disabling the rule.

Return ONLY a JSON array, no markdown fences. Each element:
{{
  "finding_index": <int>,
  "verdict": "auto_fix" | "needs_review" | "false_positive",
  "explanation": "<1-2 sentences explaining the issue and your reasoning>",
  "patch": {{
    "file": "<relative path>",
    "search": "<exact string to find in the file; must be unique>",
    "replace": "<replacement string>"
  }} | null,
  "confidence": <0.0 to 1.0>
}}

Rules:
- Only include a patch if you are >= 0.9 confident it is correct.
- The search string must be unique within the file.
- Never patch files outside the project or in .venv, __pycache__, or tests/fixtures.
- For test failures, prefer needs_review unless the fix is obvious.
- For mypy errors about fcntl/msvcrt on Windows, prefer false_positive.
"""


def ai_analyze(findings: list[Finding], provider: str = "experiential") -> list[dict]:
    """Ask the LLM to analyze findings and propose fixes."""
    if not findings:
        return []

    from app.services.llm.provider import LLMService

    finding_payload = []
    for i, f in enumerate(findings):
        snippet = ""
        if f.file and f.line:
            snippet = _read_snippet(ROOT / f.file, f.line, context=5)
        finding_payload.append(
            {
                "index": i,
                "checker": f.checker,
                "rule": f.rule,
                "file": f.file,
                "line": f.line,
                "message": f.message,
                "snippet": snippet,
            }
        )

    prompt = _AI_PROMPT.format(findings=json.dumps(finding_payload, indent=2))

    try:
        raw = LLMService.generate(prompt, provider=provider, route_mode="experiential")
    except Exception as exc:
        print(_fail(f"AI analysis failed: {exc}"))
        return []

    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(_fail(f"Could not parse AI response: {exc}"))
        print(f"{C.DIM}Raw response (first 500 chars):{C.RESET}")
        print(raw[:500])
        return []


def _read_snippet(path: Path, line: int, context: int = 5) -> str:
    """Return the lines around `line` in `path`."""
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(0, line - context - 1)
        end = min(len(lines), line + context)
        return "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
    except Exception:
        return ""


# ===========================================================================
# Patch application
# ===========================================================================


def apply_patch(patch: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Apply a patch if the search string is unique in the file."""
    if not patch:
        return False, "no patch"

    file = patch.get("file")
    search = patch.get("search", "")
    replace = patch.get("replace", "")

    if not file or not search:
        return False, "incomplete patch"

    path = ROOT / file
    if not path.exists():
        return False, f"file not found: {file}"

    try:
        if not path.resolve().is_relative_to(ROOT):
            return False, f"file outside project: {file}"
    except ValueError:
        return False, f"path resolution failed: {file}"

    if any(p in path.parts for p in (".venv", "__pycache__", ".git")):
        return False, f"refusing to patch {file}"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"read error: {exc}"

    count = content.count(search)
    if count == 0:
        return False, "search string not found"
    if count > 1:
        return False, f"search string ambiguous ({count} matches)"

    if dry_run:
        return True, "would apply"

    new_content = content.replace(search, replace, 1)
    path.write_text(new_content, encoding="utf-8")
    return True, "applied"


# ===========================================================================
# Report rendering
# ===========================================================================


def render_findings(findings: list[Finding]) -> None:
    """Pretty-print findings grouped by checker."""
    by_checker: dict[str, list[Finding]] = {}
    for f in findings:
        by_checker.setdefault(f.checker, []).append(f)

    for checker in sorted(by_checker):
        print(f"\n{_h(f'=== {checker.upper()} ===')}")
        for f in by_checker[checker]:
            loc = f"{f.file}:{f.line}" if f.line else (f.file or "—")
            sev_color = {"error": C.RED, "warning": C.YELLOW, "info": C.DIM}.get(f.severity, "")
            print(f"  {sev_color}[{f.rule}]{C.RESET} {loc}")
            print(f"    {f.message[:120]}")


def render_ai_results(ai_results: list[dict], findings: list[Finding]) -> None:
    """Pretty-print AI verdicts."""
    if not ai_results:
        print(_warn("\nNo AI analysis available"))
        return

    print(f"\n{_h('=== AI ANALYSIS ===')}")

    counts = {"auto_fix": 0, "needs_review": 0, "false_positive": 0}
    for entry in ai_results:
        idx = entry.get("finding_index")
        if idx is None or idx >= len(findings):
            continue
        f = findings[idx]
        verdict = entry.get("verdict", "needs_review")
        confidence = entry.get("confidence", 0.0)
        explanation = entry.get("explanation", "")
        counts[verdict] = counts.get(verdict, 0) + 1

        icon = {
            "auto_fix": f"{C.GREEN}[FIX]{C.RESET}",
            "needs_review": f"{C.YELLOW}[REVIEW]{C.RESET}",
            "false_positive": f"{C.DIM}[FALSE POSITIVE]{C.RESET}",
        }.get(verdict, "[?]")

        loc = f"{f.file}:{f.line}" if f.line else (f.file or "—")
        print(f"\n{icon} {C.BOLD}{f.rule}{C.RESET} at {loc}")
        print(f"  {C.DIM}{f.message[:150]}{C.RESET}")
        print(f"  {explanation}")
        print(f"  {C.DIM}confidence: {confidence:.0%}{C.RESET}")

    print(f"\n{_h('Summary:')}")
    print(f"  auto-fixable:   {counts.get('auto_fix', 0)}")
    print(f"  needs review:   {counts.get('needs_review', 0)}")
    print(f"  false positive: {counts.get('false_positive', 0)}")


# ===========================================================================
# Main
# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-powered code doctor")
    parser.add_argument("--fix", action="store_true", help="Apply safe patches")
    parser.add_argument(
        "--all", action="store_true", help="Apply all patches (even low confidence)"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--provider", default="experiential", help="LLM provider")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI analysis")
    args = parser.parse_args()

    if not args.json:
        print(f"{C.BOLD}AI Doctor{C.RESET} {C.DIM}starting...{C.RESET}\n")

    findings: list[Finding] = []
    if not args.json:
        print(_h("Running checkers..."))

    for name, checker in [
        ("ruff", check_ruff),
        ("mypy", check_mypy),
        ("imports", check_imports),
        ("pytest", check_pytest),
        ("docstrings", check_docstrings),
    ]:
        try:
            results = checker()
            findings.extend(results)
            if not args.json:
                if results:
                    print(f"  {_ok(f'{name}: {len(results)} finding(s)')}")
                else:
                    print(f"  {_ok(f'{name}: clean')}")
        except Exception as exc:
            if not args.json:
                print(f"  {_fail(f'{name}: {exc}')}")

    # Filter out docstring noise if there are other issues
    real_findings = [f for f in findings if f.checker != "docstring"]
    if real_findings:
        findings = real_findings

    if args.json:
        output = {
            "findings": [f.__dict__ for f in findings],
            "count": len(findings),
        }
        print(json.dumps(output, indent=2, default=str))
        return 0

    print(f"\n{_h(f'Total findings: {len(findings)}')}")
    if findings:
        render_findings(findings)

    # AI analysis
    if findings and not args.no_ai:
        print(f"\n{_h('Asking AI to analyze...')}")
        ai_results = ai_analyze(findings, provider=args.provider)
        render_ai_results(ai_results, findings)

        # Apply patches if requested
        if args.fix and ai_results:
            min_confidence = 0.0 if args.all else 0.9
            print(f"\n{_h('Applying patches...')}")
            applied = 0
            for entry in ai_results:
                if entry.get("verdict") != "auto_fix":
                    continue
                confidence = entry.get("confidence", 0.0)
                if confidence < min_confidence:
                    pct = f"{confidence:.0%}"
                    print(f"  {_warn(f'skipped (confidence {pct})')}")
                    continue
                patch = entry.get("patch")
                if not patch:
                    continue
                ok, msg = apply_patch(patch)
                patch_file = patch.get("file", "?")
                if ok:
                    applied += 1
                    print(f"  {_ok(f'{patch_file}: {msg}')}")
                else:
                    print(f"  {_fail(f'{patch_file}: {msg}')}")

            print(f"\n{_ok(f'Applied {applied} patch(es)')}")
            if applied:
                print(
                    f"\n{C.DIM}Re-run checkers to verify: " f"python scripts/ai_doctor.py{C.RESET}"
                )

    print(f"\n{C.DIM}Done. Run with --fix to apply safe patches.{C.RESET}")
    return 0 if not any(f.severity == "error" for f in findings) else 1


if __name__ == "__main__":
    sys.exit(main())
