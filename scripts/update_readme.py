"""Auto-update README.md with live data from the app.

Regenerates sections between <!-- BEGIN:AUTO:* --> and <!-- END:AUTO:* -->
markers. Safe to run repeatedly — only touches the marked sections.

Run: python scripts/update_readme.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Ensure project root is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def _replace_section(text: str, section: str, new_content: str) -> str:
    """Replace content between <!-- BEGIN:AUTO:X --> and <!-- END:AUTO:X -->."""
    begin = f"<!-- BEGIN:AUTO:{section} -->"
    end = f"<!-- END:AUTO:{section} -->"
    pattern = re.compile(
        rf"({re.escape(begin)})(.*?)({re.escape(end)})",
        re.DOTALL,
    )
    replacement = f"{begin}\n\n{new_content.strip()}\n\n{end}"
    if not pattern.search(text):
        print(f"  [WARN] marker {section!r} not found - skipping")
        return text
    return pattern.sub(replacement, text, count=1)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_stats() -> str:
    """Project statistics block (deterministic — no timestamps)."""
    try:
        from app.main import app

        route_count = len(
            [r for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")]
        )
    except Exception:
        route_count = 0

    # Count Python files and LOC
    py_files = list((ROOT / "app").rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]
    loc = 0
    for f in py_files:
        try:
            loc += len(f.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass

    # Count tests
    test_files = list((ROOT / "tests").rglob("test_*.py"))
    test_count = 0
    for f in test_files:
        try:
            text = f.read_text(encoding="utf-8")
            test_count += len(re.findall(r"^\s*def test_", text, re.MULTILINE))
        except Exception:
            pass

    return f"""| Metric | Value |
|--------|-------|
| API endpoints | {route_count} |
| Python files | {len(py_files)} |
| Lines of code | {loc:,} |
| Tests | {test_count} |"""


def gen_api() -> str:
    """API endpoint table from the live FastAPI app."""
    try:
        from app.main import app
    except Exception as exc:
        return f"_(failed to load app: {exc})_"

    rows: list[tuple[str, str, str, str]] = []
    for r in app.routes:
        # FastAPI 0.116+ wraps include_router in _IncludedRouter
        if type(r).__name__ == "_IncludedRouter":
            for sub in getattr(r, "routes", []):
                if not hasattr(sub, "methods") or not hasattr(sub, "path"):
                    continue
                for method in sorted(sub.methods - {"HEAD", "OPTIONS"}):
                    tag = ", ".join(getattr(sub, "tags", []) or ["—"])
                    name = getattr(sub, "name", "")
                    summary = (getattr(sub, "summary", "") or name).strip()
                    rows.append((method, sub.path, tag, summary))
        elif hasattr(r, "path") and r.path.startswith("/api"):
            for method in sorted(getattr(r, "methods", set()) - {"HEAD", "OPTIONS"}):
                tag = ", ".join(getattr(r, "tags", []) or ["—"])
                name = getattr(r, "name", "")
                summary = (getattr(r, "summary", "") or name).strip()
                rows.append((method, r.path, tag, summary))

    if not rows:
        return "_(no API routes detected)_"

    # Group by tag
    by_tag: dict[str, list[tuple[str, str, str]]] = {}
    for method, path, tag, summary in rows:
        by_tag.setdefault(tag, []).append((method, path, summary))

    parts: list[str] = []
    parts.append(f"**Total endpoints: {len(rows)}**\n")
    for tag in sorted(by_tag):
        parts.append(f"### {tag}\n")
        parts.append("| Method | Path | Description |")
        parts.append("|--------|------|-------------|")
        for method, path, summary in sorted(by_tag[tag], key=lambda x: (x[1], x[0])):
            summary = summary.replace("|", "\\|")
            parts.append(f"| `{method}` | `{path}` | {summary} |")
        parts.append("")
    parts.append("")
    return "\n".join(parts)


def gen_tree() -> str:
    """Project structure tree (app/, tests/, scripts/)."""

    def build(path: Path, prefix: str = "") -> list[str]:
        entries = sorted(
            [e for e in path.iterdir() if e.name != "__pycache__" and not e.name.startswith(".")],
            key=lambda e: (not e.is_dir(), e.name),
        )
        lines: list[str] = []
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                lines.extend(build(entry, prefix + ("    " if is_last else "│   ")))
        return lines

    parts = ["```text", "ai-resume-analyzer/"]
    for d in ("app", "tests", "scripts"):
        p = ROOT / d
        if p.exists():
            parts.append(f"├── {d}/")
            for line in build(p, "│   "):
                parts.append(line)
    for f in sorted(ROOT.glob("*.py")):
        parts.append(f"├── {f.name}")
    parts.append("```")
    parts.append("")
    parts.append("")
    return "\n".join(parts)


def gen_tests() -> str:
    """Test summary table."""
    test_files = sorted((ROOT / "tests").rglob("test_*.py"))
    if not test_files:
        return "_(no test files found)_"

    rows = []
    for f in test_files:
        try:
            text = f.read_text(encoding="utf-8")
            count = len(re.findall(r"^\s*def test_", text, re.MULTILINE))
            rel = f.relative_to(ROOT).as_posix()
            rows.append(f"| `{rel}` | {count} |")
        except Exception:
            continue

    total = sum(int(r.split("|")[2].strip()) for r in rows)
    return (
        f"**Total tests: {total}** across {len(rows)} files.\n\n"
        "| Test file | Count |\n|-----------|-------|\n" + "\n".join(rows) + "\n\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not README.exists():
        print(f"README not found at {README}")
        return 1

    text = README.read_text(encoding="utf-8")
    original = text

    sections = [
        ("STATS", gen_stats),
        ("API", gen_api),
        ("TREE", gen_tree),
        ("TESTS", gen_tests),
    ]

    print("Updating README sections:")
    for name, fn in sections:
        try:
            content = fn()
            text = _replace_section(text, name, content)
            print(f"  [OK] {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")

    if text == original:
        print("\nNo changes needed.")
        return 0

    README.write_text(text, encoding="utf-8")
    print(f"\n[OK] Updated {README.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
