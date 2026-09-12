"""Run pytest, feed the failure to the LLM, get a patch suggestion."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FIXER_PROMPT = """You are a senior Python engineer. A test failed in this project.

Project layout: FastAPI backend in app/, tests in tests/.
Stack: Python 3.11, FastAPI, Pydantic, SQLAlchemy, Streamlit.

TEST FAILURE OUTPUT:
{pytest_output}

RELEVANT FILE(S):
{files}

Return a JSON object:
{{
  "diagnosis": "<one sentence: what is broken>",
  "root_cause": "<why it broke>",
  "patches": [
    {{
      "file": "app/...py",
      "search": "<exact string to find — must be unique in the file>",
      "replace": "<replacement string>"
    }}
  ],
  "confidence": <0.0 to 1.0>,
  "explanation": "<why this fix works>"
}}
"""


def run_tests() -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not integration", "-x", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode, output[-4000:]  # last 4k chars


def extract_files(output: str) -> dict[str, str]:
    """Find 'File "path.py", line N' refs and read those files."""
    paths = set()
    for match in re.finditer(r'File "([^"]+\.py)"', output):
        p = Path(match.group(1))
        if "site-packages" in str(p) or "__pycache__" in str(p):
            continue
        if p.exists():
            paths.add(p)

    files = {}
    for p in list(paths)[:3]:  # cap at 3 files
        try:
            files[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")[:4000]
        except Exception:
            pass
    return files


def get_fix(output: str, files: dict[str, str]) -> str:
    from app.services.llm.provider import LLMService

    files_block = "\n\n".join(f"--- {name} ---\n{content}" for name, content in files.items())
    prompt = FIXER_PROMPT.format(pytest_output=output, files=files_block)

    return LLMService.generate(
        prompt,
        provider="experiential",
        route_mode="experiential",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply the suggested patches")
    args = parser.parse_args()

    print("Running tests...")
    code, output = run_tests()
    if code == 0:
        print("All tests pass. Nothing to fix.")
        return 0

    print("Tests failed. Extracting context...")
    files = extract_files(output)
    print(f"Found {len(files)} relevant files.")

    print("Asking LLM for a fix...")
    import json

    raw = get_fix(output, files)

    # Strip fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]

    try:
        fix = json.loads(text.strip())
    except Exception as exc:
        print(f"Could not parse LLM response: {exc}")
        print(raw[:1000])
        return 1

    print(f"\n=== DIAGNOSIS ===\n{fix.get('diagnosis', '?')}")
    print(f"\n=== ROOT CAUSE ===\n{fix.get('root_cause', '?')}")
    print(f"\n=== CONFIDENCE ===\n{fix.get('confidence', 0.0):.0%}")

    patches = fix.get("patches", [])
    print(f"\n=== {len(patches)} PATCH(ES) ===")
    for i, patch in enumerate(patches, 1):
        print(f"\n[{i}] {patch.get('file')}")
        print(f"  SEARCH:  {patch.get('search', '')[:80]}...")
        print(f"  REPLACE: {patch.get('replace', '')[:80]}...")

    if not args.apply:
        print("\nDry run — re-run with --apply to write the patches.")
        return 0

    for patch in patches:
        p = ROOT / patch["file"]
        if not p.exists():
            print(f"  ❌ {p} not found")
            continue
        content = p.read_text(encoding="utf-8")
        search = patch["search"]
        if content.count(search) != 1:
            print(f"  ❌ {p.name}: search string not unique ({content.count(search)} matches)")
            continue
        content = content.replace(search, patch["replace"])
        p.write_text(content, encoding="utf-8")
        print(f"  ✅ patched {p.name}")

    print("\nRun tests again to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
