"""Quick LLM sanity check."""

import sys
from pathlib import Path

# Ensure project root is importable when this file is run directly
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.services.llm.provider import LLMService

prompt = 'Return ONLY valid JSON, no markdown: {"test": "ok", "score": 42}'

print("Calling LLM...")
raw = LLMService.generate(prompt, provider="experiential", route_mode="experiential")

print("\n=== RAW RESPONSE ===")
print(repr(raw[:800]))
print("\n=== LENGTH ===")
print(len(raw))
print("\n=== FIRST 200 CHARS ===")
print(raw[:200])
