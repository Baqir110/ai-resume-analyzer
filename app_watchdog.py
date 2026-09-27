#!/usr/bin/env python
"""Read-only runtime log monitor.

Older versions of this helper rewrote source and applicant-profile YAML based
on log text.  That is unsafe: logs and model output are untrusted and a
watchdog must not silently change code or personal data.  This monitor only
reports bounded, redacted diagnostics.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("watchdog")
ROOT = Path(__file__).resolve().parent
LOG_FILES = [ROOT / "data" / n for n in ("agent_continuous.log", "linkedin_indeed.log")]
PATTERNS = {
    "maximum steps reached": "browser step limit",
    "captcha": "CAPTCHA/human verification required",
    "verification code": "email/phone verification required",
    "submission failed": "submission failed",
    "database is locked": "database contention",
}


def _tail(path: Path, count: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return list(deque(stream, maxlen=count))
    except OSError:
        return []


def _redact(line: str) -> str:
    line = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", line)
    line = re.sub(r"(?<!\w)\+?\d[\d .()/-]{7,}\d(?!\w)", "[PHONE]", line)
    for marker in ("password", "token", "secret", "api_key"):
        line = re.sub(rf"(?i)({marker}\s*[=:]\s*)\S+", r"\1[REDACTED]", line)
    return line[:300]


def scan_once() -> None:
    for path in LOG_FILES:
        for raw in _tail(path):
            lowered = raw.casefold()
            for needle, message in PATTERNS.items():
                if needle in lowered:
                    log.warning("%s: %s", path.name, message)
                    log.debug("context=%s", _redact(raw))
                    break


def main() -> None:
    print("Read-only watchdog enabled; no source or profile files will be modified.")
    try:
        while True:
            scan_once()
            time.sleep(5)
    except KeyboardInterrupt:
        print("Watchdog stopped.")


if __name__ == "__main__":
    main()
