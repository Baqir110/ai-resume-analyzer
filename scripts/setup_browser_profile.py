"""Launch an isolated local browser profile for manual ATS setup.

This helper never attaches to the user's normal browser profile and never
opens an unauthenticated CDP listener on a non-loopback interface.  The
automation worker keeps persistent profiles disabled unless its own explicit
opt-in environment variables are set.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "data" / "browser_profiles" / "manual"
EDGE_BINARIES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def main() -> int:
    if os.getenv("ALLOW_BROWSER_PROFILE_SETUP", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print("Browser profile setup is disabled. Set ALLOW_BROWSER_PROFILE_SETUP=true explicitly.")
        return 1
    edge = next((path for path in EDGE_BINARIES if path.exists()), None)
    if edge is None:
        print("Microsoft Edge was not found.")
        return 1
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={PROFILE_DIR}",
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=9222",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Launched an isolated profile at {PROFILE_DIR}")
    print("The CDP endpoint is loopback-only; do not expose port 9222.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
