#!/usr/bin/env python
"""Safe LinkedIn discovery smoke test.

This historical script used to force terminal tracker rows back to
``DISCOVERED`` and could click Submit.  It is now a read-only discovery
diagnostic; use ``run_job_agent.py --prepare`` for an explicitly reviewed
workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default="DevOps Engineer")
    parser.parse_args()
    root = Path(__file__).resolve().parent
    print("Read-only smoke test: delegating to the hardened discovery agent.")
    return subprocess.call(
        [sys.executable, str(root / "run_job_agent.py"), "--discover"],
        cwd=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
