#!/usr/bin/env python
"""Deprecated compatibility entry point.

The old implementation launched an independent continuous LinkedIn/Indeed
applier with duplicated queue, retry, and submission logic.  That path is
intentionally retired: it bypassed the hardened central state machine and
could submit without the normal package/authentication safeguards.  Use the
reviewed agent entry point instead:

    python run_job_agent.py --discover
    python run_job_agent.py --prepare
    python run_job_agent.py --apply
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true", help="run safe discovery")
    parser.add_argument(
        "--prepare", action="store_true", help="prepare packages without submitting"
    )
    args = parser.parse_args()
    flag = "--prepare" if args.prepare else "--discover"
    root = Path(__file__).resolve().parent
    return subprocess.call([sys.executable, str(root / "run_job_agent.py"), flag], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
