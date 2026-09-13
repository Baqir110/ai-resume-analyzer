"""Open Edge with remote debugging so you can log into ATS portals once.

Make sure Edge is closed, then run this. Leave the Edge window open
and start the dashboard in a second terminal.
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EDGE_BINARIES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
EDGE_EXE = next((p for p in EDGE_BINARIES if p.exists()), None)


def main() -> int:
    if EDGE_EXE is None:
        print("ERROR: Edge not found.")
        return 1

    # Kill existing Edge
    subprocess.run(
        ["taskkill", "/F", "/IM", "msedge.exe", "/T"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    # Launch with debugging
    print(f"Launching: {EDGE_EXE}")
    subprocess.Popen([str(EDGE_EXE), "--remote-debugging-port=9222"])

    # Wait for CDP to be ready
    for attempt in range(15):
        try:
            with urllib.request.urlopen("http://localhost:9222/json/version", timeout=1) as r:
                data = r.read().decode("utf-8", errors="replace")
            if "Edg/" in data:
                print(f"Ready. {data[:120]}")
                print()
                print("Log into your ATS portals (Workday, Greenhouse, Lever).")
                print("Leave Edge open. Start the dashboard in another terminal.")
                return 0
        except Exception:
            pass
        time.sleep(1)

    print("ERROR: Edge did not start with debugging enabled.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
