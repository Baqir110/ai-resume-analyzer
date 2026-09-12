"""Launch the backend + dashboard for local development."""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    print("🚀 Starting FastAPI backend on port 8000...")
    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "info",
        ],
        env=env,
        cwd=project_root,
    )

    time.sleep(2)

    print("🎯 Starting Streamlit dashboard...")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app/dashboard/main.py"],
        env=env,
        cwd=project_root,
    )

    try:
        backend_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
        backend_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()
