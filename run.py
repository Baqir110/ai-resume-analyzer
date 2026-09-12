import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    # Determine the project root (where this run.py lives)
    project_root = Path(__file__).resolve().parent

    # Build an environment with PYTHONPATH pointing to the project root
    # so that `app.*` imports resolve in both the backend and dashboard.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Start the FastAPI backend via uvicorn as a subprocess
    print("🚀 Starting FastAPI backend on port 8000...")
    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        env=env,
        cwd=project_root,
    )

    # Give the backend a brief moment to spin up
    time.sleep(2)

    # Start the Streamlit dashboard (new modular entry point)
    print("🎯 Starting Streamlit dashboard...")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app/dashboard/main.py"],
        env=env,
        cwd=project_root,
    )

    try:
        # Keep the script running while both processes are active
        backend_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
        backend_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()
