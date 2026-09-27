"""Launch the backend + dashboard for local development."""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


def _wait_for_backend(process: subprocess.Popen, timeout: float = 30.0) -> bool:
    """Wait for the authenticated local health endpoint before starting UI."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        request = urllib.request.Request("http://127.0.0.1:8000/api/v1/resume/health")
        api_key = os.getenv("API_KEY", "").strip()
        if api_key:
            request.add_header("X-API-Key", api_key)
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                if 200 <= response.status < 300:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def main():
    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    load_dotenv(project_root / ".env", override=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    app_dir = project_root / "app"
    dashboard_dir = app_dir / "dashboard"

    print("🚀 Starting FastAPI backend on port 8000 (auto-reload)...")
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
            "--reload",
            "--reload-dir",
            str(app_dir),
        ],
        env=env,
        cwd=project_root,
    )

    if not _wait_for_backend(backend_process):
        backend_process.terminate()
        print("Backend did not become healthy; startup aborted.")
        return 1

    print("🎯 Starting Streamlit dashboard on port 8501 (auto-reload)...")
    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_dir / "main.py"),
            "--server.port",
            "8501",
            "--server.address",
            "127.0.0.1",
            "--server.runOnSave",
            "true",
            "--server.fileWatcherType",
            "auto",
            "--server.headless",
            "true",
        ],
        env=env,
        cwd=project_root,
    )

    try:
        while backend_process.poll() is None and streamlit_process.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
    finally:
        for process in (streamlit_process, backend_process):
            if process.poll() is None:
                process.terminate()
        for process in (streamlit_process, backend_process):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
