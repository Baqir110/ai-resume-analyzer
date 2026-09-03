import subprocess
import sys
import time


def main():
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
        ]
    )

    # Give the backend a brief moment to spin up
    time.sleep(2)

    # Start the Streamlit dashboard
    print("🎯 Starting Streamlit dashboard...")
    streamlit_process = subprocess.Popen(["streamlit", "run", "app/dashboard.py"])

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
