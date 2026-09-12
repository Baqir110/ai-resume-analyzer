"""Helper utilities for dashboard."""

import os
import time

import requests
import streamlit as st

# ============================================================
# Base config
# ============================================================


def get_api_base() -> str:
    """Get FastAPI base URL from session or environment."""
    return (
        st.session_state.get("api_base", "").strip()
        or os.getenv("FASTAPI_API_BASE", "http://localhost:8000")
    ).rstrip("/")


def format_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    try:
        n = int(n or 0)
    except (ValueError, TypeError):
        return "0"

    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def build_file_payload() -> dict:
    """Build file payload from session state."""
    uploaded = st.session_state.get("uploaded_file_data")
    if not uploaded:
        return {}
    filename, file_bytes, mime_type = uploaded
    return {"resume_file": (filename, file_bytes, mime_type)}


def error_detail(response) -> str:
    """Extract a human-readable error message from a response."""
    if response is None:
        return "Could not connect to the backend service."
    try:
        data = response.json()
        if isinstance(data, dict) and data.get("detail"):
            return str(data["detail"])
        return response.text
    except Exception:
        return response.text or f"HTTP {response.status_code}"


# ============================================================
# Enhanced API request with status code + timing + retry
# ============================================================


def make_api_request_verbose(
    endpoint: str,
    data: dict | None = None,
    files: dict | None = None,
    method: str = "POST",
    timeout: int = 120,
    retry_on_busy: bool = True,
) -> tuple[requests.Response | None, dict]:
    """Make an API request and return (response, meta).

    meta keys:
        status_code: int | None
        elapsed: float (seconds)
        error: str | None (human-readable error message)
        retried: bool
    """
    meta: dict = {
        "status_code": None,
        "elapsed": 0.0,
        "error": None,
        "retried": False,
    }

    t0 = time.perf_counter()

    def _do_request() -> requests.Response:
        if method == "POST":
            return requests.post(endpoint, data=data, files=files, timeout=timeout)
        if method == "GET":
            return requests.get(endpoint, params=data, timeout=timeout)
        if method == "DELETE":
            return requests.delete(endpoint, timeout=timeout)
        return requests.request(method, endpoint, data=data, timeout=timeout)

    try:
        response = _do_request()
    except requests.exceptions.Timeout:
        meta["elapsed"] = time.perf_counter() - t0
        meta["error"] = f"Request timed out after {timeout}s"
        return None, meta
    except requests.exceptions.ConnectionError:
        meta["elapsed"] = time.perf_counter() - t0
        meta["error"] = f"Could not connect to backend at {endpoint}"
        return None, meta
    except requests.exceptions.RequestException as exc:
        meta["elapsed"] = time.perf_counter() - t0
        meta["error"] = f"Request failed: {exc}"
        return None, meta

    meta["status_code"] = response.status_code
    meta["elapsed"] = time.perf_counter() - t0

    # Retry on 429 / 503
    if retry_on_busy and response.status_code in (429, 503):
        wait_time = 10 if response.status_code == 429 else 5
        meta["retried"] = True
        time.sleep(wait_time)
        try:
            response = _do_request()
            meta["status_code"] = response.status_code
            meta["elapsed"] = time.perf_counter() - t0
        except requests.exceptions.RequestException as exc:
            meta["error"] = f"Retry failed: {exc}"

    if response.status_code != 200 and not meta["error"]:
        meta["error"] = error_detail(response)

    return response, meta


def make_api_request(
    endpoint: str,
    data: dict | None = None,
    files: dict | None = None,
    method: str = "POST",
    timeout: int = 120,
) -> requests.Response | None:
    """Simple wrapper — just the response, no meta."""
    response, _ = make_api_request_verbose(
        endpoint, data=data, files=files, method=method, timeout=timeout
    )
    return response


# ============================================================
# Cached fetchers
# ============================================================


@st.cache_data(ttl=15)
def fetch_quota_status(api_base: str) -> dict | None:
    """Fetch quota status from backend."""
    try:
        url = f"{api_base}/api/v1/resume/quota-status"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("providers") or {}
    except Exception:
        pass
    return None


@st.cache_data(ttl=15)
def fetch_usage_summary(api_base: str, period: str = "all") -> dict | None:
    """Fetch token usage summary."""
    try:
        url = f"{api_base}/api/v1/resume/usage-summary"
        r = requests.get(url, params={"period": period}, timeout=10)
        if r.status_code == 200:
            return r.json().get("summary")
    except Exception:
        pass
    return None


@st.cache_data(ttl=5)
def fetch_processing_log(api_base: str, limit: int = 50) -> list[dict]:
    """Fetch backend processing log events."""
    try:
        url = f"{api_base}/api/v1/resume/processing-log"
        r = requests.get(url, params={"limit": limit}, timeout=10)
        if r.status_code == 200:
            return r.json().get("logs", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=300)
def fetch_career_options(api_base: str) -> dict:
    """Fetch cover letter templates and interview families."""
    try:
        url = f"{api_base}/api/v1/resume/career-options"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"cover_letter_templates": [], "interview_families": []}


def reset_session_state() -> None:
    """Reset key session state variables."""
    st.session_state["last_analysis"] = None
    st.session_state["uploaded_file_data"] = None
    st.session_state["job_desc"] = ""
