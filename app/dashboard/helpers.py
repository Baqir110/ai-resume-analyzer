"""Helper utilities for dashboard."""

import os

import requests
import streamlit as st


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


def make_api_request(
    endpoint: str,
    data: dict = None,
    files: dict = None,
    method: str = "POST",
    timeout: int = 120,
) -> requests.Response | None:
    """Make API request with error handling.

    Returns:
        Response object or None if failed
    """
    try:
        if method == "POST":
            return requests.post(
                endpoint,
                data=data,
                files=files,
                timeout=timeout,
            )
        elif method == "GET":
            return requests.get(
                endpoint,
                params=data,
                timeout=timeout,
            )
    except requests.exceptions.Timeout:
        st.error("Request timed out. Backend may be busy.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend service")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

    return None


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
