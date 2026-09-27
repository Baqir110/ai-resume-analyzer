"""Helper utilities for dashboard."""

import os
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# ============================================================
# Base config
# ============================================================


def get_api_base() -> str:
    """Get FastAPI base URL from session or environment."""
    return (
        st.session_state.get("api_base", "").strip()
        or os.getenv("FASTAPI_API_BASE", "http://127.0.0.1:8000")
    ).rstrip("/")


def get_api_headers() -> dict[str, str]:
    """Return headers required by protected backend operations."""

    api_key = os.getenv("API_KEY", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


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
    json: dict | None = None,
) -> tuple[requests.Response | None, dict]:
    """Make an API request and return (response, meta).

    Use json for JSON bodies; data and files retain form/multipart behavior.
    Meta contains status_code, elapsed (seconds), error, and retried.
    """
    meta: dict = {
        "status_code": None,
        "elapsed": 0.0,
        "error": None,
        "retried": False,
    }

    t0 = time.perf_counter()
    method = method.upper()

    def _do_request() -> requests.Response:
        headers = get_api_headers()
        if method == "POST":
            return requests.post(
                endpoint,
                data=data,
                files=files,
                json=json,
                headers=headers,
                timeout=timeout,
            )
        if method == "GET":
            return requests.get(endpoint, params=data, headers=headers, timeout=timeout)
        if method == "DELETE":
            return requests.delete(endpoint, headers=headers, timeout=timeout)
        return requests.request(
            method,
            endpoint,
            data=data,
            json=json,
            headers=headers,
            timeout=timeout,
        )

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

    if retry_on_busy and response.status_code in (429, 503):
        wait_time = 10 if response.status_code == 429 else 5
        meta["retried"] = True
        time.sleep(wait_time)
        try:
            response = _do_request()
            meta["status_code"] = response.status_code
        except requests.exceptions.RequestException as exc:
            meta["error"] = f"Retry failed: {exc}"
        finally:
            meta["elapsed"] = time.perf_counter() - t0

    if not 200 <= response.status_code < 300 and not meta["error"]:
        meta["error"] = error_detail(response)

    return response, meta


def make_api_request(
    endpoint: str,
    data: dict | None = None,
    files: dict | None = None,
    method: str = "POST",
    timeout: int = 120,
    json: dict | None = None,
) -> requests.Response | None:
    """Simple wrapper — just the response, no meta."""
    response, _ = make_api_request_verbose(
        endpoint, data=data, files=files, method=method, timeout=timeout, json=json
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
        r = requests.get(url, headers=get_api_headers(), timeout=10)
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
        r = requests.get(url, params={"period": period}, headers=get_api_headers(), timeout=10)
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
        r = requests.get(url, params={"limit": limit}, headers=get_api_headers(), timeout=10)
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
        r = requests.get(url, headers=get_api_headers(), timeout=10)
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


# ---------------------------------------------------------------------------
# Workflow status
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def fetch_backend_status(api_base: str) -> dict | None:
    """
    Routing, provider and Ollama diagnostics, as the backend reports them.

    Returns the raw payload. The caller decides what to assert: this function
    has no opinion about whether a provider "works", because it never called one.
    """
    return _get_json(f"{api_base}/api/v1/resume/backend-status", timeout=10)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_ollama_health(api_base: str) -> dict | None:
    """Whether the local Ollama server is reachable and the model installed."""
    return _get_json(f"{api_base}/api/v1/resume/health", timeout=10)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_discovery(api_base: str, provider: str = "") -> dict | None:
    """
    The models a provider currently offers.

    ``provider=""`` asks for every provider. Discovery is optional by design, so
    a provider that exposes no listing returns a status explaining that rather
    than an error, and the caller shows the configured model instead.
    """
    url = f"{api_base}/api/v1/resume/model-discovery"
    if provider:
        url += f"?provider={provider}"
    return _get_json(url, timeout=30)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_catalog(api_base: str) -> dict | None:
    """What this deployment is configured to send, per provider."""
    return _get_json(f"{api_base}/api/v1/resume/model-catalog", timeout=15)


@st.cache_data(ttl=5, show_spinner=False)
def fetch_pipeline_metrics(api_base: str, limit: int = 40) -> dict | None:
    """
    Recent generation metrics, read from the pipeline event log.

    Provider, model, per-request LLM duration, call count, retries and PDF
    compilation time. All recorded by the backend during the request; nothing
    here estimates.
    """
    return _get_json(
        f"{api_base}/api/v1/resume/pipeline-metrics?limit={limit}",
        timeout=10,
    )


def validate_pdf(api_base: str, pdf_bytes: bytes, filename: str = "cv.pdf"):
    """
    Ask the backend to check a generated PDF.

    The validation runs server-side on purpose: the same module that produced
    the document performs the check, so the UI and the generation path cannot
    disagree about what counts as valid.

    Returns ``(response, meta)`` in the same shape as
    :func:`make_api_request_verbose`, so the caller can use the usual
    success/error handling.
    """
    return make_api_request_verbose(
        f"{api_base}/api/v1/resume/validate-pdf",
        files={"pdf_file": (filename, pdf_bytes, "application/pdf")},
        timeout=60,
    )


def _get_json(url: str, *, timeout: int) -> dict | None:
    """
    GET a JSON document, returning None on any failure.

    Deliberately quiet. The caller renders whatever it got, including nothing,
    and an unreadable status endpoint should look like an absent one rather than
    an error the user has to interpret.
    """
    try:
        response = requests.get(url, headers=get_api_headers(), timeout=timeout)
    except Exception:
        return None

    if not (200 <= response.status_code < 300):
        return None

    try:
        payload = response.json()
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def clear_cached_status(api_base: str) -> None:
    """
    Drop the cached status readers.

    Called after a generation so the overview reflects what just happened rather
    than what was true before it. Only the status caches are cleared: clearing
    all of Streamlit's data cache would also drop the quota and usage panels,
    which is not what the button says it does.
    """
    for fn in (
        fetch_backend_status,
        fetch_ollama_health,
        fetch_pipeline_metrics,
        fetch_model_discovery,
        fetch_model_catalog,
    ):
        try:
            fn.clear()
        except Exception:
            pass
