"""
Quota tracking for LLM providers.

Gemini (and most providers) do not expose a "remaining quota" endpoint for
API-key based access. Google confirmed this in their developer forum.

So we compute usage from two sources:
  1. Local call log (`data/llm_processing.jsonl`) — always available
  2. User-declared quota limits (from .env) — the denominator

For GCP-attached projects, an optional Cloud Monitoring pull can supplement
the local data. See `fetch_gcp_quota()`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider quota defaults (RPM = requests/min, RPD = requests/day, TPM = tokens/min)
# ---------------------------------------------------------------------------

DEFAULT_QUOTAS: dict[str, dict[str, int]] = {
    "gemini": {"rpm": 15, "rpd": 1500, "tpm": 1_000_000},
    "openai": {"rpm": 500, "rpd": 10_000, "tpm": 200_000},
    "groq": {"rpm": 30, "rpd": 14_400, "tpm": 6_000},
    "deepseek": {"rpm": 60, "rpd": 10_000, "tpm": 100_000},
    "claude": {"rpm": 50, "rpd": 1000, "tpm": 40_000},
    "openrouter": {"rpm": 60, "rpd": 1000, "tpm": 100_000},
    "ollama": {"rpm": 0, "rpd": 0, "tpm": 0},
    "experiential": {"rpm": 100, "rpd": 5000, "tpm": 500_000},
}


def _load_quota_from_env(provider: str) -> dict[str, int]:
    """Read `<PROVIDER>_QUOTA_RPM/RPD/TPM` from env, falling back to defaults."""
    provider_key = provider.upper()
    default = DEFAULT_QUOTAS.get(provider.lower(), {"rpm": 0, "rpd": 0, "tpm": 0})

    def _get(key: str, fallback: int) -> int:
        try:
            return int(os.getenv(f"{provider_key}_QUOTA_{key}", str(fallback)) or fallback)
        except (ValueError, TypeError):
            return fallback

    return {
        "rpm": _get("RPM", default["rpm"]),
        "rpd": _get("RPD", default["rpd"]),
        "tpm": _get("TPM", default["tpm"]),
    }


# ---------------------------------------------------------------------------
# In-memory 429 tracker
# ---------------------------------------------------------------------------

_rate_limit_events: list[dict[str, Any]] = []
_rate_lock = threading.Lock()
_MAX_EVENTS = 500


def record_rate_limit_event(
    provider: str,
    model: str,
    error_message: str = "",
) -> None:
    """Called whenever a provider returns a 429 / RESOURCE_EXHAUSTED."""
    with _rate_lock:
        _rate_limit_events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": (provider or "").lower(),
                "model": model or "",
                "error": (error_message or "")[:500],
            }
        )
        while len(_rate_limit_events) > _MAX_EVENTS:
            _rate_limit_events.pop(0)


def recent_rate_limit_events(
    provider: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with _rate_lock:
        events = list(_rate_limit_events)

    if provider:
        events = [e for e in events if e.get("provider") == provider.lower()]

    return events[-limit:]


# ---------------------------------------------------------------------------
# Usage aggregation from the local log
# ---------------------------------------------------------------------------

_WINDOW_RPM = timedelta(minutes=1)
_WINDOW_RPD = timedelta(days=1)


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def compute_provider_usage(
    provider: str,
    log_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return usage stats for one provider over the last minute and last 24h."""
    provider = (provider or "").lower()
    now = now or datetime.now(timezone.utc)

    result: dict[str, Any] = {
        "provider": provider,
        "requests_last_minute": 0,
        "tokens_last_minute": 0,
        "requests_last_24h": 0,
        "tokens_last_24h": 0,
        "last_call_at": None,
        "rate_limit_hits_24h": 0,
    }

    if not log_path.exists():
        return result

    cutoff_minute = now - _WINDOW_RPM
    cutoff_day = now - _WINDOW_RPD
    last_ts: datetime | None = None

    try:
        with log_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                ev_provider = str(ev.get("provider", "")).lower()
                if ev_provider != provider:
                    continue

                ts = _parse_iso(ev.get("timestamp", ""))
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                event_type = ev.get("event")

                if event_type == "request_completed":
                    tokens = int(ev.get("total_tokens", 0) or 0)

                    if ts >= cutoff_minute:
                        result["requests_last_minute"] += 1
                        result["tokens_last_minute"] += tokens

                    if ts >= cutoff_day:
                        result["requests_last_24h"] += 1
                        result["tokens_last_24h"] += tokens

                    if last_ts is None or ts > last_ts:
                        last_ts = ts

                elif event_type == "provider_failed":
                    err = str(ev.get("error", "")).lower()
                    if ts >= cutoff_day and ("429" in err or "resource_exhausted" in err):
                        result["rate_limit_hits_24h"] += 1

    except OSError:
        pass

    if last_ts:
        result["last_call_at"] = last_ts.isoformat()

    return result


def build_quota_status(
    provider: str,
    log_path: Path,
    model: str | None = None,
) -> dict[str, Any]:
    """Compose a UI-ready quota snapshot for one provider."""
    provider_lower = (provider or "").lower()
    limits = _load_quota_from_env(provider_lower)
    usage = compute_provider_usage(provider_lower, log_path)

    now = datetime.now(timezone.utc)

    def _pct(used: int, limit: int) -> float:
        if limit <= 0:
            return 0.0
        return round(min(100.0, (used / limit) * 100.0), 2)

    last_minute_reset = now + _WINDOW_RPM

    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    minute_window = {
        "used_requests": usage["requests_last_minute"],
        "limit_requests": limits["rpm"],
        "percent_used_requests": _pct(usage["requests_last_minute"], limits["rpm"]),
        "remaining_requests": max(0, limits["rpm"] - usage["requests_last_minute"]),
        "used_tokens": usage["tokens_last_minute"],
        "limit_tokens": limits["tpm"],
        "percent_used_tokens": _pct(usage["tokens_last_minute"], limits["tpm"]),
        "remaining_tokens": max(0, limits["tpm"] - usage["tokens_last_minute"]),
        "resets_at": last_minute_reset.isoformat(),
        "resets_in_seconds": int((last_minute_reset - now).total_seconds()),
    }

    day_window = {
        "used_requests": usage["requests_last_24h"],
        "limit_requests": limits["rpd"],
        "percent_used_requests": _pct(usage["requests_last_24h"], limits["rpd"]),
        "remaining_requests": max(0, limits["rpd"] - usage["requests_last_24h"]),
        "used_tokens": usage["tokens_last_24h"],
        "limit_tokens": limits["tpm"] * 1440 if limits["tpm"] else 0,
        "percent_used_tokens": _pct(
            usage["tokens_last_24h"],
            limits["tpm"] * 1440 if limits["tpm"] else 0,
        ),
        "remaining_tokens": max(
            0,
            (limits["tpm"] * 1440 if limits["tpm"] else 0) - usage["tokens_last_24h"],
        ),
        "resets_at": tomorrow.isoformat(),
        "resets_in_seconds": int((tomorrow - now).total_seconds()),
    }

    return {
        "provider": provider_lower,
        "model": model or "",
        "limits": limits,
        "usage": usage,
        "windows": {
            "minute": minute_window,
            "day": day_window,
        },
        "rate_limit_hits_24h": usage["rate_limit_hits_24h"],
        "quota_source": "local_log+declared_limits",
        "note": (
            "Gemini does not expose remaining quota via API. Numbers reflect "
            "calls made from this app only, compared against your declared "
            "limits in .env."
        ),
    }


def build_all_provider_quota_status(
    log_path: Path,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Return a dict of provider -> quota status."""
    if providers is None:
        providers = list(DEFAULT_QUOTAS.keys())

    return {p: build_quota_status(p, log_path) for p in providers}


# ---------------------------------------------------------------------------
# Optional: Google Cloud Monitoring
# ---------------------------------------------------------------------------


def fetch_gcp_quota() -> dict[str, Any] | None:
    """
    Optional: pull real quota data from Google Cloud Monitoring.

    Requires:
        GCP_PROJECT_ID=<your-project>
        GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
        GCP_MONITORING_ENABLED=true
    """
    if os.getenv("GCP_MONITORING_ENABLED", "").lower() not in ("true", "1", "yes"):
        return None

    project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    if not project_id:
        return None

    try:
        from google.cloud import monitoring_v3
    except ImportError:
        logger.warning(
            "google-cloud-monitoring not installed. " "Run: pip install google-cloud-monitoring"
        )
        return None

    try:
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"

        now = time.time()
        interval = monitoring_v3.TimeInterval(
            {
                "end_time": {"seconds": int(now)},
                "start_time": {"seconds": int(now) - 3600},
            }
        )

        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": (
                    'metric.type = "serviceruntime.googleapis.com/quota/allocation/usage" '
                    'AND resource.labels.service = "generativelanguage.googleapis.com"'
                ),
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )

        series = []
        for ts in results:
            points = [
                {
                    "value": p.value.int64_value or p.value.double_value,
                    "timestamp": p.interval.end_time.isoformat(),
                }
                for p in ts.points
            ]
            series.append(
                {
                    "metric_labels": dict(ts.metric.labels),
                    "resource_labels": dict(ts.resource.labels),
                    "points": points,
                }
            )

        return {"source": "gcp_monitoring", "series": series}

    except Exception as exc:
        logger.warning("GCP Monitoring fetch failed: %s", exc)
        return None
