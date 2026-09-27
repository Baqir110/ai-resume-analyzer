"""Security primitives shared by API and network-facing services.

The application accepts URLs and filesystem paths from several layers (HTTP
clients, discovery sources, and the browser worker).  Keeping validation in
one small, dependency-free module makes it harder for a new caller to
accidentally bypass the same checks.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.core.config import settings

_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
}
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home")


class UnsafeInputError(ValueError):
    """Raised when a URL or path is not safe for the requested operation."""


def _reject(message: str) -> None:
    raise UnsafeInputError(message)


def _parse_ip(value: str) -> ipaddress._BaseAddress | None:
    candidate = str(value or "").strip().strip("[]")
    if not candidate:
        return None
    if "%" in candidate:
        candidate = candidate.split("%", 1)[0]
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        # urllib/urlsplit and the OS resolver accept alternate IPv4
        # spellings such as 2130706433 and 0x7f000001.  Normalize those
        # forms before applying the private-address policy.
        if all(char in "0123456789abcdefABCDEFxXoO." for char in candidate):
            try:
                return ipaddress.ip_address(socket.inet_aton(candidate))
            except (OSError, ValueError):
                return None
    return None


def _ip_is_unsafe(value: str) -> bool:
    address = _parse_ip(value)
    if address is None:
        return False
    # ``is_global`` excludes loopback, link-local, private, multicast,
    # reserved, and unspecified ranges.  Treat IPv4-mapped IPv6 like its
    # embedded IPv4 address as well.
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return not address.is_global


def _hostname_is_unsafe(hostname: str) -> bool:
    host = hostname.rstrip(".").casefold()
    if not host or host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return True
    # Avoid alternate numeric representations of loopback/private IPv4.
    try:
        return _ip_is_unsafe(host)
    except ValueError:
        return False


def validate_public_http_url(
    url: str,
    *,
    resolve_dns: bool = True,
    allowed_hosts: set[str] | frozenset[str] | None = None,
    require_https: bool = True,
) -> str:
    """Validate an outbound HTTP(S) URL and return its normalized string.

    Only public web URLs are accepted.  URL credentials, non-web schemes,
    local hostnames, literal private addresses, and (when requested) hostnames
    resolving to private addresses are rejected.  Redirect targets must be
    passed through this function again by the caller.
    """

    if not isinstance(url, str) or not url.strip():
        _reject("URL must be a non-empty string")
    value = url.strip()
    if len(value) > 4096:
        _reject("URL is too long")
    if any(ord(char) < 32 for char in value):
        _reject("URL contains control characters")

    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise UnsafeInputError("Malformed URL") from exc

    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        _reject("Only http:// and https:// URLs are allowed")
    if require_https and scheme != "https":
        _reject("HTTPS is required for job and application URLs")
    if parsed.username is not None or parsed.password is not None:
        _reject("URLs containing credentials are not allowed")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise UnsafeInputError("Malformed URL host or port") from exc
    if not hostname:
        _reject("URL must include a hostname")
    if port is not None and port not in {80, 443}:
        _reject("Only the standard HTTP/HTTPS ports are allowed")

    host = hostname.rstrip(".").casefold()
    if _hostname_is_unsafe(host):
        raise UnsafeInputError("Private, loopback, or local network URLs are not allowed")

    if allowed_hosts:
        allowed = {str(item).rstrip(".").casefold() for item in allowed_hosts}
        if not any(host == item or host.endswith("." + item) for item in allowed):
            raise UnsafeInputError("URL host is not on the configured allowlist")

    if resolve_dns:
        try:
            addresses = socket.getaddrinfo(
                host,
                port or (443 if scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise UnsafeInputError("URL hostname could not be resolved") from exc
        for address in addresses:
            sockaddr = address[4]
            ip_value = sockaddr[0] if sockaddr else ""
            if _ip_is_unsafe(str(ip_value)):
                raise UnsafeInputError("URL resolves to a private or local network address")

    return value


def validate_job_url(url: str, *, resolve_dns: bool = True) -> str:
    """Validate a user-supplied job URL before fetching or browsing it."""

    return validate_public_http_url(
        url,
        resolve_dns=resolve_dns,
        allowed_hosts=set(getattr(settings, "JOB_ALLOWED_HOSTS", []) or []),
    )


def _configured_roots() -> list[Path]:
    raw_roots = getattr(settings, "ALLOWED_FILE_ROOTS", None) or ["data"]
    roots: list[Path] = []
    project_root = Path(__file__).resolve().parents[2]
    for raw_root in raw_roots:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = project_root / root
        roots.append(root.resolve())
    return roots


def validate_local_file(
    value: str | os.PathLike[str],
    *,
    allowed_extensions: set[str] | frozenset[str] | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve a local file and ensure it is inside an allowlisted root.

    Paths are resolved before the containment check, which prevents ``..``,
    symlink escapes, and alternate spellings from reaching arbitrary files.
    """

    if not isinstance(value, (str, os.PathLike)):
        raise UnsafeInputError("File path must be a string")
    raw = os.fspath(value)
    if not raw.strip() or len(raw) > 4096 or "\x00" in raw:
        raise UnsafeInputError("Invalid file path")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    try:
        resolved = candidate.resolve(strict=require_exists)
    except FileNotFoundError as exc:
        raise UnsafeInputError("File does not exist") from exc
    except OSError as exc:
        raise UnsafeInputError("File path could not be resolved") from exc

    roots = _configured_roots()
    project_root = Path(__file__).resolve().parents[2]
    try:
        home_root = Path.home().resolve()
    except (OSError, RuntimeError):
        home_root = project_root
    if any(root in {project_root, home_root, Path(root.anchor)} for root in roots):
        raise UnsafeInputError("Configured upload root is too broad")
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise UnsafeInputError("File is outside the configured upload roots")

    if require_exists and not resolved.is_file():
        raise UnsafeInputError("Path is not a regular file")

    if allowed_extensions:
        suffix = resolved.suffix.casefold()
        normalized = {("." + str(ext).lstrip(".")).casefold() for ext in allowed_extensions}
        if suffix not in normalized:
            raise UnsafeInputError(f"Unsupported file type: {suffix or '(none)'}")

    max_size = int(getattr(settings, "MAX_FILE_SIZE", 5 * 1024 * 1024))
    if require_exists and resolved.stat().st_size > max_size:
        raise UnsafeInputError("File exceeds the configured size limit")

    return resolved


def validate_pdf_file(value: str | os.PathLike[str], *, max_bytes: int | None = None) -> Path:
    """Validate a PDF artifact beyond its filename suffix."""
    path = validate_local_file(value, allowed_extensions={".pdf"})
    limit = int(max_bytes or getattr(settings, "MAX_PACKAGE_BYTES", 25 * 1024 * 1024))
    try:
        if path.stat().st_size > limit:
            raise UnsafeInputError("PDF exceeds the configured size limit")
        with path.open("rb") as stream:
            if b"%PDF-" not in stream.read(1024):
                raise UnsafeInputError("File is not a PDF document")
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted or len(reader.pages) < 1:
            raise UnsafeInputError("PDF is encrypted or has no pages")
    except UnsafeInputError:
        raise
    except Exception as exc:
        raise UnsafeInputError("PDF could not be parsed") from exc
    return path


def read_upload_limited(upload, *, max_bytes: int | None = None) -> bytes:
    """Read an UploadFile without allowing an unbounded allocation."""

    limit = int(max_bytes or getattr(settings, "MAX_FILE_SIZE", 5 * 1024 * 1024))
    declared = getattr(upload, "size", None)
    if declared is not None and int(declared) > limit:
        raise UnsafeInputError("Uploaded file exceeds the configured size limit")
    try:
        content = upload.file.read(limit + 1)
    except AttributeError:
        # Test doubles may expose an async UploadFile-like object.
        raise UnsafeInputError("Upload is not readable")
    if len(content) > limit:
        raise UnsafeInputError("Uploaded file exceeds the configured size limit")
    return content


API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    description="Required for API operations that read or change local data.",
    auto_error=False,
)


def require_api_key(x_api_key: str | None = Depends(API_KEY_HEADER)) -> None:
    """FastAPI dependency for endpoints that can change data or submit jobs.

    Authentication is mandatory even when the service is bound to localhost;
    an accidentally exposed development server must fail closed.  Operators
    can set ``API_KEY`` in the environment or secret manager.
    """

    configured = str(getattr(settings, "API_KEY", "") or "").strip()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured; refusing protected operation",
        )
    supplied = str(x_api_key or "")
    if not supplied or not hmac.compare_digest(supplied, configured):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


__all__ = [
    "UnsafeInputError",
    "validate_public_http_url",
    "validate_job_url",
    "validate_local_file",
    "validate_pdf_file",
    "read_upload_limited",
    "require_api_key",
    "API_KEY_HEADER",
]
