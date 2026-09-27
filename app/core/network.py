"""Small outbound-network safety helpers.

URL validation alone is vulnerable to DNS rebinding because a normal HTTP
client resolves the hostname again after validation.  The transport in this
module resolves once, rejects any private answer, and connects to the pinned
address while retaining the original Host header and TLS SNI hostname.
"""

from __future__ import annotations

import ipaddress
import socket

import httpx

from app.core.security import UnsafeInputError


def resolve_public_addresses(host: str, port: int = 443) -> list[str]:
    """Return public DNS answers, rejecting the entire hostname if any is unsafe."""
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        try:
            answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise UnsafeInputError("Outbound hostname could not be resolved") from exc
        addresses = sorted({str(answer[4][0]) for answer in answers if answer[4]})
    else:
        addresses = [str(literal)]
    if not addresses:
        raise UnsafeInputError("Outbound hostname has no addresses")
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError as exc:
            raise UnsafeInputError("Outbound hostname returned an invalid address") from exc
        if not parsed.is_global:
            raise UnsafeInputError("Outbound hostname resolved to a private address")
    return addresses


class PinnedPublicTransport(httpx.AsyncBaseTransport):
    """Async HTTP transport that pins a validated public DNS answer."""

    def __init__(self, **transport_kwargs: object) -> None:
        self._transport = httpx.AsyncHTTPTransport(
            trust_env=False,
            **transport_kwargs,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = request.url
        host = original_url.host
        port = original_url.port or (443 if original_url.scheme == "https" else 80)
        addresses = resolve_public_addresses(host, port)
        pinned_url = original_url.copy_with(host=addresses[0])
        request.headers["Host"] = original_url.netloc.decode("ascii")
        request.extensions = dict(request.extensions)
        request.extensions["sni_hostname"] = host
        request.url = pinned_url
        try:
            return await self._transport.handle_async_request(request)
        finally:
            request.url = original_url

    async def aclose(self) -> None:
        await self._transport.aclose()


def public_async_client(**kwargs: object) -> httpx.AsyncClient:
    """Create an AsyncClient with DNS pinning enabled."""
    transport = kwargs.pop("transport", None)
    if transport is None:
        transport = PinnedPublicTransport()
    return httpx.AsyncClient(transport=transport, **kwargs)


__all__ = ["PinnedPublicTransport", "public_async_client", "resolve_public_addresses"]
