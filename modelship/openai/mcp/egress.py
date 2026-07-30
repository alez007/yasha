"""Egress policy for outbound MCP server connections.

Permissive by default (decision: a homelab MCP server on ``http://localhost:3000`` is
the common case) — plain ``http://``, localhost, and private IPs are all allowed unless
explicitly locked down. Cloud metadata endpoints are blocked unconditionally. This is
hostname-level policy only, not DNS-rebinding-proof; multi-tenant deploys should set
``MSHIP_MCP_ALLOWED_HOSTS``.
"""

from __future__ import annotations

import ipaddress
import os
from http import HTTPStatus
from urllib.parse import urlsplit

from modelship.openai.protocol import create_error_response
from modelship.openai.utils.responses import ResponsesApiError

_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata.goog"})
_METADATA_NETWORK = ipaddress.ip_network("169.254.0.0/16")


def _egress_error(message: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            message, err_type="invalid_request_error", status_code=HTTPStatus.BAD_REQUEST, param="tools"
        )
    )


def validate_server_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise _egress_error(f"mcp server_url must be http or https, got {parts.scheme!r}.")

    require_https = os.environ.get("MSHIP_MCP_REQUIRE_HTTPS", "").strip().lower() in ("1", "true", "yes")
    if require_https and parts.scheme != "https":
        raise _egress_error("MSHIP_MCP_REQUIRE_HTTPS is set; mcp server_url must use https.")

    hostname = parts.hostname or ""
    if not hostname:
        raise _egress_error(f"mcp server_url {url!r} is missing a hostname.")
    if hostname.lower() in _METADATA_HOSTS:
        raise _egress_error(f"mcp server_url host {hostname!r} is a cloud metadata endpoint and is always blocked.")
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        addr = None
    if addr is not None and addr in _METADATA_NETWORK:
        raise _egress_error(f"mcp server_url host {hostname!r} is a cloud metadata endpoint and is always blocked.")

    allowed_hosts = os.environ.get("MSHIP_MCP_ALLOWED_HOSTS", "").strip()
    if allowed_hosts:
        allowed = {h.strip().lower() for h in allowed_hosts.split(",") if h.strip()}
        if hostname.lower() not in allowed:
            raise _egress_error(f"mcp server_url host {hostname!r} is not in MSHIP_MCP_ALLOWED_HOSTS.")


__all__ = ["validate_server_url"]
