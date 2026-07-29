"""Parsing and policy for the Responses ``mcp`` tool type: no I/O here, just turning
request-side ``tools[]`` dicts into typed :class:`McpToolSpec` objects and resolving
``allowed_tools`` / ``require_approval`` against a server's discovered tool list.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from modelship.openai.protocol import create_error_response
from modelship.openai.protocol.base import OpenAIBaseModel
from modelship.openai.protocol.responses.schemas import McpListToolsTool
from modelship.openai.utils.responses import ResponsesApiError


class McpToolSpec(OpenAIBaseModel):
    server_label: str
    server_url: str
    headers: dict[str, str] | None = None
    authorization: str | None = None
    allowed_tools: Any | None = None
    require_approval: Any | None = None


def _mcp_error(message: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            message, err_type="invalid_request_error", status_code=HTTPStatus.BAD_REQUEST, param="tools"
        )
    )


def split_mcp_tools(tools: list[dict[str, Any]] | None) -> tuple[list[McpToolSpec], list[dict[str, Any]]]:
    """Partition ``type: "mcp"`` tool dicts from the rest. Client ``function`` (and any
    other) tools pass through untouched in the second element."""
    if not tools:
        return [], []
    specs: list[McpToolSpec] = []
    other: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for tool in tools:
        if tool.get("type") != "mcp":
            other.append(tool)
            continue
        if tool.get("connector_id"):
            raise _mcp_error("OpenAI-hosted MCP connectors ('connector_id') are not supported; supply 'server_url'.")
        server_url = tool.get("server_url")
        if not server_url:
            raise _mcp_error("mcp tool requires a non-empty 'server_url'.")
        server_label = tool.get("server_label")
        if not server_label:
            raise _mcp_error("mcp tool requires a non-empty 'server_label'.")
        if server_label in seen_labels:
            raise _mcp_error(f"duplicate mcp server_label {server_label!r}.")
        seen_labels.add(server_label)
        specs.append(
            McpToolSpec(
                server_label=server_label,
                server_url=server_url,
                headers=tool.get("headers"),
                authorization=tool.get("authorization"),
                allowed_tools=tool.get("allowed_tools"),
                require_approval=tool.get("require_approval"),
            )
        )
    return specs, other


def _read_only(tool: McpListToolsTool) -> bool:
    annotations = tool.annotations or {}
    return bool(annotations.get("readOnlyHint") or annotations.get("read_only_hint"))


def filter_tools(spec: McpToolSpec, tools: list[McpListToolsTool]) -> list[McpListToolsTool]:
    """Apply ``allowed_tools``: a plain name list, ``{"tool_names": [...]}``, or
    ``{"read_only": true}`` (keep only tools whose annotations mark them read-only)."""
    allowed = spec.allowed_tools
    if allowed is None:
        return tools

    if isinstance(allowed, list):
        names = set(allowed)
        return [t for t in tools if t.name in names]

    if isinstance(allowed, dict):
        names = allowed.get("tool_names")
        read_only = allowed.get("read_only")
        out = tools
        if names is not None:
            name_set = set(names)
            out = [t for t in out if t.name in name_set]
        if read_only:
            out = [t for t in out if _read_only(t)]
        return out

    raise _mcp_error(f"unsupported 'allowed_tools' shape for server {spec.server_label!r}.")


def requires_approval(spec: McpToolSpec, tool: McpListToolsTool) -> bool:
    """Resolve whether *tool* needs approval. Unset ``require_approval`` defaults to
    ``"always"`` (OpenAI's default)."""
    setting = spec.require_approval
    if setting is None or setting == "always":
        return True
    if setting == "never":
        return False
    if isinstance(setting, dict):
        never = setting.get("never") or {}
        always = setting.get("always") or {}
        if not isinstance(never, dict) or not isinstance(always, dict):
            raise _mcp_error(f"unsupported 'require_approval' shape for server {spec.server_label!r}.")
        if tool.name in (never.get("tool_names") or []):
            return False
        if tool.name in (always.get("tool_names") or []):
            return True
        if never.get("read_only") and _read_only(tool):
            return False
        if always.get("read_only") and _read_only(tool):
            return True
        return True
    raise _mcp_error(f"unsupported 'require_approval' shape for server {spec.server_label!r}.")


def check_collisions(
    specs: list[McpToolSpec], discovered: dict[str, list[McpListToolsTool]], client_tools: list[dict[str, Any]]
) -> None:
    """A tool name reachable from two servers, or clashing with a client ``function``
    tool name, can't be disambiguated in the model's flat tool namespace."""
    client_names = {t.get("name") for t in client_tools if t.get("type") == "function" and t.get("name")}
    seen: dict[str, str] = {}
    for spec in specs:
        for tool in discovered.get(spec.server_label, []):
            if tool.name in client_names:
                raise _mcp_error(
                    f"tool name {tool.name!r} from server {spec.server_label!r} collides with a client tool."
                )
            owner = seen.get(tool.name)
            if owner is not None and owner != spec.server_label:
                raise _mcp_error(f"tool name {tool.name!r} is exposed by both {owner!r} and {spec.server_label!r}.")
            seen[tool.name] = spec.server_label


__all__ = [
    "McpToolSpec",
    "check_collisions",
    "filter_tools",
    "requires_approval",
    "split_mcp_tools",
]
