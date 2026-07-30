"""Thin wrapper over the official ``mcp`` SDK: one fresh, stateless ``Client`` per
operation (matching LangChain's ``MultiServerMCPClient`` default). This module is the
only place that imports ``mcp``.
"""

from __future__ import annotations

import asyncio
import json

import httpx2

from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from modelship.openai.mcp.spec import McpToolSpec
from modelship.openai.protocol.responses.schemas import McpListToolsTool

LIST_TOOLS_TIMEOUT_S = 30
CALL_TOOL_TIMEOUT_S = 60

_MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MiB


class McpCallError(Exception):
    """Raised for any MCP-side failure: network, protocol, or output-too-large.
    The loop records this on the ``mcp_call`` item's ``error`` field rather than
    letting it escape — a broken tool server doesn't take down the response."""


def _headers_for(spec: McpToolSpec) -> dict[str, str]:
    headers = dict(spec.headers or {})
    if spec.authorization and not any(k.lower() == "authorization" for k in headers):
        headers["Authorization"] = f"Bearer {spec.authorization}"
    return headers


def _http_client(spec: McpToolSpec, timeout_s: float) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(headers=_headers_for(spec), timeout=timeout_s)


async def list_mcp_tools(spec: McpToolSpec) -> list[McpListToolsTool]:
    try:
        async with asyncio.timeout(LIST_TOOLS_TIMEOUT_S):
            async with _http_client(spec, LIST_TOOLS_TIMEOUT_S) as http_client:
                transport = streamable_http_client(spec.server_url, http_client=http_client)
                async with Client(transport, mode="auto") as client:
                    result = await client.list_tools()
    except TimeoutError as exc:
        raise McpCallError(
            f"server {spec.server_label!r}: tools/list timed out after {LIST_TOOLS_TIMEOUT_S}s."
        ) from exc
    except MCPError as exc:
        raise McpCallError(f"server {spec.server_label!r}: tools/list failed: {exc.message}") from exc
    except Exception as exc:
        raise McpCallError(f"server {spec.server_label!r}: tools/list failed: {exc}") from exc

    return [
        McpListToolsTool(
            name=tool.name,
            input_schema=tool.input_schema,
            annotations=tool.annotations.model_dump(exclude_none=True) if tool.annotations else None,
            description=tool.description,
        )
        for tool in result.tools
    ]


def _flatten_content(content: list) -> str:
    parts: list[str] = []
    for part in content:
        if getattr(part, "type", None) == "text":
            parts.append(part.text)
        else:
            parts.append(part.model_dump_json(exclude_none=True))
    return "".join(parts)


async def call_mcp_tool(spec: McpToolSpec, name: str, arguments_json: str) -> str:
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        raise McpCallError(f"tool {name!r} on server {spec.server_label!r}: invalid arguments JSON: {exc}") from exc

    try:
        async with asyncio.timeout(CALL_TOOL_TIMEOUT_S):
            async with _http_client(spec, CALL_TOOL_TIMEOUT_S) as http_client:
                transport = streamable_http_client(spec.server_url, http_client=http_client)
                async with Client(transport, mode="auto") as client:
                    result = await client.call_tool(name, arguments)
    except TimeoutError as exc:
        raise McpCallError(
            f"tool {name!r} on server {spec.server_label!r} timed out after {CALL_TOOL_TIMEOUT_S}s."
        ) from exc
    except MCPError as exc:
        raise McpCallError(f"tool {name!r} on server {spec.server_label!r} failed: {exc.message}") from exc
    except Exception as exc:
        raise McpCallError(f"tool {name!r} on server {spec.server_label!r} failed: {exc}") from exc

    output = _flatten_content(result.content)
    if result.is_error:
        raise McpCallError(f"tool {name!r} on server {spec.server_label!r} returned an error: {output}")
    if len(output.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        raise McpCallError(f"tool {name!r} on server {spec.server_label!r} output exceeds {_MAX_OUTPUT_BYTES} bytes.")
    return output


__all__ = ["CALL_TOOL_TIMEOUT_S", "LIST_TOOLS_TIMEOUT_S", "McpCallError", "call_mcp_tool", "list_mcp_tools"]
