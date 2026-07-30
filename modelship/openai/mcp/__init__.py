"""Server-side MCP tool execution for ``/v1/responses``: :mod:`.spec` parses the
request-side ``mcp`` tool type, :mod:`.egress` guards outbound server URLs,
:mod:`.client` wraps the official ``mcp`` SDK, and :mod:`.loop` is the
orchestrator + stream-stitcher that ties them together.
"""
