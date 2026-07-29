"""Real-cluster integration test for server-side MCP tool execution: a real
deployed model (via the shared `mship_cluster`/`model_deployer` fixtures in
conftest.py) firing real /v1/responses requests with an `mcp` tool against a
real in-process MCP server (the official SDK's MCPServer, served over
streamable HTTP on localhost) — the gateway subprocess reaches it exactly like
it would reach any self-hosted MCP server. No mocking anywhere in this file.
"""

import asyncio
import threading

import httpx
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

OPENAI_API_BASE = "http://localhost:8000/v1"
_PORT = 8934


@pytest.fixture(scope="module")
def dice_server():
    server = MCPServer("dice")

    @server.tool()
    def roll(num_dice: int) -> str:
        """Roll num_dice six-sided dice and return the total."""
        return f"total: {num_dice * 3}"

    app = server.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=_PORT, log_level="warning")
    uv_server = uvicorn.Server(config)

    def run():
        asyncio.run(uv_server.serve())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(100):
        if uv_server.started:
            break
        threading.Event().wait(0.05)
    assert uv_server.started, "test MCP server failed to start"
    yield f"http://127.0.0.1:{_PORT}/mcp"
    uv_server.should_exit = True
    thread.join(timeout=5)


def _mcp_tool(dice_server, **overrides):
    tool = {"type": "mcp", "server_label": "dice", "server_url": dice_server, "require_approval": "never"}
    tool.update(overrides)
    return tool


@pytest.mark.integration
@pytest.mark.vllm
class TestMcpIntegration:
    """Real model + real MCP server, end to end through the actual gateway
    subprocess (mship_deploy.py --reconcile), not the loop function directly."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_model_discovers_and_calls_real_mcp_tool(self, client, dice_server):
        resp = client.responses.create(
            model="chat-capable",
            input="Use the roll tool to roll 2 dice, then tell me the total.",
            tools=[_mcp_tool(dice_server)],
            tool_choice="required",
            max_output_tokens=256,
        )

        assert resp.status in {"completed", "incomplete"}, resp.model_dump()
        types = [item.type for item in resp.output]
        assert "mcp_list_tools" in types
        assert "mcp_call" in types

        list_tools_item = next(item for item in resp.output if item.type == "mcp_list_tools")
        assert any(t.name == "roll" for t in list_tools_item.tools)

        mcp_call_item = next(item for item in resp.output if item.type == "mcp_call")
        assert mcp_call_item.name == "roll"
        assert mcp_call_item.status == "completed"
        # Real tool execution against the real dice server, not a mock.
        assert mcp_call_item.output == "total: 6"
        assert mcp_call_item.server_label == "dice"

        # The model's final answer should reflect the real tool result.
        assert resp.output_text.strip()

    def test_streaming_model_discovers_and_calls_real_mcp_tool(self, client, dice_server):
        stream = client.responses.create(
            model="chat-capable",
            input="Use the roll tool to roll 3 dice, then tell me the total.",
            tools=[_mcp_tool(dice_server)],
            tool_choice="required",
            max_output_tokens=256,
            stream=True,
        )

        types: list[str] = []
        arg_deltas: list[str] = []
        completed = None
        for event in stream:
            types.append(event.type)
            if event.type == "response.mcp_call_arguments.delta":
                arg_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response

        assert types[0] == "response.created"
        assert "response.mcp_list_tools.completed" in types
        assert "response.mcp_call.completed" in types
        assert completed is not None
        mcp_call_item = next(item for item in completed.output if item.type == "mcp_call")
        assert mcp_call_item.output == "total: 9"
        # Streamed argument fragments must reconstruct the final call's arguments.
        assert "".join(arg_deltas) == mcp_call_item.arguments

    def test_require_approval_round_trip_via_previous_response_id(self, client, dice_server):
        first = client.responses.create(
            model="chat-capable",
            input="Use the roll tool to roll 2 dice, then tell me the total.",
            tools=[_mcp_tool(dice_server, require_approval="always")],
            tool_choice="required",
            max_output_tokens=256,
        )
        assert first.status == "completed"
        approval_items = [item for item in first.output if item.type == "mcp_approval_request"]
        assert approval_items, f"expected an mcp_approval_request, got {[i.type for i in first.output]}"
        approval = approval_items[0]
        assert approval.name == "roll"

        second = client.responses.create(
            model="chat-capable",
            previous_response_id=first.id,
            input=[{"type": "mcp_approval_response", "approval_request_id": approval.id, "approve": True}],
            tools=[_mcp_tool(dice_server, require_approval="always")],
            max_output_tokens=256,
        )
        assert second.status in {"completed", "incomplete"}
        mcp_call_items = [item for item in second.output if item.type == "mcp_call"]
        assert mcp_call_items, f"expected an mcp_call after approval, got {[i.type for i in second.output]}"
        assert mcp_call_items[0].status == "completed"
        assert mcp_call_items[0].output == "total: 6"
        assert mcp_call_items[0].approval_request_id == approval.id

    def test_unreachable_mcp_server_fails_the_response(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={
                "model": "chat-capable",
                "input": "roll some dice",
                "tools": [{"type": "mcp", "server_label": "dice", "server_url": "http://127.0.0.1:1/mcp"}],
            },
            timeout=60,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert "dice" in body["error"]["message"]
