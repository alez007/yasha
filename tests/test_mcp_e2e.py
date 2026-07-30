"""End-to-end test for the MCP loop against a *real* MCP server.

Unlike test_mcp_loop.py (which patches list_mcp_tools/call_mcp_tool with canned
values), this spins up an actual in-process MCP server — the official SDK's
MCPServer, served over real streamable HTTP via uvicorn on localhost — so the
wire protocol, the official mcp client, and our client.py wrapper are all
exercised for real. Only the loader side is scripted (there's no real model
here); that part is exactly what test_mcp_loop.py already covers.

No external network and no GPU/Ray, so this is fast and reliable enough to run
in the default (non-integration) suite.
"""

import asyncio
import threading

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

from modelship.openai.mcp import loop as mcp_loop
from modelship.openai.protocol.responses.schemas import ResponsesRequest

_PORT = 8933


@pytest.fixture(scope="module")
def dice_server():
    server = MCPServer("dice")

    @server.tool()
    def roll(num_dice: int) -> str:
        """Roll num_dice six-sided dice."""
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


def _created(inner_id: str):
    return {"type": "response.created", "sequence_number": 0, "response": {"id": inner_id, "output": []}}


def _in_progress(inner_id: str):
    return {"type": "response.in_progress", "sequence_number": 1, "response": {"id": inner_id, "output": []}}


def _roll_call_turn(*, oi: int = 0):
    fc_item = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "roll",
        "arguments": "",
        "status": "in_progress",
    }
    completed_item = {**fc_item, "arguments": '{"num_dice": 2}', "status": "completed"}
    return [
        _created("i1"),
        _in_progress("i1"),
        {"type": "response.output_item.added", "sequence_number": 2, "output_index": oi, "item": fc_item},
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "item_id": "fc_1",
            "output_index": oi,
            "delta": '{"num_dice": 2}',
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 4,
            "item_id": "fc_1",
            "output_index": oi,
            "arguments": '{"num_dice": 2}',
        },
        {"type": "response.output_item.done", "sequence_number": 5, "output_index": oi, "item": completed_item},
        {
            "type": "response.completed",
            "sequence_number": 6,
            "response": {
                "id": "i1",
                "output": [completed_item],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            },
        },
    ]


def _final_answer_turn():
    msg_item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "The total is 6.", "annotations": []}],
    }
    return [
        _created("i2"),
        _in_progress("i2"),
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": {**msg_item, "status": "in_progress"},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "The total is 6.",
        },
        {"type": "response.output_item.done", "sequence_number": 4, "output_index": 0, "item": msg_item},
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": {
                "id": "i2",
                "output": [msg_item],
                "usage": {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
            },
        },
    ]


class FakeGen:
    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item

    def cancel(self):
        pass


class FakeHandle:
    """Stands in for the Ray DeploymentHandle: the loader side stays scripted
    (no real model in this test), only the MCP server side is real."""

    def __init__(self, turns: list[list]):
        self._turns = turns
        self.calls = 0

    @property
    def respond(self):
        return self

    def options(self, stream=True):
        return self

    def remote(self, *args, **kwargs):
        items = self._turns[self.calls]
        self.calls += 1
        return FakeGen(items)


class TestRealMcpServer:
    @pytest.mark.asyncio
    async def test_discovery_and_tool_call_against_real_server(self, dice_server):
        handle = FakeHandle([_roll_call_turn(), _final_answer_turn()])
        request = ResponsesRequest(
            model="m",
            input="roll 2 dice",
            stream=True,
            tools=[{"type": "mcp", "server_label": "dice", "server_url": dice_server, "require_approval": "never"}],
        )

        events = [e async for e in mcp_loop.run_mcp_response(handle, request, {}, None, "req_1", "unscoped")]

        seqs = [e["sequence_number"] for e in events]
        assert seqs == list(range(len(seqs)))
        assert sum(1 for e in events if e["type"] == "response.created") == 1

        terminals = [e for e in events if e["type"] in ("response.completed", "response.incomplete")]
        assert len(terminals) == 1
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "mcp_call", "message"]

        list_tools_item = output[0]
        assert list_tools_item["tools"][0]["name"] == "roll"

        mcp_call_item = output[1]
        assert mcp_call_item["status"] == "completed"
        # Real tool execution: the dice server actually ran and returned this.
        assert mcp_call_item["output"] == "total: 6"

    @pytest.mark.asyncio
    async def test_discovery_failure_against_unreachable_server(self):
        handle = FakeHandle([[]])
        request = ResponsesRequest(
            model="m",
            input="roll 2 dice",
            stream=True,
            tools=[{"type": "mcp", "server_label": "dice", "server_url": "http://127.0.0.1:1/mcp"}],
        )

        events = [e async for e in mcp_loop.run_mcp_response(handle, request, {}, None, "req_1", "unscoped")]

        failed = [e for e in events if e["type"] == "response.failed"]
        assert len(failed) == 1
        assert "dice" in failed[0]["response"]["error"]["message"]
        assert handle.calls == 0
