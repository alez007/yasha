"""Route-level tests for `background:true` on /v1/responses (Phase E1). The drain
task runs detached, so most tests call `_drain_background_tasks(api)` after
triggering the route before asserting on the stored snapshot."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from modelship.openai.api import ModelshipAPI
from modelship.openai.protocol import ResponsesRequest
from modelship.state import MemoryStoreActor, StateStoreUnavailableError

_ModelshipAPI = ModelshipAPI.func_or_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def _fake_disconnect_registry():
    """`utils.responses` imports `get_disconnect_registry` directly, so the
    conftest-level patch on `infer_config`'s copy doesn't reach it — fake it here."""
    registry = MagicMock()
    registry.set.remote = AsyncMock()
    with patch("modelship.openai.utils.responses.get_disconnect_registry", return_value=registry):
        yield registry


@pytest.fixture
def api():
    with (
        patch("modelship.openai.api.serve.get_replica_context") as mock_ctx,
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
    ):
        mock_ctx.return_value.app_name = "test-gateway"
        inst = _ModelshipAPI("test-gateway")
        inst._watch_task = MagicMock()
        inst._state_store = _MemoryStore()
        return inst


def _raw_request():
    raw = MagicMock()
    raw.headers = {}
    return raw


def _stored(api, response_id="resp_1", identity="unscoped", status="completed", output=None):
    api._state_store.set(
        f"responses/{identity}/{response_id}",
        {
            "response": {"id": response_id, "object": "response", "status": status, "output": output or []},
            "input_items": [],
        },
    )


def _stored_background(
    api, response_id="resp_1", identity="unscoped", status="in_progress", req_id="req-1", output=None
):
    response = {"id": response_id, "object": "response", "status": status, "background": True, "output": output or []}
    value = {"response": response, "input_items": []}
    if status in ("queued", "in_progress"):
        value["_mship"] = {"req_id": req_id, "updated_at": time.time()}
    api._state_store.set(f"responses/{identity}/{response_id}", value)


def _wire(api, side_effect):
    """Wire `handle.respond.options(...).remote(...)` to `side_effect`."""
    handle = MagicMock()
    handle.respond.options.return_value.remote.side_effect = side_effect
    api.models = {"m": {"m-a1b2c": handle}}
    api._round_robin = {"m": 0}
    return handle


def _background_gen_factory(text="hello background!", status="completed"):
    """A fake `handle.respond` remote-call side effect: reads the injected
    `response_id` (positional arg 5) so the terminal event's id matches the placeholder's."""

    def _make_gen(*args, **kwargs):
        response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

        async def gen():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            yield {
                "type": f"response.{status}",
                "sequence_number": 1,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": status,
                    "background": True,
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                    ],
                },
            }

        return gen()

    return _make_gen


def _completes_after_callback_factory(callback, text="hello background!"):
    """Like `_background_gen_factory`, but runs `callback()` between the `created`
    and `completed` events — deterministically races a genuine completion against
    whatever `callback` does (e.g. cancel/delete) landing in the store first."""

    def _make_gen(*args, **kwargs):
        response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

        async def gen():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            await callback(response_id)
            yield {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "background": True,
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                    ],
                },
            }

        return gen()

    return _make_gen


def _hanging_gen_factory(started: asyncio.Event, release: asyncio.Event):
    """A fake remote-call side effect that yields one event, signals `started`, then
    blocks on `release` — simulating a run still in flight, so a test can cancel or
    delete it mid-run before letting it wind down."""

    def _make_gen(*args, **kwargs):
        response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

        async def gen():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            started.set()
            await release.wait()

        return gen()

    return _make_gen


async def _drain_background_tasks(api):
    tasks = list(api._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


class TestBackgroundCreate:
    @pytest.mark.asyncio
    async def test_returns_queued_immediately(self, api):
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", background=True)

        result = await api.create_response(request, _raw_request())

        body = json.loads(bytes(result.body))
        assert body["status"] == "queued"
        assert body["id"].startswith("resp_")
        assert body["background"] is True
        assert "_mship" not in body

        # The drain task hasn't run yet (nothing has awaited it): an immediate GET
        # sees the same queued placeholder.
        get_result = await api.get_response(body["id"], _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "queued"

        await _drain_background_tasks(api)

    @pytest.mark.asyncio
    async def test_drains_to_completed(self, api):
        _wire(api, _background_gen_factory(text="hello background!"))
        request = ResponsesRequest(model="m", input="hi", background=True)

        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]

        await _drain_background_tasks(api)

        get_result = await api.get_response(response_id, _raw_request())
        body = json.loads(bytes(get_result.body))
        assert body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"] == "hello background!"
        assert "_mship" not in body


class TestGuards:
    @pytest.mark.asyncio
    async def test_background_and_store_false_is_400(self, api):
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", background=True, store=False)

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 400
        assert api._background_tasks == set()

    @pytest.mark.asyncio
    async def test_previous_response_id_in_progress_is_400(self, api):
        _stored_background(api, "resp_running", status="in_progress")
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_running")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 400
        assert exc.value.err.error.param == "previous_response_id"

    @pytest.mark.asyncio
    async def test_previous_response_id_queued_is_400(self, api):
        _stored_background(api, "resp_queued", status="queued")
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_queued")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 400


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_in_progress_marks_cancelled_and_survives_abnormal_end(self, api):
        started = asyncio.Event()
        release = asyncio.Event()
        _wire(api, _hanging_gen_factory(started, release))

        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]
        await started.wait()

        cancel_result = await api.cancel_response(response_id, _raw_request())
        body = json.loads(bytes(cancel_result.body))
        assert body["status"] == "cancelled"

        # Let the "generation" end abnormally (no terminal event) — the drain task's
        # fallback-failure write must not regress an already-cancelled status.
        release.set()
        await _drain_background_tasks(api)

        get_result = await api.get_response(response_id, _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_wins_race_against_a_genuine_completion(self, api):
        # A cancel that lands in the store while the generation is still mid-flight
        # must not be clobbered by that generation's own later, unconditional-looking
        # completion write.
        async def _cancel_mid_stream(response_id):
            await api.cancel_response(response_id, _raw_request())

        _wire(api, _completes_after_callback_factory(_cancel_mid_stream))
        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]

        await _drain_background_tasks(api)

        get_result = await api.get_response(response_id, _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_on_terminal_background_is_idempotent(self, api):
        _stored_background(api, "resp_done", status="completed")

        result = await api.cancel_response("resp_done", _raw_request())

        assert json.loads(bytes(result.body))["status"] == "completed"

    @pytest.mark.asyncio
    async def test_cancel_on_non_background_response_is_400(self, api):
        _stored(api, "resp_chat", status="completed")

        with pytest.raises(HTTPException) as exc:
            await api.cancel_response("resp_chat", _raw_request())
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_cancel_unknown_is_404(self, api):
        with pytest.raises(HTTPException) as exc:
            await api.cancel_response("resp_nope", _raw_request())
        assert exc.value.status_code == 404


class TestDeleteImpliesCancel:
    @pytest.mark.asyncio
    async def test_delete_in_flight_signals_cancel_and_removes_snapshot(self, api):
        started = asyncio.Event()
        release = asyncio.Event()
        _wire(api, _hanging_gen_factory(started, release))

        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]
        await started.wait()

        delete_result = await api.delete_response(response_id, _raw_request())
        assert json.loads(bytes(delete_result.body))["deleted"] is True
        assert api._state_store.get(f"responses/unscoped/{response_id}") is None

        # The drain task finishes only after the delete; it must not resurrect what
        # was just removed.
        release.set()
        await _drain_background_tasks(api)

        assert api._state_store.get(f"responses/unscoped/{response_id}") is None

    @pytest.mark.asyncio
    async def test_delete_wins_race_against_a_genuine_completion(self, api):
        # A delete that lands mid-flight must not be resurrected by that generation's
        # own later, unconditional-looking completion write.
        async def _delete_mid_stream(response_id):
            await api.delete_response(response_id, _raw_request())

        _wire(api, _completes_after_callback_factory(_delete_mid_stream))
        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]

        await _drain_background_tasks(api)

        assert api._state_store.get(f"responses/unscoped/{response_id}") is None


class TestOrphanDetection:
    @pytest.mark.asyncio
    async def test_stale_heartbeat_reports_failed_on_get(self, api):
        _stored_background(api, "resp_stale", status="in_progress")
        snapshot = api._state_store.get("responses/unscoped/resp_stale")
        snapshot["_mship"]["updated_at"] = time.time() - 3600
        api._state_store.set("responses/unscoped/resp_stale", snapshot)

        result = await api.get_response("resp_stale", _raw_request())

        body = json.loads(bytes(result.body))
        assert body["status"] == "failed"
        assert "_mship" not in body

    @pytest.mark.asyncio
    async def test_fresh_heartbeat_stays_in_progress(self, api):
        _stored_background(api, "resp_fresh", status="in_progress")

        result = await api.get_response("resp_fresh", _raw_request())

        assert json.loads(bytes(result.body))["status"] == "in_progress"


def _parse_sse(body: str) -> list[dict]:
    events = []
    for frame in body.split("\n\n"):
        lines = [line for line in frame.split("\n") if line]
        if not lines:
            continue
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        if data_line is None:
            continue
        data = data_line[len("data: ") :]
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


async def _consume_body(result) -> str:
    chunks = await asyncio.wait_for(_collect(result), timeout=5.0)
    return "".join(chunks)


async def _collect(result):
    return [chunk async for chunk in result.body_iterator]


class TestBackgroundStream:
    @pytest.mark.asyncio
    async def test_background_and_stream_streams_live_and_completes(self, api):
        # A real gap between events so the tailer's poll observes `response.created`
        # while buffered, rather than racing a drain task that finishes first.
        def _slow_gen_factory(*args, **kwargs):
            response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

            async def gen():
                yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
                await asyncio.sleep(0.4)
                yield {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "background": True,
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "hello background!"}],
                            }
                        ],
                    },
                }

            return gen()

        _wire(api, _slow_gen_factory)
        request = ResponsesRequest(model="m", input="hi", background=True, stream=True)

        result = await api.create_response(request, _raw_request())
        assert result.media_type == "text/event-stream"

        body = await _consume_body(result)
        events = _parse_sse(body)

        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["status"] == "completed"
        assert events[-1]["response"]["output"][0]["content"][0]["text"] == "hello background!"

        await _drain_background_tasks(api)

    @pytest.mark.asyncio
    async def test_get_stream_resumes_from_buffered_events(self, api):
        response_id = "resp_1"
        _stored_background(api, response_id, status="completed")
        key = f"responses-stream/unscoped/{response_id}"
        await api._state_store.append_async(key, {"type": "response.created", "sequence_number": 0, "response": {}})
        await api._state_store.append_async(
            key, {"type": "response.output_text.delta", "sequence_number": 1, "delta": "hi"}
        )
        await api._state_store.append_async(
            key,
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {"id": response_id, "object": "response", "status": "completed"},
            },
        )

        result = await api.get_response(response_id, _raw_request(), stream=True, starting_after=0)
        body = await _consume_body(result)
        events = _parse_sse(body)

        # starting_after=0 excludes the seq-0 event that was already seen before disconnect.
        assert [e["sequence_number"] for e in events] == [1, 2]
        assert events[-1]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_get_stream_on_terminal_response_synthesizes_terminal_event(self, api):
        _stored_background(api, "resp_done", status="completed")

        result = await api.get_response("resp_done", _raw_request(), stream=True)
        body = await _consume_body(result)
        events = _parse_sse(body)

        assert len(events) == 1
        assert events[0]["type"] == "response.completed"
        assert events[0]["response"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_stream_ends_cleanly_on_buffer_read_outage(self, api):
        _stored_background(api, "resp_1", status="in_progress")
        with patch.object(
            api._state_store, "read_from_async", AsyncMock(side_effect=StateStoreUnavailableError("down"))
        ):
            result = await api.get_response("resp_1", _raw_request(), stream=True)
            body = await _consume_body(result)

        # Ends the stream quietly rather than raising mid-response (headers already sent).
        assert _parse_sse(body) == []
