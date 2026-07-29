"""Tests for the /v1/responses conversation-state domain layer — snapshot round-trip,
history rebuild, identity scoping, TTL, and availability-vs-absent semantics.

Runs against every backend the store supports, so the layer is proven on the default
memory:// and on the durable redis:// alike.
"""

from unittest.mock import patch

import pytest

from modelship.openai.state import responses as responses_state
from modelship.state import MemoryStoreActor, StateStoreUnavailableError
from modelship.state.redis import RedisStateStore

_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


def _fake_redis_store():
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    s = RedisStateStore.__new__(RedisStateStore)
    s._url = "redis://fake"
    s._sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    s._async_client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    return s


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return _MemoryStore()
    return _fake_redis_store()


def _response(response_id: str = "resp_1", output: list | None = None) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "output": output if output is not None else [{"type": "message", "role": "assistant", "content": []}],
    }


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_write_then_read(self, store):
        await responses_state.write_async(
            store, "u1", "resp_1", response=_response(), input_items=[{"type": "message", "role": "user"}]
        )
        snap = await responses_state.read_async(store, "u1", "resp_1")
        assert snap is not None
        assert snap["response"]["id"] == "resp_1"
        assert snap["input_items"] == [{"type": "message", "role": "user"}]

    @pytest.mark.asyncio
    async def test_absent_returns_none(self, store):
        assert await responses_state.read_async(store, "u1", "nope") is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await responses_state.write_async(store, "u1", "resp_1", response=_response(), input_items=[])
        await responses_state.delete_async(store, "u1", "resp_1")
        assert await responses_state.read_async(store, "u1", "resp_1") is None
        # idempotent — deleting an absent snapshot is not an error
        await responses_state.delete_async(store, "u1", "resp_1")

    @pytest.mark.asyncio
    async def test_sync_read_sees_async_write(self, store):
        await responses_state.write_async(store, "u1", "resp_1", response=_response(), input_items=[])
        assert responses_state.read(store, "u1", "resp_1") is not None


class TestIdentityScoping:
    """A bare response_id would let any caller fetch another's conversation."""

    @pytest.mark.asyncio
    async def test_other_identity_cannot_read(self, store):
        await responses_state.write_async(store, "u1", "resp_1", response=_response(), input_items=[])
        assert await responses_state.read_async(store, "u2", "resp_1") is None

    @pytest.mark.asyncio
    async def test_same_id_different_identities_are_independent(self, store):
        await responses_state.write_async(store, "u1", "resp_1", response=_response("a"), input_items=[])
        await responses_state.write_async(store, "u2", "resp_1", response=_response("b"), input_items=[])
        u1 = await responses_state.read_async(store, "u1", "resp_1")
        u2 = await responses_state.read_async(store, "u2", "resp_1")
        assert u1 is not None and u2 is not None
        assert u1["response"]["id"] == "a"
        assert u2["response"]["id"] == "b"


class TestHistoryItems:
    def test_rebuild_is_input_then_output(self):
        snapshot = {
            "input_items": [{"type": "message", "role": "user", "content": "hi"}],
            "response": {"output": [{"type": "message", "role": "assistant", "content": "yo"}]},
        }
        assert responses_state.history_items(snapshot) == [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "yo"},
        ]

    def test_accumulates_across_turns(self):
        # Turn 2's input_items are turn 1's rebuild, so the snapshot stays self-contained.
        turn1 = {"input_items": [{"i": 1}], "response": {"output": [{"o": 1}]}}
        turn2_input = [*responses_state.history_items(turn1), {"i": 2}]
        turn2 = {"input_items": turn2_input, "response": {"output": [{"o": 2}]}}
        assert responses_state.history_items(turn2) == [{"i": 1}, {"o": 1}, {"i": 2}, {"o": 2}]

    @pytest.mark.parametrize(
        "snapshot",
        [
            {},
            {"input_items": None, "response": {}},
            {"input_items": "nope", "response": {"output": "nope"}},
        ],
    )
    def test_malformed_shapes_do_not_raise(self, snapshot):
        assert responses_state.history_items(snapshot) == []


class TestMalformedSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_without_response_treated_as_missing(self, store):
        await store.set_async("responses/u1/resp_1", {"input_items": []})
        assert await responses_state.read_async(store, "u1", "resp_1") is None
        assert responses_state.read(store, "u1", "resp_1") is None


class TestUnavailableVsAbsent:
    """A store outage must 503, never look like an unknown id."""

    @pytest.mark.asyncio
    async def test_unavailable_propagates(self):
        class Down:
            def get(self, *a, **k):
                raise StateStoreUnavailableError("down")

            async def get_async(self, *a, **k):
                raise StateStoreUnavailableError("down")

        with pytest.raises(StateStoreUnavailableError):
            await responses_state.read_async(Down(), "u1", "resp_1")
        with pytest.raises(StateStoreUnavailableError):
            responses_state.read(Down(), "u1", "resp_1")


class TestBackgroundHeartbeat:
    @pytest.mark.asyncio
    async def test_touch_refreshes_updated_at(self, store):
        await responses_state.write_background(
            store, "u1", "resp_1", response={**_response(), "status": "in_progress"}, input_items=[], req_id="req-1"
        )
        assert await responses_state.touch(store, "u1", "resp_1") is True
        snap = await responses_state.read_async(store, "u1", "resp_1")
        assert snap["_mship"]["updated_at"] > 0

    @pytest.mark.asyncio
    async def test_touch_on_absent_snapshot_returns_false(self, store):
        assert await responses_state.touch(store, "u1", "nope") is False

    @pytest.mark.asyncio
    async def test_touch_without_mship_returns_false(self, store):
        await responses_state.write_async(store, "u1", "resp_1", response=_response(), input_items=[])
        assert await responses_state.touch(store, "u1", "resp_1") is False

    @pytest.mark.asyncio
    async def test_touch_does_not_overwrite_an_already_terminal_snapshot(self, store):
        # Simulates the heartbeat losing a race against a terminal write: the
        # snapshot moved to `cancelled` (e.g. via write_terminal_if_not_terminal)
        # while a stale `_mship` sidecar somehow still lingers. touch() must
        # defer to the terminal status rather than blindly writing it back.
        await store.set_async(
            "responses/u1/resp_1",
            {
                "response": {**_response(), "status": "cancelled"},
                "input_items": [],
                "_mship": {"req_id": "req-1", "updated_at": 1.0},
            },
        )
        assert await responses_state.touch(store, "u1", "resp_1") is False
        snap = await responses_state.read_async(store, "u1", "resp_1")
        assert snap["response"]["status"] == "cancelled"
        assert snap["_mship"]["updated_at"] == 1.0

    @pytest.mark.asyncio
    async def test_touch_does_not_regress_a_terminal_write_landing_mid_tick(self, store):
        # Regression: touch() used to do a single read then an unconditional write of
        # that snapshot. A terminal writer (drain completion, cancel, staleness)
        # landing between touch's read and its write got silently clobbered back to
        # non-terminal — permanently, since nothing else re-corrects it once the
        # drain task has exited.
        await responses_state.write_background(
            store, "u1", "resp_1", response={**_response(), "status": "in_progress"}, input_items=[], req_id="req-1"
        )
        real_get_async = store.get_async
        calls = 0

        async def racing_get_async(key):
            nonlocal calls
            calls += 1
            value = await real_get_async(key)
            if calls == 1:
                # A concurrent terminal writer lands right between touch()'s two reads.
                await store.set_async(key, {"response": {**_response(), "status": "cancelled"}, "input_items": []})
            return value

        with patch.object(store, "get_async", racing_get_async):
            assert await responses_state.touch(store, "u1", "resp_1") is False

        snap = await responses_state.read_async(store, "u1", "resp_1")
        assert snap["response"]["status"] == "cancelled"
        assert "_mship" not in snap


class TestStreamBuffer:
    @pytest.mark.asyncio
    async def test_append_then_read_roundtrips(self, store):
        await responses_state.append_stream_event(
            store, "u1", "resp_1", {"type": "response.created", "sequence_number": 0}
        )
        await responses_state.append_stream_event(
            store, "u1", "resp_1", {"type": "response.output_text.delta", "sequence_number": 1}
        )
        events = await responses_state.read_stream_events_after(store, "u1", "resp_1", -1)
        assert [e["sequence_number"] for e in events] == [0, 1]

    @pytest.mark.asyncio
    async def test_read_after_sequence_excludes_seen(self, store):
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 0})
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 1})
        events = await responses_state.read_stream_events_after(store, "u1", "resp_1", 0)
        assert [e["sequence_number"] for e in events] == [1]

    @pytest.mark.asyncio
    async def test_read_absent_buffer_returns_empty(self, store):
        assert await responses_state.read_stream_events_after(store, "u1", "nope", -1) == []

    @pytest.mark.asyncio
    async def test_discard_removes_buffer(self, store):
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 0})
        await responses_state.discard_stream_buffer(store, "u1", "resp_1")
        assert await responses_state.read_stream_events_after(store, "u1", "resp_1", -1) == []
        # idempotent — discarding an absent buffer is not an error
        await responses_state.discard_stream_buffer(store, "u1", "resp_1")

    @pytest.mark.asyncio
    async def test_scoped_by_identity(self, store):
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 0})
        assert await responses_state.read_stream_events_after(store, "u2", "resp_1", -1) == []


class TestStreamBufferTtl:
    def test_defaults_to_600s(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", raising=False)
        assert responses_state.stream_buffer_ttl_seconds() == 600.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", "30")
        assert responses_state.stream_buffer_ttl_seconds() == 30.0

    def test_non_positive_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", "0")
        assert responses_state.stream_buffer_ttl_seconds() == 600.0

    def test_bad_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", "soon")
        assert responses_state.stream_buffer_ttl_seconds() == 600.0


class TestTtl:
    def test_defaults_to_30_days(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RESPONSES_TTL_S", raising=False)
        assert responses_state.ttl_seconds() == 30 * 24 * 60 * 60

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "60")
        assert responses_state.ttl_seconds() == 60

    def test_non_positive_disables_expiry(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "0")
        assert responses_state.ttl_seconds() is None

    def test_bad_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "soon")
        assert responses_state.ttl_seconds() == 30 * 24 * 60 * 60

    @pytest.mark.asyncio
    async def test_snapshot_expires(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "10")
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        await responses_state.write_async(store, "u1", "resp_1", response=_response(), input_items=[])
        assert await responses_state.read_async(store, "u1", "resp_1") is not None
        clock["t"] = 1011.0
        assert await responses_state.read_async(store, "u1", "resp_1") is None
