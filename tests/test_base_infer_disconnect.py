"""Unit tests for BaseInfer.run_cancellable / run_cancellable_stream, which race work
against the shared disconnect pump and call on_generation_aborted() on cancellation."""

import asyncio
import itertools
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from modelship.infer import infer_config
from modelship.infer.base_infer import _DISCONNECT_POLL_INTERVAL_S, BaseInfer, ClientDisconnectedError

_request_id_counter = itertools.count()


class _FakeRawRequest:
    """Registers itself in a class-level id -> instance map so `_Infer`'s overridden
    `_poll_disconnected_ids` can look it up by id, mirroring the real DisconnectRegistry."""

    _by_id: ClassVar[dict[str, "_FakeRawRequest"]] = {}

    def __init__(self, *, disconnect_after: float | None):
        self._disconnect_after = disconnect_after
        self._start = asyncio.get_event_loop().time()
        self.request_id = f"req-{next(_request_id_counter)}"
        self.is_watchable = True
        _FakeRawRequest._by_id[self.request_id] = self

    async def is_disconnected(self) -> bool:
        if self._disconnect_after is None:
            return False
        return asyncio.get_event_loop().time() - self._start >= self._disconnect_after


class _Infer(BaseInfer):
    """Minimal concrete BaseInfer — only run_cancellable/on_generation_aborted are exercised."""

    def __init__(self):
        super().__init__(MagicMock())
        self.aborted = False

    def shutdown(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def warmup(self) -> None:
        pass

    async def on_generation_aborted(self) -> None:
        self.aborted = True

    @staticmethod
    async def _poll_disconnected_ids(request_ids: list[str]) -> list[str]:
        return [rid for rid in request_ids if await _FakeRawRequest._by_id[rid].is_disconnected()]


@pytest.mark.asyncio
async def test_poll_disconnected_ids_degrades_on_unexpected_registry_exception(monkeypatch):
    """_poll_disconnected_ids must swallow more than RayActorError: an unhandled exception
    here would kill disconnect detection for every concurrent request on the replica."""

    class _BrokenRemote:
        def remote(self, request_ids):
            raise TypeError("boom")

    class _BrokenRegistry:
        is_set_many = _BrokenRemote()

    monkeypatch.setattr(infer_config, "get_disconnect_registry", lambda: _BrokenRegistry())

    result = await BaseInfer._poll_disconnected_ids(["req-x"])

    assert result == []


@pytest.mark.asyncio
async def test_work_finishing_first_returns_result_without_aborting():
    async def fast_work():
        await asyncio.sleep(0.01)
        return "done"

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)  # client never disconnects

    result = await infer.run_cancellable(fast_work(), raw_request)

    assert result == "done"
    assert infer.aborted is False


@pytest.mark.asyncio
async def test_disconnect_cancels_work_and_calls_abort_hook():
    cancelled = asyncio.Event()

    async def slow_work():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=0.0)  # disconnected immediately

    with pytest.raises(ClientDisconnectedError):
        await infer.run_cancellable(slow_work(), raw_request)

    assert cancelled.is_set()
    assert infer.aborted is True


@pytest.mark.asyncio
async def test_outer_cancellation_during_wait_cancels_work_without_leaking():
    """If the task driving run_cancellable is cancelled from outside while suspended in
    asyncio.wait, `work` must be cancelled too, not left running unobserved."""
    work_cancelled = asyncio.Event()

    async def slow_work():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            work_cancelled.set()
            raise

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)  # never disconnects on its own

    outer = asyncio.ensure_future(infer.run_cancellable(slow_work(), raw_request))
    await asyncio.sleep(0.01)  # let it enter asyncio.wait
    outer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await outer

    assert work_cancelled.is_set()


@pytest.mark.asyncio
async def test_work_exception_propagates_without_aborting():
    async def failing_work():
        raise ValueError("boom")

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    with pytest.raises(ValueError, match="boom"):
        await infer.run_cancellable(failing_work(), raw_request)

    assert infer.aborted is False


@pytest.mark.asyncio
async def test_disconnect_poll_interval_does_not_starve_fast_work():
    """A slow poll interval must not delay returning once work finishes —
    run_cancellable awaits asyncio.wait with FIRST_COMPLETED, not the poll loop."""

    async def fast_work():
        return 42

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    start = asyncio.get_event_loop().time()
    result = await infer.run_cancellable(fast_work(), raw_request)
    elapsed = asyncio.get_event_loop().time() - start

    assert result == 42
    assert elapsed < _DISCONNECT_POLL_INTERVAL_S


@pytest.mark.asyncio
async def test_stream_yields_items_without_aborting():
    async def gen():
        for i in range(3):
            yield i

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    items = [item async for item in infer.run_cancellable_stream(gen(), raw_request)]

    assert items == [0, 1, 2]
    assert infer.aborted is False


@pytest.mark.asyncio
async def test_stream_disconnect_cancels_work_and_calls_abort_hook():
    cancelled = asyncio.Event()

    async def gen():
        yield "first"
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield "unreachable"  # pragma: no cover - never reached

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=0.05)

    received = []
    with pytest.raises(ClientDisconnectedError):
        async for item in infer.run_cancellable_stream(gen(), raw_request):
            received.append(item)

    assert received == ["first"]
    assert cancelled.is_set()
    assert infer.aborted is True


@pytest.mark.asyncio
async def test_stream_exception_propagates_without_aborting():
    async def gen():
        yield "first"
        raise ValueError("boom")

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    received = []
    with pytest.raises(ValueError, match="boom"):
        async for item in infer.run_cancellable_stream(gen(), raw_request):
            received.append(item)

    assert received == ["first"]
    assert infer.aborted is False


@pytest.mark.asyncio
async def test_stream_disconnect_poll_interval_does_not_starve_fast_stream():
    """A slow poll interval must not delay a fast stream — each `__anext__()`
    races `asyncio.wait` with FIRST_COMPLETED, not the poll loop."""

    async def gen():
        for i in range(5):
            yield i

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    start = asyncio.get_event_loop().time()
    items = [item async for item in infer.run_cancellable_stream(gen(), raw_request)]
    elapsed = asyncio.get_event_loop().time() - start

    assert items == [0, 1, 2, 3, 4]
    assert elapsed < _DISCONNECT_POLL_INTERVAL_S


@pytest.mark.asyncio
async def test_consumer_closing_outer_generator_early_closes_work():
    """If the consumer stops iterating (e.g. Starlette aclose()s mid-stream) before any
    disconnect is detected, `work` must still be closed, not left for GC to finalize."""
    work_closed = asyncio.Event()

    async def gen():
        try:
            for i in range(5):
                yield i
                await asyncio.sleep(10)
        finally:
            work_closed.set()

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    outer = infer.run_cancellable_stream(gen(), raw_request)
    first = await outer.__anext__()
    assert first == 0

    await outer.aclose()

    assert work_closed.is_set()
    assert infer.aborted is False


@pytest.mark.asyncio
async def test_task_cancelled_while_suspended_in_asyncio_wait_does_not_raise():
    """If cancelled while suspended in asyncio.wait (not at yield), `next_item` still owns
    `work`'s frame — aclose() before awaiting it raises "already running", not CancelledError."""
    work_closed = asyncio.Event()

    async def gen():
        try:
            yield "first"
            await asyncio.sleep(10)
            yield "unreachable"  # pragma: no cover - never reached
        finally:
            work_closed.set()

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    outer = infer.run_cancellable_stream(gen(), raw_request)
    first = await outer.__anext__()
    assert first == "first"

    task = asyncio.ensure_future(outer.__anext__())
    await asyncio.sleep(0.01)  # let it create next_item and suspend in asyncio.wait
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert work_closed.is_set()
