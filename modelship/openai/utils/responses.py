"""Utility helpers for ``/v1/responses``: loader-side output shaping plus
gateway-side conversation-state plumbing (history, persistence, snapshots)."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException
from pydantic import ValidationError
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

from modelship.infer.infer_config import get_disconnect_registry
from modelship.logging import get_logger
from modelship.openai import compaction_crypto
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    ResponseObject,
    ResponsesRequest,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.base import random_uuid
from modelship.openai.protocol.responses.adapter import (
    _status_for,
    _usage_from_chat,
    build_response_object,
    messages_from_input,
)
from modelship.openai.protocol.responses.schemas import (
    CompactionItem,
    CompactResource,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummary,
)
from modelship.openai.protocol.responses.streaming import TERMINAL_EVENT_TYPES, store_failure_event
from modelship.openai.state import responses as responses_state
from modelship.openai.utils.chat import ParsedChatOutput
from modelship.state import StateStore, StateStoreUnavailableError

logger = get_logger("openai.utils.responses")

# Shape of the response ids we mint (`resp_<uuid>`). Ids arriving from a client are
# checked against it before becoming a state-store key segment, so a malformed id is
# a clean 404 rather than a lookup for something we could never have written.
RESPONSE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# --- Shaping: parsed chat output -> Responses items (loader-side) ---


def build_responses_items_from_parsed(parsed: ParsedChatOutput) -> list[ResponseOutputItem]:
    """Shape one parsed choice into Responses ``output[]`` items: reasoning, then
    message, then one ``function_call`` per tool call."""
    output: list[ResponseOutputItem] = []
    if parsed.reasoning:
        output.append(ResponseReasoningItem(summary=[ResponseReasoningSummary(text=parsed.reasoning)]))
    if parsed.content:
        output.append(ResponseOutputMessage(content=[ResponseOutputText(text=parsed.content)]))
    for call in parsed.tool_calls:
        output.append(
            ResponseFunctionToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
        )
    return output


def build_response_from_parsed(
    parsed: ParsedChatOutput,
    request: ResponsesRequest,
    *,
    usage: UsageInfo,
    finish_reason: str | None,
    model: str,
) -> ResponseObject:
    """Build a non-streaming ``ResponseObject`` from one loader's parsed chat output.
    Shared status/usage/envelope assembly for every loader's non-streaming `create_response`."""
    status, incomplete = _status_for(finish_reason)
    return build_response_object(
        request,
        status=status,
        output=build_responses_items_from_parsed(parsed),
        usage=_usage_from_chat(usage),
        incomplete=incomplete,
        model=model,
        completed_at=int(time.time()) if status == "completed" else None,
        background=bool(request.background),
    )


# Structured rather than a bare "summarize this" so a continuation loses nothing essential.
_COMPACTION_SYSTEM_PROMPT = (
    "Summarize this conversation so it can be continued from a fresh context window "
    "with nothing essential lost. Structure the summary with these sections, in order: "
    "(1) the user's explicit requests and intent, verbatim where it matters; "
    "(2) key facts, decisions, names, and identifiers established so far; "
    "(3) specific file paths, code, or command output already produced, in enough "
    "detail to avoid re-deriving them; "
    "(4) errors encountered and how they were fixed, including any correction the "
    "user gave about how to approach the task; "
    "(5) work explicitly still pending; "
    "(6) exactly what was being worked on immediately before this summary, so the "
    "next turn picks up from there rather than guessing."
)


def build_summarization_request(model: str, items: list[Any], instructions: str | None = None) -> ChatCompletionRequest:
    """The internal chat request ``/v1/responses/compact`` issues to summarize *items*.
    *instructions*, if given, is inserted as an extra system message."""
    messages = messages_from_input(items, None)
    messages.insert(0, {"role": "system", "content": _COMPACTION_SYSTEM_PROMPT})
    if instructions:
        messages.insert(1, {"role": "system", "content": instructions})
    return ChatCompletionRequest(model=model, messages=messages, stream=False)


def build_compaction(*, summary_items: list[Any], usage: UsageInfo) -> CompactResource:
    """Build a ``CompactResource`` from a ``/v1/responses/compact`` summarization call.
    ``id``/``created_at`` are freshly minted; a compaction result is never persisted."""
    encrypted_content = compaction_crypto.encrypt_items(summary_items)
    return CompactResource(output=[CompactionItem(encrypted_content=encrypted_content)], usage=_usage_from_chat(usage))


def responses_validation_error(exc: ValidationError) -> ErrorResponse:
    """400 for a pydantic ``ValidationError`` surfaced by ``responses_request_to_chat``,
    same shape as every other rejection."""
    return create_error_response(message=str(exc), err_type="invalid_request_error")


# --- Gateway-side conversation-state plumbing ---


class ResponsesApiError(HTTPException):
    """An ``HTTPException`` that also carries a full OpenAI-shaped ``ErrorResponse``,
    so one raise renders correctly for both the HTTP route and the WS turn-runner."""

    def __init__(self, err: ErrorResponse):
        self.err = err
        super().__init__(status_code=err._http_status, detail=err.error.message)


def _not_found_error(response_id: str, *, previous: bool = False) -> ResponsesApiError:
    if previous:
        # "previous_response_not_found" is OpenAI's actual code for this failure.
        return ResponsesApiError(
            create_error_response(
                f"Previous response with id '{response_id}' not found.",
                err_type="invalid_request_error",
                status_code=HTTPStatus.NOT_FOUND,
                param="previous_response_id",
                code="previous_response_not_found",
            )
        )
    return ResponsesApiError(
        create_error_response(
            f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
        )
    )


def _store_unavailable_error() -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            "Conversation state store is unavailable; retry shortly.",
            err_type="api_error",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    )


def _previous_in_progress_error(response_id: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            f"Previous response with id '{response_id}' is still in progress.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
            param="previous_response_id",
        )
    )


def background_stream_error() -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            "background:true cannot be combined with stream:true over HTTP "
            "(background streaming is not yet supported).",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
            param="background",
        )
    )


def background_store_false_error() -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            "background:true requires store:true; a background response can't be polled if it isn't stored.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
            param="background",
        )
    )


def _not_background_error(response_id: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            f"Response with id '{response_id}' is not a background response.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    )


def _mark_failed(response: dict, message: str) -> dict:
    return {**response, "status": "failed", "error": {"message": message}}


def as_input_items(input_: str | list[Any]) -> list[Any]:
    """Normalize a Responses ``input`` to item form, so stored history and this turn's
    input concatenate."""
    if isinstance(input_, str):
        return [{"type": "message", "role": "user", "content": input_}]
    return list(input_)


async def resolve_history_items(
    store: StateStore, identity: str, *, previous_response_id: str | None, input_: str | list[Any] | None
) -> list[Any]:
    """Prepend the conversation stored under ``previous_response_id`` to this turn's
    input. 404 if unknown, 503 if the store is unreachable."""
    this_turn = as_input_items(input_) if input_ is not None else []
    if previous_response_id is None:
        return this_turn
    if not RESPONSE_ID_RE.match(previous_response_id):
        raise _not_found_error(previous_response_id, previous=True)
    try:
        snapshot = await responses_state.read_async(store, identity, previous_response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable resolving previous_response_id=%s", previous_response_id)
        raise _store_unavailable_error() from None
    if snapshot is None:
        raise _not_found_error(previous_response_id, previous=True)
    status = (snapshot.get("response") or {}).get("status")
    if status in ("queued", "in_progress"):
        # No output yet to continue from.
        raise _previous_in_progress_error(previous_response_id)
    return [*responses_state.history_items(snapshot), *this_turn]


async def resolve_history(store: StateStore, identity: str, request: ResponsesRequest) -> list[Any]:
    """``resolve_history_items`` for a ``ResponsesRequest``."""
    return await resolve_history_items(
        store, identity, previous_response_id=request.previous_response_id, input_=request.input
    )


async def persist_response(gen, store: StateStore, *, identity: str, input_items: list[Any]):
    """Tee `respond`'s output, storing the snapshot as it passes. Covers both
    streaming (terminal event dict) and non-streaming (`ResponseObject` is the
    whole body); persists before yielding so a store failure changes what the
    client is told."""
    async for item in gen:
        if isinstance(item, ResponseObject):
            try:
                await responses_state.write_async(
                    store, identity, item.id, response=item.model_dump(mode="json"), input_items=input_items
                )
            except StateStoreUnavailableError:
                logger.exception("State store unavailable persisting response %s", item.id)
                yield create_error_response(
                    "Conversation state store is unavailable; the response was generated but not stored.",
                    err_type="api_error",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            yield item
            continue

        if not isinstance(item, dict) or item.get("type") not in TERMINAL_EVENT_TYPES:
            yield item
            continue

        response = item.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(response, dict) or not response_id:
            logger.warning("Terminal Responses event has no usable response id; not storing.")
            yield item
            continue

        try:
            await responses_state.write_async(store, identity, response_id, response=response, input_items=input_items)
        except StateStoreUnavailableError:
            logger.exception("State store unavailable persisting streamed response %s", response.get("id"))
            yield store_failure_event(
                item, "Conversation state store is unavailable; the response was generated but not stored."
            )
            return
        yield item


async def load_snapshot(store: StateStore, identity: str, response_id: str) -> dict:
    """The stored snapshot for *response_id*, scoped to the caller's identity."""
    if not RESPONSE_ID_RE.match(response_id):
        raise _not_found_error(response_id)
    try:
        snapshot = await responses_state.read_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reading response %s", response_id)
        raise _store_unavailable_error() from None
    if snapshot is None:
        raise _not_found_error(response_id)
    return snapshot


async def delete_snapshot(store: StateStore, identity: str, response_id: str) -> None:
    """Delete the snapshot for *response_id*. Caller must confirm existence first
    via :func:`load_snapshot`."""
    try:
        await responses_state.delete_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable deleting response %s", response_id)
        raise _store_unavailable_error() from None


# --- Background mode (background:true) ---

# Drain task's heartbeat cadence; state/responses.py's staleness threshold is a multiple of this.
_HEARTBEAT_INTERVAL_S = 5.0


async def start_background(
    store: StateStore, identity: str, request: ResponsesRequest, req_id: str, input_items: list[Any]
) -> dict:
    """Mint a response id, persist the initial `queued` placeholder, and return
    the serialized `ResponseObject` to send back to the client immediately."""
    response_id = f"resp_{random_uuid()}"
    response = build_response_object(
        request,
        status="queued",
        output=[],
        usage=None,
        incomplete=None,
        response_id=response_id,
        created_at=int(time.time()),
        background=True,
    )
    response_dict = response.model_dump(mode="json")
    try:
        await responses_state.write_background(
            store, identity, response_id, response=response_dict, input_items=input_items, req_id=req_id
        )
    except StateStoreUnavailableError:
        logger.exception("State store unavailable creating background response %s", response_id)
        raise _store_unavailable_error() from None
    return response_dict


async def run_background(
    handle: DeploymentHandle,
    request: ResponsesRequest,
    headers: dict[str, str],
    req_id: str,
    identity: str,
    *,
    store: StateStore,
    response_id: str,
    input_items: list[Any],
) -> None:
    """Detached-task body for `background:true`: drives the deployment's streaming
    Responses generator to completion, persisting every terminal transition. Never
    raises — every exit path ends in an already-terminal snapshot or a stored
    `failed` one. Always runs the inner call with `stream=True` regardless of the
    client's own request, so there are real transitions to persist."""
    request.stream = True
    registry = get_disconnect_registry()
    response_gen = cast(
        "DeploymentResponseGenerator[Any]",
        handle.respond.options(stream=True).remote(request, headers, registry, req_id, identity, response_id),
    )
    piped = persist_response(response_gen, store, identity=identity, input_items=input_items)

    drain_task = asyncio.ensure_future(_drain_background_stream(piped, response_id))
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop(store, identity, response_id))
    try:
        done, _pending = await asyncio.wait({drain_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED)
        if drain_task not in done:
            # Heartbeat found the snapshot cancelled/deleted: stop draining, no further write.
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
            response_gen.cancel()  # best-effort abort; registry signal above is the primary path
            return
        saw_ok_terminal, failed_response = drain_task.result()
    except Exception:
        logger.exception("background response %s: drain task raised unexpectedly", response_id)
        saw_ok_terminal, failed_response = False, None
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

    if failed_response is not None:
        # persist_response doesn't persist response.failed events; the poller needs it stored.
        await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=failed_response)
    elif not saw_ok_terminal:
        # Stream ended with no terminal event (cancel race or actor crash) — don't leave a poller hanging.
        snapshot = await responses_state.read_async(store, identity, response_id)
        if snapshot is not None:
            failed = _mark_failed(
                snapshot.get("response") or {}, "Generation ended unexpectedly without a terminal event."
            )
            await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=failed)


async def _drain_background_stream(piped, response_id: str) -> tuple[bool, dict | None]:
    """Consume *piped* (the `persist_response`-wrapped event stream) to completion.
    Returns `(saw_ok_terminal, failed_response)` — `failed_response` carries a
    loader-side `response.failed` payload, which `persist_response` doesn't persist itself."""
    saw_ok_terminal = False
    failed_response: dict | None = None
    try:
        async for item in piped:
            if not isinstance(item, dict):
                continue  # persist_response's own terminal-write-failed signal; nothing new to persist
            event_type = item.get("type")
            if event_type in TERMINAL_EVENT_TYPES:
                saw_ok_terminal = True
            elif event_type == "response.failed":
                failed_response = item.get("response")
    except Exception:
        logger.exception("background response %s: event stream raised mid-drain", response_id)
    return saw_ok_terminal, failed_response


async def _heartbeat_loop(store: StateStore, identity: str, response_id: str) -> None:
    """Refresh the drain task's heartbeat every `_HEARTBEAT_INTERVAL_S` until `touch`
    reports the run is no longer heartbeat-owned (cancelled or deleted)."""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        still_running = await responses_state.touch(store, identity, response_id)
        if not still_running:
            return


async def cancel_background(store: StateStore, identity: str, response_id: str) -> dict:
    """Cancel an in-flight background response: signal `DisconnectRegistry` with the
    run's request id, then mark the snapshot `cancelled`. Idempotent on an already-
    terminal response; 400s if it was never a background run."""
    snapshot = await load_snapshot(store, identity, response_id)
    response = snapshot["response"]
    if not response.get("background"):
        raise _not_background_error(response_id)
    if response.get("status") in responses_state.TERMINAL_STATUSES:
        return response

    mship = snapshot.get("_mship")
    if isinstance(mship, dict) and mship.get("req_id"):
        registry = get_disconnect_registry()
        with contextlib.suppress(Exception):
            await registry.set.remote(mship["req_id"])

    cancelled = {**response, "status": "cancelled", "completed_at": int(time.time())}
    await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=cancelled)
    updated = await load_snapshot(store, identity, response_id)
    return updated["response"]


async def signal_background_cancel_if_in_progress(snapshot: dict) -> None:
    """`DELETE /{id}` on an in-flight background run implies cancel: signal the
    registry (same mechanism `cancel_background` uses) before the caller deletes the
    snapshot, so the drain task's next heartbeat tick tears the run down."""
    response = snapshot.get("response") or {}
    if not response.get("background") or response.get("status") in responses_state.TERMINAL_STATUSES:
        return
    mship = snapshot.get("_mship")
    if isinstance(mship, dict) and mship.get("req_id"):
        registry = get_disconnect_registry()
        with contextlib.suppress(Exception):
            await registry.set.remote(mship["req_id"])


async def reconcile_staleness(store: StateStore, identity: str, response_id: str, snapshot: dict) -> dict:
    """`GET`-time orphan detection: if *snapshot* is a non-terminal background run
    with a stale heartbeat, transition it to `failed` and persist. No-op otherwise."""
    response = snapshot.get("response") or {}
    if response.get("status") not in ("queued", "in_progress"):
        return snapshot
    if not responses_state.is_stale(snapshot, responses_state.stale_seconds()):
        return snapshot
    failed = _mark_failed(response, "Background response orphaned: its worker stopped reporting progress.")
    await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=failed)
    updated = await responses_state.read_async(store, identity, response_id)
    return updated if updated is not None else snapshot
