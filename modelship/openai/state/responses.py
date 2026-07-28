"""Conversation state for ``/v1/responses`` — one snapshot per response id.

A stored snapshot is **self-contained**: it holds the full conversation as of that
response, so continuing from a ``previous_response_id`` is a single read (O(1))
rather than a walk back down a chain of pointers. Each turn mints a fresh response
id and therefore a fresh key, which is also what makes branching work — two
requests may continue from the same ``previous_response_id`` without colliding.

The cost of that shape: snapshot *N* embeds turns 1..*N*, so a conversation of *n*
turns costs O(n²) total storage. Deliberate — reads happen every turn, and TTL
bounds the total.

Keys are scoped by caller identity (``responses/<identity>/<response_id>``), never
by response id alone: a bare id would let any caller fetch another's conversation
by guessing or replaying one. A read for the wrong identity simply builds a
different key and misses, so isolation needs no comparison logic.

This is the OpenAI-domain layer over the generic ``modelship.state`` store: it takes
a store and never builds one, exactly as ``deploy.effective_config`` is the deploy
domain's layer over the same store. The store stays generic and knows nothing about
Responses.
"""

from __future__ import annotations

import os
import time

from modelship.logging import get_logger
from modelship.state import StateStore

logger = get_logger("api")

# State-store namespace; one key per response: "responses/<identity>/<response_id>".
_NAMESPACE = "responses"

# How long a stored conversation lives. Each turn writes a new key with a fresh TTL,
# so an active conversation stays alive while superseded snapshots age out.
_TTL_ENV = "MSHIP_RESPONSES_TTL_S"
_DEFAULT_TTL_S = 30 * 24 * 60 * 60.0  # 30 days, matching OpenAI's retention

# A background response's status once it can no longer change.
TERMINAL_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled"})

# How stale a background run's heartbeat (`_mship.updated_at`) may get before a
# poller gives up on it and reports `failed`.
_STALE_ENV = "MSHIP_RESPONSES_STALE_S"
_DEFAULT_STALE_S = 30.0


def ttl_seconds() -> float | None:
    """Configured conversation TTL; ``None`` (no expiry) when set to <= 0."""
    raw = os.environ.get(_TTL_ENV)
    if not raw:
        return _DEFAULT_TTL_S
    try:
        ttl = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; falling back to %ss.", _TTL_ENV, raw, _DEFAULT_TTL_S)
        return _DEFAULT_TTL_S
    return ttl if ttl > 0 else None


def stale_seconds() -> float:
    """Configured heartbeat-staleness threshold for orphan detection (always positive)."""
    raw = os.environ.get(_STALE_ENV)
    if not raw:
        return _DEFAULT_STALE_S
    try:
        threshold = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; falling back to %ss.", _STALE_ENV, raw, _DEFAULT_STALE_S)
        return _DEFAULT_STALE_S
    return threshold if threshold > 0 else _DEFAULT_STALE_S


def _key(identity: str, response_id: str) -> str:
    return f"{_NAMESPACE}/{identity}/{response_id}"


def read(store: StateStore, identity: str, response_id: str) -> dict | None:
    """Return the snapshot for *response_id* under *identity*, or ``None`` if absent.

    ``StateStoreUnavailableError`` propagates: a store outage must surface as a 503,
    never as a 404 that would look like a legitimately unknown id.
    """
    data = store.get(_key(identity, response_id))
    if data is None:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("response"), dict):
        logger.warning("Malformed response snapshot at %r; treating as missing.", _key(identity, response_id))
        return None
    return data


async def read_async(store: StateStore, identity: str, response_id: str) -> dict | None:
    """Async :func:`read` — same contract."""
    data = await store.get_async(_key(identity, response_id))
    if data is None:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("response"), dict):
        logger.warning("Malformed response snapshot at %r; treating as missing.", _key(identity, response_id))
        return None
    return data


async def write_async(
    store: StateStore,
    identity: str,
    response_id: str,
    *,
    response: dict,
    input_items: list[dict],
) -> None:
    """Persist the snapshot for *response_id*.

    *response* is the full serialized ``ResponseObject`` so ``GET`` can return it
    verbatim; *input_items* is everything that went in (resolved history + this
    turn's input), so the next turn rebuilds by appending this response's output.
    """
    await store.set_async(
        _key(identity, response_id),
        {"response": response, "input_items": input_items},
        ttl_seconds=ttl_seconds(),
    )


async def delete_async(store: StateStore, identity: str, response_id: str) -> None:
    """Drop the snapshot for *response_id*. Idempotent (per the store contract)."""
    await store.delete_async(_key(identity, response_id))


async def write_background(
    store: StateStore,
    identity: str,
    response_id: str,
    *,
    response: dict,
    input_items: list[dict],
    req_id: str,
) -> None:
    """Persist the initial ``queued`` placeholder with the ``_mship`` sidecar: ``req_id``
    (what cancel signals in ``DisconnectRegistry``) and ``updated_at`` (heartbeat).
    Dropped at terminal status — see :func:`write_terminal_if_not_terminal`."""
    await store.set_async(
        _key(identity, response_id),
        {
            "response": response,
            "input_items": input_items,
            "_mship": {"req_id": req_id, "updated_at": time.time()},
        },
        ttl_seconds=ttl_seconds(),
    )


async def touch(store: StateStore, identity: str, response_id: str) -> bool:
    """Heartbeat tick: refresh ``_mship.updated_at``. Returns ``False`` (stop
    heartbeating) if the snapshot is gone or no longer carries ``_mship``."""
    snapshot = await read_async(store, identity, response_id)
    if snapshot is None:
        return False
    mship = snapshot.get("_mship")
    if not isinstance(mship, dict):
        return False
    mship["updated_at"] = time.time()
    await store.set_async(_key(identity, response_id), snapshot, ttl_seconds=ttl_seconds())
    return True


def is_stale(snapshot: dict, threshold_s: float) -> bool:
    """Whether *snapshot*'s heartbeat is older than *threshold_s*. ``False`` for a
    snapshot with no ``_mship`` sidecar (not a live background run)."""
    mship = snapshot.get("_mship")
    if not isinstance(mship, dict):
        return False
    updated_at = mship.get("updated_at")
    if not isinstance(updated_at, int | float):
        return False
    return (time.time() - updated_at) > threshold_s


async def write_terminal_if_not_terminal(store: StateStore, identity: str, response_id: str, *, response: dict) -> None:
    """Write *response* as the snapshot's new state, unless it's already gone or
    already terminal. Drops the ``_mship`` sidecar. Shared by every writer that could
    otherwise regress a terminal status (cancel, drain failure, staleness check)."""
    snapshot = await read_async(store, identity, response_id)
    if snapshot is None:
        return
    current_status = (snapshot.get("response") or {}).get("status")
    if current_status in TERMINAL_STATUSES:
        return
    await store.set_async(
        _key(identity, response_id),
        {"response": response, "input_items": snapshot.get("input_items") or []},
        ttl_seconds=ttl_seconds(),
    )


def history_items(snapshot: dict) -> list[dict]:
    """Rebuild the conversation from a snapshot: everything that went into that turn,
    plus what it produced. This is what a continuation prepends to its own input."""
    items = snapshot.get("input_items")
    output = (snapshot.get("response") or {}).get("output")
    return [
        *(items if isinstance(items, list) else []),
        *(output if isinstance(output, list) else []),
    ]
