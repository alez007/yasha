"""Redis-backed StateStore — one JSON value per key.

Durable across cluster / head death (the value lives in Redis, not the actor), so
it's the backend for the coordinator registry and effective config in k8s, where
the same Redis also backs Ray's GCS fault tolerance. Selected by the ``redis://``
(or ``rediss://`` for TLS) URI scheme; the URL carries host/port/db/user/password,
parsed natively by ``from_url`` — so a password may be inlined (``redis://:pw@host``)
or injected by the deployment (e.g. a k8s Secret expanded into the URL).

Two clients are held, created lazily: a sync ``redis.Redis`` for the sync methods
and a native ``redis.asyncio.Redis`` for the ``*_async`` ones — so an event-loop
caller gets true async I/O, not a thread hop. TTL uses Redis's native expiry (no
value envelope needed).
"""

from __future__ import annotations

import contextlib
import json
from typing import cast

import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from modelship.logging import get_logger
from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError, normalize_prefix

logger = get_logger("startup")

_PREFIX = "modelship/state/"


def _slug(key: str) -> str:
    # Collapse the namespace path into one Redis key under a shared prefix, so
    # backends stay swappable by URI alone.
    return _PREFIX + "/".join(p for p in key.split("/") if p)


def _in_namespace(key: str, prefix: str) -> bool:
    # SCAN's MATCH glob has no path-segment concept, so "prefix*" also matches a
    # sibling like "responses/u10" under prefix "responses/u1"; re-check the boundary.
    return not prefix or key == prefix or key.startswith(f"{prefix}/")


def _decode(raw: str | bytes | None) -> JsonValue | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("Corrupt JSON in redis; treating as missing.")
        return None


@contextlib.contextmanager
def _mapped(what: str):
    """Map a Redis connectivity failure to StateStoreUnavailableError (an outage, not a
    missing key). Wraps both sync calls and awaited async calls."""
    try:
        yield
    except (RedisConnectionError, RedisTimeoutError) as exc:
        raise StateStoreUnavailableError(what) from exc


class RedisStateStore(StateStore):
    def __init__(self, url: str) -> None:
        self._url = url
        self._sync_client: redis.Redis | None = None
        self._async_client: aioredis.Redis | None = None

    def _sync(self) -> redis.Redis:
        if self._sync_client is None:
            # decode_responses so get() returns str, not bytes, for json.loads.
            self._sync_client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._sync_client

    def _async(self) -> aioredis.Redis:
        if self._async_client is None:
            self._async_client = aioredis.Redis.from_url(self._url, decode_responses=True)
        return self._async_client

    @staticmethod
    def _px(ttl_seconds: float | None) -> int | None:
        # Native expiry in ms; floor at 1ms so a sub-second TTL never becomes a
        # no-expiry / rejected ``ex=0``.
        return None if ttl_seconds is None else max(1, int(ttl_seconds * 1000))

    def get(self, key: str) -> JsonValue | None:
        with _mapped(f"redis get {key!r}"):
            raw = self._sync().get(_slug(key))
        return _decode(raw)

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        with _mapped(f"redis set {key!r}"):
            self._sync().set(_slug(key), json.dumps(value), px=self._px(ttl_seconds))

    def delete(self, key: str) -> None:
        with _mapped(f"redis delete {key!r}"):
            self._sync().delete(_slug(key))

    def list(self, prefix: str) -> list[str]:
        prefix = normalize_prefix(prefix)
        match = _slug(prefix) + "*"
        with _mapped(f"redis scan {prefix!r}"):
            keys = list(self._sync().scan_iter(match=match))
        return [k[len(_PREFIX) :] for k in keys if _in_namespace(k[len(_PREFIX) :], prefix)]

    async def get_async(self, key: str) -> JsonValue | None:
        with _mapped(f"redis get {key!r}"):
            raw = await self._async().get(_slug(key))
        return _decode(raw)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        with _mapped(f"redis set {key!r}"):
            await self._async().set(_slug(key), json.dumps(value), px=self._px(ttl_seconds))

    async def delete_async(self, key: str) -> None:
        with _mapped(f"redis delete {key!r}"):
            await self._async().delete(_slug(key))

    async def list_async(self, prefix: str) -> list[str]:
        prefix = normalize_prefix(prefix)
        match = _slug(prefix) + "*"
        keys = []
        with _mapped(f"redis scan {prefix!r}"):
            async for k in self._async().scan_iter(match=match):
                keys.append(k)
        return [k[len(_PREFIX) :] for k in keys if _in_namespace(k[len(_PREFIX) :], prefix)]

    async def append_async(
        self, key: str, event: JsonValue, *, ttl_seconds: float | None = None, max_len: int | None = None
    ) -> None:
        # Redis Streams give a native O(1) append. Entry ids are derived from the
        # event's own sequence_number (1-based: id "0-0" is reserved by Redis) so
        # read_from_async's range query maps directly onto the same cursor callers
        # already use (ResponsesStreamTranslator's sequence_number).
        seq = event.get("sequence_number") if isinstance(event, dict) else None
        entry_id = f"{int(seq) + 1}-0" if isinstance(seq, int) else "*"
        maxlen_kwargs = {"maxlen": max_len, "approximate": True} if max_len is not None else {}
        with _mapped(f"redis xadd {key!r}"):
            await self._async().xadd(_slug(key), {"data": json.dumps(event)}, id=entry_id, **maxlen_kwargs)
            if ttl_seconds is not None:
                await self._async().pexpire(_slug(key), cast(int, self._px(ttl_seconds)))

    async def read_from_async(self, key: str, *, after_sequence: int = -1) -> list[JsonValue]:
        start_id = f"({after_sequence + 1}-0"
        with _mapped(f"redis xrange {key!r}"):
            entries = await self._async().xrange(_slug(key), min=start_id, max="+")
        events = []
        for _entry_id, fields in entries or []:
            decoded = _decode(fields.get("data")) if isinstance(fields, dict) else None
            if decoded is not None:
                events.append(decoded)
        return events
