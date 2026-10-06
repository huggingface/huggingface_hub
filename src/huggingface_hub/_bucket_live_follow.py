# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Live-following of file changes in a bucket.

The Hub exposes a server-sent events feed of per-file changes for buckets:

```
GET /api/buckets/{bucket_id}/events?cursor=<opaque> | ?since=<ISO8601>
Accept: text/event-stream
```

A consumer can keep its view of a bucket fresh by applying the batches as they happen instead of re-listing
directories. This module implements the client side of that feed:

- [`parse_follow_event`] turns a raw SSE event into a typed event. Unknown or malformed events are dropped so
  the Hub can add event types without breaking older clients.
- [`BucketFollower`] is a daemon thread owning one (re)connecting stream for one bucket. It maintains the
  resume point, reconnects with exponential backoff, and hands the batches over to callbacks provided by the
  caller (see [`HfFileSystem`]).

The feed is best-effort by design: when it cannot be followed (deployment where live-follow is disabled, token
without access to the bucket), the follower stops and the consumer is left with its default behavior.

Event vocabulary:

| event       | payload                                | meaning                                                               |
| ----------- | -------------------------------------- | --------------------------------------------------------------------- |
| `ready`     | `{cursor?}`                            | replay is done, the stream now follows live changes                    |
| `changes`   | `{cursor, changes: [{path, op, ...}]}` | batch of per-file changes; `cursor` resumes strictly after the batch   |
| `reset`     | `{reason}`                             | resume point older than the replay buffer: reconcile and re-subscribe  |
| `reconnect` | `{cursor?}`                            | server-directed end of stream (pod rotation, shutdown,...)             |
"""

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union

import httpx2

from . import logging
from ._hot_reload.sse_client import SSEClient
from .utils import build_hf_headers, get_session, hf_raise_for_status, parse_datetime


logger = logging.get_logger(__name__)


STREAM_READ_TIMEOUT = 90.0
"""Seconds without a single byte (the server pings every ~30s) before a stream is considered dead."""

CONNECT_TIMEOUT = 10.0
RETRY_BASE_DELAY = 10.0
RETRY_MAX_DELAY = 80.0
RECONNECT_MIN_DELAY = 1.0
MAX_SESSIONS_WITHOUT_READY = 3

# Statuses meaning the feed cannot be followed for this bucket, as opposed to "retry later": not served by
# this deployment (404, 501, 503) or refused for this token (401, 403).
_UNFOLLOWABLE_STATUS_CODES = (401, 403, 404, 501, 503)


class BucketEventsUnavailable(Exception):
    """The live-follow feed cannot be followed for this bucket: do not retry."""


@dataclass
class BucketFileChange:
    """
    A single file change reported by the live-follow feed of a bucket.

    Only `path` and `op` are always set: an `update` carries only the fields that changed, so a field left to
    `None` means "unchanged" (or, for `xet_hash`, possibly "not readable with this token") and never "cleared".

    Args:
        path (`str`):
            Path of the file in the bucket.
        op (`Literal["add", "update", "delete"]`):
            What happened to the file.
        size (`int`, *optional*):
            Size of the file in bytes, when reported.
        xet_hash (`str`, *optional*):
            Xet hash of the file, when reported.
        uploaded_at (`datetime`, *optional*):
            Upload instant, set on every add and re-upload.
        mtime (`datetime`, *optional*):
            Client-provided modification time, when reported.
    """

    path: str
    op: Literal["add", "update", "delete"]
    size: int | None = None
    xet_hash: str | None = None
    uploaded_at: datetime | None = None
    mtime: datetime | None = None


@dataclass
class _Ready:
    cursor: str | None


@dataclass
class _Changes:
    cursor: str
    changes: list[BucketFileChange] = field(default_factory=list)


@dataclass
class _Reset:
    pass


@dataclass
class _Reconnect:
    cursor: str | None


_FollowEvent = Union[_Ready, _Changes, _Reset, _Reconnect]


def _parse_datetime(value: Any) -> datetime | None:
    return parse_datetime(value) if isinstance(value, str) and value else None


def parse_follow_event(name: str, data: str) -> _FollowEvent | None:
    """
    Parse a raw live-follow event into a typed event.

    Args:
        name (`str`):
            Value of the `event:` field: `ready`, `changes`, `reset` or `reconnect`.
        data (`str`):
            JSON payload of the `data:` field (may be empty).

    Returns:
        `Any` | `None`: The parsed event, or `None` when it must be ignored (unknown event name or malformed
        payload).
    """
    try:
        payload = json.loads(data) if data.strip() else {}
    except json.JSONDecodeError:
        logger.debug("live-follow: ignoring malformed %r payload: %s", name, data[:200])
        return None
    if not isinstance(payload, dict):
        logger.debug("live-follow: ignoring %r payload that is not an object: %s", name, data[:200])
        return None

    match name:
        case "ready":
            cursor = payload.get("cursor")
            return _Ready(cursor=cursor if isinstance(cursor, str) else None)

        case "changes":
            cursor, raw_changes = payload.get("cursor"), payload.get("changes")
            if not isinstance(cursor, str) or not isinstance(raw_changes, list):
                logger.debug("live-follow: ignoring 'changes' event without cursor/changes: %s", data[:200])
                return None
            changes = []
            for item in raw_changes:
                if not isinstance(item, dict) or item.get("op") not in ("add", "update", "delete"):
                    continue
                changes.append(
                    BucketFileChange(
                        path=item.get("path", ""),
                        op=item["op"],
                        size=item.get("size"),
                        xet_hash=item.get("xetHash"),
                        uploaded_at=_parse_datetime(item.get("uploadedAt")),
                        mtime=_parse_datetime(item.get("mtime")),
                    )
                )
            return _Changes(cursor=cursor, changes=changes)

        case "reset":
            return _Reset()

        case "reconnect":
            cursor = payload.get("cursor")
            return _Reconnect(cursor=cursor if isinstance(cursor, str) else None)

        case _:
            logger.debug("live-follow: ignoring unknown event type %r", name)
            return None


# A resume point is either absent (reconcile then probe a fresh one), a cursor to resume strictly after, an
# ISO8601 instant to replay changes from, or "live" (only the changes coming next, with no replay at all).
# The three forms are also the steps to fall back through when the server refuses a request (HTTP 400).
ResumePoint = tuple[Literal["cursor", "since", "live"], str]


def _resume_params(resume: ResumePoint | None) -> dict[str, str]:
    match resume:
        case ("cursor", cursor):
            return {"cursor": cursor}
        case ("since", since) if since:
            return {"since": since}
        case _:
            return {}


def _iter_until(chunks: Iterator[bytes], should_stop: Callable[[], bool]) -> Iterator[bytes]:
    """
    Pass response chunks through until `should_stop` returns true.

    The check runs when a chunk arrives, so a stream carrying nothing to parse is abandoned at its next
    keep-alive comment instead of being read until the server ends it.
    """
    for chunk in chunks:
        if should_stop():
            return
        yield chunk


def _follow_events(
    endpoint: str,
    bucket_id: str,
    token: bool | str | None,
    resume: ResumePoint | None,
    should_stop: Callable[[], bool],
) -> Iterator[_FollowEvent]:
    """
    Open a single live-follow session and yield the parsed events until the stream ends.

    Ends silently on EOF, on read timeout and when `should_stop` returns true. Raises
    [`BucketEventsUnavailable`] if the feed cannot be followed and [`~utils.HfHubHTTPError`] for anything else
    the caller must decide about.
    """
    timeout = httpx2.Timeout(connect=CONNECT_TIMEOUT, read=STREAM_READ_TIMEOUT, write=CONNECT_TIMEOUT, pool=10.0)
    with get_session().stream(
        "GET",
        f"{endpoint}/api/buckets/{bucket_id}/events",
        headers={**build_hf_headers(token=token), "accept": "text/event-stream"},
        params=_resume_params(resume),
        timeout=timeout,
    ) as response:
        if response.status_code in _UNFOLLOWABLE_STATUS_CODES:
            raise BucketEventsUnavailable(f"status {response.status_code} on {response.url}")
        if not (response.headers.get("content-type") or "").startswith("text/event-stream"):
            # Raises for error statuses (400, 401,...), otherwise this is not an event stream at all.
            hf_raise_for_status(response)
            raise BucketEventsUnavailable(f"unexpected content-type on {response.url}")
        hf_raise_for_status(response)

        # The vendored SSE client handles chunk-split lines, CRLF endings, the keep-alive comment lines and
        # several events per chunk. It only dispatches events carrying a `data:` field, which the feed always
        # sends (an empty payload comes as `data: {}`).
        try:
            for event in SSEClient(_iter_until(response.iter_bytes(), should_stop)).events():
                parsed = parse_follow_event(event.event, event.data)
                if parsed is not None:
                    yield parsed
        except httpx2.HTTPError as e:
            # Read timeout, connection reset,... indistinguishable from a server-side end of stream.
            logger.debug("live-follow: stream error on bucket %s (%s)", bucket_id, e)


class _Backoff:
    """Exponential backoff, capped, reset by the first healthy session."""

    def __init__(self, base: float, max_delay: float) -> None:
        self._base = base
        self.max_delay = max_delay
        self._attempt = 0

    def next(self) -> float:
        delay = min(self._base * 2**self._attempt, self.max_delay)
        self._attempt += 1
        return delay

    def reset(self) -> None:
        self._attempt = 0


def _retry_after(e: Exception) -> float | None:
    """The server `Retry-After` hint (rate limiting,...) if it sent one."""
    response = getattr(e, "response", None)
    hint = response.headers.get("retry-after") if response is not None else None
    try:
        return float(hint) if hint is not None else None
    except ValueError:
        return None


def _status_code(e: Exception) -> int | None:
    response = getattr(e, "response", None)
    return response.status_code if response is not None else None


class BucketFollower(threading.Thread):
    """
    Daemon thread keeping one bucket's event stream alive and reporting the changes it carries.

    One instance owns exactly one stream, for one bucket. Batches are handed over to `on_changes`. Whenever no
    usable resume point is known (first session, or the server cannot replay that far back), `on_reconcile` is
    called first: the consumer is expected to drop what it holds for that bucket, so that the replay that
    follows starts from a fresh state. The thread ends on its own when `is_alive` turns false, when the feed is
    not served, or after too many sessions ending before they were even ready. It keeps no strong reference to
    its consumer, which can therefore be garbage collected while the thread is running.

    Args:
        endpoint (`str`):
            Endpoint of the Hub.
        bucket_id (`str`):
            ID of the bucket to follow (e.g. `"username/my-bucket"`).
        token (`bool` or `str`, *optional*):
            Token used to consume the feed, resolved again at every (re)connect.
        on_changes (`Callable[[list[BucketFileChange]], None]`):
            Called with each batch of changes received from the feed.
        on_reconcile (`Callable[[], None]`):
            Called before (re)subscribing without a usable cursor.
        is_alive (`Callable[[], bool]`):
            Whether the consumer still cares about the updates. Checked before every session and whenever the
            stream sends anything, so a stream carrying no change is abandoned within one keep-alive.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        bucket_id: str,
        token: bool | str | None = None,
        on_changes: Callable[[list[BucketFileChange]], None],
        on_reconcile: Callable[[], None],
        is_alive: Callable[[], bool],
    ) -> None:
        super().__init__(daemon=True, name=f"hf-bucket-live-follow-{bucket_id}")
        self.bucket_id = bucket_id
        # Whether at least one session reached the `ready` state, i.e. the feed is actually being followed.
        self.subscribed = False
        self._endpoint = endpoint
        self._token = token
        self._on_changes = on_changes
        self._on_reconcile = on_reconcile
        self._is_alive = is_alive
        self._stop_event = threading.Event()
        self._resume: ResumePoint | None = None

    def stop(self) -> None:
        """Ask the follower to stop. Returns before the thread ends: it stops once the stream yields or times out."""
        self._stop_event.set()

    def _should_stop(self) -> bool:
        """Whether to give up: explicitly asked to, or the consumer holding this follower is gone."""
        return self._stop_event.is_set() or not self._is_alive()

    def run(self) -> None:
        backoff = _Backoff(RETRY_BASE_DELAY, RETRY_MAX_DELAY)
        sessions_without_ready = 0
        while not self._should_stop():
            if self._resume is None:
                # No usable resume point: let the consumer drop its view of the bucket, then resume from its
                # `updatedAt` so any change landing after that instant is replayed by the server.
                self._resume = ("since", self._fetch_updated_at())
                self._on_reconcile()
            try:
                healthy = self._follow_once()
            except BucketEventsUnavailable as e:
                logger.debug("live-follow: not following bucket %s (%s)", self.bucket_id, e)
                return
            except Exception as e:
                if _status_code(e) == 400:
                    # The server refused the request itself. Step the resume point back one notch (cheaper
                    # than a full reconcile) and give up once even a bare, live-only request is refused.
                    match self._resume:
                        case ("cursor", _):
                            self._resume = ("since", self._fetch_updated_at())
                        case ("since", _):
                            self._resume = ("live", "")
                        case _:
                            logger.debug("live-follow: giving up on bucket %s (%s)", self.bucket_id, e)
                            return
                delay = _retry_after(e)
                self._sleep(min(delay, backoff.max_delay) if delay is not None else backoff.next())
                continue

            if healthy:
                backoff.reset()
                sessions_without_ready = 0
                self._sleep(RECONNECT_MIN_DELAY)
            else:
                sessions_without_ready += 1
                if sessions_without_ready >= MAX_SESSIONS_WITHOUT_READY:
                    logger.debug(
                        "live-follow: %s sessions for bucket %s ended before 'ready'; not following it",
                        sessions_without_ready,
                        self.bucket_id,
                    )
                    return
                self._sleep(backoff.next())

    def _follow_once(self) -> bool:
        """
        Consume a single stream. Returns whether the session was healthy, i.e. whether it reached the `ready`
        state (a server-directed `reconnect` or `reset` counts as healthy).
        """
        ready = False
        for event in _follow_events(self._endpoint, self.bucket_id, self._token, self._resume, self._should_stop):
            match event:
                case _Ready(cursor=cursor):
                    # An absent cursor means the feed has seen no change yet: keep the current resume point.
                    if cursor is not None:
                        self._resume = ("cursor", cursor)
                    ready = True
                    self.subscribed = True
                case _Changes(cursor=cursor, changes=changes):
                    # On replay, `changes` batches arrive *before* the `ready` event.
                    if changes:
                        self._on_changes(changes)
                    self._resume = ("cursor", cursor)
                case _Reset():
                    # Resume point older than the server's replay buffer: reconcile and resume from `since=`.
                    self._resume = None
                    return True
                case _Reconnect(cursor=cursor):
                    if cursor is not None:
                        self._resume = ("cursor", cursor)
                    return True
            if self._should_stop():
                return ready
        return ready

    def _fetch_updated_at(self) -> str:
        """Get the instant (ISO8601) to resume from. Empty string if it cannot be probed."""
        try:
            response = get_session().get(
                f"{self._endpoint}/api/buckets/{self.bucket_id}",
                headers=build_hf_headers(token=self._token),
                timeout=CONNECT_TIMEOUT,
            )
            hf_raise_for_status(response)
            updated_at = response.json().get("updatedAt")
            return updated_at if isinstance(updated_at, str) else ""
        except Exception as e:
            logger.debug("live-follow: cannot probe updatedAt for bucket %s (%s)", self.bucket_id, e)
            return ""

    def _sleep(self, delay: float) -> None:
        self._stop_event.wait(delay)
