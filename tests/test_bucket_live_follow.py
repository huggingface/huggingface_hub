import gc
import time
import weakref
from datetime import datetime, timezone

import httpx2
import pytest

from huggingface_hub import HfFileSystem
from huggingface_hub._bucket_live_follow import (
    BucketEventsUnavailable,
    BucketFileChange,
    BucketFollower,
    _Backoff,
    _Changes,
    _iter_until,
    _Ready,
    _Reconnect,
    _Reset,
    _retry_after,
    parse_follow_event,
)
from huggingface_hub._hot_reload.sse_client import SSEClient
from huggingface_hub.utils import HfHubHTTPError

from .testing_constants import ENDPOINT_STAGING, TOKEN


def _http_error(status_code: int, headers: dict | None = None) -> HfHubHTTPError:
    response = httpx2.Response(status_code=status_code, headers=headers or {})
    response.request = httpx2.Request("GET", "https://hf.co/api/buckets/user/bucket/events")
    return HfHubHTTPError("server refused the request", response=response)


class TestParseFollowEvent:
    def test_ready_with_cursor(self):
        assert parse_follow_event("ready", '{"cursor":"abc"}') == _Ready(cursor="abc")

    def test_ready_without_payload(self):
        # An absent cursor means the feed has not seen any change yet.
        assert parse_follow_event("ready", "") == _Ready(cursor=None)
        assert parse_follow_event("ready", "{}") == _Ready(cursor=None)

    def test_changes_batch(self):
        payload = (
            '{"cursor":"c1","changes":[{"path":"data/a.txt","op":"add","size":13,"xetHash":"deadbeef",'
            '"uploadedAt":"2026-01-01T00:00:01Z","mtime":"2026-01-01T00:00:00Z","mtimeNanos":0}]}'
        )
        assert parse_follow_event("changes", payload) == _Changes(
            cursor="c1",
            changes=[
                BucketFileChange(
                    path="data/a.txt",
                    op="add",
                    size=13,
                    xet_hash="deadbeef",
                    uploaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc).replace(second=1),
                    mtime=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            ],
        )

    def test_changes_with_sparse_fields(self):
        # An `update` only carries the fields that changed: the others stay None, never "cleared".
        (change,) = parse_follow_event("changes", '{"cursor":"c","changes":[{"path":"a","op":"update"}]}').changes
        assert change == BucketFileChange(path="a", op="update")
        assert (change.size, change.xet_hash, change.uploaded_at, change.mtime) == (None, None, None, None)

    def test_changes_with_delete(self):
        (change,) = parse_follow_event(
            "changes", '{"cursor":"c","changes":[{"path":"deep/a.txt","op":"delete"}]}'
        ).changes
        assert change == BucketFileChange(path="deep/a.txt", op="delete")

    def test_changes_without_cursor_or_changes_is_ignored(self):
        assert parse_follow_event("changes", '{"changes":[]}') is None
        assert parse_follow_event("changes", '{"cursor":"c"}') is None
        assert parse_follow_event("changes", "[]") is None

    def test_changes_skips_items_with_unknown_op(self):
        payload = '{"cursor":"c","changes":[{"path":"a","op":"rename"},{"path":"b","op":"delete"},"nope"]}'
        assert parse_follow_event("changes", payload).changes == [BucketFileChange(path="b", op="delete")]

    def test_reset(self):
        assert parse_follow_event("reset", '{"reason":"cursor_too_old"}') == _Reset()

    def test_reconnect(self):
        assert parse_follow_event("reconnect", "{}") == _Reconnect(cursor=None)
        assert parse_follow_event("reconnect", '{"cursor":"c9"}') == _Reconnect(cursor="c9")

    def test_unknown_event_is_ignored(self):
        assert parse_follow_event("heartbeat", "{}") is None

    def test_malformed_payload_is_ignored(self):
        assert parse_follow_event("ready", "not json") is None


class TestSSEFraming:
    """Check the framing the vendored SSE client is expected to handle for this feed."""

    @staticmethod
    def _parse_chunks(chunks):
        events = []
        for event in SSEClient(iter(chunks)).events():
            if (parsed := parse_follow_event(event.event, event.data)) is not None:
                events.append(parsed)
        return events

    def test_events_split_at_arbitrary_byte_offsets(self):
        stream = b'event: ready\ndata: {"cursor":"c1"}\n\nevent: changes\ndata: {"cursor":"c2","changes":[]}\n\n'
        chunks = [stream[start : start + 7] for start in range(0, len(stream), 7)]
        assert self._parse_chunks(chunks) == [_Ready(cursor="c1"), _Changes(cursor="c2")]

    def test_keep_alive_comments_and_crlf(self):
        chunks = [b": ping\r\n\r\n", b"event: ready\r\ndata: {}\r\n\r\n", b": ping\r\n\r\n"]
        assert self._parse_chunks(chunks) == [_Ready(cursor=None)]

    def test_several_events_in_a_single_chunk(self):
        chunk = b'event: ready\ndata: {"cursor":"c1"}\n\nevent: reconnect\ndata: {"cursor":"c2"}\n\n'
        assert self._parse_chunks([chunk]) == [_Ready(cursor="c1"), _Reconnect(cursor="c2")]


class TestChunkIteration:
    def test_stops_passing_chunks_once_stopped(self):
        # even a stream carrying only keep-alives is abandoned: one chunk after stopping, at the latest
        stopped = False

        collected = []
        for chunk in _iter_until(iter([b"a", b"b", b"c"]), lambda: stopped):
            collected.append(chunk)
            stopped = True

        assert collected == [b"a"]


class TestBackoff:
    def test_exponential_and_capped(self):
        backoff = _Backoff(base=10, max_delay=80)
        assert [backoff.next() for _ in range(5)] == [10, 20, 40, 80, 80]

    def test_reset_after_a_healthy_session(self):
        backoff = _Backoff(base=10, max_delay=80)
        assert backoff.next() == 10
        assert backoff.next() == 20
        backoff.reset()
        assert backoff.next() == 10

    def test_retry_after_hint(self):
        assert _retry_after(_http_error(503, {"retry-after": "30"})) == 30.0

    def test_retry_after_absent_or_invalid(self):
        assert _retry_after(_http_error(500)) is None
        assert _retry_after(_http_error(503, {"retry-after": "Thu, 01 Jan 2026 00:00:00 GMT"})) is None


class _ScriptedStream:
    """Stands in for [`_follow_events`]: plays scripted sessions and records how each one was subscribed."""

    def __init__(self, *sessions):
        self.sessions = list(sessions)
        self.subscriptions = []

    def __call__(self, endpoint, bucket_id, token, resume, should_stop):
        self.subscriptions.append(resume)
        session = self.sessions.pop(0) if self.sessions else BucketEventsUnavailable("end of script")
        if isinstance(session, Exception):
            raise session
        yield from session


@pytest.fixture
def follower_factory(monkeypatch):
    from huggingface_hub import _bucket_live_follow as follow

    def factory(sessions, *, updated_at="2026-01-01T00:00:00Z", is_alive=lambda: True):
        stream = _ScriptedStream(*sessions)
        monkeypatch.setattr(follow, "_follow_events", stream)
        for name in ("RECONNECT_MIN_DELAY", "RETRY_BASE_DELAY", "RETRY_MAX_DELAY"):
            monkeypatch.setattr(follow, name, 0)

        changes, reconciles = [], []
        follower = BucketFollower(
            endpoint="https://hf.co",
            bucket_id="user/bucket",
            on_changes=changes.extend,
            on_reconcile=lambda: reconciles.append(True),
            is_alive=is_alive,
        )
        probes = [updated_at] if isinstance(updated_at, str) else list(updated_at)
        monkeypatch.setattr(follower, "_fetch_updated_at", lambda: probes.pop(0) if probes else "")
        return follower, stream, changes, reconciles

    return factory


def _run_to_completion(follower):
    follower.start()
    follower.join(timeout=10)
    assert not follower.is_alive(), "follower thread did not stop"


class TestBucketFollower:
    def test_reconciles_then_resumes_from_bucket_timestamp(self, follower_factory):
        follower, stream, _, reconciles = follower_factory([BucketEventsUnavailable("not served")])
        _run_to_completion(follower)
        assert reconciles == [True]
        assert stream.subscriptions == [("since", "2026-01-01T00:00:00Z")]
        assert not follower.subscribed

    def test_reports_changes_and_resumes_with_the_last_cursor(self, follower_factory):
        change = BucketFileChange(path="a.txt", op="add")
        follower, stream, changes, reconciles = follower_factory(
            [
                [_Ready(cursor="c1"), _Changes(cursor="c2", changes=[change])],
                BucketEventsUnavailable("end of stream"),
            ]
        )
        _run_to_completion(follower)

        assert changes == [change]
        assert follower.subscribed
        assert reconciles == [True]
        assert stream.subscriptions == [("since", "2026-01-01T00:00:00Z"), ("cursor", "c2")]

    def test_reset_triggers_a_full_reconcile(self, follower_factory):
        change = BucketFileChange(path="a.txt", op="update")
        follower, stream, changes, reconciles = follower_factory(
            [
                [_Ready(cursor="c1"), _Reset()],
                [_Changes(cursor="c2", changes=[change])],
                BucketEventsUnavailable(),
            ],
            updated_at=["2026-01-01T00:00:00Z", "2026-02-02T00:00:00Z"],
        )
        _run_to_completion(follower)

        # The bucket is dropped and re-followed from a freshly probed `updatedAt`.
        assert reconciles == [True, True]
        assert changes == [change]
        # The 3rd session ends without a `ready` event, so it is not counted as a healthy one.
        assert stream.subscriptions == [
            ("since", "2026-01-01T00:00:00Z"),
            ("since", "2026-02-02T00:00:00Z"),
            ("cursor", "c2"),
        ]

    def test_reconnect_resumes_from_the_cursor_it_gives(self, follower_factory):
        follower, stream, _, _ = follower_factory(
            [
                [_Ready(cursor="c1"), _Reconnect(cursor="c2")],
                BucketEventsUnavailable(),
            ]
        )
        _run_to_completion(follower)
        assert stream.subscriptions == [("since", "2026-01-01T00:00:00Z"), ("cursor", "c2")]

    def test_stops_when_the_feed_is_not_served(self, follower_factory):
        follower, stream, _, reconciles = follower_factory([BucketEventsUnavailable("status 404")])
        _run_to_completion(follower)
        assert stream.subscriptions == [("since", "2026-01-01T00:00:00Z")]
        assert not follower.subscribed

    def test_gives_up_when_sessions_never_get_ready(self, follower_factory):
        follower, stream, _, _ = follower_factory([[], [], [], []])
        _run_to_completion(follower)
        assert len(stream.subscriptions) == 3

    def test_a_healthy_session_resets_the_failure_counter(self, follower_factory):
        follower, stream, _, _ = follower_factory([[], [_Ready(cursor="c1")], [], [], []])
        _run_to_completion(follower)
        assert len(stream.subscriptions) == 5

    def test_bad_request_steps_the_resume_point_back(self, follower_factory):
        follower, stream, _, _ = follower_factory(
            [[_Ready(cursor="c1")], _http_error(400), _http_error(400), _http_error(400)],
            updated_at=["2026-01-01T00:00:00Z"] * 4,
        )
        _run_to_completion(follower)
        assert stream.subscriptions == [
            ("since", "2026-01-01T00:00:00Z"),
            ("cursor", "c1"),
            ("since", "2026-01-01T00:00:00Z"),
            ("live", ""),
        ]

    def test_stops_when_the_consumer_is_gone(self, follower_factory):
        follower, stream, _, _ = follower_factory([[_Ready(cursor="c1")]], is_alive=lambda: False)
        _run_to_completion(follower)
        assert stream.subscriptions == []

    def test_stop_ends_the_thread(self, follower_factory):
        change = BucketFileChange(path="a.txt", op="add")
        follower, _, changes, _ = follower_factory([[_Ready(cursor="c1"), _Changes(cursor="c2", changes=[change])]])
        follower.start()
        deadline = time.time() + 10
        while time.time() < deadline and not changes:
            time.sleep(0.05)
        follower.stop()
        follower.join(timeout=10)
        assert changes == [change]
        assert not follower.is_alive()


class TestFollowerLifetime:
    """A follower must not keep its file system alive, and must give up once that file system is gone."""

    @staticmethod
    def _only_keep_alives(endpoint, bucket_id, token, resume, should_stop):
        # a healthy stream that never ends on its own: keep-alive comments, nothing to parse
        while not should_stop():
            time.sleep(0.01)

    @pytest.fixture
    def quiet_stream(self, monkeypatch):
        from huggingface_hub import _bucket_live_follow as follow

        for name in ("RECONNECT_MIN_DELAY", "RETRY_BASE_DELAY", "RETRY_MAX_DELAY"):
            monkeypatch.setattr(follow, name, 0)
        monkeypatch.setattr(BucketFollower, "_fetch_updated_at", lambda self: "2026-01-01T00:00:00Z")
        monkeypatch.setattr(follow, "_follow_events", self._only_keep_alives)

    @staticmethod
    def _wait_for(condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)
        return condition()

    def test_thread_ends_when_the_file_system_is_garbage_collected(self, quiet_stream):
        fs = HfFileSystem(endpoint="https://hf.co", token=False, skip_instance_cache=True, live_follow=True)
        fs._ensure_bucket_follower("user/bucket")
        follower = fs._bucket_followers["user/bucket"]
        assert follower.is_alive()

        reference = weakref.ref(fs)
        del fs
        gc.collect()
        assert reference() is None, "the follower keeps its file system alive"
        assert self._wait_for(lambda: not follower.is_alive()), "the thread outlives its file system"

    def test_thread_ends_on_stop_even_though_the_stream_is_quiet(self, quiet_stream):
        fs = HfFileSystem(endpoint="https://hf.co", token=False, skip_instance_cache=True, live_follow=True)
        fs._ensure_bucket_follower("user/bucket")
        follower = fs._bucket_followers["user/bucket"]

        follower.stop()
        assert self._wait_for(lambda: not follower.is_alive())


class TestBucketInfoUpdatedAt:
    def test_updated_at_is_parsed(self):
        from huggingface_hub import BucketInfo

        info = BucketInfo(
            id="user/bucket",
            private=True,
            createdAt="2026-01-01T00:00:00.000Z",
            size=1,
            totalFiles=1,
            updatedAt="2026-01-02T03:04:05.000Z",
        )
        assert info.updated_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    def test_updated_at_is_optional(self):
        from huggingface_hub import BucketInfo

        info = BucketInfo(id="user/bucket", private=True, createdAt="2026-01-01T00:00:00.000Z", size=1, totalFiles=1)
        assert info.updated_at is None


class TestHfFileSystemCacheInvalidation:
    @pytest.fixture
    def hffs(self):
        fs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True, live_follow=True)
        # Hand-built dircache: only the keys matter to the invalidation logic.
        fs.dircache.update(
            {
                "buckets/user/bucket": [{"name": "buckets/user/bucket/data", "type": "directory"}],
                "buckets/user/bucket/data": [{"name": "buckets/user/bucket/data/a.txt", "type": "file"}],
                "buckets/user/bucket/data/deep": [],
                "buckets/other/bucket": [],
                "models/username/my-model": [],
            }
        )
        return fs

    def test_add_invalidates_parent_directories_up_to_the_root(self, hffs):
        hffs._apply_bucket_changes("user/bucket", [BucketFileChange(path="data/a.txt", op="add")])
        assert "buckets/user/bucket" not in hffs.dircache
        assert "buckets/user/bucket/data" not in hffs.dircache
        assert "buckets/user/bucket/data/deep" in hffs.dircache  # not a parent of the changed path
        assert "buckets/other/bucket" in hffs.dircache
        assert "models/username/my-model" in hffs.dircache

    def test_delete_invalidates_the_whole_chain(self, hffs):
        hffs._apply_bucket_changes("user/bucket", [BucketFileChange(path="data/a.txt", op="delete")])
        assert "buckets/user/bucket" not in hffs.dircache
        assert "buckets/user/bucket/data" not in hffs.dircache

    def test_a_directory_change_also_drops_its_subtree(self, hffs):
        hffs._apply_bucket_changes("user/bucket", [BucketFileChange(path="data", op="delete")])
        assert "buckets/user/bucket/data" not in hffs.dircache
        assert "buckets/user/bucket/data/deep" not in hffs.dircache
        assert "buckets/other/bucket" in hffs.dircache

    def test_change_on_an_unlisted_path_only_touches_its_parents(self, hffs):
        hffs._apply_bucket_changes("user/bucket", [BucketFileChange(path="new/b.txt", op="add")])
        assert set(hffs.dircache) == {
            "buckets/user/bucket/data",
            "buckets/user/bucket/data/deep",
            "buckets/other/bucket",
            "models/username/my-model",
        }

    def test_invalidate_bucket_tree_drops_every_listing(self, hffs):
        hffs._invalidate_bucket_tree("user/bucket")
        assert set(hffs.dircache) == {"buckets/other/bucket", "models/username/my-model"}
