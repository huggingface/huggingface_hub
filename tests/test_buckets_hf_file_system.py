from concurrent.futures import Future

import pytest

from huggingface_hub.hf_file_system import HfFileSystem, HfFileSystemEditFile, HfFileSystemResolvedBucketPath

from .test_hf_file_system import _HfFileSystemBaseROTests, _HfFileSystemBaseRWTests, _HfFileSystemBucketChecks
from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


pytestmark = pytest.mark.xet

try:
    from hf_xet import XetSession

    HAS_XET_RANGE_UPLOAD = hasattr(XetSession, "new_range_upload")
except ImportError:
    HAS_XET_RANGE_UPLOAD = False

# Append and edit modes rely on xet range uploads, only available in recent versions of `hf_xet`.
requires_xet_range_upload = pytest.mark.skipif(
    not HAS_XET_RANGE_UPLOAD, reason="Append/edit modes require a version of hf_xet supporting range uploads."
)


class TestHfFileSystemBucketRO(_HfFileSystemBucketChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @pytest.fixture(scope="class", autouse=True)
    def _shared_bucket(self, request):
        api = type(self).api

        # Create dummy bucket
        repo_url = api.create_bucket(repo_name())
        bucket_id = repo_url.bucket_id
        hf_path = f"buckets/{bucket_id}"
        request.cls.bucket_id = bucket_id
        request.cls.hf_path = hf_path

        # Upload files
        api.batch_bucket_files(
            bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        request.cls.readme_file_path = "README.md"
        request.cls.readme_file = hf_path + "/" + "README.md"
        request.cls.text_file_path = "data/text_data.txt"
        request.cls.text_file = hf_path + "/" + "data/text_data.txt"
        yield
        api.delete_bucket(bucket_id)

    @pytest.fixture(autouse=True)
    def _new_hffs(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

    def test_prefix_collision(self):
        colliding_path = f"{self.hf_path}/dat"

        assert not self.hffs.exists(f"{colliding_path}/new.txt")
        assert self.hffs.glob(f"{colliding_path}/*") == []
        with pytest.raises(FileNotFoundError):
            self.hffs.ls(colliding_path)


class TestHfFileSystemBucketRW(_HfFileSystemBucketChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    @pytest.fixture(autouse=True)
    def _bucket(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

        # Create dummy bucket
        repo_url = self.api.create_bucket(repo_name())
        self.bucket_id = repo_url.bucket_id
        self.hf_path = f"buckets/{self.bucket_id}"

        # Upload files
        self.api.batch_bucket_files(
            self.bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        self.readme_file_path = "README.md"
        self.readme_file = self.hf_path + "/" + self.readme_file_path
        self.text_file_path = "data/text_data.txt"
        self.text_file = self.hf_path + "/" + self.text_file_path
        yield
        self.api.delete_bucket(self.bucket_id)

    @requires_xet_range_upload
    def test_append_file(self):
        with self.hffs.open(self.text_file, "ab") as f:
            f.write(b" appended text")

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy text data appended text"

    @requires_xet_range_upload
    def test_edit_file(self):
        with self.hffs.open(self.text_file, "eb") as f:
            f.insert(0, b"this is ")
            f.edit((8, 13), b"a fantastic")
            f.delete(24, 5)
            f.append(b"!")

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "this is a fantastic text!"

    @requires_xet_range_upload
    def test_edit_file_truncate_and_seek(self):
        with self.hffs.open(self.text_file, "eb") as f:
            f.seek(6)
            f.truncate()  # truncate at current location

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy "

        with pytest.raises(ValueError):
            with self.hffs.open(self.text_file, "eb") as f:
                f.seek(1000)  # past end of file

    def test_edit_modes_are_binary_only(self):
        with pytest.raises(NotImplementedError, match="Only binary modes"):
            self.hffs.open(self.text_file, "a")

        with pytest.raises(NotImplementedError, match="Only binary modes"):
            self.hffs.open(self.text_file, "e")


class _SyncExecutor:
    """Runs 'deferred' sends inline so that the buffering tests are deterministic."""

    def submit(self, fn):
        future = Future()
        try:
            future.set_result(fn())
        except BaseException as exc:  # same semantics as ThreadPoolExecutor.submit
            future.set_exception(exc)
        return future


class FakeEditApi:
    """Stands in for `HfApi` and records the edits that would be sent to the Hub."""

    def __init__(self):
        self.calls: list[list[tuple[int, int, bytes]]] = []
        self.fail_on_call: int | None = None

    def _build_hf_headers(self, token=None):
        return {}

    def _edit_bucket_file(self, *, edits, _file_hash, _file_size, **kwargs):
        self.calls.append(edits)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("send failed")
        return f"hash-{len(self.calls)}"


class FakeBucketFs:
    """Minimal `HfFileSystem` surface needed by `HfFileSystemEditFile`, without any network call."""

    token = "api_token"
    _parent = staticmethod(lambda path: path.rsplit("/", 1)[0])

    def __init__(self, content: bytes = b"0123456789"):
        self.content = content
        self._api = FakeEditApi()

    def resolve_path(self, path):
        return HfFileSystemResolvedBucketPath(path="file.bin", bucket_id="u/b")

    def info(self, path):
        return {"name": path, "size": len(self.content), "xet_hash": "initial-hash"}

    def url(self, path):
        return f"https://fake.com/{path}"

    def touch(self, path):
        self.content = b""

    def invalidate_cache(self, path=None):
        pass


class TestHfFileSystemEditFileBuffering:
    """Offline checks of the buffering/flush policy of `HfFileSystemEditFile`."""

    @pytest.fixture(autouse=True)
    def _sync_executor(self, monkeypatch):
        monkeypatch.setattr("huggingface_hub.hf_file_system._get_deferred_executor", lambda: _SyncExecutor())

    def _open(self, **kwargs):
        fs = FakeBucketFs()
        return HfFileSystemEditFile(fs, "buckets/u/b/file.bin", mode="ab", **kwargs), fs._api

    def test_small_recent_edits_are_not_sent(self):
        f, api = self._open(block_size=1024, send_interval=10)
        f.append(b"abc")
        assert api.calls == []
        assert f.flush() is False
        assert api.calls == []

    def test_small_edits_are_sent_once_send_interval_elapsed(self):
        f, api = self._open(block_size=1024, send_interval=10)
        f.append(b"abc")
        assert api.calls == []

        f.oldest_update_time -= 20  # the first edit is now older than send_interval
        f.append(b"def")
        assert api.calls == [[(10, 10, b"abcdef")]]  # both edits sent as a single insert
        assert f.buffer_size == 0

    def test_edits_are_sent_once_buffer_is_large_enough(self):
        f, api = self._open(block_size=4, send_interval=10**6)
        f.append(b"abc")
        assert api.calls == []

        f.append(b"def")  # buffer is now larger than block_size
        assert len(api.calls) == 1

    def test_force_flush_sends_immediately(self):
        f, api = self._open(block_size=1024, send_interval=10)
        f.append(b"abc")
        assert f.flush() is False
        assert f.flush(force=True) is True
        assert api.calls == [[(10, 10, b"abc")]]

        assert f.flush(force=True) is True  # nothing buffered: nothing is sent
        assert len(api.calls) == 1

    def test_failed_send_is_not_swallowed(self):
        f, api = self._open(block_size=1, send_interval=0)
        api.fail_on_call = 2

        f.append(b"a")  # first send goes through
        f.append(b"b")  # second send fails in the background: nothing raised yet

        with pytest.raises(RuntimeError, match="send failed"):
            f.append(b"c")  # raised when waiting for the previous send

        f.close()  # the failure is reported once, it does not repeat on every later call
