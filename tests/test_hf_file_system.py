import copy
import datetime
import io
import multiprocessing
import multiprocessing.pool
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Type
from unittest.mock import Mock, patch

import fsspec
import pytest

from huggingface_hub import HfApi, constants, hf_file_system
from huggingface_hub.errors import BucketNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.hf_file_system import (
    HfFileSystem,
    HfFileSystemFile,
    HfFileSystemResolvedBucketPath,
    HfFileSystemResolvedRepositoryPath,
    HfFileSystemStreamFile,
)

from .conftest import RepoFactory
from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import OfflineSimulationMode, offline, repo_name


class _HfFileSystemBaseTests:
    __test__ = False
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    http_url_path_prefix: str
    hffs: HfFileSystem
    hf_path: str
    readme_file_path: str
    readme_file: str
    text_file_path: str
    text_file: str

    @pytest.fixture(scope="class", autouse=True)
    def _register_hf_file_system(self):
        """Register `HfFileSystem` as a `fsspec` filesystem if not already registered."""
        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    def _check_info_not_expanded(self, info):
        raise NotImplementedError

    def _check_info_expanded(self, info):
        raise NotImplementedError

    def _check_info_fields(self, info):
        raise NotImplementedError


class _HfFileSystemBaseROTests(_HfFileSystemBaseTests):
    def test_info(self):
        root_dir = self.hffs.info(self.hf_path)
        assert root_dir["type"] == "directory"
        assert root_dir["size"] == 0
        assert root_dir["name"] == self.hf_path
        self._check_info_not_expanded(root_dir)
        self._check_info_fields(root_dir)

        data_dir = self.hffs.info(self.hf_path + "/data")
        assert data_dir["type"] == "directory"
        assert data_dir["size"] == 0
        assert data_dir["name"].endswith("/data")
        self._check_info_not_expanded(data_dir)
        self._check_info_fields(data_dir)

        text_data_file = self.hffs.info(self.text_file)
        assert text_data_file["type"] == "file"
        assert text_data_file["name"].endswith("/data/text_data.txt")
        self._check_info_not_expanded(text_data_file)
        self._check_info_fields(text_data_file)

        # cached info
        assert self.hffs.info(self.text_file) == text_data_file

    def test_glob(self):
        assert self.hffs.glob(self.readme_file) == [self.readme_file]
        assert sorted(self.hffs.glob(self.hf_path + "/*")) == sorted([self.readme_file, self.hf_path + "/data"])
        assert sorted(self.hffs.glob(self.hf_path + "/doesnt-exist/*")) == []
        assert sorted(self.hffs.glob(self.hf_path + "/doesnt/exist/at/all/*")) == []

    def test_url(self):
        assert (
            self.hffs.url(self.text_file)
            == f"{ENDPOINT_STAGING}/{self.hf_path}/resolve/{self.http_url_path_prefix}data/text_data.txt"
        )
        assert (
            self.hffs.url(self.hf_path + "/data")
            == f"{ENDPOINT_STAGING}/{self.hf_path}/tree/{self.http_url_path_prefix}data"
        )

    def test_file_type(self):
        assert self.hffs.isdir(self.hf_path + "/data") and not self.hffs.isdir(self.readme_file)
        assert self.hffs.isfile(self.text_file) and not self.hffs.isfile(self.hf_path + "/data")

    def test_read_file(self):
        with self.hffs.open(self.text_file, "r") as f:
            assert isinstance(f, io.TextIOWrapper)
            assert isinstance(f.buffer, HfFileSystemFile)
            assert f.read() == "dummy text data"
            assert f.read() == ""

    def test_stream_file(self):
        with self.hffs.open(self.hf_path + "/data/binary_data.bin", block_size=0) as f:
            assert isinstance(f, HfFileSystemStreamFile)
            assert f.read() == b"dummy binary data"
            assert f.read() == b""

    def test_stream_file_retry(self):
        with self.hffs.open(self.hf_path + "/data/binary_data.bin", block_size=0) as f:
            assert isinstance(f, HfFileSystemStreamFile)
            assert f.read(6) == b"dummy "
            # Simulate that streaming fails mid-way
            f.response = None
            assert f.read(6) == b"binary"
            assert f.response is not None  # a new connection has been created

    def test_stream_file_reuse_response(self):
        with self.hffs.open(self.hf_path + "/data/binary_data.bin", block_size=0) as f:
            assert isinstance(f, HfFileSystemStreamFile)
            assert f.read(6) == b"dummy "
            first_response = f.response
            assert f.read(6) == b"binary"
            assert f.response == first_response

    def _make_stream_file_with_fake_response(self, chunks: Iterable[bytes]):
        """Helper: create iterator from specified chunks to simulate a stream response."""

        class _FakeResponse:
            def __init__(self, chunks: Iterable[bytes]):
                self._chunks = list(chunks)

            def iter_bytes(self):
                return iter(self._chunks)

        f = HfFileSystemStreamFile(self.hffs, self.hf_path + "/data/binary_data.bin")  # dummy
        f.response = _FakeResponse(chunks)
        f._stream_iterator = f.response.iter_bytes()
        return f

    def test_stream_buffer_overflow_leftover_is_buffered(self):
        # When chunk-1 is larger than read(length), the leftover should be buffered
        f = self._make_stream_file_with_fake_response([b"dummy binary", b" data"])
        assert f.read(6) == b"dummy "
        assert f.loc == 6
        assert bytes(f._stream_buffer) == b"binary"
        assert f.read() == b"binary data"
        assert f.loc == 17
        assert bytes(f._stream_buffer) == b""

    def test_stream_read_spans_buffer_and_chunks(self):
        # When there is already a buffer, read() spans the buffer and the chunks
        f = self._make_stream_file_with_fake_response([b"dummy", b"binary"])
        f._stream_buffer.extend(b"12")
        assert f.read(7) == b"12dummy"
        assert f.read() == b"binary"

    def test_stream_read_all_clears_buffer(self):
        # When read(-1) is called, it returns the buffer + all chunks and clears the buffer
        f = self._make_stream_file_with_fake_response([b"dummy", b"binary"])
        f._stream_buffer.extend(b"12")
        assert f.read(-1) == b"12dummybinary"
        assert bytes(f._stream_buffer) == b""

    def test_stream_read_negative_length_reads_all(self):
        # When length < 0, it reads all
        f = self._make_stream_file_with_fake_response([b"dummy"])
        assert f.read(-2) == b"dummy"

    def test_stream_read_partially_consumes_buffer(self):
        # When read() is called with a length shorter than the buffer,
        # it returns the shorter length and the buffer is partially consumed
        f = self._make_stream_file_with_fake_response([])
        f._stream_buffer.extend(b"dummy binary")
        assert f.read(6) == b"dummy "
        assert bytes(f._stream_buffer) == b"binary"

    def test_stream_read_past_eof_returns_shorter_then_empty(self):
        # When read() is called with a length longer than the file, it returns the shorter length and the buffer is empty
        f = self._make_stream_file_with_fake_response([b"dummy", b"binary"])
        assert f.read(100) == b"dummybinary"
        assert f.read(1) == b""

    def test_modified_time(self):
        assert isinstance(self.hffs.modified(self.text_file), datetime.datetime)
        assert isinstance(self.hffs.modified(self.hf_path + "/data"), datetime.datetime)
        # should fail on a non-existing file
        with pytest.raises(FileNotFoundError):
            self.hffs.modified(self.hf_path + "/data/not_existing_file.txt")

    def test_open_if_not_found(self):
        # Regression test: opening a missing file should raise a FileNotFoundError. This was not the case before when
        # opening a file in read mode.
        with pytest.raises(FileNotFoundError):
            self.hffs.open("hf://missing/repo/not_existing_file.txt", mode="r")

        with pytest.raises(FileNotFoundError):
            self.hffs.open("hf://missing/repo/not_existing_file.txt", mode="w")

    def test_initialize_from_fsspec(self):
        fs, _, paths = fsspec.get_fs_token_paths(
            "hf://" + self.text_file,
            storage_options={
                "endpoint": ENDPOINT_STAGING,
                "token": TOKEN,
            },
        )
        assert isinstance(fs, HfFileSystem)
        assert fs._api.endpoint == ENDPOINT_STAGING
        assert fs.token == TOKEN
        assert paths == [self.text_file]

        fs, _, paths = fsspec.get_fs_token_paths("hf://" + self.text_file)
        assert isinstance(fs, HfFileSystem)
        assert paths == [self.text_file]

    def test_list_root_directory(self):
        files = sorted(self.hffs.ls(self.hf_path), key=lambda info: info["name"])
        assert len(files) == 2

        assert files[1]["type"] == "directory"
        assert files[1]["size"] == 0
        assert files[1]["name"].endswith("/data")
        self._check_info_not_expanded(files[1])
        self._check_info_fields(files[1])

        assert files[0]["type"] == "file"
        assert files[0]["name"].endswith(self.readme_file_path)
        self._check_info_not_expanded(files[0])
        self._check_info_fields(files[0])

    def test_list_data_directory(self):
        files = sorted(self.hffs.ls(self.hf_path + "/data"), key=lambda info: info["name"])
        assert len(files) == 2

        assert files[0]["type"] == "file"
        assert files[0]["name"].endswith("/data/binary_data.bin")
        self._check_info_not_expanded(files[0])
        self._check_info_fields(files[0])

        assert files[1]["type"] == "file"
        assert files[1]["name"].endswith("/data/text_data.txt")
        self._check_info_not_expanded(files[1])
        self._check_info_fields(files[1])

    def test_list_data_file(self):
        files = self.hffs.ls(self.text_file)
        assert len(files) == 1

        assert files[0]["type"] == "file"
        assert files[0]["name"].endswith("/data/text_data.txt")
        self._check_info_not_expanded(files[0])
        self._check_info_fields(files[0])

    def test_list_data_directory_after_listing_subpaths(self):
        """Regression test for #5098."""
        data_dir = self.hf_path + "/data"
        self.hffs.ls(self.text_file)
        self.hffs.find(self.hf_path)
        self.hffs.ls(data_dir, refresh=True)
        assert sorted(self.hffs.ls(data_dir, detail=False)) == [data_dir + "/binary_data.bin", self.text_file]

    def test_list_root_directory_no_detail_then_with_detail(self):
        files = sorted(self.hffs.ls(self.hf_path, detail=False))
        assert len(files) == 2
        assert files[1].endswith("/data") and files[0].endswith(self.readme_file_path)
        self._check_info_not_expanded(self.hffs.dircache[self.hf_path][0])

        files = sorted(self.hffs.ls(self.hf_path, detail=True), key=lambda info: info["name"])
        assert len(files) == 2
        assert files[1]["name"].endswith("/data") and files[0]["name"].endswith(self.readme_file_path)
        self._check_info_not_expanded(self.hffs.dircache[self.hf_path][0])

        files = sorted(self.hffs.ls(self.hf_path, detail=True, expand_info=True), key=lambda info: info["name"])
        assert len(files) == 2
        assert files[1]["name"].endswith("/data") and files[0]["name"].endswith(self.readme_file_path)
        self._check_info_expanded(self.hffs.dircache[self.hf_path][0])

    def test_find_root_directory(self):
        files = self.hffs.find(self.hf_path, detail=False)
        assert files == (
            sorted(self.hffs.ls(self.hf_path, detail=False))[:1]
            + sorted(self.hffs.ls(self.hf_path + "/data", detail=False))
        )

        files = self.hffs.find(self.hf_path, detail=True)
        assert files == {
            f["name"]: f
            for f in sorted(self.hffs.ls(self.hf_path, detail=True), key=lambda info: info["name"])[:1]
            + sorted(self.hffs.ls(self.hf_path + "/data", detail=True), key=lambda info: info["name"])
        }

        files_with_dirs = self.hffs.find(self.hf_path, withdirs=True, detail=False)
        assert files_with_dirs == sorted(
            [self.hf_path]
            + self.hffs.ls(self.hf_path, detail=False)
            + self.hffs.ls(self.hf_path + "/data", detail=False)
        )

    def test_find_data_file(self):
        files = self.hffs.find(self.text_file, detail=False)
        assert files == [self.text_file]

        files = self.hffs.find(self.text_file, detail=True)
        assert files == {self.text_file: self.hffs.info(self.text_file)}

    def test_find_maxdepth(self):
        text_file_depth = self.text_file_path.count("/") + 1
        files = self.hffs.find(self.hf_path, detail=False, maxdepth=text_file_depth - 1)
        assert self.text_file not in files
        files = self.hffs.find(self.hf_path, detail=False, maxdepth=text_file_depth)
        assert self.text_file in files
        # we do it again once the cache is updated
        files = self.hffs.find(self.hf_path, detail=False, maxdepth=text_file_depth - 1)
        assert self.text_file not in files

    def test_read_bytes(self):
        data = self.hffs.read_bytes(self.text_file)
        assert data == b"dummy text data"

    def test_read_text(self):
        data = self.hffs.read_text(self.text_file)
        assert data == "dummy text data"

    def test_open_and_read(self):
        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy text data"

    def test_partial_read(self):
        # If partial read => should not download whole file
        with patch.object(self.hffs, "get_file") as mock:
            with self.hffs.open(self.text_file, "r") as f:
                assert f.read(5) == "dummy"
            mock.assert_not_called()

    def test_get_file_with_temporary_file(self):
        # Test passing a file object works => happens "in-memory" for posix systems
        with tempfile.TemporaryFile() as temp_file:
            self.hffs.get_file(self.text_file, temp_file)
            temp_file.seek(0)
            assert temp_file.read() == b"dummy text data"

    def test_get_file_without_lpath(self):
        with tempfile.TemporaryFile() as temp_file:
            self.hffs.get_file(self.text_file, outfile=temp_file)
            temp_file.seek(0)
            assert temp_file.read() == b"dummy text data"

    def test_get_file_outfile_takes_precedence_over_filelike_lpath(self):
        with tempfile.TemporaryFile() as lpath, tempfile.TemporaryFile() as outfile:
            self.hffs.get_file(self.text_file, lpath, outfile=outfile)
            outfile.seek(0)
            assert outfile.read() == b"dummy text data"
            assert lpath.tell() == 0

    def test_get_file_without_lpath_and_outfile_raises(self):
        with pytest.raises(ValueError, match="Either `lpath` or `outfile` must be provided"):
            self.hffs.get_file(self.text_file)

    def test_get_file_with_temporary_folder(self):
        # Test passing a file path works
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file.txt")
            self.hffs.get_file(self.text_file, temp_file)
            with open(temp_file, "rb") as f:
                assert f.read() == b"dummy text data"

    def test_get_file_with_bare_filename(self):
        # Test passing a bare filename works => downloads to the current working directory
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            os.chdir(temp_dir)
            try:
                self.hffs.get_file(self.text_file, "temp_file.txt")
                with open(os.path.join(temp_dir, "temp_file.txt"), "rb") as f:
                    assert f.read() == b"dummy text data"
            finally:
                os.chdir(cwd)

    def test_get_file_with_kwargs(self):
        # If custom kwargs are passed, the function should still work but defaults to base implementation
        with (
            patch.object(hf_file_system, "http_get") as http_mock,
            patch.object(hf_file_system, "xet_get") as xet_mock,
        ):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
                temp_file = os.path.join(temp_dir, "temp_file.txt")
                self.hffs.get_file(self.text_file, temp_file, custom_kwarg=123)
            http_mock.assert_not_called()
            xet_mock.assert_not_called()

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
                temp_file = os.path.join(temp_dir, "temp_file.txt")
                self.hffs.get_file(self.text_file, temp_file)
            assert http_mock.call_count + xet_mock.call_count == 1

    def test_get_file_on_folder(self):
        # Test it works with custom kwargs
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            assert not (Path(temp_dir) / "data").exists()
            self.hffs.get_file(self.hf_path + "/data", temp_dir + "/data")
            assert (Path(temp_dir) / "data").exists()

    def test_pickle(self):
        # Test that pickling re-populates the HfFileSystem cache and keeps the instance cache attributes.
        #
        # Two things must hold for this test to be deterministic:
        # 1. The pre-pickle fs must hit an endpoint where `text_file` actually exists, so the initial
        #    `isfile` call resolves cleanly and populates exactly one cache entry. Going against
        #    production would 404 and trigger the namespace fallback in `resolve_path`, adding 2 entries.
        # 2. The pre-pickle fs must NOT be shared via fsspec's instance cache with any sibling test
        #    (e.g. the bucket-based `test_pickle` inherited by `HfFileSystemBucketROTests`), otherwise
        #    leftover `_bucket_exists_cache` / `dircache` entries from that test leak into this one
        #    via the main-thread state-inheritance path in `_Cached.__call__`.
        # `skip_instance_cache=True` guarantees a fresh instance regardless of what earlier tests ran.
        fs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)
        fs.isfile(self.text_file)
        pickled = pickle.dumps(fs)
        HfFileSystem.clear_instance_cache()
        with offline(mode=OfflineSimulationMode.CONNECTION_FAILS):
            fs = pickle.loads(pickled)
            assert isinstance(fs, HfFileSystem)
            assert fs in HfFileSystem._cache.values()
            assert self.hf_path + "/data" in fs.dircache
            assert len(list(fs._repo_and_revision_exists_cache) + list(fs._bucket_exists_cache)) == 1
            assert fs.isfile(self.text_file)


class _HfFileSystemBaseRWTests(_HfFileSystemBaseTests):
    def test_remove_file(self):
        self.hffs.rm_file(self.text_file)
        assert self.hffs.glob(self.hf_path + "/data/*") == [self.hf_path + "/data/binary_data.bin"]

    def test_remove_directory(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        assert self.hf_path + "/data" not in self.hffs.ls(self.hf_path)

    def test_write_file(self):
        data = "new text data"
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "w") as f:
            f.write(data)
        assert self.hf_path + "/data/new_text_data.txt" in self.hffs.glob(self.hf_path + "/data/*")
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "r") as f:
            assert f.read() == data

    def test_write_file_multiple_chunks(self):
        data = "a" * (4 << 20)  # 4MB
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "w") as f:
            for _ in range(8):  # 32MB in total
                f.write(data)

        assert self.hf_path + "/data/new_text_data_big.txt" in self.hffs.glob(self.hf_path + "/data/*")
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "r") as f:
            for _ in range(8):
                assert f.read(len(data)) == data

    @pytest.mark.skip("Not implemented yet")
    def test_append_file(self):
        with self.hffs.open(self.text_file, "a") as f:
            f.write(" appended text")

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy text data appended text"

    def test_copy_file(self):
        # Non-LFS file
        self._check_info_fields(self.hffs.info(self.text_file))
        self.hffs.cp_file(self.text_file, self.hf_path + "/data/text_data_copy.txt")
        with self.hffs.open(self.hf_path + "/data/text_data_copy.txt", "r") as f:
            assert f.read() == "dummy text data"
        self._check_info_fields(self.hffs.info(self.hf_path + "/data/text_data_copy.txt"))
        # LFS file
        assert self.hffs.info(self.hf_path + "/data/binary_data.bin") is not None
        self.hffs.cp_file(self.hf_path + "/data/binary_data.bin", self.hf_path + "/data/binary_data_copy.bin")
        with self.hffs.open(self.hf_path + "/data/binary_data_copy.bin", "rb") as f:
            assert f.read() == b"dummy binary data"
        self._check_info_fields(self.hffs.info(self.hf_path + "/data/binary_data_copy.bin"))


class _HfFileSystemRepositoryChecks:
    __test__ = False
    http_url_path_prefix = "main/"

    def _check_info_not_expanded(self, info):
        assert info["last_commit"] is None

    def _check_info_expanded(self, info):
        assert info["last_commit"] is not None

    def _check_info_fields(self, info):
        if info["type"] == "file":
            assert info["blob_id"] is not None
            assert info["size"] > 0  # not empty
            assert "security" in info  # the staging endpoint does not run security checks
            if info["name"].endswith(".bin"):
                assert info["lfs"] is not None
                assert "sha256" in info["lfs"]
                assert "size" in info["lfs"]
                assert "pointer_size" in info["lfs"]


class _HfFileSystemBucketChecks:
    __test__ = False
    http_url_path_prefix = ""

    def _check_info_not_expanded(self, info):
        pass

    def _check_info_expanded(self, info):
        pass

    def _check_info_fields(self, info):
        is_bucket_root = info["name"].count("/") == 2
        if not is_bucket_root:
            assert info["uploaded_at"] is not None
        if info["type"] == "file":
            assert info["mtime"] is not None
            assert info["size"] > 0  # not empty


# File streaming uses plain HTTP instead of Xet without `hf_xet`.
@pytest.mark.transfer
class TestHfFileSystemRepositoryRO(_HfFileSystemRepositoryChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request):
        api = self.api

        # Create dummy repo
        repo_url = api.create_repo(repo_name(), repo_type="dataset")
        try:
            repo_id = repo_url.repo_id
            hf_path = f"datasets/{repo_id}"
            request.cls.repo_id = repo_id
            request.cls.hf_path = hf_path

            # Upload files
            api.upload_file(
                path_or_fileobj=b"dummy binary data on pr",
                path_in_repo="data/binary_data_for_pr.bin",
                repo_id=repo_id,
                repo_type="dataset",
                create_pr=True,
            )
            api.upload_file(
                path_or_fileobj="dummy text data".encode("utf-8"),
                path_in_repo="data/text_data.txt",
                repo_id=repo_id,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=b"dummy binary data",
                path_in_repo="data/binary_data.bin",
                repo_id=repo_id,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj="# Dataset card".encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
            api.delete_file(
                path_in_repo=".gitattributes",
                repo_id=repo_id,
                repo_type="dataset",
            )

            request.cls.readme_file_path = "README.md"
            request.cls.readme_file = hf_path + "/" + "README.md"
            request.cls.text_file_path = "data/text_data.txt"
            request.cls.text_file = hf_path + "/" + "data/text_data.txt"
            yield
        finally:
            api.delete_repo(repo_id, repo_type="dataset", missing_ok=True)

    @pytest.fixture(autouse=True)
    def _new_hffs(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

    def test_glob_with_revision(self):
        assert sorted(self.hffs.glob(self.hf_path + "/*", revision="main")) == sorted(
            [self.readme_file, self.hf_path + "/data"]
        )
        assert sorted(self.hffs.glob(self.hf_path + "@main" + "/*")) == sorted(
            [self.hf_path + "@main/" + self.readme_file_path, self.hf_path + "@main" + "/data"]
        )
        assert self.hffs.glob(self.hf_path + "@refs%2Fpr%2F1" + "/data/*") == [
            self.hf_path + "@refs%2Fpr%2F1" + "/data/binary_data_for_pr.bin"
        ]
        assert self.hffs.glob(self.hf_path + "@refs/pr/1" + "/data/*") == [
            self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin"
        ]
        assert self.hffs.glob(self.hf_path + "/data/*", revision="refs/pr/1") == [
            self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin"
        ]

        self._check_info_not_expanded(
            self.hffs.dircache[self.hf_path + "@main"][0]
        )  # no detail -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True, expand_info=False)
        assert isinstance(files, dict)
        assert len(files) == 2
        keys = sorted(files)
        assert files[keys[0]]["name"].endswith(self.readme_file_path) and files[keys[1]]["name"].endswith("/data")
        self._check_info_not_expanded(
            self.hffs.dircache[self.hf_path + "@main"][0]
        )  # detail but no expand info -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True)
        assert isinstance(files, dict)
        assert len(files) == 2
        keys = sorted(files)
        assert files[keys[0]]["name"].endswith(self.readme_file_path) and files[keys[1]]["name"].endswith("/data")
        self._check_info_not_expanded(files[keys[0]])

    def test_read_file_with_revision(self):
        with self.hffs.open(self.hf_path + "/data/binary_data_for_pr.bin", "rb", revision="refs/pr/1") as f:
            assert f.read() == b"dummy binary data on pr"

    @pytest.mark.xet
    def test_get_file_with_xet(self):
        with (
            patch.object(hf_file_system, "xet_get", wraps=hf_file_system.xet_get) as xet_mock,
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir,
        ):
            self.hffs.get_file(self.hf_path + "/data/binary_data.bin", f"{temp_dir}/main.bin")
            self.hffs.get_file(self.hf_path + "@refs/pr/1/data/binary_data_for_pr.bin", f"{temp_dir}/pr.bin")
            assert Path(temp_dir, "main.bin").read_bytes() == b"dummy binary data"
            assert Path(temp_dir, "pr.bin").read_bytes() == b"dummy binary data on pr"
        assert xet_mock.call_count == 2

    def test_list_data_directory_with_revision(self):
        files = self.hffs.ls(self.hf_path + "@refs%2Fpr%2F1" + "/data")

        for test_name, files in {
            "quoted_rev_in_path": self.hffs.ls(self.hf_path + "@refs%2Fpr%2F1" + "/data"),
            "rev_in_path": self.hffs.ls(self.hf_path + "@refs/pr/1" + "/data"),
            "rev_as_arg": self.hffs.ls(self.hf_path + "/data", revision="refs/pr/1"),
            "quoted_rev_in_path_and_rev_as_arg": self.hffs.ls(
                self.hf_path + "@refs%2Fpr%2F1" + "/data", revision="refs/pr/1"
            ),
        }.items():
            assert len(files) == 1  # only one file in PR
            assert files[0]["type"] == "file"
            assert files[0]["name"].endswith("/data/binary_data_for_pr.bin")  # PR file
            if "quoted_rev_in_path" in test_name:
                assert "@refs%2Fpr%2F1" in files[0]["name"]
            elif "rev_in_path" in test_name:
                assert "@refs/pr/1" in files[0]["name"]


# Writes go through the LFS upload protocol without `hf_xet`.
@pytest.mark.transfer
class TestHfFileSystemRepositoryRW(_HfFileSystemRepositoryChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    @pytest.fixture(autouse=True)
    def _repo(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

        # Create dummy repo
        repo_url = self.api.create_repo(repo_name(), repo_type="dataset")
        try:
            self.repo_id = repo_url.repo_id
            self.hf_path = f"datasets/{self.repo_id}"

            # Upload files
            self.api.upload_file(
                path_or_fileobj=b"dummy binary data on pr",
                path_in_repo="data/binary_data_for_pr.bin",
                repo_id=self.repo_id,
                repo_type="dataset",
                create_pr=True,
            )
            self.api.upload_file(
                path_or_fileobj="dummy text data".encode("utf-8"),
                path_in_repo="data/text_data.txt",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.api.upload_file(
                path_or_fileobj=b"dummy binary data",
                path_in_repo="data/binary_data.bin",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.api.upload_file(
                path_or_fileobj="# Dataset card".encode("utf-8"),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            self.api.delete_file(
                path_in_repo=".gitattributes",
                repo_id=self.repo_id,
                repo_type="dataset",
            )

            self.readme_file_path = "README.md"
            self.readme_file = self.hf_path + "/" + self.readme_file_path
            self.text_file_path = "data/text_data.txt"
            self.text_file = self.hf_path + "/" + self.text_file_path
            yield
        finally:
            self.api.delete_repo(self.repo_id, repo_type="dataset", missing_ok=True)

    def test_remove_file_with_revision(self):
        self.hffs.rm_file(self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin")
        assert self.hffs.glob(self.hf_path + "@refs/pr/1" + "/data/*") == []

    def test_remove_directory_with_revision(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        assert self.hf_path + "/data" not in self.hffs.ls(self.hf_path)
        self.hffs.rm(self.hf_path + "@refs/pr/1" + "/data", recursive=True)
        assert self.hf_path + "@refs/pr/1" + "/data" not in self.hffs.ls(self.hf_path)

    def test_find_root_directory_with_incomplete_cache(self):
        self.api.upload_file(
            path_or_fileobj=b"dummy text data 2",
            path_in_repo="data/sub_data/text_data2.txt",
            repo_id=self.repo_id,
            repo_type="dataset",
        )

        self.api.upload_file(
            path_or_fileobj=b"dummy binary data 2",
            path_in_repo="data1/binary_data2.bin",
            repo_id=self.repo_id,
            repo_type="dataset",
        )

        # Copy the result to make it robust to the cache modifications
        # See discussion in https://github.com/huggingface/huggingface_hub/pull/2103
        # for info on why this is not done in `HfFileSystem.find` by default
        files = copy.deepcopy(self.hffs.find(self.hf_path, detail=True))

        # some directories not in cache
        self.hffs.dircache.pop(self.hf_path + "/data/sub_data")
        # some files not expanded
        self.hffs.dircache[self.hf_path + "/data"][1]["last_commit"] = None
        out = self.hffs.find(self.hf_path, detail=True)
        assert out == files


@pytest.mark.parametrize("path_in_repo", ["", "file.txt", "path/to/file", "path/to/@not-a-revision.txt"])
@pytest.mark.parametrize(
    "root_path,revision,repo_type,repo_id,resolved_revision",
    [
        # Parse with namespace
        ("username/my_model", None, "model", "username/my_model", "main"),
        ("username/my_model", "dev", "model", "username/my_model", "dev"),
        ("username/my_model@dev", None, "model", "username/my_model", "dev"),
        ("datasets/username/my_dataset", None, "dataset", "username/my_dataset", "main"),
        ("datasets/username/my_dataset", "dev", "dataset", "username/my_dataset", "dev"),
        ("datasets/username/my_dataset@dev", None, "dataset", "username/my_dataset", "dev"),
        # Parse with `refs/convert/parquet` and `refs/pr/(\d)+` revisions.
        # Regression tests for https://github.com/huggingface/huggingface_hub/issues/1710.
        (
            "hf://datasets/username/my_dataset@refs/convert/parquet",
            None,
            "dataset",
            "username/my_dataset",
            "refs/convert/parquet",
        ),
        ("hf://username/my_model@refs/pr/10", None, "model", "username/my_model", "refs/pr/10"),
        ("hf://username/my_model@refs/pr/10", "refs/pr/10", "model", "username/my_model", "refs/pr/10"),
        ("hf://username/my_model@refs%2Fpr%2F10", "refs/pr/10", "model", "username/my_model", "refs/pr/10"),
    ],
)
def test_resolve_path(
    root_path: str,
    revision: Optional[str],
    repo_type: str,
    repo_id: str,
    resolved_revision: str,
    path_in_repo: str,
):
    fs = HfFileSystem()
    path = root_path + "/" + path_in_repo if path_in_repo else root_path

    with mock_repo_info(fs):
        resolved_path = fs.resolve_path(path, revision=revision)
        assert isinstance(resolved_path, HfFileSystemResolvedRepositoryPath)
        assert (
            resolved_path.repo_type,
            resolved_path.repo_id,
            resolved_path.revision,
            resolved_path.path_in_repo,
        ) == (repo_type, repo_id, resolved_revision, path_in_repo)
        if "@" in path and "@not-a-revision" not in path:
            assert resolved_path._raw_revision in path


@pytest.mark.parametrize("root_path", ["buckets/username/my_bucket", "hf://buckets/username/my_bucket"])
@pytest.mark.parametrize("path", ["", "file.txt", "path/to/file", "path/to/@not-a-revision.txt"])
def test_resolve_bucket_path(root_path: str, path: str):
    fs = HfFileSystem()
    bucket_id = "username/my_bucket"
    path = root_path + "/" + path if path else root_path

    with mock_bucket_info(fs):
        resolved_path = fs.resolve_path(path)
        assert isinstance(resolved_path, HfFileSystemResolvedBucketPath)
        assert resolved_path.bucket_id, resolved_path.path == (bucket_id, path)


@pytest.mark.parametrize("path_in_repo", ["", "file.txt", "path/to/file"])
@pytest.mark.parametrize(
    "path,revision,expected_path",
    [
        ("hf://username/my_model@refs/pr/10", None, "username/my_model@refs/pr/10"),
        ("username/my_model", "refs/weirdo", "username/my_model@refs%2Fweirdo"),  # not a "special revision" -> encode
    ],
)
def test_unresolve_path(path: str, revision: Optional[str], expected_path: str, path_in_repo: str) -> None:
    fs = HfFileSystem()
    path = path + "/" + path_in_repo if path_in_repo else path
    expected_path = expected_path + "/" + path_in_repo if path_in_repo else expected_path

    with mock_repo_info(fs):
        assert fs.resolve_path(path, revision=revision).unresolve() == expected_path


@pytest.mark.parametrize("root_path", ["buckets/username/my_bucket", "hf://buckets/username/my_bucket"])
@pytest.mark.parametrize("path", ["", "file.txt", "path/to/file", "path/to/@not-a-revision.txt"])
def test_unresolve_bucket_path(root_path: str, path: str) -> None:
    fs = HfFileSystem()
    bucket_id = "username/my_bucket"
    expected_path = "buckets/" + bucket_id + "/" + path if path else "buckets/" + bucket_id
    path = root_path + "/" + path if path else root_path

    with mock_bucket_info(fs):
        assert fs.resolve_path(path).unresolve() == expected_path


def test_resolve_path_with_refs_revision() -> None:
    """
    Testing a very specific edge case where a user has a repo with a revisions named "refs" and a file/directory
    named "pr/10". We can still process them but the user has to use the `revision` argument to disambiguate between
    the two.
    """
    fs = HfFileSystem()
    with mock_repo_info(fs):
        resolved = fs.resolve_path("hf://username/my_model@refs/pr/10", revision="refs")
        assert isinstance(resolved, HfFileSystemResolvedRepositoryPath)
        assert resolved.revision == "refs"
        assert resolved.path_in_repo == "pr/10"
        assert resolved.unresolve() == "username/my_model@refs/pr/10"


def mock_repo_info(fs: HfFileSystem):
    def _inner(repo_id: str, *, revision: str, repo_type: str, **kwargs):
        if repo_id not in ["username/my_dataset", "username/my_model"]:
            raise RepositoryNotFoundError(repo_id, response=Mock())
        if revision is not None and revision not in ["main", "dev", "refs"] and not revision.startswith("refs/"):
            raise RevisionNotFoundError(revision, response=Mock())

    return patch.object(fs._api, "repo_info", _inner)


def mock_bucket_info(fs: HfFileSystem):
    def _inner(bucket_id: str, *_, **kwargs):
        if bucket_id not in ["username/my_bucket"]:
            raise BucketNotFoundError(bucket_id, response=Mock())

    return patch.object(fs._api, "bucket_info", _inner)


def test_resolve_path_with_non_matching_revisions():
    fs = HfFileSystem()
    with pytest.raises(ValueError):
        fs.resolve_path("username/my_model@dev", revision="main")


def test_stream_file_open_if_not_found():
    # Regression test: opening a missing file in streaming mode (block_size=0) must raise
    # FileNotFoundError like the buffered HfFileSystemFile, not an AttributeError.
    fs = HfFileSystem()
    with patch.object(fs, "resolve_path", side_effect=FileNotFoundError("missing/repo")):
        with pytest.raises(FileNotFoundError):
            HfFileSystemStreamFile(fs, "hf://missing/repo/not_existing_file.txt")


REPO_X_PATH = "hf://datasets/username/my_dataset/marker.txt"
BUCKET_X_PATH = "hf://buckets/username/my_bucket/marker.txt"


def _mock_x_write_world(fs, exists: bool):
    """Patchers for an exclusive-create open: file existence + upload calls.

    Each test enters them with `as` to get the underlying mock, e.g.
    `with mock_repo_info(fs), exists_patch as exists_mock, ...`.
    """
    exists_patch = patch.object(fs, "exists", return_value=exists)
    upload_patch = patch.object(fs._api, "upload_file", return_value=None)
    batch_patch = patch.object(fs._api, "batch_bucket_files", return_value=None)
    return exists_patch, upload_patch, batch_patch


def test_open_xb_existing_repo_file_raises():
    fs = HfFileSystem()
    exists_patch, upload_patch, _ = _mock_x_write_world(fs, exists=True)
    with mock_repo_info(fs), exists_patch, upload_patch as upload_mock:
        with pytest.raises(FileExistsError):
            with fs.open(REPO_X_PATH, "xb") as f:
                f.write(b"second")
        upload_mock.assert_not_called()


def test_open_x_existing_repo_file_raises():
    # Text mode "x" is normalized to "xb" by fsspec before reaching _open.
    fs = HfFileSystem()
    exists_patch, upload_patch, _ = _mock_x_write_world(fs, exists=True)
    with mock_repo_info(fs), exists_patch, upload_patch as upload_mock:
        with pytest.raises(FileExistsError):
            with fs.open(REPO_X_PATH, "x") as f:
                f.write("second")
        upload_mock.assert_not_called()


def test_open_xb_existing_bucket_file_raises():
    fs = HfFileSystem()
    exists_patch, _, batch_patch = _mock_x_write_world(fs, exists=True)
    with mock_bucket_info(fs), exists_patch, batch_patch as batch_mock:
        with pytest.raises(FileExistsError):
            with fs.open(BUCKET_X_PATH, "xb") as f:
                f.write(b"second")
        batch_mock.assert_not_called()


def test_open_xb_missing_repo_file_succeeds():
    fs = HfFileSystem()
    exists_patch, upload_patch, _ = _mock_x_write_world(fs, exists=False)
    with mock_repo_info(fs), exists_patch, upload_patch as upload_mock:
        with fs.open(REPO_X_PATH, "xb") as f:
            f.write(b"first")
        upload_mock.assert_called_once()


def test_open_x_missing_repo_file_succeeds():
    fs = HfFileSystem()
    exists_patch, upload_patch, _ = _mock_x_write_world(fs, exists=False)
    with mock_repo_info(fs), exists_patch, upload_patch as upload_mock:
        with fs.open(REPO_X_PATH, "x") as f:
            f.write("first")
        upload_mock.assert_called_once()


def test_open_xb_missing_bucket_file_succeeds():
    fs = HfFileSystem()
    exists_patch, _, batch_patch = _mock_x_write_world(fs, exists=False)
    with mock_bucket_info(fs), exists_patch, batch_patch as batch_mock:
        with fs.open(BUCKET_X_PATH, "xb") as f:
            f.write(b"first")
        batch_mock.assert_called_once()


def test_open_wb_existing_repo_file_overwrites():
    # Control case: exclusive-create guard must not fire for "wb".
    fs = HfFileSystem()
    exists_patch, upload_patch, _ = _mock_x_write_world(fs, exists=True)
    with mock_repo_info(fs), exists_patch, upload_patch as upload_mock:
        with fs.open(REPO_X_PATH, "wb") as f:
            f.write(b"overwrite")
        upload_mock.assert_called_once()


def test_open_xb_existing_file_streaming_untouched():
    # Scoping pin: the exclusive-create guard lives on the buffered path only.
    # block_size=0 (streaming) keeps its current behavior; see issue #5280.
    fs = HfFileSystem()
    exists_patch, _, _ = _mock_x_write_world(fs, exists=True)
    with mock_repo_info(fs), exists_patch:
        f = fs.open(REPO_X_PATH, "xb", block_size=0)
        assert isinstance(f, HfFileSystemStreamFile)
        f.close()


def test_open_xb_passes_revision_to_existence_check():
    fs = HfFileSystem()
    exists_patch, upload_patch, _ = _mock_x_write_world(fs, exists=False)
    with mock_repo_info(fs), exists_patch as exists_mock, upload_patch:
        with fs.open(REPO_X_PATH, "xb", revision="dev") as f:
            f.write(b"first")
        exists_mock.assert_called_once()
        assert exists_mock.call_args.kwargs.get("revision") == "dev"


@pytest.mark.parametrize(
    ("not_supported_path", "expected_error"),
    [
        # empty path => not supported
        ("", NotImplementedError),
        # wrong repo_id => ValueError
        ("foo", ValueError),
        ("datasets", ValueError),
    ],
)
def test_access_repositories_lists(not_supported_path, expected_error: Type[Exception]):
    fs = HfFileSystem()
    with pytest.raises(expected_error):
        fs.info(not_supported_path)
    with pytest.raises(expected_error):
        fs.ls(not_supported_path)
    with pytest.raises(expected_error):
        fs.open(not_supported_path)


def test_exists_after_repo_creation(repo_factory: RepoFactory):
    """Regression test for #1945."""
    hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)
    repo_id = f"{USER}/{repo_name()}"
    assert not hffs.exists(repo_id)
    repo_factory(repo_id=repo_id)
    assert hffs.exists(repo_id, refresh=True)


def test_exists_after_repo_deletion(repo_factory: RepoFactory):
    """Test that exists() correctly reflects repository deletion."""
    # Initialize with staging endpoint and skip cache
    hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)
    api = hffs._api

    # Create a new repo
    repo_id = repo_factory().repo_id
    assert hffs.exists(repo_id, refresh=True)
    # Delete the repo
    api.delete_repo(repo_id=repo_id, repo_type="model")
    # Verify that the repo no longer exists.
    assert not hffs.exists(repo_id, refresh=True)


def _get_fs_token_and_dircache(fs):
    fs = HfFileSystem(endpoint=fs.endpoint, token=fs.token)
    return fs._fs_token, fs.dircache


def test_cache():
    HfFileSystem.clear_instance_cache()
    fs = HfFileSystem()
    fs.dircache = {"dummy": []}

    assert HfFileSystem() is fs
    assert HfFileSystem(endpoint=constants.ENDPOINT) is fs
    assert HfFileSystem(token=None, endpoint=constants.ENDPOINT) is fs

    another_fs = HfFileSystem(endpoint="something-else")
    assert another_fs is not fs
    assert another_fs.dircache != fs.dircache

    with multiprocessing.get_context("spawn").Pool() as pool:
        (fs_token, dircache), (_, another_dircache) = pool.map(_get_fs_token_and_dircache, [fs, another_fs])
        assert dircache == fs.dircache
        assert another_dircache != fs.dircache

    if os.name != "nt":  # "fork" is unavailable on windows
        with multiprocessing.get_context("fork").Pool() as pool:
            (fs_token, dircache), (_, another_dircache) = pool.map(_get_fs_token_and_dircache, [fs, another_fs])
            assert dircache == fs.dircache
            assert another_dircache != fs.dircache

    with multiprocessing.pool.ThreadPool() as pool:
        (fs_token, dircache), (_, another_dircache) = pool.map(_get_fs_token_and_dircache, [fs, another_fs])
        assert dircache == fs.dircache
        assert another_dircache != fs.dircache
        assert fs_token != fs._fs_token  # use a different instance for thread safety


@pytest.mark.production
def test_hf_file_system_file_can_handle_gzipped_file():
    """Test that HfFileSystemStreamFile.read() can handle gzipped files."""
    fs = HfFileSystem(endpoint=constants.ENDPOINT)
    # As of July 2025, the math_qa.py file is gzipped when queried from production:
    with fs.open("datasets/allenai/math_qa/math_qa.py", "r", encoding="utf-8") as f:
        out = f.read()
    assert "class MathQa" in out


_FAKE_REPO_ID = "fake/repo"
_FAKE_REPO = f"datasets/{_FAKE_REPO_ID}"
_FAKE_LAST_COMMIT = {"id": "oid", "title": "title", "date": "2025-01-01T00:00:00.000Z"}


class FakeTreeApi:
    """Mock of the parts of `HfApi` used by `HfFileSystem` to list the tree of a repository.

    It mimics the Hub `tree` endpoint:
    - at most 1000 entries (50 with `expand=True`) are returned per request,
    - pagination is cursor-based, i.e. pages can only be fetched one after the other,
    - a recursive listing returns all the folders and files below the requested path,
    - a non-recursive listing returns the direct children of the requested path.

    Entries are always returned "folders first, then files", each group sorted by path.
    """

    PAGE_SIZE = 1000
    EXPANDED_PAGE_SIZE = 50

    def __init__(self, files: dict[str, int], delay: float = 0.0, error_paths=()):
        self.files = dict(files)
        self.delay = delay
        self.error_paths = set(error_paths)
        self.calls = []  # one entry per request: (path in repo, recursive)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def repo_info(self, repo_id, revision=None, repo_type=None, **kwargs):
        if repo_id != _FAKE_REPO_ID or revision not in (None, "main"):
            raise RepositoryNotFoundError(f"404 Client Error: Repository Not Found for repo: {repo_id}")
        return Mock(id=repo_id)

    def _entries(self, path: str | None, recursive: bool, expand: bool) -> list:
        prefix = path + "/" if path else ""
        files = sorted(path_in_repo for path_in_repo in self.files if path_in_repo.startswith(prefix))
        relatives = [path_in_repo[len(prefix) :] for path_in_repo in files]
        if recursive:
            # a recursive listing returns all the nested folders
            folders = {
                prefix + "/".join(relative.split("/")[:nb_parts])
                for relative in relatives
                for nb_parts in range(1, len(relative.split("/")))
            }
        else:
            # a non-recursive listing returns the direct children only
            folders = {prefix + relative.split("/")[0] for relative in relatives if "/" in relative}
            files = [path_in_repo for path_in_repo, relative in zip(files, relatives) if "/" not in relative]
        entries = [RepoFolder(path=folder, oid="oid") for folder in sorted(folders)]
        entries += [
            RepoFile(
                path=path_in_repo,
                size=self.files[path_in_repo],
                oid="oid",
                last_commit=_FAKE_LAST_COMMIT if expand else None,
            )
            for path_in_repo in files
        ]
        return entries

    def _paginate(self, path: str | None, recursive: bool, expand: bool):
        """Yield pages of entries, one request at a time (as the Hub `tree` endpoint does)."""
        if path in self.error_paths:
            raise RuntimeError(f"Failed to list {path}")
        entries = self._entries(path, recursive=recursive, expand=expand)
        page_size = self.EXPANDED_PAGE_SIZE if expand else self.PAGE_SIZE
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for i in range(0, max(len(entries), 1), page_size):
                with self._lock:
                    self.calls.append((path, recursive))
                if self.delay:
                    time.sleep(self.delay)
                page = entries[i : i + page_size]
                yield page, i + page_size >= len(entries)
        finally:
            with self._lock:
                self.in_flight -= 1

    def _iter_repo_tree_pages(self, repo_id, path=None, recursive=True, expand=False, revision=None, repo_type=None):
        assert recursive  # only ever used to list a tree recursively
        yield from self._paginate(path, recursive=True, expand=expand)

    def list_repo_tree(self, repo_id, path=None, recursive=False, expand=False, revision=None, repo_type=None):
        for page, _ in self._paginate(path, recursive=recursive, expand=expand):
            yield from page


def _fake_fs(files: dict[str, int], max_workers: int, **fake_kwargs) -> tuple[HfFileSystem, FakeTreeApi]:
    api = FakeTreeApi(files, **fake_kwargs)
    fs = HfFileSystem(endpoint="https://fake.example.com", skip_instance_cache=True, max_workers=max_workers)
    fs._api = api
    return fs, api


def test_find_large_repo_is_listed_in_parallel():
    files = {f"dir_{i:03d}/file_{j:04d}.bin": 10 for i in range(20) for j in range(1100)}
    files["README.md"] = 5

    fs, api = _fake_fs(files, max_workers=16, delay=0.02)
    found = fs.find(_FAKE_REPO)

    assert api.max_in_flight > 1  # directories were listed concurrently
    # once the root pagination is known to be long, the root is listed one level deep and each of its
    # directories is listed by its own task: no task ever lists below a direct sub-directory
    assert all(path is None or "/" not in path for path, recursive in api.calls)
    assert {path for path, recursive in api.calls if path and recursive} == {f"dir_{i:03d}" for i in range(20)}
    # root pages until the split + 1 root listing + 2 pages per directory
    assert len(api.calls) <= 52

    # the same listing done sequentially (see `max_workers=1`) requires one request every 1000 entries
    sequential_fs, sequential_api = _fake_fs(files, max_workers=1, delay=0.02)
    assert sequential_fs.find(_FAKE_REPO) == found
    assert len(sequential_api.calls) == 23


def test_find_deeply_nested_repo_is_split_recursively():
    # 10 directories, each holding 10 sub-directories with 1100 files in them
    files = {f"dir_{i}/sub_{j}/file_{k:04d}.bin": 10 for i in range(10) for j in range(10) for k in range(1100)}

    fs, api = _fake_fs(files, max_workers=16)
    assert sorted(fs.find(_FAKE_REPO)) == sorted(f"{_FAKE_REPO}/{path}" for path in files)

    # each directory is listed on its own, one level deep
    assert {path for path, recursive in api.calls if path} == {f"dir_{i}" for i in range(10)} | {
        f"dir_{i}/sub_{j}" for i in range(10) for j in range(10)
    }
    assert all(path is None or path.count("/") <= 1 for path, recursive in api.calls)
    assert sorted(fs.dircache) == sorted(
        [_FAKE_REPO]
        + [f"{_FAKE_REPO}/dir_{i}" for i in range(10)]
        + [f"{_FAKE_REPO}/dir_{i}/sub_{j}" for i in range(10) for j in range(10)]
    )


def test_find_small_repo_is_listed_with_a_single_request():
    files = {"README.md": 3, "data/file.txt": 1, "data/sub/file2.txt": 2}

    fs, api = _fake_fs(files, max_workers=16)
    assert fs.find(_FAKE_REPO) == sorted(f"{_FAKE_REPO}/{path}" for path in files)
    assert api.calls == [(None, True)]  # a single page: nothing to split

    # the full tree is cached, directory by directory
    assert sorted(fs.dircache) == [_FAKE_REPO, f"{_FAKE_REPO}/data", f"{_FAKE_REPO}/data/sub"]
    assert [o["name"] for o in fs.dircache[f"{_FAKE_REPO}/data"]] == [
        f"{_FAKE_REPO}/data/sub",
        f"{_FAKE_REPO}/data/file.txt",
    ]

    # ... which can then be reused to list a sub-directory without any new request
    assert fs.ls(f"{_FAKE_REPO}/data/sub", detail=False) == [f"{_FAKE_REPO}/data/sub/file2.txt"]
    assert api.calls == [(None, True)]


def test_find_many_small_directories_is_not_split():
    # splitting a directory in (tiny) sub-directories would cost more requests than it would save
    files = {f"dir_{i:03d}/file_{j}.bin": 10 for i in range(400) for j in range(3)}

    fs, api = _fake_fs(files, max_workers=16)
    assert sorted(fs.find(_FAKE_REPO)) == sorted(f"{_FAKE_REPO}/{path}" for path in files)
    assert api.calls == [(None, True), (None, True)]  # 1600 entries listed through 2 pages


@pytest.mark.parametrize("maxdepth", [1, 2])
def test_find_maxdepth_does_not_list_deeper(maxdepth):
    files = {f"dir_{i:03d}/sub/file_{j:04d}.bin": 10 for i in range(20) for j in range(1100)}
    files["README.md"] = 5

    fs, api = _fake_fs(files, max_workers=16)
    found = fs.find(_FAKE_REPO, maxdepth=maxdepth, withdirs=True)
    expected = ["README.md"] + [f"dir_{i:03d}" for i in range(20)]
    if maxdepth >= 2:
        expected += [f"dir_{i:03d}/sub" for i in range(20)]
    # note: as per the fsspec spec, the root directory itself is part of the output
    assert sorted(found) == sorted([_FAKE_REPO] + [f"{_FAKE_REPO}/{path}" for path in expected])

    # nothing is listed below the requested depth
    assert all(path is None or path.count("/") < maxdepth for path, recursive in api.calls)
    assert all(not recursive for path, recursive in api.calls if path is not None)


def test_find_expand_info():
    files = {f"dir_{i}/file_{j}.bin": 10 for i in range(5) for j in range(60)}

    fs, api = _fake_fs(files, max_workers=16)
    found = fs.find(_FAKE_REPO, detail=True, expand_info=True)
    assert sorted(found) == sorted(f"{_FAKE_REPO}/{path}" for path in files)
    assert all(info["last_commit"] is not None for info in found.values())

    fs, api = _fake_fs(files, max_workers=16)
    found = fs.find(_FAKE_REPO, detail=True)
    assert all(info["last_commit"] is None for info in found.values())


def test_find_propagates_errors():
    files = {f"dir_{i:02d}/file_{j}.bin": 10 for i in range(10) for j in range(1100)}

    fs, api = _fake_fs(files, max_workers=4, error_paths=["dir_03"])
    with pytest.raises(RuntimeError):
        fs.find(_FAKE_REPO)
