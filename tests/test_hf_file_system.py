import copy
import datetime
import io
import multiprocessing
import multiprocessing.pool
import os
import pickle
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Optional, Type
from unittest.mock import MagicMock, Mock, patch

import fsspec
import pytest

import huggingface_hub.hf_file_system as hffs_mod
from huggingface_hub import HfApi, constants, hf_file_system
from huggingface_hub.errors import BucketNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.file_download import HfFileMetadata
from huggingface_hub.utils import XetFileData
from huggingface_hub.hf_file_system import (
    HfFileSystem,
    HfFileSystemFile,
    HfFileSystemResolvedBucketPath,
    HfFileSystemResolvedRepositoryPath,
    HfFileSystemStreamFile,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import OfflineSimulationMode, offline, repo_name, with_production_testing


class _HfFileSystemBaseTests(unittest.TestCase):
    __test__ = False
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    http_url_path_prefix: str
    hffs: HfFileSystem
    hf_path: str
    readme_file_path: str
    readme_file: str
    text_file_path: str
    text_file: str

    @classmethod
    def setUpClass(cls):
        """Register `HfFileSystem` as a `fsspec` filesystem if not already registered."""
        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    def assertInfoIsNotExpanded(self, info):
        raise NotImplementedError

    def assertInfoIsExpanded(self, info):
        raise NotImplementedError

    def assertInfoFields(self, info):
        raise NotImplementedError


class _HfFileSystemBaseROTests(_HfFileSystemBaseTests):
    def test_info(self):
        root_dir = self.hffs.info(self.hf_path)
        self.assertEqual(root_dir["type"], "directory")
        self.assertEqual(root_dir["size"], 0)
        self.assertEqual(root_dir["name"], self.hf_path)
        self.assertInfoIsNotExpanded(root_dir)
        self.assertInfoFields(root_dir)

        data_dir = self.hffs.info(self.hf_path + "/data")
        self.assertEqual(data_dir["type"], "directory")
        self.assertEqual(data_dir["size"], 0)
        self.assertTrue(data_dir["name"].endswith("/data"))
        self.assertInfoIsNotExpanded(data_dir)
        self.assertInfoFields(data_dir)

        text_data_file = self.hffs.info(self.text_file)
        self.assertEqual(text_data_file["type"], "file")
        self.assertTrue(text_data_file["name"].endswith("/data/text_data.txt"))
        self.assertInfoIsNotExpanded(text_data_file)
        self.assertInfoFields(text_data_file)

        # cached info
        self.assertEqual(self.hffs.info(self.text_file), text_data_file)

    def test_glob(self):
        self.assertEqual(
            self.hffs.glob(self.readme_file),
            [self.readme_file],
        )
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/*")),
            sorted([self.readme_file, self.hf_path + "/data"]),
        )
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/doesnt-exist/*")),
            [],
        )
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/doesnt/exist/at/all/*")),
            [],
        )

    def test_url(self):
        self.assertEqual(
            self.hffs.url(self.text_file),
            f"{ENDPOINT_STAGING}/{self.hf_path}/resolve/{self.http_url_path_prefix}data/text_data.txt",
        )
        self.assertEqual(
            self.hffs.url(self.hf_path + "/data"),
            f"{ENDPOINT_STAGING}/{self.hf_path}/tree/{self.http_url_path_prefix}data",
        )

    def test_file_type(self):
        self.assertTrue(self.hffs.isdir(self.hf_path + "/data") and not self.hffs.isdir(self.readme_file))
        self.assertTrue(self.hffs.isfile(self.text_file) and not self.hffs.isfile(self.hf_path + "/data"))

    def test_read_file(self):
        with self.hffs.open(self.text_file, "r") as f:
            self.assertIsInstance(f, io.TextIOWrapper)
            self.assertIsInstance(f.buffer, HfFileSystemFile)
            self.assertEqual(f.read(), "dummy text data")
            self.assertEqual(f.read(), "")

    def test_stream_file(self):
        with self.hffs.open(self.hf_path + "/data/binary_data.bin", block_size=0) as f:
            self.assertIsInstance(f, HfFileSystemStreamFile)
            self.assertEqual(f.read(), b"dummy binary data")
            self.assertEqual(f.read(), b"")

    def test_stream_file_retry(self):
        with self.hffs.open(self.hf_path + "/data/binary_data.bin", block_size=0) as f:
            self.assertIsInstance(f, HfFileSystemStreamFile)
            self.assertEqual(f.read(6), b"dummy ")
            # Simulate that streaming fails mid-way (HTTP mode only: drop the response)
            f.response = None
            self.assertEqual(f.read(6), b"binary")
            # In HTTP mode a new connection is opened (response becomes non-None).
            # In Xet mode there is no response object; the stream iterator drives reads.
            if not f._xet_mode:
                self.assertIsNotNone(f.response)  # a new connection has been created

    def test_stream_file_reuse_response(self):
        with self.hffs.open(self.hf_path + "/data/binary_data.bin", block_size=0) as f:
            self.assertIsInstance(f, HfFileSystemStreamFile)
            self.assertEqual(f.read(6), b"dummy ")
            first_response = f.response
            self.assertEqual(f.read(6), b"binary")
            self.assertEqual(f.response, first_response)

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
        f._stream_opened = True
        f._xet_mode = False
        return f

    def test_stream_buffer_overflow_leftover_is_buffered(self):
        # When chunk-1 is larger than read(length), the leftover should be buffered
        f = self._make_stream_file_with_fake_response([b"dummy binary", b" data"])
        self.assertEqual(f.read(6), b"dummy ")
        self.assertEqual(f.loc, 6)
        self.assertEqual(bytes(f._stream_buffer), b"binary")
        self.assertEqual(f.read(), b"binary data")
        self.assertEqual(f.loc, 17)
        self.assertEqual(bytes(f._stream_buffer), b"")

    def test_stream_read_spans_buffer_and_chunks(self):
        # When there is already a buffer, read() spans the buffer and the chunks
        f = self._make_stream_file_with_fake_response([b"dummy", b"binary"])
        f._stream_buffer.extend(b"12")
        self.assertEqual(f.read(7), b"12dummy")
        self.assertEqual(f.read(), b"binary")

    def test_stream_read_all_clears_buffer(self):
        # When read(-1) is called, it returns the buffer + all chunks and clears the buffer
        f = self._make_stream_file_with_fake_response([b"dummy", b"binary"])
        f._stream_buffer.extend(b"12")
        self.assertEqual(f.read(-1), b"12dummybinary")
        self.assertEqual(bytes(f._stream_buffer), b"")

    def test_stream_read_negative_length_reads_all(self):
        # When length < 0, it reads all
        f = self._make_stream_file_with_fake_response([b"dummy"])
        self.assertEqual(f.read(-2), b"dummy")

    def test_stream_read_partially_consumes_buffer(self):
        # When read() is called with a length shorter than the buffer,
        # it returns the shorter length and the buffer is partially consumed
        f = self._make_stream_file_with_fake_response([])
        f._stream_buffer.extend(b"dummy binary")
        self.assertEqual(f.read(6), b"dummy ")
        self.assertEqual(bytes(f._stream_buffer), b"binary")

    def test_stream_read_past_eof_returns_shorter_then_empty(self):
        # When read() is called with a length longer than the file, it returns the shorter length and the buffer is empty
        f = self._make_stream_file_with_fake_response([b"dummy", b"binary"])
        self.assertEqual(f.read(100), b"dummybinary")
        self.assertEqual(f.read(1), b"")

    def test_modified_time(self):
        self.assertIsInstance(self.hffs.modified(self.text_file), datetime.datetime)
        self.assertIsInstance(self.hffs.modified(self.hf_path + "/data"), datetime.datetime)
        # should fail on a non-existing file
        with self.assertRaises(FileNotFoundError):
            self.hffs.modified(self.hf_path + "/data/not_existing_file.txt")

    def test_open_if_not_found(self):
        # Regression test: opening a missing file should raise a FileNotFoundError. This was not the case before when
        # opening a file in read mode.
        with self.assertRaises(FileNotFoundError):
            self.hffs.open("hf://missing/repo/not_existing_file.txt", mode="r")

        with self.assertRaises(FileNotFoundError):
            self.hffs.open("hf://missing/repo/not_existing_file.txt", mode="w")

    def test_initialize_from_fsspec(self):
        fs, _, paths = fsspec.get_fs_token_paths(
            "hf://" + self.text_file,
            storage_options={
                "endpoint": ENDPOINT_STAGING,
                "token": TOKEN,
            },
        )
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(fs._api.endpoint, ENDPOINT_STAGING)
        self.assertEqual(fs.token, TOKEN)
        self.assertEqual(paths, [self.text_file])

        fs, _, paths = fsspec.get_fs_token_paths("hf://" + self.text_file)
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(paths, [self.text_file])

    def test_list_root_directory(self):
        files = sorted(self.hffs.ls(self.hf_path), key=lambda info: info["name"])
        self.assertEqual(len(files), 2)

        self.assertEqual(files[1]["type"], "directory")
        self.assertEqual(files[1]["size"], 0)
        self.assertTrue(files[1]["name"].endswith("/data"))
        self.assertInfoIsNotExpanded(files[1])
        self.assertInfoFields(files[1])

        self.assertEqual(files[0]["type"], "file")
        self.assertTrue(files[0]["name"].endswith(self.readme_file_path))
        self.assertInfoIsNotExpanded(files[0])
        self.assertInfoFields(files[0])

    def test_list_data_directory(self):
        files = sorted(self.hffs.ls(self.hf_path + "/data"), key=lambda info: info["name"])
        self.assertEqual(len(files), 2)

        self.assertEqual(files[0]["type"], "file")
        self.assertTrue(files[0]["name"].endswith("/data/binary_data.bin"))
        self.assertInfoIsNotExpanded(files[0])
        self.assertInfoFields(files[0])

        self.assertEqual(files[1]["type"], "file")
        self.assertTrue(files[1]["name"].endswith("/data/text_data.txt"))
        self.assertInfoIsNotExpanded(files[1])
        self.assertInfoFields(files[1])

    def test_list_data_file(self):
        files = self.hffs.ls(self.text_file)
        self.assertEqual(len(files), 1)

        self.assertEqual(files[0]["type"], "file")
        self.assertTrue(files[0]["name"].endswith("/data/text_data.txt"))
        self.assertInfoIsNotExpanded(files[0])
        self.assertInfoFields(files[0])

    def test_list_root_directory_no_detail_then_with_detail(self):
        files = sorted(self.hffs.ls(self.hf_path, detail=False))
        self.assertEqual(len(files), 2)
        self.assertTrue(files[1].endswith("/data") and files[0].endswith(self.readme_file_path))
        self.assertInfoIsNotExpanded(self.hffs.dircache[self.hf_path][0])

        files = sorted(self.hffs.ls(self.hf_path, detail=True), key=lambda info: info["name"])
        self.assertEqual(len(files), 2)
        self.assertTrue(files[1]["name"].endswith("/data") and files[0]["name"].endswith(self.readme_file_path))
        self.assertInfoIsNotExpanded(self.hffs.dircache[self.hf_path][0])

        files = sorted(self.hffs.ls(self.hf_path, detail=True, expand_info=True), key=lambda info: info["name"])
        self.assertEqual(len(files), 2)
        self.assertTrue(files[1]["name"].endswith("/data") and files[0]["name"].endswith(self.readme_file_path))
        self.assertInfoIsExpanded(self.hffs.dircache[self.hf_path][0])

    def test_find_root_directory(self):
        files = self.hffs.find(self.hf_path, detail=False)
        self.assertEqual(
            files,
            sorted(self.hffs.ls(self.hf_path, detail=False))[:1]
            + sorted(self.hffs.ls(self.hf_path + "/data", detail=False)),
        )

        files = self.hffs.find(self.hf_path, detail=True)
        self.assertEqual(
            files,
            {
                f["name"]: f
                for f in sorted(self.hffs.ls(self.hf_path, detail=True), key=lambda info: info["name"])[:1]
                + sorted(self.hffs.ls(self.hf_path + "/data", detail=True), key=lambda info: info["name"])
            },
        )

        files_with_dirs = self.hffs.find(self.hf_path, withdirs=True, detail=False)
        self.assertEqual(
            files_with_dirs,
            sorted(
                [self.hf_path]
                + self.hffs.ls(self.hf_path, detail=False)
                + self.hffs.ls(self.hf_path + "/data", detail=False)
            ),
        )

    def test_find_data_file(self):
        files = self.hffs.find(self.text_file, detail=False)
        self.assertEqual(files, [self.text_file])

    def test_find_maxdepth(self):
        text_file_depth = self.text_file_path.count("/") + 1
        files = self.hffs.find(self.hf_path, detail=False, maxdepth=text_file_depth - 1)
        self.assertNotIn(self.text_file, files)
        files = self.hffs.find(self.hf_path, detail=False, maxdepth=text_file_depth)
        self.assertIn(self.text_file, files)
        # we do it again once the cache is updated
        files = self.hffs.find(self.hf_path, detail=False, maxdepth=text_file_depth - 1)
        self.assertNotIn(self.text_file, files)

    def test_read_bytes(self):
        data = self.hffs.read_bytes(self.text_file)
        self.assertEqual(data, b"dummy text data")

    def test_read_text(self):
        data = self.hffs.read_text(self.text_file)
        self.assertEqual(data, "dummy text data")

    def test_open_and_read(self):
        with self.hffs.open(self.text_file, "r") as f:
            self.assertEqual(f.read(), "dummy text data")

    def test_partial_read(self):
        # If partial read => should not download whole file
        with patch.object(self.hffs, "get_file") as mock:
            with self.hffs.open(self.text_file, "r") as f:
                self.assertEqual(f.read(5), "dummy")
            mock.assert_not_called()

    def test_get_file_with_temporary_file(self):
        # Test passing a file object works => happens "in-memory" for posix systems
        with tempfile.TemporaryFile() as temp_file:
            self.hffs.get_file(self.text_file, temp_file)
            temp_file.seek(0)
            assert temp_file.read() == b"dummy text data"

    def test_get_file_with_temporary_folder(self):
        # Test passing a file path works
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file.txt")
            self.hffs.get_file(self.text_file, temp_file)
            with open(temp_file, "rb") as f:
                assert f.read() == b"dummy text data"

    def test_get_file_with_kwargs(self):
        # If custom kwargs are passed, the function should still work but defaults to base implementation
        with patch.object(hf_file_system, "http_get") as mock:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, "temp_file.txt")
                self.hffs.get_file(self.text_file, temp_file, custom_kwarg=123)
            mock.assert_not_called()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, "temp_file.txt")
                self.hffs.get_file(self.text_file, temp_file)
            mock.assert_called_once()

    def test_get_file_on_folder(self):
        # Test it works with custom kwargs
        with tempfile.TemporaryDirectory() as temp_dir:
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
        self.assertEqual(self.hffs.glob(self.hf_path + "/data/*"), [self.hf_path + "/data/binary_data.bin"])

    def test_remove_directory(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "/data", self.hffs.ls(self.hf_path))

    def test_write_file(self):
        data = "new text data"
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "w") as f:
            f.write(data)
        self.assertIn(self.hf_path + "/data/new_text_data.txt", self.hffs.glob(self.hf_path + "/data/*"))
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "r") as f:
            self.assertEqual(f.read(), data)

    def test_write_file_multiple_chunks(self):
        data = "a" * (4 << 20)  # 4MB
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "w") as f:
            for _ in range(8):  # 32MB in total
                f.write(data)

        self.assertIn(self.hf_path + "/data/new_text_data_big.txt", self.hffs.glob(self.hf_path + "/data/*"))
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "r") as f:
            for _ in range(8):
                self.assertEqual(f.read(len(data)), data)

    @unittest.skip("Not implemented yet")
    def test_append_file(self):
        with self.hffs.open(self.text_file, "a") as f:
            f.write(" appended text")

        with self.hffs.open(self.text_file, "r") as f:
            self.assertEqual(f.read(), "dummy text data appended text")

    def test_copy_file(self):
        # Non-LFS file
        self.assertInfoFields(self.hffs.info(self.text_file))
        self.hffs.cp_file(self.text_file, self.hf_path + "/data/text_data_copy.txt")
        with self.hffs.open(self.hf_path + "/data/text_data_copy.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")
        self.assertInfoFields(self.hffs.info(self.hf_path + "/data/text_data_copy.txt"))
        # LFS file
        self.assertIsNotNone(self.hffs.info(self.hf_path + "/data/binary_data.bin"))
        self.hffs.cp_file(self.hf_path + "/data/binary_data.bin", self.hf_path + "/data/binary_data_copy.bin")
        with self.hffs.open(self.hf_path + "/data/binary_data_copy.bin", "rb") as f:
            self.assertEqual(f.read(), b"dummy binary data")
        self.assertInfoFields(self.hffs.info(self.hf_path + "/data/binary_data_copy.bin"))


class _HfFileSystemRepositoryChecks(unittest.TestCase):
    __test__ = False
    http_url_path_prefix = "main/"

    def assertInfoIsNotExpanded(self, info):
        self.assertIsNone(info["last_commit"])

    def assertInfoIsExpanded(self, info):
        self.assertIsNotNone(info["last_commit"])

    def assertInfoFields(self, info):
        if info["type"] == "file":
            self.assertIsNotNone(info["blob_id"])
            self.assertGreater(info["size"], 0)  # not empty
            self.assertIn("security", info)  # the staging endpoint does not run security checks
            if info["name"].endswith(".bin"):
                self.assertIsNotNone(info["lfs"])
                self.assertIn("sha256", info["lfs"])
                self.assertIn("size", info["lfs"])
                self.assertIn("pointer_size", info["lfs"])


class _HfFileSystemBucketChecks(unittest.TestCase):
    __test__ = False
    http_url_path_prefix = ""

    def assertInfoIsNotExpanded(self, info):
        pass

    def assertInfoIsExpanded(self, info):
        pass

    def assertInfoFields(self, info):
        is_bucket_root = info["name"].count("/") == 2
        if not is_bucket_root:
            self.assertIsNotNone(info["uploaded_at"])
        if info["type"] == "file":
            self.assertIsNotNone(info["mtime"])
            self.assertGreater(info["size"], 0)  # not empty


class HfFileSystemRepositoryROTests(_HfFileSystemRepositoryChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        super(_HfFileSystemBaseROTests, cls).setUpClass()

        # Create dummy repo
        repo_url = cls.api.create_repo(repo_name(), repo_type="dataset")
        cls.repo_id = repo_url.repo_id
        cls.hf_path = f"datasets/{cls.repo_id}"

        # Upload files
        cls.api.upload_file(
            path_or_fileobj=b"dummy binary data on pr",
            path_in_repo="data/binary_data_for_pr.bin",
            repo_id=cls.repo_id,
            repo_type="dataset",
            create_pr=True,
        )
        cls.api.upload_file(
            path_or_fileobj="dummy text data".encode("utf-8"),
            path_in_repo="data/text_data.txt",
            repo_id=cls.repo_id,
            repo_type="dataset",
        )
        cls.api.upload_file(
            path_or_fileobj=b"dummy binary data",
            path_in_repo="data/binary_data.bin",
            repo_id=cls.repo_id,
            repo_type="dataset",
        )
        cls.api.upload_file(
            path_or_fileobj="# Dataset card".encode("utf-8"),
            path_in_repo="README.md",
            repo_id=cls.repo_id,
            repo_type="dataset",
        )
        cls.api.delete_file(
            path_in_repo=".gitattributes",
            repo_id=cls.repo_id,
            repo_type="dataset",
        )

        cls.readme_file_path = "README.md"
        cls.readme_file = cls.hf_path + "/" + cls.readme_file_path
        cls.text_file_path = "data/text_data.txt"
        cls.text_file = cls.hf_path + "/" + cls.text_file_path

    @classmethod
    def tearDownClass(cls):
        super(_HfFileSystemBaseROTests, cls).tearDownClass()
        cls.api.delete_repo(cls.repo_id, repo_type="dataset")

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

    def test_glob_with_revision(self):
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/*", revision="main")),
            sorted([self.readme_file, self.hf_path + "/data"]),
        )
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "@main" + "/*")),
            sorted([self.hf_path + "@main/" + self.readme_file_path, self.hf_path + "@main" + "/data"]),
        )
        self.assertEqual(
            self.hffs.glob(self.hf_path + "@refs%2Fpr%2F1" + "/data/*"),
            [self.hf_path + "@refs%2Fpr%2F1" + "/data/binary_data_for_pr.bin"],
        )
        self.assertEqual(
            self.hffs.glob(self.hf_path + "@refs/pr/1" + "/data/*"),
            [self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin"],
        )
        self.assertEqual(
            self.hffs.glob(self.hf_path + "/data/*", revision="refs/pr/1"),
            [self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin"],
        )

        self.assertInfoIsNotExpanded(
            self.hffs.dircache[self.hf_path + "@main"][0]
        )  # no detail -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True, expand_info=False)
        self.assertIsInstance(files, dict)
        self.assertEqual(len(files), 2)
        keys = sorted(files)
        self.assertTrue(
            files[keys[0]]["name"].endswith(self.readme_file_path) and files[keys[1]]["name"].endswith("/data")
        )
        self.assertInfoIsNotExpanded(
            self.hffs.dircache[self.hf_path + "@main"][0]
        )  # detail but no expand info -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True)
        self.assertIsInstance(files, dict)
        self.assertEqual(len(files), 2)
        keys = sorted(files)
        self.assertTrue(
            files[keys[0]]["name"].endswith(self.readme_file_path) and files[keys[1]]["name"].endswith("/data")
        )
        self.assertInfoIsNotExpanded(files[keys[0]])

    def test_read_file_with_revision(self):
        with self.hffs.open(self.hf_path + "/data/binary_data_for_pr.bin", "rb", revision="refs/pr/1") as f:
            self.assertEqual(f.read(), b"dummy binary data on pr")

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
            with self.subTest(test_name):
                self.assertEqual(len(files), 1)  # only one file in PR
                self.assertEqual(files[0]["type"], "file")
                self.assertTrue(files[0]["name"].endswith("/data/binary_data_for_pr.bin"))  # PR file
                if "quoted_rev_in_path" in test_name:
                    self.assertIn("@refs%2Fpr%2F1", files[0]["name"])
                elif "rev_in_path" in test_name:
                    self.assertIn("@refs/pr/1", files[0]["name"])


class HfFileSystemRepositoryRWTests(_HfFileSystemRepositoryChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

        # Create dummy repo
        repo_url = self.api.create_repo(repo_name(), repo_type="dataset")
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

    def tearDown(self):
        self.api.delete_repo(self.repo_id, repo_type="dataset")

    def test_remove_file_with_revision(self):
        self.hffs.rm_file(self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin")
        self.assertEqual(self.hffs.glob(self.hf_path + "@refs/pr/1" + "/data/*"), [])

    def test_remove_directory_with_revision(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "/data", self.hffs.ls(self.hf_path))
        self.hffs.rm(self.hf_path + "@refs/pr/1" + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "@refs/pr/1" + "/data", self.hffs.ls(self.hf_path))

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
        self.assertEqual(out, files)


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


def test_exists_after_repo_deletion():
    """Test that exists() correctly reflects repository deletion."""
    # Initialize with staging endpoint and skip cache
    hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)
    api = hffs._api

    # Create a new repo
    temp_repo_id = repo_name()
    repo_url = api.create_repo(temp_repo_id)
    repo_id = repo_url.repo_id
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


@with_production_testing
def test_hf_file_system_file_can_handle_gzipped_file():
    """Test that HfFileSystemStreamFile.read() can handle gzipped files."""
    fs = HfFileSystem(endpoint=constants.ENDPOINT)
    # As of July 2025, the math_qa.py file is gzipped when queried from production:
    with fs.open("datasets/allenai/math_qa/math_qa.py", "r", encoding="utf-8") as f:
        out = f.read()
    assert "class MathQa" in out


class TestXetMetadataResolution:
    def test_returns_none_when_xet_unavailable(self):
        with patch.object(hffs_mod, "is_xet_available", return_value=False), \
             patch.object(hffs_mod, "get_hf_file_metadata") as mock_meta:
            assert hffs_mod._try_get_xet_metadata("http://u", {}, None) == (None, None)
        mock_meta.assert_not_called()

    def test_returns_xet_file_data_and_size_when_available(self):
        meta = HfFileMetadata(
            commit_hash="c", etag="e", location="loc", size=1234,
            xet_file_data=XetFileData(file_hash="h", refresh_route="r"),
        )
        with patch.object(hffs_mod, "is_xet_available", return_value=True), \
             patch.object(hffs_mod, "get_hf_file_metadata", return_value=meta) as mock_meta:
            xfd, size = hffs_mod._try_get_xet_metadata("http://u", {"authorization": "a"}, "http://endpoint")
        assert xfd == XetFileData(file_hash="h", refresh_route="r")
        assert size == 1234
        _, kwargs = mock_meta.call_args
        assert kwargs["headers"] == {"authorization": "a"}
        assert kwargs["endpoint"] == "http://endpoint"

    def test_returns_none_when_metadata_lookup_fails(self):
        with patch.object(hffs_mod, "is_xet_available", return_value=True), \
             patch.object(hffs_mod, "get_hf_file_metadata", side_effect=RuntimeError("boom")):
            assert hffs_mod._try_get_xet_metadata("http://u", {"authorization": "a"}, None) == (None, None)


class TestHfFileSystemFileXetRouting:
    def _make_file(self):
        fs = HfFileSystem(endpoint="https://hub")
        f = HfFileSystemFile.__new__(HfFileSystemFile)
        # bypass network in __init__; set the attributes _fetch_range relies on
        f.fs = fs
        f.path = "datasets/x/data.parquet"
        f.size = 100
        f._xet_checked = False
        f._xet_group = None
        f._xet_file_hash = None
        f.url = lambda: "https://hub/resolve/data.parquet"
        return f, fs

    def test_fetch_range_uses_xet_when_backed(self):
        f, fs = self._make_file()
        fake_group = object()
        with patch.object(hffs_mod, "_try_get_xet_metadata",
                          return_value=(XetFileData(file_hash="h", refresh_route="r"), 100)), \
             patch.object(hffs_mod, "get_xet_download_stream_group", return_value=fake_group), \
             patch.object(hffs_mod, "xet_download_stream", return_value=iter([b"AB", b"CD"])) as mock_stream, \
             patch.object(fs._api, "_build_hf_headers", return_value={"authorization": "a"}):
            out = f._fetch_range(10, 14)
        assert out == b"ABCD"
        _, kwargs = mock_stream.call_args
        assert kwargs == {"start": 10, "end": 14}
        assert mock_stream.call_args[0] == (fake_group, "h", 100)

    def test_fetch_range_falls_back_to_http_when_not_xet(self):
        f, fs = self._make_file()
        response = MagicMock()
        response.content = b"http-bytes"
        with patch.object(hffs_mod, "_try_get_xet_metadata", return_value=(None, None)), \
             patch.object(hffs_mod, "http_backoff", return_value=response) as mock_http, \
             patch.object(hffs_mod, "hf_raise_for_status"), \
             patch.object(fs._api, "_build_hf_headers", return_value={"authorization": "a"}):
            out = f._fetch_range(0, 10)
        assert out == b"http-bytes"
        mock_http.assert_called_once()

    def test_fetch_range_detects_xet_only_once(self):
        f, fs = self._make_file()
        fake_group = object()
        with patch.object(hffs_mod, "_try_get_xet_metadata",
                          return_value=(XetFileData(file_hash="h", refresh_route="r"), 100)) as mock_meta, \
             patch.object(hffs_mod, "get_xet_download_stream_group", return_value=fake_group), \
             patch.object(hffs_mod, "xet_download_stream", return_value=iter([b"x"])), \
             patch.object(fs._api, "_build_hf_headers", return_value={"authorization": "a"}):
            f._fetch_range(0, 1)
            f._fetch_range(1, 2)
        mock_meta.assert_called_once()  # detection HEAD happens at most once per file


class TestHfFileSystemStreamFileXetRouting:
    def _make_stream_file(self):
        fs = HfFileSystem(endpoint="https://hub")
        f = HfFileSystemStreamFile.__new__(HfFileSystemStreamFile)
        f.fs = fs
        f.path = "datasets/x/data.parquet"
        f.loc = 0
        f.size = None
        f.response = None
        f._stream_opened = False
        f._xet_mode = False
        f._stream_iterator = None
        f._stream_buffer = bytearray()
        f._exit_stack = ExitStack()
        f.url = lambda: "https://hub/resolve/data.parquet"
        return f, fs

    def test_open_connection_uses_xet_when_backed(self):
        f, fs = self._make_stream_file()
        fake_group = object()
        with patch.object(hffs_mod, "_try_get_xet_metadata",
                          return_value=(XetFileData(file_hash="h", refresh_route="r"), 8)), \
             patch.object(hffs_mod, "get_xet_download_stream_group", return_value=fake_group), \
             patch.object(hffs_mod, "xet_download_stream", return_value=iter([b"abcd", b"efgh"])) as mock_stream, \
             patch.object(fs._api, "_build_hf_headers", return_value={"authorization": "a"}):
            data = f.read()
        assert data == b"abcdefgh"
        assert f.response is None
        assert f._xet_mode is True
        assert mock_stream.call_args[0] == (fake_group, "h", 8)
        assert mock_stream.call_args[1] == {"start": None, "end": None}

    def test_open_connection_falls_back_to_http(self):
        f, fs = self._make_stream_file()
        with patch.object(hffs_mod, "_try_get_xet_metadata", return_value=(None, None)), \
             patch.object(hffs_mod, "http_stream_backoff") as mock_http, \
             patch.object(hffs_mod, "hf_raise_for_status"), \
             patch.object(fs._api, "_build_hf_headers", return_value={}):
            response = MagicMock()
            response.iter_bytes.return_value = iter([b"xy"])
            mock_http.return_value.__enter__.return_value = response
            data = f.read()
        assert data == b"xy"
        assert f._xet_mode is False
        mock_http.assert_called_once()

    def test_http_response_none_forces_reopen(self):
        # Offline mirror of test_stream_file_retry: in HTTP mode, dropping `response`
        # mid-stream must trigger a fresh connection on the next read.
        f, fs = self._make_stream_file()
        opened = []

        def fake_http(*args, **kwargs):
            resp = MagicMock()
            resp.iter_bytes.return_value = iter([b"part"])
            opened.append(resp)
            cm = MagicMock()
            cm.__enter__.return_value = resp
            return cm

        with patch.object(hffs_mod, "_try_get_xet_metadata", return_value=(None, None)), \
             patch.object(hffs_mod, "http_stream_backoff", side_effect=fake_http), \
             patch.object(hffs_mod, "hf_raise_for_status"), \
             patch.object(fs._api, "_build_hf_headers", return_value={}):
            assert f.read(4) == b"part"
            f.response = None  # simulate mid-stream failure
            assert f.read(4) == b"part"
        assert len(opened) == 2  # a new connection was opened
        assert f.response is not None

    def test_keyboard_interrupt_aborts_xet_session(self):
        f, fs = self._make_stream_file()

        def boom(*args, **kwargs):
            raise KeyboardInterrupt

        with patch.object(hffs_mod, "_try_get_xet_metadata",
                          return_value=(XetFileData(file_hash="h", refresh_route="r"), 8)), \
             patch.object(hffs_mod, "get_xet_download_stream_group", return_value=object()), \
             patch.object(hffs_mod, "xet_download_stream", side_effect=boom), \
             patch.object(fs._api, "_build_hf_headers", return_value={}), \
             patch("huggingface_hub.utils._xet.abort_xet_session") as mock_abort:
            with pytest.raises(KeyboardInterrupt):
                f.read()
        mock_abort.assert_called_once()
