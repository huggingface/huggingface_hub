import copy
import datetime
import io
import multiprocessing
import multiprocessing.pool
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, Optional
from unittest.mock import Mock, patch

import fsspec
import pytest

from huggingface_hub import HfApi, constants, hf_file_system
from huggingface_hub.errors import BucketNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_file_system import (
    HfFileSystem,
    HfFileSystemFile,
    HfFileSystemResolvedBucketPath,
    HfFileSystemResolvedRepositoryPath,
    HfFileSystemStreamFile,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import OfflineSimulationMode, offline, repo_name, requires, with_production_testing


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
            # Simulate that streaming fails mid-way
            f.response = None
            self.assertEqual(f.read(6), b"binary")
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
        # Test that pickling re-populates the HfFileSystem cache and keeps the instance cache attributes
        fs = HfFileSystem()
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


@requires("hf_xet")
class HfFileSystemBucketROTests(_HfFileSystemBucketChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        super(_HfFileSystemBaseROTests, cls).setUpClass()

        # Create dummy bucket
        repo_url = cls.api.create_bucket(repo_name())
        cls.bucket_id = repo_url.bucket_id
        cls.hf_path = f"buckets/{cls.bucket_id}"

        # Upload files
        cls.api.batch_bucket_files(
            cls.bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        cls.readme_file_path = "README.md"
        cls.readme_file = cls.hf_path + "/" + cls.readme_file_path
        cls.text_file_path = "data/text_data.txt"
        cls.text_file = cls.hf_path + "/" + cls.text_file_path

    @classmethod
    def tearDownClass(cls):
        super(_HfFileSystemBaseROTests, cls).tearDownClass()
        cls.api.delete_bucket(cls.bucket_id)

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)


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


@requires("hf_xet")
class HfFileSystemBucketRWTests(_HfFileSystemBucketChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    def setUp(self):
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

    def tearDown(self):
        self.api.delete_bucket(self.bucket_id)

    @unittest.skip("Not implemented yet")
    def test_copy_file(self):
        pass


@pytest.mark.parametrize("path_in_repo", ["", "file.txt", "path/to/file", "path/to/@not-a-revision.txt"])
@pytest.mark.parametrize(
    "root_path,revision,repo_type,repo_id,resolved_revision",
    [
        # Parse without namespace
        ("gpt2", None, "model", "gpt2", "main"),
        ("gpt2", "dev", "model", "gpt2", "dev"),
        ("gpt2@dev", None, "model", "gpt2", "dev"),
        ("datasets/squad", None, "dataset", "squad", "main"),
        ("datasets/squad", "dev", "dataset", "squad", "dev"),
        ("datasets/squad@dev", None, "dataset", "squad", "dev"),
        # Parse with namespace
        ("username/my_model", None, "model", "username/my_model", "main"),
        ("username/my_model", "dev", "model", "username/my_model", "dev"),
        ("username/my_model@dev", None, "model", "username/my_model", "dev"),
        ("datasets/username/my_dataset", None, "dataset", "username/my_dataset", "main"),
        ("datasets/username/my_dataset", "dev", "dataset", "username/my_dataset", "dev"),
        ("datasets/username/my_dataset@dev", None, "dataset", "username/my_dataset", "dev"),
        # Parse with hf:// protocol
        ("hf://gpt2", None, "model", "gpt2", "main"),
        ("hf://gpt2", "dev", "model", "gpt2", "dev"),
        ("hf://gpt2@dev", None, "model", "gpt2", "dev"),
        ("hf://datasets/squad", None, "dataset", "squad", "main"),
        ("hf://datasets/squad", "dev", "dataset", "squad", "dev"),
        ("hf://datasets/squad@dev", None, "dataset", "squad", "dev"),
        # Parse with `refs/convert/parquet` and `refs/pr/(\d)+` revisions.
        # Regression tests for https://github.com/huggingface/huggingface_hub/issues/1710.
        ("datasets/squad@refs/convert/parquet", None, "dataset", "squad", "refs/convert/parquet"),
        (
            "hf://datasets/username/my_dataset@refs/convert/parquet",
            None,
            "dataset",
            "username/my_dataset",
            "refs/convert/parquet",
        ),
        ("gpt2@refs/pr/2", None, "model", "gpt2", "refs/pr/2"),
        ("gpt2@refs%2Fpr%2F2", None, "model", "gpt2", "refs/pr/2"),
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
        ("hf://datasets/squad@dev", None, "datasets/squad@dev"),
        ("datasets/squad@refs/convert/parquet", None, "datasets/squad@refs/convert/parquet"),
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
        if repo_id not in ["gpt2", "squad", "username/my_dataset", "username/my_model"]:
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
        fs.resolve_path("gpt2@dev", revision="main")


@pytest.mark.parametrize("not_supported_path", ["", "foo", "datasets"])
def test_access_repositories_lists(not_supported_path):
    fs = HfFileSystem()
    with pytest.raises(NotImplementedError):
        fs.info(not_supported_path)
    with pytest.raises(NotImplementedError):
        fs.ls(not_supported_path)
    with pytest.raises(NotImplementedError):
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
