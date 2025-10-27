import copy
import datetime
import io
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import fsspec
import pytest

from huggingface_hub import constants, hf_file_system
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_file_system import (
    HfFileSystem,
    HfFileSystemFile,
    HfFileSystemStreamFile,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import OfflineSimulationMode, offline, repo_name, with_production_testing


class HfFileSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Register `HfFileSystem` as a `fsspec` filesystem if not already registered."""
        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)
        self.api = self.hffs._api

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

        self.text_file_path_in_repo = "data/text_data.txt"
        self.text_file = self.hf_path + "/" + self.text_file_path_in_repo

    def tearDown(self):
        self.api.delete_repo(self.repo_id, repo_type="dataset")

    def test_info(self):
        root_dir = self.hffs.info(self.hf_path)
        self.assertEqual(root_dir["type"], "directory")
        self.assertEqual(root_dir["size"], 0)
        self.assertTrue(root_dir["name"].endswith(self.repo_id))
        self.assertIsNone(root_dir["last_commit"])

        data_dir = self.hffs.info(self.hf_path + "/data")
        self.assertEqual(data_dir["type"], "directory")
        self.assertEqual(data_dir["size"], 0)
        self.assertTrue(data_dir["name"].endswith("/data"))
        self.assertIsNone(data_dir["last_commit"])
        self.assertIsNotNone(data_dir["tree_id"])

        text_data_file = self.hffs.info(self.text_file)
        self.assertEqual(text_data_file["type"], "file")
        self.assertGreater(text_data_file["size"], 0)  # not empty
        self.assertTrue(text_data_file["name"].endswith("/data/text_data.txt"))
        self.assertIsNone(text_data_file["lfs"])
        self.assertIsNone(text_data_file["last_commit"])
        self.assertIsNotNone(text_data_file["blob_id"])
        self.assertIn("security", text_data_file)  # the staging endpoint does not run security checks

        # cached info
        self.assertEqual(self.hffs.info(self.text_file), text_data_file)

    def test_glob(self):
        self.assertEqual(
            self.hffs.glob(self.hf_path + "/.gitattributes"),
            [self.hf_path + "/.gitattributes"],
        )
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/*")),
            sorted([self.hf_path + "/.gitattributes", self.hf_path + "/data"]),
        )

        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/*", revision="main")),
            sorted([self.hf_path + "/.gitattributes", self.hf_path + "/data"]),
        )
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "@main" + "/*")),
            sorted([self.hf_path + "@main" + "/.gitattributes", self.hf_path + "@main" + "/data"]),
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

        self.assertIsNone(
            self.hffs.dircache[self.hf_path + "@main"][0]["last_commit"]
        )  # no detail -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True, expand_info=False)
        self.assertIsInstance(files, dict)
        self.assertEqual(len(files), 2)
        keys = sorted(files)
        self.assertTrue(
            files[keys[0]]["name"].endswith("/.gitattributes") and files[keys[1]]["name"].endswith("/data")
        )
        self.assertIsNone(
            self.hffs.dircache[self.hf_path + "@main"][0]["last_commit"]
        )  # detail but no expand info -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True)
        self.assertIsInstance(files, dict)
        self.assertEqual(len(files), 2)
        keys = sorted(files)
        self.assertTrue(
            files[keys[0]]["name"].endswith("/.gitattributes") and files[keys[1]]["name"].endswith("/data")
        )
        self.assertIsNone(files[keys[0]]["last_commit"])

    def test_url(self):
        self.assertEqual(
            self.hffs.url(self.text_file),
            f"{ENDPOINT_STAGING}/datasets/{self.repo_id}/resolve/main/data/text_data.txt",
        )
        self.assertEqual(
            self.hffs.url(self.hf_path + "/data"),
            f"{ENDPOINT_STAGING}/datasets/{self.repo_id}/tree/main/data",
        )

    def test_file_type(self):
        self.assertTrue(
            self.hffs.isdir(self.hf_path + "/data") and not self.hffs.isdir(self.hf_path + "/.gitattributes")
        )
        self.assertTrue(self.hffs.isfile(self.text_file) and not self.hffs.isfile(self.hf_path + "/data"))

    def test_remove_file(self):
        self.hffs.rm_file(self.text_file)
        self.assertEqual(self.hffs.glob(self.hf_path + "/data/*"), [self.hf_path + "/data/binary_data.bin"])
        self.hffs.rm_file(self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin")
        self.assertEqual(self.hffs.glob(self.hf_path + "@refs/pr/1" + "/data/*"), [])

    def test_remove_directory(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "/data", self.hffs.ls(self.hf_path))
        self.hffs.rm(self.hf_path + "@refs/pr/1" + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "@refs/pr/1" + "/data", self.hffs.ls(self.hf_path))

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

    def test_read_file_with_revision(self):
        with self.hffs.open(self.hf_path + "/data/binary_data_for_pr.bin", "rb", revision="refs/pr/1") as f:
            self.assertEqual(f.read(), b"dummy binary data on pr")

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
        self.assertIsNone(self.hffs.info(self.text_file)["lfs"])
        self.hffs.cp_file(self.text_file, self.hf_path + "/data/text_data_copy.txt")
        with self.hffs.open(self.hf_path + "/data/text_data_copy.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")
        self.assertIsNone(self.hffs.info(self.hf_path + "/data/text_data_copy.txt")["lfs"])
        # LFS file
        self.assertIsNotNone(self.hffs.info(self.hf_path + "/data/binary_data.bin")["lfs"])
        self.hffs.cp_file(self.hf_path + "/data/binary_data.bin", self.hf_path + "/data/binary_data_copy.bin")
        with self.hffs.open(self.hf_path + "/data/binary_data_copy.bin", "rb") as f:
            self.assertEqual(f.read(), b"dummy binary data")
        self.assertIsNotNone(self.hffs.info(self.hf_path + "/data/binary_data_copy.bin")["lfs"])

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
            f"hf://datasets/{self.repo_id}/data/text_data.txt",
            storage_options={
                "endpoint": ENDPOINT_STAGING,
                "token": TOKEN,
            },
        )
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(fs._api.endpoint, ENDPOINT_STAGING)
        self.assertEqual(fs.token, TOKEN)
        self.assertEqual(paths, [self.text_file])

        fs, _, paths = fsspec.get_fs_token_paths(f"hf://{self.repo_id}/data/text_data.txt")
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(paths, [f"{self.repo_id}/data/text_data.txt"])

    def test_list_root_directory_no_revision(self):
        files = self.hffs.ls(self.hf_path)
        self.assertEqual(len(files), 2)

        self.assertEqual(files[0]["type"], "directory")
        self.assertEqual(files[0]["size"], 0)
        self.assertTrue(files[0]["name"].endswith("/data"))
        self.assertIsNone(files[0]["last_commit"])
        self.assertIsNotNone(files[0]["tree_id"])

        self.assertEqual(files[1]["type"], "file")
        self.assertGreater(files[1]["size"], 0)  # not empty
        self.assertTrue(files[1]["name"].endswith("/.gitattributes"))
        self.assertIsNone(files[1]["last_commit"])
        self.assertIsNotNone(files[1]["blob_id"])
        self.assertIn("security", files[1])  # the staging endpoint does not run security checks

    def test_list_data_directory_no_revision(self):
        files = self.hffs.ls(self.hf_path + "/data")
        self.assertEqual(len(files), 2)

        self.assertEqual(files[0]["type"], "file")
        self.assertGreater(files[0]["size"], 0)  # not empty
        self.assertTrue(files[0]["name"].endswith("/data/binary_data.bin"))
        self.assertIsNotNone(files[0]["lfs"])
        self.assertIn("sha256", files[0]["lfs"])
        self.assertIn("size", files[0]["lfs"])
        self.assertIn("pointer_size", files[0]["lfs"])
        self.assertIsNone(files[0]["last_commit"])
        self.assertIsNotNone(files[0]["blob_id"])
        self.assertIn("security", files[0])  # the staging endpoint does not run security checks

        self.assertEqual(files[1]["type"], "file")
        self.assertGreater(files[1]["size"], 0)  # not empty
        self.assertTrue(files[1]["name"].endswith("/data/text_data.txt"))
        self.assertIsNone(files[1]["lfs"])
        self.assertIsNone(files[1]["last_commit"])
        self.assertIsNotNone(files[1]["blob_id"])
        self.assertIn("security", files[1])  # the staging endpoint does not run security checks

    def test_list_data_file_no_revision(self):
        files = self.hffs.ls(self.text_file)
        self.assertEqual(len(files), 1)

        self.assertEqual(files[0]["type"], "file")
        self.assertGreater(files[0]["size"], 0)  # not empty
        self.assertTrue(files[0]["name"].endswith("/data/text_data.txt"))
        self.assertIsNone(files[0]["lfs"])
        self.assertIsNone(files[0]["last_commit"])
        self.assertIsNotNone(files[0]["blob_id"])
        self.assertIn("security", files[0])  # the staging endpoint does not run security checks

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

    def test_list_root_directory_no_revision_no_detail_then_with_detail(self):
        files = self.hffs.ls(self.hf_path, detail=False)
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0].endswith("/data") and files[1].endswith("/.gitattributes"))
        self.assertIsNone(self.hffs.dircache[self.hf_path][0]["last_commit"])  # no detail -> no last_commit in cache

        files = self.hffs.ls(self.hf_path, detail=True)
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0]["name"].endswith("/data") and files[1]["name"].endswith("/.gitattributes"))
        self.assertIsNone(
            self.hffs.dircache[self.hf_path][0]["last_commit"]
        )  # no expand_info -> no last_commit in cache

        files = self.hffs.ls(self.hf_path, detail=True, expand_info=True)
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0]["name"].endswith("/data") and files[1]["name"].endswith("/.gitattributes"))
        self.assertIsNotNone(self.hffs.dircache[self.hf_path][0]["last_commit"])

    def test_find_root_directory_no_revision(self):
        files = self.hffs.find(self.hf_path, detail=False)
        self.assertEqual(
            files, self.hffs.ls(self.hf_path, detail=False)[1:] + self.hffs.ls(self.hf_path + "/data", detail=False)
        )

        files = self.hffs.find(self.hf_path, detail=True)
        self.assertEqual(
            files,
            {
                f["name"]: f
                for f in self.hffs.ls(self.hf_path, detail=True)[1:]
                + self.hffs.ls(self.hf_path + "/data", detail=True)
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

    def test_find_root_directory_no_revision_with_incomplete_cache(self):
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

    def test_find_data_file_no_revision(self):
        files = self.hffs.find(self.text_file, detail=False)
        self.assertEqual(files, [self.text_file])

    def test_find_maxdepth(self):
        text_file_depth = self.text_file_path_in_repo.count("/") + 1
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
            assert list(fs._repo_and_revision_exists_cache)[0][1] == self.repo_id
            assert fs.isfile(self.text_file)


@pytest.mark.parametrize("path_in_repo", ["", "file.txt", "path/to/file"])
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
        assert (
            resolved_path.repo_type,
            resolved_path.repo_id,
            resolved_path.revision,
            resolved_path.path_in_repo,
        ) == (repo_type, repo_id, resolved_revision, path_in_repo)
        if "@" in path:
            assert resolved_path._raw_revision in path


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


def test_resolve_path_with_refs_revision() -> None:
    """
    Testing a very specific edge case where a user has a repo with a revisions named "refs" and a file/directory
    named "pr/10". We can still process them but the user has to use the `revision` argument to disambiguate between
    the two.
    """
    fs = HfFileSystem()
    with mock_repo_info(fs):
        resolved = fs.resolve_path("hf://username/my_model@refs/pr/10", revision="refs")
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


def test_resolve_path_with_non_matching_revisions():
    fs = HfFileSystem()
    with pytest.raises(ValueError):
        fs.resolve_path("gpt2@dev", revision="main")


@pytest.mark.parametrize("not_supported_path", ["", "foo", "datasets", "datasets/foo"])
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


@with_production_testing
def test_hf_file_system_file_can_handle_gzipped_file():
    """Test that HfFileSystemStreamFile.read() can handle gzipped files."""
    fs = HfFileSystem(endpoint=constants.ENDPOINT)
    # As of July 2025, the math_qa.py file is gzipped when queried from production:
    with fs.open("datasets/allenai/math_qa/math_qa.py", "r", encoding="utf-8") as f:
        out = f.read()
    assert "class MathQa" in out
