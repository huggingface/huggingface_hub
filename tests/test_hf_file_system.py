import copy
import datetime
import io
import os
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
from .testing_utils import repo_name, with_production_testing


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

        self.text_file = self.hf_path + "/data/text_data.txt"

    def tearDown(self):
        self.api.delete_repo(self.repo_id, repo_type="dataset")

    def test_info(self):
        root_dir = self.hffs.info(self.hf_path)
        assert root_dir["type"] == "directory"
        assert root_dir["size"] == 0
        assert root_dir["name"].endswith(self.repo_id)
        assert root_dir["last_commit"] is None

        data_dir = self.hffs.info(self.hf_path + "/data")
        assert data_dir["type"] == "directory"
        assert data_dir["size"] == 0
        assert data_dir["name"].endswith("/data")
        assert data_dir["last_commit"] is None
        assert data_dir["tree_id"] is not None

        text_data_file = self.hffs.info(self.text_file)
        assert text_data_file["type"] == "file"
        assert text_data_file["size"] > 0  # not empty
        assert text_data_file["name"].endswith("/data/text_data.txt")
        assert text_data_file["lfs"] is None
        assert text_data_file["last_commit"] is None
        assert text_data_file["blob_id"] is not None
        assert "security" in text_data_file  # the staging endpoint does not run security checks

        # cached info
        assert self.hffs.info(self.text_file) == text_data_file

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
        assert isinstance(files, dict)
        assert len(files) == 2
        keys = sorted(files)
        self.assertTrue(
            files[keys[0]]["name"].endswith("/.gitattributes") and files[keys[1]]["name"].endswith("/data")
        )
        self.assertIsNone(
            self.hffs.dircache[self.hf_path + "@main"][0]["last_commit"]
        )  # detail but no expand info -> no last_commit in cache

        files = self.hffs.glob(self.hf_path + "@main" + "/*", detail=True)
        assert isinstance(files, dict)
        assert len(files) == 2
        keys = sorted(files)
        self.assertTrue(
            files[keys[0]]["name"].endswith("/.gitattributes") and files[keys[1]]["name"].endswith("/data")
        )
        assert files[keys[0]]["last_commit"] is None

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
        assert self.hffs.isfile(self.text_file and not self.hffs.isfile(self.hf_path + "/data"))

    def test_remove_file(self):
        self.hffs.rm_file(self.text_file)
        assert self.hffs.glob(self.hf_path + "/data/*") == [self.hf_path + "/data/binary_data.bin"]
        self.hffs.rm_file(self.hf_path + "@refs/pr/1" + "/data/binary_data_for_pr.bin")
        assert self.hffs.glob(self.hf_path + "@refs/pr/1" + "/data/*") == []

    def test_remove_directory(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        assert self.hf_path + "/data" not in self.hffs.ls(self.hf_path)
        self.hffs.rm(self.hf_path + "@refs/pr/1" + "/data", recursive=True)
        assert self.hf_path + "@refs/pr/1" + "/data" not in self.hffs.ls(self.hf_path)

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

    def test_read_file_with_revision(self):
        with self.hffs.open(self.hf_path + "/data/binary_data_for_pr.bin", "rb", revision="refs/pr/1") as f:
            assert f.read() == b"dummy binary data on pr"

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

    @unittest.skip("Not implemented yet")
    def test_append_file(self):
        with self.hffs.open(self.text_file, "a") as f:
            f.write(" appended text")

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy text data appended text"

    def test_copy_file(self):
        # Non-LFS file
        assert self.hffs.info(self.text_file is None["lfs"])
        self.hffs.cp_file(self.text_file, self.hf_path + "/data/text_data_copy.txt")
        with self.hffs.open(self.hf_path + "/data/text_data_copy.txt", "r") as f:
            assert f.read() == "dummy text data"
        assert self.hffs.info(self.hf_path + "/data/text_data_copy.txt" is None["lfs"])
        # LFS file
        assert self.hffs.info(self.hf_path + "/data/binary_data.bin" is not None["lfs"])
        self.hffs.cp_file(self.hf_path + "/data/binary_data.bin", self.hf_path + "/data/binary_data_copy.bin")
        with self.hffs.open(self.hf_path + "/data/binary_data_copy.bin", "rb") as f:
            assert f.read() == b"dummy binary data"
        assert self.hffs.info(self.hf_path + "/data/binary_data_copy.bin" is not None["lfs"])

    def test_modified_time(self):
        assert isinstance(self.hffs.modified(self.text_file), datetime.datetime)
        assert isinstance(self.hffs.modified(self.hf_path + "/data"), datetime.datetime)
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
        assert isinstance(fs, HfFileSystem)
        assert fs._api.endpoint == ENDPOINT_STAGING
        assert fs.token == TOKEN
        assert paths == [self.text_file]

        fs, _, paths = fsspec.get_fs_token_paths(f"hf://{self.repo_id}/data/text_data.txt")
        assert isinstance(fs, HfFileSystem)
        assert paths == [f"{self.repo_id}/data/text_data.txt"]

    def test_list_root_directory_no_revision(self):
        files = self.hffs.ls(self.hf_path)
        assert len(files) == 2

        assert files[0]["type"] == "directory"
        assert files[0]["size"] == 0
        assert files[0]["name"].endswith("/data")
        assert files[0]["last_commit"] is None
        assert files[0]["tree_id"] is not None

        assert files[1]["type"] == "file"
        assert files[1]["size"] > 0  # not empty
        assert files[1]["name"].endswith("/.gitattributes")
        assert files[1]["last_commit"] is None
        assert files[1]["blob_id"] is not None
        assert "security" in files[1]  # the staging endpoint does not run security checks

    def test_list_data_directory_no_revision(self):
        files = self.hffs.ls(self.hf_path + "/data")
        assert len(files) == 2

        assert files[0]["type"] == "file"
        assert files[0]["size"] > 0  # not empty
        assert files[0]["name"].endswith("/data/binary_data.bin")
        assert files[0]["lfs"] is not None
        assert "sha256" in files[0]["lfs"]
        assert "size" in files[0]["lfs"]
        assert "pointer_size" in files[0]["lfs"]
        assert files[0]["last_commit"] is None
        assert files[0]["blob_id"] is not None
        assert "security" in files[0]  # the staging endpoint does not run security checks

        assert files[1]["type"] == "file"
        assert files[1]["size"] > 0  # not empty
        assert files[1]["name"].endswith("/data/text_data.txt")
        assert files[1]["lfs"] is None
        assert files[1]["last_commit"] is None
        assert files[1]["blob_id"] is not None
        assert "security" in files[1]  # the staging endpoint does not run security checks

    def test_list_data_file_no_revision(self):
        files = self.hffs.ls(self.text_file)
        assert len(files) == 1

        assert files[0]["type"] == "file"
        assert files[0]["size"] > 0  # not empty
        assert files[0]["name"].endswith("/data/text_data.txt")
        assert files[0]["lfs"] is None
        assert files[0]["last_commit"] is None
        assert files[0]["blob_id"] is not None
        assert "security" in files[0]  # the staging endpoint does not run security checks

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
                assert len(files) == 1  # only one file in PR
                assert files[0]["type"] == "file"
                assert files[0]["name"].endswith("/data/binary_data_for_pr.bin")  # PR file
                if "quoted_rev_in_path" in test_name:
                    assert "@refs%2Fpr%2F1" in files[0]["name"]
                elif "rev_in_path" in test_name:
                    assert "@refs/pr/1" in files[0]["name"]

    def test_list_root_directory_no_revision_no_detail_then_with_detail(self):
        files = self.hffs.ls(self.hf_path, detail=False)
        assert len(files) == 2
        assert files[0].endswith("/data" and files[1].endswith("/.gitattributes"))
        assert self.hffs.dircache[self.hf_path][0]["last_commit"] is None  # no detail -> no last_commit in cache

        files = self.hffs.ls(self.hf_path, detail=True)
        assert len(files) == 2
        assert files[0]["name"].endswith("/data" and files[1]["name"].endswith("/.gitattributes"))
        self.assertIsNone(
            self.hffs.dircache[self.hf_path][0]["last_commit"]
        )  # no expand_info -> no last_commit in cache

        files = self.hffs.ls(self.hf_path, detail=True, expand_info=True)
        assert len(files) == 2
        assert files[0]["name"].endswith("/data" and files[1]["name"].endswith("/.gitattributes"))
        assert self.hffs.dircache[self.hf_path][0]["last_commit"] is not None

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
        assert out == files

    def test_find_data_file_no_revision(self):
        files = self.hffs.find(self.text_file, detail=False)
        assert files == [self.text_file]

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

    def test_get_file_with_temporary_folder(self):
        # Test passing a file path works => compatible with hf_transfer
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
