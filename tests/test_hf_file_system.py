import datetime
import unittest
from typing import Optional
from unittest.mock import patch

import fsspec
import pytest

from huggingface_hub.hf_file_system import HfFileSystem
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


class HfFileSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Register `HfFileSystem` as a `fsspec` filesystem if not already registered."""
        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN)
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

    def tearDown(self):
        self.api.delete_repo(self.repo_id, repo_type="dataset")

    def test_glob(self):
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

    def test_file_type(self):
        self.assertTrue(
            self.hffs.isdir(self.hf_path + "/data") and not self.hffs.isdir(self.hf_path + "/.gitattributes")
        )
        self.assertTrue(
            self.hffs.isfile(self.hf_path + "/data/text_data.txt") and not self.hffs.isfile(self.hf_path + "/data")
        )

    def test_remove_file(self):
        self.hffs.rm_file(self.hf_path + "/data/text_data.txt")
        self.assertEqual(self.hffs.glob(self.hf_path + "/data/*"), [self.hf_path + "/data/binary_data.bin"])

    def test_remove_directory(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "/data", self.hffs.ls(self.hf_path))

    def test_read_file(self):
        with self.hffs.open(self.hf_path + "/data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")

    def test_write_file(self):
        data = "new text data"
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "w") as f:
            f.write(data)
        self.assertIn(self.hf_path + "/data/new_text_data.txt", self.hffs.glob(self.hf_path + "/data/*"))
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "r") as f:
            self.assertEqual(f.read(), data)

    def test_write_file_multiple_chunks(self):
        # TODO: try with files between 10 and 50MB (as of 16 March 2023 I was getting 504 errors on hub-ci)
        data = "a" * (4 << 20)  # 4MB
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "w") as f:
            for _ in range(2):  # 8MB in total
                f.write(data)

        self.assertIn(self.hf_path + "/data/new_text_data_big.txt", self.hffs.glob(self.hf_path + "/data/*"))
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "r") as f:
            for _ in range(2):
                self.assertEqual(f.read(len(data)), data)

    @unittest.skip("Not implemented yet")
    def test_append_file(self):
        with self.hffs.open(self.hf_path + "/data/text_data.txt", "a") as f:
            f.write(" appended text")

        with self.hffs.open(self.hf_path + "/data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data appended text")

    def test_copy_file(self):
        # Non-LFS file
        self.assertIsNone(self.hffs.info(self.hf_path + "/data/text_data.txt")["lfs"])
        self.hffs.cp_file(self.hf_path + "/data/text_data.txt", self.hf_path + "/data/text_data_copy.txt")
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
        self.assertIsInstance(self.hffs.modified(self.hf_path + "/data/text_data.txt"), datetime.datetime)
        # should fail on a non-existing file
        with self.assertRaises(FileNotFoundError):
            self.hffs.modified(self.hf_path + "/data/not_existing_file.txt")
        # should fail on a directory
        with self.assertRaises(IsADirectoryError):
            self.hffs.modified(self.hf_path + "/data")

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
        self.assertEqual(paths, [self.hf_path + "/data/text_data.txt"])

        fs, _, paths = fsspec.get_fs_token_paths(f"hf://{self.repo_id}/data/text_data.txt")
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(paths, [f"{self.repo_id}/data/text_data.txt"])

    def test_list_root_directory_no_revision(self):
        files = self.hffs.ls(self.hf_path)
        self.assertEqual(len(files), 2)

        self.assertEqual(files[0]["type"], "directory")
        self.assertEqual(files[0]["size"], 0)
        self.assertTrue(files[0]["name"].endswith("/data"))

        self.assertEqual(files[1]["type"], "file")
        self.assertGreater(files[1]["size"], 0)  # not empty
        self.assertTrue(files[1]["name"].endswith("/.gitattributes"))

    def test_list_data_directory_no_revision(self):
        files = self.hffs.ls(self.hf_path + "/data")
        self.assertEqual(len(files), 2)

        self.assertEqual(files[0]["type"], "file")
        self.assertGreater(files[0]["size"], 0)  # not empty
        self.assertTrue(files[0]["name"].endswith("/data/binary_data.bin"))
        self.assertIsNotNone(files[0]["lfs"])
        self.assertIn("oid", files[0]["lfs"])
        self.assertIn("size", files[0]["lfs"])
        self.assertIn("pointerSize", files[0]["lfs"])

        self.assertEqual(files[1]["type"], "file")
        self.assertGreater(files[1]["size"], 0)  # not empty
        self.assertTrue(files[1]["name"].endswith("/data/text_data.txt"))
        self.assertIsNone(files[1]["lfs"])

    def test_list_data_directory_with_revision(self):
        files = self.hffs.ls(self.hf_path + "@refs%2Fpr%2F1" + "/data")

        for test_name, files in {
            "rev_in_path": self.hffs.ls(self.hf_path + "@refs%2Fpr%2F1" + "/data"),
            "rev_as_arg": self.hffs.ls(self.hf_path + "/data", revision="refs/pr/1"),
            "rev_in_path_and_as_arg": self.hffs.ls(self.hf_path + "@refs%2Fpr%2F1" + "/data", revision="refs/pr/1"),
        }.items():
            with self.subTest(test_name):
                self.assertEqual(len(files), 1)  # only one file in PR
                self.assertEqual(files[0]["type"], "file")
                self.assertTrue(files[0]["name"].endswith("/data/binary_data_for_pr.bin"))  # PR file


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
        ("hf://username/my_model@refs/pr/10", None, "model", "username/my_model", "refs/pr/10"),
        ("hf://username/my_model@refs/pr/10", "refs/pr/10", "model", "username/my_model", "refs/pr/10"),
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
            raise RepositoryNotFoundError(repo_id)
        if revision is not None and revision not in ["main", "dev", "refs"] and not revision.startswith("refs/"):
            raise RevisionNotFoundError(revision)

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
        fs.glob(not_supported_path + "/")
    with pytest.raises(NotImplementedError):
        fs.open(not_supported_path)
