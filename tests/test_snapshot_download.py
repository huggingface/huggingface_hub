import os
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.utils import HfFolder, SoftTemporaryDirectory

from .testing_constants import TOKEN
from .testing_utils import repo_name


class SnapshotDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls.api = HfApi(token=TOKEN)
        cls.repo_id = cls.api.create_repo(repo_name("snapshot-download")).repo_id

        # First commit on `main`
        cls.first_commit_hash = cls.api.create_commit(
            repo_id=cls.repo_id,
            operations=[
                CommitOperationAdd(path_in_repo="dummy_file.txt", path_or_fileobj=b"v1"),
                CommitOperationAdd(path_in_repo="subpath/file.txt", path_or_fileobj=b"content in subpath"),
            ],
            commit_message="Add file to main branch",
        ).oid

        # Second commit on `main`
        cls.second_commit_hash = cls.api.create_commit(
            repo_id=cls.repo_id,
            operations=[
                CommitOperationAdd(path_in_repo="dummy_file.txt", path_or_fileobj=b"v2"),
                CommitOperationAdd(path_in_repo="dummy_file_2.txt", path_or_fileobj=b"v3"),
            ],
            commit_message="Add file to main branch",
        ).oid

        # Third commit on `other`
        cls.api.create_branch(repo_id=cls.repo_id, branch="other")
        cls.third_commit_hash = cls.api.create_commit(
            repo_id=cls.repo_id,
            operations=[
                CommitOperationAdd(path_in_repo="dummy_file_2.txt", path_or_fileobj=b"v4"),
            ],
            commit_message="Add file to other branch",
            revision="other",
        ).oid

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.delete_repo(repo_id=cls.repo_id)

    def test_download_model(self):
        # Test `main` branch
        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir)

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 4)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.second_commit_hash in storage_folder)

        # Test with specific revision
        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(
                self.repo_id,
                revision=self.first_commit_hash,
                cache_dir=tmpdir,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v1")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.first_commit_hash in storage_folder)

    def test_download_private_model(self):
        self.api.update_repo_visibility(repo_id=self.repo_id, private=True)

        # Test download fails without token
        with SoftTemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(requests.exceptions.HTTPError, "401 Client Error"):
                _ = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir)

        # Test we can download with token from cache
        with SoftTemporaryDirectory() as tmpdir:
            HfFolder.save_token(TOKEN)
            storage_folder = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir)
            self.assertTrue(self.second_commit_hash in storage_folder)

        # Test we can download with explicit token
        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir, token=TOKEN)
            self.assertTrue(self.second_commit_hash in storage_folder)

        self.api.update_repo_visibility(repo_id=self.repo_id, private=False)

    def test_download_model_local_only(self):
        # Test no branch specified
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to cache it
            snapshot_download(self.repo_id, cache_dir=tmpdir)
            # now load from cache
            storage_folder = snapshot_download(self.repo_id, cache_dir=tmpdir, local_files_only=True)
            self.assertTrue(self.second_commit_hash in storage_folder)  # has expected revision

        # Test with specific revision branch
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to cache it
            snapshot_download(self.repo_id, revision="other", cache_dir=tmpdir)
            # now load from cache
            storage_folder = snapshot_download(self.repo_id, revision="other", cache_dir=tmpdir, local_files_only=True)
            self.assertTrue(self.third_commit_hash in storage_folder)  # has expected revision

        # Test with specific revision hash
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to cache it
            snapshot_download(self.repo_id, revision=self.first_commit_hash, cache_dir=tmpdir)
            # now load from cache
            storage_folder = snapshot_download(
                self.repo_id, revision=self.first_commit_hash, cache_dir=tmpdir, local_files_only=True
            )
            self.assertTrue(self.first_commit_hash in storage_folder)  # has expected revision

    def test_download_model_local_only_multiple(self):
        # cache multiple commits and make sure correct commit is taken
        with SoftTemporaryDirectory() as tmpdir:
            # download folder from main and other to cache it
            snapshot_download(self.repo_id, cache_dir=tmpdir)
            snapshot_download(self.repo_id, revision="other", cache_dir=tmpdir)

            # now make sure that loading "main" branch gives correct branch
            # folder name contains the 2nd commit sha and not the 3rd
            storage_folder = snapshot_download(self.repo_id, cache_dir=tmpdir, local_files_only=True)
            self.assertTrue(self.second_commit_hash in storage_folder)

    def check_download_model_with_pattern(self, pattern, allow=True):
        # Test `main` branch
        allow_patterns = pattern if allow else None
        ignore_patterns = pattern if not allow else None

        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(
                self.repo_id,
                revision="main",
                cache_dir=tmpdir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

            # folder contains the three text files but not the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" not in folder_contents)

    def test_download_model_with_allow_pattern(self):
        self.check_download_model_with_pattern("*.txt")

    def test_download_model_with_allow_pattern_list(self):
        self.check_download_model_with_pattern(["dummy_file.txt", "dummy_file_2.txt", "subpath/*"])

    def test_download_model_with_ignore_pattern(self):
        self.check_download_model_with_pattern(".gitattributes", allow=False)

    def test_download_model_with_ignore_pattern_list(self):
        self.check_download_model_with_pattern(["*.git*", "*.pt"], allow=False)

    @patch("huggingface_hub.constants.HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD", 10)  # >10b => "big file"
    def test_download_to_local_dir(self) -> None:
        """Download a repository to local dir.

        Cache dir is used and symlinks are adding to local dir. This test is here to check once the normal behavior
        with snapshot_download. More combinations of cache_dir/local_dir/use_symlinks are tested separately in
        `test_file_download.py`.
        """
        with SoftTemporaryDirectory() as cache_dir:
            with SoftTemporaryDirectory() as local_dir:
                returned_path = snapshot_download(self.repo_id, cache_dir=cache_dir, local_dir=local_dir)

                # Files have been downloaded
                self.assertTrue((Path(local_dir) / "dummy_file.txt").is_file())
                self.assertTrue((Path(local_dir) / "dummy_file_2.txt").is_file())

                # Files are small so duplicated from cache (no symlinks)
                self.assertFalse((Path(local_dir) / "dummy_file.txt").is_symlink())  # smaller than 10b => duplicated
                self.assertFalse((Path(local_dir) / "dummy_file_2.txt").is_symlink())  # smaller than 10b => duplicated

                # File structure is preserved (+check content)
                subpath_file = Path(local_dir) / "subpath" / "file.txt"
                self.assertTrue(subpath_file.is_file())
                self.assertEqual(subpath_file.read_text(), "content in subpath")
                if os.name != "nt":
                    self.assertTrue(subpath_file.is_symlink())  # bigger than 10b => symlinked

                # Check returns local dir and not cache dir
                self.assertEqual(returned_path, local_dir)
