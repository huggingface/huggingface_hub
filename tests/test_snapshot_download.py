import os
import shutil
import unittest

import requests
from huggingface_hub import HfApi, Repository, snapshot_download
from huggingface_hub.utils import (
    HfFolder,
    RepositoryNotFoundError,
    SoftTemporaryDirectory,
    logging,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import (
    expect_deprecation,
    repo_name,
    retry_endpoint,
    set_write_permission_and_retry,
)


logger = logging.get_logger(__name__)

REPO_NAME = repo_name("dummy-hf-hub")


class SnapshotDownloadTests(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

    @retry_endpoint
    def setUp(self) -> None:
        if os.path.exists(REPO_NAME):
            shutil.rmtree(REPO_NAME, onerror=set_write_permission_and_retry)
        logger.info(f"Does {REPO_NAME} exist: {os.path.exists(REPO_NAME)}")

        try:
            self._api.delete_repo(repo_id=REPO_NAME)
        except RepositoryNotFoundError:
            pass
        self._api.create_repo(f"{USER}/{REPO_NAME}")

        repo = Repository(
            REPO_NAME,
            clone_from=f"{USER}/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with repo.commit("Add file to main branch"):
            with open("dummy_file.txt", "w+") as f:
                f.write("v1")

        self.first_commit_hash = repo.git_head_hash()

        with repo.commit("Add file to main branch"):
            with open("dummy_file.txt", "w+") as f:
                f.write("v2")
            with open("dummy_file_2.txt", "w+") as f:
                f.write("v3")

        self.second_commit_hash = repo.git_head_hash()

        with repo.commit("Add file to other branch", branch="other"):
            with open("dummy_file_2.txt", "w+") as f:
                f.write("v4")

        self.third_commit_hash = repo.git_head_hash()

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=REPO_NAME)
        shutil.rmtree(REPO_NAME)

    def test_download_model(self):
        # Test `main` branch
        with SoftTemporaryDirectory() as tmpdirname:
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}", revision="main", cache_dir=tmpdirname
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.second_commit_hash in storage_folder)

        # Test with specific revision
        with SoftTemporaryDirectory() as tmpdirname:
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision=self.first_commit_hash,
                cache_dir=tmpdirname,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 2)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v1")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.first_commit_hash in storage_folder)

    def test_download_private_model(self):
        self._api.update_repo_visibility(repo_id=REPO_NAME, private=True)

        # Test download fails without token
        with SoftTemporaryDirectory() as tmpdirname:
            with self.assertRaisesRegex(
                requests.exceptions.HTTPError, "401 Client Error"
            ):
                _ = snapshot_download(
                    f"{USER}/{REPO_NAME}", revision="main", cache_dir=tmpdirname
                )

        # Test we can download with token from cache
        with SoftTemporaryDirectory() as tmpdirname:
            HfFolder.save_token(self._token)
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="main",
                cache_dir=tmpdirname,
                use_auth_token=True,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.second_commit_hash in storage_folder)

        # Test we can download with explicit token
        with SoftTemporaryDirectory() as tmpdirname:
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="main",
                cache_dir=tmpdirname,
                use_auth_token=self._token,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.second_commit_hash in storage_folder)

        self._api.update_repo_visibility(repo_id=REPO_NAME, private=False)

    def test_download_model_local_only(self):
        # Test no branch specified
        with SoftTemporaryDirectory() as tmpdirname:
            # first download folder to cache it
            snapshot_download(f"{USER}/{REPO_NAME}", cache_dir=tmpdirname)

            # now load from cache
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                cache_dir=tmpdirname,
                local_files_only=True,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.second_commit_hash in storage_folder)

        # Test with specific revision branch
        with SoftTemporaryDirectory() as tmpdirname:
            # first download folder to cache it
            snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="other",
                cache_dir=tmpdirname,
            )

            # now load from cache
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="other",
                cache_dir=tmpdirname,
                local_files_only=True,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.third_commit_hash in storage_folder)

        # Test with specific revision hash
        with SoftTemporaryDirectory() as tmpdirname:
            # first download folder to cache it
            snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision=self.first_commit_hash,
                cache_dir=tmpdirname,
            )

            # now load from cache
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision=self.first_commit_hash,
                cache_dir=tmpdirname,
                local_files_only=True,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 2)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v1")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.first_commit_hash in storage_folder)

    def test_download_model_local_only_multiple(self):
        # Test `main` branch
        with SoftTemporaryDirectory() as tmpdirname:
            # download both from branch and from commit
            snapshot_download(
                f"{USER}/{REPO_NAME}",
                cache_dir=tmpdirname,
            )

            snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision=self.first_commit_hash,
                cache_dir=tmpdirname,
            )

        # cache multiple commits and make sure correct commit is taken
        with SoftTemporaryDirectory() as tmpdirname:
            # first download folder to cache it
            snapshot_download(
                f"{USER}/{REPO_NAME}",
                cache_dir=tmpdirname,
            )

            # now load folder from another branch
            snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="other",
                cache_dir=tmpdirname,
            )

            # now make sure that loading "main" branch gives correct branch
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                cache_dir=tmpdirname,
                local_files_only=True,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 3)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue(".gitattributes" in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the 2nd commit sha and not the 3rd
            self.assertTrue(self.second_commit_hash in storage_folder)

    def check_download_model_with_pattern(self, pattern, allow=True):
        # Test `main` branch
        allow_patterns = pattern if allow else None
        ignore_patterns = pattern if not allow else None

        with SoftTemporaryDirectory() as tmpdirname:
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="main",
                cache_dir=tmpdirname,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            self.assertEqual(len(folder_contents), 2)
            self.assertTrue("dummy_file.txt" in folder_contents)
            self.assertTrue("dummy_file_2.txt" in folder_contents)
            self.assertTrue(".gitattributes" not in folder_contents)

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                self.assertEqual(contents, "v2")

            # folder name contains the revision's commit sha.
            self.assertTrue(self.second_commit_hash in storage_folder)

    def test_download_model_with_allow_pattern(self):
        self.check_download_model_with_pattern("*.txt")

    def test_download_model_with_allow_pattern_list(self):
        self.check_download_model_with_pattern(["dummy_file.txt", "dummy_file_2.txt"])

    def test_download_model_with_ignore_pattern(self):
        self.check_download_model_with_pattern(".gitattributes", allow=False)

    def test_download_model_with_ignore_pattern_list(self):
        self.check_download_model_with_pattern(["*.git*", "*.pt"], allow=False)
