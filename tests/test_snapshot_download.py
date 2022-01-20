import os
import shutil
import tempfile
import time
import unittest

import requests
from huggingface_hub import HfApi, Repository
from huggingface_hub.hf_api import HfFolder
from huggingface_hub.snapshot_download import snapshot_download
from tests.testing_constants import ENDPOINT_STAGING, PASS, USER


REPO_NAME = "dummy-hf-hub-{}".format(int(time.time() * 10e3))


class SnapshotDownloadTests(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def setUp(self) -> None:
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
        self._api.delete_repo(name=REPO_NAME, token=self._token)
        shutil.rmtree(REPO_NAME)

    def test_download_model(self):
        # Test `main` branch
        with tempfile.TemporaryDirectory() as tmpdirname:
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
        with tempfile.TemporaryDirectory() as tmpdirname:
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
        self._api.update_repo_visibility(
            token=self._token, name=REPO_NAME, private=True
        )

        # Test download fails without token
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaisesRegex(
                requests.exceptions.HTTPError, "404 Client Error"
            ):
                _ = snapshot_download(
                    f"{USER}/{REPO_NAME}", revision="main", cache_dir=tmpdirname
                )

        # Test we can download with token from cache
        with tempfile.TemporaryDirectory() as tmpdirname:
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
        with tempfile.TemporaryDirectory() as tmpdirname:
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

        self._api.update_repo_visibility(
            token=self._token, name=REPO_NAME, private=False
        )

    def test_download_model_local_only(self):
        # Test no branch specified
        with tempfile.TemporaryDirectory() as tmpdirname:
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
        with tempfile.TemporaryDirectory() as tmpdirname:
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
        with tempfile.TemporaryDirectory() as tmpdirname:
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
        with tempfile.TemporaryDirectory() as tmpdirname:
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

            # now load from cache and make sure warning to be raised
            with self.assertWarns(Warning):
                snapshot_download(
                    f"{USER}/{REPO_NAME}",
                    cache_dir=tmpdirname,
                    local_files_only=True,
                )

        # cache multiple commits and make sure correct commit is taken
        with tempfile.TemporaryDirectory() as tmpdirname:
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

    def check_download_model_with_regex(self, regex, allow=True):
        # Test `main` branch
        allow_regex = regex if allow else None
        ignore_regex = regex if not allow else None

        with tempfile.TemporaryDirectory() as tmpdirname:
            storage_folder = snapshot_download(
                f"{USER}/{REPO_NAME}",
                revision="main",
                cache_dir=tmpdirname,
                allow_regex=allow_regex,
                ignore_regex=ignore_regex,
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

    def test_download_model_with_allow_regex(self):
        self.check_download_model_with_regex("*.txt")

    def test_download_model_with_allow_regex_list(self):
        self.check_download_model_with_regex(["dummy_file.txt", "dummy_file_2.txt"])

    def test_download_model_with_ignore_regex(self):
        self.check_download_model_with_regex(".gitattributes", allow=False)

    def test_download_model_with_ignore_regex_list(self):
        self.check_download_model_with_regex(["*.git*", "*.pt"], allow=False)
