import os
import tempfile
import unittest

from huggingface_hub import HfApi, Repository
from huggingface_hub.snapshot_download import snapshot_download
from tests.testing_constants import ENDPOINT_STAGING, PASS, USER


REPO_NAME = "dummy-hf-hub"


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
            REPO_NAME, clone_from=f"{USER}/{REPO_NAME}", use_auth_token=self._token
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

    def tearDown(self) -> None:
        self._api.delete_repo(token=self._token, name=REPO_NAME)

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
