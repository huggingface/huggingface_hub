# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from io import BytesIO

import requests
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repository import Repository

from .testing_constants import ENDPOINT_STAGING, PASS, USER
from .testing_utils import set_write_permission_and_retry


REPO_NAME = "repo-{}".format(int(time.time() * 10e3))


WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/working_repo_2"
)


class RepositoryCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


class RepositoryTest(RepositoryCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def setUp(self):
        try:
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        except FileNotFoundError:
            pass

        self._repo_url = self._api.create_repo(token=self._token, name=REPO_NAME)
        self._api.upload_file(
            token=self._token,
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file.txt",
            repo_id=f"{USER}/{REPO_NAME}",
        )

    def tearDown(self):
        self._api.delete_repo(token=self._token, name=REPO_NAME)

    def test_init_from_existing_local_clone(self):
        subprocess.run(
            ["git", "clone", self._repo_url, WORKING_REPO_DIR],
            check=True,
        )

        repo = Repository(WORKING_REPO_DIR)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largefiles()
        repo.git_pull()

    def test_init_failure(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError):
                _ = Repository(tmpdirname)

    def test_init_clone_in_empty_folder(self):
        repo = Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largefiles()
        repo.git_pull()

        self.assertIn("random_file.txt", os.listdir(WORKING_REPO_DIR))

    def test_init_clone_in_nonempty_folder(self):
        # Create dummy files
        # one is lfs-tracked, the other is not.
        os.makedirs(WORKING_REPO_DIR, exist_ok=True)
        with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(WORKING_REPO_DIR, "model.bin"), "w") as f:
            f.write("hello")
        self.assertRaises(
            OSError, Repository, WORKING_REPO_DIR, clone_from=self._repo_url
        )

    def test_init_clone_in_nonempty_non_linked_git_repo(self):
        # Create a new repository on the HF Hub
        temp_repo_url = self._api.create_repo(
            token=self._token, name=f"{REPO_NAME}-temp"
        )
        self._api.upload_file(
            token=self._token,
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file_2.txt",
            repo_id=f"{USER}/{REPO_NAME}-temp",
        )

        # Clone the new repository
        os.makedirs(WORKING_REPO_DIR, exist_ok=True)
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

        # Try and clone another repository within the same directory. Should error out due to mismatched remotes.
        self.assertRaises(
            EnvironmentError, Repository, WORKING_REPO_DIR, clone_from=temp_repo_url
        )

        self._api.delete_repo(token=self._token, name=f"{REPO_NAME}-temp")

    def test_init_clone_in_nonempty_linked_git_repo_with_token(self):
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url, use_auth_token=self._token)
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url, use_auth_token=self._token)

    def test_init_clone_in_nonempty_linked_git_repo(self):
        # Clone the repository to disk
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

        # Add to the remote repository without doing anything to the local repository.
        self._api.upload_file(
            token=self._token,
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file_3.txt",
            repo_id=f"{USER}/{REPO_NAME}",
        )

        # Cloning the repository in the same directory should not result in a git pull.
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        self.assertNotIn("random_file_3.txt", os.listdir(WORKING_REPO_DIR))

    def test_init_clone_in_nonempty_linked_git_repo_unrelated_histories(self):
        # Clone the repository to disk
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=self._repo_url,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with open(f"{WORKING_REPO_DIR}/random_file_3.txt", "w+") as f:
            f.write("New file.")

        repo.git_add()
        repo.git_commit("Unrelated commit")

        # Add to the remote repository without doing anything to the local repository.
        self._api.upload_file(
            token=self._token,
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file_3.txt",
            repo_id=f"{USER}/{REPO_NAME}",
        )

        # The repo should initialize correctly as the remote is the same, even with unrelated historied
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

    def test_add_commit_push(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=self._repo_url,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        # Create dummy files
        # one is lfs-tracked, the other is not.
        with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(WORKING_REPO_DIR, "model.bin"), "w") as f:
            f.write("hello")

        repo.git_add()
        repo.git_commit()
        try:
            url = repo.git_push()
        except subprocess.CalledProcessError as exc:
            print(exc.stderr)
            raise exc
        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()
