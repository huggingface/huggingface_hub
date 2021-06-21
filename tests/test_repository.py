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

    def test_init_clone_in_nonempty_folder(self):
        # Create dummy files
        # one is lfs-tracked, the other is not.
        os.makedirs(WORKING_REPO_DIR, exist_ok=True)
        with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(WORKING_REPO_DIR, "model.bin"), "w") as f:
            f.write("hello")
        repo = Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largefiles()
        repo.git_pull()

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
