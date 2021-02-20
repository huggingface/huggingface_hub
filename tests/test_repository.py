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
import time
import unittest

from huggingface_hub.hf_api import HfApi
from huggingface_hub.repository import Repository

from .testing_constants import ENDPOINT_STAGING, PASS, USER


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
            shutil.rmtree(WORKING_REPO_DIR)
        except FileNotFoundError:
            pass

        cls._repo_url = self._api.create_repo(token=self._token, name=REPO_NAME)

    def tearDown(self):
        self._api.delete_repo(token=self._token, name=REPO_NAME)

    def test_init_from_existing_local_clone(self):
        subprocess.run(
            ["git", "clone", self._repo_url, WORKING_REPO_DIR],
            check=True,
            capture_output=True,
        )

        repo = Repository(WORKING_REPO_DIR)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largesfiles()
        repo.git_pull()

    def test_init_clone_in_empty_folder(self):
        repo = Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largesfiles()
        repo.git_pull()

    def test_init_clone_in_nonempty_folder(self):
        # Create dummy files
        # one is lfs-tracked, the other is not.
        with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(WORKING_REPO_DIR, "model.bin"), "w") as f:
            f.write("hello")
        repo = Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largesfiles()
        repo.git_pull()

    def test_add_commit_push(self):
        repo = Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

        # Create dummy files
        # one is lfs-tracked, the other is not.
        with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(WORKING_REPO_DIR, "model.bin"), "w") as f:
            f.write("hello")

        repo.git_add()
        repo.git_commit()
        repo.git_push()
