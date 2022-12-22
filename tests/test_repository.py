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
import json
import os
import pathlib
import shutil
import subprocess
import time
import unittest
import uuid
from io import BytesIO

import requests
from huggingface_hub._login import _currently_setup_credential_helpers
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repository import (
    Repository,
    is_tracked_upstream,
    is_tracked_with_lfs,
)
from huggingface_hub.utils import TemporaryDirectory, logging

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import (
    expect_deprecation,
    repo_name,
    retry_endpoint,
    set_write_permission_and_retry,
    with_production_testing,
)


logger = logging.get_logger(__name__)


WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/working_repo_2"
)

DATASET_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/tiny_dataset"
)
WORKING_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/working_dataset"
)


class RepositoryCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


class RepositoryTest(RepositoryCommonTest):
    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api.set_access_token(TOKEN)
        cls._token = TOKEN

    @retry_endpoint
    def setUp(self):
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        logger.info(
            f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}"
        )
        self.REPO_NAME = repo_name()
        self._repo_url = self._api.create_repo(repo_id=self.REPO_NAME)
        self._api.upload_file(
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file.txt",
            repo_id=f"{USER}/{self.REPO_NAME}",
        )

    def tearDown(self):
        try:
            self._api.delete_repo(repo_id=f"{USER}/{self.REPO_NAME}")
        except requests.exceptions.HTTPError:
            pass

        try:
            self._api.delete_repo(repo_id=self.REPO_NAME)
        except requests.exceptions.HTTPError:
            pass

        try:
            self._api.delete_repo(repo_id=f"valid_org/{self.REPO_NAME}")
        except requests.exceptions.HTTPError:
            pass

    def test_init_clone_from(self):
        temp_repo_url = self._api.create_repo(
            repo_id=f"{self.REPO_NAME}-temp", repo_type="space", space_sdk="static"
        )
        Repository(
            WORKING_REPO_DIR,
            clone_from=temp_repo_url,
            repo_type="space",
            use_auth_token=self._token,
        )
        self._api.delete_repo(
            repo_id=f"{USER}/{self.REPO_NAME}-temp", repo_type="space"
        )

    def test_clone_from_missing_repo(self):
        """If the repo does not exist an EnvironmentError is raised."""
        with self.assertRaises(EnvironmentError):
            Repository(
                WORKING_REPO_DIR, clone_from=f"{USER}/{uuid.uuid4()}", token=self._token
            )

    def test_clone_from_model(self):
        temp_repo_url = self._api.create_repo(
            repo_id=f"{self.REPO_NAME}-temp", repo_type="model"
        )
        Repository(
            WORKING_REPO_DIR,
            clone_from=temp_repo_url,
            repo_type="model",
            use_auth_token=self._token,
        )
        self._api.delete_repo(repo_id=f"{USER}/{self.REPO_NAME}-temp")

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
        with TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError):
                _ = Repository(tmpdirname)

    @retry_endpoint
    def test_init_clone_in_empty_folder(self):
        repo = Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largefiles()
        repo.git_pull()

        self.assertIn("random_file.txt", os.listdir(WORKING_REPO_DIR))

    def test_git_lfs_filename(self):
        os.mkdir(WORKING_REPO_DIR)
        subprocess.run(
            ["git", "init"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            cwd=WORKING_REPO_DIR,
        )

        repo = Repository(WORKING_REPO_DIR)

        large_file = [100] * int(4e6)
        with open(os.path.join(WORKING_REPO_DIR, "[].txt"), "w") as f:
            f.write(json.dumps(large_file))

        repo.git_add()

        repo.lfs_track(["[].txt"])
        self.assertFalse(is_tracked_with_lfs(f"{WORKING_REPO_DIR}/[].txt"))

        repo.lfs_track(["[].txt"], filename=True)
        self.assertTrue(is_tracked_with_lfs(f"{WORKING_REPO_DIR}/[].txt"))

    def test_init_clone_in_nonempty_folder(self):
        # Create dummy files
        # one is lfs-tracked, the other is not.
        os.makedirs(WORKING_REPO_DIR, exist_ok=True)
        with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(WORKING_REPO_DIR, "model.bin"), "w") as f:
            f.write("hello")
        with self.assertRaises(EnvironmentError):
            Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

    @retry_endpoint
    def test_init_clone_in_nonempty_non_linked_git_repo(self):
        # Create a new repository on the HF Hub
        temp_repo_url = self._api.create_repo(repo_id=f"{self.REPO_NAME}-temp")
        self._api.upload_file(
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file_2.txt",
            repo_id=f"{USER}/{self.REPO_NAME}-temp",
        )

        # Clone the new repository
        os.makedirs(WORKING_REPO_DIR, exist_ok=True)
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

        # Try and clone another repository within the same directory.
        # Should error out due to mismatched remotes.
        with self.assertRaises(EnvironmentError):
            Repository(WORKING_REPO_DIR, clone_from=temp_repo_url)

        self._api.delete_repo(repo_id=f"{self.REPO_NAME}-temp")

    @retry_endpoint
    def test_init_clone_in_nonempty_linked_git_repo_with_token(self):
        logger.info(
            f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}"
        )
        Repository(
            WORKING_REPO_DIR, clone_from=self._repo_url, use_auth_token=self._token
        )
        Repository(
            WORKING_REPO_DIR, clone_from=self._repo_url, use_auth_token=self._token
        )

    @retry_endpoint
    def test_init_clone_in_nonempty_linked_git_repo(self):
        # Clone the repository to disk
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

        # Add to the remote repository without doing anything to the local repository.
        self._api.upload_file(
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file_3.txt",
            repo_id=f"{USER}/{self.REPO_NAME}",
        )

        # Cloning the repository in the same directory should not result in a git pull.
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)
        self.assertNotIn("random_file_3.txt", os.listdir(WORKING_REPO_DIR))

    @retry_endpoint
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
            path_or_fileobj=BytesIO(b"some initial binary data: \x00\x01"),
            path_in_repo="random_file_3.txt",
            repo_id=f"{USER}/{self.REPO_NAME}",
        )

        # The repo should initialize correctly as the remote is the same, even with unrelated historied
        Repository(WORKING_REPO_DIR, clone_from=self._repo_url)

    @retry_endpoint
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
        url = repo.git_push()
        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()

    @retry_endpoint
    def test_add_commit_push_non_blocking(self):
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
        url, result = repo.git_push(blocking=False)
        # Check that the returned commit url
        # actually exists.

        if result._process.poll() is None:
            self.assertEqual(result.status, -1)

        while not result.is_done:
            time.sleep(0.5)

        self.assertTrue(result.is_done)
        self.assertEqual(result.status, 0)

        r = requests.head(url)
        r.raise_for_status()

    @retry_endpoint
    def test_context_manager_non_blocking(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=self._repo_url,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with repo.commit("New commit", blocking=False):
            with open(os.path.join(WORKING_REPO_DIR, "dummy.txt"), "w") as f:
                f.write("hello")

        while repo.commands_in_progress:
            time.sleep(1)

        self.assertEqual(len(repo.commands_in_progress), 0)
        self.assertEqual(len(repo.command_queue), 1)
        self.assertEqual(repo.command_queue[-1].status, 0)
        self.assertEqual(repo.command_queue[-1].is_done, True)
        self.assertEqual(repo.command_queue[-1].title, "push")

    @retry_endpoint
    def test_add_commit_push_non_blocking_process_killed(self):
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
            f.write(str([[[1] * 10000] * 1000] * 10))

        repo.git_add(auto_lfs_track=True)
        repo.git_commit()
        url, result = repo.git_push(blocking=False)

        result._process.kill()

        while result._process.poll() is None:
            time.sleep(0.5)

        self.assertTrue(result.is_done)
        self.assertEqual(result.status, -9)

    @retry_endpoint
    def test_clone_with_endpoint(self):
        self._api.create_repo(f"valid_org/{self.REPO_NAME}")

        clone = Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/valid_org/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            with open("dummy.txt", "w") as f:
                f.write("hello")
            with open("model.bin", "w") as f:
                f.write("hello")

        shutil.rmtree(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/valid_org/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    @retry_endpoint
    def test_clone_with_repo_name_and_org(self):
        self._api.create_repo(f"valid_org/{self.REPO_NAME}")

        clone = Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"valid_org/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            with open("dummy.txt", "w") as f:
                f.write("hello")
            with open("model.bin", "w") as f:
                f.write("hello")

        shutil.rmtree(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"valid_org/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    @retry_endpoint
    def test_clone_with_repo_name_and_user_namespace(self):
        clone = Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            # Create dummy files
            # one is lfs-tracked, the other is not.
            with open("dummy.txt", "w") as f:
                f.write("hello")
            with open("model.bin", "w") as f:
                f.write("hello")

        shutil.rmtree(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    @retry_endpoint
    def test_clone_with_repo_name_and_no_namespace(self):
        with self.assertRaises(EnvironmentError):
            Repository(
                f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
                clone_from=self.REPO_NAME,
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )

    @retry_endpoint
    def test_clone_with_repo_name_org_and_no_auth_token(self):
        self._api.create_repo(f"valid_org/{self.REPO_NAME}")

        # Instantiate it without token
        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=f"valid_org/{self.REPO_NAME}",
            git_user="ci",
            git_email="ci@dummy.com",
        )

    @retry_endpoint
    def test_clone_not_hf_url(self):
        # Should not error out
        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from=(
                "https://hf.co/hf-internal-testing/huggingface-hub-dummy-repository"
            ),
        )

    @with_production_testing
    @retry_endpoint
    def test_clone_repo_at_root(self):
        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from="bert-base-cased",
            skip_lfs_files=True,
        )

        shutil.rmtree(f"{WORKING_REPO_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_REPO_DIR}/{self.REPO_NAME}",
            clone_from="https://huggingface.co/bert-base-cased",
            skip_lfs_files=True,
        )

    @retry_endpoint
    def test_skip_lfs_files(self):
        repo = Repository(
            self.REPO_NAME,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with repo.commit("Add LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Bin file")

        shutil.rmtree(repo.local_dir)

        repo = Repository(
            self.REPO_NAME,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            skip_lfs_files=True,
        )

        with open(pathlib.Path(repo.local_dir) / "file.bin", "r") as f:
            content = f.read()
            self.assertTrue(content.startswith("version"))

        repo.git_pull(lfs=True)

        with open(pathlib.Path(repo.local_dir) / "file.bin", "r") as f:
            content = f.read()
            self.assertEqual(content, "Bin file")

    @retry_endpoint
    def test_is_tracked_upstream(self):
        repo = Repository(
            self.REPO_NAME,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        self.assertTrue(is_tracked_upstream(repo.local_dir))

    @retry_endpoint
    def test_push_errors_on_wrong_checkout(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        head_commit_ref = (
            subprocess.run(
                "git show --oneline -s".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                cwd=repo.local_dir,
            )
            .stdout.decode()
            .split()[0]
        )

        repo.git_checkout(head_commit_ref)

        with self.assertRaises(OSError):
            with repo.commit("New commit"):
                with open("new_file", "w+") as f:
                    f.write("Ok")

    @retry_endpoint
    def test_commits_on_correct_branch(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )
        branch = repo.current_branch
        repo.git_checkout("new-branch", create_branch_ok=True)
        repo.git_checkout(branch)

        with repo.commit("New commit"):
            with open("file.txt", "w+") as f:
                f.write("Ok")

        repo.git_checkout("new-branch")

        with repo.commit("New commit"):
            with open("new_file.txt", "w+") as f:
                f.write("Ok")

        with TemporaryDirectory() as tmp:
            clone = Repository(
                tmp,
                clone_from=f"{USER}/{self.REPO_NAME}",
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )
            files = os.listdir(clone.local_dir)
            self.assertTrue("file.txt" in files)
            self.assertFalse("new_file.txt" in files)

            clone.git_checkout("new-branch")
            files = os.listdir(clone.local_dir)
            self.assertFalse("file.txt" in files)
            self.assertTrue("new_file.txt" in files)

    @retry_endpoint
    def test_repo_checkout_push(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        repo.git_checkout("new-branch", create_branch_ok=True)
        repo.git_checkout("main")

        with open(os.path.join(repo.local_dir, "file.txt"), "w+") as f:
            f.write("Ok")

        repo.push_to_hub("Commit #1")
        repo.git_checkout("new-branch", create_branch_ok=True)

        with open(os.path.join(repo.local_dir, "new_file.txt"), "w+") as f:
            f.write("Ok")

        repo.push_to_hub("Commit #2")

        with TemporaryDirectory() as tmp:
            clone = Repository(
                tmp,
                clone_from=f"{USER}/{self.REPO_NAME}",
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )
            files = os.listdir(clone.local_dir)
            self.assertTrue("file.txt" in files)
            self.assertFalse("new_file.txt" in files)

            clone.git_checkout("new-branch")
            files = os.listdir(clone.local_dir)
            self.assertFalse("file.txt" in files)
            self.assertTrue("new_file.txt" in files)

    @retry_endpoint
    def test_repo_checkout_commit_context_manager(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        with repo.commit("Commit #1", branch="new-branch"):
            with open(os.path.join(repo.local_dir, "file.txt"), "w+") as f:
                f.write("Ok")

        with repo.commit("Commit #2", branch="main"):
            with open(os.path.join(repo.local_dir, "new_file.txt"), "w+") as f:
                f.write("Ok")

        # Maintains lastly used branch
        with repo.commit("Commit #3"):
            with open(os.path.join(repo.local_dir, "new_file-2.txt"), "w+") as f:
                f.write("Ok")

        with TemporaryDirectory() as tmp:
            clone = Repository(
                tmp,
                clone_from=f"{USER}/{self.REPO_NAME}",
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )
            files = os.listdir(clone.local_dir)
            self.assertFalse("file.txt" in files)
            self.assertTrue("new_file-2.txt" in files)
            self.assertTrue("new_file.txt" in files)

            clone.git_checkout("new-branch")
            files = os.listdir(clone.local_dir)
            self.assertTrue("file.txt" in files)
            self.assertFalse("new_file.txt" in files)
            self.assertFalse("new_file-2.txt" in files)

    @retry_endpoint
    def test_add_tag(self):
        # Eventually see why this is needed
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        repo.add_tag("v4.6.0", remote="origin")
        self.assertTrue(repo.tag_exists("v4.6.0", remote="origin"))

    @retry_endpoint
    def test_add_annotated_tag(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        repo.add_tag("v4.5.0", message="This is an annotated tag", remote="origin")

        # Unfortunately git offers no built-in way to check the annotated
        # message of a remote tag.
        # In order to check that the remote tag was correctly annotated,
        # we delete the local tag before pulling the remote tag (which
        # should be the same). We then check that this tag is correctly
        # annotated.
        repo.delete_tag("v4.5.0")

        self.assertTrue(repo.tag_exists("v4.5.0", remote="origin"))
        self.assertFalse(repo.tag_exists("v4.5.0"))

        subprocess.run(
            ["git", "pull", "--tags"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        )

        self.assertTrue(repo.tag_exists("v4.5.0"))

        result = subprocess.run(
            ["git", "tag", "-n9"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        ).stdout.strip()

        self.assertIn("This is an annotated tag", result)

    @retry_endpoint
    def test_delete_tag(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        repo.add_tag("v4.6.0", message="This is an annotated tag", remote="origin")
        self.assertTrue(repo.tag_exists("v4.6.0", remote="origin"))

        repo.delete_tag("v4.6.0")
        self.assertFalse(repo.tag_exists("v4.6.0"))
        self.assertTrue(repo.tag_exists("v4.6.0", remote="origin"))

        repo.delete_tag("v4.6.0", remote="origin")
        self.assertFalse(repo.tag_exists("v4.6.0", remote="origin"))

    @retry_endpoint
    def test_lfs_prune(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 2")

        root_directory = pathlib.Path(repo.local_dir) / ".git" / "lfs"
        git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )
        repo.lfs_prune()
        post_prune_git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        # Size of the directory holding LFS files was reduced
        self.assertLess(post_prune_git_lfs_files_size, git_lfs_files_size)

    @retry_endpoint
    def test_lfs_prune_git_push(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        root_directory = pathlib.Path(repo.local_dir) / ".git" / "lfs"
        git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        with open(os.path.join(repo.local_dir, "file.bin"), "w+") as f:
            f.write("Random string 2")

        repo.git_add()
        repo.git_commit("New commit")
        repo.git_push(auto_lfs_prune=True)

        post_prune_git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        # Size of the directory holding LFS files is the exact same
        self.assertEqual(post_prune_git_lfs_files_size, git_lfs_files_size)

    @retry_endpoint
    def test_lfs_prune_git_push_non_blocking(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        root_directory = pathlib.Path(repo.local_dir) / ".git" / "lfs"
        git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        with open(os.path.join(repo.local_dir, "file.bin"), "w+") as f:
            f.write("Random string 2")

        repo.git_add()
        repo.git_commit("New commit")
        repo.git_push(blocking=False, auto_lfs_prune=True)

        while len(repo.commands_in_progress):
            time.sleep(0.2)

        post_prune_git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        # Size of the directory holding LFS files is the exact same
        self.assertEqual(post_prune_git_lfs_files_size, git_lfs_files_size)

    @retry_endpoint
    def test_lfs_prune_context_manager(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        root_directory = pathlib.Path(repo.local_dir) / ".git" / "lfs"
        git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        with repo.commit("Committing LFS file", auto_lfs_prune=True):
            with open("file.bin", "w+") as f:
                f.write("Random string 2")

        post_prune_git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        # Size of the directory holding LFS files is the exact same
        self.assertEqual(post_prune_git_lfs_files_size, git_lfs_files_size)

    @retry_endpoint
    def test_lfs_prune_context_manager_non_blocking(self):
        repo = Repository(
            WORKING_REPO_DIR,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            revision="main",
        )

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        root_directory = pathlib.Path(repo.local_dir) / ".git" / "lfs"
        git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        with repo.commit("Committing LFS file", auto_lfs_prune=True, blocking=False):
            with open("file.bin", "w+") as f:
                f.write("Random string 2")

        while len(repo.commands_in_progress):
            time.sleep(0.2)

        post_prune_git_lfs_files_size = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        )

        # Size of the directory holding LFS files is the exact same
        self.assertEqual(post_prune_git_lfs_files_size, git_lfs_files_size)


class RepositoryOfflineTest(RepositoryCommonTest):
    @classmethod
    def setUpClass(cls) -> None:
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        logger.info(
            f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}"
        )
        os.makedirs(WORKING_REPO_DIR, exist_ok=True)
        subprocess.run(
            ["git", "init"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            cwd=WORKING_REPO_DIR,
        )

        repo = Repository(WORKING_REPO_DIR, git_user="ci", git_email="ci@dummy.ci")

        with open(f"{WORKING_REPO_DIR}/.gitattributes", "w+") as f:
            f.write("*.pt filter=lfs diff=lfs merge=lfs -text")

        repo.git_add(".gitattributes")
        repo.git_commit("Add .gitattributes")

    def tearDown(self):
        subprocess.run(
            ["git", "reset", "--hard"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            cwd=WORKING_REPO_DIR,
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            cwd=WORKING_REPO_DIR,
        )

        all_local_tags = subprocess.run(
            ["git", "tag", "-l"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            cwd=WORKING_REPO_DIR,
        ).stdout.strip()

        if len(all_local_tags):
            subprocess.run(
                ["git", "tag", "-d", all_local_tags],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                cwd=WORKING_REPO_DIR,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        logger.info(
            f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}"
        )

    def test_is_tracked_with_lfs(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is under 10MB
        small_file = [100]

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        with open(f"{WORKING_REPO_DIR}/small_file_2.txt", "w+") as f:
            f.write(json.dumps(small_file))

        with open(f"{WORKING_REPO_DIR}/model.pt", "w+") as f:
            f.write(json.dumps(small_file))

        repo.lfs_track("small_file.txt")

        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file_2.txt"))
        )
        self.assertTrue(is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "model.pt")))

    def test_is_tracked_with_lfs_with_pattern(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is 5MB (under 10MB)
        small_file = [100] * int(1e6)

        # This content is 20MB (over 10MB)
        large_file = [100] * int(4e6)

        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        os.makedirs(f"{WORKING_REPO_DIR}/dir", exist_ok=True)

        with open(f"{WORKING_REPO_DIR}/dir/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/dir/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        repo.auto_track_large_files("dir")

        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file.txt"))
        )
        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "dir/large_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "dir/small_file.txt"))
        )

    def test_auto_track_large_files(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is 5MB (under 10MB)
        small_file = [100] * int(1e6)

        # This content is 20MB (over 10MB)
        large_file = [100] * int(4e6)

        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        repo.auto_track_large_files()

        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file.txt"))
        )

    def test_auto_track_binary_files(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is non-binary
        non_binary_file = [100] * int(1e6)

        # This content is binary (contains the null character)
        binary_file = "\x00\x00\x00\x00"

        with open(f"{WORKING_REPO_DIR}/non_binary_file.txt", "w+") as f:
            f.write(json.dumps(non_binary_file))

        with open(f"{WORKING_REPO_DIR}/binary_file.txt", "w+") as f:
            f.write(binary_file)

        repo.auto_track_binary_files()

        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "non_binary)file.txt"))
        )
        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "binary_file.txt"))
        )

    def test_auto_track_large_files_ignored_with_gitignore(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is 20MB (over 10MB)
        large_file = [100] * int(4e6)

        # Test nested gitignores
        os.makedirs(f"{WORKING_REPO_DIR}/directory")

        with open(f"{WORKING_REPO_DIR}/.gitignore", "w+") as f:
            f.write("large_file.txt")

        with open(f"{WORKING_REPO_DIR}/directory/.gitignore", "w+") as f:
            f.write("large_file_3.txt")

        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/large_file_2.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/directory/large_file_3.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/directory/large_file_4.txt", "w+") as f:
            f.write(json.dumps(large_file))

        repo.auto_track_large_files()

        # Large files
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )
        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file_2.txt"))
        )

        self.assertFalse(
            is_tracked_with_lfs(
                os.path.join(WORKING_REPO_DIR, "directory/large_file_3.txt")
            )
        )
        self.assertTrue(
            is_tracked_with_lfs(
                os.path.join(WORKING_REPO_DIR, "directory/large_file_4.txt")
            )
        )

    def test_auto_track_binary_files_ignored_with_gitignore(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is binary (contains the null character)
        binary_file = "\x00\x00\x00\x00"

        # Test nested gitignores
        os.makedirs(f"{WORKING_REPO_DIR}/directory")

        with open(f"{WORKING_REPO_DIR}/.gitignore", "w+") as f:
            f.write("binary_file.txt")

        with open(f"{WORKING_REPO_DIR}/directory/.gitignore", "w+") as f:
            f.write("binary_file_3.txt")

        with open(f"{WORKING_REPO_DIR}/binary_file.txt", "w+") as f:
            f.write(binary_file)

        with open(f"{WORKING_REPO_DIR}/binary_file_2.txt", "w+") as f:
            f.write(binary_file)

        with open(f"{WORKING_REPO_DIR}/directory/binary_file_3.txt", "w+") as f:
            f.write(binary_file)

        with open(f"{WORKING_REPO_DIR}/directory/binary_file_4.txt", "w+") as f:
            f.write(binary_file)

        repo.auto_track_binary_files()

        # Binary files
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "binary_file.txt"))
        )
        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "binary_file_2.txt"))
        )

        self.assertFalse(
            is_tracked_with_lfs(
                os.path.join(WORKING_REPO_DIR, "directory/binary_file_3.txt")
            )
        )
        self.assertTrue(
            is_tracked_with_lfs(
                os.path.join(WORKING_REPO_DIR, "directory/binary_file_4.txt")
            )
        )

    def test_auto_track_large_files_through_git_add(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is 5MB (under 10MB)
        small_file = [100] * int(1e6)

        # This content is 20MB (over 10MB)
        large_file = [100] * int(4e6)

        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        repo.git_add(auto_lfs_track=True)

        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file.txt"))
        )

    def test_auto_track_binary_files_through_git_add(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is non binary
        non_binary_file = [100] * int(1e6)

        # This content is binary (contains the null character)
        binary_file = "\x00\x00\x00\x00"

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(non_binary_file))

        with open(f"{WORKING_REPO_DIR}/binary_file.txt", "w+") as f:
            f.write(binary_file)

        repo.git_add(auto_lfs_track=True)

        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "non_binary_file.txt"))
        )
        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "binary_file.txt"))
        )

    def test_auto_no_track_large_files_through_git_add(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is 5MB (under 10MB)
        small_file = [100] * int(1e6)

        # This content is 20MB (over 10MB)
        large_file = [100] * int(4e6)

        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        repo.git_add(auto_lfs_track=False)

        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file.txt"))
        )

    def test_auto_no_track_binary_files_through_git_add(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is non-binary
        non_binary_file = [100] * int(1e6)

        # This content is binary (contains the null character)
        binary_file = "\x00\x00\x00\x00"

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(non_binary_file))

        with open(f"{WORKING_REPO_DIR}/binary_file.txt", "w+") as f:
            f.write(binary_file)

        repo.git_add(auto_lfs_track=False)

        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "non_binary_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "binary_file.txt"))
        )

    def test_auto_track_updates_removed_gitattributes(self):
        repo = Repository(WORKING_REPO_DIR)

        # This content is 5MB (under 10MB)
        small_file = [100] * int(1e6)

        # This content is 20MB (over 10MB)
        large_file = [100] * int(4e6)

        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        with open(f"{WORKING_REPO_DIR}/small_file.txt", "w+") as f:
            f.write(json.dumps(small_file))

        repo.git_add(auto_lfs_track=True)

        self.assertTrue(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "small_file.txt"))
        )

        # Remove large file
        os.remove(f"{WORKING_REPO_DIR}/large_file.txt")

        # Auto track should remove the entry from .gitattributes
        repo.auto_track_large_files()

        # Recreate the large file with smaller contents
        with open(f"{WORKING_REPO_DIR}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        # Ensure the file is not LFS tracked anymore
        self.assertFalse(
            is_tracked_with_lfs(os.path.join(WORKING_REPO_DIR, "large_file.txt"))
        )

    def test_checkout_non_existant_branch(self):
        repo = Repository(WORKING_REPO_DIR)
        self.assertRaises(EnvironmentError, repo.git_checkout, "brand-new-branch")

    def test_checkout_new_branch(self):
        repo = Repository(WORKING_REPO_DIR)
        repo.git_checkout("new-branch", create_branch_ok=True)

        self.assertEqual(repo.current_branch, "new-branch")

    def test_is_not_tracked_upstream(self):
        repo = Repository(WORKING_REPO_DIR)
        repo.git_checkout("new-branch", create_branch_ok=True)
        self.assertFalse(is_tracked_upstream(repo.local_dir))

    def test_no_branch_checked_out_raises(self):
        repo = Repository(WORKING_REPO_DIR)

        head_commit_ref = (
            subprocess.run(
                "git show --oneline -s".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                cwd=WORKING_REPO_DIR,
            )
            .stdout.decode()
            .split()[0]
        )

        repo.git_checkout(head_commit_ref)
        self.assertRaises(OSError, is_tracked_upstream, repo.local_dir)

    def test_repo_init_checkout_default_revision(self):
        # Instantiate repository on a given revision
        repo = Repository(WORKING_REPO_DIR, revision="new-branch")
        self.assertEqual(repo.current_branch, "new-branch")

        # The revision should be kept when re-initializing the repo
        repo_2 = Repository(WORKING_REPO_DIR)
        self.assertEqual(repo_2.current_branch, "new-branch")

    def test_repo_init_checkout_revision(self):
        # Instantiate repository on a given revision
        repo = Repository(WORKING_REPO_DIR)
        current_head_hash = repo.git_head_hash()

        with open(os.path.join(repo.local_dir, "file.txt"), "w+") as f:
            f.write("File")

        repo.git_add()
        repo.git_commit("Add file.txt")

        new_head_hash = repo.git_head_hash()

        self.assertNotEqual(current_head_hash, new_head_hash)

        previous_head_repo = Repository(WORKING_REPO_DIR, revision=current_head_hash)
        files = os.listdir(previous_head_repo.local_dir)
        self.assertNotIn("file.txt", files)

        current_head_repo = Repository(WORKING_REPO_DIR, revision=new_head_hash)
        files = os.listdir(current_head_repo.local_dir)
        self.assertIn("file.txt", files)

    @expect_deprecation("set_access_token")
    def test_repo_user(self):
        api = HfApi(endpoint=ENDPOINT_STAGING)
        token = TOKEN
        api.set_access_token(TOKEN)

        repo = Repository(WORKING_REPO_DIR, use_auth_token=token)
        user = api.whoami(token)

        username = subprocess.run(
            ["git", "config", "user.name"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        ).stdout.strip()
        email = subprocess.run(
            ["git", "config", "user.email"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        ).stdout.strip()

        self.assertEqual(username, user["fullname"])
        self.assertEqual(email, user["email"])

    @expect_deprecation("set_access_token")
    def test_repo_passed_user(self):
        api = HfApi(endpoint=ENDPOINT_STAGING)
        token = TOKEN
        api.set_access_token(TOKEN)
        repo = Repository(
            WORKING_REPO_DIR,
            git_user="RANDOM_USER",
            git_email="EMAIL@EMAIL.EMAIL",
            use_auth_token=token,
        )
        username = subprocess.run(
            ["git", "config", "user.name"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        ).stdout.strip()
        email = subprocess.run(
            ["git", "config", "user.email"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        ).stdout.strip()

        self.assertEqual(username, "RANDOM_USER")
        self.assertEqual(email, "EMAIL@EMAIL.EMAIL")

    @expect_deprecation("_currently_setup_credential_helpers")
    def test_correct_helper(self):
        subprocess.run(
            ["git", "config", "--global", "credential.helper", "get"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        )
        repo = Repository(WORKING_REPO_DIR)
        self.assertListEqual(
            _currently_setup_credential_helpers(repo.local_dir), ["get", "store"]
        )
        self.assertEqual(_currently_setup_credential_helpers(), ["get"])

    def test_add_tag(self):
        repo = Repository(
            WORKING_REPO_DIR,
            git_user="RANDOM_USER",
            git_email="EMAIL@EMAIL.EMAIL",
        )

        repo.add_tag("v4.6.0")
        self.assertTrue(repo.tag_exists("v4.6.0"))

    def test_add_annotated_tag(self):
        repo = Repository(
            WORKING_REPO_DIR,
            git_user="RANDOM_USER",
            git_email="EMAIL@EMAIL.EMAIL",
        )

        repo.add_tag("v4.6.0", message="This is an annotated tag")
        self.assertTrue(repo.tag_exists("v4.6.0"))

        result = subprocess.run(
            ["git", "tag", "-n9"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo.local_dir,
        ).stdout.strip()

        self.assertIn("This is an annotated tag", result)

    def test_delete_tag(self):
        repo = Repository(
            WORKING_REPO_DIR,
            git_user="RANDOM_USER",
            git_email="EMAIL@EMAIL.EMAIL",
        )

        repo.add_tag("v4.6.0", message="This is an annotated tag")
        self.assertTrue(repo.tag_exists("v4.6.0"))

        repo.delete_tag("v4.6.0")
        self.assertFalse(repo.tag_exists("v4.6.0"))

    def test_repo_clean(self):
        repo = Repository(
            WORKING_REPO_DIR,
            git_user="RANDOM_USER",
            git_email="EMAIL@EMAIL.EMAIL",
        )

        self.assertTrue(repo.is_repo_clean())

        with open(os.path.join(repo.local_dir, "file.txt"), "w+") as f:
            f.write("Test")

        self.assertFalse(repo.is_repo_clean())


class RepositoryDatasetTest(RepositoryCommonTest):
    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

    def setUp(self):
        self.REPO_NAME = repo_name()
        if os.path.exists(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}"):
            shutil.rmtree(
                f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
                onerror=set_write_permission_and_retry,
            )
        logger.info(
            f"Does {WORKING_DATASET_DIR}/{self.REPO_NAME} exist:"
            f" {os.path.exists(f'{WORKING_DATASET_DIR}/{self.REPO_NAME}')}"
        )
        self._api.create_repo(repo_id=self.REPO_NAME, repo_type="dataset")

    def tearDown(self):
        try:
            self._api.delete_repo(repo_id=self.REPO_NAME, repo_type="dataset")
        except requests.exceptions.HTTPError:
            try:
                self._api.delete_repo(
                    repo_id=f"valid_org/{self.REPO_NAME}", repo_type="dataset"
                )
            except requests.exceptions.HTTPError:
                pass

    @retry_endpoint
    def test_clone_with_endpoint(self):
        clone = Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/datasets/{USER}/{self.REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            for file in os.listdir(DATASET_FIXTURE):
                shutil.copyfile(pathlib.Path(DATASET_FIXTURE) / file, file)

        shutil.rmtree(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/datasets/{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}")
        self.assertTrue("some_text.txt" in files)
        self.assertTrue("test.py" in files)

    @retry_endpoint
    def test_clone_with_repo_name_and_org(self):
        self._api.create_repo(f"valid_org/{self.REPO_NAME}", repo_type="dataset")

        clone = Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"valid_org/{self.REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            for file in os.listdir(DATASET_FIXTURE):
                shutil.copyfile(pathlib.Path(DATASET_FIXTURE) / file, file)

        shutil.rmtree(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"valid_org/{self.REPO_NAME}",
            use_auth_token=self._token,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}")
        self.assertTrue("some_text.txt" in files)
        self.assertTrue("test.py" in files)

    @retry_endpoint
    def test_clone_with_repo_name_and_user_namespace(self):
        clone = Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"{USER}/{self.REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            for file in os.listdir(DATASET_FIXTURE):
                shutil.copyfile(pathlib.Path(DATASET_FIXTURE) / file, file)

        shutil.rmtree(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}")

        Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_DATASET_DIR}/{self.REPO_NAME}")
        self.assertTrue("some_text.txt" in files)
        self.assertTrue("test.py" in files)

    @retry_endpoint
    def test_clone_with_repo_name_and_no_namespace(self):
        with self.assertRaises(EnvironmentError):
            Repository(
                f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
                clone_from=self.REPO_NAME,
                repo_type="dataset",
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )

    @retry_endpoint
    def test_clone_with_repo_name_user_and_no_auth_token(self):
        # Create repo
        Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"{USER}/{self.REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        # Instantiate it without token
        Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"{USER}/{self.REPO_NAME}",
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

    @retry_endpoint
    def test_clone_with_repo_name_org_and_no_auth_token(self):
        self._api.create_repo(f"valid_org/{self.REPO_NAME}", repo_type="dataset")

        # Instantiate it without token
        Repository(
            f"{WORKING_DATASET_DIR}/{self.REPO_NAME}",
            clone_from=f"valid_org/{self.REPO_NAME}",
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )
