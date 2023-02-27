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
import time
import unittest
from pathlib import Path

import pytest
import requests

from huggingface_hub import RepoUrl
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repository import (
    Repository,
    is_tracked_upstream,
    is_tracked_with_lfs,
)
from huggingface_hub.utils import SoftTemporaryDirectory, logging, run_subprocess

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import (
    expect_deprecation,
    repo_name,
    retry_endpoint,
    use_tmp_repo,
    with_production_testing,
)


logger = logging.get_logger(__name__)


@pytest.mark.usefixtures("fx_cache_dir")
class RepositoryTestAbstract(unittest.TestCase):
    cache_dir: Path
    repo_path: Path

    # This content is 5MB (under 10MB)
    small_content = json.dumps([100] * int(1e6))

    # This content is 20MB (over 10MB)
    large_content = json.dumps([100] * int(4e6))

    # This content is binary (contains the null character)
    binary_content = "\x00\x00\x00\x00"

    _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    @classmethod
    def setUp(self) -> None:
        self.repo_path = self.cache_dir / "working_dir"
        self.repo_path.mkdir()

    def _create_dummy_files(self):
        # Create dummy files
        # one is lfs-tracked, the other is not.
        small_file = self.repo_path / "dummy.txt"
        small_file.write_text(self.small_content)

        binary_file = self.repo_path / "model.bin"
        binary_file.write_text(self.binary_content)


class TestRepositoryShared(RepositoryTestAbstract):
    """Tests in this class shares a single repo on the Hub (common to all tests).

    These tests must not push data to it.
    """

    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        super().setUpClass()
        cls._api.set_access_token(TOKEN)
        cls._token = TOKEN

        cls.repo_url = cls._api.create_repo(repo_id=repo_name())
        cls.repo_id = cls.repo_url.repo_id
        cls._api.upload_file(
            path_or_fileobj=cls.binary_content.encode(),
            path_in_repo="random_file.txt",
            repo_id=cls.repo_id,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls._api.delete_repo(repo_id=cls.repo_id)
        except requests.exceptions.HTTPError:
            pass

    def test_clone_from_repo_url(self):
        Repository(self.repo_path, clone_from=self.repo_url)

    @retry_endpoint
    def test_clone_from_repo_id(self):
        Repository(self.repo_path, clone_from=self.repo_id)

    @retry_endpoint
    def test_clone_from_repo_name_no_namespace_fails(self):
        with self.assertRaises(EnvironmentError):
            Repository(
                self.repo_path,
                clone_from=self.repo_id.split("/")[1],
                use_auth_token=self._token,
            )

    @retry_endpoint
    def test_clone_from_not_hf_url(self):
        # Should not error out
        Repository(
            self.repo_path,
            clone_from="https://hf.co/hf-internal-testing/huggingface-hub-dummy-repository",
        )

    def test_clone_from_missing_repo(self):
        """If the repo does not exist an EnvironmentError is raised."""
        with self.assertRaises(EnvironmentError):
            Repository(self.repo_path, clone_from="missing_repo", token=self._token)

    @with_production_testing
    @retry_endpoint
    def test_clone_from_prod_canonical_repo_id(self):
        Repository(self.repo_path, clone_from="bert-base-cased", skip_lfs_files=True)

    @with_production_testing
    @retry_endpoint
    def test_clone_from_prod_canonical_repo_url(self):
        Repository(
            self.repo_path,
            clone_from="https://huggingface.co/bert-base-cased",
            skip_lfs_files=True,
        )

    def test_init_from_existing_local_clone(self):
        run_subprocess(["git", "clone", self.repo_url, str(self.repo_path)])

        repo = Repository(self.repo_path)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largefiles()
        repo.git_pull()

    def test_init_failure(self):
        with self.assertRaises(ValueError):
            Repository(self.repo_path)

    @retry_endpoint
    def test_init_clone_in_empty_folder(self):
        repo = Repository(self.repo_path, clone_from=self.repo_url)
        repo.lfs_track(["*.pdf"])
        repo.lfs_enable_largefiles()
        repo.git_pull()
        self.assertIn("random_file.txt", os.listdir(self.repo_path))

    def test_git_lfs_filename(self):
        run_subprocess("git init", folder=self.repo_path)

        repo = Repository(self.repo_path)
        large_file = self.repo_path / "large_file[].txt"
        large_file.write_text(self.large_content)

        repo.git_add()

        repo.lfs_track([large_file.name])
        self.assertFalse(is_tracked_with_lfs(large_file))

        repo.lfs_track([large_file.name], filename=True)
        self.assertTrue(is_tracked_with_lfs(large_file))

    def test_init_clone_in_nonempty_folder(self):
        self._create_dummy_files()
        with self.assertRaises(EnvironmentError):
            Repository(self.repo_path, clone_from=self.repo_url)

    @retry_endpoint
    def test_init_clone_in_nonempty_linked_git_repo_with_token(self):
        Repository(self.repo_path, clone_from=self.repo_url, use_auth_token=self._token)
        Repository(self.repo_path, clone_from=self.repo_url, use_auth_token=self._token)

    @retry_endpoint
    def test_is_tracked_upstream(self):
        Repository(self.repo_path, clone_from=self.repo_id)
        self.assertTrue(is_tracked_upstream(self.repo_path))

    @retry_endpoint
    def test_push_errors_on_wrong_checkout(self):
        repo = Repository(self.repo_path, clone_from=self.repo_id)

        head_commit_ref = run_subprocess("git show --oneline -s", folder=self.repo_path).stdout.split()[0]

        repo.git_checkout(head_commit_ref)

        with self.assertRaises(OSError):
            with repo.commit("New commit"):
                with open("new_file", "w+") as f:
                    f.write("Ok")


class TestRepositoryUniqueRepos(RepositoryTestAbstract):
    """Tests in this class use separated repos on the Hub (i.e. 1 test = 1 repo).

    These tests can push data to it.
    """

    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        super().setUpClass()
        cls._api.set_access_token(TOKEN)
        cls._token = TOKEN

    @retry_endpoint
    def setUp(self):
        super().setUp()
        self.repo_url = self._api.create_repo(repo_id=repo_name())
        self.repo_id = self.repo_url.repo_id
        self._api.upload_file(
            path_or_fileobj=self.binary_content.encode(),
            path_in_repo="random_file.txt",
            repo_id=self.repo_id,
        )

    def tearDown(self):
        try:
            self._api.delete_repo(repo_id=self.repo_id)
        except requests.exceptions.HTTPError:
            pass

    def clone_repo(self, **kwargs) -> Repository:
        if "local_dir" not in kwargs:
            kwargs["local_dir"] = self.repo_path
        if "clone_from" not in kwargs:
            kwargs["clone_from"] = self.repo_url
        if "use_auth_token" not in kwargs:
            kwargs["use_auth_token"] = self._token
        if "git_user" not in kwargs:
            kwargs["git_user"] = "ci"
        if "git_email" not in kwargs:
            kwargs["git_email"] = "ci@dummy.com"
        return Repository(**kwargs)

    @retry_endpoint
    @use_tmp_repo()
    def test_init_clone_in_nonempty_non_linked_git_repo(self, repo_url: RepoUrl):
        self.clone_repo()

        # Try and clone another repository within the same directory.
        # Should error out due to mismatched remotes.
        with self.assertRaises(EnvironmentError):
            Repository(self.repo_path, clone_from=repo_url)

    @retry_endpoint
    def test_init_clone_in_nonempty_linked_git_repo(self):
        # Clone the repository to disk
        self.clone_repo()

        # Add to the remote repository without doing anything to the local repository.
        self._api.upload_file(
            path_or_fileobj=self.binary_content.encode(),
            path_in_repo="random_file_3.txt",
            repo_id=self.repo_id,
        )

        # Cloning the repository in the same directory should not result in a git pull.
        self.clone_repo(clone_from=self.repo_url)
        self.assertNotIn("random_file_3.txt", os.listdir(self.repo_path))

    @retry_endpoint
    def test_init_clone_in_nonempty_linked_git_repo_unrelated_histories(self):
        # Clone the repository to disk
        repo = self.clone_repo()

        # Create and commit file locally
        (self.repo_path / "random_file_3.txt").write_text("hello world")
        repo.git_add()
        repo.git_commit("Unrelated commit")

        # Add to the remote repository without doing anything to the local repository.
        self._api.upload_file(
            path_or_fileobj=self.binary_content.encode(),
            path_in_repo="random_file_3.txt",
            repo_id=self.repo_url.repo_id,
        )

        # The repo should initialize correctly as the remote is the same, even with unrelated historied
        self.clone_repo()

    @retry_endpoint
    def test_add_commit_push(self):
        repo = self.clone_repo()
        self._create_dummy_files()
        repo.git_add()
        repo.git_commit()
        url = repo.git_push()

        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()

    @retry_endpoint
    def test_add_commit_push_non_blocking(self):
        repo = self.clone_repo()
        self._create_dummy_files()
        repo.git_add()
        repo.git_commit()
        url, result = repo.git_push(blocking=False)

        # Check background process
        if result._process.poll() is None:
            self.assertEqual(result.status, -1)

        while not result.is_done:
            time.sleep(0.5)

        self.assertTrue(result.is_done)
        self.assertEqual(result.status, 0)

        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()

    @retry_endpoint
    def test_context_manager_non_blocking(self):
        repo = self.clone_repo()

        with repo.commit("New commit", blocking=False):
            (self.repo_path / "dummy.txt").write_text("hello world")

        while repo.commands_in_progress:
            time.sleep(1)

        self.assertEqual(len(repo.commands_in_progress), 0)
        self.assertEqual(len(repo.command_queue), 1)
        self.assertEqual(repo.command_queue[-1].status, 0)
        self.assertEqual(repo.command_queue[-1].is_done, True)
        self.assertEqual(repo.command_queue[-1].title, "push")

    @unittest.skipIf(os.name == "nt", "Killing a process on Windows works differently.")
    def test_add_commit_push_non_blocking_process_killed(self):
        repo = self.clone_repo()

        # Far too big file: will take forever
        (self.repo_path / "dummy.txt").write_text(str([[[1] * 10000] * 1000] * 10))
        repo.git_add(auto_lfs_track=True)
        repo.git_commit()
        _, result = repo.git_push(blocking=False)

        result._process.kill()

        while result._process.poll() is None:
            time.sleep(0.5)

        self.assertTrue(result.is_done)
        self.assertEqual(result.status, -9)

    @retry_endpoint
    def test_commit_context_manager(self):
        # Clone and commit from a first folder
        folder_1 = self.repo_path / "folder_1"
        clone = self.clone_repo(local_dir=folder_1)
        with clone.commit("Commit"):
            with open("dummy.txt", "w") as f:
                f.write("hello")
            with open("model.bin", "w") as f:
                f.write("hello")

        # Clone in second folder. Check existence of committed files
        folder_2 = self.repo_path / "folder_2"
        self.clone_repo(local_dir=folder_2)
        files = os.listdir(folder_2)
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    @retry_endpoint
    def test_clone_skip_lfs_files(self):
        # Upload LFS file
        self._api.upload_file(path_or_fileobj=b"Bin file", path_in_repo="file.bin", repo_id=self.repo_id)

        repo = self.clone_repo(skip_lfs_files=True)
        file_bin = self.repo_path / "file.bin"

        self.assertTrue(file_bin.read_text().startswith("version"))

        repo.git_pull(lfs=True)

        self.assertEqual(file_bin.read_text(), "Bin file")

    @retry_endpoint
    def test_commits_on_correct_branch(self):
        repo = self.clone_repo()
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

        with SoftTemporaryDirectory() as tmp:
            clone = self.clone_repo(local_dir=tmp)
            files = os.listdir(clone.local_dir)
            self.assertTrue("file.txt" in files)
            self.assertFalse("new_file.txt" in files)

            clone.git_checkout("new-branch")
            files = os.listdir(clone.local_dir)
            self.assertFalse("file.txt" in files)
            self.assertTrue("new_file.txt" in files)

    @retry_endpoint
    def test_repo_checkout_push(self):
        repo = self.clone_repo()

        repo.git_checkout("new-branch", create_branch_ok=True)
        repo.git_checkout("main")

        (self.repo_path / "file.txt").write_text("OK")

        repo.push_to_hub("Commit #1")
        repo.git_checkout("new-branch", create_branch_ok=True)

        (self.repo_path / "new_file.txt").write_text("OK")

        repo.push_to_hub("Commit #2")

        with SoftTemporaryDirectory() as tmp:
            clone = self.clone_repo(local_dir=tmp)
            files = os.listdir(clone.local_dir)
            self.assertTrue("file.txt" in files)
            self.assertFalse("new_file.txt" in files)

            clone.git_checkout("new-branch")
            files = os.listdir(clone.local_dir)
            self.assertFalse("file.txt" in files)
            self.assertTrue("new_file.txt" in files)

    @retry_endpoint
    def test_repo_checkout_commit_context_manager(self):
        repo = self.clone_repo()

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

        with SoftTemporaryDirectory() as tmp:
            clone = self.clone_repo(local_dir=tmp)
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
        repo = self.clone_repo()
        repo.add_tag("v4.6.0", remote="origin")
        self.assertTrue(repo.tag_exists("v4.6.0", remote="origin"))

    @retry_endpoint
    def test_add_annotated_tag(self):
        repo = self.clone_repo()
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

        # Tag still exists on remote
        run_subprocess("git pull --tags", folder=self.repo_path)
        self.assertTrue(repo.tag_exists("v4.5.0"))

        # Tag is annotated
        result = run_subprocess("git tag -n9", folder=self.repo_path).stdout.strip()
        self.assertIn("This is an annotated tag", result)

    @retry_endpoint
    def test_delete_tag(self):
        repo = self.clone_repo()

        repo.add_tag("v4.6.0", message="This is an annotated tag", remote="origin")
        self.assertTrue(repo.tag_exists("v4.6.0", remote="origin"))

        repo.delete_tag("v4.6.0")
        self.assertFalse(repo.tag_exists("v4.6.0"))
        self.assertTrue(repo.tag_exists("v4.6.0", remote="origin"))

        repo.delete_tag("v4.6.0", remote="origin")
        self.assertFalse(repo.tag_exists("v4.6.0", remote="origin"))

    @retry_endpoint
    def test_lfs_prune(self):
        repo = self.clone_repo()

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 2")

        root_directory = self.repo_path / ".git" / "lfs"
        git_lfs_files_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())
        repo.lfs_prune()
        post_prune_git_lfs_files_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())

        # Size of the directory holding LFS files was reduced
        self.assertLess(post_prune_git_lfs_files_size, git_lfs_files_size)

    @retry_endpoint
    def test_lfs_prune_git_push(self):
        repo = self.clone_repo()
        with repo.commit("Committing LFS file"):
            with open("file.bin", "w+") as f:
                f.write("Random string 1")

        root_directory = self.repo_path / ".git" / "lfs"
        git_lfs_files_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())

        with open(os.path.join(repo.local_dir, "file.bin"), "w+") as f:
            f.write("Random string 2")

        repo.git_add()
        repo.git_commit("New commit")
        repo.git_push(auto_lfs_prune=True)

        post_prune_git_lfs_files_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())

        # Size of the directory holding LFS files is the exact same
        self.assertEqual(post_prune_git_lfs_files_size, git_lfs_files_size)


class TestRepositoryOffline(RepositoryTestAbstract):
    """Class to test `Repository` object on local folders only (no cloning from Hub)."""

    repo: Repository

    @classmethod
    def setUp(self) -> None:
        super().setUp()

        run_subprocess("git init", folder=self.repo_path)

        self.repo = Repository(self.repo_path, git_user="ci", git_email="ci@dummy.ci")

        git_attributes_path = self.repo_path / ".gitattributes"
        git_attributes_path.write_text("*.pt filter=lfs diff=lfs merge=lfs -text")

        self.repo.git_add(".gitattributes")
        self.repo.git_commit("Add .gitattributes")

    def test_is_tracked_with_lfs(self):
        txt_1 = self.repo_path / "small_file_1.txt"
        txt_2 = self.repo_path / "small_file_2.txt"
        pt_1 = self.repo_path / "model.pt"

        txt_1.write_text(self.small_content)
        txt_2.write_text(self.small_content)
        pt_1.write_text(self.small_content)

        self.repo.lfs_track("small_file_1.txt")

        self.assertTrue(is_tracked_with_lfs(txt_1))
        self.assertFalse(is_tracked_with_lfs(txt_2))
        self.assertTrue(pt_1)

    def test_is_tracked_with_lfs_with_pattern(self):
        txt_small_file = self.repo_path / "small_file.txt"
        txt_small_file.write_text(self.small_content)

        txt_large_file = self.repo_path / "large_file.txt"
        txt_large_file.write_text(self.large_content)

        (self.repo_path / "dir").mkdir()
        txt_small_file_in_dir = self.repo_path / "dir" / "small_file.txt"
        txt_small_file_in_dir.write_text(self.small_content)

        txt_large_file_in_dir = self.repo_path / "dir" / "large_file.txt"
        txt_large_file_in_dir.write_text(self.large_content)

        self.repo.auto_track_large_files("dir")

        self.assertFalse(is_tracked_with_lfs(txt_large_file))
        self.assertFalse(is_tracked_with_lfs(txt_small_file))
        self.assertTrue(is_tracked_with_lfs(txt_large_file_in_dir))
        self.assertFalse(is_tracked_with_lfs(txt_small_file_in_dir))

    def test_auto_track_large_files(self):
        txt_small_file = self.repo_path / "small_file.txt"
        txt_small_file.write_text(self.small_content)

        txt_large_file = self.repo_path / "large_file.txt"
        txt_large_file.write_text(self.large_content)

        self.repo.auto_track_large_files()

        self.assertTrue(is_tracked_with_lfs(txt_large_file))
        self.assertFalse(is_tracked_with_lfs(txt_small_file))

    def test_auto_track_binary_files(self):
        non_binary_file = self.repo_path / "non_binary_file.txt"
        non_binary_file.write_text(self.small_content)

        binary_file = self.repo_path / "binary_file.txt"
        binary_file.write_text(self.binary_content)

        self.repo.auto_track_binary_files()

        self.assertFalse(is_tracked_with_lfs(non_binary_file))
        self.assertTrue(is_tracked_with_lfs(binary_file))

    def test_auto_track_large_files_ignored_with_gitignore(self):
        (self.repo_path / "dir").mkdir()

        # Test nested gitignores
        gitignore_file = self.repo_path / ".gitignore"
        gitignore_file.write_text("large_file.txt")

        gitignore_file_in_dir = self.repo_path / "dir" / ".gitignore"
        gitignore_file_in_dir.write_text("large_file_3.txt")

        large_file = self.repo_path / "large_file.txt"
        large_file.write_text(self.large_content)

        large_file_2 = self.repo_path / "large_file_2.txt"
        large_file_2.write_text(self.large_content)

        large_file_3 = self.repo_path / "dir" / "large_file_3.txt"
        large_file_3.write_text(self.large_content)

        large_file_4 = self.repo_path / "dir" / "large_file_4.txt"
        large_file_4.write_text(self.large_content)

        self.repo.auto_track_large_files()

        # Large files
        self.assertFalse(is_tracked_with_lfs(large_file))
        self.assertTrue(is_tracked_with_lfs(large_file_2))

        self.assertFalse(is_tracked_with_lfs(large_file_3))
        self.assertTrue(is_tracked_with_lfs(large_file_4))

    def test_auto_track_binary_files_ignored_with_gitignore(self):
        (self.repo_path / "dir").mkdir()

        # Test nested gitignores
        gitignore_file = self.repo_path / ".gitignore"
        gitignore_file.write_text("binary_file.txt")

        gitignore_file_in_dir = self.repo_path / "dir" / ".gitignore"
        gitignore_file_in_dir.write_text("binary_file_3.txt")

        binary_file = self.repo_path / "binary_file.txt"
        binary_file.write_text(self.binary_content)

        binary_file_2 = self.repo_path / "binary_file_2.txt"
        binary_file_2.write_text(self.binary_content)

        binary_file_3 = self.repo_path / "dir" / "binary_file_3.txt"
        binary_file_3.write_text(self.binary_content)

        binary_file_4 = self.repo_path / "dir" / "binary_file_4.txt"
        binary_file_4.write_text(self.binary_content)

        self.repo.auto_track_binary_files()

        # Binary files
        self.assertFalse(is_tracked_with_lfs(binary_file))
        self.assertTrue(is_tracked_with_lfs(binary_file_2))
        self.assertFalse(is_tracked_with_lfs(binary_file_3))
        self.assertTrue(is_tracked_with_lfs(binary_file_4))

    def test_auto_track_large_files_through_git_add(self):
        txt_small_file = self.repo_path / "small_file.txt"
        txt_small_file.write_text(self.small_content)

        txt_large_file = self.repo_path / "large_file.txt"
        txt_large_file.write_text(self.large_content)

        self.repo.git_add(auto_lfs_track=True)

        self.assertTrue(is_tracked_with_lfs(txt_large_file))
        self.assertFalse(is_tracked_with_lfs(txt_small_file))

    def test_auto_track_binary_files_through_git_add(self):
        non_binary_file = self.repo_path / "small_file.txt"
        non_binary_file.write_text(self.small_content)

        binary_file = self.repo_path / "binary.txt"
        binary_file.write_text(self.binary_content)

        self.repo.git_add(auto_lfs_track=True)

        self.assertTrue(is_tracked_with_lfs(binary_file))
        self.assertFalse(is_tracked_with_lfs(non_binary_file))

    def test_auto_no_track_large_files_through_git_add(self):
        txt_small_file = self.repo_path / "small_file.txt"
        txt_small_file.write_text(self.small_content)

        txt_large_file = self.repo_path / "large_file.txt"
        txt_large_file.write_text(self.large_content)

        self.repo.git_add(auto_lfs_track=False)

        self.assertFalse(is_tracked_with_lfs(txt_large_file))
        self.assertFalse(is_tracked_with_lfs(txt_small_file))

    def test_auto_no_track_binary_files_through_git_add(self):
        non_binary_file = self.repo_path / "small_file.txt"
        non_binary_file.write_text(self.small_content)

        binary_file = self.repo_path / "binary.txt"
        binary_file.write_text(self.binary_content)

        self.repo.git_add(auto_lfs_track=False)

        self.assertFalse(is_tracked_with_lfs(binary_file))
        self.assertFalse(is_tracked_with_lfs(non_binary_file))

    def test_auto_track_updates_removed_gitattributes(self):
        txt_small_file = self.repo_path / "small_file.txt"
        txt_small_file.write_text(self.small_content)

        txt_large_file = self.repo_path / "large_file.txt"
        txt_large_file.write_text(self.large_content)

        self.repo.git_add(auto_lfs_track=True)

        self.assertTrue(is_tracked_with_lfs(txt_large_file))
        self.assertFalse(is_tracked_with_lfs(txt_small_file))

        # Remove large file
        txt_large_file.unlink()

        # Auto track should remove the entry from .gitattributes
        self.repo.auto_track_large_files()

        # Recreate the large file with smaller contents
        txt_large_file.write_text(self.small_content)

        # Ensure the file is not LFS tracked anymore
        self.repo.auto_track_large_files()
        self.assertFalse(is_tracked_with_lfs(txt_large_file))

    def test_checkout_non_existing_branch(self):
        self.assertRaises(EnvironmentError, self.repo.git_checkout, "brand-new-branch")

    def test_checkout_new_branch(self):
        self.repo.git_checkout("new-branch", create_branch_ok=True)
        self.assertEqual(self.repo.current_branch, "new-branch")

    def test_is_not_tracked_upstream(self):
        self.repo.git_checkout("new-branch", create_branch_ok=True)
        self.assertFalse(is_tracked_upstream(self.repo.local_dir))

    def test_no_branch_checked_out_raises(self):
        head_commit_ref = run_subprocess("git show --oneline -s", folder=self.repo_path).stdout.split()[0]

        self.repo.git_checkout(head_commit_ref)
        self.assertRaises(OSError, is_tracked_upstream, self.repo.local_dir)

    def test_repo_init_checkout_default_revision(self):
        # Instantiate repository on a given revision
        repo = Repository(self.repo_path, revision="new-branch")
        self.assertEqual(repo.current_branch, "new-branch")

        # The revision should be kept when re-initializing the repo
        repo_2 = Repository(self.repo_path)
        self.assertEqual(repo_2.current_branch, "new-branch")

    def test_repo_init_checkout_revision(self):
        current_head_hash = self.repo.git_head_hash()

        (self.repo_path / "file.txt").write_text("hello world")

        self.repo.git_add()
        self.repo.git_commit("Add file.txt")

        new_head_hash = self.repo.git_head_hash()

        self.assertNotEqual(current_head_hash, new_head_hash)

        previous_head_repo = Repository(self.repo_path, revision=current_head_hash)
        files = os.listdir(previous_head_repo.local_dir)
        self.assertNotIn("file.txt", files)

        current_head_repo = Repository(self.repo_path, revision=new_head_hash)
        files = os.listdir(current_head_repo.local_dir)
        self.assertIn("file.txt", files)

    def test_repo_user(self):
        _ = Repository(self.repo_path, use_auth_token=TOKEN)
        username = run_subprocess("git config user.name", folder=self.repo_path).stdout
        email = run_subprocess("git config user.email", folder=self.repo_path).stdout

        # hardcode values to avoid another api call to whoami
        self.assertEqual(username.strip(), "Dummy User")
        self.assertEqual(email.strip(), "julien@huggingface.co")

    def test_repo_passed_user(self):
        _ = Repository(
            self.repo_path,
            use_auth_token=TOKEN,  # token ignored
            git_user="RANDOM_USER",
            git_email="EMAIL@EMAIL.EMAIL",
        )
        username = run_subprocess("git config user.name", folder=self.repo_path).stdout
        email = run_subprocess("git config user.email", folder=self.repo_path).stdout

        self.assertEqual(username.strip(), "RANDOM_USER")
        self.assertEqual(email.strip(), "EMAIL@EMAIL.EMAIL")

    def test_add_tag(self):
        self.repo.add_tag("v4.6.0")
        self.assertTrue(self.repo.tag_exists("v4.6.0"))

    def test_add_annotated_tag(self):
        self.repo.add_tag("v4.6.0", message="This is an annotated tag")
        self.assertTrue(self.repo.tag_exists("v4.6.0"))

        result = run_subprocess("git tag -n9", folder=self.repo_path).stdout.strip()
        self.assertIn("This is an annotated tag", result)

    def test_delete_tag(self):
        self.repo.add_tag("v4.6.0", message="This is an annotated tag")
        self.assertTrue(self.repo.tag_exists("v4.6.0"))

        self.repo.delete_tag("v4.6.0")
        self.assertFalse(self.repo.tag_exists("v4.6.0"))

    def test_repo_clean(self):
        self.assertTrue(self.repo.is_repo_clean())
        (self.repo_path / "file.txt").write_text("hello world")
        self.assertFalse(self.repo.is_repo_clean())


class TestRepositoryDataset(RepositoryTestAbstract):
    """Class to test that cloning from a different repo_type works fine."""

    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        super().setUpClass()
        cls._api.set_access_token(TOKEN)
        cls._token = TOKEN

        cls.repo_url = cls._api.create_repo(repo_id=repo_name(), repo_type="dataset")
        cls.repo_id = cls.repo_url.repo_id
        cls._api.upload_file(
            path_or_fileobj=cls.binary_content.encode(),
            path_in_repo="file.txt",
            repo_id=cls.repo_id,
            repo_type="dataset",
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        try:
            cls._api.delete_repo(repo_id=cls.repo_id)
        except requests.exceptions.HTTPError:
            pass

    @retry_endpoint
    def test_clone_dataset_with_endpoint_explicit_repo_type(self):
        Repository(
            self.repo_path,
            clone_from=self.repo_url,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )
        self.assertTrue((self.repo_path / "file.txt").exists())

    @retry_endpoint
    def test_clone_dataset_with_endpoint_implicit_repo_type(self):
        self.assertIn("dataset", self.repo_url)  # Implicit
        Repository(
            self.repo_path,
            clone_from=self.repo_url,
            git_user="ci",
            git_email="ci@dummy.com",
        )
        self.assertTrue((self.repo_path / "file.txt").exists())

    @retry_endpoint
    def test_clone_dataset_with_repo_id_and_repo_type(self):
        Repository(
            self.repo_path,
            clone_from=self.repo_id,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )
        self.assertTrue((self.repo_path / "file.txt").exists())

    @retry_endpoint
    def test_clone_dataset_no_ci_user_and_email(self):
        Repository(self.repo_path, clone_from=self.repo_id, repo_type="dataset")
        self.assertTrue((self.repo_path / "file.txt").exists())

    @retry_endpoint
    def test_clone_dataset_with_repo_name_and_repo_type_fails(self):
        with self.assertRaises(EnvironmentError):
            Repository(
                self.repo_path,
                clone_from=self.repo_id.split("/")[1],
                repo_type="dataset",
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )
