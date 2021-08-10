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
import tempfile
import time
import unittest
from io import BytesIO

import requests
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repository import Repository, is_tracked_with_lfs

from .testing_constants import ENDPOINT_STAGING, PASS, USER
from .testing_utils import set_write_permission_and_retry, with_production_testing


REPO_NAME = "repo-{}".format(int(time.time() * 10e3))


WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.path.join("fixtures", "working_repo_2")
)

DATASET_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.path.join("fixtures", "tiny_dataset")
)
WORKING_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.path.join("fixtures", "working_dataset")
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
        try:
            self._api.delete_repo(token=self._token, name=REPO_NAME)
        except requests.exceptions.HTTPError:
            pass

        try:
            self._api.delete_repo(
                token=self._token, organization="valid_org", name=REPO_NAME
            )
        except requests.exceptions.HTTPError:
            pass

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
        Repository(
            WORKING_REPO_DIR, clone_from=self._repo_url, use_auth_token=self._token
        )
        Repository(
            WORKING_REPO_DIR, clone_from=self._repo_url, use_auth_token=self._token
        )

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

        shutil.rmtree(WORKING_REPO_DIR)

    def test_clone_with_endpoint(self):
        clone = Repository(
            REPO_NAME,
            clone_from=f"{ENDPOINT_STAGING}/valid_org/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            with open("dummy.txt", "w") as f:
                f.write("hello")
            with open("model.bin", "w") as f:
                f.write("hello")

        shutil.rmtree(REPO_NAME)

        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/valid_org/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    def test_clone_with_repo_name_and_org(self):
        clone = Repository(
            REPO_NAME,
            clone_from=f"valid_org/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            with open("dummy.txt", "w") as f:
                f.write("hello")
            with open("model.bin", "w") as f:
                f.write("hello")

        shutil.rmtree(REPO_NAME)

        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=f"valid_org/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    def test_clone_with_repo_name_and_user_namespace(self):
        clone = Repository(
            REPO_NAME,
            clone_from=f"{USER}/{REPO_NAME}",
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

        shutil.rmtree(REPO_NAME)

        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("dummy.txt" in files)
        self.assertTrue("model.bin" in files)

    def test_clone_with_repo_name_and_no_namespace(self):
        self.assertRaises(
            OSError,
            Repository,
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=REPO_NAME,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

    def test_clone_with_repo_name_user_and_no_auth_token(self):
        # Create repo
        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        # Instantiate it without token
        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            git_user="ci",
            git_email="ci@dummy.com",
        )

    def test_clone_with_repo_name_org_and_no_auth_token(self):
        # Create repo
        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            use_auth_token=self._token,
            clone_from=f"valid_org/{REPO_NAME}",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        # Instantiate it without token
        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from=f"valid_org/{REPO_NAME}",
            git_user="ci",
            git_email="ci@dummy.com",
        )

    def test_clone_not_hf_url(self):
        # Should not error out
        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from="https://hf.co/hf-internal-testing/huggingface-hub-dummy-repository",
        )

    @with_production_testing
    def test_clone_repo_at_root(self):
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from="bert-base-cased",
        )

        shutil.rmtree(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        Repository(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            clone_from="https://huggingface.co/bert-base-cased",
        )


class RepositoryAutoLFSTrackingTest(RepositoryCommonTest):
    @classmethod
    def setUpClass(cls) -> None:
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)

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

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)

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


class RepositoryDatasetTest(RepositoryCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def tearDown(self):
        try:
            self._api.delete_repo(
                token=self._token, name=REPO_NAME, repo_type="dataset"
            )
        except requests.exceptions.HTTPError:
            try:
                self._api.delete_repo(
                    token=self._token,
                    organization="valid_org",
                    name=REPO_NAME,
                    repo_type="dataset",
                )
            except requests.exceptions.HTTPError:
                pass

        shutil.rmtree(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}", onerror=set_write_permission_and_retry
        )

    def test_clone_with_endpoint(self):
        clone = Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/datasets/{USER}/{REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            for file in os.listdir(DATASET_FIXTURE):
                shutil.copyfile(pathlib.Path(DATASET_FIXTURE) / file, file)

        shutil.rmtree(f"{WORKING_DATASET_DIR}/{REPO_NAME}")

        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"{ENDPOINT_STAGING}/datasets/{USER}/{REPO_NAME}",
            use_auth_token=self._token,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_DATASET_DIR}/{REPO_NAME}")
        self.assertTrue("some_text.txt" in files)
        self.assertTrue("test.py" in files)

    def test_clone_with_repo_name_and_org(self):
        clone = Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"valid_org/{REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            for file in os.listdir(DATASET_FIXTURE):
                shutil.copyfile(pathlib.Path(DATASET_FIXTURE) / file, file)

        shutil.rmtree(f"{WORKING_DATASET_DIR}/{REPO_NAME}")

        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"valid_org/{REPO_NAME}",
            use_auth_token=self._token,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_DATASET_DIR}/{REPO_NAME}")
        self.assertTrue("some_text.txt" in files)
        self.assertTrue("test.py" in files)

    def test_clone_with_repo_name_and_user_namespace(self):
        clone = Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with clone.commit("Commit"):
            for file in os.listdir(DATASET_FIXTURE):
                shutil.copyfile(pathlib.Path(DATASET_FIXTURE) / file, file)

        shutil.rmtree(f"{WORKING_DATASET_DIR}/{REPO_NAME}")

        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            use_auth_token=self._token,
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

        files = os.listdir(f"{WORKING_DATASET_DIR}/{REPO_NAME}")
        self.assertTrue("some_text.txt" in files)
        self.assertTrue("test.py" in files)

    def test_clone_with_repo_name_and_no_namespace(self):
        self.assertRaises(
            OSError,
            Repository,
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=REPO_NAME,
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

    def test_clone_with_repo_name_user_and_no_auth_token(self):
        # Create repo
        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        # Instantiate it without token
        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"{USER}/{REPO_NAME}",
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )

    def test_clone_with_repo_name_org_and_no_auth_token(self):
        # Create repo
        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"valid_org/{REPO_NAME}",
            repo_type="dataset",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        # Instantiate it without token
        Repository(
            f"{WORKING_DATASET_DIR}/{REPO_NAME}",
            clone_from=f"valid_org/{REPO_NAME}",
            repo_type="dataset",
            git_user="ci",
            git_email="ci@dummy.com",
        )
