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
import datetime
import os
import re
import subprocess
import tempfile
import time
import types
import unittest
import uuid
import warnings
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import fields
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set, Union
from unittest.mock import Mock, patch
from urllib.parse import quote, urlparse

import pytest
import requests
from requests.exceptions import HTTPError

import huggingface_hub.lfs
from huggingface_hub import HfApi, SpaceHardware, SpaceStage, SpaceStorage
from huggingface_hub._commit_api import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    _fetch_upload_modes,
)
from huggingface_hub.community import DiscussionComment, DiscussionWithDetails
from huggingface_hub.constants import (
    REPO_TYPE_DATASET,
    REPO_TYPE_MODEL,
    REPO_TYPE_SPACE,
    SPACES_SDK_TYPES,
)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import (
    AccessRequest,
    Collection,
    CommitInfo,
    DatasetInfo,
    MetricInfo,
    ModelInfo,
    RepoSibling,
    RepoUrl,
    SpaceInfo,
    SpaceRuntime,
    WebhookInfo,
    WebhookWatchedItem,
    repo_type_and_id_from_hf_id,
)
from huggingface_hub.repocard_data import DatasetCardData, ModelCardData
from huggingface_hub.utils import (
    BadRequestError,
    EntryNotFoundError,
    HfHubHTTPError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    SafetensorsFileMetadata,
    SafetensorsParsingError,
    SafetensorsRepoMetadata,
    SoftTemporaryDirectory,
    TensorInfo,
    get_session,
    hf_raise_for_status,
    logging,
)
from huggingface_hub.utils.endpoint_helpers import (
    _is_emission_within_threshold,
)

from .testing_constants import (
    ENDPOINT_STAGING,
    FULL_NAME,
    OTHER_TOKEN,
    OTHER_USER,
    TOKEN,
    USER,
)
from .testing_utils import (
    DUMMY_DATASET_ID,
    DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT,
    DUMMY_MODEL_ID,
    DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
    SAMPLE_DATASET_IDENTIFIER,
    repo_name,
    require_git_lfs,
    rmtree_with_retry,
    use_tmp_repo,
    with_production_testing,
)


logger = logging.get_logger(__name__)

WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/working_repo")
LARGE_FILE_14MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.epub"
LARGE_FILE_18MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.pdf"

INVALID_MODELCARD = """
---
model-index: foo
---

This is a modelcard with an invalid metadata section.
"""


class HfApiCommonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Share the valid token in all tests below."""
        cls._token = TOKEN
        cls._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


def test_repo_id_no_warning():
    # tests that passing repo_id as positional arg doesn't raise any warnings
    # for {create, delete}_repo and update_repo_visibility
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    with warnings.catch_warnings(record=True) as record:
        repo_id = api.create_repo(repo_name()).repo_id
        api.update_repo_visibility(repo_id, private=True)
        api.delete_repo(repo_id)
    assert not len(record)


class HfApiRepoFileExistsTest(HfApiCommonTest):
    def setUp(self) -> None:
        super().setUp()
        self.repo_id = self._api.create_repo(repo_name(), private=True).repo_id
        self.upload = self._api.upload_file(repo_id=self.repo_id, path_in_repo="file.txt", path_or_fileobj=b"content")

    def tearDown(self) -> None:
        self._api.delete_repo(self.repo_id)
        return super().tearDown()

    def test_repo_exists(self):
        self.assertTrue(self._api.repo_exists(self.repo_id))
        self.assertFalse(self._api.repo_exists(self.repo_id, token=False))  # private repo
        self.assertFalse(self._api.repo_exists("repo-that-does-not-exist"))  # missing repo

    def test_revision_exists(self):
        assert self._api.revision_exists(self.repo_id, "main")
        assert not self._api.revision_exists(self.repo_id, "revision-that-does-not-exist")  # missing revision
        assert not self._api.revision_exists(self.repo_id, "main", token=False)  # private repo
        assert not self._api.revision_exists("repo-that-does-not-exist", "main")  # missing repo

    @patch("huggingface_hub.file_download.ENDPOINT", "https://hub-ci.huggingface.co")
    @patch(
        "huggingface_hub.file_download.HUGGINGFACE_CO_URL_TEMPLATE",
        "https://hub-ci.huggingface.co/{repo_id}/resolve/{revision}/{filename}",
    )
    def test_file_exists(self):
        self.assertTrue(self._api.file_exists(self.repo_id, "file.txt"))
        self.assertFalse(self._api.file_exists("repo-that-does-not-exist", "file.txt"))  # missing repo
        self.assertFalse(self._api.file_exists(self.repo_id, "file-does-not-exist"))  # missing file
        self.assertFalse(
            self._api.file_exists(self.repo_id, "file.txt", revision="revision-that-does-not-exist")
        )  # missing revision
        self.assertFalse(self._api.file_exists(self.repo_id, "file.txt", token=False))  # private repo


class HfApiEndpointsTest(HfApiCommonTest):
    def test_whoami_with_passing_token(self):
        info = self._api.whoami(token=self._token)
        self.assertEqual(info["name"], USER)
        self.assertEqual(info["fullname"], FULL_NAME)
        self.assertIsInstance(info["orgs"], list)
        valid_org = [org for org in info["orgs"] if org["name"] == "valid_org"][0]
        self.assertEqual(valid_org["fullname"], "Dummy Org")

    @patch("huggingface_hub.utils._headers.get_token", return_value=TOKEN)
    def test_whoami_with_implicit_token_from_login(self, mock_get_token: Mock) -> None:
        """Test using `whoami` after a `huggingface-cli login`."""
        with patch.object(self._api, "token", None):  # no default token
            info = self._api.whoami()
        self.assertEqual(info["name"], USER)

    @patch("huggingface_hub.utils._headers.get_token")
    def test_whoami_with_implicit_token_from_hf_api(self, mock_get_token: Mock) -> None:
        """Test using `whoami` with token from the HfApi client."""
        info = self._api.whoami()
        self.assertEqual(info["name"], USER)
        mock_get_token.assert_not_called()

    def test_delete_repo_error_message(self):
        # test for #751
        # See https://github.com/huggingface/huggingface_hub/issues/751
        with self.assertRaisesRegex(
            requests.exceptions.HTTPError,
            re.compile(
                r"404 Client Error(.+)\(Request ID: .+\)(.*)Repository Not Found",
                flags=re.DOTALL,
            ),
        ):
            self._api.delete_repo("repo-that-does-not-exist")

    def test_delete_repo_missing_ok(self) -> None:
        self._api.delete_repo("repo-that-does-not-exist", missing_ok=True)

    def test_create_update_and_delete_repo(self):
        repo_id = self._api.create_repo(repo_id=repo_name()).repo_id
        res = self._api.update_repo_visibility(repo_id=repo_id, private=True)
        assert res["private"]
        res = self._api.update_repo_visibility(repo_id=repo_id, private=False)
        assert not res["private"]
        self._api.delete_repo(repo_id=repo_id)

    def test_create_update_and_delete_model_repo(self):
        repo_id = self._api.create_repo(repo_id=repo_name(), repo_type=REPO_TYPE_MODEL).repo_id
        res = self._api.update_repo_visibility(repo_id=repo_id, private=True, repo_type=REPO_TYPE_MODEL)
        assert res["private"]
        res = self._api.update_repo_visibility(repo_id=repo_id, private=False, repo_type=REPO_TYPE_MODEL)
        assert not res["private"]
        self._api.delete_repo(repo_id=repo_id, repo_type=REPO_TYPE_MODEL)

    def test_create_update_and_delete_dataset_repo(self):
        repo_id = self._api.create_repo(repo_id=repo_name(), repo_type=REPO_TYPE_DATASET).repo_id
        res = self._api.update_repo_visibility(repo_id=repo_id, private=True, repo_type=REPO_TYPE_DATASET)
        assert res["private"]
        res = self._api.update_repo_visibility(repo_id=repo_id, private=False, repo_type=REPO_TYPE_DATASET)
        assert not res["private"]
        self._api.delete_repo(repo_id=repo_id, repo_type=REPO_TYPE_DATASET)

    def test_create_update_and_delete_space_repo(self):
        with pytest.raises(ValueError, match=r"No space_sdk provided.*"):
            self._api.create_repo(repo_id=repo_name(), repo_type=REPO_TYPE_SPACE, space_sdk=None)
        with pytest.raises(ValueError, match=r"Invalid space_sdk.*"):
            self._api.create_repo(repo_id=repo_name(), repo_type=REPO_TYPE_SPACE, space_sdk="something")

        for sdk in SPACES_SDK_TYPES:
            repo_id = self._api.create_repo(repo_id=repo_name(), repo_type=REPO_TYPE_SPACE, space_sdk=sdk).repo_id
            res = self._api.update_repo_visibility(repo_id=repo_id, private=True, repo_type=REPO_TYPE_SPACE)
            assert res["private"]
            res = self._api.update_repo_visibility(repo_id=repo_id, private=False, repo_type=REPO_TYPE_SPACE)
            assert not res["private"]
            self._api.delete_repo(repo_id=repo_id, repo_type=REPO_TYPE_SPACE)

    def test_move_repo_normal_usage(self):
        repo_id = f"{USER}/{repo_name()}"
        new_repo_id = f"{USER}/{repo_name()}"

        # Spaces not tested on staging (error 500)
        for repo_type in [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET]:
            self._api.create_repo(repo_id=repo_id, repo_type=repo_type)
            self._api.move_repo(from_id=repo_id, to_id=new_repo_id, repo_type=repo_type)
            self._api.delete_repo(repo_id=new_repo_id, repo_type=repo_type)

    def test_move_repo_target_already_exists(self) -> None:
        repo_id_1 = f"{USER}/{repo_name()}"
        repo_id_2 = f"{USER}/{repo_name()}"

        self._api.create_repo(repo_id=repo_id_1)
        self._api.create_repo(repo_id=repo_id_2)

        with pytest.raises(HfHubHTTPError, match=r"A model repository called .* already exists"):
            self._api.move_repo(from_id=repo_id_1, to_id=repo_id_2, repo_type=REPO_TYPE_MODEL)

        self._api.delete_repo(repo_id=repo_id_1)
        self._api.delete_repo(repo_id=repo_id_2)

    def test_move_repo_invalid_repo_id(self) -> None:
        """Test from_id and to_id must be in the form `"namespace/repo_name"`."""
        with pytest.raises(ValueError, match=r"Invalid repo_id*"):
            self._api.move_repo(from_id="namespace/repo_name", to_id="invalid_repo_id")

        with pytest.raises(ValueError, match=r"Invalid repo_id*"):
            self._api.move_repo(from_id="invalid_repo_id", to_id="namespace/repo_name")


class CommitApiTest(HfApiCommonTest):
    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_file = os.path.join(self.tmp_dir, "temp")
        self.tmp_file_content = "Content of the file"
        with open(self.tmp_file, "w+") as f:
            f.write(self.tmp_file_content)
        os.makedirs(os.path.join(self.tmp_dir, "nested"))
        self.nested_tmp_file = os.path.join(self.tmp_dir, "nested", "file.bin")
        with open(self.nested_tmp_file, "wb+") as f:
            f.truncate(1024 * 1024)

        self.addCleanup(rmtree_with_retry, self.tmp_dir)

    def test_upload_file_validation(self) -> None:
        with self.assertRaises(ValueError, msg="Wrong repo type"):
            self._api.upload_file(
                path_or_fileobj=self.tmp_file,
                path_in_repo="README.md",
                repo_id="something",
                repo_type="this type does not exist",
            )

    def test_commit_operation_validation(self):
        with open(self.tmp_file, "rt") as ftext:
            with self.assertRaises(
                ValueError,
                msg="If you passed a file-like object, make sure it is in binary mode",
            ):
                CommitOperationAdd(path_or_fileobj=ftext, path_in_repo="README.md")  # type: ignore

        with self.assertRaises(ValueError, msg="path_or_fileobj is str but does not point to a file"):
            CommitOperationAdd(
                path_or_fileobj=os.path.join(self.tmp_dir, "nofile.pth"),
                path_in_repo="README.md",
            )

    @use_tmp_repo()
    def test_upload_file_str_path(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        return_val = self._api.upload_file(
            path_or_fileobj=self.tmp_file,
            path_in_repo="temp/new_file.md",
            repo_id=repo_id,
        )
        self.assertEqual(return_val, f"{repo_url}/blob/main/temp/new_file.md")
        self.assertIsInstance(return_val, CommitInfo)

        with SoftTemporaryDirectory() as cache_dir:
            with open(hf_hub_download(repo_id=repo_id, filename="temp/new_file.md", cache_dir=cache_dir)) as f:
                self.assertEqual(f.read(), self.tmp_file_content)

    @use_tmp_repo()
    def test_upload_file_pathlib_path(self, repo_url: RepoUrl) -> None:
        """Regression test for https://github.com/huggingface/huggingface_hub/issues/1246."""
        self._api.upload_file(path_or_fileobj=Path(self.tmp_file), path_in_repo="README.md", repo_id=repo_url.repo_id)
        self.assertIn("README.md", self._api.list_repo_files(repo_id=repo_url.repo_id))

    @use_tmp_repo()
    def test_upload_file_fileobj(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        with open(self.tmp_file, "rb") as filestream:
            return_val = self._api.upload_file(
                path_or_fileobj=filestream,
                path_in_repo="temp/new_file.md",
                repo_id=repo_id,
            )
        self.assertEqual(return_val, f"{repo_url}/blob/main/temp/new_file.md")

        with SoftTemporaryDirectory() as cache_dir:
            with open(hf_hub_download(repo_id=repo_id, filename="temp/new_file.md", cache_dir=cache_dir)) as f:
                self.assertEqual(f.read(), self.tmp_file_content)

    @use_tmp_repo()
    def test_upload_file_bytesio(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        content = BytesIO(b"File content, but in bytes IO")
        return_val = self._api.upload_file(
            path_or_fileobj=content,
            path_in_repo="temp/new_file.md",
            repo_id=repo_id,
        )
        self.assertEqual(return_val, f"{repo_url}/blob/main/temp/new_file.md")

        with SoftTemporaryDirectory() as cache_dir:
            with open(hf_hub_download(repo_id=repo_id, filename="temp/new_file.md", cache_dir=cache_dir)) as f:
                self.assertEqual(f.read(), content.getvalue().decode())

    def test_create_repo_return_value(self) -> None:
        REPO_NAME = repo_name("org")
        url = self._api.create_repo(repo_id=REPO_NAME)
        self.assertIsInstance(url, str)
        self.assertIsInstance(url, RepoUrl)
        self.assertEqual(url.repo_id, f"{USER}/{REPO_NAME}")
        self._api.delete_repo(repo_id=url.repo_id)

    def test_create_repo_org_token_fail(self):
        REPO_NAME = repo_name("org")
        with pytest.raises(HfHubHTTPError, match="Invalid username or password"):
            self._api.create_repo(repo_id=REPO_NAME, token="api_org_dummy_token")

    @patch("huggingface_hub.utils._headers.get_token", return_value="api_org_dummy_token")
    def test_create_repo_org_token_none_fail(self, mock_get_token: Mock):
        with pytest.raises(HfHubHTTPError, match="Invalid username or password"):
            with patch.object(self._api, "token", None):  # no default token
                self._api.create_repo(repo_id=repo_name("org"))

    def test_create_repo_already_exists_but_no_write_permission(self):
        # Create under other user namespace
        repo_id = self._api.create_repo(repo_id=repo_name(), token=OTHER_TOKEN).repo_id

        # Try to create with our namespace -> should not fail as the repo already exists
        self._api.create_repo(repo_id=repo_id, token=TOKEN, exist_ok=True)

        # Clean up
        self._api.delete_repo(repo_id=repo_id, token=OTHER_TOKEN)

    @use_tmp_repo()
    def test_upload_file_create_pr(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        return_val = self._api.upload_file(
            path_or_fileobj=self.tmp_file_content.encode(),
            path_in_repo="temp/new_file.md",
            repo_id=repo_id,
            create_pr=True,
        )
        self.assertEqual(return_val, f"{repo_url}/blob/{quote('refs/pr/1', safe='')}/temp/new_file.md")
        self.assertIsInstance(return_val, CommitInfo)

        with SoftTemporaryDirectory() as cache_dir:
            with open(
                hf_hub_download(
                    repo_id=repo_id, filename="temp/new_file.md", revision="refs/pr/1", cache_dir=cache_dir
                )
            ) as f:
                self.assertEqual(f.read(), self.tmp_file_content)

    @use_tmp_repo()
    def test_delete_file(self, repo_url: RepoUrl) -> None:
        self._api.upload_file(
            path_or_fileobj=self.tmp_file,
            path_in_repo="temp/new_file.md",
            repo_id=repo_url.repo_id,
        )
        return_val = self._api.delete_file(path_in_repo="temp/new_file.md", repo_id=repo_url.repo_id)
        self.assertIsInstance(return_val, CommitInfo)

        with self.assertRaises(EntryNotFoundError):
            # Should raise a 404
            hf_hub_download(repo_url.repo_id, "temp/new_file.md")

    def test_get_full_repo_name(self):
        repo_name_with_no_org = self._api.get_full_repo_name("model")
        self.assertEqual(repo_name_with_no_org, f"{USER}/model")

        repo_name_with_no_org = self._api.get_full_repo_name("model", organization="org")
        self.assertEqual(repo_name_with_no_org, "org/model")

    @use_tmp_repo()
    def test_upload_folder(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id

        # Upload folder
        url = self._api.upload_folder(folder_path=self.tmp_dir, path_in_repo="temp/dir", repo_id=repo_id)
        self.assertEqual(
            url,
            f"{self._api.endpoint}/{repo_id}/tree/main/temp/dir",
        )
        self.assertIsInstance(url, CommitInfo)

        # Check files are uploaded
        for rpath in ["temp", "nested/file.bin"]:
            local_path = os.path.join(self.tmp_dir, rpath)
            remote_path = f"temp/dir/{rpath}"
            filepath = hf_hub_download(
                repo_id=repo_id, filename=remote_path, revision="main", use_auth_token=self._token
            )
            assert filepath is not None
            with open(filepath, "rb") as downloaded_file:
                content = downloaded_file.read()
            with open(local_path, "rb") as local_file:
                expected_content = local_file.read()
            self.assertEqual(content, expected_content)

        # Re-uploading the same folder twice should be fine
        return_val = self._api.upload_folder(folder_path=self.tmp_dir, path_in_repo="temp/dir", repo_id=repo_id)
        self.assertIsInstance(return_val, CommitInfo)

    @use_tmp_repo()
    def test_upload_folder_create_pr(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id

        # Upload folder as a new PR
        return_val = self._api.upload_folder(
            folder_path=self.tmp_dir, path_in_repo="temp/dir", repo_id=repo_id, create_pr=True
        )
        self.assertEqual(return_val, f"{self._api.endpoint}/{repo_id}/tree/refs%2Fpr%2F1/temp/dir")

        # Check files are uploaded
        for rpath in ["temp", "nested/file.bin"]:
            local_path = os.path.join(self.tmp_dir, rpath)
            filepath = hf_hub_download(repo_id=repo_id, filename=f"temp/dir/{rpath}", revision="refs/pr/1")
            assert Path(local_path).read_bytes() == Path(filepath).read_bytes()

    def test_upload_folder_default_path_in_repo(self):
        REPO_NAME = repo_name("upload_folder_to_root")
        self._api.create_repo(repo_id=REPO_NAME, exist_ok=False)
        url = self._api.upload_folder(folder_path=self.tmp_dir, repo_id=f"{USER}/{REPO_NAME}")
        # URL to root of repository
        self.assertEqual(url, f"{self._api.endpoint}/{USER}/{REPO_NAME}/tree/main/")

    @use_tmp_repo()
    def test_upload_folder_git_folder_excluded(self, repo_url: RepoUrl) -> None:
        # Simulate a folder with a .git folder
        def _create_file(*parts) -> None:
            path = Path(self.tmp_dir, *parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content")

        _create_file(".git", "file.txt")
        _create_file(".cache", "huggingface", "file.txt")
        _create_file(".git", "folder", "file.txt")
        _create_file("folder", ".git", "file.txt")
        _create_file("folder", ".cache", "huggingface", "file.txt")
        _create_file("folder", ".git", "folder", "file.txt")
        _create_file(".git_something", "file.txt")
        _create_file("file.git")

        # Upload folder and check that .git folder is excluded
        self._api.upload_folder(folder_path=self.tmp_dir, repo_id=repo_url.repo_id)
        self.assertEqual(
            set(self._api.list_repo_files(repo_id=repo_url.repo_id)),
            {".gitattributes", ".git_something/file.txt", "file.git", "temp", "nested/file.bin"},
        )

    @use_tmp_repo()
    def test_upload_folder_gitignore_already_exists(self, repo_url: RepoUrl) -> None:
        # Ignore nested folder
        self._api.upload_file(path_or_fileobj=b"nested/*\n", path_in_repo=".gitignore", repo_id=repo_url.repo_id)

        # Upload folder
        self._api.upload_folder(folder_path=self.tmp_dir, repo_id=repo_url.repo_id)

        # Check nested file not uploaded
        assert not self._api.file_exists(repo_url.repo_id, "nested/file.bin")

    @use_tmp_repo()
    def test_upload_folder_gitignore_in_commit(self, repo_url: RepoUrl) -> None:
        # Create .gitignore file locally
        (Path(self.tmp_dir) / ".gitignore").write_text("nested/*\n")

        # Upload folder
        self._api.upload_folder(folder_path=self.tmp_dir, repo_id=repo_url.repo_id)

        # Check nested file not uploaded
        assert not self._api.file_exists(repo_url.repo_id, "nested/file.bin")

    @use_tmp_repo()
    def test_create_commit_create_pr(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id

        # Upload a first file
        self._api.upload_file(path_or_fileobj=self.tmp_file, path_in_repo="temp/new_file.md", repo_id=repo_id)

        # Create a commit with a PR
        operations = [
            CommitOperationDelete(path_in_repo="temp/new_file.md"),
            CommitOperationAdd(path_in_repo="buffer", path_or_fileobj=b"Buffer data"),
        ]
        resp = self._api.create_commit(
            operations=operations, commit_message="Test create_commit", repo_id=repo_id, create_pr=True
        )

        # Check commit info
        self.assertIsInstance(resp, CommitInfo)
        commit_id = resp.oid
        self.assertIn("pr_revision='refs/pr/1'", repr(resp))
        self.assertIsInstance(commit_id, str)
        self.assertGreater(len(commit_id), 0)
        self.assertEqual(resp.commit_url, f"{self._api.endpoint}/{repo_id}/commit/{commit_id}")
        self.assertEqual(resp.commit_message, "Test create_commit")
        self.assertEqual(resp.commit_description, "")
        self.assertEqual(resp.pr_url, f"{self._api.endpoint}/{repo_id}/discussions/1")
        self.assertEqual(resp.pr_num, 1)
        self.assertEqual(resp.pr_revision, "refs/pr/1")

        # File doesn't exist on main...
        with self.assertRaises(HTTPError) as ctx:
            # Should raise a 404
            self._api.hf_hub_download(repo_id, "buffer")
            self.assertEqual(ctx.exception.response.status_code, 404)

        # ...but exists on PR
        filepath = self._api.hf_hub_download(filename="buffer", repo_id=repo_id, revision="refs/pr/1")
        with open(filepath, "rb") as downloaded_file:
            content = downloaded_file.read()
        self.assertEqual(content, b"Buffer data")

    def test_create_commit_create_pr_against_branch(self):
        repo_id = f"{USER}/{repo_name()}"

        # Create repo and create a non-main branch
        self._api.create_repo(repo_id=repo_id, exist_ok=False)
        self._api.create_branch(repo_id=repo_id, branch="test_branch")
        head = self._api.list_repo_refs(repo_id=repo_id).branches[0].target_commit

        # Create PR against non-main branch works
        resp = self._api.create_commit(
            operations=[],
            commit_message="PR against existing branch",
            repo_id=repo_id,
            revision="test_branch",
            create_pr=True,
        )
        self.assertIsInstance(resp, CommitInfo)

        # Create PR against a oid fails
        with self.assertRaises(RevisionNotFoundError):
            self._api.create_commit(
                operations=[],
                commit_message="PR against a oid",
                repo_id=repo_id,
                revision=head,
                create_pr=True,
            )

        # Create PR against a non-existing branch fails
        with self.assertRaises(RevisionNotFoundError):
            self._api.create_commit(
                operations=[],
                commit_message="PR against missing branch",
                repo_id=repo_id,
                revision="missing_branch",
                create_pr=True,
            )

        # Cleanup
        self._api.delete_repo(repo_id=repo_id)

    def test_create_commit_create_pr_on_foreign_repo(self):
        # Create a repo with another user. The normal CI user don't have rights on it.
        # We must be able to create a PR on it
        foreign_api = HfApi(token=OTHER_TOKEN)
        foreign_repo_url = foreign_api.create_repo(repo_id=repo_name("repo-for-hfh-ci"))

        self._api.create_commit(
            operations=[
                CommitOperationAdd(path_in_repo="regular.txt", path_or_fileobj=b"File content"),
                CommitOperationAdd(path_in_repo="lfs.pkl", path_or_fileobj=b"File content"),
            ],
            commit_message="PR on foreign repo",
            repo_id=foreign_repo_url.repo_id,
            create_pr=True,
        )

        foreign_api.delete_repo(repo_id=foreign_repo_url.repo_id)

    @use_tmp_repo()
    def test_create_commit(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        self._api.upload_file(path_or_fileobj=self.tmp_file, path_in_repo="temp/new_file.md", repo_id=repo_id)
        with open(self.tmp_file, "rb") as fileobj:
            operations = [
                CommitOperationDelete(path_in_repo="temp/new_file.md"),
                CommitOperationAdd(path_in_repo="buffer", path_or_fileobj=b"Buffer data"),
                CommitOperationAdd(
                    path_in_repo="bytesio",
                    path_or_fileobj=BytesIO(b"BytesIO data"),
                ),
                CommitOperationAdd(path_in_repo="fileobj", path_or_fileobj=fileobj),
                CommitOperationAdd(
                    path_in_repo="nested/path",
                    path_or_fileobj=self.tmp_file,
                ),
            ]
            resp = self._api.create_commit(operations=operations, commit_message="Test create_commit", repo_id=repo_id)
            # Check commit info
            self.assertIsInstance(resp, CommitInfo)
            self.assertIsNone(resp.pr_url)  # No pr created
            self.assertIsNone(resp.pr_num)
            self.assertIsNone(resp.pr_revision)

        with self.assertRaises(HTTPError):
            # Should raise a 404
            hf_hub_download(repo_id, "temp/new_file.md")

        for path, expected_content in [
            ("buffer", b"Buffer data"),
            ("bytesio", b"BytesIO data"),
            ("fileobj", self.tmp_file_content.encode()),
            ("nested/path", self.tmp_file_content.encode()),
        ]:
            filepath = hf_hub_download(repo_id=repo_id, filename=path, revision="main")
            self.assertTrue(filepath is not None)
            with open(filepath, "rb") as downloaded_file:
                content = downloaded_file.read()
            self.assertEqual(content, expected_content)

    @use_tmp_repo()
    def test_create_commit_conflict(self, repo_url: RepoUrl) -> None:
        # Get commit on main
        repo_id = repo_url.repo_id
        parent_commit = self._api.model_info(repo_id).sha

        # Upload new file
        self._api.upload_file(path_or_fileobj=self.tmp_file, path_in_repo="temp/new_file.md", repo_id=repo_id)

        # Creating a commit with a parent commit that is not the current main should fail
        operations = [
            CommitOperationAdd(path_in_repo="buffer", path_or_fileobj=b"Buffer data"),
        ]
        with self.assertRaises(HTTPError) as exc_ctx:
            self._api.create_commit(
                operations=operations,
                commit_message="Test create_commit",
                repo_id=repo_id,
                parent_commit=parent_commit,
            )
        self.assertEqual(exc_ctx.exception.response.status_code, 412)
        self.assertIn(
            # Check the server message is added to the exception
            "A commit has happened since. Please refresh and try again.",
            str(exc_ctx.exception),
        )

    def test_create_commit_repo_does_not_exist(self) -> None:
        """Test error message is detailed when creating a commit on a missing repo."""
        # Test once with empty commit and once with an addition commit.
        for route, operations in (
            ("commit", []),
            ("preupload", [CommitOperationAdd("config.json", b"content")]),
        ):
            with self.subTest():
                with self.assertRaises(RepositoryNotFoundError) as context:
                    self._api.create_commit(
                        repo_id=f"{USER}/repo_that_do_not_exist",
                        operations=operations,
                        commit_message="fake_message",
                    )

                request_id = context.exception.response.headers.get("X-Request-Id")
                expected_message = (
                    f"404 Client Error. (Request ID: {request_id})\n\nRepository Not"
                    " Found for url:"
                    f" {self._api.endpoint}/api/models/{USER}/repo_that_do_not_exist/{route}/main.\nPlease"
                    " make sure you specified the correct `repo_id` and"
                    " `repo_type`.\nIf you are trying to access a private or gated"
                    " repo, make sure you are authenticated.\nNote: Creating a commit"
                    " assumes that the repo already exists on the Huggingface Hub."
                    " Please use `create_repo` if it's not the case."
                )

                self.assertEqual(str(context.exception), expected_message)

    @patch("huggingface_hub.utils._headers.get_token", return_value=TOKEN)
    def test_create_commit_lfs_file_implicit_token(self, get_token_mock: Mock) -> None:
        """Test that uploading a file as LFS works with cached token.

        Regression test for https://github.com/huggingface/huggingface_hub/pull/1084.
        """
        REPO_NAME = repo_name("create_commit_with_lfs")
        repo_id = f"{USER}/{REPO_NAME}"

        with patch.object(self._api, "token", None):  # no default token
            # Create repo
            self._api.create_repo(repo_id=REPO_NAME, exist_ok=False)

            # Set repo to track png files as LFS
            self._api.create_commit(
                operations=[
                    CommitOperationAdd(
                        path_in_repo=".gitattributes",
                        path_or_fileobj=b"*.png filter=lfs diff=lfs merge=lfs -text",
                    ),
                ],
                commit_message="Update .gitattributes",
                repo_id=repo_id,
            )

            # Upload a PNG file
            self._api.create_commit(
                operations=[
                    CommitOperationAdd(path_in_repo="image.png", path_or_fileobj=b"image data"),
                ],
                commit_message="Test upload lfs file",
                repo_id=repo_id,
            )

            # Check uploaded as LFS
            info = self._api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = {file.rfilename: file for file in info.siblings}
            self.assertIsInstance(siblings["image.png"].lfs, dict)  # LFS file

            # Delete repo
            self._api.delete_repo(repo_id=REPO_NAME)

    @use_tmp_repo()
    def test_create_commit_huge_regular_files(self, repo_url: RepoUrl) -> None:
        """Test committing 12 text files (>100MB in total) at once.

        This was not possible when using `json` format instead of `ndjson`
        on the `/create-commit` endpoint.

        See https://github.com/huggingface/huggingface_hub/pull/1117.
        """
        operations = [
            CommitOperationAdd(
                path_in_repo=f"file-{num}.text",
                path_or_fileobj=b"Hello regular " + b"a" * 1024 * 1024 * 9,
            )
            for num in range(12)
        ]
        self._api.create_commit(
            operations=operations,  # 12*9MB regular => too much for "old" method
            commit_message="Test create_commit with huge regular files",
            repo_id=repo_url.repo_id,
        )

    @use_tmp_repo()
    def test_commit_preflight_on_lots_of_lfs_files(self, repo_url: RepoUrl):
        """Test committing 1300 LFS files at once.

        This was not possible when `_fetch_upload_modes` was not fetching metadata by
        chunks. We are not testing the full upload as it would require to upload 1300
        files which is unnecessary for the test. Having an overall large payload (for
        `/create-commit` endpoint) is tested in `test_create_commit_huge_regular_files`.

        There is also a 25k LFS files limit on the Hub but this is not tested.

        See https://github.com/huggingface/huggingface_hub/pull/1117.
        """
        operations = [
            CommitOperationAdd(
                path_in_repo=f"file-{num}.bin",  # considered as LFS
                path_or_fileobj=b"Hello LFS" + b"a" * 2048,  # big enough sample
            )
            for num in range(1300)
        ]

        # Test `_fetch_upload_modes` preflight ("are they regular or LFS files?")
        _fetch_upload_modes(
            additions=operations,
            repo_type="model",
            repo_id=repo_url.repo_id,
            headers=self._api._build_hf_headers(),
            revision="main",
            endpoint=ENDPOINT_STAGING,
        )
        for operation in operations:
            self.assertEqual(operation._upload_mode, "lfs")
            self.assertFalse(operation._is_committed)
            self.assertFalse(operation._is_uploaded)

    def test_create_commit_repo_id_case_insensitive(self):
        """Test create commit but repo_id is lowercased.

        Regression test for #1371. Hub API is already case insensitive. Somehow the issue was with the `requests`
        streaming implementation when generating the ndjson payload "on the fly". It seems that the server was
        receiving only the first line which causes a confusing "400 Bad Request - Add a line with the key `lfsFile`,
        `file` or `deletedFile`". Passing raw bytes instead of a generator fixes the problem.

        See https://github.com/huggingface/huggingface_hub/issues/1371.
        """
        REPO_NAME = repo_name("CaSe_Is_ImPoRtAnT")
        repo_id = self._api.create_repo(repo_id=REPO_NAME, exist_ok=False).repo_id

        self._api.create_commit(
            repo_id=repo_id.lower(),  # API is case-insensitive!
            commit_message="Add 1 regular and 1 LFs files.",
            operations=[
                CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
                CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"LFS content"),
            ],
        )
        repo_files = self._api.list_repo_files(repo_id=repo_id)
        self.assertIn("file.txt", repo_files)
        self.assertIn("lfs.bin", repo_files)

    @use_tmp_repo()
    def test_commit_copy_file(self, repo_url: RepoUrl) -> None:
        """Test CommitOperationCopy.

        Works only when copying an LFS file.
        """
        repo_id = repo_url.repo_id

        self._api.upload_file(path_or_fileobj=b"content", repo_id=repo_id, path_in_repo="file.txt")
        self._api.upload_file(path_or_fileobj=b"LFS content", repo_id=repo_id, path_in_repo="lfs.bin")

        self._api.create_commit(
            repo_id=repo_id,
            commit_message="Copy LFS file.",
            operations=[
                CommitOperationCopy(src_path_in_repo="lfs.bin", path_in_repo="lfs Copy.bin"),
                CommitOperationCopy(src_path_in_repo="lfs.bin", path_in_repo="lfs Copy (1).bin"),
            ],
        )
        self._api.create_commit(
            repo_id=repo_id,
            commit_message="Copy regular file.",
            operations=[CommitOperationCopy(src_path_in_repo="file.txt", path_in_repo="file Copy.txt")],
        )
        with self.assertRaises(EntryNotFoundError):
            self._api.create_commit(
                repo_id=repo_id,
                commit_message="Copy a file that doesn't exist.",
                operations=[
                    CommitOperationCopy(src_path_in_repo="doesnt-exist.txt", path_in_repo="doesnt-exist Copy.txt")
                ],
            )

        # Check repo files
        repo_files = self._api.list_repo_files(repo_id=repo_id)
        self.assertIn("file.txt", repo_files)
        self.assertIn("file Copy.txt", repo_files)
        self.assertIn("lfs.bin", repo_files)
        self.assertIn("lfs Copy.bin", repo_files)
        self.assertIn("lfs Copy (1).bin", repo_files)

        # Check same LFS file
        repo_file1, repo_file2 = self._api.get_paths_info(repo_id=repo_id, paths=["lfs.bin", "lfs Copy.bin"])
        self.assertEqual(repo_file1.lfs["sha256"], repo_file2.lfs["sha256"])

    @use_tmp_repo()
    def test_create_commit_mutates_operations(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id

        operations = [
            CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"content"),
            CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
        ]
        self._api.create_commit(
            repo_id=repo_id,
            commit_message="Copy LFS file.",
            operations=operations,
        )

        self.assertTrue(operations[0]._is_committed)
        self.assertTrue(operations[0]._is_uploaded)  # LFS file
        self.assertEqual(operations[0].path_or_fileobj, b"content")  # not removed by default
        self.assertTrue(operations[1]._is_committed)
        self.assertEqual(operations[1].path_or_fileobj, b"content")

    @use_tmp_repo()
    def test_pre_upload_before_commit(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id

        operations = [
            CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"content1"),
            CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
            CommitOperationAdd(path_in_repo="lfs2.bin", path_or_fileobj=b"content2"),
            CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
        ]

        # First: preupload 1 by 1
        for operation in operations:
            self._api.preupload_lfs_files(repo_id, [operation])
        self.assertTrue(operations[0]._is_uploaded)
        self.assertEqual(operations[0].path_or_fileobj, b"")  # Freed memory
        self.assertTrue(operations[2]._is_uploaded)
        self.assertEqual(operations[2].path_or_fileobj, b"")  # Freed memory

        # create commit and capture debug logs
        with self.assertLogs("huggingface_hub", level="DEBUG") as debug_logs:
            self._api.create_commit(
                repo_id=repo_id,
                commit_message="Copy LFS file.",
                operations=operations,
            )

        # No LFS files uploaded during commit
        self.assertTrue(any("No LFS files to upload." in log for log in debug_logs.output))

    @use_tmp_repo()
    def test_commit_modelcard_invalid_metadata(self, repo_url: RepoUrl) -> None:
        with patch.object(self._api, "preupload_lfs_files") as mock:
            with self.assertRaisesRegex(ValueError, "Invalid metadata in README.md"):
                self._api.create_commit(
                    repo_id=repo_url.repo_id,
                    operations=[
                        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=INVALID_MODELCARD.encode()),
                        CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
                        CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"content"),
                    ],
                    commit_message="Test commit",
                )

        # Failed early => no LFS files uploaded
        mock.assert_not_called()

    @use_tmp_repo()
    def test_commit_modelcard_empty_metadata(self, repo_url: RepoUrl) -> None:
        modelcard = "This is a modelcard without metadata"
        with self.assertWarnsRegex(UserWarning, "Warnings while validating metadata in README.md"):
            commit = self._api.create_commit(
                repo_id=repo_url.repo_id,
                operations=[
                    CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=modelcard.encode()),
                    CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
                    CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"content"),
                ],
                commit_message="Test commit",
            )

        # Commit still happened correctly
        assert isinstance(commit, CommitInfo)

    def test_create_file_with_relative_path(self):
        """Creating a file with a relative path_in_repo is forbidden.

        Previously taken from a regression test for HackerOne report 1928845. The bug enabled attackers to create files
        outside of the local dir if users downloaded a file with a relative path_in_repo on Windows.

        This is not relevant anymore as the API now forbids such paths.
        """
        repo_id = self._api.create_repo(repo_id=repo_name()).repo_id
        with self.assertRaises(HfHubHTTPError) as cm:
            self._api.upload_file(path_or_fileobj=b"content", path_in_repo="..\\ddd", repo_id=repo_id)
        assert cm.exception.response.status_code == 422


class HfApiUploadEmptyFileTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create repo for all tests as they are not dependent on each other.
        cls.repo_id = f"{USER}/{repo_name('upload_empty_file')}"
        cls._api.create_repo(repo_id=cls.repo_id, exist_ok=False)

    @classmethod
    def tearDownClass(cls):
        cls._api.delete_repo(repo_id=cls.repo_id)
        super().tearDownClass()

    def test_upload_empty_lfs_file(self) -> None:
        # Should have been an LFS file, but uploaded as regular (would fail otherwise)
        self._api.upload_file(repo_id=self.repo_id, path_in_repo="empty.pkl", path_or_fileobj=b"")
        info = self._api.repo_info(repo_id=self.repo_id, files_metadata=True)

        repo_file = {file.rfilename: file for file in info.siblings}["empty.pkl"]
        self.assertEqual(repo_file.size, 0)
        self.assertIsNone(repo_file.lfs)  # As regular


class HfApiDeleteFolderTest(HfApiCommonTest):
    def setUp(self):
        self.repo_id = f"{USER}/{repo_name('create_commit_delete_folder')}"
        self._api.create_repo(repo_id=self.repo_id, exist_ok=False)

        self._api.create_commit(
            repo_id=self.repo_id,
            commit_message="Init repo",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_1.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_2.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="2/file_3.md"),
            ],
        )

    def tearDown(self):
        self._api.delete_repo(repo_id=self.repo_id)

    def test_create_commit_delete_folder_implicit(self):
        self._api.create_commit(
            operations=[CommitOperationDelete(path_in_repo="1/")],
            commit_message="Test delete folder implicit",
            repo_id=self.repo_id,
        )

        with self.assertRaises(EntryNotFoundError):
            hf_hub_download(self.repo_id, "1/file_1.md", use_auth_token=self._token)

        with self.assertRaises(EntryNotFoundError):
            hf_hub_download(self.repo_id, "1/file_2.md", use_auth_token=self._token)

        # Still exists
        hf_hub_download(self.repo_id, "2/file_3.md", use_auth_token=self._token)

    def test_create_commit_delete_folder_explicit(self):
        self._api.delete_folder(path_in_repo="1", repo_id=self.repo_id)
        with self.assertRaises(EntryNotFoundError):
            hf_hub_download(self.repo_id, "1/file_1.md", use_auth_token=self._token)

    def test_create_commit_implicit_delete_folder_is_ok(self):
        self._api.create_commit(
            operations=[CommitOperationDelete(path_in_repo="1")],
            commit_message="Failing delete folder",
            repo_id=self.repo_id,
        )


class HfApiListFilesInfoTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_id = cls._api.create_repo(repo_id=repo_name()).repo_id

        cls._api.create_commit(
            repo_id=cls.repo_id,
            commit_message="A first repo",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="file.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_1.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/2/file_1_2.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="2/file_2.md"),
            ],
        )

        cls._api.create_commit(
            repo_id=cls.repo_id,
            commit_message="Another commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"data2", path_in_repo="3/file_3.md"),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls._api.delete_repo(repo_id=cls.repo_id)


class HfApiListRepoTreeTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_id = cls._api.create_repo(repo_id=repo_name()).repo_id

        cls._api.create_commit(
            repo_id=cls.repo_id,
            commit_message="A first repo",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="file.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_1.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/2/file_1_2.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="2/file_2.md"),
            ],
        )

        cls._api.create_commit(
            repo_id=cls.repo_id,
            commit_message="Another commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"data2", path_in_repo="3/file_3.md"),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls._api.delete_repo(repo_id=cls.repo_id)

    def test_list_tree(self):
        tree = list(self._api.list_repo_tree(repo_id=self.repo_id))
        self.assertEqual(len(tree), 6)
        self.assertEqual({tree_obj.path for tree_obj in tree}, {"file.md", "lfs.bin", "1", "2", "3", ".gitattributes"})

        tree = list(self._api.list_repo_tree(repo_id=self.repo_id, path_in_repo="1"))
        self.assertEqual(len(tree), 2)
        self.assertEqual({tree_obj.path for tree_obj in tree}, {"1/file_1.md", "1/2"})

    def test_list_tree_recursively(self):
        tree = list(self._api.list_repo_tree(repo_id=self.repo_id, recursive=True))
        self.assertEqual(len(tree), 11)
        self.assertEqual(
            {tree_obj.path for tree_obj in tree},
            {
                "file.md",
                "lfs.bin",
                "1/file_1.md",
                "1/2/file_1_2.md",
                "2/file_2.md",
                "3/file_3.md",
                "1",
                "2",
                "1/2",
                "3",
                ".gitattributes",
            },
        )

    def test_list_unknown_tree(self):
        with self.assertRaises(EntryNotFoundError):
            list(self._api.list_repo_tree(repo_id=self.repo_id, path_in_repo="unknown"))

    def test_list_with_empty_path(self):
        self.assertEqual(
            set(tree_obj.path for tree_obj in self._api.list_repo_tree(repo_id=self.repo_id, path_in_repo="")),
            set(tree_obj.path for tree_obj in self._api.list_repo_tree(repo_id=self.repo_id)),
        )

    @with_production_testing
    def test_list_tree_with_expand(self):
        tree = list(
            HfApi().list_repo_tree(
                repo_id="prompthero/openjourney-v4",
                expand=True,
                revision="c9211c53404dd6f4cfac5f04f33535892260668e",
            )
        )
        self.assertEqual(len(tree), 11)

        # check last_commit and security are present for a file
        model_ckpt = next(tree_obj for tree_obj in tree if tree_obj.path == "openjourney-v4.ckpt")
        self.assertIsNotNone(model_ckpt.last_commit)
        self.assertEqual(model_ckpt.last_commit["oid"], "bda967fdb79a50844e4a02cccae3217a8ecc86cd")
        self.assertIsNotNone(model_ckpt.security)
        self.assertTrue(model_ckpt.security["safe"])
        self.assertTrue(isinstance(model_ckpt.security["av_scan"], dict))  # all details in here

        # check last_commit is present for a folder
        feature_extractor = next(tree_obj for tree_obj in tree if tree_obj.path == "feature_extractor")
        self.assertIsNotNone(feature_extractor.last_commit)
        self.assertEqual(feature_extractor.last_commit["oid"], "47b62b20b20e06b9de610e840282b7e6c3d51190")

    @with_production_testing
    def test_list_files_without_expand(self):
        tree = list(
            HfApi().list_repo_tree(
                repo_id="prompthero/openjourney-v4",
                revision="c9211c53404dd6f4cfac5f04f33535892260668e",
            )
        )
        self.assertEqual(len(tree), 11)

        # check last_commit and security are missing for a file
        model_ckpt = next(tree_obj for tree_obj in tree if tree_obj.path == "openjourney-v4.ckpt")
        self.assertIsNone(model_ckpt.last_commit)
        self.assertIsNone(model_ckpt.security)

        # check last_commit is missing for a folder
        feature_extractor = next(tree_obj for tree_obj in tree if tree_obj.path == "feature_extractor")
        self.assertIsNone(feature_extractor.last_commit)


class HfApiTagEndpointTest(HfApiCommonTest):
    @use_tmp_repo("model")
    def test_create_tag_on_main(self, repo_url: RepoUrl) -> None:
        """Check `create_tag` on default main branch works."""
        self._api.create_tag(repo_url.repo_id, tag="v0", tag_message="This is a tag message.")

        # Check tag  is on `main`
        tag_info = self._api.model_info(repo_url.repo_id, revision="v0")
        main_info = self._api.model_info(repo_url.repo_id, revision="main")
        self.assertEqual(tag_info.sha, main_info.sha)

    @use_tmp_repo("model")
    def test_create_tag_on_pr(self, repo_url: RepoUrl) -> None:
        """Check `create_tag` on a PR ref works."""
        # Create a PR with a readme
        commit_info: CommitInfo = self._api.create_commit(
            repo_id=repo_url.repo_id,
            create_pr=True,
            commit_message="upload readme",
            operations=[CommitOperationAdd(path_or_fileobj=b"this is a file content", path_in_repo="readme.md")],
        )

        # Tag the PR
        self._api.create_tag(repo_url.repo_id, tag="v0", revision=commit_info.pr_revision)

        # Check tag  is on `refs/pr/1`
        tag_info = self._api.model_info(repo_url.repo_id, revision="v0")
        pr_info = self._api.model_info(repo_url.repo_id, revision=commit_info.pr_revision)
        main_info = self._api.model_info(repo_url.repo_id)

        self.assertEqual(tag_info.sha, pr_info.sha)
        self.assertNotEqual(tag_info.sha, main_info.sha)

    @use_tmp_repo("dataset")
    def test_create_tag_on_commit_oid(self, repo_url: RepoUrl) -> None:
        """Check `create_tag` on specific commit oid works (both long and shorthands).

        Test it on a `dataset` repo.
        """
        # Create a PR with a readme
        commit_info_1: CommitInfo = self._api.create_commit(
            repo_id=repo_url.repo_id,
            repo_type="dataset",
            commit_message="upload readme",
            operations=[CommitOperationAdd(path_or_fileobj=b"this is a file content", path_in_repo="readme.md")],
        )
        commit_info_2: CommitInfo = self._api.create_commit(
            repo_id=repo_url.repo_id,
            repo_type="dataset",
            commit_message="upload config",
            operations=[CommitOperationAdd(path_or_fileobj=b"{'hello': 'world'}", path_in_repo="config.json")],
        )

        # Tag commits
        self._api.create_tag(
            repo_url.repo_id,
            tag="commit_1",
            repo_type="dataset",
            revision=commit_info_1.oid,  # long version
        )
        self._api.create_tag(
            repo_url.repo_id,
            tag="commit_2",
            repo_type="dataset",
            revision=commit_info_2.oid[:7],  # use shorthand !
        )

        # Check tags
        tag_1_info = self._api.dataset_info(repo_url.repo_id, revision="commit_1")
        tag_2_info = self._api.dataset_info(repo_url.repo_id, revision="commit_2")

        self.assertEqual(tag_1_info.sha, commit_info_1.oid)
        self.assertEqual(tag_2_info.sha, commit_info_2.oid)

    @use_tmp_repo("model")
    def test_invalid_tag_name(self, repo_url: RepoUrl) -> None:
        """Check `create_tag` with an invalid tag name."""
        with self.assertRaises(HTTPError):
            self._api.create_tag(repo_url.repo_id, tag="invalid tag")

    @use_tmp_repo("model")
    def test_create_tag_on_missing_revision(self, repo_url: RepoUrl) -> None:
        """Check `create_tag` on a missing revision."""
        with self.assertRaises(RevisionNotFoundError):
            self._api.create_tag(repo_url.repo_id, tag="invalid tag", revision="foobar")

    @use_tmp_repo("model")
    def test_create_tag_twice(self, repo_url: RepoUrl) -> None:
        """Check `create_tag` called twice on same tag should fail with HTTP 409."""
        self._api.create_tag(repo_url.repo_id, tag="tag_1")
        with self.assertRaises(HfHubHTTPError) as err:
            self._api.create_tag(repo_url.repo_id, tag="tag_1")
        self.assertEqual(err.exception.response.status_code, 409)

        # exist_ok=True => doesn't fail
        self._api.create_tag(repo_url.repo_id, tag="tag_1", exist_ok=True)

    @use_tmp_repo("model")
    def test_create_and_delete_tag(self, repo_url: RepoUrl) -> None:
        """Check `delete_tag` deletes the tag."""
        self._api.create_tag(repo_url.repo_id, tag="v0")
        self._api.model_info(repo_url.repo_id, revision="v0")

        self._api.delete_tag(repo_url.repo_id, tag="v0")
        with self.assertRaises(RevisionNotFoundError):
            self._api.model_info(repo_url.repo_id, revision="v0")

    @use_tmp_repo("model")
    def test_delete_tag_missing_tag(self, repo_url: RepoUrl) -> None:
        """Check cannot `delete_tag` if tag doesn't exist."""
        with self.assertRaises(RevisionNotFoundError):
            self._api.delete_tag(repo_url.repo_id, tag="v0")

    @use_tmp_repo("model")
    def test_delete_tag_with_branch_name(self, repo_url: RepoUrl) -> None:
        """Try to `delete_tag` if tag is a branch name.

        Currently getting a HTTP 500.
        See https://github.com/huggingface/moon-landing/issues/4223.
        """
        with self.assertRaises(HfHubHTTPError):
            self._api.delete_tag(repo_url.repo_id, tag="main")


class HfApiBranchEndpointTest(HfApiCommonTest):
    @use_tmp_repo()
    def test_create_and_delete_branch(self, repo_url: RepoUrl) -> None:
        """Test `create_branch` from main branch."""
        self._api.create_branch(repo_url.repo_id, branch="cool-branch")

        # Check `cool-branch` branch exists
        self._api.model_info(repo_url.repo_id, revision="cool-branch")

        # Delete it
        self._api.delete_branch(repo_url.repo_id, branch="cool-branch")

        # Check doesn't exist anymore
        with self.assertRaises(RevisionNotFoundError):
            self._api.model_info(repo_url.repo_id, revision="cool-branch")

    @use_tmp_repo()
    def test_create_branch_existing_branch_fails(self, repo_url: RepoUrl) -> None:
        """Test `create_branch` on existing branch."""
        self._api.create_branch(repo_url.repo_id, branch="cool-branch")

        with self.assertRaisesRegex(HfHubHTTPError, "Reference already exists"):
            self._api.create_branch(repo_url.repo_id, branch="cool-branch")

        with self.assertRaisesRegex(HfHubHTTPError, "Reference already exists"):
            self._api.create_branch(repo_url.repo_id, branch="main")

        # exist_ok=True => doesn't fail
        self._api.create_branch(repo_url.repo_id, branch="cool-branch", exist_ok=True)
        self._api.create_branch(repo_url.repo_id, branch="main", exist_ok=True)

    @use_tmp_repo()
    def test_create_branch_existing_tag_does_not_fail(self, repo_url: RepoUrl) -> None:
        """Test `create_branch` on existing tag."""
        self._api.create_tag(repo_url.repo_id, tag="tag")
        self._api.create_branch(repo_url.repo_id, branch="tag")

    @unittest.skip(
        "Test user is flagged as isHF which gives permissions to create invalid references."
        "Not relevant to test it anyway (i.e. it's more a server-side test)."
    )
    @use_tmp_repo()
    def test_create_branch_forbidden_ref_branch_fails(self, repo_url: RepoUrl) -> None:
        """Test `create_branch` on forbidden ref branch."""
        with self.assertRaisesRegex(BadRequestError, "Invalid reference for a branch"):
            self._api.create_branch(repo_url.repo_id, branch="refs/pr/5")

        with self.assertRaisesRegex(BadRequestError, "Invalid reference for a branch"):
            self._api.create_branch(repo_url.repo_id, branch="refs/something/random")

    @use_tmp_repo()
    def test_delete_branch_on_protected_branch_fails(self, repo_url: RepoUrl) -> None:
        """Test `delete_branch` fails on protected branch."""
        with self.assertRaisesRegex(HfHubHTTPError, "Cannot delete refs/heads/main"):
            self._api.delete_branch(repo_url.repo_id, branch="main")

    @use_tmp_repo()
    def test_delete_branch_on_missing_branch_fails(self, repo_url: RepoUrl) -> None:
        """Test `delete_branch` fails on missing branch."""
        with self.assertRaisesRegex(HfHubHTTPError, "Invalid rev id"):
            self._api.delete_branch(repo_url.repo_id, branch="cool-branch")

        # Using a tag instead of branch -> fails
        self._api.create_tag(repo_url.repo_id, tag="cool-tag")
        with self.assertRaisesRegex(HfHubHTTPError, "Invalid rev id"):
            self._api.delete_branch(repo_url.repo_id, branch="cool-tag")

    @use_tmp_repo()
    def test_create_branch_from_revision(self, repo_url: RepoUrl) -> None:
        """Test `create_branch` from a different revision than main HEAD."""
        # Create commit and remember initial/latest commit
        initial_commit = self._api.model_info(repo_url.repo_id).sha
        commit = self._api.create_commit(
            repo_url.repo_id,
            operations=[CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=b"content")],
            commit_message="test commit",
        )
        latest_commit = commit.oid

        # Create branches
        self._api.create_branch(repo_url.repo_id, branch="from-head")
        self._api.create_branch(repo_url.repo_id, branch="from-initial", revision=initial_commit)
        self._api.create_branch(repo_url.repo_id, branch="from-branch", revision="from-initial")
        time.sleep(0.2)  # hack: wait for server to update cache?

        # Checks branches start from expected commits
        self.assertEqual(
            {
                "main": latest_commit,
                "from-head": latest_commit,
                "from-initial": initial_commit,
                "from-branch": initial_commit,
            },
            {ref.name: ref.target_commit for ref in self._api.list_repo_refs(repo_id=repo_url.repo_id).branches},
        )


class HfApiDeleteFilesTest(HfApiCommonTest):
    def setUp(self) -> None:
        super().setUp()

        self.repo_id = self._api.create_repo(repo_id=repo_name()).repo_id
        self._api.create_commit(
            repo_id=self.repo_id,
            operations=[
                # Regular files
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="file.txt"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="nested/file.txt"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="nested/sub/file.txt"),
                # LFS files
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="nested/lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="nested/sub/lfs.bin"),
            ],
            commit_message="Init repo structure",
        )

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=self.repo_id)
        super().tearDown()

    def remote_files(self) -> Set[set]:
        return set(self._api.list_repo_files(repo_id=self.repo_id))

    def test_delete_single_file(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["file.txt"])
        assert "file.txt" not in self.remote_files()

    def test_delete_multiple_files(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["file.txt", "lfs.bin"])
        files = self.remote_files()
        assert "file.txt" not in files
        assert "lfs.bin" not in files

    def test_delete_folder_with_pattern(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["nested/*"])
        assert self.remote_files() == {".gitattributes", "file.txt", "lfs.bin"}

    def test_delete_folder_without_pattern(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["nested/"])
        assert self.remote_files() == {".gitattributes", "file.txt", "lfs.bin"}

    def test_unknown_path_do_not_raise(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["not_existing", "nested/*"])
        assert self.remote_files() == {".gitattributes", "file.txt", "lfs.bin"}

    def test_delete_bin_files_with_patterns(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["*.bin"])
        files = self.remote_files()
        assert "lfs.bin" not in files
        assert "nested/lfs.bin" not in files
        assert "nested/sub/lfs.bin" not in files

    def test_delete_files_in_folders_with_patterns(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["*/file.txt"])
        files = self.remote_files()
        assert "file.txt" in files
        assert "nested/file.txt" not in files
        assert "nested/sub/file.txt" not in files

    def test_delete_all_files(self):
        self._api.delete_files(repo_id=self.repo_id, delete_patterns=["*"])
        assert self.remote_files() == {".gitattributes"}


class HfApiPublicStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._api = HfApi()

    def test_staging_list_datasets(self):
        self._api.list_datasets()

    def test_staging_list_models(self):
        self._api.list_models()

    def test_staging_list_metrics(self):
        self._api.list_metrics()


class HfApiPublicProductionTest(unittest.TestCase):
    @with_production_testing
    def setUp(self) -> None:
        self._api = HfApi()

    def test_list_models(self):
        models = list(self._api.list_models(limit=500))
        self.assertGreater(len(models), 100)
        self.assertIsInstance(models[0], ModelInfo)

    def test_list_models_author(self):
        models = list(self._api.list_models(author="google"))
        self.assertGreater(len(models), 10)
        self.assertIsInstance(models[0], ModelInfo)
        for model in models:
            self.assertTrue(model.modelId.startswith("google/"))

    def test_list_models_search(self):
        models = list(self._api.list_models(search="bert"))
        self.assertGreater(len(models), 10)
        self.assertIsInstance(models[0], ModelInfo)
        for model in models[:10]:
            # Rough rule: at least first 10 will have "bert" in the name
            # Not optimal since it is dependent on how the Hub implements the search
            # (and changes it in the future) but for now it should do the trick.
            self.assertTrue("bert" in model.modelId.lower())

    def test_list_models_complex_query(self):
        # Let's list the 10 most recent models
        # with tags "bert" and "jax",
        # ordered by last modified date.
        models = list(self._api.list_models(filter=("bert", "jax"), sort="last_modified", direction=-1, limit=10))
        # we have at least 1 models
        self.assertGreater(len(models), 1)
        self.assertLessEqual(len(models), 10)
        model = models[0]
        self.assertIsInstance(model, ModelInfo)
        self.assertTrue(all(tag in model.tags for tag in ["bert", "jax"]))

    def test_list_models_with_config(self):
        for model in self._api.list_models(filter="adapter-transformers", fetch_config=True, limit=20):
            self.assertIsNotNone(model.config)

    def test_list_models_without_config(self):
        for model in self._api.list_models(filter="adapter-transformers", fetch_config=False, limit=20):
            self.assertIsNone(model.config)

    def test_model_info(self):
        model = self._api.model_info(repo_id=DUMMY_MODEL_ID)
        self.assertIsInstance(model, ModelInfo)
        self.assertNotEqual(model.sha, DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)
        self.assertEqual(model.created_at, datetime.datetime(2022, 3, 2, 23, 29, 5, tzinfo=datetime.timezone.utc))

        # One particular commit (not the top of `main`)
        model = self._api.model_info(repo_id=DUMMY_MODEL_ID, revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)
        self.assertIsInstance(model, ModelInfo)
        self.assertEqual(model.sha, DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)

    # TODO; un-skip this test once it's fixed.
    @unittest.skip(
        "Security status is currently unreliable on the server endpoint, so this"
        " test occasionally fails. Issue is tracked in"
        " https://github.com/huggingface/huggingface_hub/issues/1002 and"
        " https://github.com/huggingface/moon-landing/issues/3695. TODO: un-skip"
        " this test once it's fixed."
    )
    def test_model_info_with_security(self):
        model = self._api.model_info(
            repo_id=DUMMY_MODEL_ID,
            revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
            securityStatus=True,
        )
        self.assertEqual(model.securityStatus, {"containsInfected": False})

    def test_model_info_with_file_metadata(self):
        model = self._api.model_info(
            repo_id=DUMMY_MODEL_ID,
            revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
            files_metadata=True,
        )
        files = model.siblings
        assert files is not None
        self._check_siblings_metadata(files)

    def test_model_info_corrupted_model_index(self) -> None:
        """Loading model info from a model with corrupted data card should still work.

        Here we use a model with a "model-index" that is not an array. Git hook should prevent this from happening
        on the server, but models uploaded before we implemented the check might have this issue.

        Example data from https://huggingface.co/Waynehillsdev/Waynehills-STT-doogie-server.
        """
        with self.assertLogs("huggingface_hub", level="WARNING") as warning_logs:
            model = ModelInfo(
                **{
                    "_id": "621ffdc036468d709f1751d8",
                    "id": "Waynehillsdev/Waynehills-STT-doogie-server",
                    "cardData": {
                        "license": "apache-2.0",
                        "tags": ["generated_from_trainer"],
                        "model-index": {"name": "Waynehills-STT-doogie-server"},
                    },
                    "gitalyUid": "53c57f29a007fc728c968127061b7b740dcf2b1ad401d907f703b27658559413",
                    "likes": 0,
                    "private": False,
                    "config": {"architectures": ["Wav2Vec2ForCTC"], "model_type": "wav2vec2"},
                    "downloads": 1,
                    "tags": [
                        "transformers",
                        "pytorch",
                        "tensorboard",
                        "wav2vec2",
                        "automatic-speech-recognition",
                        "generated_from_trainer",
                        "license:apache-2.0",
                        "endpoints_compatible",
                        "region:us",
                    ],
                    "pipeline_tag": "automatic-speech-recognition",
                    "createdAt": "2022-03-02T23:29:04.000Z",
                    "modelId": "Waynehillsdev/Waynehills-STT-doogie-server",
                    "siblings": None,
                }
            )
            self.assertIsNone(model.card_data.eval_results)
        self.assertTrue(any("Invalid model-index" in log for log in warning_logs.output))

    def test_model_info_with_widget_data(self):
        info = self._api.model_info("HuggingFaceH4/zephyr-7b-beta")
        assert info.widget_data is not None

    def test_list_repo_files(self):
        files = self._api.list_repo_files(repo_id=DUMMY_MODEL_ID)
        expected_files = [
            ".gitattributes",
            "README.md",
            "config.json",
            "flax_model.msgpack",
            "merges.txt",
            "pytorch_model.bin",
            "tf_model.h5",
            "vocab.json",
        ]
        self.assertListEqual(files, expected_files)

    def test_list_datasets_no_filter(self):
        datasets = list(self._api.list_datasets(limit=500))
        self.assertGreater(len(datasets), 100)
        self.assertIsInstance(datasets[0], DatasetInfo)

    def test_filter_datasets_by_author_and_name(self):
        datasets = list(self._api.list_datasets(author="huggingface", dataset_name="DataMeasurementsFiles"))
        assert len(datasets) > 0
        assert "huggingface" in datasets[0].author
        assert "DataMeasurementsFiles" in datasets[0].id

    def test_filter_datasets_by_benchmark(self):
        datasets = list(self._api.list_datasets(benchmark="raft"))
        assert len(datasets) > 0
        assert "benchmark:raft" in datasets[0].tags

    def test_filter_datasets_by_language_creator(self):
        datasets = list(self._api.list_datasets(language_creators="crowdsourced"))
        assert len(datasets) > 0
        assert "language_creators:crowdsourced" in datasets[0].tags

    def test_filter_datasets_by_language_only(self):
        datasets = list(self._api.list_datasets(language="en", limit=10))
        assert len(datasets) > 0
        assert "language:en" in datasets[0].tags

        datasets = list(self._api.list_datasets(language=("en", "fr"), limit=10))
        assert len(datasets) > 0
        assert "language:en" in datasets[0].tags
        assert "language:fr" in datasets[0].tags

    def test_filter_datasets_by_multilinguality(self):
        datasets = list(self._api.list_datasets(multilinguality="multilingual", limit=10))
        assert len(datasets) > 0
        assert "multilinguality:multilingual" in datasets[0].tags

    def test_filter_datasets_by_size_categories(self):
        datasets = list(self._api.list_datasets(size_categories="100K<n<1M", limit=10))
        assert len(datasets) > 0
        assert "size_categories:100K<n<1M" in datasets[0].tags

    def test_filter_datasets_by_task_categories(self):
        datasets = list(self._api.list_datasets(task_categories="audio-classification", limit=10))
        assert len(datasets) > 0
        assert "task_categories:audio-classification" in datasets[0].tags

    def test_filter_datasets_by_task_ids(self):
        datasets = list(self._api.list_datasets(task_ids="natural-language-inference", limit=10))
        assert len(datasets) > 0
        assert "task_ids:natural-language-inference" in datasets[0].tags

    def test_list_datasets_full(self):
        datasets = list(self._api.list_datasets(full=True, limit=500))
        assert len(datasets) > 100
        assert isinstance(datasets[0], DatasetInfo)
        assert any(dataset.card_data for dataset in datasets)

    def test_list_datasets_author(self):
        datasets = list(self._api.list_datasets(author="huggingface", limit=10))
        assert len(datasets) > 0
        assert datasets[0].author == "huggingface"

    def test_list_datasets_search(self):
        datasets = list(self._api.list_datasets(search="wikipedia", limit=10))
        assert len(datasets) > 5
        for dataset in datasets:
            assert "wikipedia" in dataset.id.lower()

    def test_filter_datasets_with_card_data(self):
        assert any(dataset.card_data is not None for dataset in self._api.list_datasets(full=True, limit=50))
        assert all(dataset.card_data is None for dataset in self._api.list_datasets(full=False, limit=50))

    def test_filter_datasets_by_tag(self):
        for dataset in self._api.list_datasets(tags="fiftyone", limit=5):
            assert "fiftyone" in dataset.tags

    def test_dataset_info(self):
        dataset = self._api.dataset_info(repo_id=DUMMY_DATASET_ID)
        self.assertTrue(isinstance(dataset.card_data, DatasetCardData) and len(dataset.card_data) > 0)
        self.assertTrue(isinstance(dataset.siblings, list) and len(dataset.siblings) > 0)
        self.assertIsInstance(dataset, DatasetInfo)
        self.assertNotEqual(dataset.sha, DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT)
        dataset = self._api.dataset_info(
            repo_id=DUMMY_DATASET_ID,
            revision=DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT,
        )
        self.assertIsInstance(dataset, DatasetInfo)
        self.assertEqual(dataset.sha, DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT)

    def test_dataset_info_with_file_metadata(self):
        dataset = self._api.dataset_info(
            repo_id=SAMPLE_DATASET_IDENTIFIER,
            files_metadata=True,
        )
        files = dataset.siblings
        assert files is not None
        self._check_siblings_metadata(files)

    def _check_siblings_metadata(self, files: List[RepoSibling]):
        """Check requested metadata has been received from the server."""
        at_least_one_lfs = False
        for file in files:
            self.assertTrue(isinstance(file.blob_id, str))
            self.assertTrue(isinstance(file.size, int))
            if file.lfs is not None:
                at_least_one_lfs = True
                self.assertTrue(isinstance(file.lfs, dict))
                self.assertTrue("sha256" in file.lfs)
        self.assertTrue(at_least_one_lfs)

    def test_space_info(self) -> None:
        space = self._api.space_info(repo_id="HuggingFaceH4/zephyr-chat")
        assert space.id == "HuggingFaceH4/zephyr-chat"
        assert space.author == "HuggingFaceH4"
        assert isinstance(space.runtime, SpaceRuntime)

    def test_list_metrics(self):
        metrics = self._api.list_metrics()
        self.assertGreater(len(metrics), 10)
        self.assertIsInstance(metrics[0], MetricInfo)
        self.assertTrue(any(metric.description for metric in metrics))

    def test_filter_models_by_author(self):
        models = list(self._api.list_models(author="muellerzr"))
        assert len(models) > 0
        assert "muellerzr" in models[0].modelId

    def test_filter_models_by_author_and_name(self):
        # Test we can search by an author and a name, but the model is not found
        models = list(self._api.list_models(author="facebook", model_name="bart-base"))
        assert "facebook/bart-base" in models[0].modelId

    def test_failing_filter_models_by_author_and_model_name(self):
        # Test we can search by an author and a name, but the model is not found
        models = list(self._api.list_models(author="muellerzr", model_name="testme"))
        assert len(models) == 0

    def test_filter_models_with_library(self):
        models = list(self._api.list_models(author="microsoft", model_name="wavlm-base-sd", library="tensorflow"))
        assert len(models) == 0

        models = list(self._api.list_models(author="microsoft", model_name="wavlm-base-sd", library="pytorch"))
        assert len(models) > 0

    def test_filter_models_with_task(self):
        models = list(self._api.list_models(task="fill-mask", model_name="albert-base-v2"))
        assert models[0].pipeline_tag == "fill-mask"
        assert "albert" in models[0].modelId
        assert "base" in models[0].modelId
        assert "v2" in models[0].modelId

        models = list(self._api.list_models(task="dummytask"))
        assert len(models) == 0

    def test_filter_models_by_language(self):
        for language in ["en", "fr", "zh"]:
            for model in self._api.list_models(language=language, limit=5):
                assert language in model.tags

    def test_filter_models_with_tag(self):
        models = list(self._api.list_models(author="HuggingFaceBR4", tags=["tensorboard"]))
        assert models[0].id.startswith("HuggingFaceBR4/")
        assert "tensorboard" in models[0].tags

        models = list(self._api.list_models(tags="dummytag"))
        assert len(models) == 0

    def test_filter_models_with_card_data(self):
        models = self._api.list_models(filter="co2_eq_emissions", cardData=True)
        assert any(model.card_data is not None for model in models)

        models = self._api.list_models(filter="co2_eq_emissions")
        assert all(model.card_data is None for model in models)

    def test_is_emission_within_threshold(self):
        # tests that dictionary is handled correctly as "emissions" and that
        # 17g is accepted and parsed correctly as a value
        # regression test for #753
        kwargs = {field.name: None for field in fields(ModelInfo) if field.init}
        kwargs = {**kwargs, "card_data": ModelCardData(co2_eq_emissions={"emissions": "17g"})}
        model = ModelInfo(**kwargs)
        assert _is_emission_within_threshold(model, -1, 100)

    def test_filter_emissions_with_max(self):
        assert all(
            model.card_data["co2_eq_emissions"] <= 100
            for model in self._api.list_models(emissions_thresholds=(None, 100), cardData=True, limit=1000)
            if isinstance(model.card_data["co2_eq_emissions"], (float, int))
        )

    def test_filter_emissions_with_min(self):
        assert all(
            [
                model.card_data["co2_eq_emissions"] >= 5
                for model in self._api.list_models(emissions_thresholds=(5, None), cardData=True, limit=1000)
                if isinstance(model.card_data["co2_eq_emissions"], (float, int))
            ]
        )

    def test_filter_emissions_with_min_and_max(self):
        models = list(self._api.list_models(emissions_thresholds=(5, 100), cardData=True, limit=1000))
        assert all(
            [
                model.card_data["co2_eq_emissions"] >= 5
                for model in models
                if isinstance(model.card_data["co2_eq_emissions"], (float, int))
            ]
        )

        assert all(
            [
                model.card_data["co2_eq_emissions"] <= 100
                for model in models
                if isinstance(model.card_data["co2_eq_emissions"], (float, int))
            ]
        )

    def test_list_spaces_full(self):
        spaces = list(self._api.list_spaces(full=True, limit=500))
        assert len(spaces) > 100
        space = spaces[0]
        assert isinstance(space, SpaceInfo)
        assert any(space.card_data for space in spaces)

    def test_list_spaces_author(self):
        spaces = list(self._api.list_spaces(author="julien-c"))
        assert len(spaces) > 10
        for space in spaces:
            assert space.id.startswith("julien-c/")

    def test_list_spaces_search(self):
        spaces = list(self._api.list_spaces(search="wikipedia", limit=10))
        assert "wikipedia" in spaces[0].id.lower()

    def test_list_spaces_sort_and_direction(self):
        # Descending order => first item has more likes than second
        spaces_descending_likes = list(self._api.list_spaces(sort="likes", direction=-1, limit=100))
        assert spaces_descending_likes[0].likes > spaces_descending_likes[1].likes

    def test_list_spaces_limit(self):
        spaces = list(self._api.list_spaces(limit=5))
        assert len(spaces) == 5

    def test_list_spaces_with_models(self):
        spaces = list(self._api.list_spaces(models="bert-base-uncased"))
        assert "bert-base-uncased" in spaces[0].models

    def test_list_spaces_with_datasets(self):
        spaces = list(self._api.list_spaces(datasets="wikipedia"))
        assert "wikipedia" in spaces[0].datasets

    def test_list_spaces_linked(self):
        space_id = "open-llm-leaderboard/open_llm_leaderboard"

        spaces = list(self._api.list_spaces(search=space_id))
        assert spaces[0].models is None
        assert spaces[0].datasets is None

        spaces = list(self._api.list_spaces(search=space_id, linked=True))
        assert spaces[0].models is not None
        assert spaces[0].datasets is not None

    def test_get_paths_info(self):
        paths_info = self._api.get_paths_info(
            "allenai/c4",
            ["en", "en/c4-train.00001-of-01024.json.gz", "non_existing_path"],
            expand=True,
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            repo_type="dataset",
        )
        assert len(paths_info) == 2

        assert paths_info[0].path == "en"
        assert paths_info[0].tree_id is not None
        assert paths_info[0].last_commit is not None

        assert paths_info[1].path == "en/c4-train.00001-of-01024.json.gz"
        assert paths_info[1].blob_id is not None
        assert paths_info[1].last_commit is not None
        assert paths_info[1].lfs is not None
        assert paths_info[1].security is not None
        assert paths_info[1].size > 0

    def test_get_safetensors_metadata_single_file(self) -> None:
        info = self._api.get_safetensors_metadata("bigscience/bloomz-560m")
        assert isinstance(info, SafetensorsRepoMetadata)

        assert not info.sharded
        assert info.metadata is None  # Never populated on non-sharded models
        assert len(info.files_metadata) == 1
        assert "model.safetensors" in info.files_metadata

        file_metadata = info.files_metadata["model.safetensors"]
        assert isinstance(file_metadata, SafetensorsFileMetadata)
        assert file_metadata.metadata == {"format": "pt"}
        assert len(file_metadata.tensors) == 293

        assert isinstance(info.weight_map, dict)
        assert info.weight_map["h.0.input_layernorm.bias"] == "model.safetensors"

        assert info.parameter_count == {"F16": 559214592}

    def test_get_safetensors_metadata_sharded_model(self) -> None:
        info = self._api.get_safetensors_metadata("HuggingFaceH4/zephyr-7b-beta")
        assert isinstance(info, SafetensorsRepoMetadata)

        assert info.sharded
        assert isinstance(info.metadata, dict)  # populated for sharded model
        assert len(info.files_metadata) == 8

        for file_metadata in info.files_metadata.values():
            assert isinstance(file_metadata, SafetensorsFileMetadata)

        assert info.parameter_count == {"BF16": 7241732096}

    def test_not_a_safetensors_repo(self) -> None:
        with self.assertRaises(NotASafetensorsRepoError):
            self._api.get_safetensors_metadata("huggingface-hub-ci/test_safetensors_metadata")

    def test_get_safetensors_metadata_from_revision(self) -> None:
        info = self._api.get_safetensors_metadata("huggingface-hub-ci/test_safetensors_metadata", revision="refs/pr/1")
        assert isinstance(info, SafetensorsRepoMetadata)

    def test_parse_safetensors_metadata(self) -> None:
        info = self._api.parse_safetensors_file_metadata(
            "HuggingFaceH4/zephyr-7b-beta", "model-00003-of-00008.safetensors"
        )
        assert isinstance(info, SafetensorsFileMetadata)

        assert info.metadata == {"format": "pt"}
        assert isinstance(info.tensors, dict)
        tensor = info.tensors["model.layers.10.input_layernorm.weight"]

        assert tensor == TensorInfo(dtype="BF16", shape=[4096], data_offsets=(0, 8192))

        assert tensor.parameter_count == 4096
        assert info.parameter_count == {"BF16": 989888512}

    def test_not_a_safetensors_file(self) -> None:
        with self.assertRaises(SafetensorsParsingError):
            self._api.parse_safetensors_file_metadata(
                "HuggingFaceH4/zephyr-7b-beta", "pytorch_model-00001-of-00008.bin"
            )


class HfApiPrivateTest(HfApiCommonTest):
    def setUp(self) -> None:
        super().setUp()
        self.REPO_NAME = repo_name("private")
        self._api.create_repo(repo_id=self.REPO_NAME, private=True)
        self._api.create_repo(repo_id=self.REPO_NAME, private=True, repo_type="dataset")

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=self.REPO_NAME)
        self._api.delete_repo(repo_id=self.REPO_NAME, repo_type="dataset")

    @patch("huggingface_hub.utils._headers.get_token", return_value=None)
    def test_model_info(self, mock_get_token: Mock) -> None:
        with patch.object(self._api, "token", None):  # no default token
            # Test we cannot access model info without a token
            with self.assertRaisesRegex(
                requests.exceptions.HTTPError,
                re.compile(
                    r"401 Client Error(.+)\(Request ID: .+\)(.*)Repository Not Found",
                    flags=re.DOTALL,
                ),
            ):
                _ = self._api.model_info(repo_id=f"{USER}/{self.REPO_NAME}")

            model_info = self._api.model_info(repo_id=f"{USER}/{self.REPO_NAME}", use_auth_token=self._token)
            self.assertIsInstance(model_info, ModelInfo)

    @patch("huggingface_hub.utils._headers.get_token", return_value=None)
    def test_dataset_info(self, mock_get_token: Mock) -> None:
        with patch.object(self._api, "token", None):  # no default token
            # Test we cannot access model info without a token
            with self.assertRaisesRegex(
                requests.exceptions.HTTPError,
                re.compile(
                    r"401 Client Error(.+)\(Request ID: .+\)(.*)Repository Not Found",
                    flags=re.DOTALL,
                ),
            ):
                _ = self._api.dataset_info(repo_id=f"{USER}/{self.REPO_NAME}")

            dataset_info = self._api.dataset_info(repo_id=f"{USER}/{self.REPO_NAME}", use_auth_token=self._token)
            self.assertIsInstance(dataset_info, DatasetInfo)

    def test_list_private_datasets(self):
        orig = len(list(self._api.list_datasets(use_auth_token=False)))
        new = len(list(self._api.list_datasets(use_auth_token=self._token)))
        self.assertGreater(new, orig)

    def test_list_private_models(self):
        orig = len(list(self._api.list_models(use_auth_token=False)))
        new = len(list(self._api.list_models(use_auth_token=self._token)))
        self.assertGreater(new, orig)

    @with_production_testing
    def test_list_private_spaces(self):
        orig = len(list(self._api.list_spaces(use_auth_token=False)))
        new = len(list(self._api.list_spaces(use_auth_token=self._token)))
        self.assertGreaterEqual(new, orig)


@pytest.mark.usefixtures("fx_cache_dir")
class UploadFolderMockedTest(unittest.TestCase):
    api = HfApi()
    cache_dir: Path

    def setUp(self) -> None:
        (self.cache_dir / "file.txt").write_text("content")
        (self.cache_dir / "lfs.bin").write_text("content")

        (self.cache_dir / "sub").mkdir()
        (self.cache_dir / "sub" / "file.txt").write_text("content")
        (self.cache_dir / "sub" / "lfs_in_sub.bin").write_text("content")

        (self.cache_dir / "subdir").mkdir()
        (self.cache_dir / "subdir" / "file.txt").write_text("content")
        (self.cache_dir / "subdir" / "lfs_in_subdir.bin").write_text("content")

        self.all_local_files = {
            "lfs.bin",
            "file.txt",
            "sub/file.txt",
            "sub/lfs_in_sub.bin",
            "subdir/file.txt",
            "subdir/lfs_in_subdir.bin",
        }

        self.repo_files_mock = Mock()
        self.repo_files_mock.return_value = [  # all remote files
            ".gitattributes",
            "file.txt",
            "file1.txt",
            "sub/file.txt",
            "sub/file1.txt",
            "subdir/file.txt",
            "subdir/lfs_in_subdir.bin",
        ]
        self.api.list_repo_files = self.repo_files_mock

        self.create_commit_mock = Mock()
        self.create_commit_mock.return_value.pr_url = None
        self.api.create_commit = self.create_commit_mock

    def _upload_folder_alias(self, **kwargs) -> List[Union[CommitOperationAdd, CommitOperationDelete]]:
        """Alias to call `upload_folder` + retrieve the CommitOperation list passed to `create_commit`."""
        if "folder_path" not in kwargs:
            kwargs["folder_path"] = self.cache_dir
        self.api.upload_folder(repo_id="repo_id", **kwargs)
        return self.create_commit_mock.call_args_list[0][1]["operations"]

    def test_allow_everything(self):
        operations = self._upload_folder_alias()
        self.assertTrue(all(isinstance(op, CommitOperationAdd) for op in operations))
        self.assertEqual({op.path_in_repo for op in operations}, self.all_local_files)

    def test_allow_everything_in_subdir_no_trailing_slash(self):
        operations = self._upload_folder_alias(folder_path=self.cache_dir / "subdir", path_in_repo="subdir")
        self.assertTrue(all(isinstance(op, CommitOperationAdd) for op in operations))
        self.assertEqual(
            {op.path_in_repo for op in operations},
            {"subdir/file.txt", "subdir/lfs_in_subdir.bin"},  # correct `path_in_repo`
        )

    def test_allow_everything_in_subdir_with_trailing_slash(self):
        operations = self._upload_folder_alias(folder_path=self.cache_dir / "subdir", path_in_repo="subdir/")
        self.assertTrue(all(isinstance(op, CommitOperationAdd) for op in operations))
        self.assertEqual(
            {op.path_in_repo for op in operations},
            {"subdir/file.txt", "subdir/lfs_in_subdir.bin"},  # correct `path_in_repo`
        )

    def test_allow_txt_ignore_subdir(self):
        operations = self._upload_folder_alias(allow_patterns="*.txt", ignore_patterns="subdir/*")
        self.assertTrue(all(isinstance(op, CommitOperationAdd) for op in operations))
        self.assertEqual(
            {op.path_in_repo for op in operations},
            {"sub/file.txt", "file.txt"},  # only .txt files, not in subdir
        )

    def test_allow_txt_not_root_ignore_subdir(self):
        operations = self._upload_folder_alias(allow_patterns="**/*.txt", ignore_patterns="subdir/*")
        self.assertTrue(all(isinstance(op, CommitOperationAdd) for op in operations))
        self.assertEqual(
            {op.path_in_repo for op in operations},
            {"sub/file.txt"},  # only .txt files, not in subdir, not at root
        )

    def test_path_in_repo_dot(self):
        """Regression test for #1382 when using `path_in_repo="."`.

        Using `path_in_repo="."` or `path_in_repo=None` should be equivalent.
        See https://github.com/huggingface/huggingface_hub/pull/1382.
        """
        operation_with_dot = self._upload_folder_alias(path_in_repo=".", allow_patterns=["file.txt"])[0]
        operation_with_none = self._upload_folder_alias(path_in_repo=None, allow_patterns=["file.txt"])[0]
        self.assertEqual(operation_with_dot.path_in_repo, "file.txt")
        self.assertEqual(operation_with_none.path_in_repo, "file.txt")

    def test_delete_txt(self):
        operations = self._upload_folder_alias(delete_patterns="*.txt")
        added_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}

        self.assertEqual(added_files, self.all_local_files)
        self.assertEqual(deleted_files, {"file1.txt", "sub/file1.txt"})

        # since "file.txt" and "sub/file.txt" are overwritten, no need to delete them first
        self.assertIn("file.txt", added_files)
        self.assertIn("sub/file.txt", added_files)

    def test_delete_txt_in_sub(self):
        operations = self._upload_folder_alias(
            path_in_repo="sub/", folder_path=self.cache_dir / "sub", delete_patterns="*.txt"
        )
        added_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}

        self.assertEqual(added_files, {"sub/file.txt", "sub/lfs_in_sub.bin"})  # added only in sub/
        self.assertEqual(deleted_files, {"sub/file1.txt"})  # delete only in sub/

    def test_delete_txt_in_sub_ignore_sub_file_txt(self):
        operations = self._upload_folder_alias(
            path_in_repo="sub", folder_path=self.cache_dir / "sub", ignore_patterns="file.txt", delete_patterns="*.txt"
        )
        added_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}

        # since "sub/file.txt" should be deleted and is not overwritten (ignore_patterns), we delete it explicitly
        self.assertEqual(added_files, {"sub/lfs_in_sub.bin"})  # no "sub/file.txt"
        self.assertEqual(deleted_files, {"sub/file1.txt", "sub/file.txt"})

    def test_delete_if_path_in_repo(self):
        # Regression test for https://github.com/huggingface/huggingface_hub/pull/2129
        operations = self._upload_folder_alias(path_in_repo=".", folder_path=self.cache_dir, delete_patterns="*")
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}
        assert deleted_files == {"file1.txt", "sub/file1.txt"}  # all the 'old' files


@pytest.mark.usefixtures("fx_cache_dir")
class HfLargefilesTest(HfApiCommonTest):
    cache_dir: Path

    def tearDown(self):
        self._api.delete_repo(repo_id=self.repo_id)

    def setup_local_clone(self) -> None:
        scheme = urlparse(self.repo_url).scheme
        repo_url_auth = self.repo_url.replace(f"{scheme}://", f"{scheme}://user:{TOKEN}@")

        subprocess.run(
            ["git", "clone", repo_url_auth, str(self.cache_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(["git", "lfs", "track", "*.pdf"], check=True, cwd=self.cache_dir)
        subprocess.run(["git", "lfs", "track", "*.epub"], check=True, cwd=self.cache_dir)

    @require_git_lfs
    def test_end_to_end_thresh_6M(self):
        # Little-hack: create repo with defined `_lfsmultipartthresh`. Only for tests purposes
        self._api._lfsmultipartthresh = 6 * 10**6
        self.repo_url = self._api.create_repo(repo_id=repo_name())
        self.repo_id = self.repo_url.repo_id
        self._api._lfsmultipartthresh = None
        self.setup_local_clone()

        subprocess.run(
            ["wget", LARGE_FILE_18MB], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cache_dir
        )
        subprocess.run(["git", "add", "*"], check=True, cwd=self.cache_dir)
        subprocess.run(["git", "commit", "-m", "commit message"], check=True, cwd=self.cache_dir)

        # This will fail as we haven't set up our custom transfer agent yet.
        failed_process = subprocess.run(
            ["git", "push"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cache_dir,
        )
        self.assertEqual(failed_process.returncode, 1)
        self.assertIn("cli lfs-enable-largefiles", failed_process.stderr.decode())
        # ^ Instructions on how to fix this are included in the error message.
        subprocess.run(["huggingface-cli", "lfs-enable-largefiles", self.cache_dir], check=True)

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=self.cache_dir)
        print("took", time.time() - start_time)

        # To be 100% sure, let's download the resolved file
        pdf_url = f"{self.repo_url}/resolve/main/progit.pdf"
        DEST_FILENAME = "uploaded.pdf"
        subprocess.run(
            ["wget", pdf_url, "-O", DEST_FILENAME],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cache_dir,
        )
        dest_filesize = (self.cache_dir / DEST_FILENAME).stat().st_size
        self.assertEqual(dest_filesize, 18685041)

    @require_git_lfs
    def test_end_to_end_thresh_16M(self):
        # Here we'll push one multipart and one non-multipart file in the same commit, and see what happens
        # Little-hack: create repo with defined `_lfsmultipartthresh`. Only for tests purposes
        self._api._lfsmultipartthresh = 16 * 10**6
        self.repo_url = self._api.create_repo(repo_id=repo_name())
        self.repo_id = self.repo_url.repo_id
        self._api._lfsmultipartthresh = None
        self.setup_local_clone()

        subprocess.run(
            ["wget", LARGE_FILE_18MB], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cache_dir
        )
        subprocess.run(
            ["wget", LARGE_FILE_14MB], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cache_dir
        )
        subprocess.run(["git", "add", "*"], check=True, cwd=self.cache_dir)
        subprocess.run(["git", "commit", "-m", "both files in same commit"], check=True, cwd=self.cache_dir)
        subprocess.run(["huggingface-cli", "lfs-enable-largefiles", self.cache_dir], check=True)

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=self.cache_dir)
        print("took", time.time() - start_time)

    def test_upload_lfs_file_multipart(self):
        """End to end test to check upload an LFS file using multipart upload works."""
        self._api._lfsmultipartthresh = 16 * 10**6
        self.repo_id = self._api.create_repo(repo_id=repo_name()).repo_id
        self._api._lfsmultipartthresh = None

        with patch.object(
            huggingface_hub.lfs,
            "_upload_parts_iteratively",
            wraps=huggingface_hub.lfs._upload_parts_iteratively,
        ) as mock:
            self._api.upload_file(repo_id=self.repo_id, path_or_fileobj=b"0" * 18 * 10**6, path_in_repo="lfs.bin")
            mock.assert_called_once()  # It used multipart upload


class ParseHFUrlTest(unittest.TestCase):
    def test_repo_type_and_id_from_hf_id_on_correct_values(self):
        possible_values = {
            "https://huggingface.co/id": [None, None, "id"],
            "https://huggingface.co/user/id": [None, "user", "id"],
            "https://huggingface.co/datasets/user/id": ["dataset", "user", "id"],
            "https://huggingface.co/spaces/user/id": ["space", "user", "id"],
            "user/id": [None, "user", "id"],
            "dataset/user/id": ["dataset", "user", "id"],
            "space/user/id": ["space", "user", "id"],
            "id": [None, None, "id"],
            "hf://id": [None, None, "id"],
            "hf://user/id": [None, "user", "id"],
            "hf://model/user/name": ["model", "user", "name"],  # 's' is optional
            "hf://models/user/name": ["model", "user", "name"],
        }

        for key, value in possible_values.items():
            self.assertEqual(
                repo_type_and_id_from_hf_id(key, hub_url="https://huggingface.co"),
                tuple(value),
            )

    def test_repo_type_and_id_from_hf_id_on_wrong_values(self):
        for hub_id in [
            "https://unknown-endpoint.co/id",
            "https://huggingface.co/datasets/user/id@revision",  # @ forbidden
            "datasets/user/id/subpath",
            "hffs://model/user/name",
            "spaeces/user/id",  # with typo in repo type
        ]:
            with self.assertRaises(ValueError):
                repo_type_and_id_from_hf_id(hub_id, hub_url="https://huggingface.co")


class HfApiDiscussionsTest(HfApiCommonTest):
    def setUp(self):
        self.repo_id = self._api.create_repo(repo_id=repo_name()).repo_id
        self.pull_request = self._api.create_discussion(
            repo_id=self.repo_id, pull_request=True, title="Test Pull Request"
        )
        self.discussion = self._api.create_discussion(
            repo_id=self.repo_id, pull_request=False, title="Test Discussion"
        )

    def tearDown(self):
        self._api.delete_repo(repo_id=self.repo_id)

    def test_create_discussion(self):
        discussion = self._api.create_discussion(repo_id=self.repo_id, title=" Test discussion !  ")
        self.assertEqual(discussion.num, 3)
        self.assertEqual(discussion.author, USER)
        self.assertEqual(discussion.is_pull_request, False)
        self.assertEqual(discussion.title, "Test discussion !")

    @use_tmp_repo("dataset")
    def test_create_discussion_space(self, repo_url: RepoUrl):
        """Regression test for #1463.

        Computed URL was malformed with `dataset` and `space` repo_types.
        See https://github.com/huggingface/huggingface_hub/issues/1463.
        """
        discussion = self._api.create_discussion(repo_id=repo_url.repo_id, repo_type="dataset", title="title")
        self.assertEqual(discussion.url, f"{repo_url}/discussions/1")

    def test_create_pull_request(self):
        discussion = self._api.create_discussion(repo_id=self.repo_id, title=" Test PR !  ", pull_request=True)
        self.assertEqual(discussion.num, 3)
        self.assertEqual(discussion.author, USER)
        self.assertEqual(discussion.is_pull_request, True)
        self.assertEqual(discussion.title, "Test PR !")

        model_info = self._api.repo_info(repo_id=self.repo_id, revision="refs/pr/1")
        self.assertIsInstance(model_info, ModelInfo)

    def test_get_repo_discussion(self):
        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id)
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(
            list([d.num for d in discussions_generator]), [self.discussion.num, self.pull_request.num]
        )

    def test_get_repo_discussion_by_type(self):
        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, discussion_type="pull_request")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(list([d.num for d in discussions_generator]), [self.pull_request.num])

        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, discussion_type="discussion")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(list([d.num for d in discussions_generator]), [self.discussion.num])

        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, discussion_type="all")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(
            list([d.num for d in discussions_generator]), [self.discussion.num, self.pull_request.num]
        )

    def test_get_repo_discussion_by_author(self):
        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, author="unknown")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(list([d.num for d in discussions_generator]), [])

    def test_get_repo_discussion_by_status(self):
        self._api.change_discussion_status(self.repo_id, self.discussion.num, "closed")

        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, discussion_status="open")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(list([d.num for d in discussions_generator]), [self.pull_request.num])

        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, discussion_status="closed")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(list([d.num for d in discussions_generator]), [self.discussion.num])

        discussions_generator = self._api.get_repo_discussions(repo_id=self.repo_id, discussion_status="all")
        self.assertIsInstance(discussions_generator, types.GeneratorType)
        self.assertListEqual(
            list([d.num for d in discussions_generator]), [self.discussion.num, self.pull_request.num]
        )

    @with_production_testing
    def test_get_repo_discussion_pagination(self):
        discussions = list(
            HfApi().get_repo_discussions(repo_id="open-llm-leaderboard/open_llm_leaderboard", repo_type="space")
        )
        assert len(discussions) > 50

    def test_get_discussion_details(self):
        retrieved = self._api.get_discussion_details(repo_id=self.repo_id, discussion_num=2)
        self.assertEqual(retrieved, self.discussion)

    def test_edit_discussion_comment(self):
        def get_first_comment(discussion: DiscussionWithDetails) -> DiscussionComment:
            return [evt for evt in discussion.events if evt.type == "comment"][0]

        edited_comment = self._api.edit_discussion_comment(
            repo_id=self.repo_id,
            discussion_num=self.pull_request.num,
            comment_id=get_first_comment(self.pull_request).id,
            new_content="**Edited** comment 🤗",
        )
        retrieved = self._api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.pull_request.num)
        self.assertEqual(get_first_comment(retrieved).edited, True)
        self.assertEqual(get_first_comment(retrieved).id, get_first_comment(self.pull_request).id)
        self.assertEqual(get_first_comment(retrieved).content, "**Edited** comment 🤗")

        self.assertEqual(get_first_comment(retrieved), edited_comment)

    def test_comment_discussion(self):
        new_comment = self._api.comment_discussion(
            repo_id=self.repo_id,
            discussion_num=self.discussion.num,
            comment="""\
                # Multi-line comment

                **With formatting**, including *italic text* & ~strike through~
                And even [links](http://hf.co)! 💥🤯
            """,
        )
        retrieved = self._api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.discussion.num)
        self.assertEqual(len(retrieved.events), 2)
        self.assertIn(new_comment.id, {event.id for event in retrieved.events})

    def test_rename_discussion(self):
        rename_event = self._api.rename_discussion(
            repo_id=self.repo_id, discussion_num=self.discussion.num, new_title="New title2"
        )
        retrieved = self._api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.discussion.num)
        self.assertIn(rename_event.id, (event.id for event in retrieved.events))
        self.assertEqual(rename_event.old_title, self.discussion.title)
        self.assertEqual(rename_event.new_title, "New title2")

    def test_change_discussion_status(self):
        status_change_event = self._api.change_discussion_status(
            repo_id=self.repo_id, discussion_num=self.discussion.num, new_status="closed"
        )
        retrieved = self._api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.discussion.num)
        self.assertIn(status_change_event.id, (event.id for event in retrieved.events))
        self.assertEqual(status_change_event.new_status, "closed")

        with self.assertRaises(ValueError):
            self._api.change_discussion_status(
                repo_id=self.repo_id, discussion_num=self.discussion.num, new_status="published"
            )

    def test_merge_pull_request(self):
        self._api.create_commit(
            repo_id=self.repo_id,
            commit_message="Commit some file",
            operations=[CommitOperationAdd(path_in_repo="file.test", path_or_fileobj=b"Content")],
            revision=self.pull_request.git_reference,
        )
        self._api.change_discussion_status(
            repo_id=self.repo_id, discussion_num=self.pull_request.num, new_status="open"
        )
        self._api.merge_pull_request(self.repo_id, self.pull_request.num)

        retrieved = self._api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.pull_request.num)
        self.assertEqual(retrieved.status, "merged")
        self.assertIsNotNone(retrieved.merge_commit_oid)


class ActivityApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = HfApi()  # no auth!

    def test_like_and_unlike_repo(self) -> None:
        # Create and like a private and a public repo
        repo_id_private = self.api.create_repo(repo_name(), token=TOKEN, private=True).repo_id
        self.api.like(repo_id_private, token=TOKEN)

        repo_id_public = self.api.create_repo(repo_name(), token=TOKEN, private=False).repo_id
        self.api.like(repo_id_public, token=TOKEN)

        # Get likes as public and authenticated
        likes = self.api.list_liked_repos(USER)
        likes_with_auth = self.api.list_liked_repos(USER, token=TOKEN)

        # Public repo is shown in liked repos
        self.assertIn(repo_id_public, likes.models)
        self.assertIn(repo_id_public, likes_with_auth.models)

        # Private repo is NOT shown in liked repos, even when authenticated
        # This is by design. See https://github.com/huggingface/moon-landing/pull/4879 (internal link)
        self.assertNotIn(repo_id_private, likes.models)
        self.assertNotIn(repo_id_private, likes_with_auth.models)

        # Unlike repo and check not in liked list
        self.api.unlike(repo_id_public, token=TOKEN)
        self.api.unlike(repo_id_private, token=TOKEN)
        likes_after_unlike = self.api.list_liked_repos(USER)
        self.assertNotIn(repo_id_public, likes_after_unlike.models)  # Unliked

        # Cleanup
        self.api.delete_repo(repo_id_public, token=TOKEN)
        self.api.delete_repo(repo_id_private, token=TOKEN)

    def test_like_missing_repo(self) -> None:
        with self.assertRaises(RepositoryNotFoundError):
            self.api.like("missing_repo_id", token=TOKEN)

        with self.assertRaises(RepositoryNotFoundError):
            self.api.unlike("missing_repo_id", token=TOKEN)

    def test_like_twice(self) -> None:
        # Create and like repo
        repo_id = self.api.create_repo(repo_name(), token=TOKEN, private=True).repo_id

        # Can like twice
        self.api.like(repo_id, token=TOKEN)
        self.api.like(repo_id, token=TOKEN)

        # Can unlike twice
        self.api.unlike(repo_id, token=TOKEN)
        self.api.unlike(repo_id, token=TOKEN)

        # Cleanup
        self.api.delete_repo(repo_id, token=TOKEN)

    def test_list_liked_repos_no_auth(self) -> None:
        # Create a repo + like
        repo_id = self.api.create_repo(repo_name(), exist_ok=True, token=TOKEN).repo_id
        self.api.like(repo_id, token=TOKEN)

        # Fetch liked repos without auth
        likes = self.api.list_liked_repos(USER, token=False)
        self.assertEqual(likes.user, USER)
        self.assertGreater(len(likes.models) + len(likes.datasets) + len(likes.spaces), 0)
        self.assertIn(repo_id, likes.models)

        # Cleanup
        self.api.delete_repo(repo_id, token=TOKEN)

    def test_list_likes_repos_auth_and_implicit_user(self) -> None:
        # User is implicit
        likes = self.api.list_liked_repos(token=TOKEN)
        self.assertEqual(likes.user, USER)

    def test_list_likes_repos_auth_and_explicit_user(self) -> None:
        # User is explicit even if auth
        likes = self.api.list_liked_repos(user=OTHER_USER, token=TOKEN)
        self.assertEqual(likes.user, OTHER_USER)

    def test_list_repo_likers(self) -> None:
        # Create a repo + like
        repo_id = self.api.create_repo(repo_name(), token=TOKEN).repo_id
        self.api.like(repo_id, token=TOKEN)

        # Use list_repo_likers to get the list of users who liked this repo
        likers = self.api.list_repo_likers(repo_id, token=TOKEN)

        # Check if the test user is in the list of likers
        liker_usernames = [user.username for user in likers]
        self.assertGreater(len(likers), 0)
        self.assertIn(USER, liker_usernames)

        # Cleanup
        self.api.delete_repo(repo_id, token=TOKEN)

    @with_production_testing
    def test_list_likes_on_production(self) -> None:
        # Test julien-c likes a lot of repos !
        likes = HfApi().list_liked_repos("julien-c")
        self.assertEqual(len(likes.models) + len(likes.datasets) + len(likes.spaces), likes.total)
        self.assertGreater(len(likes.models), 0)
        self.assertGreater(len(likes.datasets), 0)
        self.assertGreater(len(likes.spaces), 0)


class TestSquashHistory(HfApiCommonTest):
    @use_tmp_repo()
    def test_super_squash_history(self, repo_url: RepoUrl) -> None:
        # Upload + update file on main
        repo_id = repo_url.repo_id
        self._api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"content")
        self._api.upload_file(repo_id=repo_id, path_in_repo="lfs.bin", path_or_fileobj=b"content")
        self._api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"another_content")

        # Upload file on a new branch
        self._api.create_branch(repo_id=repo_id, branch="v0.1", exist_ok=True)
        self._api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"foo", revision="v0.1")

        # Squash history on main
        self._api.super_squash_history(repo_id=repo_id)

        # List history
        squashed_main_commits = self._api.list_repo_commits(repo_id=repo_id, revision="main")
        branch_commits = self._api.list_repo_commits(repo_id=repo_id, revision="v0.1")

        # Main branch has been squashed but initial commits still exists on other branch
        self.assertEqual(len(squashed_main_commits), 1)
        self.assertEqual(squashed_main_commits[0].title, "Super-squash branch 'main' using huggingface_hub")
        self.assertEqual(len(branch_commits), 5)
        self.assertEqual(branch_commits[-1].title, "initial commit")

        # Squash history on branch
        self._api.super_squash_history(repo_id=repo_id, branch="v0.1")
        squashed_branch_commits = self._api.list_repo_commits(repo_id=repo_id, revision="v0.1")
        self.assertEqual(len(squashed_branch_commits), 1)
        self.assertEqual(squashed_branch_commits[0].title, "Super-squash branch 'v0.1' using huggingface_hub")


@pytest.mark.vcr
class TestSpaceAPIProduction(unittest.TestCase):
    """
    Testing Space API is not possible on staging. We use VCR-ed to mimic server requests.
    """

    repo_id: str
    api: HfApi

    _BASIC_APP_PY_TEMPLATE = """
import gradio as gr


def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
""".encode()

    def setUp(self):
        super().setUp()

        # If generating new VCR => use personal token and REMOVE IT from the VCR
        self.repo_id = "user/tmp_test_space"  # no need to be unique as it's a VCRed test
        self.api = HfApi(token="hf_fake_token", endpoint="https://huggingface.co")

        # Create a Space
        self.api.create_repo(repo_id=self.repo_id, repo_type="space", space_sdk="gradio", private=True)
        self.api.upload_file(
            path_or_fileobj=self._BASIC_APP_PY_TEMPLATE,
            repo_id=self.repo_id,
            repo_type="space",
            path_in_repo="app.py",
        )

    def tearDown(self):
        self.api.delete_repo(repo_id=self.repo_id, repo_type="space")
        super().tearDown()

    def test_manage_secrets(self) -> None:
        # Add 3 secrets
        self.api.add_space_secret(self.repo_id, "foo", "123")
        self.api.add_space_secret(self.repo_id, "token", "hf_api_123456")
        self.api.add_space_secret(self.repo_id, "gh_api_key", "******")

        # Add secret with optional description
        self.api.add_space_secret(self.repo_id, "bar", "123", description="This is a secret")

        # Update secret
        self.api.add_space_secret(self.repo_id, "foo", "456")

        # Update secret with optional description
        self.api.add_space_secret(self.repo_id, "foo", "789", description="This is a secret")
        self.api.add_space_secret(self.repo_id, "bar", "456", description="This is another secret")

        # Delete secret
        self.api.delete_space_secret(self.repo_id, "gh_api_key")

        # Doesn't fail on missing key
        self.api.delete_space_secret(self.repo_id, "missing_key")

    def test_manage_variables(self) -> None:
        # Get variables
        self.api.get_space_variables(self.repo_id)

        # Add 3 variables
        self.api.add_space_variable(self.repo_id, "foo", "123")
        self.api.add_space_variable(self.repo_id, "MODEL_REPO_ID", "user/repo")

        # Add 1 variable with optional description
        self.api.add_space_variable(self.repo_id, "MODEL_PAPER", "arXiv", description="found it there")

        # Update variable
        self.api.add_space_variable(self.repo_id, "foo", "456")

        # Update variable with optional description
        self.api.add_space_variable(self.repo_id, "foo", "456", description="updated description")

        # Delete variable
        self.api.delete_space_variable(self.repo_id, "gh_api_key")

        # Doesn't fail on missing key
        self.api.delete_space_variable(self.repo_id, "missing_key")

        # Returning all variables created
        variables = self.api.get_space_variables(self.repo_id)
        self.assertEqual(len(variables), 3)

    def test_space_runtime(self) -> None:
        runtime = self.api.get_space_runtime(self.repo_id)

        # Space has just been created: hardware might not be set yet.
        self.assertIn(runtime.hardware, (None, SpaceHardware.CPU_BASIC))
        self.assertIn(runtime.requested_hardware, (None, SpaceHardware.CPU_BASIC))

        # Space is either "BUILDING" (if not yet done) or "NO_APP_FILE" (if building failed)
        self.assertIn(runtime.stage, (SpaceStage.NO_APP_FILE, SpaceStage.BUILDING))
        self.assertIn(runtime.stage, ("NO_APP_FILE", "BUILDING"))  # str works as well

        # Raw response from Hub
        self.assertIsInstance(runtime.raw, dict)

    def test_static_space_runtime(self) -> None:
        """
        Regression test for static Spaces.
        See https://github.com/huggingface/huggingface_hub/pull/1754.
        """
        runtime = self.api.get_space_runtime("victor/static-space")
        self.assertIsInstance(runtime.raw, dict)

    def test_pause_and_restart_space(self) -> None:
        # Upload a fake app.py file
        self.api.upload_file(path_or_fileobj=b"", path_in_repo="app.py", repo_id=self.repo_id, repo_type="space")

        # Wait for the Space to be "BUILDING"
        count = 0
        while True:
            if self.api.get_space_runtime(self.repo_id).stage == SpaceStage.BUILDING:
                break
            time.sleep(1.0)
            count += 1
            if count > 10:
                raise Exception("Space is not building after 10 seconds.")

        # Pause it
        runtime_after_pause = self.api.pause_space(self.repo_id)
        self.assertEqual(runtime_after_pause.stage, SpaceStage.PAUSED)

        # Restart
        self.api.restart_space(self.repo_id)
        time.sleep(0.5)
        runtime_after_restart = self.api.get_space_runtime(self.repo_id)
        self.assertNotEqual(runtime_after_restart.stage, SpaceStage.PAUSED)


@pytest.mark.usefixtures("fx_cache_dir")
class TestCommitInBackground(HfApiCommonTest):
    cache_dir: Path

    @use_tmp_repo()
    def test_commit_to_repo_in_background(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        (self.cache_dir / "file.txt").write_text("content")
        (self.cache_dir / "lfs.bin").write_text("content")

        t0 = time.time()
        upload_future_1 = self._api.upload_file(
            path_or_fileobj=b"1", path_in_repo="1.txt", repo_id=repo_id, commit_message="Upload 1", run_as_future=True
        )
        upload_future_2 = self._api.upload_file(
            path_or_fileobj=b"2", path_in_repo="2.txt", repo_id=repo_id, commit_message="Upload 2", run_as_future=True
        )
        upload_future_3 = self._api.upload_folder(
            repo_id=repo_id, folder_path=self.cache_dir, commit_message="Upload folder", run_as_future=True
        )
        t1 = time.time()

        # all futures are queued instantly
        self.assertLessEqual(t1 - t0, 0.01)

        # wait for the last job to complete
        upload_future_3.result()

        # all of them are now complete (ran in order)
        self.assertTrue(upload_future_1.done())
        self.assertTrue(upload_future_2.done())
        self.assertTrue(upload_future_3.done())

        # 4 commits, sorted in reverse order of creation
        commits = self._api.list_repo_commits(repo_id=repo_id)
        self.assertEqual(len(commits), 4)
        self.assertEqual(commits[0].title, "Upload folder")
        self.assertEqual(commits[1].title, "Upload 2")
        self.assertEqual(commits[2].title, "Upload 1")
        self.assertEqual(commits[3].title, "initial commit")

    @use_tmp_repo()
    def test_run_as_future(self, repo_url: RepoUrl) -> None:
        repo_id = repo_url.repo_id
        self._api.run_as_future(self._api.like, repo_id)
        future_1 = self._api.run_as_future(self._api.model_info, repo_id=repo_id)
        self._api.run_as_future(self._api.unlike, repo_id)
        future_2 = self._api.run_as_future(self._api.model_info, repo_id=repo_id)

        self.assertIsInstance(future_1, Future)
        self.assertIsInstance(future_2, Future)

        # Wait for first info future
        info_1 = future_1.result()
        self.assertFalse(future_2.done())

        # Wait for second info future
        info_2 = future_2.result()
        self.assertTrue(future_2.done())

        # Like/unlike is correct
        self.assertEqual(info_1.likes, 1)
        self.assertEqual(info_2.likes, 0)


class TestDownloadHfApiAlias(unittest.TestCase):
    def setUp(self) -> None:
        self.api = HfApi(
            endpoint="https://hf.co",
            token="user_token",
            library_name="cool_one",
            library_version="1.0.0",
            user_agent="myself",
        )
        return super().setUp()

    @patch("huggingface_hub.file_download.hf_hub_download")
    def test_hf_hub_download_alias(self, mock: Mock) -> None:
        self.api.hf_hub_download("my_repo_id", "file.txt")
        mock.assert_called_once_with(
            # Call values
            repo_id="my_repo_id",
            filename="file.txt",
            # HfAPI values
            endpoint="https://hf.co",
            library_name="cool_one",
            library_version="1.0.0",
            user_agent="myself",
            token="user_token",
            # Default values
            subfolder=None,
            repo_type=None,
            revision=None,
            cache_dir=None,
            local_dir=None,
            local_dir_use_symlinks="auto",
            force_download=False,
            force_filename=None,
            proxies=None,
            etag_timeout=10,
            resume_download=None,
            local_files_only=False,
            legacy_cache_layout=False,
            headers=None,
        )

    @patch("huggingface_hub._snapshot_download.snapshot_download")
    def test_snapshot_download_alias(self, mock: Mock) -> None:
        self.api.snapshot_download("my_repo_id")
        mock.assert_called_once_with(
            # Call values
            repo_id="my_repo_id",
            # HfAPI values
            endpoint="https://hf.co",
            library_name="cool_one",
            library_version="1.0.0",
            user_agent="myself",
            token="user_token",
            # Default values
            repo_type=None,
            revision=None,
            cache_dir=None,
            local_dir=None,
            local_dir_use_symlinks="auto",
            proxies=None,
            etag_timeout=10,
            resume_download=None,
            force_download=False,
            local_files_only=False,
            allow_patterns=None,
            ignore_patterns=None,
            max_workers=8,
            tqdm_class=None,
        )


class TestSpaceAPIMocked(unittest.TestCase):
    """
    Testing Space hardware requests is resource intensive for the server (need to spawn
    GPUs). Tests are mocked to check the correct values are sent.
    """

    def setUp(self) -> None:
        self.api = HfApi(token="fake_token")
        self.repo_id = "fake_repo_id"

        get_session_mock = Mock()
        self.post_mock = get_session_mock().post
        self.post_mock.return_value.json.return_value = {
            "url": f"{self.api.endpoint}/spaces/user/repo_id",
            "stage": "RUNNING",
            "sdk": "gradio",
            "sdkVersion": "3.17.0",
            "hardware": {
                "current": "t4-medium",
                "requested": "t4-medium",
            },
            "storage": "large",
            "gcTimeout": None,
        }
        self.delete_mock = get_session_mock().delete
        self.delete_mock.return_value.json.return_value = {
            "url": f"{self.api.endpoint}/spaces/user/repo_id",
            "stage": "RUNNING",
            "sdk": "gradio",
            "sdkVersion": "3.17.0",
            "hardware": {
                "current": "t4-medium",
                "requested": "t4-medium",
            },
            "storage": None,
            "gcTimeout": None,
        }
        self.patcher = patch("huggingface_hub.hf_api.get_session", get_session_mock)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_create_space_with_hardware(self) -> None:
        self.api.create_repo(
            self.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.T4_MEDIUM,
        )
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/repos/create",
            headers=self.api._build_hf_headers(),
            json={
                "name": self.repo_id,
                "organization": None,
                "private": False,
                "type": "space",
                "sdk": "gradio",
                "hardware": "t4-medium",
            },
        )

    def test_create_space_with_hardware_and_sleep_time(self) -> None:
        self.api.create_repo(
            self.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.T4_MEDIUM,
            space_sleep_time=123,
        )
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/repos/create",
            headers=self.api._build_hf_headers(),
            json={
                "name": self.repo_id,
                "organization": None,
                "private": False,
                "type": "space",
                "sdk": "gradio",
                "hardware": "t4-medium",
                "sleepTimeSeconds": 123,
            },
        )

    def test_create_space_with_storage(self) -> None:
        self.api.create_repo(
            self.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_storage=SpaceStorage.LARGE,
        )
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/repos/create",
            headers=self.api._build_hf_headers(),
            json={
                "name": self.repo_id,
                "organization": None,
                "private": False,
                "type": "space",
                "sdk": "gradio",
                "storageTier": "large",
            },
        )

    def test_create_space_with_secrets_and_variables(self) -> None:
        self.api.create_repo(
            self.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_secrets=[
                {"key": "Testsecret", "value": "Testvalue", "description": "Testdescription"},
                {"key": "Testsecret2", "value": "Testvalue"},
            ],
            space_variables=[
                {"key": "Testvariable", "value": "Testvalue", "description": "Testdescription"},
                {"key": "Testvariable2", "value": "Testvalue"},
            ],
        )
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/repos/create",
            headers=self.api._build_hf_headers(),
            json={
                "name": self.repo_id,
                "organization": None,
                "private": False,
                "type": "space",
                "sdk": "gradio",
                "secrets": [
                    {"key": "Testsecret", "value": "Testvalue", "description": "Testdescription"},
                    {"key": "Testsecret2", "value": "Testvalue"},
                ],
                "variables": [
                    {"key": "Testvariable", "value": "Testvalue", "description": "Testdescription"},
                    {"key": "Testvariable2", "value": "Testvalue"},
                ],
            },
        )

    def test_duplicate_space(self) -> None:
        self.api.duplicate_space(
            self.repo_id,
            to_id=f"{USER}/new_repo_id",
            private=True,
            hardware=SpaceHardware.T4_MEDIUM,
            storage=SpaceStorage.LARGE,
            sleep_time=123,
            secrets=[
                {"key": "Testsecret", "value": "Testvalue", "description": "Testdescription"},
                {"key": "Testsecret2", "value": "Testvalue"},
            ],
            variables=[
                {"key": "Testvariable", "value": "Testvalue", "description": "Testdescription"},
                {"key": "Testvariable2", "value": "Testvalue"},
            ],
        )
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/duplicate",
            headers=self.api._build_hf_headers(),
            json={
                "repository": f"{USER}/new_repo_id",
                "private": True,
                "hardware": "t4-medium",
                "storageTier": "large",
                "sleepTimeSeconds": 123,
                "secrets": [
                    {"key": "Testsecret", "value": "Testvalue", "description": "Testdescription"},
                    {"key": "Testsecret2", "value": "Testvalue"},
                ],
                "variables": [
                    {"key": "Testvariable", "value": "Testvalue", "description": "Testdescription"},
                    {"key": "Testvariable2", "value": "Testvalue"},
                ],
            },
        )

    def test_request_space_hardware_no_sleep_time(self) -> None:
        self.api.request_space_hardware(self.repo_id, SpaceHardware.T4_MEDIUM)
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/hardware",
            headers=self.api._build_hf_headers(),
            json={"flavor": "t4-medium"},
        )

    def test_request_space_hardware_with_sleep_time(self) -> None:
        self.api.request_space_hardware(self.repo_id, SpaceHardware.T4_MEDIUM, sleep_time=123)
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/hardware",
            headers=self.api._build_hf_headers(),
            json={"flavor": "t4-medium", "sleepTimeSeconds": 123},
        )

    def test_set_space_sleep_time_upgraded_hardware(self) -> None:
        self.api.set_space_sleep_time(self.repo_id, sleep_time=123)
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/sleeptime",
            headers=self.api._build_hf_headers(),
            json={"seconds": 123},
        )

    def test_set_space_sleep_time_cpu_basic(self) -> None:
        self.post_mock.return_value.json.return_value["hardware"]["requested"] = "cpu-basic"
        with self.assertWarns(UserWarning):
            self.api.set_space_sleep_time(self.repo_id, sleep_time=123)

    def test_request_space_storage(self) -> None:
        runtime = self.api.request_space_storage(self.repo_id, SpaceStorage.LARGE)
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/storage",
            headers=self.api._build_hf_headers(),
            json={"tier": "large"},
        )
        assert runtime.storage == SpaceStorage.LARGE

    def test_delete_space_storage(self) -> None:
        runtime = self.api.delete_space_storage(self.repo_id)
        self.delete_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/storage",
            headers=self.api._build_hf_headers(),
        )
        assert runtime.storage is None

    def test_restart_space_factory_reboot(self) -> None:
        self.api.restart_space(self.repo_id, factory_reboot=True)
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/restart",
            headers=self.api._build_hf_headers(),
            params={"factory": "true"},
        )


class ListGitRefsTest(unittest.TestCase):
    @classmethod
    @with_production_testing
    def setUpClass(cls) -> None:
        cls.api = HfApi()
        return super().setUpClass()

    def test_list_refs_gpt2(self) -> None:
        refs = self.api.list_repo_refs("gpt2")
        self.assertGreater(len(refs.branches), 0)
        main_branch = [branch for branch in refs.branches if branch.name == "main"][0]
        self.assertEqual(main_branch.ref, "refs/heads/main")
        self.assertIsNone(refs.pull_requests)
        # Can get info by revision
        self.api.repo_info("gpt2", revision=main_branch.target_commit)

    def test_list_refs_bigcode(self) -> None:
        refs = self.api.list_repo_refs("bigcode/admin", repo_type="dataset")
        self.assertGreater(len(refs.branches), 0)
        self.assertGreater(len(refs.converts), 0)
        self.assertIsNone(refs.pull_requests)
        main_branch = [branch for branch in refs.branches if branch.name == "main"][0]
        self.assertEqual(main_branch.ref, "refs/heads/main")

        convert_branch = [branch for branch in refs.converts if branch.name == "parquet"][0]
        self.assertEqual(convert_branch.ref, "refs/convert/parquet")

        # Can get info by convert revision
        self.api.repo_info(
            "bigcode/admin",
            repo_type="dataset",
            revision=convert_branch.target_commit,
        )

    def test_list_refs_with_prs(self) -> None:
        refs = self.api.list_repo_refs("openchat/openchat_3.5", include_pull_requests=True)
        self.assertGreater(len(refs.pull_requests), 1)
        self.assertTrue(refs.pull_requests[0].ref.startswith("refs/pr/"))


class ListGitCommitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = HfApi(token=TOKEN)
        # Create repo (with initial commit)
        cls.repo_id = cls.api.create_repo(repo_name()).repo_id

        # Create a commit on `main` branch
        cls.api.upload_file(repo_id=cls.repo_id, path_or_fileobj=b"content", path_in_repo="content.txt")

        # Create a commit in a PR
        cls.api.upload_file(repo_id=cls.repo_id, path_or_fileobj=b"on_pr", path_in_repo="on_pr.txt", create_pr=True)

        # Create another commit on `main` branch
        cls.api.upload_file(repo_id=cls.repo_id, path_or_fileobj=b"on_main", path_in_repo="on_main.txt")
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.delete_repo(cls.repo_id)
        return super().tearDownClass()

    def test_list_commits_on_main(self) -> None:
        commits = self.api.list_repo_commits(self.repo_id)

        # "on_pr" commit not returned
        self.assertEqual(len(commits), 3)
        self.assertTrue(all("on_pr" not in commit.title for commit in commits))

        # USER is always the author
        self.assertTrue(all(commit.authors == [USER] for commit in commits))

        # latest commit first
        self.assertEqual(commits[0].title, "Upload on_main.txt with huggingface_hub")

        # Formatted field not returned by default
        for commit in commits:
            self.assertIsNone(commit.formatted_title)
            self.assertIsNone(commit.formatted_message)

    def test_list_commits_on_pr(self) -> None:
        commits = self.api.list_repo_commits(self.repo_id, revision="refs/pr/1")

        # "on_pr" commit returned but not the "on_main" one
        self.assertEqual(len(commits), 3)
        self.assertTrue(all("on_main" not in commit.title for commit in commits))
        self.assertEqual(commits[0].title, "Upload on_pr.txt with huggingface_hub")

    def test_list_commits_include_formatted(self) -> None:
        for commit in self.api.list_repo_commits(self.repo_id, formatted=True):
            self.assertIsNotNone(commit.formatted_title)
            self.assertIsNotNone(commit.formatted_message)

    def test_list_commits_on_missing_repo(self) -> None:
        with self.assertRaises(RepositoryNotFoundError):
            self.api.list_repo_commits("missing_repo_id")

    def test_list_commits_on_missing_revision(self) -> None:
        with self.assertRaises(RevisionNotFoundError):
            self.api.list_repo_commits(self.repo_id, revision="missing_revision")


@patch("huggingface_hub.hf_api.build_hf_headers")
class HfApiTokenAttributeTest(unittest.TestCase):
    def test_token_passed(self, mock_build_hf_headers: Mock) -> None:
        HfApi(token="default token")._build_hf_headers(token="A token")
        self._assert_token_is(mock_build_hf_headers, "A token")

    def test_no_token_passed(self, mock_build_hf_headers: Mock) -> None:
        HfApi(token="default token")._build_hf_headers()
        self._assert_token_is(mock_build_hf_headers, "default token")

    def test_token_true_passed(self, mock_build_hf_headers: Mock) -> None:
        HfApi(token="default token")._build_hf_headers(token=True)
        self._assert_token_is(mock_build_hf_headers, True)

    def test_token_false_passed(self, mock_build_hf_headers: Mock) -> None:
        HfApi(token="default token")._build_hf_headers(token=False)
        self._assert_token_is(mock_build_hf_headers, False)

    def test_no_token_at_all(self, mock_build_hf_headers: Mock) -> None:
        HfApi()._build_hf_headers(token=None)
        self._assert_token_is(mock_build_hf_headers, None)

    def _assert_token_is(self, mock_build_hf_headers: Mock, expected_value: str) -> None:
        self.assertEqual(mock_build_hf_headers.call_args[1]["token"], expected_value)

    def test_library_name_and_version_are_set(self, mock_build_hf_headers: Mock) -> None:
        HfApi(library_name="a", library_version="b")._build_hf_headers()
        self.assertEqual(mock_build_hf_headers.call_args[1]["library_name"], "a")
        self.assertEqual(mock_build_hf_headers.call_args[1]["library_version"], "b")

    def test_library_name_and_version_are_overwritten(self, mock_build_hf_headers: Mock) -> None:
        api = HfApi(library_name="a", library_version="b")
        api._build_hf_headers(library_name="A", library_version="B")
        self.assertEqual(mock_build_hf_headers.call_args[1]["library_name"], "A")
        self.assertEqual(mock_build_hf_headers.call_args[1]["library_version"], "B")

    def test_user_agent_is_set(self, mock_build_hf_headers: Mock) -> None:
        HfApi(user_agent={"a": "b"})._build_hf_headers()
        self.assertEqual(mock_build_hf_headers.call_args[1]["user_agent"], {"a": "b"})

    def test_user_agent_is_overwritten(self, mock_build_hf_headers: Mock) -> None:
        HfApi(user_agent={"a": "b"})._build_hf_headers(user_agent={"A": "B"})
        self.assertEqual(mock_build_hf_headers.call_args[1]["user_agent"], {"A": "B"})


@patch("huggingface_hub.hf_api.ENDPOINT", "https://huggingface.co")
class RepoUrlTest(unittest.TestCase):
    def test_repo_url_class(self):
        url = RepoUrl("https://huggingface.co/gpt2")

        # RepoUrl Is a string
        self.assertIsInstance(url, str)
        self.assertEqual(url, "https://huggingface.co/gpt2")

        # Any str-method can be applied
        self.assertEqual(url.split("/"), "https://huggingface.co/gpt2".split("/"))

        # String formatting and concatenation work
        self.assertEqual(f"New repo: {url}", "New repo: https://huggingface.co/gpt2")
        self.assertEqual("New repo: " + url, "New repo: https://huggingface.co/gpt2")

        # __repr__ is modified for debugging purposes
        self.assertEqual(
            repr(url),
            "RepoUrl('https://huggingface.co/gpt2',"
            " endpoint='https://huggingface.co', repo_type='model', repo_id='gpt2')",
        )

    def test_repo_url_endpoint(self):
        # Implicit endpoint
        url = RepoUrl("https://huggingface.co/gpt2")
        self.assertEqual(url.endpoint, "https://huggingface.co")

        # Explicit endpoint
        url = RepoUrl("https://example.com/gpt2", endpoint="https://example.com")
        self.assertEqual(url.endpoint, "https://example.com")

    def test_repo_url_repo_type(self):
        # Explicit repo type
        url = RepoUrl("https://huggingface.co/user/repo_name")
        self.assertEqual(url.repo_type, "model")

        url = RepoUrl("https://huggingface.co/datasets/user/repo_name")
        self.assertEqual(url.repo_type, "dataset")

        url = RepoUrl("https://huggingface.co/spaces/user/repo_name")
        self.assertEqual(url.repo_type, "space")

        # Implicit repo type (model)
        url = RepoUrl("https://huggingface.co/user/repo_name")
        self.assertEqual(url.repo_type, "model")

    def test_repo_url_namespace(self):
        # Canonical model (e.g. no username)
        url = RepoUrl("https://huggingface.co/gpt2")
        self.assertIsNone(url.namespace)
        self.assertEqual(url.repo_id, "gpt2")

        # "Normal" model
        url = RepoUrl("https://huggingface.co/dummy_user/dummy_model")
        self.assertEqual(url.namespace, "dummy_user")
        self.assertEqual(url.repo_id, "dummy_user/dummy_model")

    def test_repo_url_url_property(self):
        # RepoUrl.url returns a pure `str` value
        url = RepoUrl("https://huggingface.co/gpt2")
        self.assertEqual(url, "https://huggingface.co/gpt2")
        self.assertEqual(url.url, "https://huggingface.co/gpt2")
        self.assertIsInstance(url, RepoUrl)
        self.assertNotIsInstance(url.url, RepoUrl)

    def test_repo_url_canonical_model(self):
        for _id in ("gpt2", "hf://gpt2", "https://huggingface.co/gpt2"):
            with self.subTest(_id):
                url = RepoUrl(_id)
                self.assertEqual(url.repo_id, "gpt2")
                self.assertEqual(url.repo_type, "model")

    def test_repo_url_canonical_dataset(self):
        for _id in ("datasets/squad", "hf://datasets/squad", "https://huggingface.co/datasets/squad"):
            with self.subTest(_id):
                url = RepoUrl(_id)
                self.assertEqual(url.repo_id, "squad")
                self.assertEqual(url.repo_type, "dataset")


class HfApiDuplicateSpaceTest(HfApiCommonTest):
    def test_duplicate_space_success(self) -> None:
        """Check `duplicate_space` works."""
        from_repo_name = repo_name()
        from_repo_id = self._api.create_repo(
            repo_id=from_repo_name,
            repo_type="space",
            space_sdk="static",
            token=OTHER_TOKEN,
        ).repo_id
        self._api.upload_file(
            path_or_fileobj=b"data",
            path_in_repo="temp/new_file.md",
            repo_id=from_repo_id,
            repo_type="space",
            token=OTHER_TOKEN,
        )

        to_repo_id = self._api.duplicate_space(from_repo_id).repo_id

        assert to_repo_id == f"{USER}/{from_repo_name}"
        assert self._api.list_repo_files(repo_id=from_repo_id, repo_type="space") == [
            ".gitattributes",
            "README.md",
            "index.html",
            "style.css",
            "temp/new_file.md",
        ]
        assert self._api.list_repo_files(repo_id=to_repo_id, repo_type="space") == self._api.list_repo_files(
            repo_id=from_repo_id, repo_type="space"
        )

        self._api.delete_repo(repo_id=from_repo_id, repo_type="space", token=OTHER_TOKEN)
        self._api.delete_repo(repo_id=to_repo_id, repo_type="space")

    def test_duplicate_space_from_missing_repo(self) -> None:
        """Check `duplicate_space` fails when the from_repo doesn't exist."""

        with self.assertRaises(RepositoryNotFoundError):
            self._api.duplicate_space(f"{OTHER_USER}/repo_that_does_not_exist")


class CollectionAPITest(HfApiCommonTest):
    def setUp(self) -> None:
        id = uuid.uuid4()
        self.title = f"My cool stuff {id}"
        self.slug_prefix = f"{USER}/my-cool-stuff-{id}"
        self.slug: Optional[str] = None  # Populated by the tests => use to delete in tearDown
        return super().setUp()

    def tearDown(self) -> None:
        if self.slug is not None:  # Delete collection even if test failed
            self._api.delete_collection(self.slug, missing_ok=True)
        return super().tearDown()

    @with_production_testing
    def test_list_collections(self) -> None:
        item_id = "teknium/OpenHermes-2.5-Mistral-7B"
        item_type = "model"
        limit = 3
        collections = HfApi().list_collections(item=f"{item_type}s/{item_id}", limit=limit)

        # Check return type
        self.assertIsInstance(collections, Iterable)
        collections = list(collections)

        # Check length
        self.assertEqual(len(collections), limit)

        # Check all collections contain the item
        for collection in collections:
            # all items are not necessarily returned when listing collections => retrieve complete one
            full_collection = HfApi().get_collection(collection.slug)
            assert any(item.item_id == item_id and item.item_type == item_type for item in full_collection.items)

    def test_create_collection_with_description(self) -> None:
        collection = self._api.create_collection(self.title, description="Contains a lot of cool stuff")
        self.slug = collection.slug

        self.assertIsInstance(collection, Collection)
        self.assertEqual(collection.title, self.title)
        self.assertEqual(collection.description, "Contains a lot of cool stuff")
        self.assertEqual(collection.items, [])
        self.assertTrue(collection.slug.startswith(self.slug_prefix))
        self.assertEqual(collection.url, f"{ENDPOINT_STAGING}/collections/{collection.slug}")

    def test_create_collection_exists_ok(self) -> None:
        # Create collection once without description
        collection_1 = self._api.create_collection(self.title)
        self.slug = collection_1.slug

        # Cannot create twice with same title
        with self.assertRaises(HTTPError):  # already exists
            self._api.create_collection(self.title)

        # Can ignore error
        collection_2 = self._api.create_collection(self.title, description="description", exists_ok=True)

        self.assertEqual(collection_1.slug, collection_2.slug)
        self.assertIsNone(collection_1.description)
        self.assertIsNone(collection_2.description)  # Did not got updated!

    def test_create_private_collection(self) -> None:
        collection = self._api.create_collection(self.title, private=True)
        self.slug = collection.slug

        # Get private collection
        self._api.get_collection(collection.slug)  # no error
        with self.assertRaises(HTTPError):
            self._api.get_collection(collection.slug, token=OTHER_TOKEN)  # not authorized

        # Get public collection
        self._api.update_collection_metadata(collection.slug, private=False)
        self._api.get_collection(collection.slug)  # no error
        self._api.get_collection(collection.slug, token=OTHER_TOKEN)  # no error

    def test_update_collection(self) -> None:
        # Create collection
        collection_1 = self._api.create_collection(self.title)
        self.slug = collection_1.slug

        # Update metadata
        new_title = f"New title {uuid.uuid4()}"
        collection_2 = self._api.update_collection_metadata(
            collection_slug=collection_1.slug,
            title=new_title,
            description="New description",
            private=True,
            theme="pink",
        )

        self.assertEqual(collection_2.title, new_title)
        self.assertEqual(collection_2.description, "New description")
        self.assertEqual(collection_2.private, True)
        self.assertEqual(collection_2.theme, "pink")
        self.assertNotEqual(collection_1.slug, collection_2.slug)

        # Different slug, same id
        self.assertEqual(collection_1.slug.split("-")[-1], collection_2.slug.split("-")[-1])

        # Works with both slugs, same collection returned
        self.assertEqual(self._api.get_collection(collection_1.slug).slug, collection_2.slug)
        self.assertEqual(self._api.get_collection(collection_2.slug).slug, collection_2.slug)

    def test_delete_collection(self) -> None:
        collection = self._api.create_collection(self.title)

        self._api.delete_collection(collection.slug)

        # Cannot delete twice the same collection
        with self.assertRaises(HTTPError):  # already exists
            self._api.delete_collection(collection.slug)

        # Possible to ignore error
        self._api.delete_collection(collection.slug, missing_ok=True)

    def test_collection_items(self) -> None:
        # Create some repos
        model_id = self._api.create_repo(repo_name()).repo_id
        dataset_id = self._api.create_repo(repo_name(), repo_type="dataset").repo_id

        # Create collection + add items to it
        collection = self._api.create_collection(self.title)
        self._api.add_collection_item(collection.slug, model_id, "model", note="This is my model")
        self._api.add_collection_item(collection.slug, dataset_id, "dataset")  # note is optional

        # Check consistency
        collection = self._api.get_collection(collection.slug)
        self.assertEqual(len(collection.items), 2)
        self.assertEqual(collection.items[0].item_id, model_id)
        self.assertEqual(collection.items[0].item_type, "model")
        self.assertEqual(collection.items[0].note, "This is my model")

        self.assertEqual(collection.items[1].item_id, dataset_id)
        self.assertEqual(collection.items[1].item_type, "dataset")
        self.assertIsNone(collection.items[1].note)

        # Add existing item fails (except if ignore error)
        with self.assertRaises(HTTPError):
            self._api.add_collection_item(collection.slug, model_id, "model")
        self._api.add_collection_item(collection.slug, model_id, "model", exists_ok=True)

        # Add inexistent item fails
        with self.assertRaises(HTTPError):
            self._api.add_collection_item(collection.slug, model_id, "dataset")

        # Update first item
        self._api.update_collection_item(
            collection.slug, collection.items[0].item_object_id, note="New note", position=1
        )

        # Check consistency
        collection = self._api.get_collection(collection.slug)
        self.assertEqual(collection.items[0].item_id, dataset_id)  # position got updated
        self.assertEqual(collection.items[1].item_id, model_id)
        self.assertEqual(collection.items[1].note, "New note")  # note got updated

        # Delete last item
        self._api.delete_collection_item(collection.slug, collection.items[1].item_object_id)
        self._api.delete_collection_item(collection.slug, collection.items[1].item_object_id, missing_ok=True)

        # Check consistency
        collection = self._api.get_collection(collection.slug)
        self.assertEqual(len(collection.items), 1)  # only 1 item remaining
        self.assertEqual(collection.items[0].item_id, dataset_id)  # position got updated

        # Delete everything
        self._api.delete_repo(model_id)
        self._api.delete_repo(dataset_id, repo_type="dataset")
        self._api.delete_collection(collection.slug)


class AccessRequestAPITest(HfApiCommonTest):
    def setUp(self) -> None:
        # Setup test with a gated repo
        super().setUp()
        self.repo_id = self._api.create_repo(repo_name()).repo_id
        response = get_session().put(
            f"{self._api.endpoint}/api/models/{self.repo_id}/settings",
            json={"gated": "auto"},
            headers=self._api._build_hf_headers(),
        )
        hf_raise_for_status(response)

    def tearDown(self) -> None:
        self._api.delete_repo(self.repo_id)
        return super().tearDown()

    def test_access_requests_normal_usage(self) -> None:
        # No access requests initially
        requests = self._api.list_accepted_access_requests(self.repo_id)
        assert len(requests) == 0
        requests = self._api.list_pending_access_requests(self.repo_id)
        assert len(requests) == 0
        requests = self._api.list_rejected_access_requests(self.repo_id)
        assert len(requests) == 0

        # Grant access to a user
        self._api.grant_access(self.repo_id, OTHER_USER)

        # User is in accepted list
        requests = self._api.list_accepted_access_requests(self.repo_id)
        assert len(requests) == 1
        request = requests[0]
        assert isinstance(request, AccessRequest)
        assert request.username == OTHER_USER
        assert request.status == "accepted"
        assert isinstance(request.timestamp, datetime.datetime)

        # Cancel access
        self._api.cancel_access_request(self.repo_id, OTHER_USER)
        requests = self._api.list_accepted_access_requests(self.repo_id)
        assert len(requests) == 0  # not accepted anymore
        requests = self._api.list_pending_access_requests(self.repo_id)
        assert len(requests) == 1
        assert requests[0].username == OTHER_USER

        # Reject access
        self._api.reject_access_request(self.repo_id, OTHER_USER)
        requests = self._api.list_pending_access_requests(self.repo_id)
        assert len(requests) == 0  # not pending anymore
        requests = self._api.list_rejected_access_requests(self.repo_id)
        assert len(requests) == 1
        assert requests[0].username == OTHER_USER

        # Accept again
        self._api.accept_access_request(self.repo_id, OTHER_USER)
        requests = self._api.list_accepted_access_requests(self.repo_id)
        assert len(requests) == 1
        assert requests[0].username == OTHER_USER

    def test_access_request_error(self):
        # Grant access to a user
        self._api.grant_access(self.repo_id, OTHER_USER)

        # Cannot grant twice
        with self.assertRaises(HTTPError):
            self._api.grant_access(self.repo_id, OTHER_USER)

        # Cannot accept to already accepted
        with self.assertRaises(HTTPError):
            self._api.accept_access_request(self.repo_id, OTHER_USER)

        # Cannot reject to already rejected
        self._api.reject_access_request(self.repo_id, OTHER_USER)
        with self.assertRaises(HTTPError):
            self._api.reject_access_request(self.repo_id, OTHER_USER)

        # Cannot cancel to already cancelled
        self._api.cancel_access_request(self.repo_id, OTHER_USER)
        with self.assertRaises(HTTPError):
            self._api.cancel_access_request(self.repo_id, OTHER_USER)


@with_production_testing
class UserApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = HfApi()  # no auth!

    def test_user_overview(self) -> None:
        overview = self.api.get_user_overview("julien-c")
        self.assertEqual(overview.user_type, "user")
        self.assertGreater(overview.num_likes, 10)
        self.assertGreater(overview.num_upvotes, 10)

    def test_organization_members(self) -> None:
        members = self.api.list_organization_members("huggingface")
        self.assertGreater(len(list(members)), 1)

    def test_user_followers(self) -> None:
        followers = self.api.list_user_followers("julien-c")
        self.assertGreater(len(list(followers)), 10)

    def test_user_following(self) -> None:
        following = self.api.list_user_following("julien-c")
        self.assertGreater(len(list(following)), 10)


class WebhookApiTest(HfApiCommonTest):
    def setUp(self) -> None:
        super().setUp()
        self.webhook_url = "https://webhook.site/test"
        self.watched_items = [
            WebhookWatchedItem(type="user", name="julien-c"),  # can be either a dataclass
            {"type": "org", "name": "HuggingFaceH4"},  # or a simple dictionary
        ]
        self.domains = ["repo", "discussion"]
        self.secret = "my-secret"

        # Create a webhook to be used in the tests
        self.webhook = self._api.create_webhook(
            url=self.webhook_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )

    def tearDown(self) -> None:
        # Clean up the created webhook
        self._api.delete_webhook(self.webhook.id)
        super().tearDown()

    def test_get_webhook(self) -> None:
        webhook = self._api.get_webhook(self.webhook.id)
        self.assertIsInstance(webhook, WebhookInfo)
        self.assertEqual(webhook.id, self.webhook.id)
        self.assertEqual(webhook.url, self.webhook_url)

    def test_list_webhooks(self) -> None:
        webhooks = self._api.list_webhooks()
        self.assertTrue(any(webhook.id == self.webhook.id for webhook in webhooks))

    def test_create_webhook(self) -> None:
        new_webhook = self._api.create_webhook(
            url=self.webhook_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        self.assertIsInstance(new_webhook, WebhookInfo)
        self.assertEqual(new_webhook.url, self.webhook_url)

        # Clean up the newly created webhook
        self._api.delete_webhook(new_webhook.id)

    def test_update_webhook(self) -> None:
        updated_url = "https://webhook.site/new"
        updated_webhook = self._api.update_webhook(
            self.webhook.id, url=updated_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        self.assertEqual(updated_webhook.url, updated_url)

    def test_enable_webhook(self) -> None:
        enabled_webhook = self._api.enable_webhook(self.webhook.id)
        self.assertFalse(enabled_webhook.disabled)

    def test_disable_webhook(self) -> None:
        disabled_webhook = self._api.disable_webhook(self.webhook.id)
        self.assertTrue(disabled_webhook.disabled)

    def test_delete_webhook(self) -> None:
        # Create another webhook to test deletion
        webhook_to_delete = self._api.create_webhook(
            url=self.webhook_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        self._api.delete_webhook(webhook_to_delete.id)
        with self.assertRaises(HTTPError):
            self._api.get_webhook(webhook_to_delete.id)
