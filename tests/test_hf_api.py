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
import time
import types
import uuid
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import fields
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, get_args
from unittest.mock import Mock, patch
from urllib.parse import urlparse

import pytest

from huggingface_hub import HfApi, SpaceHardware, SpaceStage, SpaceStorage, constants
from huggingface_hub._commit_api import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    _fetch_upload_modes,
)
from huggingface_hub.community import DiscussionComment, DiscussionWithDetails
from huggingface_hub.errors import (
    BadRequestError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import (
    AccessRequest,
    Collection,
    CommitInfo,
    DatasetInfo,
    DatasetLeaderboardEntry,
    ExpandDatasetProperty_T,
    ExpandModelProperty_T,
    ExpandSpaceProperty_T,
    InferenceEndpoint,
    InferenceProviderMapping,
    ModelInfo,
    Organization,
    RepoSibling,
    RepoUrl,
    SpaceInfo,
    SpaceRuntime,
    SpaceSearchResult,
    User,
    WebhookInfo,
    WebhookWatchedItem,
    repo_type_and_id_from_hf_id,
)
from huggingface_hub.repocard_data import DatasetCardData, ModelCardData
from huggingface_hub.utils import (
    NotASafetensorsRepoError,
    SafetensorsFileMetadata,
    SafetensorsParsingError,
    SafetensorsRepoMetadata,
    SoftTemporaryDirectory,
    TensorInfo,
    get_session,
    hf_raise_for_status,
    logging,
)
from huggingface_hub.utils.endpoint_helpers import _is_emission_within_threshold

from .conftest import RepoFactory
from .testing_constants import (
    DUMMY_DATASET_ID,
    DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT,
    DUMMY_MODEL_ID,
    DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
    ENDPOINT_PRODUCTION,
    ENDPOINT_STAGING,
    ENTERPRISE_ORG,
    ENTERPRISE_TOKEN,
    FULL_NAME,
    OTHER_TOKEN,
    OTHER_USER,
    SAMPLE_DATASET_IDENTIFIER,
    TOKEN,
    USER,
)
from .testing_utils import repo_name


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


class TestHfApiRepoFileExists:
    @pytest.fixture(autouse=True)
    def _repo(self, api: HfApi):
        self.repo_id = api.create_repo(repo_name(), private=True).repo_id
        api.upload_file(repo_id=self.repo_id, path_in_repo="file.txt", path_or_fileobj=b"content")
        yield
        api.delete_repo(self.repo_id)

    def test_repo_exists(self, api: HfApi):
        assert api.repo_exists(self.repo_id)
        assert not api.repo_exists(self.repo_id, token=False)  # private repo
        assert not api.repo_exists("repo-that-does-not-exist")  # missing repo

    def test_revision_exists(self, api: HfApi):
        assert api.revision_exists(self.repo_id, "main")
        assert not api.revision_exists(self.repo_id, "revision-that-does-not-exist")  # missing revision
        assert not api.revision_exists(self.repo_id, "main", token=False)  # private repo
        assert not api.revision_exists("repo-that-does-not-exist", "main")  # missing repo

    def test_file_exists(self, api: HfApi, monkeypatch):
        monkeypatch.setattr(constants, "ENDPOINT", "https://hub-ci.huggingface.co")
        monkeypatch.setattr(
            constants,
            "HUGGINGFACE_CO_URL_TEMPLATE",
            "https://hub-ci.huggingface.co/{repo_id}/resolve/{revision}/{filename}",
        )
        assert api.file_exists(self.repo_id, "file.txt")
        assert not api.file_exists("repo-that-does-not-exist", "file.txt")  # missing repo
        assert not api.file_exists(self.repo_id, "file-does-not-exist")  # missing file
        assert not api.file_exists(
            self.repo_id, "file.txt", revision="revision-that-does-not-exist"
        )  # missing revision
        assert not api.file_exists(self.repo_id, "file.txt", token=False)  # private repo


class TestHfApiEndpoints:
    def test_whoami_with_passing_token(self, api: HfApi):
        info = api.whoami(token=TOKEN)
        assert info["name"] == USER
        assert info["fullname"] == FULL_NAME
        assert isinstance(info["orgs"], list)
        valid_org = [org for org in info["orgs"] if org["name"] == "valid_org_hub"][0]
        assert valid_org["fullname"] == "Dummy Hub Org"

    def test_whoami_with_implicit_token_from_login(self, api: HfApi, mocker) -> None:
        """Test using `whoami` after a `hf auth login`."""
        mocker.patch("huggingface_hub.hf_api.get_token", return_value=TOKEN)
        with patch.object(api, "token", None):  # no default token
            info = api.whoami()
        assert info["name"] == USER

    def test_whoami_with_implicit_token_from_hf_api(self, api: HfApi, mocker) -> None:
        """Test using `whoami` with token from the HfApi client."""
        mock_get_token = mocker.patch("huggingface_hub.utils._headers.get_token")
        info = api.whoami()
        assert info["name"] == USER
        mock_get_token.assert_not_called()

    def test_whoami_with_caching(self) -> None:
        # Don't use class instance to avoid cache sharing
        api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        assert api._whoami_cache == {}

        assert api.whoami(cache=True)["name"] == USER

        # Value in cache
        assert len(api._whoami_cache) == 1
        assert TOKEN in api._whoami_cache
        mocked_value = Mock()
        api._whoami_cache[TOKEN] = mocked_value

        # Call again => use cache
        assert api.whoami(cache=True) == mocked_value

        # Cache not shared between HfApi instances
        api_bis = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        assert api_bis._whoami_cache == {}
        assert api_bis.whoami(cache=True)["name"] == USER

    def test_whoami_rate_limit_suggest_caching(self, api: HfApi) -> None:
        with patch("huggingface_hub.hf_api.hf_raise_for_status") as mock:
            mock.side_effect = HfHubHTTPError(message="Fake error.", response=Mock(status_code=429))
            with pytest.raises(
                HfHubHTTPError, match=r".*consider caching the response with `whoami\(..., cache=True\)`.*"
            ):
                api.whoami()

    def test_whoami_with_token_false(self, api: HfApi):
        """Test that using `token=False` raises an error.

        Regression test for https://github.com/huggingface/huggingface_hub/pull/3568#discussion_r2557248898.

        Before the fix, local token was used even when `token=False` was passed (which is not intended).
        """
        with pytest.raises(ValueError):
            api.whoami(token=False)

        with pytest.raises(ValueError):
            HfApi(token=False).whoami()

    def test_delete_repo_error_message(self, api: HfApi):
        # test for #751
        # See https://github.com/huggingface/huggingface_hub/issues/751
        with pytest.raises(
            HfHubHTTPError,
            match=re.compile(
                r"404 Client Error(.+)\(Request ID: .+\)(.*)Repository Not Found",
                flags=re.DOTALL,
            ),
        ):
            api.delete_repo("repo-that-does-not-exist")

    def test_delete_repo_missing_ok(self, api: HfApi) -> None:
        api.delete_repo("repo-that-does-not-exist", missing_ok=True)

    def test_move_repo_normal_usage(self, api: HfApi):
        # Spaces not tested on staging (error 500)
        for repo_type in [None, constants.REPO_TYPE_MODEL, constants.REPO_TYPE_DATASET]:
            repo_id = f"{USER}/{repo_name()}"
            new_repo_id = f"{USER}/{repo_name()}"
            api.create_repo(repo_id=repo_id, repo_type=repo_type)
            api.move_repo(from_id=repo_id, to_id=new_repo_id, repo_type=repo_type)
            api.delete_repo(repo_id=new_repo_id, repo_type=repo_type)

    def test_move_repo_target_already_exists(self, api: HfApi) -> None:
        repo_id_1 = f"{USER}/{repo_name()}"
        repo_id_2 = f"{USER}/{repo_name()}"

        api.create_repo(repo_id=repo_id_1)
        api.create_repo(repo_id=repo_id_2)

        with pytest.raises(HfHubHTTPError, match=r"A model repository called .* already exists"):
            api.move_repo(from_id=repo_id_1, to_id=repo_id_2, repo_type=constants.REPO_TYPE_MODEL)

        api.delete_repo(repo_id=repo_id_1)
        api.delete_repo(repo_id=repo_id_2)

    def test_move_repo_invalid_repo_id(self, api: HfApi) -> None:
        """Test from_id and to_id must be in the form `"namespace/repo_name"`."""
        with pytest.raises(ValueError, match=r"Invalid repo_id*"):
            api.move_repo(from_id="namespace/repo_name", to_id="invalid_repo_id")

        with pytest.raises(ValueError, match=r"Invalid repo_id*"):
            api.move_repo(from_id="invalid_repo_id", to_id="namespace/repo_name")

    def test_update_repo_settings(self, api: HfApi, repo_factory: RepoFactory):
        repo_url = repo_factory("model")
        repo_id = repo_url.repo_id

        for gated_value in ["auto", "manual", False]:
            for private_value in [True, False]:  # Test both private and public settings
                api.update_repo_settings(repo_id=repo_id, gated=gated_value, private=private_value)
                info = api.model_info(repo_id)
                assert info.gated == gated_value
                assert info.private == private_value  # Verify the private setting

    def test_update_dataset_repo_settings(self, api: HfApi, repo_factory: RepoFactory):
        repo_url = repo_factory("dataset")
        repo_id = repo_url.repo_id
        repo_type = repo_url.repo_type

        for gated_value in ["auto", "manual", False]:
            for private_value in [True, False]:
                api.update_repo_settings(
                    repo_id=repo_id, repo_type=repo_type, gated=gated_value, private=private_value
                )
                info = api.dataset_info(repo_id)
                assert info.gated == gated_value
                assert info.private == private_value


class TestCommitApi:
    @pytest.fixture(autouse=True)
    def _tmp_files(self, tmp_path: Path):
        self.tmp_dir = str(tmp_path)
        self.tmp_file = os.path.join(self.tmp_dir, "temp")
        self.tmp_file_content = "Content of the file"
        with open(self.tmp_file, "w+") as f:
            f.write(self.tmp_file_content)
        os.makedirs(os.path.join(self.tmp_dir, "nested"))
        self.nested_tmp_file = os.path.join(self.tmp_dir, "nested", "file.bin")
        with open(self.nested_tmp_file, "wb+") as f:
            f.truncate(1024 * 1024)

    def test_upload_file_validation(self, api: HfApi) -> None:
        with pytest.raises(ValueError):
            api.upload_file(
                path_or_fileobj=self.tmp_file,
                path_in_repo="README.md",
                repo_id="something",
                repo_type="this type does not exist",
            )

    def test_commit_operation_validation(self):
        with open(self.tmp_file, "rt") as ftext:
            with pytest.raises(ValueError):
                CommitOperationAdd(path_or_fileobj=ftext, path_in_repo="README.md")  # type: ignore

        with pytest.raises(ValueError):
            CommitOperationAdd(
                path_or_fileobj=os.path.join(self.tmp_dir, "nofile.pth"),
                path_in_repo="README.md",
            )

    def test_upload_file_str_path(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        return_val = api.upload_file(
            path_or_fileobj=self.tmp_file,
            path_in_repo="temp/new_file.md",
            repo_id=repo_id,
        )
        assert isinstance(return_val, CommitInfo)
        assert return_val.startswith(f"{repo_url}/commit/")

        with SoftTemporaryDirectory() as cache_dir:
            with open(hf_hub_download(repo_id=repo_id, filename="temp/new_file.md", cache_dir=cache_dir)) as f:
                assert f.read() == self.tmp_file_content

    def test_upload_file_pathlib_path(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Regression test for https://github.com/huggingface/huggingface_hub/issues/1246."""
        repo_url = repo_factory()
        api.upload_file(path_or_fileobj=Path(self.tmp_file), path_in_repo="file.txt", repo_id=repo_url.repo_id)
        assert "file.txt" in api.list_repo_files(repo_id=repo_url.repo_id)

    def test_upload_file_fileobj(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        with open(self.tmp_file, "rb") as filestream:
            return_val = api.upload_file(
                path_or_fileobj=filestream,
                path_in_repo="temp/new_file.md",
                repo_id=repo_id,
            )
        assert isinstance(return_val, CommitInfo)
        assert return_val.startswith(f"{repo_url}/commit/")

        with SoftTemporaryDirectory() as cache_dir:
            with open(hf_hub_download(repo_id=repo_id, filename="temp/new_file.md", cache_dir=cache_dir)) as f:
                assert f.read() == self.tmp_file_content

    def test_upload_file_bytesio(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        content = BytesIO(b"File content, but in bytes IO")
        return_val = api.upload_file(
            path_or_fileobj=content,
            path_in_repo="temp/new_file.md",
            repo_id=repo_id,
        )
        assert isinstance(return_val, CommitInfo)
        assert return_val.startswith(f"{repo_url}/commit/")

        with SoftTemporaryDirectory() as cache_dir:
            with open(hf_hub_download(repo_id=repo_id, filename="temp/new_file.md", cache_dir=cache_dir)) as f:
                assert f.read() == content.getvalue().decode()

    def test_upload_data_files_to_model_repo(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        # If a .parquet file is uploaded to a model repo, it should be uploaded correctly but a warning is raised.
        with pytest.warns(UserWarning) as cm:
            api.upload_file(
                path_or_fileobj=b"content",
                path_in_repo="data.parquet",
                repo_id=repo_url.repo_id,
            )
        assert (
            cm[0].message.args[0]
            == "It seems that you are about to commit a data file (data.parquet) to a model repository. You are sure this is intended? If you are trying to upload a dataset, please set `repo_type='dataset'` or `--repo-type=dataset` in a CLI."
        )

        # Same for arrow file
        with pytest.warns(UserWarning):
            api.upload_file(
                path_or_fileobj=b"content",
                path_in_repo="data.arrow",
                repo_id=repo_url.repo_id,
            )

        # Still correctly uploaded
        files = api.list_repo_files(repo_url.repo_id)
        assert "data.parquet" in files
        assert "data.arrow" in files

    def test_create_repo_return_value(self, api: HfApi) -> None:
        REPO_NAME = repo_name("org")
        url = api.create_repo(repo_id=REPO_NAME)
        assert isinstance(url, str)
        assert isinstance(url, RepoUrl)
        assert url.repo_id == f"{USER}/{REPO_NAME}"
        api.delete_repo(repo_id=url.repo_id)

    def test_create_repo_already_exists_but_no_write_permission(self, api: HfApi):
        # Create under other user namespace
        repo_id = api.create_repo(repo_id=repo_name(), token=OTHER_TOKEN).repo_id

        # Try to create with our namespace -> should not fail as the repo already exists
        api.create_repo(repo_id=repo_id, token=TOKEN, exist_ok=True)

        # Clean up
        api.delete_repo(repo_id=repo_id, token=OTHER_TOKEN)

    def test_create_repo_already_exists_but_no_write_permission_returns_correct_repo_id(self, api: HfApi):
        """Regression test for https://github.com/huggingface/huggingface_hub/issues/3632."""
        # Create dataset under other user namespace
        repo_id = api.create_repo(repo_id=repo_name(), repo_type="dataset", token=OTHER_TOKEN).repo_id

        # Try to create with our token -> triggers 403 fallback path
        returned_url = api.create_repo(repo_id=repo_id, repo_type="dataset", token=TOKEN, exist_ok=True)

        # Verify the returned RepoUrl has the correct repo_id
        assert returned_url.repo_id == repo_id
        assert returned_url.repo_type == "dataset"

        # Clean up
        api.delete_repo(repo_id=repo_id, repo_type="dataset", token=OTHER_TOKEN)

    def test_create_repo_private_by_default(self, api: HfApi):
        """Enterprise Hub allows creating private repos by default. Let's test that."""
        repo_id = f"{ENTERPRISE_ORG}/{repo_name()}"
        api.create_repo(repo_id, token=ENTERPRISE_TOKEN)
        info = api.model_info(repo_id, token=ENTERPRISE_TOKEN, expand="private")
        assert info.private

        api.delete_repo(repo_id, token=ENTERPRISE_TOKEN)

    def test_create_repo_with_visibility(self, api: HfApi):
        repo_id = repo_name()
        url = api.create_repo(repo_id, visibility="private")
        info = api.model_info(url.repo_id, expand="private")
        assert info.private
        api.delete_repo(url.repo_id)

    def test_update_repo_settings_with_visibility(self, api: HfApi, repo_factory: RepoFactory):
        repo_url = repo_factory("model")
        repo_id = repo_url.repo_id
        api.update_repo_settings(repo_id=repo_id, visibility="private")
        info = api.model_info(repo_id, expand="private")
        assert info.private

        api.update_repo_settings(repo_id=repo_id, visibility="public")
        info = api.model_info(repo_id, expand="private")
        assert not info.private

    def test_upload_file_create_pr(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        return_val = api.upload_file(
            path_or_fileobj=self.tmp_file_content.encode(),
            path_in_repo="temp/new_file.md",
            repo_id=repo_id,
            create_pr=True,
        )
        assert isinstance(return_val, CommitInfo)
        assert return_val.startswith(f"{repo_url}/commit/")
        assert return_val.pr_revision == "refs/pr/1"

        with SoftTemporaryDirectory() as cache_dir:
            with open(
                hf_hub_download(
                    repo_id=repo_id, filename="temp/new_file.md", revision="refs/pr/1", cache_dir=cache_dir
                )
            ) as f:
                assert f.read() == self.tmp_file_content

    def test_delete_file(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        api.upload_file(
            path_or_fileobj=self.tmp_file,
            path_in_repo="temp/new_file.md",
            repo_id=repo_url.repo_id,
        )
        return_val = api.delete_file(path_in_repo="temp/new_file.md", repo_id=repo_url.repo_id)
        assert isinstance(return_val, CommitInfo)

        with pytest.raises(EntryNotFoundError):
            # Should raise a 404
            hf_hub_download(repo_url.repo_id, "temp/new_file.md")

    def test_get_full_repo_name(self, api: HfApi):
        repo_name_with_no_org = api.get_full_repo_name("model")
        assert repo_name_with_no_org == f"{USER}/model"

        repo_name_with_no_org = api.get_full_repo_name("model", organization="org")
        assert repo_name_with_no_org == "org/model"

    def test_upload_folder(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id

        # Upload folder
        url = api.upload_folder(folder_path=self.tmp_dir, path_in_repo="temp/dir", repo_id=repo_id)
        assert isinstance(url, CommitInfo)
        assert url.startswith(f"{repo_url}/commit/")

        # Check files are uploaded
        for rpath in ["temp", "nested/file.bin"]:
            local_path = os.path.join(self.tmp_dir, rpath)
            remote_path = f"temp/dir/{rpath}"
            filepath = hf_hub_download(repo_id=repo_id, filename=remote_path, revision="main", token=TOKEN)
            assert filepath is not None
            with open(filepath, "rb") as downloaded_file:
                content = downloaded_file.read()
            with open(local_path, "rb") as local_file:
                expected_content = local_file.read()
            assert content == expected_content

        # Re-uploading the same folder twice should be fine
        return_val = api.upload_folder(folder_path=self.tmp_dir, path_in_repo="temp/dir", repo_id=repo_id)
        assert isinstance(return_val, CommitInfo)

    def test_upload_folder_create_pr(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id

        # Upload folder as a new PR
        return_val = api.upload_folder(
            folder_path=self.tmp_dir, path_in_repo="temp/dir", repo_id=repo_id, create_pr=True
        )
        assert isinstance(return_val, CommitInfo)
        assert return_val.startswith(f"{repo_url}/commit/")
        assert return_val.pr_revision == "refs/pr/1"

        # Check files are uploaded
        for rpath in ["temp", "nested/file.bin"]:
            local_path = os.path.join(self.tmp_dir, rpath)
            filepath = hf_hub_download(repo_id=repo_id, filename=f"temp/dir/{rpath}", revision="refs/pr/1")
            assert Path(local_path).read_bytes() == Path(filepath).read_bytes()

    def test_upload_folder_git_folder_excluded(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()

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
        api.upload_folder(folder_path=self.tmp_dir, repo_id=repo_url.repo_id)
        assert set(api.list_repo_files(repo_id=repo_url.repo_id)) == {
            ".gitattributes",
            ".git_something/file.txt",
            "file.git",
            "temp",
            "nested/file.bin",
        }

    def test_upload_folder_gitignore_already_exists(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        # Ignore nested folder
        api.upload_file(path_or_fileobj=b"nested/*\n", path_in_repo=".gitignore", repo_id=repo_url.repo_id)

        # Upload folder
        api.upload_folder(folder_path=self.tmp_dir, repo_id=repo_url.repo_id)

        # Check nested file not uploaded
        assert not api.file_exists(repo_url.repo_id, "nested/file.bin")

    def test_upload_folder_gitignore_in_commit(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        # Create .gitignore file locally
        (Path(self.tmp_dir) / ".gitignore").write_text("nested/*\n")

        # Upload folder
        api.upload_folder(folder_path=self.tmp_dir, repo_id=repo_url.repo_id)

        # Check nested file not uploaded
        assert not api.file_exists(repo_url.repo_id, "nested/file.bin")

    def test_create_commit_create_pr(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id

        # Upload a first file
        api.upload_file(path_or_fileobj=self.tmp_file, path_in_repo="temp/new_file.md", repo_id=repo_id)

        # Create a commit with a PR
        operations = [
            CommitOperationDelete(path_in_repo="temp/new_file.md"),
            CommitOperationAdd(path_in_repo="buffer", path_or_fileobj=b"Buffer data"),
        ]
        resp = api.create_commit(
            operations=operations, commit_message="Test create_commit", repo_id=repo_id, create_pr=True
        )

        # Check commit info
        assert isinstance(resp, CommitInfo)
        commit_id = resp.oid
        assert "pr_revision='refs/pr/1'" in repr(resp)
        assert isinstance(commit_id, str)
        assert len(commit_id) > 0
        assert resp.commit_url == f"{api.endpoint}/{repo_id}/commit/{commit_id}"
        assert resp.commit_message == "Test create_commit"
        assert resp.commit_description == ""
        assert resp.pr_url == f"{api.endpoint}/{repo_id}/discussions/1"
        assert resp.pr_num == 1
        assert resp.pr_revision == "refs/pr/1"

        # File doesn't exist on main...
        with pytest.raises(HfHubHTTPError) as ctx:
            # Should raise a 404
            api.hf_hub_download(repo_id, "buffer")
            assert ctx.value.response.status_code == 404

        # ...but exists on PR
        filepath = api.hf_hub_download(filename="buffer", repo_id=repo_id, revision="refs/pr/1")
        with open(filepath, "rb") as downloaded_file:
            content = downloaded_file.read()
        assert content == b"Buffer data"

    def test_create_commit_create_pr_against_branch(self, api: HfApi):
        repo_id = f"{USER}/{repo_name()}"

        # Create repo and create a non-main branch
        api.create_repo(repo_id=repo_id, exist_ok=False)
        api.create_branch(repo_id=repo_id, branch="test_branch")
        head = api.list_repo_refs(repo_id=repo_id).branches[0].target_commit

        # Create PR against non-main branch works
        resp = api.create_commit(
            operations=[
                CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
            ],
            commit_message="PR against existing branch",
            repo_id=repo_id,
            revision="test_branch",
            create_pr=True,
        )
        assert isinstance(resp, CommitInfo)

        # Create PR against a oid fails
        with pytest.raises(RevisionNotFoundError):
            api.create_commit(
                operations=[
                    CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
                ],
                commit_message="PR against a oid",
                repo_id=repo_id,
                revision=head,
                create_pr=True,
            )

        # Create PR against a non-existing branch fails
        with pytest.raises(RevisionNotFoundError):
            api.create_commit(
                operations=[
                    CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
                ],
                commit_message="PR against missing branch",
                repo_id=repo_id,
                revision="missing_branch",
                create_pr=True,
            )

        # Cleanup
        api.delete_repo(repo_id=repo_id)

    def test_create_commit_create_pr_on_foreign_repo(self, api: HfApi):
        # Create a repo with another user. The normal CI user don't have rights on it.
        # We must be able to create a PR on it
        foreign_api = HfApi(token=OTHER_TOKEN)
        foreign_repo_url = foreign_api.create_repo(repo_id=repo_name("repo-for-hfh-ci"))

        api.create_commit(
            operations=[
                CommitOperationAdd(path_in_repo="regular.txt", path_or_fileobj=b"File content"),
                CommitOperationAdd(path_in_repo="lfs.pkl", path_or_fileobj=b"File content"),
            ],
            commit_message="PR on foreign repo",
            repo_id=foreign_repo_url.repo_id,
            create_pr=True,
        )

        foreign_api.delete_repo(repo_id=foreign_repo_url.repo_id)

    def test_create_commit(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        api.upload_file(path_or_fileobj=self.tmp_file, path_in_repo="temp/new_file.md", repo_id=repo_id)
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
            resp = api.create_commit(operations=operations, commit_message="Test create_commit", repo_id=repo_id)
            # Check commit info
            assert isinstance(resp, CommitInfo)
            assert resp.pr_url is None  # No pr created
            assert resp.pr_num is None
            assert resp.pr_revision is None

        with pytest.raises(HfHubHTTPError):
            # Should raise a 404
            hf_hub_download(repo_id, "temp/new_file.md")

        for path, expected_content in [
            ("buffer", b"Buffer data"),
            ("bytesio", b"BytesIO data"),
            ("fileobj", self.tmp_file_content.encode()),
            ("nested/path", self.tmp_file_content.encode()),
        ]:
            filepath = hf_hub_download(repo_id=repo_id, filename=path, revision="main")
            assert filepath is not None
            with open(filepath, "rb") as downloaded_file:
                content = downloaded_file.read()
            assert content == expected_content

    def test_create_commit_conflict(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        # Get commit on main
        repo_id = repo_url.repo_id
        parent_commit = api.model_info(repo_id).sha

        # Upload new file
        api.upload_file(path_or_fileobj=self.tmp_file, path_in_repo="temp/new_file.md", repo_id=repo_id)

        # Creating a commit with a parent commit that is not the current main should fail
        operations = [
            CommitOperationAdd(path_in_repo="buffer", path_or_fileobj=b"Buffer data"),
        ]
        with pytest.raises(HfHubHTTPError) as exc_ctx:
            api.create_commit(
                operations=operations,
                commit_message="Test create_commit",
                repo_id=repo_id,
                parent_commit=parent_commit,
            )
        assert exc_ctx.value.response.status_code == 412
        assert "The branch was updated since you opened this page. Please refresh and try again." in str(exc_ctx.value)

    def test_create_commit_repo_does_not_exist(self, api: HfApi) -> None:
        """Test error message is detailed when creating a commit on a missing repo."""
        with pytest.raises(RepositoryNotFoundError) as context:
            api.create_commit(
                repo_id=f"{USER}/repo_that_do_not_exist",
                operations=[CommitOperationAdd("config.json", b"content")],
                commit_message="fake_message",
            )

        request_id = context.value.response.headers.get("X-Request-Id")
        expected_message = (
            f"404 Client Error. (Request ID: {request_id})\n\nRepository Not"
            " Found for url:"
            f" {api.endpoint}/api/models/{USER}/repo_that_do_not_exist/preupload/main.\nPlease"
            " make sure you specified the correct `repo_id` and"
            " `repo_type`.\nIf you are trying to access a private or gated"
            " repo, make sure you are authenticated and your token has the required permissions."
            "\nFor more details, see https://huggingface.co/docs/huggingface_hub/authentication"
            "\nNote: Creating a commit assumes that the repo already exists on the Huggingface Hub."
            " Please use `create_repo` if it's not the case."
        )

        assert str(context.value) == expected_message

    def test_create_commit_lfs_file_implicit_token(self, api: HfApi, mocker) -> None:
        """Test that uploading a file as LFS works with cached token.

        Regression test for https://github.com/huggingface/huggingface_hub/pull/1084.
        """
        mocker.patch("huggingface_hub.utils._headers.get_token", return_value=TOKEN)
        REPO_NAME = repo_name("create_commit_with_lfs")
        repo_id = f"{USER}/{REPO_NAME}"

        with patch.object(api, "token", None):  # no default token
            # Create repo
            api.create_repo(repo_id=REPO_NAME, exist_ok=False)

            # Set repo to track png files as LFS
            api.create_commit(
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
            api.create_commit(
                operations=[
                    CommitOperationAdd(path_in_repo="image.png", path_or_fileobj=b"image data"),
                ],
                commit_message="Test upload lfs file",
                repo_id=repo_id,
            )

            # Check uploaded as LFS
            info = api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = {file.rfilename: file for file in info.siblings}
            assert isinstance(siblings["image.png"].lfs, dict)  # LFS file

            # Delete repo
            api.delete_repo(repo_id=REPO_NAME)

    def test_create_commit_huge_regular_files(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test committing 12 text files (>100MB in total) at once.

        This was not possible when using `json` format instead of `ndjson`
        on the `/create-commit` endpoint.

        See https://github.com/huggingface/huggingface_hub/pull/1117.
        """
        repo_url = repo_factory()
        operations = [
            CommitOperationAdd(
                path_in_repo=f"file-{num}.text",
                path_or_fileobj=b"Hello regular " + b"a" * 1024 * 1024 * 9,
            )
            for num in range(12)
        ]
        api.create_commit(
            operations=operations,  # 12*9MB regular => too much for "old" method
            commit_message="Test create_commit with huge regular files",
            repo_id=repo_url.repo_id,
        )

    def test_commit_preflight_on_lots_of_lfs_files(self, api: HfApi, repo_factory: RepoFactory):
        """Test committing 1300 LFS files at once.

        This was not possible when `_fetch_upload_modes` was not fetching metadata by
        chunks. We are not testing the full upload as it would require to upload 1300
        files which is unnecessary for the test. Having an overall large payload (for
        `/create-commit` endpoint) is tested in `test_create_commit_huge_regular_files`.

        There is also a 25k LFS files limit on the Hub but this is not tested.

        See https://github.com/huggingface/huggingface_hub/pull/1117.
        """
        repo_url = repo_factory()
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
            headers=api._build_hf_headers(),
            revision="main",
            endpoint=ENDPOINT_STAGING,
        )
        for operation in operations:
            assert operation._upload_mode == "lfs"
            assert not operation._is_committed
            assert not operation._is_uploaded

    def test_create_commit_repo_id_case_insensitive(self, api: HfApi):
        """Test create commit but repo_id is lowercased.

        Regression test for #1371. Hub API is already case-insensitive. Somehow the issue was with the `requests`
        streaming implementation when generating the ndjson payload "on the fly". It seems that the server was
        receiving only the first line which causes a confusing "400 Bad Request - Add a line with the key `lfsFile`,
        `file` or `deletedFile`". Passing raw bytes instead of a generator fixes the problem.

        See https://github.com/huggingface/huggingface_hub/issues/1371.
        """
        REPO_NAME = repo_name("CaSe_Is_ImPoRtAnT")
        repo_id = api.create_repo(repo_id=REPO_NAME, exist_ok=False).repo_id

        api.create_commit(
            repo_id=repo_id.lower(),  # API is case-insensitive!
            commit_message="Add 1 regular and 1 LFs files.",
            operations=[
                CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
                CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"LFS content"),
            ],
        )
        repo_files = api.list_repo_files(repo_id=repo_id)
        assert "file.txt" in repo_files
        assert "lfs.bin" in repo_files

    def test_create_commit_mutates_operations(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id

        operations = [
            CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"content"),
            CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
        ]
        api.create_commit(
            repo_id=repo_id,
            commit_message="Copy LFS file.",
            operations=operations,
        )

        assert operations[0]._is_committed
        assert operations[0]._is_uploaded  # LFS file
        assert operations[0].path_or_fileobj == b"content"  # not removed by default
        assert operations[1]._is_committed
        assert operations[1].path_or_fileobj == b"content"

    def test_pre_upload_before_commit(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id

        operations = [
            CommitOperationAdd(path_in_repo="lfs.bin", path_or_fileobj=b"content1"),
            CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"content"),
            CommitOperationAdd(path_in_repo="lfs2.bin", path_or_fileobj=b"content2"),
            CommitOperationAdd(path_in_repo="file2.txt", path_or_fileobj=b"content"),
        ]

        # First: preupload 1 by 1
        for operation in operations:
            api.preupload_lfs_files(repo_id, [operation])
        assert operations[0]._is_uploaded
        assert operations[0].path_or_fileobj == b""  # Freed memory
        assert operations[2]._is_uploaded
        assert operations[2].path_or_fileobj == b""  # Freed memory

        # create commit and capture debug logs
        with caplog.at_level("DEBUG", logger="huggingface_hub"):
            api.create_commit(
                repo_id=repo_id,
                commit_message="Copy LFS file.",
                operations=operations,
            )

        # No LFS files uploaded during commit
        assert any("No LFS files to upload." in record.message for record in caplog.records)

    def test_commit_modelcard_invalid_metadata(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        with patch.object(api, "preupload_lfs_files") as mock:
            with pytest.raises(ValueError, match="Invalid metadata in README.md"):
                api.create_commit(
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

    def test_commit_modelcard_empty_metadata(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        modelcard = "This is a modelcard without metadata"
        with pytest.warns(UserWarning, match="Warnings while validating metadata in README.md"):
            commit = api.create_commit(
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

    def test_create_file_with_relative_path(self, api: HfApi):
        """Creating a file with a relative path_in_repo is forbidden.

        Previously taken from a regression test for HackerOne report 1928845. The bug enabled attackers to create files
        outside of the local dir if users downloaded a file with a relative path_in_repo on Windows.

        This is not relevant anymore as the API now forbids such paths.
        """
        repo_id = api.create_repo(repo_id=repo_name()).repo_id
        with pytest.raises(HfHubHTTPError) as cm:
            api.upload_file(path_or_fileobj=b"content", path_in_repo="..\\ddd", repo_id=repo_id)
        assert cm.value.response.status_code == 422

    def test_prevent_empty_commit_if_no_op(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        with caplog.at_level("INFO", logger="huggingface_hub"):
            api.create_commit(repo_id=repo_url.repo_id, commit_message="Empty commit", operations=[])
        records = [record for record in caplog.records if record.name.startswith("huggingface_hub")]
        assert records[0].message == "No files have been modified since last commit. Skipping to prevent empty commit."
        assert records[0].levelname == "WARNING"

    def test_prevent_empty_commit_if_no_new_addition(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        api.create_commit(
            repo_id=repo_url.repo_id,
            commit_message="initial commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"Regular file content", path_in_repo="file.txt"),
                CommitOperationAdd(path_or_fileobj=b"LFS content", path_in_repo="lfs.bin"),
            ],
        )
        with caplog.at_level("INFO", logger="huggingface_hub"):
            api.create_commit(
                repo_id=repo_url.repo_id,
                commit_message="Empty commit",
                operations=[
                    CommitOperationAdd(path_or_fileobj=b"Regular file content", path_in_repo="file.txt"),
                    CommitOperationAdd(path_or_fileobj=b"LFS content", path_in_repo="lfs.bin"),
                ],
            )
        records = [record for record in caplog.records if record.name.startswith("huggingface_hub")]
        assert records[0].message == "Removing 2 file(s) from commit that have not changed."
        assert records[0].levelname == "INFO"

        assert records[1].message == "No files have been modified since last commit. Skipping to prevent empty commit."
        assert records[1].levelname == "WARNING"

    def test_prevent_empty_commit_if_no_new_copy(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        # Add 2 regular identical files and 2 LFS identical files
        api.create_commit(
            repo_id=repo_url.repo_id,
            commit_message="initial commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"Regular file content", path_in_repo="file.txt"),
                CommitOperationAdd(path_or_fileobj=b"Regular file content", path_in_repo="file_copy.txt"),
                CommitOperationAdd(path_or_fileobj=b"LFS content", path_in_repo="lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"LFS content", path_in_repo="lfs_copy.bin"),
            ],
        )
        with caplog.at_level("INFO", logger="huggingface_hub"):
            api.create_commit(
                repo_id=repo_url.repo_id,
                commit_message="Empty commit",
                operations=[
                    CommitOperationCopy(src_path_in_repo="file.txt", path_in_repo="file_copy.txt"),
                    CommitOperationCopy(src_path_in_repo="lfs.bin", path_in_repo="lfs_copy.bin"),
                ],
            )
        records = [record for record in caplog.records if record.name.startswith("huggingface_hub")]
        assert records[0].message == "Removing 2 file(s) from commit that have not changed."
        assert records[0].levelname == "INFO"

        assert records[1].message == "No files have been modified since last commit. Skipping to prevent empty commit."
        assert records[1].levelname == "WARNING"

    def test_empty_commit_on_pr(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        """
        Regression test for #2411. Revision was quoted twice, leading to a HTTP 404.

        See https://github.com/huggingface/huggingface_hub/issues/2411.
        """
        repo_url = repo_factory()
        pr = api.create_pull_request(repo_id=repo_url.repo_id, title="Test PR")

        with caplog.at_level("WARNING", logger="huggingface_hub"):
            url = api.create_commit(
                repo_id=repo_url.repo_id,
                operations=[],
                commit_message="Empty commit",
                revision=pr.git_reference,
            )
        # a warning is emitted when skipping the empty commit
        assert any(record.levelname == "WARNING" for record in caplog.records)

        commits = api.list_repo_commits(repo_id=repo_url.repo_id, revision=pr.git_reference)
        assert len(commits) == 1  # no 2nd commit
        assert url.oid == commits[0].commit_id

    def test_continue_commit_without_existing_files(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        api.create_commit(
            repo_id=repo_url.repo_id,
            commit_message="initial commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file.txt"),
                CommitOperationAdd(path_or_fileobj=b"content 2.0", path_in_repo="file2.txt"),
                CommitOperationAdd(path_or_fileobj=b"LFS content 1.0", path_in_repo="lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"LFS content 2.0", path_in_repo="lfs2.bin"),
            ],
        )
        with caplog.at_level("DEBUG", logger="huggingface_hub"):
            api.create_commit(
                repo_id=repo_url.repo_id,
                commit_message="second commit",
                operations=[
                    # Did not change => will be removed from commit
                    CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file.txt"),
                    # Change => will be kept
                    CommitOperationAdd(path_or_fileobj=b"content 2.1", path_in_repo="file2.txt"),
                    # New file => will be kept
                    CommitOperationAdd(path_or_fileobj=b"content 3.0", path_in_repo="file3.txt"),
                    # Did not change => will be removed from commit
                    CommitOperationAdd(path_or_fileobj=b"LFS content 1.0", path_in_repo="lfs.bin"),
                    # Change => will be kept
                    CommitOperationAdd(path_or_fileobj=b"LFS content 2.1", path_in_repo="lfs2.bin"),
                    # New file => will be kept
                    CommitOperationAdd(path_or_fileobj=b"LFS content 3.0", path_in_repo="lfs3.bin"),
                ],
            )
        records = [record for record in caplog.records if record.name.startswith("huggingface_hub")]
        debug_logs = [record.message for record in records if record.levelname == "DEBUG"]
        info_logs = [record.message for record in records if record.levelname == "INFO"]
        warning_logs = [record.message for record in records if record.levelname == "WARNING"]

        assert "Skipping upload for 'file.txt' as the file has not changed." in debug_logs
        assert "Skipping upload for 'lfs.bin' as the file has not changed." in debug_logs
        assert "Removing 2 file(s) from commit that have not changed." in info_logs
        assert len(warning_logs) == 0  # no warnings since the commit is not empty

        paths_info = {
            item.path: item.last_commit
            for item in api.get_paths_info(
                repo_id=repo_url.repo_id,
                paths=["file.txt", "file2.txt", "file3.txt", "lfs.bin", "lfs2.bin", "lfs3.bin"],
                expand=True,
            )
        }

        # Check which files are in the last commit
        assert paths_info["file.txt"].title == "initial commit"
        assert paths_info["file2.txt"].title == "second commit"
        assert paths_info["file3.txt"].title == "second commit"
        assert paths_info["lfs.bin"].title == "initial commit"
        assert paths_info["lfs2.bin"].title == "second commit"
        assert paths_info["lfs3.bin"].title == "second commit"

    def test_continue_commit_if_copy_is_identical(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        api.create_commit(
            repo_id=repo_url.repo_id,
            commit_message="initial commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file.txt"),
                CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file_copy.txt"),
                CommitOperationAdd(path_or_fileobj=b"content 2.0", path_in_repo="file2.txt"),
                CommitOperationAdd(path_or_fileobj=b"LFS content 1.0", path_in_repo="lfs.bin"),
                CommitOperationAdd(path_or_fileobj=b"LFS content 1.0", path_in_repo="lfs_copy.bin"),
                CommitOperationAdd(path_or_fileobj=b"LFS content 2.0", path_in_repo="lfs2.bin"),
            ],
        )
        with caplog.at_level("DEBUG", logger="huggingface_hub"):
            api.create_commit(
                repo_id=repo_url.repo_id,
                commit_message="second commit",
                operations=[
                    # Did not change => will be removed from commit
                    CommitOperationCopy(src_path_in_repo="file.txt", path_in_repo="file_copy.txt"),
                    # Change => will be kept
                    CommitOperationCopy(src_path_in_repo="file2.txt", path_in_repo="file.txt"),
                    # New file => will be kept
                    CommitOperationCopy(src_path_in_repo="file2.txt", path_in_repo="file3.txt"),
                    # Did not change => will be removed from commit
                    CommitOperationCopy(src_path_in_repo="lfs.bin", path_in_repo="lfs_copy.bin"),
                    # Change => will be kept
                    CommitOperationCopy(src_path_in_repo="lfs2.bin", path_in_repo="lfs.bin"),
                    # New file => will be kept
                    CommitOperationCopy(src_path_in_repo="lfs2.bin", path_in_repo="lfs3.bin"),
                ],
            )
        records = [record for record in caplog.records if record.name.startswith("huggingface_hub")]
        debug_logs = [record.message for record in records if record.levelname == "DEBUG"]
        info_logs = [record.message for record in records if record.levelname == "INFO"]
        warning_logs = [record.message for record in records if record.levelname == "WARNING"]

        assert (
            "Skipping copy for 'file.txt' -> 'file_copy.txt' as the content of the source file is the same as the destination file."
            in debug_logs
        )
        assert (
            "Skipping copy for 'lfs.bin' -> 'lfs_copy.bin' as the content of the source file is the same as the destination file."
            in debug_logs
        )
        assert "Removing 2 file(s) from commit that have not changed." in info_logs
        assert len(warning_logs) == 0  # no warnings since the commit is not empty

        paths_info = {
            item.path: item.last_commit
            for item in api.get_paths_info(
                repo_id=repo_url.repo_id,
                paths=[
                    "file.txt",
                    "file_copy.txt",
                    "file3.txt",
                    "lfs.bin",
                    "lfs_copy.bin",
                    "lfs3.bin",
                ],
                expand=True,
            )
        }

        # Check which files are in the last commit
        assert paths_info["file.txt"].title == "second commit"
        assert paths_info["file_copy.txt"].title == "initial commit"
        assert paths_info["file3.txt"].title == "second commit"
        assert paths_info["lfs.bin"].title == "second commit"
        assert paths_info["lfs_copy.bin"].title == "initial commit"
        assert paths_info["lfs3.bin"].title == "second commit"

    def test_continue_commit_if_only_deletion(self, api: HfApi, repo_factory: RepoFactory, caplog) -> None:
        repo_url = repo_factory()
        api.create_commit(
            repo_id=repo_url.repo_id,
            commit_message="initial commit",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file.txt"),
                CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file_copy.txt"),
                CommitOperationAdd(path_or_fileobj=b"content 2.0", path_in_repo="file2.txt"),
            ],
        )
        with caplog.at_level("DEBUG", logger="huggingface_hub"):
            api.create_commit(
                repo_id=repo_url.repo_id,
                commit_message="second commit",
                operations=[
                    # Did not change => will be removed from commit
                    CommitOperationAdd(path_or_fileobj=b"content 1.0", path_in_repo="file.txt"),
                    # identical to file.txt => will be removed from commit
                    CommitOperationCopy(src_path_in_repo="file.txt", path_in_repo="file_copy.txt"),
                    # Delete operation => kept in any case
                    CommitOperationDelete(path_in_repo="file2.txt"),
                ],
            )
        records = [record for record in caplog.records if record.name.startswith("huggingface_hub")]
        debug_logs = [record.message for record in records if record.levelname == "DEBUG"]
        info_logs = [record.message for record in records if record.levelname == "INFO"]
        warning_logs = [record.message for record in records if record.levelname == "WARNING"]

        assert "Skipping upload for 'file.txt' as the file has not changed." in debug_logs
        assert (
            "Skipping copy for 'file.txt' -> 'file_copy.txt' as the content of the source file is the same as the destination file."
            in debug_logs
        )
        assert "Removing 2 file(s) from commit that have not changed." in info_logs
        assert len(warning_logs) == 0  # no warnings since the commit is not empty

        remote_files = api.list_repo_files(repo_id=repo_url.repo_id)
        assert "file.txt" in remote_files
        assert "file2.txt" not in remote_files


class TestHfApiUploadEmptyFile:
    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request, api: HfApi):
        # Create repo for all tests as they are not dependent on each other.
        repo_id = f"{USER}/{repo_name('upload_empty_file')}"
        api.create_repo(repo_id=repo_id, exist_ok=False)
        request.cls.repo_id = repo_id
        yield
        api.delete_repo(repo_id=repo_id)

    def test_upload_empty_lfs_file(self, api: HfApi) -> None:
        # Should have been an LFS file, but uploaded as regular (would fail otherwise)
        api.upload_file(repo_id=self.repo_id, path_in_repo="empty.pkl", path_or_fileobj=b"")
        info = api.repo_info(repo_id=self.repo_id, files_metadata=True)

        repo_file = {file.rfilename: file for file in info.siblings}["empty.pkl"]
        assert repo_file.size == 0
        assert repo_file.lfs is None  # As regular


class TestHfApiDeleteFolder:
    @pytest.fixture(autouse=True)
    def _repo(self, api: HfApi):
        self.repo_id = f"{USER}/{repo_name('create_commit_delete_folder')}"
        api.create_repo(repo_id=self.repo_id, exist_ok=False)

        api.create_commit(
            repo_id=self.repo_id,
            commit_message="Init repo",
            operations=[
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_1.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_2.md"),
                CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="2/file_3.md"),
            ],
        )
        yield
        api.delete_repo(repo_id=self.repo_id)

    def test_create_commit_delete_folder_implicit(self, api: HfApi):
        api.create_commit(
            operations=[CommitOperationDelete(path_in_repo="1/")],
            commit_message="Test delete folder implicit",
            repo_id=self.repo_id,
        )

        with pytest.raises(EntryNotFoundError):
            hf_hub_download(self.repo_id, "1/file_1.md", token=TOKEN)

        with pytest.raises(EntryNotFoundError):
            hf_hub_download(self.repo_id, "1/file_2.md", token=TOKEN)

        # Still exists
        hf_hub_download(self.repo_id, "2/file_3.md", token=TOKEN)

    def test_create_commit_delete_folder_explicit(self, api: HfApi):
        api.delete_folder(path_in_repo="1", repo_id=self.repo_id)
        with pytest.raises(EntryNotFoundError):
            hf_hub_download(self.repo_id, "1/file_1.md", token=TOKEN)

    def test_create_commit_implicit_delete_folder_is_ok(self, api: HfApi):
        api.create_commit(
            operations=[CommitOperationDelete(path_in_repo="1")],
            commit_message="Failing delete folder",
            repo_id=self.repo_id,
        )


def _create_nested_files_repo(api: HfApi) -> str:
    """Create a repo with a nested file structure shared by the list-files/list-tree tests."""
    repo_id = api.create_repo(repo_id=repo_name()).repo_id
    api.create_commit(
        repo_id=repo_id,
        commit_message="A first repo",
        operations=[
            CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="file.md"),
            CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="lfs.bin"),
            CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/file_1.md"),
            CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="1/2/file_1_2.md"),
            CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="2/file_2.md"),
        ],
    )
    api.create_commit(
        repo_id=repo_id,
        commit_message="Another commit",
        operations=[
            CommitOperationAdd(path_or_fileobj=b"data2", path_in_repo="3/file_3.md"),
        ],
    )
    return repo_id


class TestHfApiListRepoTree:
    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request, api: HfApi):
        repo_id = _create_nested_files_repo(api)
        request.cls.repo_id = repo_id
        yield
        api.delete_repo(repo_id=repo_id)

    def test_list_tree(self, api: HfApi):
        tree = list(api.list_repo_tree(repo_id=self.repo_id))
        assert len(tree) == 6
        assert {tree_obj.path for tree_obj in tree} == {"file.md", "lfs.bin", "1", "2", "3", ".gitattributes"}

        tree = list(api.list_repo_tree(repo_id=self.repo_id, path_in_repo="1"))
        assert len(tree) == 2
        assert {tree_obj.path for tree_obj in tree} == {"1/file_1.md", "1/2"}

    def test_list_tree_recursively(self, api: HfApi):
        tree = list(api.list_repo_tree(repo_id=self.repo_id, recursive=True))
        assert len(tree) == 11
        assert {tree_obj.path for tree_obj in tree} == {
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
        }

    def test_list_unknown_tree(self, api: HfApi):
        with pytest.raises(EntryNotFoundError):
            list(api.list_repo_tree(repo_id=self.repo_id, path_in_repo="unknown"))

    def test_list_with_empty_path(self, api: HfApi):
        assert set(tree_obj.path for tree_obj in api.list_repo_tree(repo_id=self.repo_id, path_in_repo="")) == set(
            tree_obj.path for tree_obj in api.list_repo_tree(repo_id=self.repo_id)
        )

    @pytest.mark.production
    def test_list_tree_with_expand(self):
        tree = list(
            HfApi().list_repo_tree(
                repo_id="prompthero/openjourney-v4",
                expand=True,
                revision="c9211c53404dd6f4cfac5f04f33535892260668e",
            )
        )
        assert len(tree) == 11

        # check last_commit and security are present for a file
        model_ckpt = next(tree_obj for tree_obj in tree if tree_obj.path == "openjourney-v4.ckpt")
        assert model_ckpt.last_commit is not None
        assert model_ckpt.last_commit["oid"] == "bda967fdb79a50844e4a02cccae3217a8ecc86cd"
        # `security` is computed asynchronously by the backend and may be absent from the response.
        # Only assert its structure when present to avoid flakiness.
        if model_ckpt.security is not None:
            assert model_ckpt.security["safe"]
            assert isinstance(model_ckpt.security["av_scan"], dict)  # all details in here

        # check last_commit is present for a folder
        feature_extractor = next(tree_obj for tree_obj in tree if tree_obj.path == "feature_extractor")
        assert feature_extractor.last_commit is not None
        assert feature_extractor.last_commit["oid"] == "47b62b20b20e06b9de610e840282b7e6c3d51190"

    @pytest.mark.production
    def test_list_files_without_expand(self):
        tree = list(
            HfApi().list_repo_tree(
                repo_id="prompthero/openjourney-v4",
                revision="c9211c53404dd6f4cfac5f04f33535892260668e",
            )
        )
        assert len(tree) == 11

        # check last_commit and security are missing for a file
        model_ckpt = next(tree_obj for tree_obj in tree if tree_obj.path == "openjourney-v4.ckpt")
        assert model_ckpt.last_commit is None
        assert model_ckpt.security is None

        # check last_commit is missing for a folder
        feature_extractor = next(tree_obj for tree_obj in tree if tree_obj.path == "feature_extractor")
        assert feature_extractor.last_commit is None

    @pytest.mark.production
    def test_list_tree_with_xethash(self):
        tree = list(HfApi().list_repo_tree(repo_id="openai-community/gpt2"))
        model_entry = next(tree_obj for tree_obj in tree if tree_obj.path == "model.safetensors")
        assert model_entry.xet_hash == "63bed80836ee0758c8fd4f8975d59bb0b864263ee2753547c358e8a37cde8758"


class TestHfApiTagEndpoint:
    def test_create_tag_on_main(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `create_tag` on default main branch works."""
        repo_url = repo_factory("model")
        api.create_tag(repo_url.repo_id, tag="v0", tag_message="This is a tag message.")

        # Check tag  is on `main`
        tag_info = api.model_info(repo_url.repo_id, revision="v0")
        main_info = api.model_info(repo_url.repo_id, revision="main")
        assert tag_info.sha == main_info.sha

    def test_create_tag_on_pr(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `create_tag` on a PR ref works."""
        repo_url = repo_factory("model")
        # Create a PR with a readme
        commit_info: CommitInfo = api.create_commit(
            repo_id=repo_url.repo_id,
            create_pr=True,
            commit_message="upload readme",
            operations=[CommitOperationAdd(path_or_fileobj=b"this is a file content", path_in_repo="readme.md")],
        )

        # Tag the PR
        api.create_tag(repo_url.repo_id, tag="v0", revision=commit_info.pr_revision)

        # Check tag  is on `refs/pr/1`
        tag_info = api.model_info(repo_url.repo_id, revision="v0")
        pr_info = api.model_info(repo_url.repo_id, revision=commit_info.pr_revision)
        main_info = api.model_info(repo_url.repo_id)

        assert tag_info.sha == pr_info.sha
        assert tag_info.sha != main_info.sha

    def test_create_tag_on_commit_oid(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `create_tag` on specific commit oid works (both long and shorthands).

        Test it on a `dataset` repo.
        """
        repo_url = repo_factory("dataset")
        # Create a PR with a readme
        commit_info_1: CommitInfo = api.create_commit(
            repo_id=repo_url.repo_id,
            repo_type="dataset",
            commit_message="upload readme",
            operations=[CommitOperationAdd(path_or_fileobj=b"this is a file content", path_in_repo="readme.md")],
        )
        commit_info_2: CommitInfo = api.create_commit(
            repo_id=repo_url.repo_id,
            repo_type="dataset",
            commit_message="upload config",
            operations=[CommitOperationAdd(path_or_fileobj=b"{'hello': 'world'}", path_in_repo="config.json")],
        )

        # Tag commits
        api.create_tag(
            repo_url.repo_id,
            tag="commit_1",
            repo_type="dataset",
            revision=commit_info_1.oid,  # long version
        )
        api.create_tag(
            repo_url.repo_id,
            tag="commit_2",
            repo_type="dataset",
            revision=commit_info_2.oid[:7],  # use shorthand !
        )

        # Check tags
        tag_1_info = api.dataset_info(repo_url.repo_id, revision="commit_1")
        tag_2_info = api.dataset_info(repo_url.repo_id, revision="commit_2")

        assert tag_1_info.sha == commit_info_1.oid
        assert tag_2_info.sha == commit_info_2.oid

    def test_invalid_tag_name(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `create_tag` with an invalid tag name."""
        repo_url = repo_factory("model")
        with pytest.raises(HfHubHTTPError):
            api.create_tag(repo_url.repo_id, tag="invalid tag")

    def test_create_tag_on_missing_revision(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `create_tag` on a missing revision."""
        repo_url = repo_factory("model")
        with pytest.raises(RevisionNotFoundError):
            api.create_tag(repo_url.repo_id, tag="invalid tag", revision="foobar")

    def test_create_tag_twice(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `create_tag` called twice on same tag should fail with HTTP 409."""
        repo_url = repo_factory("model")
        api.create_tag(repo_url.repo_id, tag="tag_1")
        with pytest.raises(HfHubHTTPError) as err:
            api.create_tag(repo_url.repo_id, tag="tag_1")
        assert err.value.response.status_code == 409

        # exist_ok=True => doesn't fail
        api.create_tag(repo_url.repo_id, tag="tag_1", exist_ok=True)

    def test_create_and_delete_tag(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check `delete_tag` deletes the tag."""
        repo_url = repo_factory("model")
        api.create_tag(repo_url.repo_id, tag="v0")
        api.model_info(repo_url.repo_id, revision="v0")

        api.delete_tag(repo_url.repo_id, tag="v0")
        with pytest.raises(RevisionNotFoundError):
            api.model_info(repo_url.repo_id, revision="v0")

    def test_delete_tag_missing_tag(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Check cannot `delete_tag` if tag doesn't exist."""
        repo_url = repo_factory("model")
        with pytest.raises(RevisionNotFoundError):
            api.delete_tag(repo_url.repo_id, tag="v0")

    def test_delete_tag_with_branch_name(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Try to `delete_tag` if tag is a branch name.

        Currently getting a HTTP 500.
        See https://github.com/huggingface/moon-landing/issues/4223.
        """
        repo_url = repo_factory("model")
        with pytest.raises(HfHubHTTPError):
            api.delete_tag(repo_url.repo_id, tag="main")


class TestHfApiBranchEndpoint:
    def test_create_and_delete_branch(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `create_branch` from main branch."""
        repo_url = repo_factory()
        api.create_branch(repo_url.repo_id, branch="cool-branch")

        # Check `cool-branch` branch exists
        api.model_info(repo_url.repo_id, revision="cool-branch")

        # Delete it
        api.delete_branch(repo_url.repo_id, branch="cool-branch")

        # Check doesn't exist anymore
        with pytest.raises(RevisionNotFoundError):
            api.model_info(repo_url.repo_id, revision="cool-branch")

    def test_create_branch_existing_branch_fails(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `create_branch` on existing branch."""
        repo_url = repo_factory()
        api.create_branch(repo_url.repo_id, branch="cool-branch")

        with pytest.raises(HfHubHTTPError, match=r"Reference refs/heads/cool-branch already exists"):
            api.create_branch(repo_url.repo_id, branch="cool-branch")

        with pytest.raises(HfHubHTTPError, match=r"Reference refs/heads/main already exists"):
            api.create_branch(repo_url.repo_id, branch="main")

        # exist_ok=True => doesn't fail
        api.create_branch(repo_url.repo_id, branch="cool-branch", exist_ok=True)
        api.create_branch(repo_url.repo_id, branch="main", exist_ok=True)

    def test_create_branch_existing_tag_does_not_fail(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `create_branch` on existing tag."""
        repo_url = repo_factory()
        api.create_tag(repo_url.repo_id, tag="tag")
        api.create_branch(repo_url.repo_id, branch="tag")

    @pytest.mark.skip(
        "Test user is flagged as isHF which gives permissions to create invalid references."
        "Not relevant to test it anyway (i.e. it's more a server-side test)."
    )
    def test_create_branch_forbidden_ref_branch_fails(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `create_branch` on forbidden ref branch."""
        repo_url = repo_factory()
        with pytest.raises(BadRequestError, match="Invalid reference for a branch"):
            api.create_branch(repo_url.repo_id, branch="refs/pr/5")

        with pytest.raises(BadRequestError, match="Invalid reference for a branch"):
            api.create_branch(repo_url.repo_id, branch="refs/something/random")

    def test_delete_branch_on_protected_branch_fails(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `delete_branch` fails on protected branch."""
        repo_url = repo_factory()
        with pytest.raises(HfHubHTTPError, match="Cannot delete refs/heads/main"):
            api.delete_branch(repo_url.repo_id, branch="main")

    def test_delete_branch_on_missing_branch_fails(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `delete_branch` fails on missing branch."""
        repo_url = repo_factory()
        with pytest.raises(HfHubHTTPError, match="Invalid rev id"):
            api.delete_branch(repo_url.repo_id, branch="cool-branch")

        # Using a tag instead of branch -> fails
        api.create_tag(repo_url.repo_id, tag="cool-tag")
        with pytest.raises(HfHubHTTPError, match="Invalid rev id"):
            api.delete_branch(repo_url.repo_id, branch="cool-tag")

    def test_create_branch_from_revision(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Test `create_branch` from a different revision than main HEAD."""
        repo_url = repo_factory()
        # Create commit and remember initial/latest commit
        initial_commit = api.model_info(repo_url.repo_id).sha
        commit = api.create_commit(
            repo_url.repo_id,
            operations=[CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=b"content")],
            commit_message="test commit",
        )
        latest_commit = commit.oid

        # Create branches
        api.create_branch(repo_url.repo_id, branch="from-head")
        api.create_branch(repo_url.repo_id, branch="from-initial", revision=initial_commit)
        api.create_branch(repo_url.repo_id, branch="from-branch", revision="from-initial")
        time.sleep(0.2)  # hack: wait for server to update cache?

        # Checks branches start from expected commits
        assert {
            "main": latest_commit,
            "from-head": latest_commit,
            "from-initial": initial_commit,
            "from-branch": initial_commit,
        } == {ref.name: ref.target_commit for ref in api.list_repo_refs(repo_id=repo_url.repo_id).branches}


class TestHfApiDeleteFiles:
    @pytest.fixture(autouse=True)
    def _repo(self, api: HfApi):
        self.api = api
        self.repo_id = api.create_repo(repo_id=repo_name()).repo_id
        api.create_commit(
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
        yield
        api.delete_repo(repo_id=self.repo_id)

    def remote_files(self) -> set[str]:
        return set(self.api.list_repo_files(repo_id=self.repo_id))

    def test_delete_single_file(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["file.txt"])
        assert "file.txt" not in self.remote_files()

    def test_delete_multiple_files(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["file.txt", "lfs.bin"])
        files = self.remote_files()
        assert "file.txt" not in files
        assert "lfs.bin" not in files

    def test_delete_folder_with_pattern(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["nested/*"])
        assert self.remote_files() == {".gitattributes", "file.txt", "lfs.bin"}

    def test_delete_folder_without_pattern(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["nested/"])
        assert self.remote_files() == {".gitattributes", "file.txt", "lfs.bin"}

    def test_unknown_path_do_not_raise(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["not_existing", "nested/*"])
        assert self.remote_files() == {".gitattributes", "file.txt", "lfs.bin"}

    def test_delete_bin_files_with_patterns(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["*.bin"])
        files = self.remote_files()
        assert "lfs.bin" not in files
        assert "nested/lfs.bin" not in files
        assert "nested/sub/lfs.bin" not in files

    def test_delete_files_in_folders_with_patterns(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["*/file.txt"])
        files = self.remote_files()
        assert "file.txt" in files
        assert "nested/file.txt" not in files
        assert "nested/sub/file.txt" not in files

    def test_delete_all_files(self, api: HfApi):
        api.delete_files(repo_id=self.repo_id, delete_patterns=["*"])
        assert self.remote_files() == {".gitattributes"}


class TestHfApiPublicStaging:
    def test_staging_list_datasets(self, api: HfApi):
        api.list_datasets()

    def test_staging_list_models(self, api: HfApi):
        api.list_models()


@pytest.mark.production
class TestHfApiPublicProduction:
    @pytest.fixture
    def api(self) -> HfApi:
        # Unauthenticated production client (these are public, read-only endpoints).
        return HfApi(endpoint=ENDPOINT_PRODUCTION)

    def test_list_models(self, api: HfApi):
        models = list(api.list_models(limit=500))
        assert len(models) > 100
        assert isinstance(models[0], ModelInfo)

    def test_list_models_author(self, api: HfApi):
        models = list(api.list_models(author="google"))
        assert len(models) > 10
        assert isinstance(models[0], ModelInfo)
        for model in models:
            assert model.id.startswith("google/")

    def test_list_models_apps(self, api: HfApi):
        models = list(api.list_models(apps="ollama", full=True, limit=500))
        assert len(models) > 1
        for model in models:
            assert any(sibling.rfilename.lower().endswith(".gguf") for sibling in model.siblings)

    def test_list_models_search(self, api: HfApi):
        models = list(api.list_models(search="bert"))
        assert len(models) > 10
        assert isinstance(models[0], ModelInfo)
        for model in models[:10]:
            # Rough rule: at least first 10 will have "bert" in the name
            # Not optimal since it is dependent on how the Hub implements the search
            # (and changes it in the future) but for now it should do the trick.
            assert "bert" in model.id.lower()

    def test_list_models_num_parameters(self, api: HfApi):
        models = list(api.list_models(num_parameters="min:6B,max:128B", limit=5))
        assert len(models) == 5
        assert all(isinstance(model, ModelInfo) for model in models)

    def test_list_models_complex_query(self, api: HfApi):
        # Let's list the 10 most recent models
        # with tags "bert" and "jax",
        # ordered by last modified date.
        models = list(api.list_models(filter=("bert", "jax"), sort="last_modified", limit=10))
        # we have at least 1 models
        assert len(models) > 1
        assert len(models) <= 10
        model = models[0]
        assert isinstance(model, ModelInfo)
        assert all(tag in model.tags for tag in ["bert", "jax"])

    def test_list_models_sort_trending_score(self, api: HfApi):
        models = list(api.list_models(sort="trending_score", limit=10))
        assert len(models) == 10
        assert isinstance(models[0], ModelInfo)
        assert all(model.trending_score is not None for model in models)

    def test_list_models_sort_created_at(self, api: HfApi):
        models = list(api.list_models(sort="created_at", limit=10))
        assert len(models) == 10
        assert isinstance(models[0], ModelInfo)
        assert all(model.created_at is not None for model in models)

    def test_list_models_sort_downloads(self, api: HfApi):
        models = list(api.list_models(sort="downloads", limit=10))
        assert len(models) == 10
        assert isinstance(models[0], ModelInfo)
        assert all(model.downloads is not None for model in models)

    def test_list_models_sort_likes(self, api: HfApi):
        models = list(api.list_models(sort="likes", limit=10))
        assert len(models) == 10
        assert isinstance(models[0], ModelInfo)
        assert all(model.likes is not None for model in models)

    def test_list_models_with_config(self, api: HfApi):
        for model in api.list_models(filter=("adapter-transformers", "bert"), fetch_config=True, limit=20):
            assert model.config is not None

    def test_list_models_without_config(self, api: HfApi):
        for model in api.list_models(filter=("adapter-transformers", "bert"), fetch_config=False, limit=20):
            assert model.config is None

    def test_list_models_expand_author(self, api: HfApi):
        # Only the selected field is returned
        models = list(api.list_models(expand=["author"], limit=5))
        for model in models:
            assert model.author is not None
            assert model.id is not None
            assert model.downloads is None
            assert model.created_at is None
            assert model.last_modified is None

    def test_list_models_expand_multiple(self, api: HfApi):
        # Only the selected fields are returned
        models = list(api.list_models(expand=["author", "downloadsAllTime"], limit=5))
        for model in models:
            assert model.author is not None
            assert model.downloads_all_time is not None
            assert model.downloads is None

    def test_list_models_expand_unexpected_value(self, api: HfApi):
        # Unexpected value => HTTP 400
        with pytest.raises(HfHubHTTPError) as cm:
            list(api.list_models(expand=["foo"]))
        assert cm.value.response.status_code == 400

    def test_list_models_expand_cannot_be_used_with_other_params(self, api: HfApi):
        # `expand` cannot be used with other params
        with pytest.raises(ValueError):
            next(api.list_models(expand=["author"], full=True))
        with pytest.raises(ValueError):
            next(api.list_models(expand=["author"], fetch_config=True))
        with pytest.raises(ValueError):
            next(api.list_models(expand=["author"], cardData=True))

    def test_list_models_gated_only(self, api: HfApi):
        for model in api.list_models(expand=["gated"], gated=True, limit=5):
            assert model.gated in ("auto", "manual")

    def test_list_models_non_gated_only(self, api: HfApi):
        for model in api.list_models(expand=["gated"], gated=False, limit=5):
            assert model.gated is False

    @pytest.mark.skip("Inference parameter is being revamped")
    def test_list_models_inference_warm(self, api: HfApi):
        for model in api.list_models(inference=["warm"], expand="inference", limit=5):
            assert model.inference == "warm"

    @pytest.mark.skip("Inference parameter is being revamped")
    def test_list_models_inference_cold(self, api: HfApi):
        for model in api.list_models(inference=["cold"], expand="inference", limit=5):
            assert model.inference == "cold"

    def test_model_info(self, api: HfApi):
        model = api.model_info(repo_id=DUMMY_MODEL_ID)
        assert isinstance(model, ModelInfo)
        assert model.sha != DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT
        assert model.created_at == datetime.datetime(2022, 3, 2, 23, 29, 5, tzinfo=datetime.timezone.utc)

        # One particular commit (not the top of `main`)
        model = api.model_info(repo_id=DUMMY_MODEL_ID, revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)
        assert isinstance(model, ModelInfo)
        assert model.sha == DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT

    def test_model_info_with_security(self, api: HfApi):
        # Note: this test might break in the future if `security_repo_status` object structure gets updated server-side
        # (not yet fully stable)
        model = api.model_info(
            repo_id=DUMMY_MODEL_ID,
            revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
            securityStatus=True,
        )
        assert model.security_repo_status is not None
        assert isinstance(model.security_repo_status["scansDone"], bool)
        assert "filesWithIssues" in model.security_repo_status

    def test_model_info_with_file_metadata(self, api: HfApi):
        model = api.model_info(
            repo_id=DUMMY_MODEL_ID,
            revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
            files_metadata=True,
        )
        files = model.siblings
        assert files is not None
        self._check_siblings_metadata(files)

    def test_model_info_corrupted_model_index(self, caplog) -> None:
        """Loading model info from a model with corrupted data card should still work.

        Here we use a model with a "model-index" that is not an array. Git hook should prevent this from happening
        on the server, but models uploaded before we implemented the check might have this issue.

        Example data from https://huggingface.co/Waynehillsdev/Waynehills-STT-doogie-server.
        """
        with caplog.at_level("WARNING", logger="huggingface_hub"):
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
                    "siblings": None,
                }
            )
            assert model.card_data.eval_results is None
        assert any("Invalid model-index" in record.message for record in caplog.records)

    def test_model_info_with_widget_data(self, api: HfApi):
        info = api.model_info("HuggingFaceH4/zephyr-7b-beta")
        assert info.widget_data is not None

    def test_model_info_expand_author(self, api: HfApi):
        # Only the selected field is returned
        model = api.model_info(repo_id="HuggingFaceH4/zephyr-7b-beta", expand=["author"])
        assert model.author == "HuggingFaceH4"
        assert model.downloads is None
        assert model.created_at is None
        assert model.last_modified is None

    def test_model_info_expand_multiple(self, api: HfApi):
        # Only the selected fields are returned
        model = api.model_info(repo_id="HuggingFaceH4/zephyr-7b-beta", expand=["author", "downloadsAllTime"])
        assert model.author == "HuggingFaceH4"
        assert model.downloads is None
        assert model.downloads_all_time is not None
        assert model.created_at is None
        assert model.last_modified is None

    def test_model_info_expand_unexpected_value(self, api: HfApi):
        # Unexpected value => HTTP 400
        with pytest.raises(HfHubHTTPError) as cm:
            api.model_info("HuggingFaceH4/zephyr-7b-beta", expand=["foo"])
        assert cm.value.response.status_code == 400

    def test_model_info_expand_all_are_official_attributes(self):
        """All expand properties should be official ModelInfo attributes, not just __dict__ extras."""
        all_expand_values = list(get_args(ExpandModelProperty_T))

        dataclass_fields = {f.name for f in ModelInfo.__dataclass_fields__.values()}
        missing_attrs = []
        for expand_param in all_expand_values:
            attr_name = _to_snake_case(expand_param)
            if attr_name not in dataclass_fields:
                missing_attrs.append(f"{expand_param!r} -> {attr_name!r}")
        assert not missing_attrs, (
            f"The following expand parameters are not official ModelInfo attributes "
            f"(they fall through to __dict__.update): {missing_attrs}"
        )

    def test_model_info_expand_cannot_be_used_with_other_params(self, api: HfApi):
        # `expand` cannot be used with other params
        with pytest.raises(ValueError):
            api.model_info("HuggingFaceH4/zephyr-7b-beta", expand=["author"], securityStatus=True)
        with pytest.raises(ValueError):
            api.model_info("HuggingFaceH4/zephyr-7b-beta", expand=["author"], files_metadata=True)

    def test_list_repo_files(self, api: HfApi):
        files = api.list_repo_files(repo_id=DUMMY_MODEL_ID)
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
        assert files == expected_files

    def test_list_datasets_no_filter(self, api: HfApi):
        datasets = list(api.list_datasets(limit=500))
        assert len(datasets) > 100
        assert isinstance(datasets[0], DatasetInfo)

    def test_list_dataset_parquet_files(self, api: HfApi):
        entries = api.list_dataset_parquet_files(repo_id="nvidia/Llama-Nemotron-Post-Training-Dataset", token=False)
        assert len(entries) > 0
        assert entries[0].config
        assert entries[0].split
        assert entries[0].url.endswith(".parquet")
        assert entries[0].size > 0

    def test_filter_datasets_by_author_and_name(self, api: HfApi):
        datasets = list(api.list_datasets(author="huggingface", dataset_name="DataMeasurementsFiles"))
        assert len(datasets) > 0
        assert "huggingface" in datasets[0].author
        assert "DataMeasurementsFiles" in datasets[0].id

    def test_filter_datasets_by_benchmark_official(self, api: HfApi):
        datasets = list(api.list_datasets(benchmark="official", limit=10))
        assert len(datasets) > 0
        assert all("benchmark:official" in dataset.tags for dataset in datasets)

    def test_filter_datasets_by_benchmark_true_alias(self, api: HfApi):
        # benchmark=True should be an alias for benchmark="official"
        with patch("huggingface_hub.hf_api.paginate") as mock_paginate:
            mock_paginate.side_effect = lambda *args, **kwargs: []
            list(api.list_datasets(benchmark=True))
            list(api.list_datasets(benchmark="official"))

        # Exact same calls to paginate
        assert mock_paginate.call_count == 2
        assert mock_paginate.call_args_list[0][1]["params"] == {"filter": ["benchmark:official"]}
        assert mock_paginate.call_args_list[1][1]["params"] == {"filter": ["benchmark:official"]}

    def test_list_models_num_parameters_are_forwarded(self, api: HfApi):
        with patch("huggingface_hub.hf_api.paginate") as mock_paginate:
            mock_paginate.side_effect = lambda *args, **kwargs: []
            list(api.list_models(num_parameters="min:6B,max:128B"))

        assert mock_paginate.call_count == 1
        assert mock_paginate.call_args[1]["params"] == {"num_parameters": "min:6B,max:128B"}

    def test_filter_datasets_by_language_creator(self, api: HfApi):
        datasets = list(api.list_datasets(language_creators="crowdsourced"))
        assert len(datasets) > 0
        assert "language_creators:crowdsourced" in datasets[0].tags

    def test_filter_datasets_by_language_only(self, api: HfApi):
        datasets = list(api.list_datasets(language="en", limit=10))
        assert len(datasets) > 0
        assert "language:en" in datasets[0].tags

        datasets = list(api.list_datasets(language=("en", "fr"), limit=10))
        assert len(datasets) > 0
        assert "language:en" in datasets[0].tags
        assert "language:fr" in datasets[0].tags

    def test_filter_datasets_by_multilinguality(self, api: HfApi):
        datasets = list(api.list_datasets(multilinguality="multilingual", limit=10))
        assert len(datasets) > 0
        assert "multilinguality:multilingual" in datasets[0].tags

    def test_filter_datasets_by_size_categories(self, api: HfApi):
        datasets = list(api.list_datasets(size_categories="100K<n<1M", limit=10))
        assert len(datasets) > 0
        assert "size_categories:100K<n<1M" in datasets[0].tags

    def test_filter_datasets_by_task_categories(self, api: HfApi):
        datasets = list(api.list_datasets(task_categories="audio-classification", limit=10))
        assert len(datasets) > 0
        assert "task_categories:audio-classification" in datasets[0].tags

    def test_filter_datasets_by_task_ids(self, api: HfApi):
        datasets = list(api.list_datasets(task_ids="natural-language-inference", limit=10))
        assert len(datasets) > 0
        assert "task_ids:natural-language-inference" in datasets[0].tags

    def test_list_datasets_full(self, api: HfApi):
        datasets = list(api.list_datasets(full=True, limit=500))
        assert len(datasets) > 100
        assert isinstance(datasets[0], DatasetInfo)
        assert any(dataset.card_data for dataset in datasets)

    def test_list_datasets_author(self, api: HfApi):
        datasets = list(api.list_datasets(author="huggingface", limit=10))
        assert len(datasets) > 0
        assert datasets[0].author == "huggingface"

    def test_list_datasets_search(self, api: HfApi):
        datasets = list(api.list_datasets(search="wikipedia", limit=10))
        assert len(datasets) > 5
        for dataset in datasets:
            assert "wikipedia" in dataset.id.lower()

    def test_list_datasets_expand_author(self, api: HfApi):
        # Only the selected field is returned
        datasets = list(api.list_datasets(expand=["author"], limit=5))
        for dataset in datasets:
            assert dataset.author is not None
            assert dataset.id is not None
            assert dataset.downloads is None
            assert dataset.created_at is None
            assert dataset.last_modified is None

    def test_list_datasets_expand_multiple(self, api: HfApi):
        # Only the selected fields are returned
        datasets = list(api.list_datasets(expand=["author", "downloadsAllTime"], limit=5))
        for dataset in datasets:
            assert dataset.author is not None
            assert dataset.downloads_all_time is not None
            assert dataset.downloads is None

    def test_list_datasets_expand_unexpected_value(self, api: HfApi):
        # Unexpected value => HTTP 400
        with pytest.raises(HfHubHTTPError) as cm:
            list(api.list_datasets(expand=["foo"]))
        assert cm.value.response.status_code == 400

    def test_list_datasets_expand_cannot_be_used_with_full(self, api: HfApi):
        # `expand` cannot be used with `full`
        with pytest.raises(ValueError):
            next(api.list_datasets(expand=["author"], full=True))

    def test_list_datasets_gated_only(self, api: HfApi):
        for dataset in api.list_datasets(expand=["gated"], gated=True, limit=5):
            assert dataset.gated in ("auto", "manual")

    def test_list_datasets_non_gated_only(self, api: HfApi):
        for dataset in api.list_datasets(expand=["gated"], gated=False, limit=5):
            assert dataset.gated is False

    def test_filter_datasets_with_card_data(self, api: HfApi):
        assert any(dataset.card_data is not None for dataset in api.list_datasets(full=True, limit=50))
        assert all(dataset.card_data is None for dataset in api.list_datasets(full=False, limit=50))

    def test_filter_datasets_by_tag(self, api: HfApi):
        for dataset in api.list_datasets(filter="fiftyone", limit=5):
            assert "fiftyone" in dataset.tags

    def test_dataset_info(self, api: HfApi):
        dataset = api.dataset_info(repo_id=DUMMY_DATASET_ID)
        assert isinstance(dataset.card_data, DatasetCardData) and len(dataset.card_data) > 0
        assert isinstance(dataset.siblings, list) and len(dataset.siblings) > 0
        assert isinstance(dataset, DatasetInfo)
        assert dataset.sha != DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT
        dataset = api.dataset_info(
            repo_id=DUMMY_DATASET_ID,
            revision=DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT,
        )
        assert isinstance(dataset, DatasetInfo)
        assert dataset.sha == DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT

    def test_dataset_info_with_file_metadata(self, api: HfApi):
        dataset = api.dataset_info(repo_id=SAMPLE_DATASET_IDENTIFIER, files_metadata=True)
        files = dataset.siblings
        assert files is not None
        self._check_siblings_metadata(files)

    def _check_siblings_metadata(self, files: list[RepoSibling]):
        """Check requested metadata has been received from the server."""
        at_least_one_lfs = False
        for file in files:
            assert isinstance(file.blob_id, str)
            assert isinstance(file.size, int)
            if file.lfs is not None:
                at_least_one_lfs = True
                assert isinstance(file.lfs, dict)
                assert "sha256" in file.lfs
        assert at_least_one_lfs

    def test_dataset_info_expand_author(self, api: HfApi):
        # Only the selected field is returned
        dataset = api.dataset_info(repo_id="HuggingFaceH4/no_robots", expand=["author"])
        assert dataset.author == "HuggingFaceH4"
        assert dataset.downloads is None
        assert dataset.created_at is None
        assert dataset.last_modified is None

    def test_dataset_info_expand_multiple(self, api: HfApi):
        # Only the selected fields are returned
        dataset = api.dataset_info(repo_id="HuggingFaceH4/no_robots", expand=["author", "downloadsAllTime"])
        assert dataset.author == "HuggingFaceH4"
        assert dataset.downloads is None
        assert dataset.downloads_all_time is not None
        assert dataset.created_at is None
        assert dataset.last_modified is None

    def test_dataset_info_expand_unexpected_value(self, api: HfApi):
        # Unexpected value => HTTP 400
        with pytest.raises(HfHubHTTPError) as cm:
            api.dataset_info("HuggingFaceH4/no_robots", expand=["foo"])
        assert cm.value.response.status_code == 400

    def test_dataset_info_expand_all_are_official_attributes(self):
        """All expand properties should be official DatasetInfo attributes, not just __dict__ extras."""
        all_expand_values = list(get_args(ExpandDatasetProperty_T))

        dataclass_fields = {f.name for f in DatasetInfo.__dataclass_fields__.values()}
        missing_attrs = []
        for expand_param in all_expand_values:
            attr_name = _to_snake_case(expand_param)
            if attr_name not in dataclass_fields:
                missing_attrs.append(f"{expand_param!r} -> {attr_name!r}")
        assert not missing_attrs, (
            f"The following expand parameters are not official DatasetInfo attributes "
            f"(they fall through to __dict__.update): {missing_attrs}"
        )

    def test_dataset_info_expand_cannot_be_used_with_files_metadata(self, api: HfApi):
        # `expand` cannot be used with other `files_metadata`
        with pytest.raises(ValueError):
            api.dataset_info("HuggingFaceH4/no_robots", expand=["author"], files_metadata=True)

    @pytest.mark.production
    def test_get_dataset_leaderboard(self):
        leaderboard = HfApi().get_dataset_leaderboard("allenai/olmOCR-bench")
        assert isinstance(leaderboard, list)
        assert len(leaderboard) > 0
        entry = leaderboard[0]
        assert isinstance(entry, DatasetLeaderboardEntry)
        assert isinstance(entry.rank, int)
        assert entry.rank == 1
        assert isinstance(entry.model_id, str)
        assert isinstance(entry.value, (int, float))
        assert isinstance(entry.filename, str)
        assert isinstance(entry.verified, bool)
        assert isinstance(entry.source, dict)
        assert isinstance(entry.author, (User, Organization))
        # Optional fields should be accessible (may be None)
        assert entry.pull_request is None or isinstance(entry.pull_request, int)
        assert entry.notes is None or isinstance(entry.notes, str)

    @pytest.mark.production
    def test_get_dataset_leaderboard_not_found(self):
        with pytest.raises(RepositoryNotFoundError):
            HfApi().get_dataset_leaderboard("this-repo-does-not-exist/404")

    def test_dataset_leaderboard_entry_missing_source(self):
        # Some leaderboard entries returned by the server omit the "source" key entirely
        # (e.g. https://huggingface.co/api/datasets/Idavidrein/gpqa/leaderboard). This should
        # not raise a KeyError.
        entry = DatasetLeaderboardEntry(
            rank=1,
            modelId="some-org/some-model",
            value=42.0,
            filename="results.yaml",
            verified=False,
            author={"type": "org", "name": "some-org"},
        )
        assert entry.source is None

    def test_space_info(self, api: HfApi) -> None:
        space = api.space_info(repo_id="HuggingFaceH4/zephyr-chat")
        assert space.id == "HuggingFaceH4/zephyr-chat"
        assert space.author == "HuggingFaceH4"
        assert isinstance(space.runtime, SpaceRuntime)

    def test_space_info_expand_author(self, api: HfApi):
        # Only the selected field is returned
        space = api.space_info(repo_id="HuggingFaceH4/zephyr-chat", expand=["author"])
        assert space.author == "HuggingFaceH4"
        assert space.created_at is None
        assert space.last_modified is None

    def test_space_info_expand_multiple(self, api: HfApi):
        # Only the selected fields are returned
        space = api.space_info(repo_id="HuggingFaceH4/zephyr-chat", expand=["author", "likes"])
        assert space.author == "HuggingFaceH4"
        assert space.created_at is None
        assert space.last_modified is None
        assert space.likes is not None

    def test_space_info_expand_unexpected_value(self, api: HfApi):
        # Unexpected value => HTTP 400
        with pytest.raises(HfHubHTTPError) as cm:
            api.space_info("HuggingFaceH4/zephyr-chat", expand=["foo"])
        assert cm.value.response.status_code == 400

    def test_space_info_expand_all_are_official_attributes(self):
        """All expand properties should be official SpaceInfo attributes, not just __dict__ extras."""
        all_expand_values = list(get_args(ExpandSpaceProperty_T))

        dataclass_fields = {f.name for f in SpaceInfo.__dataclass_fields__.values()}
        missing_attrs = []
        for expand_param in all_expand_values:
            attr_name = _to_snake_case(expand_param)
            if attr_name not in dataclass_fields:
                missing_attrs.append(f"{expand_param!r} -> {attr_name!r}")
        assert not missing_attrs, (
            f"The following expand parameters are not official SpaceInfo attributes "
            f"(they fall through to __dict__.update): {missing_attrs}"
        )

    def test_space_info_expand_cannot_be_used_with_files_metadata(self, api: HfApi):
        # `expand` cannot be used with other files_metadata
        with pytest.raises(ValueError):
            api.space_info("HuggingFaceH4/zephyr-chat", expand=["author"], files_metadata=True)

    def test_filter_models_by_author(self, api: HfApi):
        models = list(api.list_models(author="muellerzr"))
        assert len(models) > 0
        assert "muellerzr" in models[0].id

    def test_filter_models_by_author_and_name(self, api: HfApi):
        # Test we can search by an author and a name, but the model is not found
        models = list(api.list_models(author="facebook", search="bart-base"))
        assert "facebook/bart-base" in models[0].id

    def test_failing_filter_models_by_author_and_search(self, api: HfApi):
        # Test we can search by an author and a name, but the model is not found
        models = list(api.list_models(author="muellerzr", search="testme"))
        assert len(models) == 0

    def test_filter_models_with_library(self, api: HfApi):
        models = list(api.list_models(author="microsoft", search="wavlm-base-sd", filter="tensorflow"))
        assert len(models) == 0

        models = list(api.list_models(author="microsoft", search="wavlm-base-sd", filter="pytorch"))
        assert len(models) > 0

    def test_filter_models_with_task(self, api: HfApi):
        models = list(api.list_models(filter="fill-mask", search="albert-base-v2"))
        assert models[0].pipeline_tag == "fill-mask"
        assert "albert" in models[0].id
        assert "base" in models[0].id
        assert "v2" in models[0].id

        models = list(api.list_models(filter="dummytask"))
        assert len(models) == 0

    def test_filter_models_by_language(self, api: HfApi):
        for language in ["en", "fr", "zh"]:
            for model in api.list_models(filter=language, limit=5):
                assert language in model.tags

    def test_filter_models_with_tag(self, api: HfApi):
        models = list(api.list_models(author="HuggingFaceBR4", filter=["tensorboard"]))
        assert models[0].id.startswith("HuggingFaceBR4/")
        assert "tensorboard" in models[0].tags

        models = list(api.list_models(filter=["dummytag"]))
        assert len(models) == 0

    def test_filter_models_with_card_data(self, api: HfApi):
        models = api.list_models(filter="co2_eq_emissions", cardData=True)
        assert any(model.card_data is not None for model in models)

        models = api.list_models(filter="co2_eq_emissions")
        assert all(model.card_data is None for model in models)

    def test_is_emission_within_threshold(self):
        # tests that dictionary is handled correctly as "emissions" and that
        # 17g is accepted and parsed correctly as a value
        # regression test for #753
        kwargs = {field.name: None for field in fields(ModelInfo) if field.init}
        kwargs = {**kwargs, "card_data": ModelCardData(co2_eq_emissions={"emissions": "17g"})}
        model = ModelInfo(**kwargs)
        assert _is_emission_within_threshold(model, -1, 100)

    def test_filter_emissions_with_max(self, api: HfApi):
        assert all(
            model.card_data["co2_eq_emissions"] <= 100
            for model in api.list_models(emissions_thresholds=(None, 100), cardData=True, limit=1000)
            if isinstance(model.card_data["co2_eq_emissions"], (float, int))
        )

    def test_filter_emissions_with_min(self, api: HfApi):
        assert all(
            [
                model.card_data["co2_eq_emissions"] >= 5
                for model in api.list_models(emissions_thresholds=(5, None), cardData=True, limit=1000)
                if isinstance(model.card_data["co2_eq_emissions"], (float, int))
            ]
        )

    def test_filter_emissions_with_min_and_max(self, api: HfApi):
        models = list(api.list_models(emissions_thresholds=(5, 100), cardData=True, limit=1000))
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

    def test_list_spaces_full(self, api: HfApi):
        spaces = list(api.list_spaces(full=True, limit=500))
        assert len(spaces) > 100
        space = spaces[0]
        assert isinstance(space, SpaceInfo)
        assert any(space.card_data for space in spaces)

    def test_list_spaces_author(self, api: HfApi):
        spaces = list(api.list_spaces(author="julien-c"))
        assert len(spaces) > 10
        for space in spaces:
            assert space.id.startswith("julien-c/")

    def test_list_spaces_search(self, api: HfApi):
        spaces = list(api.list_spaces(search="wikipedia", limit=10))
        assert "wikipedia" in spaces[0].id.lower()

    def test_list_spaces_sort(self, api: HfApi):
        # sort by likes in descending order => first item has more likes than second
        spaces_descending_likes = list(api.list_spaces(sort="likes", limit=100))
        assert spaces_descending_likes[0].likes > spaces_descending_likes[1].likes

    def test_list_spaces_limit(self, api: HfApi):
        spaces = list(api.list_spaces(limit=5))
        assert len(spaces) == 5

    def test_list_spaces_with_models(self, api: HfApi):
        spaces = list(api.list_spaces(models="bert-base-uncased"))
        assert "bert-base-uncased" in spaces[0].models

    def test_list_spaces_with_datasets(self, api: HfApi):
        spaces = list(api.list_spaces(datasets="wikipedia"))
        assert "wikipedia" in spaces[0].datasets

    def test_list_spaces_linked(self, api: HfApi):
        space_id = "black-forest-labs/FLUX.1-dev"

        spaces = [space for space in api.list_spaces(search=space_id) if space.id == space_id]
        assert spaces[0].models is None

        spaces = [space for space in api.list_spaces(search=space_id, linked=True) if space.id == space_id]
        assert spaces[0].models is not None

    def test_list_spaces_expand_author(self, api: HfApi):
        # Only the selected field is returned
        spaces = list(api.list_spaces(expand=["author"], limit=5))
        for space in spaces:
            assert space.author is not None
            assert space.id is not None
            assert space.created_at is None
            assert space.last_modified is None

    def test_list_spaces_expand_multiple(self, api: HfApi):
        # Only the selected fields are returned
        spaces = list(api.list_spaces(expand=["author", "likes"], limit=5))
        for space in spaces:
            assert space.author is not None
            assert space.likes is not None

    def test_list_spaces_expand_unexpected_value(self, api: HfApi):
        # Unexpected value => HTTP 400
        with pytest.raises(HfHubHTTPError) as cm:
            list(api.list_spaces(expand=["foo"]))
        assert cm.value.response.status_code == 400

    def test_list_spaces_expand_cannot_be_used_with_full(self, api: HfApi):
        # `expand` cannot be used with full
        with pytest.raises(ValueError):
            next(api.list_spaces(expand=["author"], full=True))

    def test_search_spaces(self, api: HfApi):
        results = list(api.search_spaces("generate image"))
        assert len(results) > 0
        result = results[0]
        assert isinstance(result, SpaceSearchResult)
        assert result.id
        assert result.author
        assert result.title
        assert result.likes >= 0
        assert result.semantic_relevancy_score is not None
        assert result.semantic_relevancy_score > 0
        assert result.runtime is not None
        assert isinstance(result.runtime, SpaceRuntime)

    def test_get_paths_info(self, api: HfApi):
        paths_info = api.get_paths_info(
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
        assert paths_info[1].size > 0

    def test_get_safetensors_metadata_single_file(self, api: HfApi) -> None:
        info = api.get_safetensors_metadata("bigscience/bloomz-560m")
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

    def test_get_safetensors_metadata_sharded_model(self, api: HfApi) -> None:
        info = api.get_safetensors_metadata("HuggingFaceH4/zephyr-7b-beta")
        assert isinstance(info, SafetensorsRepoMetadata)

        assert info.sharded
        assert isinstance(info.metadata, dict)  # populated for sharded model
        assert len(info.files_metadata) == 8

        for file_metadata in info.files_metadata.values():
            assert isinstance(file_metadata, SafetensorsFileMetadata)

        assert info.parameter_count == {"BF16": 7241732096}

    def test_not_a_safetensors_repo(self, api: HfApi) -> None:
        with pytest.raises(NotASafetensorsRepoError):
            api.get_safetensors_metadata("huggingface-hub-ci/test_safetensors_metadata")

    def test_get_safetensors_metadata_from_revision(self, api: HfApi) -> None:
        info = api.get_safetensors_metadata("huggingface-hub-ci/test_safetensors_metadata", revision="refs/pr/1")
        assert isinstance(info, SafetensorsRepoMetadata)

    def test_parse_safetensors_metadata(self, api: HfApi) -> None:
        info = api.parse_safetensors_file_metadata("HuggingFaceH4/zephyr-7b-beta", "model-00003-of-00008.safetensors")
        assert isinstance(info, SafetensorsFileMetadata)

        assert info.metadata == {"format": "pt"}
        assert isinstance(info.tensors, dict)
        tensor = info.tensors["model.layers.10.input_layernorm.weight"]

        assert tensor == TensorInfo(dtype="BF16", shape=[4096], data_offsets=(0, 8192))

        assert tensor.parameter_count == 4096
        assert info.parameter_count == {"BF16": 989888512}

    def test_not_a_safetensors_file(self, api: HfApi) -> None:
        with pytest.raises(SafetensorsParsingError):
            api.parse_safetensors_file_metadata("HuggingFaceH4/zephyr-7b-beta", "pytorch_model-00001-of-00008.bin")

    def test_inference_provider_mapping_model_info(self, api: HfApi):
        model = api.model_info("deepseek-ai/DeepSeek-R1-0528", expand="inferenceProviderMapping")
        mapping = model.inference_provider_mapping
        assert isinstance(mapping, list)
        assert len(mapping) > 0
        for item in mapping:
            assert isinstance(item, InferenceProviderMapping)
            assert item.provider is not None
            assert item.hf_model_id == "deepseek-ai/DeepSeek-R1-0528"
            assert item.provider_id is not None

    def test_inference_provider_mapping_list_models(self, api: HfApi):
        models = list(
            api.list_models(author="deepseek-ai", expand="inferenceProviderMapping", limit=1, inference_provider="all")
        )
        assert len(models) > 0
        mapping = models[0].inference_provider_mapping
        assert isinstance(mapping, list)
        assert len(mapping) > 0
        for item in mapping:
            assert isinstance(item, InferenceProviderMapping)
            assert item.provider is not None
            assert item.hf_model_id is not None
            assert item.provider_id is not None

    def test_filter_models_by_inference_provider(self, api: HfApi):
        models = list(
            api.list_models(inference_provider="hf-inference", expand=["inferenceProviderMapping"], limit=10)
        )
        assert len(models) > 0
        for model in models:
            assert model.inference_provider_mapping is not None
            assert any(mapping.provider == "hf-inference" for mapping in model.inference_provider_mapping)


class TestHfApiPrivate:
    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request, api: HfApi):
        repo_id = f"{USER}/{repo_name('private')}"
        api.create_repo(repo_id=repo_id, private=True)
        api.create_repo(repo_id=repo_id, private=True, repo_type="dataset")
        request.cls.repo_id = repo_id
        yield
        api.delete_repo(repo_id=repo_id)
        api.delete_repo(repo_id=repo_id, repo_type="dataset")

    def test_model_info(self, api: HfApi, mocker) -> None:
        mocker.patch("huggingface_hub.utils._headers.get_token", return_value=None)
        # Auth => retrieve private model
        api.model_info(repo_id=self.repo_id)

        # No auth => cannot access private model
        with patch.object(api, "token", None):
            with pytest.raises(HfHubHTTPError, match=r".*Repository Not Found.*"):
                _ = api.model_info(repo_id=self.repo_id)

    def test_dataset_info(self, api: HfApi, mocker) -> None:
        mocker.patch("huggingface_hub.utils._headers.get_token", return_value=None)
        # Auth => retrieve private dataset
        api.dataset_info(repo_id=self.repo_id)

        # No auth => cannot access private dataset
        with patch.object(api, "token", None):
            with pytest.raises(HfHubHTTPError, match=r".*Repository Not Found.*"):
                _ = api.dataset_info(repo_id=self.repo_id)

    def test_list_private_datasets(self, api: HfApi):
        kwargs = {"sort": "created_at", "limit": 100, "author": USER}
        assert all(dataset.id != self.repo_id for dataset in api.list_datasets(token=False, **kwargs))
        assert any(dataset.id == self.repo_id for dataset in api.list_datasets(token=TOKEN, **kwargs))

    def test_list_private_models(self, api: HfApi):
        kwargs = {"sort": "created_at", "limit": 100, "author": USER}
        assert all(model.id != self.repo_id for model in api.list_models(token=False, **kwargs))
        assert any(model.id == self.repo_id for model in api.list_models(token=TOKEN, **kwargs))


@pytest.mark.xet
class TestUploadFolderMocked:
    api = HfApi()

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, mocker) -> None:
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "lfs.bin").write_text("content")

        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.txt").write_text("content")
        (tmp_path / "sub" / "lfs_in_sub.bin").write_text("content")

        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file.txt").write_text("content")
        (tmp_path / "subdir" / "lfs_in_subdir.bin").write_text("content")

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

        # `upload_folder` now delegates the actual upload to the streamed xet pipeline. We mock
        # `pipelined_upload` to capture the add/delete operations it would commit, and force the
        # xet path so the test is deterministic regardless of whether `hf_xet` is installed.
        self.pipeline_mock = Mock()
        self.pipeline_mock.return_value.commit_url = f"{ENDPOINT_STAGING}/username/repo_id/commit/dummy_sha"
        self.pipeline_mock.return_value.pr_url = None
        mocker.patch("huggingface_hub.hf_api.is_xet_available", return_value=True)
        mocker.patch("huggingface_hub.hf_api.pipelined_upload", self.pipeline_mock)

    def _upload_folder_alias(self, tmp_path, **kwargs) -> list[Union[CommitOperationAdd, CommitOperationDelete]]:
        """Alias to call `upload_folder` + retrieve the CommitOperation list passed to the pipeline."""
        if "folder_path" not in kwargs:
            kwargs["folder_path"] = tmp_path
        self.api.upload_folder(repo_id="repo_id", **kwargs)
        call_kwargs = self.pipeline_mock.call_args_list[0][1]
        # `upload_folder` passes additions and deletions separately to the pipeline. Recombine them
        # (deletions first, as `create_commit` used to receive them) for the assertions below.
        return call_kwargs["delete_operations"] + call_kwargs["add_operations"]

    def test_allow_everything(self, tmp_path):
        operations = self._upload_folder_alias(tmp_path)
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert {op.path_in_repo for op in operations} == self.all_local_files

    def test_allow_everything_in_subdir_no_trailing_slash(self, tmp_path):
        operations = self._upload_folder_alias(tmp_path, folder_path=tmp_path / "subdir", path_in_repo="subdir")
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert {op.path_in_repo for op in operations} == {
            # correct `path_in_repo`
            "subdir/file.txt",
            "subdir/lfs_in_subdir.bin",
        }

    def test_allow_everything_in_subdir_with_trailing_slash(self, tmp_path):
        operations = self._upload_folder_alias(tmp_path, folder_path=tmp_path / "subdir", path_in_repo="subdir/")
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert {op.path_in_repo for op in operations} == {"subdir/file.txt", "subdir/lfs_in_subdir.bin"}

    def test_allow_txt_ignore_subdir(self, tmp_path):
        operations = self._upload_folder_alias(tmp_path, allow_patterns="*.txt", ignore_patterns="subdir/*")
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert {op.path_in_repo for op in operations} == {"sub/file.txt", "file.txt"}  # only .txt files, not in subdir

    def test_allow_txt_not_root_ignore_subdir(self, tmp_path):
        operations = self._upload_folder_alias(tmp_path, allow_patterns="**/*.txt", ignore_patterns="subdir/*")
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        assert {op.path_in_repo for op in operations} == {
            # only .txt files, not in subdir, not at root
            "sub/file.txt"
        }

    def test_path_in_repo_dot(self, tmp_path):
        """Regression test for #1382 when using `path_in_repo="."`.

        Using `path_in_repo="."` or `path_in_repo=None` should be equivalent.
        See https://github.com/huggingface/huggingface_hub/pull/1382.
        """
        operation_with_dot = self._upload_folder_alias(tmp_path, path_in_repo=".", allow_patterns=["file.txt"])[0]
        operation_with_none = self._upload_folder_alias(tmp_path, path_in_repo=None, allow_patterns=["file.txt"])[0]
        assert operation_with_dot.path_in_repo == "file.txt"
        assert operation_with_none.path_in_repo == "file.txt"

    def test_delete_txt(self, tmp_path):
        operations = self._upload_folder_alias(tmp_path, delete_patterns="*.txt")
        added_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}

        assert added_files == self.all_local_files
        assert deleted_files == {"file1.txt", "sub/file1.txt"}

        # since "file.txt" and "sub/file.txt" are overwritten, no need to delete them first
        assert "file.txt" in added_files
        assert "sub/file.txt" in added_files

    def test_delete_txt_in_sub(self, tmp_path):
        operations = self._upload_folder_alias(
            tmp_path, path_in_repo="sub/", folder_path=tmp_path / "sub", delete_patterns="*.txt"
        )
        added_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}

        assert added_files == {"sub/file.txt", "sub/lfs_in_sub.bin"}  # added only in sub/
        assert deleted_files == {"sub/file1.txt"}  # delete only in sub/

    def test_delete_txt_in_sub_ignore_sub_file_txt(self, tmp_path):
        operations = self._upload_folder_alias(
            tmp_path,
            path_in_repo="sub",
            folder_path=tmp_path / "sub",
            ignore_patterns="file.txt",
            delete_patterns="*.txt",
        )
        added_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}

        # since "sub/file.txt" should be deleted and is not overwritten (ignore_patterns), we delete it explicitly
        assert added_files == {"sub/lfs_in_sub.bin"}  # no "sub/file.txt"
        assert deleted_files == {"sub/file1.txt", "sub/file.txt"}

    def test_delete_if_path_in_repo(self, tmp_path):
        # Regression test for https://github.com/huggingface/huggingface_hub/pull/2129
        operations = self._upload_folder_alias(tmp_path, path_in_repo=".", folder_path=tmp_path, delete_patterns="*")
        deleted_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}
        assert deleted_files == {"file1.txt", "sub/file1.txt"}  # all the 'old' files


@pytest.mark.skip(
    # See https://huggingface.slack.com/archives/C02EMARJ65P/p1772636713600769 for more details (private link)
    reason="Skipping git clone test on CI."
)
class TestHfLargefiles:
    @pytest.fixture(autouse=True)
    def _cleanup(self, api: HfApi):
        yield
        api.delete_repo(repo_id=self.repo_id)

    def setup_local_clone(self, tmp_path) -> None:
        scheme = urlparse(self.repo_url).scheme
        repo_url_auth = self.repo_url.replace(f"{scheme}://", f"{scheme}://user:{TOKEN}@")

        subprocess.run(
            ["git", "clone", repo_url_auth, str(tmp_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(["git", "lfs", "track", "*.pdf"], check=True, cwd=tmp_path)
        subprocess.run(["git", "lfs", "track", "*.epub"], check=True, cwd=tmp_path)

    @pytest.mark.git_lfs
    def test_git_push_end_to_end(self, api: HfApi, tmp_path):
        self.repo_url = api.create_repo(repo_id=repo_name())
        self.repo_id = self.repo_url.repo_id
        self.setup_local_clone(tmp_path)

        subprocess.run(
            ["wget", LARGE_FILE_18MB], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp_path
        )
        subprocess.run(["git", "add", "*"], check=True, cwd=tmp_path)
        subprocess.run(["git", "commit", "-m", "commit message"], check=True, cwd=tmp_path)
        subprocess.run(["hf", "lfs-enable-largefiles", tmp_path], check=True)

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=tmp_path)
        print("took", time.time() - start_time)

        # To be 100% sure, let's download the resolved file
        with SoftTemporaryDirectory() as tmp_dir:
            filepath = hf_hub_download(
                repo_id=self.repo_id,
                filename="progit.pdf",
                cache_dir=tmp_dir,
            )
            assert Path(filepath).stat().st_size == 18685041


class TestParseHFUrl:
    def test_repo_type_and_id_from_hf_id_on_correct_values(self):
        possible_values = {
            "hub": {
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
            },
            "self-hosted": {
                "http://localhost:8080/hf/user/id": [None, "user", "id"],
                "http://localhost:8080/hf/datasets/user/id": ["dataset", "user", "id"],
                "http://localhost:8080/hf/models/user/id": ["model", "user", "id"],
            },
        }

        for key, value in possible_values.items():
            hub_url = ENDPOINT_PRODUCTION if key == "hub" else "http://localhost:8080/hf"
            for key, value in value.items():
                assert repo_type_and_id_from_hf_id(key, hub_url=hub_url) == tuple(value)

    def test_repo_type_and_id_from_hf_id_on_wrong_values(self):
        for hub_id in [
            "https://unknown-endpoint.co/id",
            "https://huggingface.co/datasets/user/id@revision",  # @ forbidden
            "datasets/user/id/subpath",
            "hffs://model/user/name",
            "spaeces/user/id",  # with typo in repo type
        ]:
            with pytest.raises(ValueError):
                repo_type_and_id_from_hf_id(hub_id, hub_url=ENDPOINT_PRODUCTION)


class TestHfApiDiscussions:
    @pytest.fixture(autouse=True)
    def _repo(self, api: HfApi):
        self.repo_id = api.create_repo(repo_id=repo_name()).repo_id
        self.pull_request = api.create_discussion(repo_id=self.repo_id, pull_request=True, title="Test Pull Request")
        self.discussion = api.create_discussion(repo_id=self.repo_id, pull_request=False, title="Test Discussion")
        yield
        api.delete_repo(repo_id=self.repo_id)

    def test_create_discussion(self, api: HfApi):
        discussion = api.create_discussion(repo_id=self.repo_id, title=" Test discussion !  ")
        assert discussion.num == 3
        assert discussion.author == USER
        assert discussion.is_pull_request is False
        assert discussion.title == "Test discussion !"

    def test_create_discussion_space(self, api: HfApi, repo_factory: RepoFactory):
        """Regression test for #1463.

        Computed URL was malformed with `dataset` and `space` repo_types.
        See https://github.com/huggingface/huggingface_hub/issues/1463.
        """
        repo_url = repo_factory("dataset")
        discussion = api.create_discussion(repo_id=repo_url.repo_id, repo_type="dataset", title="title")
        assert discussion.url == f"{repo_url}/discussions/1"

    def test_create_pull_request(self, api: HfApi):
        discussion = api.create_discussion(repo_id=self.repo_id, title=" Test PR !  ", pull_request=True)
        assert discussion.num == 3
        assert discussion.author == USER
        assert discussion.is_pull_request is True
        assert discussion.title == "Test PR !"

        model_info = api.repo_info(repo_id=self.repo_id, revision="refs/pr/1")
        assert isinstance(model_info, ModelInfo)

    def test_get_repo_discussion(self, api: HfApi):
        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id)
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.discussion.num, self.pull_request.num]

    def test_get_repo_discussion_by_type(self, api: HfApi):
        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, discussion_type="pull_request")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.pull_request.num]

        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, discussion_type="discussion")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.discussion.num]

        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, discussion_type="all")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.discussion.num, self.pull_request.num]

    def test_get_repo_discussion_by_author(self, api: HfApi):
        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, author="unknown")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == []

    def test_get_repo_discussion_by_status(self, api: HfApi):
        api.change_discussion_status(self.repo_id, self.discussion.num, "closed")

        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, discussion_status="open")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.pull_request.num]

        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, discussion_status="closed")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.discussion.num]

        discussions_generator = api.get_repo_discussions(repo_id=self.repo_id, discussion_status="all")
        assert isinstance(discussions_generator, types.GeneratorType)
        assert list([d.num for d in discussions_generator]) == [self.discussion.num, self.pull_request.num]

    @pytest.mark.production
    def test_get_repo_discussion_pagination(self):
        discussions = list(
            HfApi().get_repo_discussions(repo_id="open-llm-leaderboard/open_llm_leaderboard", repo_type="space")
        )
        assert len(discussions) > 50

    def test_get_discussion_details(self, api: HfApi):
        retrieved = api.get_discussion_details(repo_id=self.repo_id, discussion_num=2)
        assert retrieved == self.discussion

    def test_edit_discussion_comment(self, api: HfApi):
        def get_first_comment(discussion: DiscussionWithDetails) -> DiscussionComment:
            return [evt for evt in discussion.events if evt.type == "comment"][0]

        edited_comment = api.edit_discussion_comment(
            repo_id=self.repo_id,
            discussion_num=self.pull_request.num,
            comment_id=get_first_comment(self.pull_request).id,
            new_content="**Edited** comment 🤗",
        )
        retrieved = api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.pull_request.num)
        assert get_first_comment(retrieved).edited is True
        assert get_first_comment(retrieved).id == get_first_comment(self.pull_request).id
        assert get_first_comment(retrieved).content == "**Edited** comment 🤗"

        assert get_first_comment(retrieved) == edited_comment

    def test_comment_discussion(self, api: HfApi):
        new_comment = api.comment_discussion(
            repo_id=self.repo_id,
            discussion_num=self.discussion.num,
            comment="""\
                # Multi-line comment

                **With formatting**, including *italic text* & ~strike through~
                And even [links](http://hf.co)! 💥🤯
            """,
        )
        retrieved = api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.discussion.num)
        assert len(retrieved.events) == 2
        assert new_comment.id in {event.id for event in retrieved.events}

    def test_rename_discussion(self, api: HfApi):
        rename_event = api.rename_discussion(
            repo_id=self.repo_id, discussion_num=self.discussion.num, new_title="New title2"
        )
        retrieved = api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.discussion.num)
        assert rename_event.id in (event.id for event in retrieved.events)
        assert rename_event.old_title == self.discussion.title
        assert rename_event.new_title == "New title2"

    def test_change_discussion_status(self, api: HfApi):
        status_change_event = api.change_discussion_status(
            repo_id=self.repo_id, discussion_num=self.discussion.num, new_status="closed"
        )
        retrieved = api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.discussion.num)
        assert status_change_event.id in (event.id for event in retrieved.events)
        assert status_change_event.new_status == "closed"

        with pytest.raises(ValueError):
            api.change_discussion_status(
                repo_id=self.repo_id, discussion_num=self.discussion.num, new_status="published"
            )

    def test_merge_pull_request(self, api: HfApi):
        api.create_commit(
            repo_id=self.repo_id,
            commit_message="Commit some file",
            operations=[CommitOperationAdd(path_in_repo="file.test", path_or_fileobj=b"Content")],
            revision=self.pull_request.git_reference,
        )
        api.change_discussion_status(repo_id=self.repo_id, discussion_num=self.pull_request.num, new_status="open")
        api.merge_pull_request(self.repo_id, self.pull_request.num)

        retrieved = api.get_discussion_details(repo_id=self.repo_id, discussion_num=self.pull_request.num)
        assert retrieved.status == "merged"
        assert retrieved.merge_commit_oid is not None


class TestActivityApi:
    @pytest.fixture(autouse=True)
    def _api(self):
        self.api = HfApi()  # no auth!

    def test_unlike_missing_repo(self) -> None:
        with pytest.raises(RepositoryNotFoundError):
            self.api.unlike("missing_repo_id", token=TOKEN)

    def test_list_likes_repos_auth_and_implicit_user(self) -> None:
        # User is implicit
        likes = self.api.list_liked_repos(token=TOKEN)
        assert likes.user == USER

    def test_list_likes_repos_auth_and_explicit_user(self) -> None:
        # User is explicit even if auth
        likes = self.api.list_liked_repos(user=OTHER_USER, token=TOKEN)
        assert likes.user == OTHER_USER

    @pytest.mark.production
    def test_list_repo_likers(self) -> None:
        # a repo with > 5000 likes
        all_likers = list(
            HfApi().list_repo_likers(repo_id="open-llm-leaderboard/open_llm_leaderboard", repo_type="space")
        )
        assert isinstance(all_likers[0], User)
        assert len(all_likers) > 5000

    @pytest.mark.production
    def test_list_likes_on_production(self) -> None:
        # Test julien-c likes a lot of repos !
        likes = HfApi().list_liked_repos("julien-c")
        assert len(likes.models) + len(likes.datasets) + len(likes.spaces) + len(likes.kernels) == likes.total
        assert len(likes.models) > 0
        assert len(likes.datasets) > 0
        assert len(likes.spaces) > 0
        assert len(likes.kernels) > 0


class TestSquashHistory:
    def test_super_squash_history_on_branch(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        # Upload + update file on main
        repo_id = repo_url.repo_id
        api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"content")
        api.upload_file(repo_id=repo_id, path_in_repo="lfs.bin", path_or_fileobj=b"content")
        api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"another_content")

        # Upload file on a new branch
        api.create_branch(repo_id=repo_id, branch="v0.1", exist_ok=True)
        api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"foo", revision="v0.1")

        # Squash history on main
        api.super_squash_history(repo_id=repo_id)

        # List history
        squashed_main_commits = api.list_repo_commits(repo_id=repo_id, revision="main")
        branch_commits = api.list_repo_commits(repo_id=repo_id, revision="v0.1")

        # Main branch has been squashed but initial commits still exists on other branch
        assert len(squashed_main_commits) == 1
        assert squashed_main_commits[0].title == "Super-squash branch 'main' using huggingface_hub"
        assert len(branch_commits) == 5
        assert branch_commits[-1].title == "initial commit"

        # Squash history on branch
        api.super_squash_history(repo_id=repo_id, branch="v0.1")
        squashed_branch_commits = api.list_repo_commits(repo_id=repo_id, revision="v0.1")
        assert len(squashed_branch_commits) == 1
        assert squashed_branch_commits[0].title == "Super-squash branch 'v0.1' using huggingface_hub"

    def test_super_squash_history_on_special_ref(self, api: HfApi, repo_factory: RepoFactory) -> None:
        """Regression test for https://github.com/huggingface/dataset-viewer/pull/3131.

        In practice, it doesn't make any sense to super squash a PR as it will not be mergeable anymore.
        The only case where it's useful is for the dataset-viewer on refs/convert/parquet.
        """
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        pr = api.create_pull_request(repo_id=repo_id, title="Test super squash on PR")

        # Upload + update file on PR
        api.upload_file(
            repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"content", revision=pr.git_reference
        )
        api.upload_file(repo_id=repo_id, path_in_repo="lfs.bin", path_or_fileobj=b"content", revision=pr.git_reference)
        api.upload_file(
            repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"another_content", revision=pr.git_reference
        )

        # Squash history PR
        api.super_squash_history(repo_id=repo_id, branch=pr.git_reference)

        squashed_branch_commits = api.list_repo_commits(repo_id=repo_id, revision=pr.git_reference)
        assert len(squashed_branch_commits) == 1


class TestListAndPermanentlyDeleteLFSFiles:
    def test_list_and_delete_lfs_files(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id

        # Main files
        api.upload_file(path_or_fileobj=b"LFS content", path_in_repo="lfs_file.bin", repo_id=repo_id)
        api.upload_file(path_or_fileobj=b"TXT content", path_in_repo="txt_file.txt", repo_id=repo_id)
        api.upload_file(path_or_fileobj=b"LFS content 2", path_in_repo="lfs_file_2.bin", repo_id=repo_id)
        api.upload_file(path_or_fileobj=b"TXT content 2", path_in_repo="txt_file_2.txt", repo_id=repo_id)

        # Branch files
        api.create_branch(repo_id=repo_id, branch="my-branch")
        api.upload_file(
            path_or_fileobj=b"LFS content branch",
            path_in_repo="lfs_file_branch.bin",
            repo_id=repo_id,
            revision="my-branch",
        )
        api.upload_file(
            path_or_fileobj=b"TXT content branch",
            path_in_repo="txt_file_branch.txt",
            repo_id=repo_id,
            revision="my-branch",
        )

        # List LFS files
        lfs_files = [file for file in api.list_lfs_files(repo_id=repo_id)]
        assert len(lfs_files) == 3
        assert {file.filename for file in lfs_files} == {
            "lfs_file.bin",
            "lfs_file_2.bin",
            "lfs_file_branch.bin",
        }

        # Select LFS files that are on main
        lfs_files_on_main = [file for file in lfs_files if file.ref in ("main", "refs/heads/main")]
        assert len(lfs_files_on_main) == 2

        # Permanently delete LFS files
        api.permanently_delete_lfs_files(repo_id=repo_id, lfs_files=lfs_files_on_main)

        # LFS file from the branch remains
        lfs_files = [file for file in api.list_lfs_files(repo_id=repo_id)]
        assert len(lfs_files) == 1
        assert {file.filename for file in lfs_files} == {"lfs_file_branch.bin"}

        # Downloading "lfs_file.bin" fails with EntryNotFoundError
        files = api.list_repo_files(repo_id=repo_id)
        assert set(files) == {".gitattributes", "txt_file.txt", "txt_file_2.txt"}
        with pytest.raises(EntryNotFoundError):
            api.hf_hub_download(repo_id=repo_id, filename="lfs_file.bin")


@pytest.mark.vcr
@pytest.mark.production
class TestSpaceAPIProduction:
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

    @pytest.fixture(autouse=True)
    def _space(self):
        # If generating new VCR => use personal token and REMOVE IT from the VCR
        self.repo_id = "user/tmp_test_space"  # no need to be unique as it's a VCRed test
        self.api = HfApi(token="hf_fake_token", endpoint=ENDPOINT_PRODUCTION)

        # Create a Space
        self.api.create_repo(repo_id=self.repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
        self.api.upload_file(
            path_or_fileobj=self._BASIC_APP_PY_TEMPLATE,
            repo_id=self.repo_id,
            repo_type="space",
            path_in_repo="app.py",
        )
        yield
        self.api.delete_repo(repo_id=self.repo_id, repo_type="space")

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
        assert len(variables) == 3

    def test_space_runtime(self) -> None:
        runtime = self.api.get_space_runtime(self.repo_id)

        # Space has just been created: hardware might not be set yet.
        assert runtime.hardware in (None, SpaceHardware.CPU_BASIC)
        assert runtime.requested_hardware in (None, SpaceHardware.CPU_BASIC)

        # Space is either "BUILDING" (if not yet done) or "NO_APP_FILE" (if building failed)
        assert runtime.stage in (SpaceStage.NO_APP_FILE, SpaceStage.BUILDING)
        assert runtime.stage in ("NO_APP_FILE", "BUILDING")  # str works as well

        # Raw response from Hub
        assert isinstance(runtime.raw, dict)

    def test_static_space_runtime(self) -> None:
        """
        Regression test for static Spaces.
        See https://github.com/huggingface/huggingface_hub/pull/1754.
        """
        runtime = self.api.get_space_runtime("victor/static-space")
        assert isinstance(runtime.raw, dict)

    @pytest.mark.production
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
        assert runtime_after_pause.stage == SpaceStage.PAUSED

        # Restart
        self.api.restart_space(self.repo_id)
        time.sleep(0.5)
        runtime_after_restart = self.api.get_space_runtime(self.repo_id)
        assert runtime_after_restart.stage != SpaceStage.PAUSED


class TestCommitInBackground:
    def test_commit_to_repo_in_background(self, api: HfApi, repo_factory: RepoFactory, tmp_path) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "lfs.bin").write_text("content")

        t0 = time.time()
        upload_future_1 = api.upload_file(
            path_or_fileobj=b"1", path_in_repo="1.txt", repo_id=repo_id, commit_message="Upload 1", run_as_future=True
        )
        upload_future_2 = api.upload_file(
            path_or_fileobj=b"2", path_in_repo="2.txt", repo_id=repo_id, commit_message="Upload 2", run_as_future=True
        )
        upload_future_3 = api.upload_folder(
            repo_id=repo_id, folder_path=tmp_path, commit_message="Upload folder", run_as_future=True
        )
        t1 = time.time()

        # all futures are queued instantly
        assert t1 - t0 <= 0.01

        # wait for the last job to complete
        upload_future_3.result()

        # all of them are now complete (ran in order)
        assert upload_future_1.done()
        assert upload_future_2.done()
        assert upload_future_3.done()

        # 4 commits, sorted in reverse order of creation
        commits = api.list_repo_commits(repo_id=repo_id)
        assert len(commits) == 4
        assert commits[0].title == "Upload folder"
        assert commits[1].title == "Upload 2"
        assert commits[2].title == "Upload 1"
        assert commits[3].title == "initial commit"

    def test_run_as_future(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory()
        repo_id = repo_url.repo_id
        # update repo visibility to private
        api.run_as_future(api.update_repo_settings, repo_id=repo_id, private=True)
        future_1 = api.run_as_future(api.model_info, repo_id=repo_id)

        # update repo visibility to public
        api.run_as_future(api.update_repo_settings, repo_id=repo_id, private=False)
        future_2 = api.run_as_future(api.model_info, repo_id=repo_id)

        assert isinstance(future_1, Future)
        assert isinstance(future_2, Future)

        # Wait for first info future
        info_1 = future_1.result()
        assert not future_2.done()

        # Wait for second info future
        info_2 = future_2.result()
        assert future_2.done()

        # Like/unlike is correct
        assert info_1.private is True
        assert info_2.private is False


class TestDownloadHfApiAlias:
    @pytest.fixture(autouse=True)
    def _api(self):
        self.api = HfApi(
            endpoint="https://hf.co",
            token="user_token",
            library_name="cool_one",
            library_version="1.0.0",
            user_agent="myself",
        )

    def test_hf_hub_download_alias(self, mocker) -> None:
        mock = mocker.patch("huggingface_hub.file_download.hf_hub_download")
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
            force_download=False,
            etag_timeout=10,
            local_files_only=False,
            headers=None,
            tqdm_class=None,
            dry_run=False,
        )

    def test_snapshot_download_alias(self, mocker) -> None:
        mock = mocker.patch("huggingface_hub._snapshot_download.snapshot_download")
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
            etag_timeout=10,
            force_download=False,
            local_files_only=False,
            allow_patterns=None,
            ignore_patterns=None,
            max_workers=8,
            tqdm_class=None,
            headers=None,
            dry_run=False,
        )


class TestSpaceAPIMocked:
    """
    Testing Space hardware requests is resource intensive for the server (need to spawn
    GPUs). Tests are mocked to check the correct values are sent.
    """

    @pytest.fixture(autouse=True)
    def _mocked_session(self, mocker):
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
        mocker.patch("huggingface_hub.hf_api.get_session", get_session_mock)

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
                "type": "space",
                "sdk": "gradio",
                "hardware": "t4-medium",
                "sleepTimeSeconds": 123,
            },
        )

    @pytest.mark.deprecated("create_repo")
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
                "type": "space",
                "sdk": "gradio",
                "storageTier": "large",
            },
        )

    def test_protected_visibility_is_only_supported_for_spaces(self) -> None:
        with pytest.raises(
            ValueError, match=r"Only Spaces can be 'protected'. Please set visibility to 'public' or 'private'."
        ):
            self.api.create_repo(self.repo_id, visibility="protected")
        self.post_mock.assert_not_called()

    def test_private_and_visibility_are_mutually_exclusive(self) -> None:
        with pytest.raises(
            ValueError, match=r"Received both `private` and `visibility` arguments. Please provide only one of them."
        ):
            self.api.create_repo(self.repo_id, private=True, visibility="private")
        self.post_mock.assert_not_called()

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

    @pytest.mark.deprecated("duplicate_space", "duplicate_repo")
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
                "visibility": "private",
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
        with pytest.warns(UserWarning):
            self.api.set_space_sleep_time(self.repo_id, sleep_time=123)

    @pytest.mark.deprecated("request_space_storage")
    def test_request_space_storage(self) -> None:
        runtime = self.api.request_space_storage(self.repo_id, SpaceStorage.LARGE)
        self.post_mock.assert_called_once_with(
            f"{self.api.endpoint}/api/spaces/{self.repo_id}/storage",
            headers=self.api._build_hf_headers(),
            json={"tier": "large"},
        )
        assert runtime.storage == SpaceStorage.LARGE

    @pytest.mark.deprecated("delete_space_storage")
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


@pytest.mark.production
class TestListGitRefs:
    @pytest.fixture(autouse=True)
    def _api(self):
        self.api = HfApi(endpoint=ENDPOINT_PRODUCTION)

    def test_list_refs_gpt2(self) -> None:
        refs = self.api.list_repo_refs("gpt2")
        assert len(refs.branches) > 0
        main_branch = [branch for branch in refs.branches if branch.name == "main"][0]
        assert main_branch.ref == "refs/heads/main"
        assert refs.pull_requests is None
        # Can get info by revision
        self.api.repo_info("gpt2", revision=main_branch.target_commit)

    def test_list_refs_bigcode(self) -> None:
        refs = self.api.list_repo_refs("bigcode/admin", repo_type="dataset")
        assert len(refs.branches) > 0
        assert len(refs.converts) > 0
        assert refs.pull_requests is None
        main_branch = [branch for branch in refs.branches if branch.name == "main"][0]
        assert main_branch.ref == "refs/heads/main"

        convert_branch = [branch for branch in refs.converts if branch.name == "parquet"][0]
        assert convert_branch.ref == "refs/convert/parquet"

        # Can get info by convert revision
        self.api.repo_info(
            "bigcode/admin",
            repo_type="dataset",
            revision=convert_branch.target_commit,
        )

    def test_list_refs_with_prs(self) -> None:
        refs = self.api.list_repo_refs("openchat/openchat_3.5", include_pull_requests=True)
        assert len(refs.pull_requests) > 1
        assert refs.pull_requests[0].ref.startswith("refs/pr/")


class TestListGitCommits:
    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request, api: HfApi):
        request.cls.api = api
        # Create repo (with initial commit)
        repo_id = api.create_repo(repo_name()).repo_id

        # Create a commit on `main` branch
        api.upload_file(repo_id=repo_id, path_or_fileobj=b"content", path_in_repo="content.txt")

        # Create a commit in a PR
        api.upload_file(repo_id=repo_id, path_or_fileobj=b"on_pr", path_in_repo="on_pr.txt", create_pr=True)

        # Create another commit on `main` branch
        api.upload_file(repo_id=repo_id, path_or_fileobj=b"on_main", path_in_repo="on_main.txt")
        request.cls.repo_id = repo_id
        yield
        api.delete_repo(repo_id)

    def test_list_commits_on_main(self) -> None:
        commits = self.api.list_repo_commits(self.repo_id)

        # "on_pr" commit not returned
        assert len(commits) == 3
        assert all("on_pr" not in commit.title for commit in commits)

        # USER is always the author
        assert all(commit.authors == [USER] for commit in commits)

        # latest commit first
        assert commits[0].title == "Upload on_main.txt with huggingface_hub"

        # Formatted field not returned by default
        for commit in commits:
            assert commit.formatted_title is None
            assert commit.formatted_message is None

    def test_list_commits_on_pr(self) -> None:
        commits = self.api.list_repo_commits(self.repo_id, revision="refs/pr/1")

        # "on_pr" commit returned but not the "on_main" one
        assert len(commits) == 3
        assert all("on_main" not in commit.title for commit in commits)
        assert commits[0].title == "Upload on_pr.txt with huggingface_hub"

    def test_list_commits_include_formatted(self) -> None:
        for commit in self.api.list_repo_commits(self.repo_id, formatted=True):
            assert commit.formatted_title is not None
            assert commit.formatted_message is not None

    def test_list_commits_on_missing_repo(self) -> None:
        with pytest.raises(RepositoryNotFoundError):
            self.api.list_repo_commits("missing_repo_id")

    def test_list_commits_on_missing_revision(self) -> None:
        with pytest.raises(RevisionNotFoundError):
            self.api.list_repo_commits(self.repo_id, revision="missing_revision")


class TestHfApiTokenAttribute:
    @pytest.fixture(autouse=True)
    def _mock_build_headers(self, mocker):
        self.mock_build_hf_headers = mocker.patch("huggingface_hub.hf_api.build_hf_headers")

    def _assert_token_is(self, expected_value) -> None:
        assert self.mock_build_hf_headers.call_args[1]["token"] == expected_value

    def test_token_passed(self) -> None:
        HfApi(token="default token")._build_hf_headers(token="A token")
        self._assert_token_is("A token")

    def test_no_token_passed(self) -> None:
        HfApi(token="default token")._build_hf_headers()
        self._assert_token_is("default token")

    def test_token_true_passed(self) -> None:
        HfApi(token="default token")._build_hf_headers(token=True)
        self._assert_token_is(True)

    def test_token_false_passed(self) -> None:
        HfApi(token="default token")._build_hf_headers(token=False)
        self._assert_token_is(False)

    def test_no_token_at_all(self) -> None:
        HfApi()._build_hf_headers(token=None)
        self._assert_token_is(None)

    def test_library_name_and_version_are_set(self) -> None:
        HfApi(library_name="a", library_version="b")._build_hf_headers()
        assert self.mock_build_hf_headers.call_args[1]["library_name"] == "a"
        assert self.mock_build_hf_headers.call_args[1]["library_version"] == "b"

    def test_library_name_and_version_are_overwritten(self) -> None:
        api = HfApi(library_name="a", library_version="b")
        api._build_hf_headers(library_name="A", library_version="B")
        assert self.mock_build_hf_headers.call_args[1]["library_name"] == "A"
        assert self.mock_build_hf_headers.call_args[1]["library_version"] == "B"

    def test_user_agent_is_set(self) -> None:
        HfApi(user_agent={"a": "b"})._build_hf_headers()
        assert self.mock_build_hf_headers.call_args[1]["user_agent"] == {"a": "b"}

    def test_user_agent_is_overwritten(self) -> None:
        HfApi(user_agent={"a": "b"})._build_hf_headers(user_agent={"A": "B"})
        assert self.mock_build_hf_headers.call_args[1]["user_agent"] == {"A": "B"}


@pytest.mark.production
class TestRepoUrl:
    def test_repo_url_class(self):
        url = RepoUrl("https://huggingface.co/user/repo_name")

        # RepoUrl Is a string
        assert isinstance(url, str)
        assert url == "https://huggingface.co/user/repo_name"

        # Any str-method can be applied
        assert url.split("/") == "https://huggingface.co/user/repo_name".split("/")

        # String formatting and concatenation work
        assert f"New repo: {url}" == "New repo: https://huggingface.co/user/repo_name"
        assert "New repo: " + url == "New repo: https://huggingface.co/user/repo_name"

        # __repr__ is modified for debugging purposes
        assert repr(url) == (
            "RepoUrl('https://huggingface.co/user/repo_name',"
            " endpoint='https://huggingface.co', repo_type='model', repo_id='user/repo_name')"
        )

    def test_repo_url_endpoint(self):
        # Implicit endpoint
        url = RepoUrl("https://huggingface.co/user/repo_name")
        assert url.endpoint == ENDPOINT_PRODUCTION

        # Explicit (custom / self-hosted) endpoint: the endpoint prefix is stripped before parsing.
        url = RepoUrl("https://example.com/user/repo_name", endpoint="https://example.com")
        assert url.endpoint == "https://example.com"
        assert url.repo_id == "user/repo_name"
        assert url.repo_type == "model"

    def test_repo_url_repo_type(self):
        # Explicit repo type
        url = RepoUrl("https://huggingface.co/user/repo_name")
        assert url.repo_type == "model"

        url = RepoUrl("https://huggingface.co/datasets/user/repo_name")
        assert url.repo_type == "dataset"

        url = RepoUrl("https://huggingface.co/spaces/user/repo_name")
        assert url.repo_type == "space"

        # Implicit repo type (model)
        url = RepoUrl("https://huggingface.co/user/repo_name")
        assert url.repo_type == "model"

    def test_repo_url_namespace(self):
        url = RepoUrl("https://huggingface.co/dummy_user/dummy_model")
        assert url.namespace == "dummy_user"
        assert url.repo_name == "dummy_model"
        assert url.repo_id == "dummy_user/dummy_model"

    def test_repo_url_url_property(self):
        # RepoUrl.url returns a pure `str` value
        url = RepoUrl("https://huggingface.co/user/repo_name")
        assert url == "https://huggingface.co/user/repo_name"
        assert url.url == "https://huggingface.co/user/repo_name"
        assert isinstance(url, RepoUrl)
        assert not isinstance(url.url, RepoUrl)

    def test_repo_url_accepts_bare_and_hf_ids(self):
        # Bare '<namespace>/<name>' ids and 'hf://' URIs are normalized through `parse_hf_uri`.
        for _id in ("user/repo_name", "hf://user/repo_name"):
            url = RepoUrl(_id)
            assert url.repo_id == "user/repo_name"
            assert url.repo_type == "model"

        url = RepoUrl("hf://datasets/user/squad")
        assert url.repo_id == "user/squad"
        assert url.repo_type == "dataset"

    def test_repo_url_canonical_repo_not_supported(self):
        # Canonical single-segment repos (no namespace) are intentionally rejected.
        for _id in ("gpt2", "hf://gpt2", "https://huggingface.co/gpt2", "https://huggingface.co/datasets/squad"):
            with pytest.raises(ValueError):
                RepoUrl(_id)

    def test_repo_url_in_commit_info(self):
        info = CommitInfo(
            commit_url="https://huggingface.co/Wauplin/test-repo-id-mixin/commit/52d172a8b276e529d5260d6f3f76c85be5889dee",
            commit_message="Dummy message",
            commit_description="Dummy description",
            oid="52d172a8b276e529d5260d6f3f76c85be5889dee",
            pr_url=None,
            _endpoint=None,
        )
        assert isinstance(info.repo_url, RepoUrl)
        assert info.repo_url.endpoint == "https://huggingface.co"
        assert info.repo_url.repo_id == "Wauplin/test-repo-id-mixin"
        assert info.repo_url.repo_type == "model"

    def test_custom_endpoint_in_commit_info(self):
        """Regression test for #3679

        See https://github.com/huggingface/huggingface_hub/pulls/3679 for more details.
        """
        info = CommitInfo(
            commit_url="http://localhost:5564/Wauplin/dummy/commit/52d172a8b276e529d5260d6f3f76c85be5889dee",
            commit_message="Dummy message",
            commit_description="Dummy description",
            oid="52d172a8b276e529d5260d6f3f76c85be5889dee",
            pr_url=None,
            _endpoint="http://localhost:5564",
        )
        assert info.repo_url.endpoint == "http://localhost:5564"
        assert info.repo_url.repo_id == "Wauplin/dummy"
        assert info.repo_url.repo_type == "model"


class TestHfApiDuplicateSpace:
    @pytest.mark.deprecated("duplicate_space")
    @pytest.mark.skip("Duplicating Space doesn't work on staging.")
    def test_duplicate_space_success(self, api: HfApi) -> None:
        """Check `duplicate_space` works."""
        from_repo_name = repo_name()
        from_repo_id = api.create_repo(
            repo_id=from_repo_name,
            repo_type="space",
            space_sdk="static",
            token=OTHER_TOKEN,
        ).repo_id
        api.upload_file(
            path_or_fileobj=b"data",
            path_in_repo="temp/new_file.md",
            repo_id=from_repo_id,
            repo_type="space",
            token=OTHER_TOKEN,
        )

        to_repo_id = api.duplicate_space(from_repo_id).repo_id

        assert to_repo_id == f"{USER}/{from_repo_name}"
        assert api.list_repo_files(repo_id=from_repo_id, repo_type="space") == [
            ".gitattributes",
            "README.md",
            "index.html",
            "style.css",
            "temp/new_file.md",
        ]
        assert api.list_repo_files(repo_id=to_repo_id, repo_type="space") == api.list_repo_files(
            repo_id=from_repo_id, repo_type="space"
        )

        api.delete_repo(repo_id=from_repo_id, repo_type="space", token=OTHER_TOKEN)
        api.delete_repo(repo_id=to_repo_id, repo_type="space")

    @pytest.mark.deprecated("duplicate_space")
    def test_duplicate_space_from_missing_repo(self, api: HfApi) -> None:
        """Check `duplicate_space` fails when the from_repo doesn't exist."""

        with pytest.raises(RepositoryNotFoundError):
            api.duplicate_space(f"{OTHER_USER}/repo_that_does_not_exist")


class TestCollectionAPI:
    @pytest.fixture(autouse=True)
    def _collection(self, api: HfApi):
        id = uuid.uuid4()
        self.title = f"My cool stuff {id}"
        self.slug_prefix = f"{USER}/my-cool-stuff-{id}"
        self.slug: Optional[str] = None  # Populated by the tests => use to delete in teardown
        yield
        if self.slug is not None:  # Delete collection even if test failed
            api.delete_collection(self.slug, missing_ok=True)

    @pytest.mark.production
    def test_list_collections(self) -> None:
        item_id = "teknium/OpenHermes-2.5-Mistral-7B"
        item_type = "model"
        limit = 3
        collections = HfApi().list_collections(item=f"{item_type}s/{item_id}", limit=limit)

        # Check return type
        assert isinstance(collections, Iterable)
        collections = list(collections)

        # Check length
        assert len(collections) == limit

        # Check all collections contain the item
        for collection in collections:
            # all items are not necessarily returned when listing collections => retrieve complete one
            full_collection = HfApi().get_collection(collection.slug)
            assert any(item.item_id == item_id and item.item_type == item_type for item in full_collection.items)

    def test_create_collection_with_description(self, api: HfApi) -> None:
        collection = api.create_collection(self.title, description="Contains a lot of cool stuff")
        self.slug = collection.slug

        assert isinstance(collection, Collection)
        assert collection.title == self.title
        assert collection.description == "Contains a lot of cool stuff"
        assert collection.items == []
        assert collection.slug.startswith(self.slug_prefix)
        assert collection.url == f"{ENDPOINT_STAGING}/collections/{collection.slug}"

    @pytest.mark.skip("Creating duplicated collections work on staging")
    def test_create_collection_exists_ok(self, api: HfApi) -> None:
        # Create collection once without description
        collection_1 = api.create_collection(self.title)
        self.slug = collection_1.slug

        # Cannot create twice with same title
        with pytest.raises(HfHubHTTPError):  # already exists
            api.create_collection(self.title)

        # Can ignore error
        collection_2 = api.create_collection(self.title, description="description", exists_ok=True)

        assert collection_1.slug == collection_2.slug
        assert collection_1.description is None
        assert collection_2.description is None  # Did not get updated!

    def test_create_private_collection(self, api: HfApi) -> None:
        collection = api.create_collection(self.title, private=True)
        self.slug = collection.slug

        # Get private collection
        api.get_collection(collection.slug)  # no error
        with pytest.raises(HfHubHTTPError):
            api.get_collection(collection.slug, token=OTHER_TOKEN)  # not authorized

        # Get public collection
        api.update_collection_metadata(collection.slug, private=False)
        api.get_collection(collection.slug)  # no error
        api.get_collection(collection.slug, token=OTHER_TOKEN)  # no error

    def test_update_collection(self, api: HfApi) -> None:
        # Create collection
        collection_1 = api.create_collection(self.title)
        self.slug = collection_1.slug

        # Update metadata
        new_title = f"New title {uuid.uuid4()}"
        collection_2 = api.update_collection_metadata(
            collection_slug=collection_1.slug,
            title=new_title,
            description="New description",
            private=True,
            theme="pink",
        )

        assert collection_2.title == new_title
        assert collection_2.description == "New description"
        assert collection_2.private is True
        assert collection_2.theme == "pink"
        assert collection_1.slug != collection_2.slug

        # Different slug, same id
        assert collection_1.slug.split("-")[-1] == collection_2.slug.split("-")[-1]

        # Works with both slugs, same collection returned
        assert api.get_collection(collection_1.slug).slug == collection_2.slug
        assert api.get_collection(collection_2.slug).slug == collection_2.slug

    def test_delete_collection(self, api: HfApi) -> None:
        collection = api.create_collection(self.title)

        api.delete_collection(collection.slug)

        # Cannot delete twice the same collection
        with pytest.raises(HfHubHTTPError):  # already exists
            api.delete_collection(collection.slug)

        # Possible to ignore error
        api.delete_collection(collection.slug, missing_ok=True)

    def test_collection_items(self, api: HfApi) -> None:
        # Create some repos
        model_id = api.create_repo(repo_name()).repo_id
        dataset_id = api.create_repo(repo_name(), repo_type="dataset").repo_id
        nested_collection_slug = api.create_collection(f"nested collection {repo_name()}").slug

        # Create collection + add items to it
        collection = api.create_collection(self.title)
        api.add_collection_item(collection.slug, model_id, "model", note="This is my model")
        api.add_collection_item(collection.slug, dataset_id, "dataset")  # note is optional
        api.add_collection_item(collection.slug, nested_collection_slug, "collection")

        # Check consistency
        collection = api.get_collection(collection.slug)
        assert len(collection.items) == 3
        assert collection.items[0].item_id == model_id
        assert collection.items[0].item_type == "model"
        assert collection.items[0].note == "This is my model"

        assert collection.items[1].item_id == dataset_id
        assert collection.items[1].item_type == "dataset"
        assert collection.items[1].note is None

        assert collection.items[2].item_id == nested_collection_slug
        assert collection.items[2].item_type == "collection"

        # Add existing item fails (except if ignore error)
        with pytest.raises(HfHubHTTPError):
            api.add_collection_item(collection.slug, model_id, "model")
        api.add_collection_item(collection.slug, model_id, "model", exists_ok=True)

        # Add inexistent item fails
        with pytest.raises(HfHubHTTPError):
            api.add_collection_item(collection.slug, model_id, "dataset")

        # Update first item
        api.update_collection_item(collection.slug, collection.items[0].item_object_id, note="New note", position=1)

        # Check consistency
        collection = api.get_collection(collection.slug)
        assert collection.items[0].item_id == dataset_id  # position got updated
        assert collection.items[1].item_id == model_id
        assert collection.items[1].note == "New note"  # note got updated

        # Delete last item
        api.delete_collection_item(collection.slug, collection.items[1].item_object_id)
        api.delete_collection_item(collection.slug, collection.items[1].item_object_id, missing_ok=True)

        # Check consistency
        collection = api.get_collection(collection.slug)
        assert len(collection.items) == 2  # only 1 item remaining
        assert collection.items[0].item_id == dataset_id  # position got updated

        # Delete everything
        api.delete_repo(model_id)
        api.delete_repo(dataset_id, repo_type="dataset")
        api.delete_collection(collection.slug)
        api.delete_collection(nested_collection_slug)

    @pytest.mark.production
    def test_collection_items_with_collections(self) -> None:
        collection = HfApi().get_collection("celinah/inference-providers-function-calling-6826023e8ae9b24b3039ee5f")
        assert len(collection.items) > 1
        assert collection.items[0].item_type == "collection"
        assert collection.items[0].item_id.startswith("celinah/")


class TestAccessRequestAPI:
    @pytest.fixture(autouse=True)
    def _gated_repo(self, api: HfApi):
        # Setup test with a gated repo
        self.repo_id = api.create_repo(repo_name()).repo_id
        response = get_session().put(
            f"{api.endpoint}/api/models/{self.repo_id}/settings",
            json={"gated": "auto"},
            headers=api._build_hf_headers(),
        )
        hf_raise_for_status(response)
        yield
        api.delete_repo(self.repo_id)

    def test_access_requests_normal_usage(self, api: HfApi) -> None:
        # No access requests initially
        requests = list(api.list_accepted_access_requests(self.repo_id))
        assert len(requests) == 0
        requests = list(api.list_pending_access_requests(self.repo_id))
        assert len(requests) == 0
        requests = list(api.list_rejected_access_requests(self.repo_id))
        assert len(requests) == 0

        # Grant access to a user
        api.grant_access(self.repo_id, OTHER_USER)

        # User is in accepted list
        requests = list(api.list_accepted_access_requests(self.repo_id))
        assert len(requests) == 1
        request = requests[0]
        assert isinstance(request, AccessRequest)
        assert request.username == OTHER_USER
        assert request.email is None  # email not shared when granted access manually
        assert request.status == "accepted"
        assert isinstance(request.timestamp, datetime.datetime)

        # Cancel access
        api.cancel_access_request(self.repo_id, OTHER_USER)
        requests = list(api.list_accepted_access_requests(self.repo_id))
        assert len(requests) == 0  # not accepted anymore
        requests = list(api.list_pending_access_requests(self.repo_id))
        assert len(requests) == 1
        assert requests[0].username == OTHER_USER

        # Reject access
        api.reject_access_request(self.repo_id, OTHER_USER, rejection_reason="This is a rejection reason")
        requests = list(api.list_pending_access_requests(self.repo_id))
        assert len(requests) == 0  # not pending anymore
        requests = list(api.list_rejected_access_requests(self.repo_id))
        assert len(requests) == 1
        assert requests[0].username == OTHER_USER

        # Accept again
        api.accept_access_request(self.repo_id, OTHER_USER)
        requests = list(api.list_accepted_access_requests(self.repo_id))
        assert len(requests) == 1
        assert requests[0].username == OTHER_USER

    def test_access_request_error(self, api: HfApi):
        # Grant access to a user
        api.grant_access(self.repo_id, OTHER_USER)

        # Cannot grant twice
        with pytest.raises(HfHubHTTPError):
            api.grant_access(self.repo_id, OTHER_USER)

        # Cannot accept to already accepted
        with pytest.raises(HfHubHTTPError):
            api.accept_access_request(self.repo_id, OTHER_USER)

        # Cannot reject to already rejected
        api.reject_access_request(self.repo_id, OTHER_USER, rejection_reason="This is a rejection reason")
        with pytest.raises(HfHubHTTPError):
            api.reject_access_request(self.repo_id, OTHER_USER, rejection_reason="This is a rejection reason")

        # Cannot cancel to already cancelled
        api.cancel_access_request(self.repo_id, OTHER_USER)
        with pytest.raises(HfHubHTTPError):
            api.cancel_access_request(self.repo_id, OTHER_USER)


@pytest.mark.production
class TestUserApi:
    @pytest.fixture(autouse=True)
    def _api(self):
        self.api = HfApi(endpoint=ENDPOINT_PRODUCTION)  # no auth!

    def test_user_overview(self) -> None:
        overview = self.api.get_user_overview("julien-c")
        assert overview.user_type == "user"
        assert overview.username == "julien-c"
        assert overview.num_likes > 10
        assert overview.num_upvotes > 10
        assert len(overview.orgs) > 0
        assert any(org.name == "huggingface" for org in overview.orgs)
        assert overview.num_following > 300
        assert overview.num_followers > 1000

    def test_organization_overview(self) -> None:
        overview = self.api.get_organization_overview("huggingface")
        assert overview.name == "huggingface"
        assert overview.fullname == "Hugging Face"
        assert overview.avatar_url.startswith("https://")
        assert overview.num_users is None or overview.num_users > 10
        assert overview.num_models is None or overview.num_models > 10
        assert overview.num_followers is None or overview.num_followers > 1000
        assert overview.num_papers is None or overview.num_papers >= 0

    def test_organization_members(self) -> None:
        members = self.api.list_organization_members("huggingface")
        assert len(list(members)) > 1

    def test_organization_followers(self) -> None:
        followers = self.api.list_organization_followers("huggingface")
        first_follower = next(followers)
        assert isinstance(first_follower, User)
        assert first_follower.username
        assert first_follower.fullname
        assert first_follower.avatar_url

    def test_user_followers(self) -> None:
        followers = self.api.list_user_followers("clem")
        assert len(list(followers)) > 500

    def test_user_following(self) -> None:
        following = self.api.list_user_following("clem")
        assert len(list(following)) > 500


@pytest.mark.production
class TestPaperApi:
    @pytest.fixture(autouse=True)
    def _api(self):
        self.api = HfApi(endpoint=ENDPOINT_PRODUCTION)

    def test_papers_by_query(self) -> None:
        papers = list(self.api.list_papers(query="llama"))
        assert len(papers) > 0
        assert "The Llama 3 Herd of Models" in [paper.title for paper in papers]

    def test_papers_by_query_with_limit(self) -> None:
        papers = list(self.api.list_papers(query="llama", limit=2))
        assert len(papers) == 2

    def test_get_paper_by_id_success(self) -> None:
        paper = self.api.paper_info("2407.21783")
        assert paper.title == "The Llama 3 Herd of Models"

    def test_get_paper_by_id_returns_linked_repos(self) -> None:
        paper = self.api.paper_info("2601.15621")
        assert paper.linked_models is not None and len(paper.linked_models) > 0
        assert all(isinstance(m, ModelInfo) for m in paper.linked_models)
        assert paper.num_total_models is not None and paper.num_total_models > 0
        assert paper.linked_datasets is not None
        assert all(isinstance(d, DatasetInfo) for d in paper.linked_datasets)
        assert paper.num_total_datasets is not None
        assert paper.linked_spaces is not None and len(paper.linked_spaces) > 0
        assert all(isinstance(s, SpaceInfo) for s in paper.linked_spaces)

    def test_get_paper_by_id_not_found(self) -> None:
        with pytest.raises(HfHubHTTPError) as context:
            self.api.paper_info("1234.56789")
        assert context.value.response.status_code == 404

    def test_list_daily_papers_by_date(self) -> None:
        papers = list(self.api.list_daily_papers(date="2025-10-29"))
        assert len(papers) > 0
        assert hasattr(papers[0], "id")
        assert hasattr(papers[0], "title")

    def test_list_daily_papers_by_date_invalid_date(self) -> None:
        with pytest.raises(BadRequestError):
            list(self.api.list_daily_papers(date="2025-13-40"))

    def test_list_daily_papers_default_date(self) -> None:
        papers = list(self.api.list_daily_papers())
        assert len(papers) > 0
        assert hasattr(papers[0], "id")
        assert hasattr(papers[0], "title")

    def test_list_daily_papers_week(self) -> None:
        week = 44
        papers = list(self.api.list_daily_papers(week=f"2025-W{week}"))
        assert len(papers) > 0
        first_paper = papers[0]
        last_paper = papers[-1]

        # friday of previous week
        week_start = datetime.datetime.fromisocalendar(2025, week - 1, 5).replace(tzinfo=datetime.timezone.utc)
        week_end = datetime.datetime.fromisocalendar(2025, week, 7).replace(tzinfo=datetime.timezone.utc)
        assert week_start <= first_paper.submitted_at <= week_end
        assert week_start <= last_paper.submitted_at <= week_end

    def test_list_daily_papers_month(self) -> None:
        month = 10
        papers = list(self.api.list_daily_papers(month=f"2025-{month}"))
        assert len(papers) > 0
        first_paper = papers[0]
        last_paper = papers[-1]
        # last day of previous month
        month_start = datetime.datetime(2025, month, 1, tzinfo=datetime.timezone.utc) - datetime.timedelta(days=1)
        month_end = datetime.datetime(2025, month + 1, 1, tzinfo=datetime.timezone.utc) - datetime.timedelta(days=1)
        assert month_start <= first_paper.submitted_at <= month_end
        assert month_start <= last_paper.submitted_at <= month_end

    def test_daily_papers_submitter(self) -> None:
        papers = list(self.api.list_daily_papers(submitter="akhaliq"))
        assert len(papers) > 0
        assert papers[0].submitted_by.fullname == "AK"

    def test_daily_papers_p(self) -> None:
        papers = list(self.api.list_daily_papers(date="2025-10-29", p=100))
        assert len(papers) == 0

    def test_daily_papers_limit(self) -> None:
        papers = list(self.api.list_daily_papers(date="2025-10-29", limit=10))
        assert len(papers) == 10


class TestWebhookApi:
    @pytest.fixture(autouse=True)
    def _webhook(self, api: HfApi):
        self.webhook_url = "https://webhook.site/test"
        self.watched_items = [
            WebhookWatchedItem(type="user", name="julien-c"),  # can be either a dataclass
            {"type": "org", "name": "HuggingFaceH4"},  # or a simple dictionary
        ]
        self.domains = ["repo", "discussion"]
        self.secret = "my-secret"

        # Create a webhook to be used in the tests
        self.webhook = api.create_webhook(
            url=self.webhook_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        yield
        # Clean up the created webhook
        api.delete_webhook(self.webhook.id)

    def test_get_webhook(self, api: HfApi) -> None:
        webhook = api.get_webhook(self.webhook.id)
        assert isinstance(webhook, WebhookInfo)
        assert webhook.id == self.webhook.id
        assert webhook.url == self.webhook_url

    def test_list_webhooks(self, api: HfApi) -> None:
        webhooks = api.list_webhooks()
        assert any(webhook.id == self.webhook.id for webhook in webhooks)

    def test_create_webhook(self, api: HfApi) -> None:
        new_webhook = api.create_webhook(
            url=self.webhook_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        assert isinstance(new_webhook, WebhookInfo)
        assert new_webhook.url == self.webhook_url

        # Clean up the newly created webhook
        api.delete_webhook(new_webhook.id)

    def test_update_webhook(self, api: HfApi) -> None:
        updated_url = "https://webhook.site/new"
        updated_webhook = api.update_webhook(
            self.webhook.id, url=updated_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        assert updated_webhook.url == updated_url

    def test_enable_webhook(self, api: HfApi) -> None:
        enabled_webhook = api.enable_webhook(self.webhook.id)
        assert not enabled_webhook.disabled

    def test_disable_webhook(self, api: HfApi) -> None:
        disabled_webhook = api.disable_webhook(self.webhook.id)
        assert disabled_webhook.disabled

    def test_delete_webhook(self, api: HfApi) -> None:
        # Create another webhook to test deletion
        webhook_to_delete = api.create_webhook(
            url=self.webhook_url, watched=self.watched_items, domains=self.domains, secret=self.secret
        )
        api.delete_webhook(webhook_to_delete.id)
        with pytest.raises(HfHubHTTPError):
            api.get_webhook(webhook_to_delete.id)


class TestExpandPropertyType:
    def test_expand_model_property_type_is_up_to_date(self, api: HfApi, repo_factory: RepoFactory):
        repo_url = repo_factory("model")
        self._check_expand_property_is_up_to_date(api, repo_url)

    def test_expand_dataset_property_type_is_up_to_date(self, api: HfApi, repo_factory: RepoFactory):
        repo_url = repo_factory("dataset")
        self._check_expand_property_is_up_to_date(api, repo_url)

    def test_expand_space_property_type_is_up_to_date(self, api: HfApi, repo_factory: RepoFactory):
        repo_url = repo_factory("space")
        self._check_expand_property_is_up_to_date(api, repo_url)

    def _check_expand_property_is_up_to_date(self, api: HfApi, repo_url: RepoUrl):
        repo_id = repo_url.repo_id
        repo_type = repo_url.repo_type
        property_type = (
            ExpandModelProperty_T
            if repo_type == "model"
            else (ExpandDatasetProperty_T if repo_type == "dataset" else ExpandSpaceProperty_T)
        )
        property_type_name = (
            "ExpandModelProperty_T"
            if repo_type == "model"
            else ("ExpandDatasetProperty_T" if repo_type == "dataset" else "ExpandSpaceProperty_T")
        )

        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, expand=["does_not_exist"])
            raise Exception("Should have raised an exception")
        except HfHubHTTPError as e:
            assert e.response.status_code == 400
            message = e.response.json()["error"]

        assert message.startswith('"expand" must be one of ')
        defined_args = set(get_args(property_type))
        expected_args = set(message.replace('"expand" must be one of ', "").strip("[]").split(", "))
        expected_args.discard("gitalyUid")  # internal one, do not document
        expected_args.discard("xetEnabled")  # all repos are xetEnabled now, so we don't document it anymore

        if defined_args != expected_args:
            should_be_removed = defined_args - expected_args
            should_be_added = expected_args - defined_args

            msg = f"Literal `{property_type_name}` is outdated! This is probably due to a server-side update."
            if should_be_removed:
                msg += f"\nArg(s) not supported anymore: {', '.join(should_be_removed)}"
            if should_be_added:
                msg += f"\nNew arg(s) to support: {', '.join(should_be_added)}"
            msg += f"\nPlease open a PR to update `./src/huggingface_hub/hf_api.py` accordingly. `{property_type_name}` should be updated as well as `{repo_type}_info` and `list_{repo_type}s` docstrings."
            msg += "\nThank you in advance!"
            raise ValueError(msg)


class TestLargeUpload:
    def test_upload_large_folder(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory("dataset")
        N_FILES_PER_FOLDER = 4

        with SoftTemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "large_folder"
            # Create 16 LFS files + 16 regular files
            for i in range(N_FILES_PER_FOLDER):
                subfolder = folder / f"subfolder_{i}"
                subfolder.mkdir(parents=True, exist_ok=True)
                for j in range(N_FILES_PER_FOLDER):
                    (subfolder / f"file_lfs_{i}_{j}.bin").write_bytes(f"content_lfs_{i}_{j}".encode())
                    (subfolder / f"file_regular_{i}_{j}.txt").write_bytes(f"content_regular_{i}_{j}".encode())

            # Upload the folder
            with pytest.warns(FutureWarning, match="`upload_large_folder` is DEPRECATED"):
                api.upload_large_folder(
                    repo_id=repo_url.repo_id, repo_type=repo_url.repo_type, folder_path=folder, num_workers=4
                )

        # Check all files have been uploaded
        uploaded_files = api.list_repo_files(repo_url.repo_id, repo_type=repo_url.repo_type)
        for i in range(N_FILES_PER_FOLDER):
            for j in range(N_FILES_PER_FOLDER):
                assert f"subfolder_{i}/file_lfs_{i}_{j}.bin" in uploaded_files
                assert f"subfolder_{i}/file_regular_{i}_{j}.txt" in uploaded_files


class TestHfApiAuthCheck:
    def test_auth_check_success(self, api: HfApi, repo_factory: RepoFactory) -> None:
        repo_url = repo_factory("dataset")
        api.auth_check(repo_id=repo_url.repo_id, repo_type=repo_url.repo_type)

    def test_auth_check_repo_missing(self, api: HfApi) -> None:
        with pytest.raises(RepositoryNotFoundError):
            api.auth_check(repo_id="username/missing_repo_id")

    def test_auth_check_gated_repo(self, api: HfApi) -> None:
        repo_id = api.create_repo(repo_name()).repo_id

        response = get_session().put(
            f"{api.endpoint}/api/models/{repo_id}/settings",
            json={"gated": "auto"},
            headers=api._build_hf_headers(token=TOKEN),
        )

        hf_raise_for_status(response)

        with pytest.raises(GatedRepoError):
            api.auth_check(repo_id=repo_id, token=OTHER_TOKEN)


class TestHfApiInferenceCatalog:
    def test_list_inference_catalog(self, api: HfApi) -> None:
        models = api.list_inference_catalog()  # note: @experimental api
        # Check that server returns a list[str] => at least if it changes in the future, we'll notice
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    def test_create_inference_endpoint_from_catalog(self, api: HfApi, mocker) -> None:
        mock_get_session = mocker.patch("huggingface_hub.hf_api.get_session")
        mock_response = Mock()
        mock_response.json.return_value = {
            "endpoint": {
                "compute": {
                    "accelerator": "gpu",
                    "id": "aws-us-east-1-nvidia-l4-x1",
                    "instanceSize": "x1",
                    "instanceType": "nvidia-l4",
                    "scaling": {
                        "maxReplica": 1,
                        "measure": {"hardwareUsage": None},
                        "metric": "hardwareUsage",
                        "minReplica": 0,
                        "scaleToZeroTimeout": 15,
                    },
                },
                "model": {
                    "env": {},
                    "framework": "pytorch",
                    "image": {
                        "tgi": {
                            "disableCustomKernels": False,
                            "healthRoute": "/health",
                            "port": 80,
                            "url": "ghcr.io/huggingface/text-generation-inference:3.1.1",
                        }
                    },
                    "repository": "meta-llama/Llama-3.2-3B-Instruct",
                    "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
                    "secrets": {},
                    "task": "text-generation",
                },
                "name": "llama-3-2-3b-instruct-eey",
                "provider": {"region": "us-east-1", "vendor": "aws"},
                "healthRoute": "/health",
                "status": {
                    "createdAt": "2025-03-07T15:30:13.949Z",
                    "createdBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
                    "message": "Endpoint waiting to be scheduled",
                    "readyReplica": 0,
                    "state": "pending",
                    "targetReplica": 1,
                    "updatedAt": "2025-03-07T15:30:13.949Z",
                    "updatedBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
                },
                "type": "protected",
            }
        }
        mock_get_session.return_value.post.return_value = mock_response

        endpoint = api.create_inference_endpoint_from_catalog(
            repo_id="meta-llama/Llama-3.2-3B-Instruct", namespace="Wauplin"
        )
        assert isinstance(endpoint, InferenceEndpoint)
        assert endpoint.name == "llama-3-2-3b-instruct-eey"


@pytest.mark.parametrize(
    "custom_image, expected_image_payload",
    [
        # Case 1: No custom_image provided
        (
            None,
            {
                "huggingface": {},
            },
        ),
        # Case 2: Flat dictionary custom_image provided
        (
            {
                "url": "my.registry/my-image:latest",
                "port": 8080,
            },
            {
                "custom": {
                    "url": "my.registry/my-image:latest",
                    "port": 8080,
                }
            },
        ),
        # Case 3: Explicitly keyed ('tgi') custom_image provided
        (
            {
                "tgi": {
                    "url": "ghcr.io/huggingface/text-generation-inference:latest",
                }
            },
            {
                "tgi": {
                    "url": "ghcr.io/huggingface/text-generation-inference:latest",
                }
            },
        ),
        # Case 4: Explicitly keyed ('custom') custom_image provided
        (
            {
                "custom": {
                    "url": "another.registry/custom:v2",
                }
            },
            {
                "custom": {
                    "url": "another.registry/custom:v2",
                }
            },
        ),
    ],
    ids=["no_custom_image", "flat_dict_custom_image", "keyed_tgi_custom_image", "keyed_custom_custom_image"],
)
def test_create_inference_endpoint_custom_image_payload(
    mocker,
    custom_image: Optional[dict],
    expected_image_payload: dict,
):
    mock_post = mocker.patch("huggingface_hub.hf_api.get_session")
    common_args = {
        "name": "test-endpoint-custom-img",
        "repository": "meta-llama/Llama-2-7b-chat-hf",
        "framework": "pytorch",
        "accelerator": "gpu",
        "instance_size": "medium",
        "instance_type": "nvidia-a10g",
        "region": "us-east-1",
        "vendor": "aws",
        "type": "authenticated",
        "task": "text-generation",
        "namespace": "Wauplin",
    }
    mock_session = mock_post.return_value
    mock_post_method = mock_session.post
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "compute": {
            "accelerator": "gpu",
            "id": "aws-us-east-1-nvidia-l4-x1",
            "instanceSize": "x1",
            "instanceType": "nvidia-l4",
            "scaling": {
                "maxReplica": 1,
                "measure": {"hardwareUsage": None},
                "metric": "hardwareUsage",
                "minReplica": 0,
                "scaleToZeroTimeout": 15,
            },
        },
        "model": {
            "env": {},
            "framework": "pytorch",
            "image": {
                "tgi": {
                    "disableCustomKernels": False,
                    "healthRoute": "/health",
                    "port": 80,
                    "url": "ghcr.io/huggingface/text-generation-inference:3.1.1",
                }
            },
            "repository": "meta-llama/Llama-3.2-3B-Instruct",
            "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
            "secrets": {},
            "task": "text-generation",
        },
        "name": "llama-3-2-3b-instruct-eey",
        "provider": {"region": "us-east-1", "vendor": "aws"},
        "healthRoute": "/health",
        "status": {
            "createdAt": "2025-03-07T15:30:13.949Z",
            "createdBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
            "message": "Endpoint waiting to be scheduled",
            "readyReplica": 0,
            "state": "pending",
            "targetReplica": 1,
            "updatedAt": "2025-03-07T15:30:13.949Z",
            "updatedBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        },
        "type": "protected",
    }
    mock_post_method.return_value = mock_response

    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    if custom_image is not None:
        api.create_inference_endpoint(custom_image=custom_image, **common_args)
    else:
        api.create_inference_endpoint(**common_args)

    mock_post_method.assert_called_once()
    _, call_kwargs = mock_post_method.call_args
    payload = call_kwargs.get("json", {})

    assert "model" in payload and "image" in payload["model"]
    assert payload["model"]["image"] == expected_image_payload


def test_create_inference_endpoint_container_command_and_args_payload(mocker):
    mock_post = mocker.patch("huggingface_hub.hf_api.get_session")
    mock_session = mock_post.return_value
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "name": "sglang-endpoint",
        "model": {"repository": "nex-agi/Nex-N2-Pro", "framework": "custom", "revision": None, "task": None},
        "status": {
            "state": "pending",
            "createdAt": "2025-03-07T15:30:13.949Z",
            "updatedAt": "2025-03-07T15:30:13.949Z",
        },
        "healthRoute": "/health",
        "type": "authenticated",
    }
    mock_session.post.return_value = mock_response

    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    api.create_inference_endpoint(
        name="sglang-endpoint",
        repository="nex-agi/Nex-N2-Pro",
        framework="custom",
        accelerator="gpu",
        instance_size="x8",
        instance_type="nvidia-h200",
        region="us-east-1",
        vendor="aws",
        type="authenticated",
        namespace="Wauplin",
        custom_image={"url": "nexagi/sglang:v0.5.12", "healthRoute": "/health", "port": 30000},
        container_command=["python", "-m", "sglang.launch_server"],
        container_args=["--tp", "8", "--reasoning-parser", "qwen3"],
    )

    _, call_kwargs = mock_session.post.call_args
    payload = call_kwargs.get("json", {})
    assert payload["model"]["command"] == ["python", "-m", "sglang.launch_server"]
    assert payload["model"]["args"] == ["--tp", "8", "--reasoning-parser", "qwen3"]
    assert payload["model"]["image"] == {
        "custom": {"url": "nexagi/sglang:v0.5.12", "healthRoute": "/health", "port": 30000}
    }


class TestHfApiVerifyChecksums:
    def test_verify_repo_checksums_with_local_cache(self, api: HfApi) -> None:
        repo_id = api.create_repo(repo_name()).repo_id
        api.create_commit(
            repo_id=repo_id,
            commit_message="add file",
            operations=[CommitOperationAdd(path_or_fileobj=b"data", path_in_repo="file.txt")],
        )

        # minimal cache layout
        info = api.repo_info(repo_id)
        commit = info.sha
        parts = [f"{constants.REPO_TYPE_MODEL}s", *repo_id.split("/")]
        repo_folder_name = constants.REPO_ID_SEPARATOR.join(parts)

        storage = Path(constants.HF_HUB_CACHE) / repo_folder_name
        snapshot = storage / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "file.txt").write_bytes(b"data")

        res = api.verify_repo_checksums(repo_id=repo_id, revision=commit, cache_dir=storage.parent)
        assert res.revision == commit and res.checked_count == 1 and not res.mismatches


def _to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case (e.g. 'downloadsAllTime' -> 'downloads_all_time', 'model-index' -> 'model_index')."""
    import re

    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).replace("-", "_").lower()
