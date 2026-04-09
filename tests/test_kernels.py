# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
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
"""Tests specific to kernel repo type."""

import os

import pytest

from huggingface_hub import HfApi, KernelInfo, RepoUrl
from huggingface_hub.errors import RemoteEntryNotFoundError, RevisionNotFoundError

from .testing_constants import ENDPOINT_PRODUCTION, ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


KERNEL_TEST_REPO_ID = "kernels-community/relu"
KERNEL_TEST_REPO_FILE = "build/torch-ext/torch_binding.cpp"


def kernel_name() -> str:
    return repo_name(prefix="kernel")


@pytest.fixture(scope="module")
def staging_api():
    """HfApi for staging environment."""
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture(scope="module")
def api():
    """HfApi for production environment (unauthenticated)"""
    return HfApi(endpoint=ENDPOINT_PRODUCTION)


def test_create_kernel(staging_api: HfApi) -> None:
    name = kernel_name()
    repo_url = staging_api.create_repo(name, repo_type="kernel")
    assert isinstance(repo_url, RepoUrl)

    assert repo_url.repo_type == "kernel"
    assert repo_url.namespace == USER
    assert repo_url.repo_name == name
    assert repo_url.repo_id == f"{USER}/{name}"
    assert repo_url.url == f"https://hub-ci.huggingface.co/kernels/{USER}/{name}"
    assert repo_url.endpoint == ENDPOINT_STAGING

    staging_api.delete_repo(repo_url.repo_id, repo_type="kernel")


def test_kernel_info(api: HfApi) -> None:
    kernel_info = api.kernel_info(KERNEL_TEST_REPO_ID)
    assert isinstance(kernel_info, KernelInfo)
    assert kernel_info.id == KERNEL_TEST_REPO_ID
    assert kernel_info.author == "kernels-community"


def test_list_repo_files(api: HfApi) -> None:
    """Test listing files from kernel repo works."""
    files = api.list_repo_files(KERNEL_TEST_REPO_ID, repo_type="kernel")
    assert len(files) > 0
    assert KERNEL_TEST_REPO_FILE in files

    files = api.list_repo_files(KERNEL_TEST_REPO_ID, repo_type="kernel", revision="v1")
    assert len(files) > 0
    assert KERNEL_TEST_REPO_FILE in files

    with pytest.raises(RevisionNotFoundError):
        api.list_repo_files(KERNEL_TEST_REPO_ID, repo_type="kernel", revision="invalid")


def test_list_repo_refs(api: HfApi) -> None:
    """Test listing refs from kernel repo works."""
    refs = api.list_repo_refs(KERNEL_TEST_REPO_ID, repo_type="kernel")
    assert any(ref.name == "main" for ref in refs.branches)
    assert any(ref.name == "v1" for ref in refs.branches)


def test_list_repo_tree(api: HfApi) -> None:
    """Test listing tree from kernel repo works."""
    tree = list(api.list_repo_tree(KERNEL_TEST_REPO_ID, repo_type="kernel"))
    assert len(tree) > 0
    assert any(tree_obj.path == ".gitattributes" for tree_obj in tree)

    # specific revision + recursive
    tree = list(api.list_repo_tree(KERNEL_TEST_REPO_ID, repo_type="kernel", revision="v1", recursive=True))
    assert any(tree_obj.path == KERNEL_TEST_REPO_FILE for tree_obj in tree)  # file in subfolder

    with pytest.raises(RevisionNotFoundError):
        list(api.list_repo_tree(KERNEL_TEST_REPO_ID, repo_type="kernel", revision="invalid"))


def test_download_existing_file(api: HfApi, tmp_path) -> None:
    """Test downloading file from kernel repo works."""
    file_path = api.hf_hub_download(KERNEL_TEST_REPO_ID, KERNEL_TEST_REPO_FILE, repo_type="kernel", cache_dir=tmp_path)
    assert os.path.isfile(file_path)
    assert "kernels--kernels-community--relu" in file_path  # kernel path


def test_download_missing_file(api: HfApi, tmp_path) -> None:
    """Test downloading missing file from kernel repo works."""
    with pytest.raises(RemoteEntryNotFoundError):
        api.hf_hub_download(KERNEL_TEST_REPO_ID, "missing.md", repo_type="kernel", cache_dir=tmp_path)


def test_redownload_file_offline_mode(api: HfApi, tmp_path) -> None:
    """Test re-downloading file from kernel repo works in offline mode."""
    path_1 = api.hf_hub_download(KERNEL_TEST_REPO_ID, KERNEL_TEST_REPO_FILE, repo_type="kernel", cache_dir=tmp_path)

    path_2 = api.hf_hub_download(
        KERNEL_TEST_REPO_ID, KERNEL_TEST_REPO_FILE, repo_type="kernel", local_files_only=True, cache_dir=tmp_path
    )
    assert path_1 == path_2


def test_download_file_from_revision(api: HfApi, tmp_path) -> None:
    """Test downloading file from revision works."""
    path_from_main = api.hf_hub_download(
        KERNEL_TEST_REPO_ID, KERNEL_TEST_REPO_FILE, repo_type="kernel", cache_dir=tmp_path
    )
    path_from_v1 = api.hf_hub_download(
        KERNEL_TEST_REPO_ID, KERNEL_TEST_REPO_FILE, repo_type="kernel", cache_dir=tmp_path, revision="v1"
    )
    assert path_from_main != path_from_v1


def test_snapshot_download_allow_patterns(api: HfApi, tmp_path) -> None:
    """Test partial downloading from kernel repo works."""
    path = api.snapshot_download(
        KERNEL_TEST_REPO_ID, repo_type="kernel", cache_dir=tmp_path, allow_patterns="build/torch-ext/*"
    )
    assert os.path.isdir(path)
    assert "kernels--kernels-community--relu" in path  # kernel path
    assert "snapshots" in path
    assert os.path.isfile(os.path.join(path, "build", "torch-ext", "torch_binding.cpp"))
