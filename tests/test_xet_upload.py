# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import HfApi, RepoUrl
from huggingface_hub._commit_api import _upload_lfs_files, _upload_xet_files
from huggingface_hub.file_download import get_hf_file_metadata, hf_hub_download, hf_hub_url

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name, requires


@contextmanager
def assert_xet_upload_used(should_be_called: bool):
    with patch("huggingface_hub.hf_api._upload_xet_files", wraps=_upload_xet_files) as mock_upload_xet_api:
        yield
        assert mock_upload_xet_api.called == should_be_called


@contextmanager
def assert_lfs_upload_used(should_be_called: bool):
    with patch("huggingface_hub.hf_api._upload_lfs_files", wraps=_upload_lfs_files) as mock_upload_lfs_api:
        yield
        assert mock_upload_lfs_api.called == should_be_called


@pytest.fixture(scope="module")
def api():
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture
def repo_url(api, repo_type: str = "model"):
    repo_url = api.create_repo(repo_id=repo_name(prefix=repo_type), repo_type=repo_type)
    api.update_repo_settings(repo_id=repo_url.repo_id, xet_enabled=True)

    yield repo_url

    api.delete_repo(repo_id=repo_url.repo_id, repo_type=repo_type)


@requires("hf_xet")
@pytest.mark.timeout(60)
class TestXetUpload:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.folder_path = tmp_path
        # Create a regular text file
        text_file = self.folder_path / "text_file.txt"
        self.text_content = "This is a regular text file"
        text_file.write_text(self.text_content)

        # Create a binary file
        self.bin_file = self.folder_path / "binary_file.bin"
        self.bin_content = b"0" * (1 * 1024 * 1024)
        self.bin_file.write_bytes(self.bin_content)

        # Create nested directory structure
        nested_dir = self.folder_path / "nested"
        nested_dir.mkdir()

        # Create a nested text file
        nested_text_file = nested_dir / "nested_text.txt"
        self.nested_text_content = "This is a nested text file"
        nested_text_file.write_text(self.nested_text_content)

        # Create a nested binary file
        nested_bin_file = nested_dir / "nested_binary.safetensors"
        self.nested_bin_content = b"1" * (1 * 1024 * 1024)
        nested_bin_file.write_bytes(self.nested_bin_content)

    def test_upload_file(self, api, tmp_path, repo_url):
        filename_in_repo = "binary_file.bin"
        repo_id = repo_url.repo_id
        with assert_xet_upload_used(should_be_called=True):
            return_val = api.upload_file(
                path_or_fileobj=self.bin_file,
                path_in_repo=filename_in_repo,
                repo_id=repo_id,
            )

        assert return_val == f"{api.endpoint}/{repo_id}/blob/main/{filename_in_repo}"
        # Download and verify content
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename_in_repo, cache_dir=tmp_path)
        with open(downloaded_file, "rb") as f:
            downloaded_content = f.read()
            assert downloaded_content == self.bin_content

        # Check xet metadata
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename_in_repo,
        )
        metadata = get_hf_file_metadata(url)
        xet_metadata = metadata.xet_metadata
        assert xet_metadata is not None

    def test_upload_file_with_bytesio(self, api, tmp_path, repo_url):
        repo_id = repo_url.repo_id
        content = BytesIO(self.bin_content)
        with assert_lfs_upload_used(should_be_called=True):
            api.upload_file(
                path_or_fileobj=content,
                path_in_repo="bytesio_file.bin",
                repo_id=repo_id,
            )
        # Download and verify content
        downloaded_file = hf_hub_download(repo_id=repo_id, filename="bytesio_file.bin", cache_dir=tmp_path)
        with open(downloaded_file, "rb") as f:
            downloaded_content = f.read()
            assert downloaded_content == self.bin_content

    def test_fallback_to_lfs_when_xet_not_available(self, api, repo_url):
        repo_id = repo_url.repo_id
        with patch("huggingface_hub.hf_api.is_xet_available", return_value=False):
            with assert_lfs_upload_used(should_be_called=True):
                with assert_xet_upload_used(should_be_called=False):
                    api.upload_file(
                        path_or_fileobj=self.bin_file,
                        path_in_repo="fallback_file.bin",
                        repo_id=repo_id,
                    )

    def test_upload_based_on_xet_enabled_setting(self, api, repo_url):
        repo_id = repo_url.repo_id

        # Test when xet is enabled -> use Xet upload
        with patch("huggingface_hub.hf_api.HfApi.repo_info") as mock_repo_info:
            mock_repo_info.return_value.xet_enabled = True
            with assert_xet_upload_used(should_be_called=True):
                with assert_lfs_upload_used(should_be_called=False):
                    api.upload_file(
                        path_or_fileobj=self.bin_file,
                        path_in_repo="xet_enabled.bin",
                        repo_id=repo_id,
                    )

        # Test when xet is disabled -> use LFS upload
        with patch("huggingface_hub.hf_api.HfApi.repo_info") as mock_repo_info:
            mock_repo_info.return_value.xet_enabled = False
            with assert_xet_upload_used(should_be_called=False):
                with assert_lfs_upload_used(should_be_called=True):
                    api.upload_file(
                        path_or_fileobj=self.bin_file,
                        path_in_repo="xet_disabled.bin",
                        repo_id=repo_id,
                    )

    @pytest.mark.timeout(6)
    def test_upload_folder(self, api, repo_url):
        repo_id = repo_url.repo_id
        folder_in_repo = "temp"
        with assert_xet_upload_used(should_be_called=True):
            return_val = api.upload_folder(
                folder_path=self.folder_path,
                path_in_repo=folder_in_repo,
                repo_id=repo_id,
            )

        assert return_val == f"{api.endpoint}/{repo_id}/tree/main/{folder_in_repo}"
        files_in_repo = set(api.list_repo_files(repo_id=repo_id))
        files = {
            f"{folder_in_repo}/text_file.txt",
            f"{folder_in_repo}/binary_file.bin",
            f"{folder_in_repo}/nested/nested_text.txt",
            f"{folder_in_repo}/nested/nested_binary.safetensors",
        }
        assert all(file in files_in_repo for file in files)

        for rpath in files:
            local_file = Path(rpath).relative_to(folder_in_repo)
            local_path = self.folder_path / local_file
            filepath = hf_hub_download(repo_id=repo_id, filename=rpath)
            assert Path(local_path).read_bytes() == Path(filepath).read_bytes()

    @pytest.mark.timeout(6)
    def test_upload_folder_create_pr(self, api, repo_url) -> None:
        repo_id = repo_url.repo_id
        folder_in_repo = "temp_create_pr"
        with assert_xet_upload_used(should_be_called=True):
            return_val = api.upload_folder(
                folder_path=self.folder_path,
                path_in_repo=folder_in_repo,
                repo_id=repo_id,
                create_pr=True,
            )

        assert return_val == f"{api.endpoint}/{repo_id}/tree/refs%2Fpr%2F1/{folder_in_repo}"

        for rpath in ["text_file.txt", "nested/nested_binary.safetensors"]:
            local_path = self.folder_path / rpath
            filepath = hf_hub_download(
                repo_id=repo_id, filename=f"{folder_in_repo}/{rpath}", revision=return_val.pr_revision
            )
            assert Path(local_path).read_bytes() == Path(filepath).read_bytes()


@requires("hf_xet")
@pytest.mark.skip("Skipping large upload to debug")
class TestXetLargeUpload:
    def test_upload_large_folder(self, api, tmp_path, repo_url: RepoUrl) -> None:
        N_FILES_PER_FOLDER = 4
        repo_id = repo_url.repo_id

        folder = Path(tmp_path) / "large_folder"
        for i in range(N_FILES_PER_FOLDER):
            subfolder = folder / f"subfolder_{i}"
            subfolder.mkdir(parents=True, exist_ok=True)
            for j in range(N_FILES_PER_FOLDER):
                (subfolder / f"file_xet_{i}_{j}.bin").write_bytes(f"content_lfs_{i}_{j}".encode())
                (subfolder / f"file_regular_{i}_{j}.txt").write_bytes(f"content_regular_{i}_{j}".encode())

            with assert_xet_upload_used(should_be_called=True):
                api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=folder, num_workers=4)

        # Check all files have been uploaded
        uploaded_files = api.list_repo_files(repo_id=repo_id)
        for i in range(N_FILES_PER_FOLDER):
            for j in range(N_FILES_PER_FOLDER):
                assert f"subfolder_{i}/file_xet_{i}_{j}.bin" in uploaded_files
                assert f"subfolder_{i}/file_regular_{i}_{j}.txt" in uploaded_files

            # Check xet metadata
            url = hf_hub_url(
                repo_id=repo_id,
                filename=f"subfolder_{i}/file_xet_{i}_{j}.bin",
            )
            metadata = get_hf_file_metadata(url)
            xet_metadata = metadata.xet_metadata
            assert xet_metadata is not None
