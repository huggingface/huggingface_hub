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
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub import HfApi, RepoUrl
from huggingface_hub._commit_api import CommitOperationAdd, _upload_files, _upload_lfs_files, _upload_xet_files
from huggingface_hub.file_download import (
    _get_metadata_or_catch_error,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from huggingface_hub.utils import build_hf_headers, refresh_xet_connection_info

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name, requires


@contextmanager
def assert_upload_mode(mode: str):
    if mode not in ("xet", "lfs"):
        raise ValueError("Mode must be either 'xet' or 'lfs'")

    with patch("huggingface_hub._commit_api._upload_xet_files", wraps=_upload_xet_files) as mock_xet:
        with patch("huggingface_hub._commit_api._upload_lfs_files", wraps=_upload_lfs_files) as mock_lfs:
            yield
            assert mock_xet.called == (mode == "xet"), (
                f"Expected {'XET' if mode == 'xet' else 'LFS'} upload to be used"
            )
            assert mock_lfs.called == (mode == "lfs"), (
                f"Expected {'LFS' if mode == 'lfs' else 'XET'} upload to be used"
            )


@pytest.fixture(scope="module")
def api():
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture
def repo_url(api, repo_type: str = "model"):
    repo_url = api.create_repo(repo_id=repo_name(prefix=repo_type), repo_type=repo_type)

    yield repo_url

    api.delete_repo(repo_id=repo_url.repo_id, repo_type=repo_type)


@pytest.fixture
def xet_setup(request, tmp_path):
    instance = getattr(request, "instance", None)
    if instance is None:
        yield
        return
    instance.folder_path = tmp_path
    # Create a regular text file
    text_file = instance.folder_path / "text_file.txt"
    instance.text_content = "This is a regular text file"
    text_file.write_text(instance.text_content)

    # Create a binary file
    instance.bin_file = instance.folder_path / "binary_file.bin"
    instance.bin_content = b"0" * (1 * 1024 * 1024)
    instance.bin_file.write_bytes(instance.bin_content)

    # Create nested directory structure
    nested_dir = instance.folder_path / "nested"
    nested_dir.mkdir()

    # Create a nested text file
    nested_text_file = nested_dir / "nested_text.txt"
    instance.nested_text_content = "This is a nested text file"
    nested_text_file.write_text(instance.nested_text_content)

    # Create a nested binary file
    nested_bin_file = nested_dir / "nested_binary.safetensors"
    instance.nested_bin_content = b"1" * (1 * 1024 * 1024)
    nested_bin_file.write_bytes(instance.nested_bin_content)
    yield


@requires("hf_xet")
@pytest.mark.usefixtures("xet_setup")
class TestXetUpload:
    def test_upload_file(self, api, tmp_path, repo_url):
        filename_in_repo = "binary_file.bin"
        repo_id = repo_url.repo_id
        with assert_upload_mode("xet"):
            return_val = api.upload_file(
                path_or_fileobj=self.bin_file,
                path_in_repo=filename_in_repo,
                repo_id=repo_id,
            )
        assert return_val.startswith(f"{api.endpoint}/{repo_id}/commit")

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
        assert metadata.xet_file_data is not None
        xet_connection = refresh_xet_connection_info(file_data=metadata.xet_file_data, headers={})
        assert xet_connection is not None

    def test_upload_file_with_bytesio(self, api, tmp_path, repo_url):
        repo_id = repo_url.repo_id
        content = BytesIO(self.bin_content)
        with assert_upload_mode("lfs"):
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

    def test_upload_file_with_byte_array(self, api, tmp_path, repo_url):
        repo_id = repo_url.repo_id
        content = self.bin_content
        with assert_upload_mode("xet"):
            api.upload_file(
                path_or_fileobj=content,
                path_in_repo="bytearray_file.bin",
                repo_id=repo_id,
            )
        # Download and verify content
        downloaded_file = hf_hub_download(repo_id=repo_id, filename="bytearray_file.bin", cache_dir=tmp_path)
        with open(downloaded_file, "rb") as f:
            downloaded_content = f.read()
            assert downloaded_content == self.bin_content

    def test_fallback_to_lfs_when_xet_not_available(self, api, repo_url):
        repo_id = repo_url.repo_id
        with patch("huggingface_hub._commit_api.is_xet_available", return_value=False):
            with assert_upload_mode("lfs"):
                api.upload_file(
                    path_or_fileobj=self.bin_file,
                    path_in_repo="fallback_file.bin",
                    repo_id=repo_id,
                )

    def test_transfers_to_xet_when_server_returns_xet(self):
        addition = CommitOperationAdd(path_in_repo="xet.bin", path_or_fileobj=self.bin_file)

        def fake_batch(
            upload_infos, token, repo_type, repo_id, revision=None, endpoint=None, headers=None, transfers=None
        ):
            action = {
                "oid": upload_infos[0].sha256.hex(),
                "size": upload_infos[0].size,
                "actions": {"upload": {"href": "https://example.invalid", "header": {}}},
            }
            return ([action], [], "xet")

        with patch("huggingface_hub._commit_api.post_lfs_batch_info", side_effect=fake_batch) as mock_batch:
            with patch("huggingface_hub._commit_api._upload_lfs_files") as mock_lfs:
                with patch("huggingface_hub._commit_api._upload_xet_files") as mock_xet:
                    _upload_files(
                        additions=[addition],
                        repo_type="model",
                        repo_id="dummy/user-repo",
                        headers={},
                        endpoint="https://hub-ci.huggingface.co",
                        revision="main",
                        create_pr=False,
                    )
            assert mock_batch.call_count == 1
            mock_xet.assert_called_once()
            mock_lfs.assert_not_called()

    def test_transfers_bytesio_renegotiates_to_lfs_when_server_returns_xet(self):
        addition = CommitOperationAdd(path_in_repo="bytesio.bin", path_or_fileobj=BytesIO(self.bin_content))

        def fake_batch(
            upload_infos, token, repo_type, repo_id, revision=None, endpoint=None, headers=None, transfers=None
        ):
            action = {
                "oid": upload_infos[0].sha256.hex(),
                "size": upload_infos[0].size,
                "actions": {"upload": {"href": "https://example.invalid", "header": {}}},
            }
            return ([action], [], "xet")

        with patch("huggingface_hub._commit_api.post_lfs_batch_info", side_effect=fake_batch) as mock_batch:
            with patch("huggingface_hub._commit_api._upload_lfs_files") as mock_lfs:
                with patch("huggingface_hub._commit_api._upload_xet_files") as mock_xet:
                    _upload_files(
                        additions=[addition],
                        repo_type="model",
                        repo_id="dummy/user-repo",
                        headers={},
                        endpoint="https://hub-ci.huggingface.co",
                        revision="main",
                        create_pr=False,
                    )

            # Ensure we retried negotiation and routed to LFS, not XET
            assert mock_batch.call_count == 1
            mock_xet.assert_not_called()
            mock_lfs.assert_called_once()

    def test_upload_folder(self, api, repo_url):
        repo_id = repo_url.repo_id
        folder_in_repo = "temp"
        with assert_upload_mode("xet"):
            return_val = api.upload_folder(
                folder_path=self.folder_path,
                path_in_repo=folder_in_repo,
                repo_id=repo_id,
            )

        assert return_val.startswith(f"{api.endpoint}/{repo_id}/commit")
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

    def test_upload_folder_create_pr(self, api, repo_url) -> None:
        repo_id = repo_url.repo_id
        folder_in_repo = "temp_create_pr"
        with assert_upload_mode("xet"):
            return_val = api.upload_folder(
                folder_path=self.folder_path,
                path_in_repo=folder_in_repo,
                repo_id=repo_id,
                create_pr=True,
            )

        assert return_val.startswith(f"{api.endpoint}/{repo_id}/commit")

        for rpath in ["text_file.txt", "nested/nested_binary.safetensors"]:
            local_path = self.folder_path / rpath
            filepath = hf_hub_download(
                repo_id=repo_id, filename=f"{folder_in_repo}/{rpath}", revision=return_val.pr_revision
            )
            assert Path(local_path).read_bytes() == Path(filepath).read_bytes()


@requires("hf_xet")
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

        with assert_upload_mode("xet"):
            api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=folder, num_workers=4)

        # Check all files have been uploaded
        uploaded_files = api.list_repo_files(repo_id=repo_id)

        # Download and verify content
        local_dir = Path(tmp_path) / "snapshot"
        local_dir.mkdir()
        api.snapshot_download(repo_id=repo_id, local_dir=local_dir, cache_dir=None)

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
            xet_filedata = metadata.xet_file_data
            assert xet_filedata is not None

            # Verify xet files
            xet_file = local_dir / f"subfolder_{i}/file_xet_{i}_{j}.bin"
            assert xet_file.read_bytes() == f"content_lfs_{i}_{j}".encode()

            # Verify regular files
            regular_file = local_dir / f"subfolder_{i}/file_regular_{i}_{j}.txt"
            assert regular_file.read_bytes() == f"content_regular_{i}_{j}".encode()

    def test_upload_large_folder_batch_size_greater_than_one(self, api, tmp_path, repo_url: RepoUrl) -> None:
        from hf_xet import upload_files as real_upload_files

        N_FILES = 500
        repo_id = repo_url.repo_id

        folder = Path(tmp_path) / "large_folder"
        folder.mkdir()
        for i in range(N_FILES):
            (folder / f"file_xet_{i}.bin").write_bytes(f"content_lfs_{i}".encode())

        # capture the number of files passed in per call to hf_xet.upload_files
        # to ensure that the batch size is respected.
        num_files_per_call = []

        def spy_upload_files(*args, **kwargs):
            num_files = len(args[0])
            num_files_per_call.append(num_files)
            return real_upload_files(*args, **kwargs)

        with assert_upload_mode("xet"):
            with patch("hf_xet.upload_files", side_effect=spy_upload_files):
                api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=folder, num_workers=4)

        # the batch size is set to 256 however due to speed of hashing and get_upload_mode calls it's not always guaranteed
        # that the files will be uploaded in batches of 256. They may be uploaded in smaller batches if no other jobs
        # are available to run; even as small as 1 file per call.
        #
        # However, it would be unlikely that all files are uploaded in batches of 1 if batching was correctly implemented.
        # So we assert that not all files were uploaded in batches of 1, although it is possible even with batching.

        assert any(n > 1 for n in num_files_per_call)


@requires("hf_xet")
@pytest.mark.usefixtures("xet_setup")
class TestXetE2E:
    def test_hf_xet_with_token_refresher(self, api, tmp_path, repo_url):
        """
        Test the hf_xet.download_files function with a token refresher.

        This test manually calls the hf_xet.download_files function with a token refresher
        function to verify that the token refresh mechanism works as expected. It aims to
        identify regressions in the hf_xet.download_files function.

        * Define a token refresher function that issues a token refresh by returning a new
           access token and expiration time.
        * Mock the token refresher function.
        * Construct the necessary headers and metadata for the file to be downloaded.
        * Call the download_files function with the token refresher, forcing a token refresh.
        * Assert that the token refresher function was called as expected.

        This test ensures that the downloaded file is the same as the uploaded file.
        """
        from hf_xet import PyXetDownloadInfo, download_files

        filename_in_repo = "binary_file.bin"
        repo_id = repo_url.repo_id

        # Upload a file
        api.upload_file(
            path_or_fileobj=self.bin_file,
            path_in_repo=filename_in_repo,
            repo_id=repo_id,
        )

        # headers
        headers = build_hf_headers(token=TOKEN)

        # metadata for url
        (url_to_download, etag, commit_hash, expected_size, xet_filedata, head_call_error) = (
            _get_metadata_or_catch_error(
                repo_id=repo_id,
                filename=filename_in_repo,
                revision="main",
                repo_type="model",
                headers=headers,
                endpoint=api.endpoint,
                token=TOKEN,
                etag_timeout=None,
                local_files_only=False,
            )
        )

        xet_connection_info = refresh_xet_connection_info(file_data=xet_filedata, headers=headers)

        # manually construct parameters to hf_xet.download_files and use a locally defined token_refresher function
        # to verify that token refresh works as expected.
        def token_refresher() -> tuple[str, int]:
            # Issue a token refresh by returning a new access token and expiration time
            new_connection = refresh_xet_connection_info(file_data=xet_filedata, headers=headers)
            return new_connection.access_token, new_connection.expiration_unix_epoch

        mock_token_refresher = MagicMock(side_effect=token_refresher)

        incomplete_path = Path(tmp_path) / "file.bin.incomplete"
        file_info = [
            PyXetDownloadInfo(
                destination_path=str(incomplete_path.absolute()), hash=xet_filedata.file_hash, file_size=expected_size
            )
        ]

        # Call the download_files function with the token refresher, set expiration to 0 forcing a refresh
        download_files(
            file_info,
            endpoint=xet_connection_info.endpoint,
            token_info=(xet_connection_info.access_token, 0),
            token_refresher=mock_token_refresher,
            progress_updater=None,
        )

        # assert that our local token_refresher function was called by hfxet as expected.
        mock_token_refresher.assert_called_once()

        # Check that the downloaded file is the same as the uploaded file
        with open(incomplete_path, "rb") as f:
            downloaded_content = f.read()
        assert downloaded_content == self.bin_content
