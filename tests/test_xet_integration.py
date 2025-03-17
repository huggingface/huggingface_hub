import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest import TestCase, skip
from unittest.mock import MagicMock, patch

from huggingface_hub import HfApi
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.file_download import _get_metadata_or_catch_error
from huggingface_hub.file_download import xet_get as original_xet_get
from huggingface_hub.utils import SoftTemporaryDirectory, XetMetadata, build_hf_headers, tqdm
from huggingface_hub.utils._xet import refresh_xet_metadata

from .testing_constants import TOKEN
from .testing_utils import repo_name


WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR)


def is_hf_xet_available():
    try:
        from hf_xet import PyPointerFile

        _p = PyPointerFile("path", "hash", 100)
    except ImportError:
        return False
    return True


def require_hf_xet(test_case):
    """
    Decorator marking a test that requires hf_xet.
    These tests are skipped when hf_xet is not installed.
    """
    if not is_hf_xet_available():
        return skip("Test requires hf_xet")(test_case)
    else:
        return test_case


@require_hf_xet
class TestHfXet(TestCase):
    def setUp(self) -> None:
        self.content = b"RandOm Xet ConTEnT" * 1024
        self._token = TOKEN
        self._api = HfApi(endpoint=ENDPOINT, token=self._token)

        self._repo_id = self._api.create_repo(repo_name()).repo_id

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=self._repo_id)

    def test__xet_available(self):
        # renaming to this runs first
        self.assertTrue(is_hf_xet_available())

    def test_upload_and_download_with_hf_xet(self):
        # create a temporary file
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            file_path = os.path.join(tmpdir, "file.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)

            # Upload a file
            self._api.upload_file(repo_id=self._repo_id, path_or_fileobj=file_path, path_in_repo="file.bin")

            # Download a file
            downloaded_file = self._api.hf_hub_download(
                repo_id=self._repo_id, filename="file.bin", cache_dir=cache_dir
            )

            # Check that the downloaded file is the same as the uploaded file
            with open(downloaded_file, "rb") as file:
                downloaded_content = file.read()
            self.assertEqual(downloaded_content, self.content)

    def test_upload_folder_download_snapshot_with_hf_xet(self):
        # create a temporary directory
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            folder_path = os.path.join(tmpdir, "folder")
            os.makedirs(folder_path)
            file_path = os.path.join(folder_path, "file.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)
            file_path = os.path.join(folder_path, "file2.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)

            # Upload a folder
            self._api.upload_folder(repo_id=self._repo_id, folder_path=folder_path, path_in_repo="folder")

            # Download & verify files
            local_dir = os.path.join(tmpdir, "snapshot")
            os.makedirs(local_dir)
            snapshot_path = self._api.snapshot_download(
                repo_id=self._repo_id, local_dir=local_dir, cache_dir=cache_dir
            )

            for downloaded_file in ["folder/file.bin", "folder/file2.bin"]:
                # Check that the downloaded file is the same as the uploaded file
                print(downloaded_file)
                with open(os.path.join(snapshot_path, downloaded_file), "rb") as file:
                    downloaded_content = file.read()
                self.assertEqual(downloaded_content, self.content)

    def test_upload_large_folder_download_snapshot_with_hf_xet(self):
        # create a temporary directory
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            folder_path = os.path.join(tmpdir, "folder")
            os.makedirs(folder_path)
            for i in range(200):
                file_path = os.path.join(folder_path, f"file_{i}.bin")
                with open(file_path, "wb+") as file:
                    file.write(self.content)

            # Upload a large folder - should require two batches
            self._api.upload_large_folder(repo_id=self._repo_id, folder_path=folder_path, repo_type="model")

            # Download & verify files
            local_dir = os.path.join(tmpdir, "snapshot")
            os.makedirs(local_dir)
            snapshot_path = self._api.snapshot_download(
                repo_id=self._repo_id, local_dir=local_dir, cache_dir=cache_dir
            )

            for i in range(200):
                downloaded_file = f"file_{i}.bin"
                # Check that the downloaded file is the same as the uploaded file
                with open(os.path.join(snapshot_path, downloaded_file), "rb") as file:
                    downloaded_content = file.read()
                self.assertEqual(downloaded_content, self.content)

    def test_force_refresh_token(self):
        """
        Test that the token refresher is called when the expiration time is set to now.
        """
        # create a temporary file
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            file_path = os.path.join(tmpdir, "file.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)

            # Upload a file
            self._api.upload_file(repo_id=self._repo_id, path_or_fileobj=file_path, path_in_repo="file.bin")

            # shim xet_get so can muck with xet_metadata parameter,
            # setting the timeout to now()
            # and then calling through to real xet_get(),
            # forcing a refresh token to occur.

            def xet_get_shim(
                incomplete_path: Path,
                xet_metadata: XetMetadata,
                headers: Dict[str, str],
                expected_size: Optional[int] = None,
                displayed_filename: Optional[str] = None,
                _tqdm_bar: Optional[tqdm] = None,
            ):
                modified_xet_metadata = XetMetadata(
                    endpoint=xet_metadata.endpoint,
                    access_token=xet_metadata.access_token,
                    expiration_unix_epoch=int(datetime.timestamp(datetime.now())),
                    refresh_route=xet_metadata.refresh_route,
                    file_hash=xet_metadata.file_hash,
                )

                return original_xet_get(
                    incomplete_path=incomplete_path,
                    xet_metadata=modified_xet_metadata,
                    headers=headers,
                    expected_size=expected_size,
                    displayed_filename=displayed_filename,
                    _tqdm_bar=_tqdm_bar,
                )

            with patch("huggingface_hub.file_download.xet_get", side_effect=xet_get_shim) as mock_xet_get:
                # Download a file
                downloaded_file = self._api.hf_hub_download(
                    repo_id=self._repo_id, filename="file.bin", cache_dir=cache_dir
                )

            mock_xet_get.assert_called_once()

            # Check that the downloaded file is the same as the uploaded file
            with open(downloaded_file, "rb") as file:
                downloaded_content = file.read()
            self.assertEqual(downloaded_content, self.content)

    def test_hfxet_direct_download_files_token_refresher(self):
        """
        Manually call hf_xet.download_files with a token refresher function to verify that
        the token refresh works as expected. This is test to identify regressions in the
        hf_xet.download_files function.
        """
        from hf_xet import PyPointerFile, download_files

        # create a temporary file
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            file_path = os.path.join(tmpdir, "file.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)

            # Upload a file
            self._api.upload_file(repo_id=self._repo_id, path_or_fileobj=file_path, path_in_repo="file.bin")

            # manually construct parameters to hf_xet.download_files and use a locally defined token_refresher function
            # to verify that token refresh works as expected.
            def token_refresher() -> Tuple[str, int]:
                # Issue a token refresh by returning a new access token and expiration time
                new_xet_metadata = refresh_xet_metadata(xet_metadata=xet_metadata, headers=headers)
                return new_xet_metadata.access_token, new_xet_metadata.expiration_unix_epoch

            mock_token_refresher = MagicMock(side_effect=token_refresher)

            # headers
            headers = build_hf_headers(token=self._token)

            # metadata for url
            (url_to_download, etag, commit_hash, expected_size, xet_metadata, head_call_error) = (
                _get_metadata_or_catch_error(
                    repo_id=self._repo_id,
                    filename="file.bin",
                    revision="main",
                    repo_type="model",
                    headers=headers,
                    endpoint=self._api.endpoint,
                    token=self._token,
                    proxies=None,
                    etag_timeout=None,
                    local_files_only=False,
                )
            )

            incomplete_path = Path(cache_dir) / "file.bin.incomplete"
            py_file = [
                PyPointerFile(
                    path=str(incomplete_path.absolute()), hash=xet_metadata.file_hash, filesize=expected_size
                )
            ]

            # Call the download_files function with the token refresher, set expiration to 0 forcing a refresh
            download_files(
                py_file,
                endpoint=xet_metadata.endpoint,
                token_info=(xet_metadata.access_token, 0),
                token_refresher=mock_token_refresher,
                progress_updater=None,
            )

            # assert that our local token_refresher function was called by hfxet as expected.
            mock_token_refresher.assert_called_once()

            # Check that the downloaded file is the same as the uploaded file
            with open(incomplete_path, "rb") as file:
                downloaded_content = file.read()
            self.assertEqual(downloaded_content, self.content)
