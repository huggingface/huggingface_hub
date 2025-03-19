import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest import TestCase
from unittest.mock import DEFAULT, MagicMock, Mock, patch

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.file_download import (
    HfFileMetadata,
    _get_metadata_or_catch_error,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
    xet_get,
)
from huggingface_hub.utils import SoftTemporaryDirectory, XetMetadata, build_hf_headers, tqdm

from .testing_constants import TOKEN
from .testing_utils import (
    DUMMY_XET_FILE,
    DUMMY_XET_MODEL_ID,
    repo_name,
    requires,
    with_production_testing,
)


@requires("hf_xet")
@with_production_testing
class TestXetFileDownload:
    @contextmanager
    def _patch_xet_file_metadata(self, with_xet_metadata: bool):
        xet_metadata = (
            XetMetadata(
                endpoint="mock_endpoint",
                access_token="mock_token",
                expiration_unix_epoch=9999999999,
                file_hash="mock_hash",
            )
            if with_xet_metadata
            else None
        )
        patcher = patch("huggingface_hub.file_download.get_hf_file_metadata")
        mock_metadata = patcher.start()
        mock_metadata.return_value = HfFileMetadata(
            commit_hash="mock_commit",
            etag="mock_etag",
            location="mock_location",
            size=1024,
            xet_metadata=xet_metadata,
        )
        try:
            yield mock_metadata
        finally:
            patcher.stop()

    def test_xet_get_called_when_xet_metadata_present(self, tmp_path):
        """Test that xet_get is called when xet metadata is present."""
        with self._patch_xet_file_metadata(with_xet_metadata=True) as mock_metadata:
            with patch("huggingface_hub.file_download.xet_get") as mock_xet_get:
                with patch("huggingface_hub.file_download._create_symlink"):
                    hf_hub_download(
                        DUMMY_XET_MODEL_ID,
                        filename=DUMMY_XET_FILE,
                        cache_dir=tmp_path,
                        force_download=True,
                    )

                    # Verify xet_get was called with correct parameters
                    mock_xet_get.assert_called_once()
                    _, kwargs = mock_xet_get.call_args
                    assert "xet_metadata" in kwargs
                    assert kwargs["xet_metadata"] == mock_metadata.return_value.xet_metadata

    def test_backward_compatibility_no_xet_metadata(self, tmp_path):
        """Test backward compatibility when response has no xet metadata."""
        with self._patch_xet_file_metadata(with_xet_metadata=False):
            with patch("huggingface_hub.file_download.http_get") as mock_http_get:
                with patch("huggingface_hub.file_download._create_symlink"):
                    hf_hub_download(
                        DUMMY_XET_MODEL_ID,
                        filename=DUMMY_XET_FILE,
                        cache_dir=tmp_path,
                        force_download=True,
                    )

                    # Verify http_get was called
                    mock_http_get.assert_called_once()

    def test_get_xet_file_metadata_basic(self) -> None:
        """Test getting metadata from a file on the Hub."""
        url = hf_hub_url(
            repo_id=DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
        )
        metadata = get_hf_file_metadata(url)
        xet_metadata = metadata.xet_metadata
        assert xet_metadata is not None
        assert xet_metadata.endpoint is not None
        assert xet_metadata.access_token is not None
        assert isinstance(xet_metadata.expiration_unix_epoch, int)
        assert xet_metadata.file_hash is not None
        assert xet_metadata.refresh_route is not None

    def test_basic_download(self, tmp_path):
        # Make sure that xet_get is called
        with patch("huggingface_hub.file_download.xet_get", wraps=xet_get) as _xet_get:
            filepath = hf_hub_download(
                DUMMY_XET_MODEL_ID,
                filename=DUMMY_XET_FILE,
                cache_dir=tmp_path,
            )

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

            _xet_get.assert_called_once()

    def test_try_to_load_from_cache(self, tmp_path):
        cached_path = try_to_load_from_cache(
            DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
            cache_dir=tmp_path,
        )
        assert cached_path is None

        downloaded_path = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
            cache_dir=tmp_path,
        )

        # Now should find it in cache
        cached_path = try_to_load_from_cache(
            DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
            cache_dir=tmp_path,
        )
        assert cached_path == downloaded_path

    def test_cache_reuse(self, tmp_path):
        path1 = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
            cache_dir=tmp_path,
        )

        assert os.path.exists(path1)

        with patch("huggingface_hub.file_download._download_to_tmp_and_move") as mock:
            # Second download should use cache
            path2 = hf_hub_download(
                DUMMY_XET_MODEL_ID,
                filename=DUMMY_XET_FILE,
                cache_dir=tmp_path,
            )

            assert path1 == path2
            mock.assert_not_called()

    def test_download_to_local_dir(self, tmp_path):
        local_dir = tmp_path / "local_dir"
        local_dir.mkdir(exist_ok=True, parents=True)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        # Download to local dir
        returned_path = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
            local_dir=local_dir,
            cache_dir=cache_dir,
        )
        assert local_dir in Path(returned_path).parents

        for path in cache_dir.glob("**/blobs/**"):
            assert not path.is_file()
        for path in cache_dir.glob("**/snapshots/**"):
            assert not path.is_file()

    def test_force_download(self, tmp_path):
        # First download
        path1 = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=DUMMY_XET_FILE,
            cache_dir=tmp_path,
        )

        # Force download should re-download even if in cache
        with patch("huggingface_hub.file_download.xet_get") as mock_xet_get:
            path2 = hf_hub_download(
                DUMMY_XET_MODEL_ID,
                filename=DUMMY_XET_FILE,
                cache_dir=tmp_path,
                force_download=True,
            )

            assert path1 == path2
            mock_xet_get.assert_called_once()

    def test_fallback_to_http_when_xet_not_available(self, tmp_path):
        """Test that http_get is used when hf_xet is not available."""
        with self._patch_xet_file_metadata(with_xet_metadata=True):
            # Mock is_xet_available to return False
            with patch.multiple(
                "huggingface_hub.file_download",
                is_xet_available=Mock(return_value=False),
                http_get=DEFAULT,
                xet_get=DEFAULT,
                _create_symlink=DEFAULT,
            ) as mocks:
                hf_hub_download(
                    DUMMY_XET_MODEL_ID,
                    filename=DUMMY_XET_FILE,
                    cache_dir=tmp_path,
                    force_download=True,
                )

                # Verify http_get was called and xet_get was not
                mocks["http_get"].assert_called_once()
                mocks["xet_get"].assert_not_called()

    def test_use_xet_when_available(self, tmp_path):
        """Test that xet_get is used when hf_xet is available."""
        with self._patch_xet_file_metadata(with_xet_metadata=True):
            with patch.multiple(
                "huggingface_hub.file_download",
                is_xet_available=Mock(return_value=True),
                http_get=DEFAULT,
                xet_get=DEFAULT,
                _create_symlink=DEFAULT,
            ) as mocks:
                hf_hub_download(
                    DUMMY_XET_MODEL_ID,
                    filename=DUMMY_XET_FILE,
                    cache_dir=tmp_path,
                    force_download=True,
                )

                # Verify xet_get was called and http_get was not
                mocks["xet_get"].assert_called_once()
                mocks["http_get"].assert_not_called()


@requires("hf_xet")
@with_production_testing
class TestXetSnapshotDownload:
    def test_download_model(self, tmp_path):
        """Test that snapshot_download works with Xet storage."""
        storage_folder = snapshot_download(
            DUMMY_XET_MODEL_ID,
            cache_dir=tmp_path,
        )

        assert os.path.exists(storage_folder)
        assert os.path.isdir(storage_folder)
        assert os.path.exists(os.path.join(storage_folder, DUMMY_XET_FILE))

        with open(os.path.join(storage_folder, DUMMY_XET_FILE), "rb") as f:
            content = f.read()
            assert len(content) > 0

    def test_snapshot_download_cache_reuse(self, tmp_path):
        """Test that snapshot_download reuses cached files."""
        # First download
        storage_folder1 = snapshot_download(
            DUMMY_XET_MODEL_ID,
            cache_dir=tmp_path,
        )

        with patch("huggingface_hub.file_download.xet_get") as mock_xet_get:
            # Second download should use cache
            storage_folder2 = snapshot_download(
                DUMMY_XET_MODEL_ID,
                cache_dir=tmp_path,
            )

            # Verify same folder is returned
            assert storage_folder1 == storage_folder2

            # Verify xet_get was not called (files were cached)
            mock_xet_get.assert_not_called()


@requires("hf_xet")
class TestXetTokenRefresh(TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment.
        This method initializes the following attributes:
        - `self.content`: A byte string containing random content for testing.
        - `self._token`: The authentication token for accessing the Hugging Face API.
        - `self._api`: An instance of `HfApi` initialized with the endpoint and token.
        - `self._repo_id`: The repository ID created by the Hugging Face API.
        """
        self.content = b"RandOm Xet ConTEnT" * 1024
        self._token = TOKEN
        self._api = HfApi(endpoint=ENDPOINT, token=self._token)

        self._repo_id = self._api.create_repo(repo_name()).repo_id

    def tearDown(self) -> None:
        """
        Tear down the test environment.
        This method deletes the repository created for testing.
        """
        self._api.delete_repo(repo_id=self._repo_id)

    def test_force_refresh_token(self):
        """
        Test the token refresh process when the expiration time is set to now.

        * Patches the `xet_get` function to simulate an expired token by setting the expiration time to the current time.
        * Downloads the file from the repository, triggering the token refresh process.
        * Asserts that the `xet_get` function was called once.

        This test ensures that the downloaded file is the same as the uploaded file.
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

                return xet_get(
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
        from hf_xet import PyPointerFile, download_files

        from huggingface_hub.utils._xet import refresh_xet_metadata

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
