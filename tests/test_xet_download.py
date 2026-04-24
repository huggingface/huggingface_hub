import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, Mock, patch

from huggingface_hub import snapshot_download
from huggingface_hub.file_download import (
    HfFileMetadata,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
    xet_get,
)
from huggingface_hub.utils import (
    XetFileData,
    refresh_xet_connection_info,
)

from .testing_utils import (
    DUMMY_XET_FILE,
    DUMMY_XET_MODEL_ID,
    requires,
    with_production_testing,
)


@requires("hf_xet")
@with_production_testing
class TestXetFileDownload:
    @contextmanager
    def _patch_xet_file_metadata(self, with_xet_data: bool):
        patcher = patch("huggingface_hub.file_download.get_hf_file_metadata")
        mock_metadata = patcher.start()
        mock_metadata.return_value = HfFileMetadata(
            commit_hash="mock_commit",
            etag="mock_etag",
            location="mock_location",
            size=1024,
            xet_file_data=XetFileData(file_hash="mock_hash", refresh_route="mock/route") if with_xet_data else None,
        )
        try:
            yield mock_metadata
        finally:
            patcher.stop()

    def test_xet_get_called_when_xet_metadata_present(self, tmp_path):
        """Test that xet_get is called when xet metadata is present."""
        with self._patch_xet_file_metadata(with_xet_data=True) as mock_file_metadata:
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
                    assert "xet_file_data" in kwargs
                    assert kwargs["xet_file_data"] == mock_file_metadata.return_value.xet_file_data

    def test_backward_compatibility_no_xet_metadata(self, tmp_path):
        """Test backward compatibility when response has no xet metadata."""
        with self._patch_xet_file_metadata(with_xet_data=False):
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
        assert metadata.xet_file_data is not None
        assert metadata.xet_file_data.file_hash is not None

        connection_info = refresh_xet_connection_info(file_data=metadata.xet_file_data, headers={})
        assert connection_info is not None
        assert connection_info.endpoint is not None
        assert connection_info.access_token is not None
        assert isinstance(connection_info.expiration_unix_epoch, int)

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
        with self._patch_xet_file_metadata(with_xet_data=True):
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
        with self._patch_xet_file_metadata(with_xet_data=True):
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

    def test_request_headers_passed_to_download_files(self, tmp_path):
        """Test that headers (minus authorization) are passed as custom_headers to the download group builder."""
        headers = {
            "authorization": "Bearer my_token",
            "x-custom-header": "custom_value",
            "user-agent": "test-agent",
        }

        mock_group = MagicMock()
        mock_group.download_file.return_value = MagicMock()
        mock_group.finish.return_value = MagicMock()

        mock_builder = MagicMock()
        for method in ("with_token_refresh_url", "with_custom_headers", "with_progress_callback"):
            getattr(mock_builder, method).return_value = mock_builder
        mock_builder.build.return_value = mock_group

        mock_session = MagicMock()
        mock_session.new_file_download_group.return_value = mock_builder

        with self._patch_xet_file_metadata(with_xet_data=True):
            with patch("hf_xet.XetSession", return_value=mock_session):
                with patch("huggingface_hub.file_download._create_symlink"):
                    hf_hub_download(
                        DUMMY_XET_MODEL_ID,
                        filename=DUMMY_XET_FILE,
                        cache_dir=tmp_path,
                        force_download=True,
                        headers=headers,
                    )

        mock_group.download_file.assert_called_once()
        # Verify custom_headers excludes authorization
        custom_headers_arg = mock_builder.with_custom_headers.call_args[0][0]
        assert custom_headers_arg.get("x-custom-header") == "custom_value"
        assert custom_headers_arg.get("user-agent") == "test-agent"
        assert "authorization" not in custom_headers_arg
        # Verify full headers (incl. auth) passed to with_token_refresh_url
        refresh_headers_arg = mock_builder.with_token_refresh_url.call_args[0][1]
        assert "authorization" in refresh_headers_arg


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

    def test_download_backward_compatibility(self, tmp_path):
        """Test that xet download works with the old pointer file protocol.

        Until the next major version of hf-xet is released, we need to support the old
        pointer file based download to support old huggingface_hub versions.
        """

        file_path = os.path.join(tmp_path, DUMMY_XET_FILE)

        file_metadata = get_hf_file_metadata(
            hf_hub_url(
                repo_id=DUMMY_XET_MODEL_ID,
                filename=DUMMY_XET_FILE,
            )
        )

        xet_file_data = file_metadata.xet_file_data

        # Mock the response to not include xet metadata
        from hf_xet import PyPointerFile, download_files

        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers={})

        def token_refresher() -> tuple[str, int]:
            connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers={})
            return connection_info.access_token, connection_info.expiration_unix_epoch

        pointer_files = [PyPointerFile(path=file_path, hash=xet_file_data.file_hash, filesize=file_metadata.size)]

        download_files(
            pointer_files,
            endpoint=connection_info.endpoint,
            token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher=token_refresher,
            progress_updater=None,
        )

        assert os.path.exists(file_path)
