import os
from pathlib import Path
from unittest.mock import DEFAULT, Mock, patch

from huggingface_hub import snapshot_download
from huggingface_hub.file_download import (
    HfFileMetadata,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
    xet_get,
)
from huggingface_hub.utils import XetMetadata

from .testing_utils import (
    DUMMY_XET_FILE,
    DUMMY_XET_MODEL_ID,
    requires,
    with_production_testing,
)


@requires("hf_xet")
@with_production_testing
class TestXetFileDownload:
    def test_xet_get_called_when_xet_metadata_present(self, tmp_path):
        """Test that xet_get is called when xet metadata is present."""
        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            xet_metadata = XetMetadata(
                endpoint="mock_endpoint",
                access_token="mock_token",
                expiration_unix_epoch=9999999999,
                file_hash="mock_hash",
            )
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=1024,
                xet_metadata=xet_metadata,
            )

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
                    assert kwargs["xet_metadata"] == xet_metadata

    def test_backward_compatibility_no_xet_metadata(self, tmp_path):
        """Test backward compatibility when response has no xet metadata."""
        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=1024,
                xet_metadata=None,  # No xet_metadata
            )

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
        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=1024,
                xet_metadata=XetMetadata(
                    endpoint="mock_endpoint",
                    access_token="mock_token",
                    expiration_unix_epoch=9999999999,
                    file_hash="mock_hash",
                ),
            )

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
        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=1024,
                xet_metadata=XetMetadata(
                    endpoint="mock_endpoint",
                    access_token="mock_token",
                    expiration_unix_epoch=9999999999,
                    file_hash="mock_hash",
                ),
            )

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
