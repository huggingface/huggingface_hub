import os
from contextlib import contextmanager
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
from huggingface_hub.utils import (
    XetConnectionInfo,
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

    @contextmanager
    def _patch_get_refresh_xet_connection_info(self):
        patcher = patch("huggingface_hub.file_download.refresh_xet_connection_info")
        connection_info = XetConnectionInfo(
            endpoint="mock_endpoint",
            access_token="mock_token",
            expiration_unix_epoch=9999999999,
        )

        mock_xet_connection = patcher.start()
        mock_xet_connection.return_value = connection_info
        try:
            yield mock_xet_connection
        finally:
            patcher.stop()

    def test_xet_get_called_when_xet_metadata_present(self, tmp_path):
        """Test that xet_get is called when xet metadata is present."""
        with self._patch_xet_file_metadata(with_xet_data=True) as mock_file_metadata:
            with self._patch_get_refresh_xet_connection_info():
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
            with self._patch_get_refresh_xet_connection_info():
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
            with self._patch_get_refresh_xet_connection_info():
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
        """Test that headers (minus authorization) are passed as request_headers to hf_xet.download_files."""
        headers = {
            "authorization": "Bearer my_token",
            "x-custom-header": "custom_value",
            "user-agent": "test-agent",
        }

        with self._patch_xet_file_metadata(with_xet_data=True):
            with self._patch_get_refresh_xet_connection_info():
                with patch("hf_xet.download_files") as mock:
                    hf_hub_download(
                        DUMMY_XET_MODEL_ID,
                        filename=DUMMY_XET_FILE,
                        cache_dir=tmp_path,
                        force_download=True,
                        headers=headers,
                    )
                    mock.assert_called_once()
                    request_headers = mock.call_args.kwargs["request_headers"]
                    assert request_headers.get("x-custom-header") == "custom_value"
                    assert request_headers.get("user-agent") == "test-agent"
                    assert "authorization" not in request_headers


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


def _make_mock_total_update(transfer_increment: float, transfer_total: float):
    """Create a mock PyTotalProgressUpdate with the given transfer fields."""
    update = Mock()
    update.total_transfer_bytes_completion_increment = transfer_increment
    update.total_transfer_bytes = transfer_total
    return update


@requires("hf_xet")
class TestXetProgressGranularity:
    """Test that xet_get uses the fine-grained 2-arg callback for tqdm progress."""

    _XET_FILE_DATA = XetFileData(file_hash="mock_hash", refresh_route="mock/route")

    def _call_xet_get_and_capture(self, tmp_path, mock_download, expected_size=1000, mock_progress_cm=None):
        """Call xet_get and return the captured progress callback."""
        incomplete_path = tmp_path / "test_file.bin"
        incomplete_path.touch()

        captured = {}

        def capture(*args, **kwargs):
            captured["callback"] = kwargs["progress_updater"][0]

        mock_download.side_effect = capture

        if mock_progress_cm is not None:
            mock_bar = Mock()
            mock_bar.n = 0
            mock_progress_cm.return_value.__enter__ = Mock(return_value=mock_bar)
            mock_progress_cm.return_value.__exit__ = Mock(return_value=False)
        else:
            mock_bar = None

        xet_get(
            incomplete_path=incomplete_path,
            xet_file_data=self._XET_FILE_DATA,
            headers={"authorization": "Bearer token"},
            expected_size=expected_size,
        )

        return captured["callback"], mock_bar

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_callback_uses_two_arg_signature(self, _mock_conn, mock_download, mock_progress_cm, tmp_path):
        """Verify xet_get passes a 2-arg callback to download_files, triggering
        xet-core's fine-grained network-level progress dispatch."""
        callback, _ = self._call_xet_get_and_capture(tmp_path, mock_download, mock_progress_cm=mock_progress_cm)

        # Call with 2 args (total_update, item_updates) to confirm it accepts them.
        # A 1-arg callback would raise TypeError here.
        total_update = _make_mock_total_update(transfer_increment=200, transfer_total=1000)
        callback(total_update, [])  # should not raise

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_progress_bar_scales_network_to_file_size(self, _mock_conn, mock_download, mock_progress_cm, tmp_path):
        """When transfer bytes differ from file size, the progress bar should
        scale to expected_size so it always reaches 100%."""
        expected_size = 10_000
        transfer_total = 5_000  # fewer bytes due to deduplication

        callback, mock_bar = self._call_xet_get_and_capture(
            tmp_path, mock_download, expected_size=expected_size, mock_progress_cm=mock_progress_cm
        )

        def update_side_effect(n):
            mock_bar.n += n

        mock_bar.update = Mock(side_effect=update_side_effect)

        # Simulate 5 updates of 1000 transfer bytes each (total: 5000 transfer bytes)
        for _ in range(5):
            total_update = _make_mock_total_update(transfer_increment=1000, transfer_total=transfer_total)
            callback(total_update, [])

        # After transferring 5000/5000 bytes, bar should be at expected_size (10000)
        assert mock_bar.n == expected_size

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_progress_bar_capped_at_expected_size(self, _mock_conn, mock_download, mock_progress_cm, tmp_path):
        """Progress bar should never exceed expected_size."""
        expected_size = 1000

        callback, mock_bar = self._call_xet_get_and_capture(
            tmp_path, mock_download, expected_size=expected_size, mock_progress_cm=mock_progress_cm
        )

        def update_side_effect(n):
            mock_bar.n += n

        mock_bar.update = Mock(side_effect=update_side_effect)

        # Send more transfer bytes than total (edge case)
        total_update = _make_mock_total_update(transfer_increment=1200, transfer_total=1000)
        callback(total_update, [])

        assert mock_bar.n <= expected_size

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_zero_increment_skipped(self, _mock_conn, mock_download, mock_progress_cm, tmp_path):
        """Zero-increment updates should not call progress.update."""
        callback, mock_bar = self._call_xet_get_and_capture(tmp_path, mock_download, mock_progress_cm=mock_progress_cm)

        total_update = _make_mock_total_update(transfer_increment=0, transfer_total=1000)
        callback(total_update, [])

        mock_bar.update.assert_not_called()

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_expected_size_none_passes_raw_bytes(self, _mock_conn, mock_download, mock_progress_cm, tmp_path):
        """When expected_size is None, raw transfer bytes are passed through."""
        callback, mock_bar = self._call_xet_get_and_capture(
            tmp_path, mock_download, expected_size=None, mock_progress_cm=mock_progress_cm
        )

        def update_side_effect(n):
            mock_bar.n += n

        mock_bar.update = Mock(side_effect=update_side_effect)

        total_update = _make_mock_total_update(transfer_increment=500, transfer_total=0)
        callback(total_update, [])

        assert mock_bar.n == 500

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_expected_size_none_with_known_transfer_total(self, _mock_conn, mock_download, mock_progress_cm, tmp_path):
        """When expected_size is None, raw bytes pass through even if transfer_total is known."""
        callback, mock_bar = self._call_xet_get_and_capture(
            tmp_path, mock_download, expected_size=None, mock_progress_cm=mock_progress_cm
        )

        def update_side_effect(n):
            mock_bar.n += n

        mock_bar.update = Mock(side_effect=update_side_effect)

        total_update = _make_mock_total_update(transfer_increment=500, transfer_total=2000)
        callback(total_update, [])

        assert mock_bar.n == 500

    @patch("huggingface_hub.file_download._get_progress_bar_context")
    @patch("hf_xet.download_files")
    @patch(
        "huggingface_hub.file_download.refresh_xet_connection_info",
        return_value=XetConnectionInfo(
            endpoint="mock_endpoint", access_token="mock_token", expiration_unix_epoch=9999999999
        ),
    )
    def test_transfer_total_zero_skips_when_expected_size_set(
        self, _mock_conn, mock_download, mock_progress_cm, tmp_path
    ):
        """When expected_size is set but transfer_total is 0 (not yet known),
        updates are skipped to avoid injecting unscaled bytes."""
        callback, mock_bar = self._call_xet_get_and_capture(
            tmp_path, mock_download, expected_size=1000, mock_progress_cm=mock_progress_cm
        )

        total_update = _make_mock_total_update(transfer_increment=500, transfer_total=0)
        callback(total_update, [])

        mock_bar.update.assert_not_called()


@requires("hf_xet")
class TestMakeXetProgressCallback:
    """Direct tests for make_xet_progress_callback shared helper."""

    def test_multi_file_shared_bar(self):
        """Multiple callbacks sharing one bar should each contribute independently."""
        from huggingface_hub.file_download import make_xet_progress_callback

        mock_bar = Mock()
        mock_bar.n = 0

        def update_side_effect(n):
            mock_bar.n += n

        mock_bar.update = Mock(side_effect=update_side_effect)

        # Two files: 600 bytes and 400 bytes, sharing a bar with total=1000
        cb_a = make_xet_progress_callback(mock_bar, file_size=600)
        cb_b = make_xet_progress_callback(mock_bar, file_size=400)

        # File A: 50% done (transfers 500/1000 network bytes -> contributes 300 of 600 file bytes)
        cb_a(_make_mock_total_update(transfer_increment=500, transfer_total=1000), [])
        assert mock_bar.n == 300

        # File B: 100% done (transfers 800/800 -> contributes 400 of 400 file bytes)
        cb_b(_make_mock_total_update(transfer_increment=800, transfer_total=800), [])
        assert mock_bar.n == 700  # 300 + 400

        # File A: 100% done (transfers remaining 500/1000 -> contributes remaining 300)
        cb_a(_make_mock_total_update(transfer_increment=500, transfer_total=1000), [])
        assert mock_bar.n == 1000  # 600 + 400

    def test_no_regression_on_duplicate_progress(self):
        """When cumulative doesn't advance (e.g. duplicate update), bar should not update."""
        from huggingface_hub.file_download import make_xet_progress_callback

        mock_bar = Mock()
        mock_bar.n = 0

        def update_side_effect(n):
            mock_bar.n += n

        mock_bar.update = Mock(side_effect=update_side_effect)

        cb = make_xet_progress_callback(mock_bar, file_size=1000)

        # First update: 500/1000 transfer -> 500 file bytes
        cb(_make_mock_total_update(transfer_increment=500, transfer_total=1000), [])
        assert mock_bar.n == 500
        assert mock_bar.update.call_count == 1

        # Tiny increment that doesn't move int() forward (1 byte of 1000 transfer = 0.001 * 1000 = 1)
        # contributed = int(501/1000 * 1000) = 501, advance = 501 - 500 = 1
        cb(_make_mock_total_update(transfer_increment=1, transfer_total=1000), [])
        assert mock_bar.n == 501
        assert mock_bar.update.call_count == 2
