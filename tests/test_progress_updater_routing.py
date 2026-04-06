"""Integration tests for progress_updater routing in xet downloads.

These tests verify that the progress_updater parameter is correctly passed
through from hf_hub_download to the internal xet_get function.
"""

from unittest.mock import Mock, patch

from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import XetFileData


class TestProgressUpdaterRouting:
    """Tests for progress_updater parameter routing in hf_hub_download."""

    @patch("huggingface_hub.file_download._download_to_tmp_and_move")
    @patch("huggingface_hub.file_download._create_symlink")
    @patch("huggingface_hub.file_download._get_metadata_or_catch_error")
    def test_progress_updater_passed_to_xet_get(self, mock_metadata, mock_symlink, mock_download, tmp_path):
        """Test that progress_updater is passed from hf_hub_download to _download_to_tmp_and_move."""
        progress_callback = Mock()

        mock_metadata.return_value = (
            "https://example.com/xet-file",
            "mock_etag",
            "mock_commit",
            45,
            XetFileData(file_hash="mock_hash", refresh_route="mock/route"),
            None,
        )

        hf_hub_download(
            "celinah/dummy-xet-testing",
            filename="tiny.safetensors",
            cache_dir=tmp_path,
            force_download=True,
            progress_updater=progress_callback,
        )

        mock_download.assert_called_once()
        _, kwargs = mock_download.call_args
        assert kwargs["progress_updater"] == progress_callback

    @patch("huggingface_hub.file_download._download_to_tmp_and_move")
    @patch("huggingface_hub.file_download._create_symlink")
    @patch("huggingface_hub.file_download._get_metadata_or_catch_error")
    def test_progress_updater_passed_to_http_get(self, mock_metadata, mock_symlink, mock_download, tmp_path):
        """Test that progress_updater is passed from hf_hub_download to _download_to_tmp_and_move."""
        progress_callback = Mock()

        mock_metadata.return_value = (
            "https://huggingface.co/model/resolve/main/file.bin",
            "mock_etag",
            "mock_commit",
            100,
            None,
            None,
        )

        hf_hub_download(
            "org/repo",
            filename="file.bin",
            cache_dir=tmp_path,
            force_download=True,
            progress_updater=progress_callback,
        )

        mock_download.assert_called_once()
        _, kwargs = mock_download.call_args
        assert kwargs.get("progress_updater") == progress_callback


class TestMakeXetProgressAdapter:
    """Tests for _make_xet_progress_adapter helper."""

    def test_accumulates_transfer_bytes(self):
        """Adapter converts xet-core detailed updates to cumulative ProgressCallback calls."""
        from huggingface_hub.file_download import _make_xet_progress_adapter

        results: list[tuple[int, int | None]] = []
        adapter = _make_xet_progress_adapter(lambda d, t=None: results.append((d, t)), total=1000)

        update1 = Mock(total_transfer_bytes_completion_increment=200, total_transfer_bytes=1000)
        update2 = Mock(total_transfer_bytes_completion_increment=300, total_transfer_bytes=1000)

        adapter(update1, [])
        adapter(update2, [])

        assert results == [(200, 1000), (500, 1000)]

    def test_skips_zero_increment(self):
        """Adapter ignores updates with zero transfer bytes."""
        from huggingface_hub.file_download import _make_xet_progress_adapter

        results: list[tuple[int, int | None]] = []
        adapter = _make_xet_progress_adapter(lambda d, t=None: results.append((d, t)), total=1000)

        adapter(Mock(total_transfer_bytes_completion_increment=100, total_transfer_bytes=1000), [])
        adapter(Mock(total_transfer_bytes_completion_increment=0, total_transfer_bytes=1000), [])

        assert results == [(100, 1000)]

    def test_uses_fallback_total(self):
        """Adapter uses provided total when xet-core reports 0 total_transfer_bytes."""
        from huggingface_hub.file_download import _make_xet_progress_adapter

        results: list[tuple[int, int | None]] = []
        adapter = _make_xet_progress_adapter(lambda d, t=None: results.append((d, t)), total=5000)

        adapter(Mock(total_transfer_bytes_completion_increment=100, total_transfer_bytes=0), [])

        assert results == [(100, 5000)]

    def test_uses_fallback_total_when_none(self):
        """Adapter uses provided total when xet-core reports None total_transfer_bytes."""
        from huggingface_hub.file_download import _make_xet_progress_adapter

        results: list[tuple[int, int | None]] = []
        adapter = _make_xet_progress_adapter(lambda d, t=None: results.append((d, t)), total=3000)

        adapter(Mock(total_transfer_bytes_completion_increment=50, total_transfer_bytes=None), [])

        assert results == [(50, 3000)]
