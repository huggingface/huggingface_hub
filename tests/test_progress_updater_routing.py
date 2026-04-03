"""Integration tests for progress_updater routing in xet downloads.

These tests verify that the progress_updater parameter is correctly passed
through from hf_hub_download to the internal xet_get function.

Run with:
    cd /Users/tobias/projects/huggingface_hub
    uv run pytest tests/test_progress_updater_routing.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import DEFAULT, Mock, patch


class TestProgressUpdaterRouting:
    """Tests for progress_updater parameter routing in hf_hub_download."""

    def test_progress_updater_passed_to_xet_get_via_download_to_tmp(self, tmp_path):
        """Test that progress_updater is passed from hf_hub_download to xet_get.

        This is the core bug: progress_updater exists in hf_hub_download API
        but is never passed to _download_to_tmp_and_move, which then doesn't
        pass it to xet_get.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.file_download import HfFileMetadata, XetFileData
        from huggingface_hub.utils import XetConnectionInfo

        progress_callback = Mock()

        # Use the dummy xet testing repo
        model_id = "celinah/dummy-xet-testing"
        filename = "tiny.safetensors"  # 45 bytes

        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=45,
                xet_file_data=XetFileData(file_hash="mock_hash", refresh_route="mock/route"),
            )

            with patch("huggingface_hub.file_download.refresh_xet_connection_info") as mock_xet_conn:
                mock_xet_conn.return_value = XetConnectionInfo(
                    endpoint="mock_endpoint",
                    access_token="mock_token",
                    expiration_unix_epoch=9999999999,
                )

                with patch("huggingface_hub.file_download.xet_get") as mock_xet_get:
                    with patch("huggingface_hub.file_download._create_symlink"):
                        hf_hub_download(
                            model_id,
                            filename=filename,
                            cache_dir=tmp_path,
                            force_download=True,
                            progress_updater=progress_callback,
                        )

                        # Verify xet_get was called
                        mock_xet_get.assert_called_once()
                        _, kwargs = mock_xet_get.call_args

                        # THIS IS THE KEY ASSERTION - will fail with the bug
                        # Because progress_updater is never passed through to xet_get
                        assert "progress_updater" in kwargs, (
                            "progress_updater was not passed to xet_get! "
                            "This is the bug - _download_to_tmp_and_move doesn't "
                            "propagate the progress_updater parameter."
                        )
                        assert kwargs["progress_updater"] == progress_callback

    def test_progress_updater_passed_to_xet_get_with_detailed_callback(self, tmp_path):
        """Test that detailed (2-arg) callback works correctly."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.file_download import HfFileMetadata, XetFileData
        from huggingface_hub.utils import XetConnectionInfo

        def detailed_callback(total_update, item_updates):
            """xet-core detailed mode callback with 2 args."""
            pass

        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=45,
                xet_file_data=XetFileData(file_hash="mock_hash", refresh_route="mock/route"),
            )

            with patch("huggingface_hub.file_download.refresh_xet_connection_info") as mock_xet_conn:
                mock_xet_conn.return_value = XetConnectionInfo(
                    endpoint="mock_endpoint",
                    access_token="mock_token",
                    expiration_unix_epoch=9999999999,
                )

                with patch("huggingface_hub.file_download.xet_get") as mock_xet_get:
                    with patch("huggingface_hub.file_download._create_symlink"):
                        hf_hub_download(
                            "celinah/dummy-xet-testing",
                            filename="tiny.safetensors",
                            cache_dir=tmp_path,
                            force_download=True,
                            progress_updater=detailed_callback,
                        )

                        mock_xet_get.assert_called_once()
                        _, kwargs = mock_xet_get.call_args

                        assert "progress_updater" in kwargs
                        assert kwargs["progress_updater"] == detailed_callback


class TestProgressUpdaterWithTqdm:
    """Tests verifying tqdm is disabled when progress_updater is provided."""

    def test_tqdm_class_is_none_when_progress_updater_provided(self, tmp_path):
        """Verify tqdm_class is set to None when progress_updater is provided.

        This prevents the "bad value(s) in fds_to_keep" error that occurs
        when tqdm is used in Textual worker threads.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.file_download import HfFileMetadata, XetFileData
        from huggingface_hub.utils import XetConnectionInfo

        with patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_metadata:
            mock_metadata.return_value = HfFileMetadata(
                commit_hash="mock_commit",
                etag="mock_etag",
                location="mock_location",
                size=45,
                xet_file_data=XetFileData(file_hash="mock_hash", refresh_route="mock/route"),
            )

            with patch("huggingface_hub.file_download.refresh_xet_connection_info") as mock_xet_conn:
                mock_xet_conn.return_value = XetConnectionInfo(
                    endpoint="mock_endpoint",
                    access_token="mock_token",
                    expiration_unix_epoch=9999999999,
                )

                with patch("huggingface_hub.file_download.xet_get") as mock_xet_get:
                    with patch("huggingface_hub.file_download._create_symlink"):
                        hf_hub_download(
                            "celinah/dummy-xet-testing",
                            filename="tiny.safetensors",
                            cache_dir=tmp_path,
                            force_download=True,
                            progress_updater=lambda x: None,
                            tqdm_class=None,  # User explicitly passing None
                        )

                        mock_xet_get.assert_called_once()
                        _, kwargs = mock_xet_get.call_args

                        # tqdm_class should be None when progress_updater is provided
                        assert kwargs.get("tqdm_class") is None, (
                            "tqdm_class should be None when progress_updater is provided "
                            "to prevent fd errors in Textual worker threads"
                        )
