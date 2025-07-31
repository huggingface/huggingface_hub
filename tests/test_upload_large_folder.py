# tests/test_upload_large_folder.py
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub._upload_large_folder import (
    COMMIT_SIZE_SCALE,
    MAX_FILES_PER_FOLDER,
    MAX_FILES_PER_REPO,
    LargeUploadStatus,
    upload_large_folder_internal,
)
from huggingface_hub.utils import SoftTemporaryDirectory


@pytest.fixture
def status():
    return LargeUploadStatus(items=[])


def test_target_chunk_default(status):
    assert status.target_chunk() == COMMIT_SIZE_SCALE[1]


@pytest.mark.parametrize(
    "start_idx, success, delta_items, duration, expected_idx",
    [
        (2, False, 0, 10, 1),  # drop by one on failure
        (0, False, 0, 10, 0),  # never go below zero
        (1, True, 0, 50, 1),  # duration >= 40 --> no bump
        (1, True, -1, 30, 1),  # nb_items < threshold --> no bump
        (1, True, 0, 30, 2),  # fast enough and enough items
        (len(COMMIT_SIZE_SCALE) - 1, True, 0, 10, len(COMMIT_SIZE_SCALE) - 1),  # never exceed last index
    ],
)
def test_update_chunk_transitions(status, start_idx, success, delta_items, duration, expected_idx):
    status._chunk_idx = start_idx
    threshold = COMMIT_SIZE_SCALE[start_idx]
    nb_items = threshold + delta_items
    status.update_chunk(success=success, nb_items=nb_items, duration=duration)

    assert status._chunk_idx == expected_idx
    assert status.target_chunk() == COMMIT_SIZE_SCALE[expected_idx]


class UploadLargeFolderValidationTest(unittest.TestCase):
    """Test validation warnings for upload_large_folder - focusing on file count checks."""

    def setUp(self):
        self.api = MagicMock()
        self.api.create_repo.return_value.repo_id = "test-user/test-repo"
        self.api.repo_info.return_value.xet_enabled = False
        self.api._build_hf_headers.return_value = {}
        self.api.endpoint = "https://huggingface.co"

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_validation_warns_too_many_files(self, mock_logger):
        """Test warning when total files exceed MAX_FILES_PER_REPO."""
        with SoftTemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            # Create actual test files that will be found
            num_files = 5
            for i in range(num_files):
                (folder / f"file_{i}.txt").write_text("content")

            # Mock the validation check by directly calling logger.warning
            # since we're focusing on file count validation only
            expected_count = MAX_FILES_PER_REPO + 100
            with patch("huggingface_hub._upload_large_folder.len") as mock_len:
                # First call returns actual file count for initial check
                # Second call returns our mocked large number for validation
                mock_len.side_effect = [num_files, expected_count, expected_count]

                with patch("huggingface_hub._upload_large_folder.threading.Thread"):
                    with patch("huggingface_hub._upload_large_folder.time.sleep"):
                        with patch("huggingface_hub._upload_large_folder.LargeUploadStatus.is_done") as mock_done:
                            mock_done.return_value = True

                            upload_large_folder_internal(
                                api=self.api,
                                repo_id="test-repo",
                                folder_path=folder,
                                repo_type="dataset",
                            )

                # Check that warning was logged
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any(
                    f"You are about to upload {expected_count:,} files" in str(call) for call in warning_calls
                ), "Expected warning about too many files not found"

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_validation_warns_too_many_files_per_folder(self, mock_logger):
        """Test warning when a folder has too many files."""
        with SoftTemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            subfolder = folder / "data"
            subfolder.mkdir()
            # Create a couple actual files
            (subfolder / "file1.txt").write_text("content")
            (subfolder / "file2.txt").write_text("content")

            # Use Counter directly to simulate the file count per folder
            with patch("huggingface_hub._upload_large_folder.Counter") as mock_counter:
                # Mock Counter to return high file count for 'data' folder
                mock_counter.return_value = {"data": MAX_FILES_PER_FOLDER + 100}

                with patch("huggingface_hub._upload_large_folder.threading.Thread"):
                    with patch("huggingface_hub._upload_large_folder.time.sleep"):
                        with patch("huggingface_hub._upload_large_folder.LargeUploadStatus.is_done") as mock_done:
                            mock_done.return_value = True

                            upload_large_folder_internal(
                                api=self.api,
                                repo_id="test-repo",
                                folder_path=folder,
                                repo_type="dataset",
                            )

                # Check warning
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any(
                    f"Folder 'data' contains {MAX_FILES_PER_FOLDER + 100:,} files" in str(call)
                    for call in warning_calls
                ), "Expected warning about too many files per folder"
