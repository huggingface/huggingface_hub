# tests/test_upload_large_folder.py
import unittest
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub._upload_large_folder import (
    COMMIT_SIZE_SCALE,
    MAX_FILES_PER_FOLDER,
    MAX_FILES_PER_REPO,
    LargeUploadStatus,
    _validate_upload_limits,
)


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


class TestValidateUploadLimits(unittest.TestCase):
    """Test the _validate_upload_limits function directly."""

    class MockPath:
        """Mock object to simulate LocalUploadFilePaths."""

        def __init__(self, path_in_repo, size_bytes=1000):
            self.path_in_repo = path_in_repo
            self.file_path = MagicMock()
            self.file_path.stat.return_value.st_size = size_bytes

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_no_warnings_under_limits(self, mock_logger):
        """Test that no warnings are issued when under all limits."""
        paths = [
            self.MockPath("file1.txt"),
            self.MockPath("data/file2.txt"),
            self.MockPath("data/sub/file3.txt"),
        ]
        _validate_upload_limits(paths)

        # Should only have info messages, no warnings
        mock_logger.warning.assert_not_called()

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_warns_too_many_total_files(self, mock_logger):
        """Test warning when total files exceed MAX_FILES_PER_REPO."""
        # Create a list with more files than the limit
        paths = [self.MockPath(f"file{i}.txt") for i in range(MAX_FILES_PER_REPO + 10)]
        _validate_upload_limits(paths)

        # Check that the appropriate warning was logged
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any(f"{MAX_FILES_PER_REPO + 10:,} files" in call for call in warning_calls)
        assert any("exceeds the recommended limit" in call for call in warning_calls)

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_warns_too_many_subdirectories(self, mock_logger):
        """Test warning when a folder has too many subdirectories."""
        # Create files in many subdirectories under "data"
        paths = []
        for i in range(MAX_FILES_PER_FOLDER + 10):
            paths.append(self.MockPath(f"data/subdir{i:05d}/file.txt"))

        _validate_upload_limits(paths)

        # Check that warning mentions subdirectories in "data" folder
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("data" in call and "subdirectories" in call for call in warning_calls)
        assert any(f"{MAX_FILES_PER_FOLDER + 10:,} subdirectories" in call for call in warning_calls)

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_counts_files_and_subdirs_separately(self, mock_logger):
        """Test that files and subdirectories are counted separately and correctly."""
        # Create a structure with both files and subdirs in "data"
        paths = []
        # Add 5000 files directly in data/
        for i in range(5000):
            paths.append(self.MockPath(f"data/file{i}.txt"))
        # Add 5100 subdirectories with files (exceeds limit when combined)
        for i in range(5100):
            paths.append(self.MockPath(f"data/subdir{i}/file.txt"))

        _validate_upload_limits(paths)

        # Should warn about "data" having 10,100 entries (5000 files + 5100 subdirs)
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("data" in call and "10,100 entries" in call for call in warning_calls)
        assert any("5,000 files" in call and "5,100 subdirectories" in call for call in warning_calls)

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_file_size_decimal_gb(self, mock_logger):
        """Test that file sizes are calculated using decimal GB (10^9 bytes)."""
        # Create a file that's 21 GB in decimal (21 * 10^9 bytes)
        size_bytes = 21 * 1_000_000_000
        paths = [self.MockPath("large_file.bin", size_bytes=size_bytes)]

        _validate_upload_limits(paths)

        # Should warn about file being larger than 20GB recommended
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("21.0GB" in call or "21GB" in call for call in warning_calls)
        assert any("20GB (recommended limit)" in call for call in warning_calls)

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_very_large_file_warning(self, mock_logger):
        """Test warning for files exceeding hard limit (50GB)."""
        # Create a file that's 51 GB
        size_bytes = 51 * 1_000_000_000
        paths = [self.MockPath("huge_file.bin", size_bytes=size_bytes)]

        _validate_upload_limits(paths)

        # Should warn about file exceeding 50GB hard limit
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("51.0GB" in call or "51GB" in call for call in warning_calls)
        assert any("50GB hard limit" in call for call in warning_calls)

    @patch("huggingface_hub._upload_large_folder.logger")
    def test_nested_directory_structure(self, mock_logger):
        """Test correct handling of deeply nested directory structures."""
        paths = [
            self.MockPath("a/b/c/d/e/file1.txt"),
            self.MockPath("a/b/c/d/e/file2.txt"),
            self.MockPath("a/b/c/d/f/file3.txt"),
            self.MockPath("a/b/c/g/file4.txt"),
        ]

        _validate_upload_limits(paths)

        # Should not warn - each folder has at most 2 entries
        mock_logger.warning.assert_not_called()
