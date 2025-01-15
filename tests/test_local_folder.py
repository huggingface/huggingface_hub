# coding=utf-8
# Copyright 2024-present, the HuggingFace Inc. team.
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
"""Contains tests for the `.cache/huggingface` folder in local directories.

See `huggingface_hub/src/_local_folder.py` for the implementation.
"""

import logging
import os
import threading
import time
from pathlib import Path, WindowsPath

import pytest

from huggingface_hub._local_folder import (
    LocalDownloadFileMetadata,
    LocalDownloadFilePaths,
    LocalUploadFilePaths,
    _huggingface_dir,
    get_local_download_paths,
    get_local_upload_paths,
    read_download_metadata,
    write_download_metadata,
)


def test_creates_huggingface_dir_with_gitignore(tmp_path: Path):
    """Test `.cache/huggingface/` dir is ignored by git."""
    local_dir = tmp_path / "path" / "to" / "local"
    huggingface_dir = _huggingface_dir(local_dir)
    assert huggingface_dir == local_dir / ".cache" / "huggingface"
    assert huggingface_dir.exists()  # all subdirectories have been created
    assert huggingface_dir.is_dir()

    # Whole folder must be ignored
    assert (huggingface_dir / ".gitignore").exists()
    assert (huggingface_dir / ".gitignore").read_text() == "*"


def test_gitignore_lock_timeout_is_ignored(tmp_path: Path):
    local_dir = tmp_path / "path" / "to" / "local"

    threads = [threading.Thread(target=_huggingface_dir, args=(local_dir,)) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (local_dir / ".cache" / "huggingface" / ".gitignore").exists()
    assert not (local_dir / ".cache" / "huggingface" / ".gitignore.lock").exists()


def test_local_download_paths(tmp_path: Path):
    """Test local download paths are valid + usable."""
    paths = get_local_download_paths(tmp_path, "path/in/repo.txt")

    # Correct paths (also sanitized on windows)
    assert isinstance(paths, LocalDownloadFilePaths)
    assert paths.file_path == tmp_path / "path" / "in" / "repo.txt"
    assert (
        paths.metadata_path == tmp_path / ".cache" / "huggingface" / "download" / "path" / "in" / "repo.txt.metadata"
    )
    assert paths.lock_path == tmp_path / ".cache" / "huggingface" / "download" / "path" / "in" / "repo.txt.lock"

    # Paths are usable (parent directories have been created)
    assert paths.file_path.parent.is_dir()
    assert paths.metadata_path.parent.is_dir()
    assert paths.lock_path.parent.is_dir()

    # Incomplete path are etag-based
    assert (
        paths.incomplete_path("etag123")
        == tmp_path / ".cache" / "huggingface" / "download" / "path" / "in" / "repo.txt.etag123.incomplete"
    )
    assert paths.incomplete_path("etag123").parent.is_dir()


def test_local_download_paths_are_recreated_each_time(tmp_path: Path):
    paths1 = get_local_download_paths(tmp_path, "path/in/repo.txt")
    assert paths1.file_path.parent.is_dir()
    assert paths1.metadata_path.parent.is_dir()

    paths1.file_path.parent.rmdir()
    paths1.metadata_path.parent.rmdir()

    paths2 = get_local_download_paths(tmp_path, "path/in/repo.txt")
    assert paths2.file_path.parent.is_dir()
    assert paths2.metadata_path.parent.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test.")
def test_local_download_paths_long_paths(tmp_path: Path):
    """Test long path handling on Windows."""
    long_file_name = "a" * 255
    paths = get_local_download_paths(tmp_path, f"path/long/{long_file_name}.txt")

    # WindowsPath on Windows platform
    assert isinstance(paths.file_path, WindowsPath)
    assert isinstance(paths.lock_path, WindowsPath)
    assert isinstance(paths.metadata_path, WindowsPath)

    # Correct path prefixes
    assert str(paths.file_path).startswith("\\\\?\\")
    assert str(paths.lock_path).startswith("\\\\?\\")
    assert str(paths.metadata_path).startswith("\\\\?\\")


def test_write_download_metadata(tmp_path: Path):
    """Test download metadata content is valid."""
    # Write metadata
    write_download_metadata(tmp_path, filename="file.txt", commit_hash="commit_hash", etag="123456789")
    metadata_path = tmp_path / ".cache" / "huggingface" / "download" / "file.txt.metadata"
    assert metadata_path.exists()

    # Metadata is valid
    with metadata_path.open() as f:
        assert f.readline() == "commit_hash\n"
        assert f.readline() == "123456789\n"
        timestamp = float(f.readline().strip())
    assert timestamp <= time.time()  # in the past
    assert timestamp >= time.time() - 1  # but less than 1 seconds ago (we're not that slow)

    time.sleep(0.2)  # for deterministic tests

    # Overwriting works as expected
    write_download_metadata(tmp_path, filename="file.txt", commit_hash="commit_hash2", etag="987654321")
    with metadata_path.open() as f:
        assert f.readline() == "commit_hash2\n"
        assert f.readline() == "987654321\n"
        timestamp2 = float(f.readline().strip())
    assert timestamp <= timestamp2  # updated timestamp


def test_read_download_metadata_valid_metadata(tmp_path: Path):
    """Test reading download metadata when metadata is valid."""
    # Create file + write correct metadata
    (tmp_path / "file.txt").write_text("content")
    write_download_metadata(tmp_path, filename="file.txt", commit_hash="commit_hash", etag="123456789")

    # Read metadata
    metadata = read_download_metadata(tmp_path, filename="file.txt")
    assert isinstance(metadata, LocalDownloadFileMetadata)
    assert metadata.filename == "file.txt"
    assert metadata.commit_hash == "commit_hash"
    assert metadata.etag == "123456789"
    assert isinstance(metadata.timestamp, float)


def test_read_download_metadata_no_metadata(tmp_path: Path):
    """Test reading download metadata when there is no metadata."""
    # No metadata file => return None
    assert read_download_metadata(tmp_path, filename="file.txt") is None


def test_read_download_metadata_corrupted_metadata(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Test reading download metadata when metadata is corrupted."""
    # Write corrupted metadata
    metadata_path = tmp_path / ".cache" / "huggingface" / "download" / "file.txt.metadata"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("invalid content")

    # Corrupted metadata file => delete it + warn + return None
    with caplog.at_level(logging.WARNING):
        assert read_download_metadata(tmp_path, filename="file.txt") is None
        assert not metadata_path.exists()
    assert "Invalid metadata file" in caplog.text


def test_read_download_metadata_correct_metadata_missing_file(tmp_path: Path):
    """Test reading download metadata when metadata is correct but file is missing."""
    # Write correct metadata
    write_download_metadata(tmp_path, filename="file.txt", commit_hash="commit_hash", etag="123456789")

    # File missing => return None
    assert read_download_metadata(tmp_path, filename="file.txt") is None


def test_read_download_metadata_correct_metadata_but_outdated(tmp_path: Path):
    """Test reading download metadata when metadata is correct but outdated."""
    # Write correct metadata
    write_download_metadata(tmp_path, filename="file.txt", commit_hash="commit_hash", etag="123456789")
    time.sleep(2)  # We allow for a 1s difference in practice, so let's wait a bit

    # File is outdated => return None
    (tmp_path / "file.txt").write_text("content")
    assert read_download_metadata(tmp_path, filename="file.txt") is None


def test_local_upload_paths(tmp_path: Path):
    """Test local upload paths are valid + usable."""
    paths = get_local_upload_paths(tmp_path, "path/in/repo.txt")

    # Correct paths (also sanitized on windows)
    assert isinstance(paths, LocalUploadFilePaths)
    assert paths.file_path == tmp_path / "path" / "in" / "repo.txt"
    assert paths.metadata_path == tmp_path / ".cache" / "huggingface" / "upload" / "path" / "in" / "repo.txt.metadata"
    assert paths.lock_path == tmp_path / ".cache" / "huggingface" / "upload" / "path" / "in" / "repo.txt.lock"

    # Paths are usable (parent directories have been created)
    assert paths.file_path.parent.is_dir()
    assert paths.metadata_path.parent.is_dir()
    assert paths.lock_path.parent.is_dir()


def test_local_upload_paths_are_recreated_each_time(tmp_path: Path):
    paths1 = get_local_upload_paths(tmp_path, "path/in/repo.txt")
    assert paths1.file_path.parent.is_dir()
    assert paths1.metadata_path.parent.is_dir()

    paths1.file_path.parent.rmdir()
    paths1.metadata_path.parent.rmdir()

    paths2 = get_local_upload_paths(tmp_path, "path/in/repo.txt")
    assert paths2.file_path.parent.is_dir()
    assert paths2.metadata_path.parent.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test.")
def test_local_upload_paths_long_paths(tmp_path: Path):
    """Test long path handling on Windows."""
    long_file_name = "a" * 255
    paths = get_local_upload_paths(tmp_path, f"path/long/{long_file_name}.txt")

    # WindowsPath on Windows platform
    assert isinstance(paths.file_path, WindowsPath)
    assert isinstance(paths.lock_path, WindowsPath)
    assert isinstance(paths.metadata_path, WindowsPath)

    # Correct path prefixes
    assert str(paths.file_path).startswith("\\\\?\\")
    assert str(paths.lock_path).startswith("\\\\?\\")
    assert str(paths.metadata_path).startswith("\\\\?\\")
