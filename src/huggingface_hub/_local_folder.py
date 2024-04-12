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
"""Contains utilities to handle the `../.huggingface` folder in local directories.

First discussed in https://github.com/huggingface/huggingface_hub/issues/1738 to store
download metadata when downloading files from the hub to a local directory (without
using the cache).

./.huggingface folder structure:
[4.0K]  data
├── [4.0K]  .huggingface
│   └── [4.0K]  download
│       ├── [  16]  file.parquet.metadata
│       ├── [  16]  file.txt.metadata
│       └── [4.0K]  folder
│           └── [  16]  file.parquet.metadata
│
├── [6.5G]  file.parquet
├── [1.5K]  file.txt
└── [4.0K]  folder
    └── [   16]  file.parquet


Metadata file structure:
```
# file.txt.metadata
{
    "commit_hash": "11c5a3d5811f50298f278a704980280950aedb10",
    "etag": "a16a55fda99d2f2e7b69cce5cf93ff4ad3049930",
    "timestamp": 1712656091
}


# file.parquet.metadata
{
    "commit_hash": "11c5a3d5811f50298f278a704980280950aedb10",
    "etag": "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421",
    "timestamp": 1712656091
}
```
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .utils import WeakFileLock


logger = logging.getLogger(__name__)


@dataclass
class LocalDownloadFileMetadata:
    """
    Metadata about a file in the local directory related to a download process.

    Attributes:
        filename (`str`):
            Path of the file in the repo.
        commit_hash (`str`):
            Commit hash of the file in the repo.
        etag (`str`):
            ETag of the file in the repo. Used to check if the file has changed.
            For LFS files, this is the sha256 of the file. For regular, it correspond to the git hash.
        timestamp (`int`):
            Unix timestamp of when the metadata was saved i.e. when the metadata was accurate.
    """

    filename: str
    commit_hash: str
    etag: str
    timestamp: int


def local_file_path(local_dir: Path, filename: str) -> Path:
    """Compute path to a file in the local directory.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `Path`: the path to the file.
    """
    return local_dir / _normalize_filename(filename)


def local_lock_path(local_dir: Path, filename: str) -> Path:
    """Compute path to the lock file for a file in the local directory.

    It is the same lock as for the metadata file.
    Folder containing the lock file is guaranteed to exist.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `Path`: the path to the lock file.
    """
    return _download_metadata_file_path(local_dir, filename)[0]


def local_tmp_path(local_dir: Path, filename: str) -> Path:
    """Compute path where the file will be downloaded to before being moved to correct destination.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `Path`: the path to the temporary file.
    """
    return _download_metadata_file_path(local_dir, filename)[1].with_suffix(".incomplete")


def read_download_metadata(local_dir: Path, filename: str) -> Optional[LocalDownloadFileMetadata]:
    """Read metadata about a file in the local directory related to a download process.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `[LocalDownloadFileMetadata]` or `None`: the metadata if it exists, `None` otherwise.
    """
    # file_path => path where file is downloaded
    # metadata_path => path where metadata is stored
    # lock_path => path to lock file to ensure atomic read/write of metadata
    file_path = local_file_path(local_dir, filename)
    lock_path, metadata_path = _download_metadata_file_path(local_dir, filename)
    with WeakFileLock(lock_path):
        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    metadata = json.load(f)
                    metadata = LocalDownloadFileMetadata(
                        filename=filename,
                        commit_hash=metadata["commit_hash"],
                        etag=metadata["etag"],
                        timestamp=metadata["timestamp"],
                    )
            except Exception as e:
                # remove the metadata file if it is corrupted / not a json / not the right format
                logger.warning(f"Invalid metadata file {metadata_path}: {e}. Removing it from disk and continue.")
                try:
                    metadata_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove corrupted metadata file {metadata_path}: {e}")

            try:
                # check if the file exists and hasn't been modified since the metadata was saved
                stat = file_path.stat()
                if stat.st_mtime <= metadata.timestamp:
                    return metadata
                logger.info(f"Ignoring metadata as file has been modified since metadata was saved ({filename}).")
            except FileNotFoundError:
                # file does not exist => metadata is outdated
                return None
    return None


def write_download_metadata(local_dir: Path, filename: str, commit_hash: str, etag: str) -> None:
    """Write metadata about a file in the local directory related to a download process.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
    """
    lock_path, metadata_path = _download_metadata_file_path(local_dir, filename)
    with WeakFileLock(lock_path):
        with metadata_path.open("w") as f:
            json.dump({"commit_hash": commit_hash, "etag": etag, "timestamp": int(time.time())}, f, indent=4)


def _download_metadata_file_path(local_dir: Path, filename: str) -> Tuple[Path, Path]:
    """Compute path to the metadata file for a given file in the local directory.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `Tuple[Path, Path]`: a tuple (`lock_path`, `metadata_path`). You must use the lock_path to read or write in the
                             metadata file. You are guaranteed the folder that should contain the files exists but
                             the files themselves might not exist.
    """
    path = local_dir / ".huggingface" / "download" / f"{_normalize_filename(filename)}.metadata"
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    return lock_path, path


def _normalize_filename(filename: str) -> str:
    """Normalize a filename to be cross-platform.

    Args:
        filename (`str`):
            Path of the file in the repo.

    Return:
        `str`: the normalized filename.
    """
    # filename is the path in the Hub repository (separated by '/')
    # make sure to have a cross platform transcription
    return os.path.join(*filename.split("/"))
