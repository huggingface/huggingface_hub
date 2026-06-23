# Copyright 2026-present, the HuggingFace Inc. team.
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
"""On-disk cache for repository tree listings.

A tree listing is the set of files (with their download metadata) contained in a repo at a given commit. Because a
commit hash is immutable, its tree listing never changes and can be cached forever without any invalidation logic.

The listing is stored as a human-readable JSON file under `<storage_folder>/trees/<commit_hash>.json`, next to the
existing `refs/`, `blobs/` and `snapshots/` folders.

```json
{
  "format_version": 1,
  "files": {
    "config.json": {"size": 519, "blob_id": "<git sha1>"},
    "model.safetensors": {"size": 1234, "blob_id": "...", "lfs_sha256": "<sha256>", "lfs_size": 1234, "xet_hash": "..."}
  }
}
```
"""

import json
import os
import tempfile
import threading
from dataclasses import dataclass

from .utils import logging


logger = logging.get_logger(__name__)

TREE_CACHE_FORMAT_VERSION = 1

# In-memory cache of parsed tree listings, keyed by absolute file path.
_IN_MEMORY_TREE_CACHE: dict[str, "dict[str, TreeCacheEntry]"] = {}
_IN_MEMORY_TREE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class TreeCacheEntry:
    """Raw metadata of a single file in a cached tree listing.

    This is a dumb container that mirrors the fields returned by the `/tree` endpoint. It does not derive any
    download metadata (e.g. which field is the `/resolve` ETag); that logic lives in `file_download.py`, which
    owns the knowledge of how the server serves files.
    """

    path: str
    size: int
    blob_id: str
    lfs_sha256: str | None = None
    lfs_size: int | None = None
    xet_hash: str | None = None


def _tree_cache_path(storage_folder: str, commit_hash: str) -> str:
    return os.path.join(storage_folder, "trees", f"{commit_hash}.json")


def read_tree_cache(storage_folder: str, commit_hash: str) -> dict[str, TreeCacheEntry] | None:
    """Return the cached tree listing for a commit hash, or `None` if not cached (or unreadable)."""
    path = _tree_cache_path(storage_folder, commit_hash)
    with _IN_MEMORY_TREE_CACHE_LOCK:
        if path in _IN_MEMORY_TREE_CACHE:
            return _IN_MEMORY_TREE_CACHE[path]
    entries = _read_tree_cache_from_disk(path)
    if entries is not None:
        with _IN_MEMORY_TREE_CACHE_LOCK:
            _IN_MEMORY_TREE_CACHE[path] = entries
    return entries


def _read_tree_cache_from_disk(path: str) -> dict[str, TreeCacheEntry] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format_version") != TREE_CACHE_FORMAT_VERSION:
            # Unknown format (e.g. written by a newer version) => ignore and re-fetch.
            return None
        return {
            file_path: TreeCacheEntry(
                path=file_path,
                size=info["size"],
                blob_id=info["blob_id"],
                lfs_sha256=info.get("lfs_sha256"),
                lfs_size=info.get("lfs_size"),
                xet_hash=info.get("xet_hash"),
            )
            for file_path, info in data["files"].items()
        }
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Ignoring corrupted tree cache file {path}: {e}")
        return None


def write_tree_cache(storage_folder: str, commit_hash: str, entries: dict[str, TreeCacheEntry]) -> None:
    """Write the tree listing of a commit hash to the cache (ignoring any failures)."""
    path = _tree_cache_path(storage_folder, commit_hash)
    files: dict[str, dict] = {}
    for file_path in sorted(entries):
        entry = entries[file_path]
        info: dict = {"size": entry.size, "blob_id": entry.blob_id}
        if entry.lfs_sha256 is not None:
            info["lfs_sha256"] = entry.lfs_sha256
            info["lfs_size"] = entry.lfs_size
        if entry.xet_hash is not None:
            info["xet_hash"] = entry.xet_hash
        files[file_path] = info
    data = {
        "format_version": TREE_CACHE_FORMAT_VERSION,
        "files": files,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning(f"Ignored error while writing tree cache file {path}: {e}")
        return
