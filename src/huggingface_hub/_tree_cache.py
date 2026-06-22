"""On-disk cache for repository tree listings.

A tree listing is the set of files (with their download metadata) contained in a repo at a given
commit. Because a commit hash is immutable, its tree listing never changes and can be cached forever
without any invalidation logic.

The listing is stored as a human-readable JSON file under `<storage_folder>/trees/<commit_hash>.json`,
next to the existing `refs/`, `blobs/` and `snapshots/` folders. It maps each file path to the
metadata needed to download it without a per-file HEAD call:

```json
{
  "format_version": 1,
  "repo_id": "user/repo",
  "repo_type": "model",
  "commit_hash": "...",
  "files": {
    "config.json": {"size": 519, "blob_id": "<git sha1>"},
    "model.safetensors": {"size": 1234, "blob_id": "...", "lfs_sha256": "<sha256>", "lfs_size": 1234, "xet_hash": "..."}
  }
}
```

`snapshot_download` reads this cache to skip both the repo-wide tree listing call and the per-file
HEAD call on warm pulls. See `_snapshot_download.py` for usage.
"""

import json
import os
import tempfile
from dataclasses import dataclass

from .utils import logging


logger = logging.get_logger(__name__)

TREE_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TreeCacheEntry:
    """Metadata of a single file in a cached tree listing."""

    path: str
    size: int
    blob_id: str
    lfs_sha256: str | None = None
    lfs_size: int | None = None
    xet_hash: str | None = None

    @property
    def etag(self) -> str:
        """The value the `/resolve` endpoint returns as ETag: LFS sha256 for LFS files, git blob id otherwise.

        This is what the blob file is named after in the cache, so it must match `/resolve` exactly to keep
        caches built by different code paths interoperable.
        """
        return self.lfs_sha256 if self.lfs_sha256 is not None else self.blob_id

    @property
    def file_size(self) -> int:
        """The real file size: LFS size for LFS files, git blob size otherwise."""
        return self.lfs_size if self.lfs_size is not None else self.size


def _tree_cache_path(storage_folder: str, commit_hash: str) -> str:
    return os.path.join(storage_folder, "trees", f"{commit_hash}.json")


def read_tree_cache(storage_folder: str, commit_hash: str) -> dict[str, TreeCacheEntry] | None:
    """Return the cached tree listing for a commit hash, or `None` if not cached (or unreadable).

    The returned dict is keyed by file path for direct lookup.
    """
    path = _tree_cache_path(storage_folder, commit_hash)
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


def write_tree_cache(
    storage_folder: str,
    commit_hash: str,
    repo_id: str,
    repo_type: str,
    entries: dict[str, TreeCacheEntry],
) -> None:
    """Write the tree listing of a commit hash to the cache (atomic, best-effort).

    Failures are logged and ignored: the tree cache is an optimization, never a requirement.
    """
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
        "repo_id": repo_id,
        "repo_type": repo_type,
        "commit_hash": commit_hash,
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
