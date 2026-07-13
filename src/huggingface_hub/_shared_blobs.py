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
"""Cache-wide shared blob store, deduplicating Xet-backed files across cached repos.

Layout: `<cache_dir>/blobs/<2-hex-prefix>/<xet_hash>`. The store is content-addressed by
the Xet file hash, which is verified server-side on upload (unlike the sha256 etag, which
is never checked). Entries are only ever created from files downloaded through the
verified `hf_xet` path.

Every store entry is a **hardlink** of a per-repo `blobs/<etag>` file (same inode, two
directory entries). Hardlinks - rather than symlinks - are what make the store safe to
introduce in an existing cache:

- Old versions of `huggingface_hub` and external tools see regular files. Their deletion
  code unlinks the per-repo entry only; the filesystem refcount keeps the data alive for
  other repos. A symlink-based store would get corrupted by older `delete_revisions`
  implementations, which remove the *resolved* blob path.
- Deleting a store entry can never break a repo: at worst, a future download is not
  deduplicated. This also makes every concurrency scenario benign, so no locking is
  needed: `os.link` is atomic, and a garbage collection racing with a new link only
  loses a deduplication opportunity.
- Hardlinks work on Windows/NTFS without administrator rights or Developer Mode.

The store requires the per-repo blobs and the store to live on the same filesystem,
which is guaranteed by placing it inside the cache directory. On filesystems without
hardlink support, everything degrades gracefully to the regular download path.
"""

import logging
import os
import re
import uuid
from pathlib import Path

from . import constants


# This module is imported by `utils/_cache_manager.py` while the `utils` package is
# still initializing, so it must only depend on stdlib and `constants`. The stdlib
# logger propagates to the `huggingface_hub` root logger configured in `utils.logging`,
# so log behavior is identical to `logging.get_logger(__name__)`.
logger = logging.getLogger(__name__)

# Xet file hashes are 64 lowercase hex characters (merkle root). Validating the format
# also guarantees the server-provided value cannot be used for path traversal.
_XET_HASH_REGEX = re.compile(r"^[0-9a-f]{64}$")

# Name of the store directory, at the root of the cache dir (next to `models--*` folders).
# Old versions of `huggingface_hub` report it as a single captured scan warning and
# otherwise ignore it.
SHARED_BLOBS_DIR_NAME = "blobs"

_are_hardlinks_supported_in_dir: dict[str, bool] = {}


def shared_blobs_dir(cache_dir: str | Path) -> Path:
    """Return the path of the shared blob store inside a cache directory."""
    return Path(cache_dir) / SHARED_BLOBS_DIR_NAME


def shared_blob_path(cache_dir: str | Path, xet_hash: str) -> Path:
    """Return the store path for a given Xet file hash.

    Raises `ValueError` if `xet_hash` is not a valid Xet hash.
    """
    if _XET_HASH_REGEX.match(xet_hash) is None:
        raise ValueError(f"Invalid Xet file hash: '{xet_hash}'.")
    return shared_blobs_dir(cache_dir) / xet_hash[:2] / xet_hash


def are_hardlinks_supported(cache_dir: str | Path) -> bool:
    """Return whether hardlinks are supported in the given directory.

    Like `are_symlinks_supported`, the check is done once per cache directory as support
    depends on the mounted filesystem (e.g. some network filesystems reject `link(2)`).
    """
    cache_dir = str(Path(cache_dir).expanduser().resolve())  # make it unique

    if cache_dir not in _are_hardlinks_supported_in_dir:
        os.makedirs(cache_dir, exist_ok=True)
        src_path = os.path.join(cache_dir, f".hardlink_probe_{uuid.uuid4().hex[:8]}")
        dst_path = src_path + "_dst"
        try:
            with open(src_path, "wb"):
                pass
            os.link(src_path, dst_path)
            _are_hardlinks_supported_in_dir[cache_dir] = True
        except OSError:
            _are_hardlinks_supported_in_dir[cache_dir] = False
            logger.debug(f"Hardlinks are not supported in '{cache_dir}'. Shared blob store is disabled.")
        finally:
            for path in (dst_path, src_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return _are_hardlinks_supported_in_dir[cache_dir]


def shared_blobs_enabled(cache_dir: str | Path) -> bool:
    """Return whether the shared blob store can be used for the given cache directory.

    The download path must ALSO require `are_symlinks_supported(cache_dir)` (checked at
    the call site in `file_download.py`, as this module cannot depend on it): in the
    degraded no-symlink mode, blobs are moved into `snapshots/` where users legitimately
    edit files in place - a shared inode there would propagate edits (and corruption)
    across repos, and there is no per-repo `blobs/<etag>` left to link anyway.
    """
    return constants.HF_HUB_ENABLE_SHARED_BLOBS and are_hardlinks_supported(cache_dir)


def try_link_from_shared_store(
    *, blob_path: str, xet_hash: str, cache_dir: str | Path, expected_size: int | None
) -> bool:
    """Materialize `blobs/<etag>` as a hardlink to an existing store entry, if any.

    Returns `True` if the blob is now available locally (no download needed). An entry
    with an unexpected size is treated as corrupted and evicted from the store (other
    repos hardlinking it are unaffected).
    """
    if _XET_HASH_REGEX.match(xet_hash) is None:
        return False
    store_path = shared_blob_path(cache_dir, xet_hash)
    try:
        if expected_size is not None and store_path.stat().st_size != expected_size:
            logger.warning(f"Shared blob '{store_path}' has an unexpected size. Evicting it from the store.")
            store_path.unlink(missing_ok=True)
            return False
        os.link(store_path, blob_path)
    except FileNotFoundError:
        return False
    except FileExistsError:
        # Blob created concurrently by another process: it is available either way.
        return True
    except OSError as e:
        logger.debug(f"Could not hardlink '{blob_path}' from shared blob store: {e}")
        return False
    logger.debug(f"Blob '{blob_path}' reused from shared blob store (no download needed).")
    return True


def publish_blob_to_shared_store(*, blob_path: str, xet_hash: str, cache_dir: str | Path) -> None:
    """Hardlink a freshly downloaded (and Xet-verified) blob into the shared store.

    Best-effort: any failure is logged and ignored, the cache stays fully functional
    without the store. If an entry already exists for this hash, it is replaced by the
    fresh copy: equality with the existing entry cannot be cheaply proven (a same-size
    corrupted entry is indistinguishable without a full re-hash), so the copy this
    process just verified always wins. Replacing a store entry never mutates other
    repos' blobs - existing hardlinks keep their inode.
    """
    if _XET_HASH_REGEX.match(xet_hash) is None:
        return
    store_path = shared_blob_path(cache_dir, xet_hash)
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(blob_path, store_path)
    except FileExistsError:
        try:
            if os.path.samestat(os.stat(blob_path), os.stat(store_path)):
                return  # already the same inode, nothing to do
            store_path.unlink(missing_ok=True)
            os.link(blob_path, store_path)
        except OSError as e:
            # e.g. lost a race against a concurrent publisher: its copy is verified too.
            logger.debug(f"Could not replace shared blob store entry for '{blob_path}': {e}")
    except OSError as e:
        logger.debug(f"Could not publish '{blob_path}' to shared blob store: {e}")


def sweep_shared_blobs(cache_dir: str | Path) -> int:
    """Remove store entries that are no longer referenced by any cached repo.

    An entry with `st_nlink == 1` has no per-repo hardlink left (e.g. all repos using it
    have been deleted). Removing it is what actually frees disk space. Returns the number
    of bytes freed.

    Racing with a concurrent download is benign: at worst the download loses a
    deduplication opportunity and re-publishes the entry.
    """
    store_dir = shared_blobs_dir(cache_dir)
    if not store_dir.is_dir():
        return 0

    freed_bytes = 0
    for prefix_dir in store_dir.iterdir():
        if not prefix_dir.is_dir():
            continue
        for entry in prefix_dir.iterdir():
            try:
                stat = entry.lstat()
                if stat.st_nlink <= 1:
                    entry.unlink()
                    freed_bytes += stat.st_size
            except OSError:
                continue
        try:
            prefix_dir.rmdir()  # cleanup empty prefix dirs, fails silently if non-empty
        except OSError:
            pass
    return freed_bytes


def shared_store_inodes(cache_dir: str | Path) -> set[int]:
    """Return the inodes of all store entries.

    Used by `delete_revisions` to predict freed sizes: a blob whose only remaining
    hardlink is a store entry will be freed by the post-deletion sweep.
    """
    store_dir = shared_blobs_dir(cache_dir)
    inodes: set[int] = set()
    if not store_dir.is_dir():
        return inodes
    for prefix_dir in store_dir.iterdir():
        if not prefix_dir.is_dir():
            continue
        for entry in prefix_dir.iterdir():
            try:
                inodes.add(entry.lstat().st_ino)
            except OSError:
                continue
    return inodes
