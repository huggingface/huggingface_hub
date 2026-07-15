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
"""Cache-wide shared blob store, deduplicating Xet files across cached repos.

Store entries at `<cache_dir>/blobs/<2-hex-prefix>/<xet_hash>` are hardlinks of per-repo
`blobs/<etag>` files: one copy on disk, shared across repos, safe for older versions.
"""

import os
import re
import uuid
from pathlib import Path

from . import constants
from .utils import logging


logger = logging.get_logger(__name__)

_XET_HASH_REGEX = re.compile(r"[0-9a-f]{64}")

SHARED_BLOBS_DIR_NAME = "blobs"

_are_hardlinks_supported_in_dir: dict[str, bool] = {}


def shared_blobs_dir(cache_dir: str | Path) -> Path:
    """Return the path of the shared blob store inside a cache directory."""
    return Path(cache_dir) / SHARED_BLOBS_DIR_NAME


def shared_blob_path(cache_dir: str | Path, xet_hash: str) -> Path:
    """Return the store path for a given Xet file hash.

    Raises `ValueError` if `xet_hash` is not a valid Xet hash.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        raise ValueError(f"Invalid Xet file hash: '{xet_hash}'.")
    return shared_blobs_dir(cache_dir) / xet_hash[:2] / xet_hash


def are_hardlinks_supported(cache_dir: str | Path) -> bool:
    """Return whether hardlinks are supported in the given directory.

    Checked once per directory, as support depends on the mounted filesystem.
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

    `HF_HUB_DISABLE_XET` also disables store reads (escape hatch for a misbehaving
    `hf_xet`). The download path must additionally require `are_symlinks_supported`:
    the store is unusable in the degraded no-symlink layout.
    """
    return (
        constants.HF_HUB_ENABLE_SHARED_BLOBS
        and not constants.HF_HUB_DISABLE_XET
        and are_hardlinks_supported(cache_dir)
    )


def try_link_from_shared_store(
    *, blob_path: str, xet_hash: str, cache_dir: str | Path, expected_size: int | None
) -> bool:
    """Materialize `blobs/<etag>` as a hardlink to an existing store entry, if any.

    Returns `True` if the blob is now available locally (no download needed). An entry
    with an unexpected size is treated as corrupted and evicted.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
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


def has_shared_blob(*, xet_hash: str, cache_dir: str | Path, expected_size: int | None) -> bool:
    """Return whether a valid store entry exists for this hash, without touching anything.

    Read-only variant of `try_link_from_shared_store` for `dry_run`: no eviction, no repo
    link, no filesystem probes.
    """
    if not constants.HF_HUB_ENABLE_SHARED_BLOBS or constants.HF_HUB_DISABLE_XET:
        return False
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        return False
    try:
        store_stat = shared_blob_path(cache_dir, xet_hash).stat()
    except OSError:
        return False
    return expected_size is None or store_stat.st_size == expected_size


def publish_blob_to_shared_store(*, blob_path: str, xet_hash: str, cache_dir: str | Path) -> None:
    """Hardlink a freshly downloaded (and Xet-verified) blob into the shared store.

    Best-effort: failures are logged and ignored. An existing entry is replaced by the
    fresh copy; other repos' hardlinks keep their inode.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
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
    """Remove store entries no longer referenced by any repo (`st_nlink == 1`).

    Returns the number of bytes freed. Racing a concurrent download is benign: at worst
    the entry is re-published.
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


def shared_store_inodes(cache_dir: str | Path) -> set[tuple[int, int]]:
    """Return the `(st_dev, st_ino)` of all store entries.

    Used by `delete_revisions` to predict freed sizes.
    """
    store_dir = shared_blobs_dir(cache_dir)
    inodes: set[tuple[int, int]] = set()
    if not store_dir.is_dir():
        return inodes
    for prefix_dir in store_dir.iterdir():
        if not prefix_dir.is_dir():
            continue
        for entry in prefix_dir.iterdir():
            try:
                stat = entry.lstat()
            except OSError:
                continue
            inodes.add((stat.st_dev, stat.st_ino))
    return inodes
