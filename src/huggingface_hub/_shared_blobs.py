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

Store entries at `<cache_dir>/blobs/<2-hex-prefix>/<xet_hash>` are regular files.
Per-repo `blobs/<etag>` entries are relative symlinks to them. A companion
`<xet_hash>.refs` manifest lists those symlinks so cache deletion only needs to inspect
the blobs it touches instead of scanning the entire store.

The manifest is a hint, not trusted metadata: every entry is validated against the
filesystem before use, and any missing or unreadable manifest makes garbage collection
leak the shared blob rather than risk deleting referenced content.
"""

import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from filelock import FileLock, SoftFileLock

from . import constants
from .utils import logging


logger = logging.get_logger(__name__)

_XET_HASH_REGEX = re.compile(r"[0-9a-f]{64}")
_REPO_DIR_REGEX = re.compile(r"(?:models|datasets|spaces)--.+")

SHARED_BLOBS_DIR_NAME = "blobs"
SHARED_BLOBS_MARKER_NAME = ".huggingface-shared-blobs"
SHARED_BLOBS_LAYOUT_VERSION = "1"
_MANIFEST_SUFFIX = ".refs"
_LOCK_SUFFIX = ".lock"
_SOFT_LOCK_TIMEOUT = 10
_MARKER_TMP_REGEX = re.compile(rf"{re.escape(SHARED_BLOBS_MARKER_NAME)}\.[0-9a-f]{{8}}\.tmp")


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


def shared_blob_manifest_path(cache_dir: str | Path, xet_hash: str) -> Path:
    """Return the reference manifest path for a given Xet file hash."""
    return shared_blob_path(cache_dir, xet_hash).with_name(f"{xet_hash}{_MANIFEST_SUFFIX}")


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _is_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _is_usable_store_entry(store_path: Path, expected_size: int) -> bool:
    """Return whether `store_path` is a regular, readable payload of the expected size."""
    try:
        store_stat = store_path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(store_stat.st_mode):
        return False
    if store_stat.st_size != expected_size:
        logger.warning(
            f"Shared blob '{store_path}' has an unexpected size ({store_stat.st_size} instead of {expected_size}). "
            "Not using it."
        )
        return False
    if not os.access(store_path, os.R_OK):
        logger.warning(f"Shared blob '{store_path}' is not readable. Not using it.")
        return False
    return True


def is_shared_blobs_dir(path: str | Path) -> bool:
    """Return whether `path` is an owned, supported shared blob store."""
    store_dir = Path(path)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    if not _is_directory(store_dir) or not _is_regular_file(marker_path):
        return False
    try:
        return marker_path.read_text() == f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    except OSError:
        return False


def _cleanup_abandoned_marker_temps(store_dir: Path) -> bool:
    """Remove owned marker temporaries, refusing unexpected directory content."""
    expected_content = f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    entries = list(store_dir.iterdir())
    for entry in entries:
        if _MARKER_TMP_REGEX.fullmatch(entry.name) is None or not _is_regular_file(entry):
            return False
        try:
            if not expected_content.startswith(entry.read_text()):
                return False
        except (OSError, UnicodeError):
            return False
    for entry in entries:
        entry.unlink(missing_ok=True)
    return not any(store_dir.iterdir())


def _ensure_shared_blobs_dir(cache_dir: str | Path) -> bool:
    """Create and mark the shared store, refusing to adopt an unmarked directory."""
    store_dir = shared_blobs_dir(cache_dir)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    try:
        store_dir.mkdir(exist_ok=True)
        if is_shared_blobs_dir(store_dir):
            return True
        if not _is_directory(store_dir) or not _cleanup_abandoned_marker_temps(store_dir):
            logger.debug(f"Refusing to use unmarked shared blob directory '{store_dir}'.")
            return False
        _repair_shared_directory_mode(store_dir, cache_dir)

        tmp_marker = marker_path.with_name(f"{marker_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp_marker.write_text(f"{SHARED_BLOBS_LAYOUT_VERSION}\n")
            tmp_marker.chmod(_shared_blob_mode(cache_dir))
            os.replace(tmp_marker, marker_path)
        finally:
            tmp_marker.unlink(missing_ok=True)
        return is_shared_blobs_dir(store_dir)
    except OSError as e:
        logger.debug(f"Could not initialize shared blob directory '{store_dir}': {e}")
        return False


def _ensure_prefix_dir(cache_dir: str | Path, xet_hash: str) -> Path | None:
    if not _ensure_shared_blobs_dir(cache_dir):
        return None
    prefix_dir = shared_blob_path(cache_dir, xet_hash).parent
    try:
        prefix_dir.mkdir(exist_ok=True)
    except OSError as e:
        logger.debug(f"Could not create shared blob prefix directory '{prefix_dir}': {e}")
        return None
    if not _is_directory(prefix_dir):
        logger.debug(f"Refusing to use non-directory shared blob prefix '{prefix_dir}'.")
        return None
    try:
        _repair_shared_directory_mode(prefix_dir, cache_dir)
    except OSError as e:
        logger.debug(f"Could not set shared blob prefix permissions on '{prefix_dir}': {e}")
        return None
    return prefix_dir


def _shared_blob_mode(cache_dir: str | Path) -> int:
    """Return a read-only mode accessible to users who can traverse the cache root."""
    if os.name == "nt":
        # Windows access is governed by ACLs; 0444 would set the read-only attribute
        # and prevent later atomic replacement or garbage collection.
        return 0o666
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o400
    return 0o400 | (0o040 if cache_mode & stat.S_IXGRP else 0) | (0o004 if cache_mode & stat.S_IXOTH else 0)


def _shared_directory_mode(cache_dir: str | Path) -> int:
    """Mirror cache-root access and inheritance bits on newly created store directories."""
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o700
    return cache_mode & (0o777 | stat.S_ISGID | stat.S_ISVTX)


def _repair_shared_directory_mode(path: Path, cache_dir: str | Path) -> None:
    expected_mode = _shared_directory_mode(cache_dir)
    if stat.S_IMODE(path.lstat().st_mode) != expected_mode:
        path.chmod(expected_mode)


def _prepare_shared_blob_permissions(blob_path: Path, prefix_dir: Path, cache_dir: str | Path) -> None:
    """Make a payload immutable and readable according to the shared cache policy."""
    blob_mode = _shared_blob_mode(cache_dir)
    os.chmod(blob_path, blob_mode)
    if os.name == "nt" or not hasattr(os, "chown"):
        return
    target_gid = prefix_dir.stat().st_gid
    if blob_path.stat().st_gid == target_gid:
        return
    try:
        os.chown(blob_path, -1, target_gid)
    except OSError:
        # Without other-read, publishing with the wrong group would create a shared
        # entry that other cache users cannot read. Fall back to repo-local storage.
        if blob_mode & stat.S_IRGRP and not blob_mode & stat.S_IROTH:
            raise


def _path_for_comparison(path: str | Path) -> Path:
    """Return an absolute path without the Windows extended-length prefix."""
    path_str = os.fspath(path)
    if path_str[:8].lower() == "\\\\?\\unc\\":
        path_str = f"\\\\{path_str[8:]}"
    elif path_str.startswith("\\\\?\\"):
        path_str = path_str[4:]
    return Path(os.path.abspath(path_str))


def _relative_blob_path(blob_path: str | Path, cache_dir: str | Path) -> str | None:
    blob_path = _path_for_comparison(blob_path)
    cache_dir = _path_for_comparison(cache_dir)
    try:
        relative_path = blob_path.relative_to(cache_dir)
    except ValueError:
        return None
    if (
        len(relative_path.parts) != 3
        or _REPO_DIR_REGEX.fullmatch(relative_path.parts[0]) is None
        or relative_path.parts[1] != "blobs"
        or not relative_path.parts[2]
    ):
        return None
    relative_str = relative_path.as_posix()
    return None if "\n" in relative_str or "\r" in relative_str else relative_str


def _shared_blob_target(
    blob_path: str | Path, cache_dir: str | Path, *, raise_on_io_error: bool = False
) -> Path | None:
    """Return the canonical store target if `blob_path` is a managed shared symlink."""
    blob_path = Path(blob_path)
    if _relative_blob_path(blob_path, cache_dir) is None:
        return None
    try:
        if not stat.S_ISLNK(blob_path.lstat().st_mode):
            return None
        link_target = Path(os.readlink(blob_path))
    except FileNotFoundError:
        return None
    except OSError:
        if raise_on_io_error:
            raise
        return None

    comparable_blob_path = _path_for_comparison(blob_path)
    target = link_target if link_target.is_absolute() else comparable_blob_path.parent / link_target
    target = _path_for_comparison(target)
    store_dir = _path_for_comparison(shared_blobs_dir(cache_dir))
    try:
        relative_target = target.relative_to(store_dir)
    except ValueError:
        return None
    if (
        len(relative_target.parts) != 2
        or _XET_HASH_REGEX.fullmatch(relative_target.parts[1]) is None
        or relative_target.parts[0] != relative_target.parts[1][:2]
    ):
        return None
    return shared_blobs_dir(cache_dir) / relative_target


def shared_blob_target(blob_path: str | Path, cache_dir: str | Path) -> Path | None:
    """Return the canonical store target if `blob_path` is a valid shared symlink.

    Callers check `is_shared_blobs_dir` once before resolving many blobs.
    """
    target = _shared_blob_target(blob_path, cache_dir)
    return target if target is not None and _is_regular_file(target) else None


def _manifest_path_for_store_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}{_MANIFEST_SUFFIX}")


def _lock_path_for_store_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}{_LOCK_SUFFIX}")


@contextmanager
def _shared_blob_lock(store_path: Path) -> Generator[None, None, None]:
    """Lock publication, reference creation, and GC for one content hash."""
    lock_path = _lock_path_for_store_path(store_path)
    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"Shared blob lock is not a regular file: '{lock_path}'.")
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
    finally:
        os.close(fd)
    lock = FileLock(lock_path, mode=0o666)
    try:
        lock.acquire()
    except NotImplementedError:
        # A SoftFileLock uses file existence as the lock, so it cannot reuse the
        # persistent, cross-user-writable flock file prepared above.
        lock = SoftFileLock(f"{lock_path}.soft", mode=0o666)
        lock.acquire(timeout=_SOFT_LOCK_TIMEOUT)
    try:
        yield
    finally:
        try:
            lock.release()
        except OSError:
            pass


def _append_manifest_reference(manifest_path: Path, relative_blob_path: str) -> None:
    """Append and flush one reference before its symlink is made visible."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(manifest_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"Shared blob manifest is not a regular file: '{manifest_path}'.")
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
        line = f"{relative_blob_path}\n".encode()
        if os.write(fd, line) != len(line):
            raise OSError(f"Could not append a complete reference to '{manifest_path}'.")
        os.fsync(fd)
    finally:
        os.close(fd)


def _make_temporary_symlink(blob_path: Path, store_path: Path) -> Path:
    tmp_link = blob_path.with_name(f".{blob_path.name}.{uuid.uuid4().hex[:8]}.shared")
    relative_target = os.path.relpath(_path_for_comparison(store_path), start=_path_for_comparison(blob_path).parent)
    os.symlink(relative_target, tmp_link)
    return tmp_link


def shared_blobs_enabled() -> bool:
    """Return whether the shared blob store may be used."""
    return not constants.HF_HUB_DISABLE_SHARED_BLOBS and not constants.HF_HUB_DISABLE_XET


def try_link_from_shared_store(*, blob_path: str, xet_hash: str, cache_dir: str | Path, expected_size: int) -> bool:
    """Materialize `blobs/<etag>` as a symlink to an existing store entry, if any.

    The reference manifest is flushed before the symlink becomes visible. Failures are
    best-effort misses and leave the regular download path untouched.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None or not is_shared_blobs_dir(shared_blobs_dir(cache_dir)):
        return False
    store_path = shared_blob_path(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if relative_blob_path is None:
        return False

    tmp_link: Path | None = None
    try:
        with _shared_blob_lock(store_path):
            if not _is_usable_store_entry(store_path, expected_size):
                return False

            tmp_link = _make_temporary_symlink(Path(blob_path), store_path)
            _append_manifest_reference(_manifest_path_for_store_path(store_path), relative_blob_path)
            os.replace(tmp_link, blob_path)
    except OSError as e:
        logger.debug(f"Could not symlink '{blob_path}' from shared blob store: {e}")
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(f"Blob '{blob_path}' reused from shared blob store (no download needed).")
    return True


def has_shared_blob(*, xet_hash: str, cache_dir: str | Path, expected_size: int) -> bool:
    """Return whether a usable store entry exists, without touching the cache."""
    if (
        not shared_blobs_enabled()
        or _XET_HASH_REGEX.fullmatch(xet_hash) is None
        or not is_shared_blobs_dir(shared_blobs_dir(cache_dir))
    ):
        return False
    return _is_usable_store_entry(shared_blob_path(cache_dir, xet_hash), expected_size)


def publish_blob_to_shared_store(
    *,
    blob_path: str,
    xet_hash: str,
    cache_dir: str | Path,
    expected_size: int,
    replace_existing: bool = False,
) -> bool:
    """Move a fresh Xet download into the store and replace its repo blob with a symlink.

    Best-effort: on failure the repo blob remains (or is restored as) a regular local
    file. Returns whether the repo blob was successfully shared.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        return False
    prefix_dir = _ensure_prefix_dir(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if prefix_dir is None or relative_blob_path is None:
        return False

    blob_path_obj = Path(blob_path)
    store_path = shared_blob_path(cache_dir, xet_hash)
    manifest_path = _manifest_path_for_store_path(store_path)
    tmp_link: Path | None = None
    blob_moved = False
    try:
        with _shared_blob_lock(store_path):
            tmp_link = _make_temporary_symlink(blob_path_obj, store_path)
            store_is_usable = not replace_existing and _is_usable_store_entry(store_path, expected_size)
            _append_manifest_reference(manifest_path, relative_blob_path)
            if not store_is_usable:
                _prepare_shared_blob_permissions(blob_path_obj, prefix_dir, cache_dir)
                os.replace(blob_path_obj, store_path)
                blob_moved = True

            os.replace(tmp_link, blob_path_obj)
    except OSError as e:
        logger.debug(f"Could not publish '{blob_path}' to shared blob store: {e}")
        if blob_moved and not os.path.lexists(blob_path_obj):
            try:
                shutil.copyfile(store_path, blob_path_obj)
            except OSError as restore_error:
                raise OSError(
                    f"Could not restore repo blob '{blob_path}' after shared-store failure"
                ) from restore_error
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(f"Blob '{blob_path}' published to shared blob store.")
    return True


def _read_valid_manifest_references(store_path: Path, cache_dir: str | Path) -> set[Path] | None:
    manifest_path = _manifest_path_for_store_path(store_path)
    if not _is_regular_file(manifest_path):
        return None
    try:
        lines = manifest_path.read_text().splitlines()
    except (OSError, UnicodeError):
        return None

    references: set[Path] = set()
    cache_dir = Path(os.path.abspath(cache_dir))
    for line in lines:
        relative_path = Path(line)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        candidate = cache_dir / relative_path
        if _relative_blob_path(candidate, cache_dir) is None:
            continue
        try:
            target = _shared_blob_target(candidate, cache_dir, raise_on_io_error=True)
        except OSError:
            return None
        if target == store_path:
            references.add(candidate)
    return references


def expected_shared_blob_freed_size(
    store_path: Path, *, blob_paths_to_delete: set[Path], cache_dir: str | Path
) -> int:
    """Predict bytes freed if all known references are part of a deletion plan."""
    references = _read_valid_manifest_references(store_path, cache_dir)
    if references is None or not references or not references.issubset(blob_paths_to_delete):
        return 0
    try:
        return store_path.lstat().st_size if _is_regular_file(store_path) else 0
    except OSError:
        return 0


def shared_store_blob_paths(cache_dir: str | Path) -> set[Path]:
    """Return all valid canonical payload paths in a marked shared store."""
    store_dir = shared_blobs_dir(cache_dir)
    if not is_shared_blobs_dir(store_dir):
        return set()
    paths: set[Path] = set()
    try:
        prefix_dirs = list(store_dir.iterdir())
    except OSError:
        return paths
    for prefix_dir in prefix_dirs:
        if re.fullmatch(r"[0-9a-f]{2}", prefix_dir.name) is None or not _is_directory(prefix_dir):
            continue
        try:
            entries = list(prefix_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if (
                _XET_HASH_REGEX.fullmatch(entry.name) is not None
                and entry.name.startswith(prefix_dir.name)
                and _is_regular_file(entry)
            ):
                paths.add(entry)
    return paths


def unreferenced_shared_blobs(cache_dir: str | Path) -> dict[Path, int]:
    """Return store payloads without any valid manifest reference, with their sizes.

    Payloads with a missing or unreadable manifest are not listed, consistent with `sweep_shared_blob`.
    """
    unreferenced: dict[Path, int] = {}
    for store_path in shared_store_blob_paths(cache_dir):
        references = _read_valid_manifest_references(store_path, cache_dir)
        if references is not None and not references:
            try:
                unreferenced[store_path] = store_path.lstat().st_size
            except OSError:
                pass
    return unreferenced


def _rewrite_manifest(manifest_path: Path, references: set[Path], cache_dir: str | Path) -> None:
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    cache_dir = Path(os.path.abspath(cache_dir))
    try:
        content = "".join(f"{path.relative_to(cache_dir).as_posix()}\n" for path in sorted(references))
        tmp_path.write_text(content)
        tmp_path.chmod(0o666)
        with tmp_path.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(tmp_path, manifest_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def sweep_shared_blob(store_path: Path, *, cache_dir: str | Path) -> int:
    """Collect one shared blob if its manifest has no remaining valid references.

    Missing, invalid, or unreadable metadata is never grounds for deletion. Returns the
    number of payload bytes removed.
    """
    store_path = Path(os.path.abspath(store_path))
    cache_dir = Path(os.path.abspath(cache_dir))
    expected_store_dir = Path(os.path.abspath(shared_blobs_dir(cache_dir)))
    try:
        relative_store_path = store_path.relative_to(expected_store_dir)
    except ValueError:
        return 0
    if (
        not is_shared_blobs_dir(expected_store_dir)
        or len(relative_store_path.parts) != 2
        or _XET_HASH_REGEX.fullmatch(store_path.name) is None
        or store_path.parent.name != store_path.name[:2]
    ):
        return 0

    try:
        with _shared_blob_lock(store_path):
            references = _read_valid_manifest_references(store_path, cache_dir)
            if references is None:
                return 0
            manifest_path = _manifest_path_for_store_path(store_path)
            if references:
                _rewrite_manifest(manifest_path, references, cache_dir)
                return 0
            if not _is_regular_file(store_path):
                return 0
            size = store_path.lstat().st_size
            store_path.unlink()
            manifest_path.unlink(missing_ok=True)
            return size
    except OSError as e:
        logger.debug(f"Could not collect shared blob '{store_path}': {e}")
        return 0
