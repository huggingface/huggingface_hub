# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to manage the HF cache directory."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Union

from ..constants import HUGGINGFACE_HUB_CACHE
from ._typing import Literal


REPO_TYPE_T = Literal["model", "dataset", "space"]


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the huggingface cache."""


@dataclass(frozen=True)
class CachedFileInfo:
    """Frozen data structure holding information about a single cached file.

    Args:
        file_name (`str`):
            Name of the file. Example: `config.json`.
        file_path (`Path`):
            Path of the file in the `snapshots` directory. The file path is a symlink
            referring to a blob in the `blobs` folder.
        blob_path (`Path`):
            Path of the blob file. This is equivalent to `file_path.resolve()`.
        size_on_disk (`int`):
            Size of the blob file in bytes.
        size_on_disk_str (`str`):
            Size of the blob file as a human-readable string. Example: "36M".
    """

    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int

    @property
    def size_on_disk_str(self) -> str:
        """
        Return size of the blob file as a human-readable string.

        Example: "36M".
        """
        return _format_size(self.size_on_disk)


@dataclass(frozen=True)
class CachedRevisionInfo:
    r"""Frozen data structure holding information about a revision.

    A revision correspond to a folder in the `snapshots` folder and is populated with
    the exact tree structure as the repo on the Hub but contains only symlinks. A
    revision can be either referenced by 1 or more `refs` or be "detached" (no refs).

    Args:
        commit_hash (`str`):
            Hash of the revision (unique).
            Example: `"9338f7b671827df886678df2bdd7cc7b4f36dffd"`.
        snapshot_path (`Path`):
            Path to the revision directory in the `snapshots` folder. It contains the
            exact tree structure as the repo on the Hub.
        size_on_disk (`int`):
            Sum of the blob files sizes that are symlink-ed by the revision.
        files: (`Set['CachedFileInfo']`):
            Set of `CachedFileInfo` describing all files contained in the snapshot.
        refs (`Set[str]`):
            Immutable set of `refs` pointing to this revision. If the revision has no
            `refs`, it is considered detached.
            Example: `{"main", "2.4.0"}` or `{"refs/pr/1"}`.

    <Tip warning={true}>

    `size_on_disk` is not necessarily the sum of all file sizes because of possible
    duplicated files. Besides, only blobs are taken into account, not the (negligible)
    size of folders and symlinks.

    </Tip>
    """

    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: FrozenSet[CachedFileInfo]
    refs: FrozenSet[str]

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        return len(self.files)


@dataclass(frozen=True)
class CachedRepoInfo:
    r"""Frozen data structure holding information about a cached repository.

    <Tip warning={true}>

    `size_on_disk` is not necessarily the sum of all revisions sizes because of
    duplicated files. Besides, only blobs are taken into account, not the (negligible)
    size of folders and symlinks.

    </Tip>
    """

    repo_id: str
    repo_type: REPO_TYPE_T
    repo_path: Path
    revisions: FrozenSet[CachedRevisionInfo]
    size_on_disk: int
    nb_files: int

    @property
    def size_on_disk_str(self) -> str:
        """Human-readable sizes"""
        return _format_size(self.size_on_disk)

    @property
    def refs(self) -> Dict[str, CachedRevisionInfo]:
        """Mapping between refs and revision infos."""
        refs = {}
        for revision in self.revisions:
            if revision.refs is not None:
                for ref in revision.refs:
                    refs[ref] = revision
        return refs


@dataclass(frozen=True)
class HFCacheInfo:
    r"""Frozen data structure holding information about the entire cache-system.

    <Tip warning={true}>

    Here `size_on_disk` is equal to the sum of all repo sizes (only blobs).

    </Tip>
    """

    repos: FrozenSet[CachedRepoInfo]
    size_on_disk: int
    errors: List[CorruptedCacheException]

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)


def scan_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> HFCacheInfo:
    """Scan the entire cache directory and return information about it."""
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    repos: Set[CachedRepoInfo] = set()
    errors: List[CorruptedCacheException] = []
    for repo_path in Path(cache_dir).resolve().iterdir():
        try:
            if repo_path.is_dir():
                repo_info = _scan_cached_repo(repo_path)
                repos.add(repo_info)
        except CorruptedCacheException as e:
            errors.append(e)

    return HFCacheInfo(
        repos=frozenset(repos),
        size_on_disk=sum(repo.size_on_disk for repo in repos),
        errors=errors,
    )


def _scan_cached_repo(repo_path: Path) -> CachedRepoInfo:
    """Scan a single cache repo and return information about it.

    Any unexpected behavior will raise a `CorruptedCacheException`.
    """
    if not repo_path.is_dir():
        raise CorruptedCacheException(f"Repo path is not a directory: {repo_path}")

    if "--" not in repo_path.name:
        raise CorruptedCacheException(
            f"Repo path is not a valid HuggingFace cache directory: {repo_path}"
        )

    repo_type, repo_id = repo_path.name.split("--", maxsplit=1)
    repo_type = repo_type[:-1]  # "models" -> "model"
    repo_id = repo_id.replace("--", "/")  # google/fleurs -> "google/fleurs"

    if repo_type not in {"dataset", "model", "space"}:
        raise CorruptedCacheException(
            f"Repo type must be `dataset`, `model` or `space`, found `{repo_type}`"
            f" ({repo_path})."
        )

    blob_sizes: Dict[Path, int] = {}  # Key is blob_path, value is blob size (in bytes)

    snapshots_path = repo_path / "snapshots"
    refs_path = repo_path / "refs"
    blobs_path = repo_path / "blobs"

    if not snapshots_path.exists() or not snapshots_path.is_dir():
        raise CorruptedCacheException(
            f"Snapshots dir doesn't exist in cached repo: {snapshots_path}"
        )

    # Scan over `refs` directory
    refs_by_hash: Dict[str, Set[str]] = {}  # key is revision hash, value is set of refs
    if refs_path.exists():
        # Example of `refs` directory
        # ── refs
        #     ├── main
        #     └── refs
        #         └── pr
        #             └── 1
        if refs_path.is_file():
            raise CorruptedCacheException(
                f"Refs directory cannot be a file: {refs_path}"
            )

        for ref_path in refs_path.glob("**/*"):
            # glob("**/*") iterates over all files and directories -> skip directories
            if ref_path.is_dir():
                continue

            ref_name = str(ref_path.relative_to(refs_path))
            with ref_path.open() as f:
                commit_hash = f.read()

            if commit_hash in refs_by_hash:
                refs_by_hash[commit_hash].add(ref_name)
            else:
                refs_by_hash[commit_hash] = {ref_name}

    # Scan snapshots directory
    cached_revisions: Set[CachedRevisionInfo] = set()
    for revision_path in snapshots_path.iterdir():
        if revision_path.is_file():
            raise CorruptedCacheException(
                f"Snapshots folder corrupted. Found a file: {revision_path}"
            )

        cached_files = set()
        for file_path in revision_path.glob("**/*"):
            # glob("**/*") iterates over all files and directories -> skip directories
            if file_path.is_dir():
                continue

            if not file_path.is_symlink():
                raise CorruptedCacheException(
                    f"Revision folder corrupted. Found a non-symlink file: {file_path}"
                )

            blob_path = Path(file_path).resolve()
            if not blob_path.exists():
                raise CorruptedCacheException(
                    f"Blob missing (broken symlink): {blob_path}"
                )

            if blobs_path not in blob_path.parents:
                raise CorruptedCacheException(
                    f"Blob symlink points outside of blob directory: {blob_path}"
                )

            if blob_path not in blob_sizes:
                blob_sizes[blob_path] = blob_path.stat().st_size

            cached_files.add(
                CachedFileInfo(
                    file_name=file_path.name,
                    file_path=file_path,
                    size_on_disk=blob_sizes[blob_path],
                    blob_path=blob_path,
                )
            )

        cached_revisions.add(
            CachedRevisionInfo(
                commit_hash=revision_path.name,
                size_on_disk=sum(
                    blob_sizes[blob_path]
                    for blob_path in set(file.blob_path for file in cached_files)
                ),
                files=frozenset(cached_files),
                snapshot_path=revision_path,
                refs=frozenset(refs_by_hash.pop(revision_path.name, set())),
            )
        )

    # Check that all refs referred to an existing revision
    if len(refs_by_hash) > 0:
        raise CorruptedCacheException(
            "Reference(s) refer to missing commit hashes:"
            f" {refs_by_hash} ({repo_path})."
        )

    # Build and return frozen structure
    return CachedRepoInfo(
        repo_id=repo_id,
        repo_type=repo_type,  # type: ignore
        revisions=frozenset(cached_revisions),
        repo_path=repo_path,
        size_on_disk=sum(blob_sizes.values()),
        nb_files=len(blob_sizes),
    )


def _format_size(num: int) -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    """
    num_f = float(num)
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_f) < 1000.0:
            return f"{num_f:3.1f}{unit}"
        num_f /= 1000.0
    return f"{num_f:.1f}Y"
