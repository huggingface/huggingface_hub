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
from typing import Dict, List, Optional, Set, Union

from ..constants import HUGGINGFACE_HUB_CACHE
from ._typing import Literal


REPO_TYPE_T = Literal["model", "dataset", "space"]


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the huggingface cache."""


@dataclass
class CachedFileInfo:
    """Information about a single file.

    The file path is a symlink existing in the `snapshots` folder and referring to a
    blob in the `blobs` folder.
    """

    file_name: str
    file_path: Path
    size_on_disk: int
    blob_path: Path

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)


@dataclass
class CachedRevisionInfo:
    r"""Information about a revision (a snapshot).

    A revision can be either referenced by 1 or more `refs` or be "detached" (no refs).

    /!\ `size_on_disk` is not necesarily the sum of all file sizes because of possible
        duplicated files. Also only blobs are taken into account, not the (neglectable)
        size of folders and symlinks.
    """

    commit_hash: str
    size_on_disk: int
    files: List[CachedFileInfo]  # sorted by name
    snapshot_path: Path
    refs: Optional[Set[str]] = None

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        return len(self.files)


@dataclass
class CachedRepoInfo:
    r"""Information about a cached repository (dataset or model).

    /!\ `size_on_disk` is not necesarily the sum of all revisions sizes because of
        duplicated files. Also only blobs are taken into account, not the (neglectable)
        size of folders and symlinks.
    """

    repo_id: str
    repo_type: REPO_TYPE_T
    repo_path: Path
    revisions: Set[CachedRevisionInfo]
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


@dataclass
class HFCacheInfo:
    r"""Information about the entire cache repository.

    /!\ Here `size_on_disk` is equal to the sum of all repo sizes (only blobs).
    """

    repos: List[CachedRepoInfo]
    size_on_disk: int
    errors: List[CorruptedCacheException]

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)


def scan_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> HFCacheInfo:
    """Scan the entire cache directory and return information about it."""
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    repos: List[CachedRepoInfo] = []
    errors: List[CorruptedCacheException] = []
    for repo_path in Path(cache_dir).iterdir():
        try:
            if repo_path.is_dir():
                repo_info = _scan_cached_repo(repo_path)
                repos.append(repo_info)
        except CorruptedCacheException as e:
            errors.append(e)

    return HFCacheInfo(
        repos=sorted(repos, key=lambda repo: repo.repo_path),
        size_on_disk=sum(repo.size_on_disk for repo in repos),
        errors=errors,
    )


def _scan_cached_repo(repo_path: Path) -> Optional[CachedRepoInfo]:
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
    if repo_type not in {"datasets", "models", "spaces"}:
        raise CorruptedCacheException(
            f"Repo type must be `dataset`, `model` or `space`, found `{repo_type}`"
            f" ({repo_path})."
        )
    repo_id = repo_id.replace("--", "/")  # google/fleurs -> "google/fleurs"
    repo_type = repo_type[:-1]  # "models" -> "model"

    blob_sizes: Dict[Path, int] = {}  # Key is blob_path, value is blob size (in bytes)

    snapshots_path = repo_path / "snapshots"
    refs_path = repo_path / "refs"
    blobs_path = repo_path / "blobs"

    if not snapshots_path.exists() or not snapshots_path.is_dir():
        raise CorruptedCacheException(
            f"Snapshots dir doesn't exist in cached repo: {snapshots_path}"
        )

    # Scan snapshots directory
    cached_revisions_by_hash: Dict[str, CachedRevisionInfo] = {}
    for revision_path in snapshots_path.iterdir():
        if revision_path.is_file():
            raise CorruptedCacheException(
                f"Snapshots folder corrupted. Found a file: {revision_path}"
            )

        cached_files = []
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

            cached_files.append(
                CachedFileInfo(
                    file_name=file_path.name,
                    file_path=file_path,
                    size_on_disk=blob_sizes[blob_path],
                    blob_path=blob_path,
                )
            )

        cached_revisions_by_hash[revision_path.name] = CachedRevisionInfo(
            commit_hash=revision_path.name,
            size_on_disk=sum(
                blob_sizes[blob_path]
                for blob_path in set(file.blob_path for file in cached_files)
            ),
            files=sorted(cached_files, key=lambda file: file.file_path),
            snapshot_path=revision_path,
        )

    # Scan over `refs` directory
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

            if commit_hash not in cached_revisions_by_hash:
                raise CorruptedCacheException(
                    f"Reference refers to a missing commit hash: {commit_hash}."
                    f" Existing hashes: {', '.join(cached_revisions_by_hash.keys())}"
                )

            cached_revision = cached_revisions_by_hash[commit_hash]
            if cached_revision.refs is None:
                cached_revision.refs = {ref_name}
            else:
                cached_revision.refs.add(ref_name)

    return CachedRepoInfo(
        repo_id=repo_id,
        repo_type=repo_type,
        revisions=sorted(
            cached_revisions_by_hash.values(), key=lambda revision: revision.commit_hash
        ),
        repo_path=repo_path,
        size_on_disk=sum(blob_sizes.values()),
        nb_files=len(blob_sizes),
    )


def _format_size(num: int) -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}Y"
