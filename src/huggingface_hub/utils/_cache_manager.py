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
    """Exception for any unexpected structure in the Huggingface cache-system."""


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
    """

    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Size of the blob file as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)


@dataclass(frozen=True)
class CachedRevisionInfo:
    """Frozen data structure holding information about a revision.

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
            Sum of the blob file sizes that are symlink-ed by the revision.
        files: (`FrozenSet[CachedFileInfo]`):
            Set of [`~CachedFileInfo`] describing all files contained in the snapshot.
        refs (`FrozenSet[str]`):
            Set of `refs` pointing to this revision. If the revision has no `refs`, it
            is considered detached.
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
        """
        (property) Sum of the blob file sizes as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        """
        (property) Total number of files in the revision.
        """
        return len(self.files)


@dataclass(frozen=True)
class CachedRepoInfo:
    """Frozen data structure holding information about a cached repository.

    Args:
        repo_id (`str`):
            Repo id of the repo on the Hub. Example: `"google/fleurs"`.
        repo_type (`Literal["dataset", "model", "space"]`):
            Type of the cached repo.
        repo_path (`Path`):
            Local path to the cached repo.
        size_on_disk (`int`):
            Sum of the blob file sizes in the cached repo.
        nb_files (`int`):
            Total number of blob files in the cached repo.
        revisions (`FrozenSet[CachedRevisionInfo]`):
            Set of [`~CachedRevisionInfo`] describing all revisions cached in the repo.

    <Tip warning={true}>

    `size_on_disk` is not necessarily the sum of all revisions sizes because of
    duplicated files. Besides, only blobs are taken into account, not the (negligible)
    size of folders and symlinks.

    </Tip>
    """

    repo_id: str
    repo_type: REPO_TYPE_T
    repo_path: Path
    size_on_disk: int
    nb_files: int
    revisions: FrozenSet[CachedRevisionInfo]

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of the blob file sizes as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)

    @property
    def refs(self) -> Dict[str, CachedRevisionInfo]:
        """
        (property) Mapping between `refs` and revision data structures.
        """
        return {ref: revision for revision in self.revisions for ref in revision.refs}


@dataclass(frozen=True)
class HFCacheInfo:
    """Frozen data structure holding information about the entire cache-system.

    This data structure is returned by [`scan_cache_dir`] and is immutable.

    Args:
        size_on_disk (`int`):
            Sum of all valid repo sizes in the cache-system.
        repos (`FrozenSet[CachedRepoInfo]`):
            Set of [`~CachedRepoInfo`] describing all valid cached repos found on the
            cache-system while scanning.
        errors (`List[CorruptedCacheException]`):
            List of [`~CorruptedCacheException`] that occurred while scanning the cache.
            Those exceptions are captured so that the scan can continue. Corrupted repos
            are skipped from the scan.

    <Tip warning={true}>

    Here `size_on_disk` is equal to the sum of all repo sizes (only blobs). However if
    some cached repos are corrupted, their sizes are not taken into account.

    </Tip>
    """

    size_on_disk: int
    repos: FrozenSet[CachedRepoInfo]
    errors: List[CorruptedCacheException]

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of all valid repo sizes in the cache-system as a human-readable
        string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)


def scan_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> HFCacheInfo:
    """Scan the entire HF cache-system and return a [`~HFCacheInfo`] structure.

    Use `scan_cache_dir` in order to programmatically scan your cache-system. The cache
    will be scanned repo by repo. If a repo is corrupted, a [`~CorruptedCacheException`]
    will be thrown internally but captured and returned in the [`~HFCacheInfo`]
    structure. Only valid repos get a proper report.

    ```py
    >>> from huggingface_hub import scan_cache_dir

    >>> hf_cache_info = scan_cache_dir()
    HFCacheInfo(
        size_on_disk=3398085269,
        repos=frozenset({
            CachedRepoInfo(
                repo_id='t5-small',
                repo_type='model',
                repo_path=PosixPath(...),
                size_on_disk=970726914,
                nb_files=11,
                revisions=frozenset({
                    CachedRevisionInfo(
                        commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                        size_on_disk=970726339,
                        snapshot_path=PosixPath(...),
                        files=frozenset({
                            CachedFileInfo(
                                file_name='config.json',
                                size_on_disk=1197
                                file_path=PosixPath(...),
                                blob_path=PosixPath(...),
                            ),
                            CachedFileInfo(...),
                            ...
                        }),
                    ),
                    CachedRevisionInfo(...),
                    ...
                }),
            ),
            CachedRepoInfo(...),
            ...
        }),
        errors=[
            CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
            CorruptedCacheException(...),
            ...
        ],
    )
    ```

    You can also print a detailed report directly from the `huggingface-cli` using:
    ```text
    > huggingface-cli scan-cache
    REPO ID                     REPO TYPE SIZE ON DISK NB FILES REFS                LOCAL PATH
    --------------------------- --------- ------------ -------- ------------------- -------------------------------------------------------------------------
    glue                        dataset         116.3K       15 1.17.0, main, 2.4.0 /Users/lucain/.cache/huggingface/hub/datasets--glue
    google/fleurs               dataset          64.9M        6 main, refs/pr/1     /Users/lucain/.cache/huggingface/hub/datasets--google--fleurs
    Jean-Baptiste/camembert-ner model           441.0M        7 main                /Users/lucain/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
    bert-base-cased             model             1.9G       13 main                /Users/lucain/.cache/huggingface/hub/models--bert-base-cased
    t5-base                     model            10.1K        3 main                /Users/lucain/.cache/huggingface/hub/models--t5-base
    t5-small                    model           970.7M       11 refs/pr/1, main     /Users/lucain/.cache/huggingface/hub/models--t5-small

    Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
    Got 1 error(s) while scanning. Use -vvv to print details.
    ```

    Args:
        cache_dir (`str` or `Path`, `optional`):
            Cache directory to cache. Defaults to the default HF cache directory.

    Returns: a [`~HFCacheInfo`] object.
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    repos: Set[CachedRepoInfo] = set()
    errors: List[CorruptedCacheException] = []
    for repo_path in Path(cache_dir).resolve().iterdir():
        try:
            repos.add(_scan_cached_repo(repo_path))
        except CorruptedCacheException as e:
            errors.append(e)

    return HFCacheInfo(
        repos=frozenset(repos),
        size_on_disk=sum(repo.size_on_disk for repo in repos),
        errors=errors,
    )


def _scan_cached_repo(repo_path: Path) -> CachedRepoInfo:
    """Scan a single cache repo and return information about it.

    Any unexpected behavior will raise a [`~CorruptedCacheException`].
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
