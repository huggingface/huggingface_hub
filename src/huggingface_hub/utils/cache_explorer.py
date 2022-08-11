from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Union

from ..constants import HUGGINGFACE_HUB_CACHE


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
    """Information about a revision (a snapshot).

    A revision can be either referenced by 1 or more `refs` or be "detached" (no refs).
    
    /!\ `size_on_disk` is not necesarily the sum of all file sizes because of possible
        duplicated files. Also only blobs are taken into account, not the (neglectable)
        size of folders and symlinks.
    """    
    commit_hash: str
    size_on_disk: int
    files: List[CachedFileInfo]  # sorted by name
    refs: Optional[Set[str]] = None

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        return len(self.files)


@dataclass
class CachedRepoInfo:
    """Information about a cached repository (dataset or model).

    /!\ `size_on_disk` is not necesarily the sum of all revisions sizes because of
        duplicated files. Also only blobs are taken into account, not the (neglectable)
        size of folders and symlinks.
    """
    repo_id: str
    repo_type: Literal["model", "dataset"]
    repo_path: Path
    revisions: Set[CachedRevisionInfo]
    size_on_disk: int
    nb_files: int

    @property
    def size_on_disk_str(self) -> str:
        """Human-readable sizes"""
        return _format_size(self.size_on_disk)


@dataclass
class HFCacheInfo:
    """Information about the entire cache repository.
    
    /!\ Here `size_on_disk` is equal to the sum of all repo sizes (only blobs).
    """
    datasets: List[CachedRepoInfo]
    models: List[CachedRepoInfo]
    size_on_disk: int

    @property
    def size_on_disk_str(self) -> str:
        return _format_size(self.size_on_disk)


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the huggingface cache."""


def scan_cache_dir(
    cache_dir: Optional[Union["str", Path]] = None
) -> List[CachedRepoInfo]:
    """Scan the entire cache directory and return information about it."""
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    datasets: List[CachedRepoInfo] = []
    models: List[CachedRepoInfo] = []
    for repo_path in Path(cache_dir).iterdir():
        try:
            if repo_path.is_dir():
                repo_info = _scan_cached_repo(repo_path)
                if repo_info.repo_type == "dataset":
                    datasets.append(repo_info)
                elif repo_info.repo_type == "model":
                    models.append(repo_info)
                else:
                    raise ValueError()
        except CorruptedCacheException as e:
            print(f"Error while scanning {repo_path}: {e}")

    return HFCacheInfo(
        datasets=sorted(datasets, key=lambda repo: repo.repo_path),
        models=sorted(models, key=lambda repo: repo.repo_path),
        size_on_disk=sum(repo.size_on_disk for repo in datasets + models),
    )


def _scan_cached_repo(repo_path: Path) -> Optional[CachedRepoInfo]:
    """Scan a single cache repo and return information about it.
    
    Any unexpected behavior will raise a `CorruptedCacheException`.
    """
    if not repo_path.is_dir():
        raise CorruptedCacheException("Not a directory !")

    if "--" not in repo_path.name:
        raise CorruptedCacheException("Not a valid HuggingFace cache directory !")

    repo_type, repo_id = repo_path.name.split("--", maxsplit=1)
    if repo_type not in {"datasets", "models"}:
        raise CorruptedCacheException("Repo type must be `dataset` or `model` !")
    repo_type = repo_type[:-1]  # "models" -> "model"

    blob_sizes: Dict[Path, int] = {}  # Key is blob_path, value is blob size (in bytes)

    snapshots_path = repo_path / "snapshots"
    refs_path = repo_path / "refs"
    blobs_path = repo_path / "blobs"

    if not snapshots_path.exists() or not snapshots_path.is_dir():
        raise CorruptedCacheException("Snapshots dir doesn't exist !")

    # Scan snapshots directory
    cached_revisions_by_hash: Dict[str, CachedRevisionInfo] = {}
    for revision_path in snapshots_path.iterdir():
        if revision_path.is_file():
            raise CorruptedCacheException("Snapshots folder corrupted: found a file  !")

        cached_files = []
        for file_path in revision_path.glob("**/*"):
            # glob("**/*") iterates over all files and directories -> skip directories
            if file_path.is_dir():
                continue

            if not file_path.is_symlink():
                raise CorruptedCacheException(
                    "Revision folder corrupted: found a non-symlink file !"
                )

            blob_path = Path(file_path).resolve()
            if not blob_path.exists():
                raise CorruptedCacheException("Blob missing (broken symlink) !")

            if blobs_path not in blob_path.parents:
                raise CorruptedCacheException(
                    "Blob symlink points outside of blob directory !"
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
            raise CorruptedCacheException("Refs directory cannot be a file !")

        for ref_path in refs_path.glob("**/*"):
            # glob("**/*") iterates over all files and directories -> skip directories
            if ref_path.is_dir():
                continue

            ref_name = str(ref_path.relative_to(refs_path))
            with ref_path.open() as f:
                commit_hash = f.read()

            if commit_hash not in cached_revisions_by_hash:
                raise CorruptedCacheException(
                    "Reference refers to a missing commit hash !"
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


def _format_size(num: int, suffix: str = "B") -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
