import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

from .. import constants
from ..file_download import repo_folder_name
from .sha import git_hash, sha_fileobj


# using fullmatch for clarity and strictness
_REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


# Typed structure describing a checksum mismatch
class Mismatch(TypedDict):
    path: str
    expected: str
    actual: str
    algorithm: str


HashAlgo = Literal["sha256", "git-sha1"]


@dataclass(frozen=True)
class Verification:
    revision: str
    checked_count: int
    mismatches: list[Mismatch]
    missing_paths: list[str]
    extra_paths: list[str]

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.missing_paths or self.extra_paths)


def _collect_files_from_directory(root: Path) -> dict[str, Path]:
    """
    Return a mapping of repo-relative path -> absolute path for all files under `root`.
    """
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


def _resolve_commit_hash_from_cache(storage_folder: Path, revision: Optional[str]) -> str:
    """
    Resolve a commit hash from a cache repo folder and an optional revision.
    """
    if revision and _REGEX_COMMIT_HASH.fullmatch(revision):
        return revision

    refs_dir = storage_folder / "refs"
    snapshots_dir = storage_folder / "snapshots"

    if revision:
        ref_path = refs_dir / revision
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        raise ValueError(f"Revision '{revision}' could not be resolved in cache (expected file '{ref_path}').")

    # No revision provided: try common defaults
    main_ref = refs_dir / "main"
    if main_ref.is_file():
        return main_ref.read_text(encoding="utf-8").strip()

    if not snapshots_dir.is_dir():
        raise ValueError(f"Cache repo is missing snapshots directory: {snapshots_dir}. Provide --revision explicitly.")

    candidates = [p.name for p in snapshots_dir.iterdir() if p.is_dir() and _REGEX_COMMIT_HASH.fullmatch(p.name)]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Ambiguous cached revision: multiple snapshots found and no refs to disambiguate. Please pass --revision."
    )


def resolve_expected_hash(entry: object) -> tuple[HashAlgo, str]:
    """
    Return the algorithm and expected hash for a remote entry.
    Prefers LFS sha256 if available; falls back to git blob_id (sha1).
    """
    lfs = getattr(entry, "lfs", None)
    lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if lfs_sha is None and isinstance(lfs, dict):
        lfs_sha = lfs.get("sha256")

    if lfs_sha:
        return ("sha256", str(lfs_sha).lower())

    blob_id = getattr(entry, "blob_id", None)
    if blob_id:
        return ("git-sha1", str(blob_id).lower())

    raise ValueError("Remote entry missing checksum (no blob_id or lfs.sha256)")


def compute_file_hash(path: Path, algorithm: HashAlgo, *, git_hash_cache: dict[Path, str]) -> str:
    """
    Compute the checksum of a local file using the requested algorithm.
    """

    def _sha256(p: Path, cache: dict[Path, str]) -> str:
        with p.open("rb") as stream:
            return sha_fileobj(stream).hex()

    def _git_sha1(p: Path, cache: dict[Path, str]) -> str:
        try:
            return cache[p]
        except KeyError:
            with p.open("rb") as stream:
                data = stream.read()
            digest = git_hash(data)
            cache[p] = digest
            return digest

    HASHERS: dict[HashAlgo, Callable[[Path, dict[Path, str]], str]] = {
        "sha256": _sha256,
        "git-sha1": _git_sha1,
    }

    try:
        return HASHERS[algorithm](path, git_hash_cache)
    except KeyError:
        # Should be unreachable, but keeps type checker happy
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def verify_maps(*, remote_by_path: dict[str, object], local_by_path: dict[str, Path], revision: str) -> Verification:
    """Compare remote entries and local files and return a verification result."""
    remote_paths = set(remote_by_path)
    local_paths = set(local_by_path)

    missing = sorted(remote_paths - local_paths)
    extra = sorted(local_paths - remote_paths)
    both = sorted(remote_paths & local_paths)

    mismatches: list[Mismatch] = []
    git_hash_cache: dict[Path, str] = {}

    for rel_path in both:
        entry = remote_by_path[rel_path]
        local_path = local_by_path[rel_path]

        try:
            algorithm, expected = resolve_expected_hash(entry)
            actual = compute_file_hash(local_path, algorithm, git_hash_cache=git_hash_cache)
        except OSError as exc:
            mismatches.append(
                Mismatch(path=rel_path, expected="<unavailable>", actual=f"io-error:{exc}", algorithm="io")
            )
            continue
        except ValueError as exc:
            mismatches.append(
                Mismatch(path=rel_path, expected="<unavailable>", actual=f"meta-error:{exc}", algorithm="meta")
            )
            continue

        if actual != expected:
            mismatches.append(Mismatch(path=rel_path, expected=expected, actual=actual, algorithm=algorithm))

    return Verification(
        revision=revision,
        checked_count=len(both),
        mismatches=mismatches,
        missing_paths=missing,
        extra_paths=extra,
    )


def resolve_local_root(
    *,
    repo_id: str,
    repo_type: str,
    revision: Optional[str],
    cache_dir: Optional[Path],
    local_dir: Optional[Path],
) -> tuple[Path, str]:
    """
    Resolve the root directory to scan locally and the remote revision to verify.
    """
    if local_dir is not None:
        root = Path(local_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Local directory does not exist or is not a directory: {root}")
        return root, (revision or constants.DEFAULT_REVISION)

    cache_root = Path(cache_dir or constants.HF_HUB_CACHE).expanduser().resolve()
    storage_folder = cache_root / repo_folder_name(repo_id=repo_id, repo_type=repo_type)
    if not storage_folder.exists():
        raise ValueError(
            f"Repo is not present in cache: {storage_folder}. Use 'hf download' first or pass --local-dir."
        )
    commit = _resolve_commit_hash_from_cache(storage_folder, revision)
    snapshot_dir = storage_folder / "snapshots" / commit
    if not snapshot_dir.is_dir():
        raise ValueError(f"Snapshot directory does not exist for revision '{commit}': {snapshot_dir}.")
    return snapshot_dir, commit


def collect_local_files(root: Path) -> dict[str, Path]:
    """Collect all files under a root directory (either a cache snapshot or a regular folder)."""
    return _collect_files_from_directory(root)
