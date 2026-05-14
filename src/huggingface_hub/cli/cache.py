# Copyright 2025-present, the HuggingFace Inc. team.
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
"""Contains the 'hf cache' command group with cache management subcommands."""

import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer

from huggingface_hub.errors import CLIError

from ..utils import (
    ANSI,
    CachedFileInfo,
    CachedRepoInfo,
    CachedRevisionInfo,
    CacheNotFound,
    DeleteCacheStrategy,
    HFCacheInfo,
    _format_size,
    scan_cache_dir,
)
from ..utils._parsing import parse_duration, parse_size
from ._cli_utils import RepoIdArg, RepoTypeOpt, RevisionOpt, TokenOpt, get_hf_api, typer_factory
from ._output import out


cache_cli = typer_factory(help="Manage local cache directory.")


#### Cache helper utilities


@dataclass(frozen=True)
class _DeletionResolution:
    revisions: frozenset[str]
    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]]
    selected_files: dict[CachedRepoInfo, dict[CachedRevisionInfo, frozenset[CachedFileInfo]]]
    missing: tuple[str, ...]


_FILTER_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)\s*(?P<op>==|!=|>=|<=|>|<|=)\s*(?P<value>.+)$")
_ALLOWED_OPERATORS = {"=", "!=", ">", "<", ">=", "<="}
_FILTER_KEYS = {"accessed", "modified", "refs", "size", "type"}
_SORT_KEYS = {"accessed", "modified", "name", "size"}
_SORT_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)(?::(?P<order>asc|desc))?$")
_SORT_DEFAULT_ORDER = {
    # Default ordering: accessed/modified/size are descending (newest/biggest first), name is ascending
    "accessed": "desc",
    "modified": "desc",
    "size": "desc",
    "name": "asc",
}


# Dynamically generate SortOptions enum from _SORT_KEYS
_sort_options_dict = {}
for key in sorted(_SORT_KEYS):
    _sort_options_dict[key] = key
    _sort_options_dict[f"{key}_asc"] = f"{key}:asc"
    _sort_options_dict[f"{key}_desc"] = f"{key}:desc"

SortOptions = Enum("SortOptions", _sort_options_dict, type=str, module=__name__)  # type: ignore


@dataclass(frozen=True)
class CacheDeletionCounts:
    """Simple counters summarizing cache deletions for CLI messaging."""

    repo_count: int
    partial_revision_count: int
    total_revision_count: int
    file_count: int = 0


CacheEntry = tuple[CachedRepoInfo, CachedRevisionInfo | None]
RepoRefsMap = dict[CachedRepoInfo, frozenset[str]]


def summarize_deletions(
    selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]],
    *,
    selected_files: Mapping[CachedRepoInfo, Mapping[CachedRevisionInfo, frozenset[CachedFileInfo]]] | None = None,
) -> CacheDeletionCounts:
    """Summarize deletions across repositories."""
    repo_count = 0
    total_revisions = 0
    revisions_in_full_repos = 0

    for repo, revisions in selected_by_repo.items():
        total_revisions += len(revisions)
        if len(revisions) == len(repo.revisions):
            repo_count += 1
            revisions_in_full_repos += len(revisions)

    partial_revision_count = total_revisions - revisions_in_full_repos
    file_count = 0
    if selected_files is not None:
        file_count = sum(len(files) for revisions in selected_files.values() for files in revisions.values())

    return CacheDeletionCounts(repo_count, partial_revision_count, total_revisions, file_count=file_count)


def print_cache_selected_revisions(
    selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]],
    selected_files: Mapping[CachedRepoInfo, Mapping[CachedRevisionInfo, frozenset[CachedFileInfo]]] | None = None,
) -> None:
    """Pretty-print selected cache revisions during confirmation prompts."""
    selected_files = selected_files or {}
    repos = sorted(
        set(selected_by_repo) | set(selected_files), key=lambda repo: (repo.repo_type, repo.repo_id.lower())
    )
    for repo in repos:
        repo_key = f"{repo.repo_type}/{repo.repo_id}"
        revisions = sorted(selected_by_repo.get(repo, frozenset()), key=lambda rev: rev.commit_hash)
        revision_files = selected_files.get(repo, {})

        if len(revisions) == len(repo.revisions) and not revision_files:
            out.text(f"  - {repo_key} (entire repo)")
            continue

        out.text(f"  - {repo_key}:")
        for revision in revisions:
            refs = " ".join(sorted(revision.refs)) or "(detached)"
            if revision not in revision_files:
                out.text(f"      {revision.commit_hash} [{refs}] {revision.size_on_disk_str}")
                continue

            out.text(f"      {revision.commit_hash} [{refs}]")
            for file in sorted(
                revision_files[revision],
                key=lambda file: file.file_path.relative_to(revision.snapshot_path).as_posix(),
            ):
                relative_path = file.file_path.relative_to(revision.snapshot_path).as_posix()
                out.text(f"        {relative_path} {file.size_on_disk_str}")

        for revision in sorted(
            (rev for rev in revision_files.keys() if rev not in revisions), key=lambda rev: rev.commit_hash
        ):
            refs = " ".join(sorted(revision.refs)) or "(detached)"
            out.text(f"      {revision.commit_hash} [{refs}]")
            for file in sorted(
                revision_files[revision],
                key=lambda file: file.file_path.relative_to(revision.snapshot_path).as_posix(),
            ):
                relative_path = file.file_path.relative_to(revision.snapshot_path).as_posix()
                out.text(f"        {relative_path} {file.size_on_disk_str}")


def build_cache_index(
    hf_cache_info: HFCacheInfo,
) -> tuple[
    dict[str, CachedRepoInfo],
    dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]],
]:
    """Create lookup tables so CLI commands can resolve repo ids and revisions quickly."""
    repo_lookup: dict[str, CachedRepoInfo] = {}
    revision_lookup: dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]] = {}
    for repo in hf_cache_info.repos:
        repo_key = repo.cache_id.lower()
        repo_lookup[repo_key] = repo
        for revision in repo.revisions:
            revision_lookup[revision.commit_hash.lower()] = (repo, revision)
    return repo_lookup, revision_lookup


def _normalize_cache_id(repo_target: str) -> str:
    target = repo_target.strip().strip("/")
    target = re.sub(r"^https?://huggingface\.co/", "", target).strip("/")
    target = target.removeprefix("hf://").strip("/")

    parts = target.split("/")
    if len(parts) == 1:
        repo_type = "model"
        namespace = None
        repo_id = parts[0]
    elif parts[0] in {"model", "models", "dataset", "datasets", "space", "spaces", "kernel", "kernels"}:
        repo_type = {
            "model": "model",
            "models": "model",
            "dataset": "dataset",
            "datasets": "dataset",
            "space": "space",
            "spaces": "space",
            "kernel": "kernel",
            "kernels": "kernel",
        }[parts[0]]
        remaining = parts[1:]
        if len(remaining) == 1:
            namespace = None
            repo_id = remaining[0]
        else:
            namespace = remaining[0]
            repo_id = remaining[1]
    else:
        repo_type = "model"
        namespace = parts[0]
        repo_id = parts[1]

    if namespace is None:
        return f"{repo_type}/{repo_id}"
    return f"{repo_type}/{namespace}/{repo_id}"


def _resolve_file_targets(
    repo: CachedRepoInfo,
    path_target: str,
) -> dict[CachedRevisionInfo, frozenset[CachedFileInfo]]:
    selected_files: dict[CachedRevisionInfo, set[CachedFileInfo]] = defaultdict(set)
    normalized_path_target = path_target.strip("/")

    for revision in repo.revisions:
        for file in revision.files:
            relative_path = file.file_path.relative_to(revision.snapshot_path).as_posix()
            if "/" in normalized_path_target:
                matches = relative_path == normalized_path_target
            else:
                matches = file.file_name == normalized_path_target or relative_path.endswith(
                    f"/{normalized_path_target}"
                )
            if matches:
                selected_files[revision].add(file)

    return {revision: frozenset(files) for revision, files in selected_files.items()}


def collect_cache_entries(
    hf_cache_info: HFCacheInfo, *, include_revisions: bool
) -> tuple[list[CacheEntry], RepoRefsMap]:
    """Flatten cache metadata into rows consumed by `hf cache ls`."""
    entries: list[CacheEntry] = []
    repo_refs_map: RepoRefsMap = {}
    sorted_repos = sorted(hf_cache_info.repos, key=lambda repo: (repo.repo_type, repo.repo_id.lower()))
    for repo in sorted_repos:
        repo_refs_map[repo] = frozenset({ref for revision in repo.revisions for ref in revision.refs})
        if include_revisions:
            for revision in sorted(repo.revisions, key=lambda rev: rev.commit_hash):
                entries.append((repo, revision))
        else:
            entries.append((repo, None))
    if include_revisions:
        entries.sort(
            key=lambda entry: (
                entry[0].cache_id,
                entry[1].commit_hash if entry[1] is not None else "",
            )
        )
    else:
        entries.sort(key=lambda entry: entry[0].cache_id)
    return entries, repo_refs_map


def compile_cache_filter(
    expr: str, repo_refs_map: RepoRefsMap
) -> Callable[[CachedRepoInfo, CachedRevisionInfo | None, float], bool]:
    """Convert a `hf cache ls` filter expression into the yes/no test we apply to each cache entry before displaying it."""
    match = _FILTER_PATTERN.match(expr.strip())
    if not match:
        raise ValueError(f"Invalid filter expression: '{expr}'.")

    key = match.group("key").lower()
    op = match.group("op")
    value_raw = match.group("value").strip()

    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Unsupported operator '{op}' in filter '{expr}'. Must be one of {list(_ALLOWED_OPERATORS)}.")

    if key not in _FILTER_KEYS:
        raise ValueError(f"Unsupported filter key '{key}' in '{expr}'. Must be one of {list(_FILTER_KEYS)}.")
    # at this point we know that key is in `_FILTER_KEYS`
    if key == "size":
        size_threshold = parse_size(value_raw)
        return lambda repo, revision, _: _compare_numeric(
            revision.size_on_disk if revision is not None else repo.size_on_disk,
            op,
            size_threshold,
        )

    if key in {"modified", "accessed"}:
        seconds = parse_duration(value_raw.strip())

        def _time_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, now: float) -> bool:
            timestamp = (
                repo.last_accessed
                if key == "accessed"
                else revision.last_modified
                if revision is not None
                else repo.last_modified
            )
            if timestamp is None:
                return False
            return _compare_numeric(now - timestamp, op, seconds)

        return _time_filter

    if key == "type":
        expected = value_raw.lower()

        if op != "=":
            raise ValueError(f"Only '=' is supported for 'type' filters. Got '{op}'.")

        def _type_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, _: float) -> bool:
            return repo.repo_type.lower() == expected

        return _type_filter

    else:  # key == "refs"
        if op != "=":
            raise ValueError(f"Only '=' is supported for 'refs' filters. Got {op}.")

        def _refs_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, _: float) -> bool:
            refs = revision.refs if revision is not None else repo_refs_map.get(repo, frozenset())
            return value_raw.lower() in [ref.lower() for ref in refs]

        return _refs_filter


def _compare_numeric(left: float | None, op: str, right: float) -> bool:
    """Evaluate numeric comparisons for filters."""
    if left is None:
        return False

    comparisons = {
        "=": left == right,
        "!=": left != right,
        ">": left > right,
        "<": left < right,
        ">=": left >= right,
        "<=": left <= right,
    }

    if op not in comparisons:
        raise ValueError(f"Unsupported numeric comparison operator: {op}")

    return comparisons[op]


def compile_cache_sort(sort_expr: str) -> tuple[Callable[[CacheEntry], tuple[Any, ...]], bool]:
    """Convert a `hf cache ls` sort expression into a key function for sorting entries.

    Returns:
        A tuple of (key_function, reverse_flag) where reverse_flag indicates whether
        to sort in descending order (True) or ascending order (False).
    """
    match = _SORT_PATTERN.match(sort_expr.strip().lower())
    if not match:
        raise ValueError(f"Invalid sort expression: '{sort_expr}'. Expected format: 'key' or 'key:asc' or 'key:desc'.")

    key = match.group("key").lower()
    explicit_order = match.group("order")

    if key not in _SORT_KEYS:
        raise ValueError(f"Unsupported sort key '{key}' in '{sort_expr}'. Must be one of {list(_SORT_KEYS)}.")

    # Use explicit order if provided, otherwise use default for the key
    order = explicit_order if explicit_order else _SORT_DEFAULT_ORDER[key]
    reverse = order == "desc"

    def _sort_key(entry: CacheEntry) -> tuple[Any, ...]:
        repo, revision = entry

        if key == "name":
            # Sort by cache_id (repo type/id)
            value: Any = repo.cache_id.lower()
            return (value,)

        if key == "size":
            # Use revision size if available, otherwise repo size
            value = revision.size_on_disk if revision is not None else repo.size_on_disk
            return (value,)

        if key == "accessed":
            # For revisions, accessed is not available per-revision, use repo's last_accessed
            # For repos, use repo's last_accessed
            value = repo.last_accessed if repo.last_accessed is not None else 0.0
            return (value,)

        if key == "modified":
            # Use revision's last_modified if available, otherwise repo's last_modified
            if revision is not None:
                value = revision.last_modified if revision.last_modified is not None else 0.0
            else:
                value = repo.last_modified if repo.last_modified is not None else 0.0
            return (value,)

        # Should never reach here due to validation above
        raise ValueError(f"Unsupported sort key: {key}")

    return _sort_key, reverse


def _resolve_deletion_targets(hf_cache_info: HFCacheInfo, targets: list[str]) -> _DeletionResolution:
    """Resolve the deletion targets into a deletion resolution."""
    repo_lookup, revision_lookup = build_cache_index(hf_cache_info)

    selected: dict[CachedRepoInfo, set[CachedRevisionInfo]] = defaultdict(set)
    selected_files: dict[CachedRepoInfo, dict[CachedRevisionInfo, set[CachedFileInfo]]] = defaultdict(
        lambda: defaultdict(set)
    )
    revisions: set[str] = set()
    missing: list[str] = []

    for raw_target in targets:
        target = raw_target.strip()
        if not target:
            continue
        lowered = target.lower()

        if re.fullmatch(r"[0-9a-fA-F]{40}", lowered):
            match = revision_lookup.get(lowered)
            if match is None:
                missing.append(raw_target)
                continue
            repo, revision = match
            selected[repo].add(revision)
            revisions.add(revision.commit_hash)
            continue

        if ":" in target:
            repo_target, path_target = target.split(":", 1)
            matched_repo = repo_lookup.get(_normalize_cache_id(repo_target).lower())
            if matched_repo is None:
                missing.append(raw_target)
                continue

            matching_files = _resolve_file_targets(matched_repo, path_target)
            if not matching_files:
                missing.append(raw_target)
                continue

            for revision, files in matching_files.items():
                selected_files[matched_repo][revision].update(files)
            continue

        matched_repo = repo_lookup.get(_normalize_cache_id(target).lower())
        if matched_repo is None:
            missing.append(raw_target)
            continue

        for revision in matched_repo.revisions:
            selected[matched_repo].add(revision)
            revisions.add(revision.commit_hash)

    # Promote file deletions that fully remove a revision into revision deletions.
    for repo, revision_to_files in list(selected_files.items()):
        for revision, files in list(revision_to_files.items()):
            if len(files) == len(revision.files):
                selected[repo].add(revision)
                revisions.add(revision.commit_hash)
                del revision_to_files[revision]

        if len(revision_to_files) == 0:
            del selected_files[repo]

    # Promote repos whose selected revisions cover all revisions.
    for repo, revisions_to_delete in list(selected.items()):
        revisions.update(revision.commit_hash for revision in revisions_to_delete)
        if len(revisions_to_delete) == len(repo.revisions):
            selected_files.pop(repo, None)

    frozen_selected = {repo: frozenset(revs) for repo, revs in selected.items()}
    frozen_selected_files = {
        repo: {revision: frozenset(files) for revision, files in revision_to_files.items()}
        for repo, revision_to_files in selected_files.items()
    }
    return _DeletionResolution(
        revisions=frozenset(revisions),
        selected=frozen_selected,
        selected_files=frozen_selected_files,
        missing=tuple(missing),
    )


def _build_delete_strategy(
    hf_cache_info: HFCacheInfo, resolution: _DeletionResolution
) -> tuple[DeleteCacheStrategy, CacheDeletionCounts]:
    blobs_to_delete: set[Path] = set()
    blob_sizes: dict[Path, int] = {}
    files_to_delete: set[Path] = set()
    refs_to_delete: set[Path] = set()
    repos_to_delete: set[Path] = set()
    snapshots_to_delete: set[Path] = set()

    repo_count = 0
    total_revision_count = 0
    partial_revision_count = 0
    file_count = 0

    for repo in hf_cache_info.repos:
        selected_revisions = resolution.selected.get(repo, frozenset())
        selected_files_by_revision = resolution.selected_files.get(repo, {})

        is_repo_fully_selected = len(selected_revisions) == len(repo.revisions) and not selected_files_by_revision
        if is_repo_fully_selected:
            repo_count += 1
            total_revision_count += len(repo.revisions)
            repos_to_delete.add(repo.repo_path)
            continue

        total_revision_count += len(selected_revisions)
        partial_revision_count += len(selected_files_by_revision)
        file_count += sum(len(files) for files in selected_files_by_revision.values())

        total_blob_refs: dict[Path, int] = defaultdict(int)
        selected_blob_refs: dict[Path, int] = defaultdict(int)

        for revision in repo.revisions:
            for file in revision.files:
                total_blob_refs[file.blob_path] += 1
                blob_sizes.setdefault(file.blob_path, file.size_on_disk)

        for revision in selected_revisions:
            snapshots_to_delete.add(revision.snapshot_path)
            refs_to_delete.update(repo.repo_path / "refs" / ref for ref in revision.refs)
            for file in revision.files:
                selected_blob_refs[file.blob_path] += 1

        for revision, files in selected_files_by_revision.items():
            for file in files:
                files_to_delete.add(file.file_path)
                selected_blob_refs[file.blob_path] += 1

        for blob_path, selected_count in selected_blob_refs.items():
            if selected_count == total_blob_refs[blob_path]:
                blobs_to_delete.add(blob_path)

    expected_freed_size = sum(blob_sizes[blob_path] for blob_path in blobs_to_delete) + sum(
        repo.size_on_disk for repo in hf_cache_info.repos if repo.repo_path in repos_to_delete
    )

    strategy = DeleteCacheStrategy(
        blobs=frozenset(blobs_to_delete),
        files=frozenset(files_to_delete),
        refs=frozenset(refs_to_delete),
        repos=frozenset(repos_to_delete),
        snapshots=frozenset(snapshots_to_delete),
        expected_freed_size=expected_freed_size,
    )
    counts = CacheDeletionCounts(
        repo_count=repo_count,
        partial_revision_count=partial_revision_count,
        total_revision_count=total_revision_count,
        file_count=file_count,
    )
    return strategy, counts


#### Cache CLI commands


@cache_cli.command(
    "list | ls",
    examples=[
        "hf cache ls",
        "hf cache ls --revisions",
        'hf cache ls --filter "size>1GB" --limit 20',
        "hf cache ls --format json",
    ],
)
def ls(
    cache_dir: Annotated[
        str | None,
        typer.Option(
            help="Cache directory to scan (defaults to Hugging Face cache).",
        ),
    ] = None,
    revisions: Annotated[
        bool,
        typer.Option(
            help="Include revisions in the output instead of aggregated repositories.",
        ),
    ] = False,
    filter: Annotated[
        list[str] | None,
        typer.Option(
            "-f",
            "--filter",
            help="Filter entries (e.g. 'size>1GB', 'type=model', 'accessed>7d'). Can be used multiple times.",
        ),
    ] = None,
    sort: Annotated[
        SortOptions | None,
        typer.Option(
            help="Sort entries by key. Supported keys: 'accessed', 'modified', 'name', 'size'. "
            "Append ':asc' or ':desc' to explicitly set the order (e.g., 'modified:asc'). "
            "Defaults: 'accessed', 'modified', 'size' default to 'desc' (newest/biggest first); "
            "'name' defaults to 'asc' (alphabetical).",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            help="Limit the number of results returned. Returns only the top N entries after sorting.",
        ),
    ] = None,
) -> None:
    """List cached repositories or revisions."""
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    filters = filter or []

    entries, repo_refs_map = collect_cache_entries(hf_cache_info, include_revisions=revisions)
    try:
        filter_fns = [compile_cache_filter(expr, repo_refs_map) for expr in filters]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    now = time.time()
    for fn in filter_fns:
        entries = [entry for entry in entries if fn(entry[0], entry[1], now)]

    # Apply sorting if requested
    if sort:
        try:
            sort_key_fn, reverse = compile_cache_sort(sort.value)
            entries.sort(key=sort_key_fn, reverse=reverse)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    # Apply limit if requested
    if limit is not None:
        if limit < 0:
            raise typer.BadParameter(f"Limit must be a positive integer, got {limit}.")
        entries = entries[:limit]

    if revisions:
        items = [
            {
                "id": repo.cache_id,
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "revision": revision.commit_hash,
                "snapshot_path": str(revision.snapshot_path),
                "size": revision.size_on_disk_str,
                "last_modified": revision.last_modified_str,
                "refs": sorted(revision.refs),
            }
            for repo, revision in entries
            if revision is not None
        ]
        out.table(
            items,
            headers=["id", "revision", "size", "last_modified", "refs"],
            id_key="revision",
            alignments={"size": "right"},
        )
    else:
        items = [
            {
                "id": repo.cache_id,
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size": repo.size_on_disk_str,
                "last_accessed": repo.last_accessed_str or "",
                "last_modified": repo.last_modified_str,
                "refs": sorted(repo_refs_map.get(repo, frozenset())),
            }
            for repo, _ in entries
        ]
        out.table(
            items,
            headers=["id", "size", "last_accessed", "last_modified", "refs"],
            id_key="id",
            alignments={"size": "right"},
        )

    if entries:
        unique_repos = {repo for repo, _ in entries}
        repo_count = len(unique_repos)
        if revisions:
            revision_count = sum(1 for _, rev in entries if rev is not None)
            total_size = sum(rev.size_on_disk for _, rev in entries if rev is not None)
        else:
            revision_count = sum(len(repo.revisions) for repo in unique_repos)
            total_size = sum(repo.size_on_disk for repo in unique_repos)
        out.text(
            ANSI.bold(
                f"\nFound {repo_count} repo(s) for a total of {revision_count} revision(s)"
                f" and {_format_size(total_size)} on disk."
            )
        )


@cache_cli.command(
    examples=[
        "hf cache rm model/gpt2",
        "hf cache rm <revision_hash>",
        "hf cache rm unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL",
        "hf cache rm model/gpt2 --dry-run",
        "hf cache rm model/gpt2 --yes",
    ],
)
def rm(
    targets: Annotated[
        list[str],
        typer.Argument(
            help="One or more repo IDs (e.g. model/bert-base-uncased), revision hashes, or repo:path targets to delete.",
        ),
    ],
    cache_dir: Annotated[
        str | None,
        typer.Option(
            help="Cache directory to scan (defaults to Hugging Face cache).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Preview deletions without removing anything.",
        ),
    ] = False,
) -> None:
    """Remove cached repositories, revisions, or files."""
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    resolution = _resolve_deletion_targets(hf_cache_info, targets)

    if resolution.missing:
        details = "\n".join(f"  - {entry}" for entry in resolution.missing)
        out.warning(f"Could not find in cache:\n{details}")

    if len(resolution.revisions) == 0 and not resolution.selected_files:
        out.text("Nothing to delete.")
        raise typer.Exit(code=0)

    strategy, counts = _build_delete_strategy(hf_cache_info, resolution)

    summary_parts: list[str] = []
    if counts.repo_count:
        summary_parts.append(f"{counts.repo_count} repo(s)")
    if counts.total_revision_count:
        summary_parts.append(f"{counts.total_revision_count} revision(s)")
    if counts.file_count:
        summary_parts.append(f"{counts.file_count} file(s)")
    if not summary_parts:
        summary_parts.append("Nothing")

    summary_text = " and ".join(summary_parts)
    out.text(f"About to delete {summary_text} totalling {strategy.expected_freed_size_str}.")
    print_cache_selected_revisions(resolution.selected, resolution.selected_files)

    if dry_run:
        out.result(
            "Dry run: no files were deleted.",
            dry_run=True,
            repos=counts.repo_count,
            revisions=counts.total_revision_count,
            files=counts.file_count,
            size=strategy.expected_freed_size_str,
        )
        return

    out.confirm("Proceed with deletion?", yes=yes)

    strategy.execute()
    out.result(
        f"Deleted {summary_text}; freed {strategy.expected_freed_size_str}.",
        repos_deleted=counts.repo_count,
        revisions_deleted=counts.total_revision_count,
        files_deleted=counts.file_count,
        freed=strategy.expected_freed_size_str,
    )


@cache_cli.command(examples=["hf cache prune", "hf cache prune --dry-run"])
def prune(
    cache_dir: Annotated[
        str | None,
        typer.Option(
            help="Cache directory to scan (defaults to Hugging Face cache).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Preview deletions without removing anything.",
        ),
    ] = False,
) -> None:
    """Remove detached revisions from the cache."""
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]] = {}
    revisions: set[str] = set()
    for repo in hf_cache_info.repos:
        detached = frozenset(revision for revision in repo.revisions if len(revision.refs) == 0)
        if not detached:
            continue
        selected[repo] = detached
        revisions.update(revision.commit_hash for revision in detached)

    if len(revisions) == 0:
        out.text("No unreferenced revisions found. Nothing to prune.")
        return

    resolution = _DeletionResolution(
        revisions=frozenset(revisions),
        selected=selected,
        missing=(),
    )
    strategy = hf_cache_info.delete_revisions(*sorted(resolution.revisions))
    counts = summarize_deletions(selected)

    out.text(
        f"About to delete {counts.total_revision_count} unreferenced revision(s) ({strategy.expected_freed_size_str} total)."
    )
    print_cache_selected_revisions(selected)

    if dry_run:
        out.result(
            "Dry run: no files were deleted.",
            dry_run=True,
            revisions=counts.total_revision_count,
            size=strategy.expected_freed_size_str,
        )
        return

    out.confirm("Proceed?", yes=yes)

    strategy.execute()
    out.result(
        f"Deleted {counts.total_revision_count} unreferenced revision(s); freed {strategy.expected_freed_size_str}.",
        revisions_deleted=counts.total_revision_count,
        freed=strategy.expected_freed_size_str,
    )


@cache_cli.command(
    examples=[
        "hf cache verify gpt2",
        "hf cache verify gpt2 --revision refs/pr/1",
        "hf cache verify my-dataset --repo-type dataset",
    ],
)
def verify(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoTypeOpt.model,
    revision: RevisionOpt = None,
    cache_dir: Annotated[
        str | None,
        typer.Option(
            help="Cache directory to use when verifying files from cache (defaults to Hugging Face cache).",
        ),
    ] = None,
    local_dir: Annotated[
        str | None,
        typer.Option(
            help="If set, verify files under this directory instead of the cache.",
        ),
    ] = None,
    fail_on_missing_files: Annotated[
        bool,
        typer.Option(
            "--fail-on-missing-files",
            help="Fail if some files exist on the remote but are missing locally.",
        ),
    ] = False,
    fail_on_extra_files: Annotated[
        bool,
        typer.Option(
            "--fail-on-extra-files",
            help="Fail if some files exist locally but are not present on the remote revision.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Verify checksums for a single repo revision from cache or a local directory.

    Examples:
      - Verify main revision in cache: `hf cache verify gpt2`
      - Verify specific revision: `hf cache verify gpt2 --revision refs/pr/1`
      - Verify dataset: `hf cache verify karpathy/fineweb-edu-100b-shuffle --repo-type dataset`
      - Verify local dir: `hf cache verify deepseek-ai/DeepSeek-OCR --local-dir /path/to/repo`
    """

    if local_dir is not None and cache_dir is not None:
        out.error("Cannot pass both --local-dir and --cache-dir. Use one or the other.")
        raise typer.Exit(code=2)

    api = get_hf_api(token=token)

    result = api.verify_repo_checksums(
        repo_id=repo_id,
        repo_type=repo_type.value if hasattr(repo_type, "value") else str(repo_type),
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
        token=token,
    )

    exit_code = 0

    if result.mismatches:
        details = "\n".join(
            f"  - {m['path']}: expected {m['expected']} ({m['algorithm']}), got {m['actual']}"
            for m in result.mismatches
        )
        out.text(f"❌ Checksum verification failed for the following file(s):\n{details}")
        exit_code = 1

    if result.missing_paths:
        if fail_on_missing_files:
            details = "\n".join(f"  - {p}" for p in result.missing_paths)
            out.text(f"❌ Missing files (present remotely, absent locally):\n{details}")
            exit_code = 1
        else:
            out.warning(
                f"{len(result.missing_paths)} remote file(s) are missing locally. "
                "Use --fail-on-missing-files for details."
            )

    if result.extra_paths:
        if fail_on_extra_files:
            details = "\n".join(f"  - {p}" for p in result.extra_paths)
            out.text(f"❌ Extra files (present locally, absent remotely):\n{details}")
            exit_code = 1
        else:
            out.warning(
                f"{len(result.extra_paths)} local file(s) do not exist on the remote repo. "
                "Use --fail-on-extra-files for details."
            )

    verified_location = result.verified_path

    if exit_code != 0:
        out.error(
            f"Verification failed for '{repo_id}' ({repo_type.value}) in {verified_location}.\n  Revision: {result.revision}"
        )
        raise typer.Exit(code=exit_code)

    out.result(
        f"Verified {result.checked_count} file(s) for {repo_type.value} '{repo_id}'. All checksums match.",
        repo_id=repo_id,
        repo_type=repo_type.value,
        checked=result.checked_count,
        path=str(verified_location),
    )
