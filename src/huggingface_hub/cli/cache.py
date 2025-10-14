# coding=utf-8
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
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Optional

import typer

from ..utils import CachedRepoInfo, CachedRevisionInfo, CacheNotFound, DeleteCacheStrategy, HFCacheInfo, scan_cache_dir
from ._cli_utils import (
    build_cache_index,
    collect_cache_entries,
    compile_cache_filter,
    format_cache_repo_id,
    print_cache_entries_csv,
    print_cache_entries_json,
    print_cache_entries_table,
    print_cache_selected_revisions,
    summarize_cache_deletion_counts,
    typer_factory,
)


cache_cli = typer_factory(help="Manage local cache directory.")


class OutputFormat(str, Enum):
    table = "table"
    json = "json"
    csv = "csv"


@dataclass(frozen=True)
class _DeletionResolution:
    revisions: frozenset[str]
    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]]
    missing: tuple[str, ...]


@cache_cli.command(help="List cached repositories or revisions.")
def ls(
    cache_dir: Annotated[
        Optional[str],
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
        Optional[list[str]],
        typer.Option(
            "-f",
            "--filter",
            help="Filter entries (e.g. size>1GB, type=model, accessed>7d). Can be used multiple times.",
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(
            help="Output format.",
        ),
    ] = OutputFormat.table,
    quiet: Annotated[
        bool,
        typer.Option(
            "-q",
            "--quiet",
            help="Print only IDs (repo IDs or revision hashes).",
        ),
    ] = False,
) -> None:
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        print(f"Cache directory not found: {str(exc.cache_dir)}")
        raise typer.Exit(code=1)

    filters = filter or []

    entries, repo_refs_map = collect_cache_entries(hf_cache_info, include_revisions=revisions)
    try:
        filter_fns = [compile_cache_filter(expr, repo_refs_map) for expr in filters]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    now = time.time()
    for fn in filter_fns:
        entries = [entry for entry in entries if fn(entry[0], entry[1], now)]

    if quiet:
        for repo, revision in entries:
            print(revision.commit_hash if revision is not None else format_cache_repo_id(repo))
        return

    formatters = {
        OutputFormat.table: print_cache_entries_table,
        OutputFormat.json: print_cache_entries_json,
        OutputFormat.csv: print_cache_entries_csv,
    }
    return formatters[format](entries, include_revisions=revisions, repo_refs_map=repo_refs_map)


@cache_cli.command(help="Remove cached repositories or revisions.")
def rm(
    targets: Annotated[
        list[str],
        typer.Argument(
            help="One or more repo IDs (e.g. model/bert-base-uncased) or revision hashes to delete.",
        ),
    ],
    cache_dir: Annotated[
        Optional[str],
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
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        print(f"Cache directory not found: {str(exc.cache_dir)}")
        raise typer.Exit(code=1)

    resolution = _resolve_deletion_targets(hf_cache_info, targets)

    if resolution.missing:
        print("Could not find the following targets in the cache:")
        for entry in resolution.missing:
            print(f"  - {entry}")

    if len(resolution.revisions) == 0:
        print("Nothing to delete.")
        raise typer.Exit(code=0)

    strategy = hf_cache_info.delete_revisions(*sorted(resolution.revisions))
    _print_deletion_summary(resolution, strategy)

    if dry_run:
        print("Dry run: no files were deleted.")
        return

    if not yes and not typer.confirm("Proceed with deletion?", default=False):
        print("Deletion cancelled.")
        return

    strategy.execute()
    counts = summarize_cache_deletion_counts(resolution.selected)
    print(
        f"Deleted {counts.repo_count} repo(s) and {counts.total_revision_count} revision(s); freed {strategy.expected_freed_size_str}."
    )


@cache_cli.command(help="Remove detached revisions from the cache.")
def prune(
    cache_dir: Annotated[
        Optional[str],
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
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        print(f"Cache directory not found: {str(exc.cache_dir)}")
        raise typer.Exit(code=1)

    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]] = {}
    revisions: set[str] = set()
    for repo in hf_cache_info.repos:
        detached = frozenset(revision for revision in repo.revisions if len(revision.refs) == 0)
        if not detached:
            continue
        selected[repo] = detached
        revisions.update(revision.commit_hash for revision in detached)

    if len(revisions) == 0:
        print("No unreferenced revisions found. Nothing to prune.")
        return

    resolution = _DeletionResolution(
        revisions=frozenset(revisions),
        selected=selected,
        missing=(),
    )
    strategy = hf_cache_info.delete_revisions(*sorted(resolution.revisions))
    counts = summarize_cache_deletion_counts(selected)

    print(
        f"About to delete {counts.total_revision_count} unreferenced revision(s) ({strategy.expected_freed_size_str} total)."
    )
    print_cache_selected_revisions(selected)

    if dry_run:
        print("Dry run: no files were deleted.")
        return

    if not yes and not typer.confirm("Proceed?", default=False):
        print("Pruning cancelled.")
        return

    strategy.execute()
    print(f"Deleted {counts.total_revision_count} unreferenced revision(s); freed {strategy.expected_freed_size_str}.")


def _resolve_deletion_targets(hf_cache_info: HFCacheInfo, targets: list[str]) -> _DeletionResolution:
    repo_lookup, revision_lookup = build_cache_index(hf_cache_info)

    selected: dict[CachedRepoInfo, set[CachedRevisionInfo]] = defaultdict(set)
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

        matched_repo = repo_lookup.get(lowered)
        if matched_repo is None:
            missing.append(raw_target)
            continue

        for revision in matched_repo.revisions:
            selected[matched_repo].add(revision)
            revisions.add(revision.commit_hash)

    frozen_selected = {repo: frozenset(revs) for repo, revs in selected.items()}
    return _DeletionResolution(
        revisions=frozenset(revisions),
        selected=frozen_selected,
        missing=tuple(missing),
    )


def _print_deletion_summary(resolution: _DeletionResolution, strategy: DeleteCacheStrategy) -> None:
    selected_by_repo = resolution.selected
    counts = summarize_cache_deletion_counts(selected_by_repo)

    summary_parts: list[str] = []
    if counts.repo_count:
        summary_parts.append(f"{counts.repo_count} repo(s)")
    if counts.partial_revision_count:
        summary_parts.append(f"{counts.partial_revision_count} revision(s)")
    if not summary_parts:
        summary_parts.append(f"{counts.total_revision_count} revision(s)")

    summary_text = " and ".join(summary_parts)
    print(f"About to delete {summary_text} totalling {strategy.expected_freed_size_str}.")
    print_cache_selected_revisions(selected_by_repo)
