# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Contains CLI utilities (styling, helpers)."""

import csv
import importlib.metadata
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Dict, List, Mapping, Optional, Tuple

import click
import typer

from huggingface_hub import __version__, constants
from huggingface_hub.utils import (
    ANSI,
    CachedRepoInfo,
    CachedRevisionInfo,
    HFCacheInfo,
    get_session,
    hf_raise_for_status,
    installation_method,
    logging,
    tabulate,
)


logger = logging.get_logger()


if TYPE_CHECKING:
    from huggingface_hub.hf_api import HfApi


def get_hf_api(token: Optional[str] = None) -> "HfApi":
    # Import here to avoid circular import
    from huggingface_hub.hf_api import HfApi

    return HfApi(token=token, library_name="hf", library_version=__version__)


#### TYPER UTILS


class AlphabeticalMixedGroup(typer.core.TyperGroup):
    """
    Typer Group that lists commands and sub-apps mixed and alphabetically.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:  # type: ignore[name-defined]
        # click.Group stores both commands and sub-groups in `self.commands`
        return sorted(self.commands.keys())


def typer_factory(help: str) -> typer.Typer:
    return typer.Typer(
        help=help,
        add_completion=True,
        no_args_is_help=True,
        cls=AlphabeticalMixedGroup,
        # Disable rich completely for consistent experience
        rich_markup_mode=None,
        rich_help_panel=None,
        pretty_exceptions_enable=False,
    )


class RepoType(str, Enum):
    model = "model"
    dataset = "dataset"
    space = "space"


RepoIdArg = Annotated[
    str,
    typer.Argument(
        help="The ID of the repo (e.g. `username/repo-name`).",
    ),
]


RepoTypeOpt = Annotated[
    RepoType,
    typer.Option(
        help="The type of repository (model, dataset, or space).",
    ),
]

TokenOpt = Annotated[
    Optional[str],
    typer.Option(
        help="A User Access Token generated from https://huggingface.co/settings/tokens.",
    ),
]

PrivateOpt = Annotated[
    bool,
    typer.Option(
        help="Whether to create a private repo if repo doesn't exist on the Hub. Ignored if the repo already exists.",
    ),
]

RevisionOpt = Annotated[
    Optional[str],
    typer.Option(
        help="Git revision id which can be a branch name, a tag, or a commit hash.",
    ),
]


### PyPI VERSION CHECKER


def check_cli_update() -> None:
    """
    Check whether a newer version of `huggingface_hub` is available on PyPI.

    If a newer version is found, notify the user and suggest updating.
    If current version is a pre-release (e.g. `1.0.0.rc1`), or a dev version (e.g. `1.0.0.dev1`), no check is performed.

    This function is called at the entry point of the CLI. It only performs the check once every 24 hours, and any error
    during the check is caught and logged, to avoid breaking the CLI.
    """
    try:
        _check_cli_update()
    except Exception:
        # We don't want the CLI to fail on version checks, no matter the reason.
        logger.debug("Error while checking for CLI update.", exc_info=True)


def _check_cli_update() -> None:
    current_version = importlib.metadata.version("huggingface_hub")

    # Skip if current version is a pre-release or dev version
    if any(tag in current_version for tag in ["rc", "dev"]):
        return

    # Skip if already checked in the last 24 hours
    if os.path.exists(constants.CHECK_FOR_UPDATE_DONE_PATH):
        mtime = os.path.getmtime(constants.CHECK_FOR_UPDATE_DONE_PATH)
        if (time.time() - mtime) < 24 * 3600:
            return

    # Touch the file to mark that we did the check now
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).touch()

    # Check latest version from PyPI
    response = get_session().get("https://pypi.org/pypi/huggingface_hub/json", timeout=2)
    hf_raise_for_status(response)
    data = response.json()
    latest_version = data["info"]["version"]

    # If latest version is different from current, notify user
    if current_version != latest_version:
        method = installation_method()
        if method == "brew":
            update_command = "brew upgrade huggingface-cli"
        elif method == "hf_installer" and os.name == "nt":
            update_command = 'powershell -NoProfile -Command "iwr -useb https://hf.co/cli/install.ps1 | iex"'
        elif method == "hf_installer":
            update_command = "curl -LsSf https://hf.co/cli/install.sh | sh -"
        else:  # unknown => likely pip
            update_command = "pip install -U huggingface_hub"

        click.echo(
            ANSI.yellow(
                f"A new version of huggingface_hub ({latest_version}) is available! "
                f"You are using version {current_version}.\n"
                f"To update, run: {ANSI.bold(update_command)}\n",
            )
        )


def _ask_for_confirmation_no_tui(message: str, default: bool = True) -> bool:
    YES = ("y", "yes", "1")
    NO = ("n", "no", "0")
    DEFAULT = ""
    ALL = YES + NO + (DEFAULT,)
    full_message = message + (" (Y/n) " if default else " (y/N) ")
    while True:
        answer = input(full_message).lower()
        if answer == DEFAULT:
            return default
        if answer in YES:
            return True
        if answer in NO:
            return False
        print(f"Invalid input. Must be one of {ALL}")


#### hf cache utils

_FILTER_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)\s*(?P<op>==|!=|>=|<=|>|<|=)\s*(?P<value>.+)$")
_ALLOWED_OPERATORS = {"=", "!=", ">", "<", ">=", "<="}


class ByteUnit(IntEnum):
    BYTE = 1
    KB = 1_000
    MB = 1_000**2
    GB = 1_000**3
    TB = 1_000**4
    PB = 1_000**5

    @classmethod
    def suffixes(cls) -> Dict[str, int]:
        return {
            "b": cls.BYTE,
            "kb": cls.KB,
            "k": cls.KB,
            "mb": cls.MB,
            "m": cls.MB,
            "gb": cls.GB,
            "g": cls.GB,
            "tb": cls.TB,
            "t": cls.TB,
            "pb": cls.PB,
            "p": cls.PB,
        }


class TimeUnit(IntEnum):
    SECOND = 1
    MINUTE = 60
    HOUR = 60 * 60
    DAY = 24 * 60 * 60
    WEEK = 7 * 24 * 60 * 60
    MONTH = 30 * 24 * 60 * 60
    YEAR = 365 * 24 * 60 * 60

    @classmethod
    def suffixes(cls) -> Dict[str, int]:
        return {
            "s": cls.SECOND,
            "m": cls.MINUTE,
            "h": cls.HOUR,
            "d": cls.DAY,
            "w": cls.WEEK,
            "mo": cls.MONTH,
            "y": cls.YEAR,
        }


@dataclass(frozen=True)
class CacheDeletionCounts:
    repo_count: int
    partial_revision_count: int
    total_revision_count: int


CacheEntry = Tuple[CachedRepoInfo, Optional[CachedRevisionInfo]]
RepoRefsMap = Dict[CachedRepoInfo, frozenset[str]]


def parse_cache_size(value: str) -> int:
    stripped = value.strip()
    try:
        return int(float(stripped))
    except ValueError:
        pass

    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<suffix>[a-zA-Z]+)", stripped)
    if not match:
        raise ValueError(f"Invalid size value '{value}'.")

    number = float(match.group("number"))
    suffix = match.group("suffix").lower()

    multiplier = ByteUnit.suffixes().get(suffix)
    if multiplier is None:
        raise ValueError(f"Unknown size suffix '{match.group('suffix')}'.")

    if number < 0:
        raise ValueError(f"Size value cannot be negative: {value}")

    return int(number * multiplier)


def parse_cache_duration(value: str) -> float:
    stripped = value.strip().lower()
    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)(?P<suffix>s|m|h|d|w|mo|y)", stripped)
    if not match:
        raise ValueError(f"Invalid time value '{value}'.")

    number = float(match.group("number"))
    suffix = match.group("suffix")
    multiplier = TimeUnit.suffixes().get(suffix)
    if multiplier is None:
        raise ValueError(f"Unknown time suffix '{match.group('suffix')}'.")

    if number < 0:
        raise ValueError(f"Time value cannot be negative: {value}")

    return number * multiplier


def summarize_cache_deletion_counts(
    selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]],
) -> CacheDeletionCounts:
    repo_count = 0
    total_revisions = 0
    revisions_in_full_repos = 0

    for repo, revisions in selected_by_repo.items():
        total_revisions += len(revisions)
        if len(revisions) == len(repo.revisions):
            repo_count += 1
            revisions_in_full_repos += len(revisions)

    partial_revision_count = total_revisions - revisions_in_full_repos
    return CacheDeletionCounts(repo_count, partial_revision_count, total_revisions)


def print_cache_selected_revisions(selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]]) -> None:
    for repo in sorted(selected_by_repo.keys(), key=lambda repo: (repo.repo_type, repo.repo_id.lower())):
        repo_key = f"{repo.repo_type}/{repo.repo_id}"
        revisions = sorted(selected_by_repo[repo], key=lambda rev: rev.commit_hash)
        if len(revisions) == len(repo.revisions):
            print(f"  - {repo_key} (entire repo)")
            continue

        print(f"  - {repo_key}:")
        for revision in revisions:
            refs = ", ".join(sorted(revision.refs)) or "(detached)"
            print(f"      {revision.commit_hash} [{refs}] {revision.size_on_disk_str}")


def build_cache_index(
    hf_cache_info: HFCacheInfo,
) -> Tuple[
    Dict[str, CachedRepoInfo],
    Dict[str, Tuple[CachedRepoInfo, CachedRevisionInfo]],
]:
    repo_lookup: dict[str, CachedRepoInfo] = {}
    revision_lookup: dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]] = {}
    for repo in hf_cache_info.repos:
        repo_key = f"{repo.repo_type}/{repo.repo_id}".lower()
        repo_lookup[repo_key] = repo
        for revision in repo.revisions:
            revision_lookup[revision.commit_hash.lower()] = (repo, revision)
    return repo_lookup, revision_lookup


def collect_cache_entries(
    hf_cache_info: HFCacheInfo, *, include_revisions: bool
) -> Tuple[List[CacheEntry], RepoRefsMap]:
    entries: List[CacheEntry] = []
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
                format_cache_repo_id(entry[0]),
                entry[1].commit_hash if entry[1] is not None else "",
            )
        )
    else:
        entries.sort(key=lambda entry: format_cache_repo_id(entry[0]))
    return entries, repo_refs_map


def format_cache_repo_id(repo: CachedRepoInfo) -> str:
    return f"{repo.repo_type}/{repo.repo_id}"


def compile_cache_filter(
    expr: str, repo_refs_map: RepoRefsMap
) -> Callable[[CachedRepoInfo, Optional[CachedRevisionInfo], float], bool]:
    match = _FILTER_PATTERN.match(expr.strip())
    if not match:
        raise ValueError(f"Invalid filter expression: '{expr}'.")

    key = match.group("key").lower()
    op = match.group("op")
    op = "=" if op == "==" else op
    value_raw = match.group("value").strip()

    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Unsupported operator '{op}' in filter '{expr}'.")

    if key == "size":
        size_threshold = parse_cache_size(value_raw)
        return lambda repo, revision, _: _compare_numeric(
            revision.size_on_disk if revision is not None else repo.size_on_disk,
            op,
            size_threshold,
        )

    if key in {"modified", "accessed"}:
        stripped = value_raw.strip().lower()
        if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
            time_kind, time_value = ("absolute", float(stripped))
        else:
            time_kind, time_value = ("relative", parse_cache_duration(value_raw))

        if key == "modified":

            def _time_filter(repo: CachedRepoInfo, revision: Optional[CachedRevisionInfo], now: float) -> bool:
                timestamp = revision.last_modified if revision is not None else repo.last_modified
                if timestamp is None:
                    return False
                if time_kind == "relative":
                    return _compare_numeric(now - timestamp, op, time_value)
                return _compare_numeric(timestamp, op, time_value)

        else:

            def _time_filter(repo: CachedRepoInfo, revision: Optional[CachedRevisionInfo], now: float) -> bool:
                timestamp = repo.last_accessed
                if timestamp is None:
                    return False
                if time_kind == "relative":
                    return _compare_numeric(now - timestamp, op, time_value)
                return _compare_numeric(timestamp, op, time_value)

        return _time_filter

    if key == "type":
        expected = value_raw.lower()

        if op not in {"=", "!="}:
            raise ValueError("Only '=' and '!=' are supported for 'type' filters.")

        def _type_filter(repo: CachedRepoInfo, revision: Optional[CachedRevisionInfo], _: float) -> bool:
            actual = repo.repo_type.lower()
            return actual == expected if op == "=" else actual != expected

        return _type_filter

    if key == "id":
        needle = value_raw.lower()

        if op not in {"=", "!="}:
            raise ValueError("Only '=' and '!=' are supported for 'id' filters.")

        def _id_filter(repo: CachedRepoInfo, revision: Optional[CachedRevisionInfo], _: float) -> bool:
            haystack = format_cache_repo_id(repo).lower()
            contains = needle in haystack
            return contains if op == "=" else not contains

        return _id_filter

    if key == "refs":
        needle = value_raw.lower()

        if op not in {"=", "!="}:
            raise ValueError("Only '=' and '!=' are supported for 'refs' filters.")

        def _refs_filter(repo: CachedRepoInfo, revision: Optional[CachedRevisionInfo], _: float) -> bool:
            refs = revision.refs if revision is not None else repo_refs_map.get(repo, frozenset())
            refs_lower = [ref.lower() for ref in refs]
            contains = any(needle in ref for ref in refs_lower)
            return contains if op == "=" else not contains

        return _refs_filter

    raise ValueError(f"Unsupported filter key '{key}' in '{expr}'.")


def print_cache_entries_table(
    entries: List[CacheEntry], *, include_revisions: bool, repo_refs_map: RepoRefsMap
) -> None:
    if not entries:
        message = "No cached revisions found." if include_revisions else "No cached repositories found."
        print(message)
        return

    if include_revisions:
        table_rows: List[List[str]] = []
        headers = ["ID", "REVISION", "SIZE", "LAST_MODIFIED", "REFS"]
        table_rows = [
            [
                format_cache_repo_id(repo),
                revision.commit_hash,
                revision.size_on_disk_str.rjust(8),
                revision.last_modified_str,
                ", ".join(sorted(revision.refs)),
            ]
            for repo, revision in entries
            if revision is not None
        ]
    else:
        headers = ["ID", "SIZE", "LAST_ACCESSED", "LAST_MODIFIED", "REFS"]
        table_rows = [
            [
                format_cache_repo_id(repo),
                repo.size_on_disk_str.rjust(8),
                repo.last_accessed_str or "",
                repo.last_modified_str,
                ", ".join(sorted(repo_refs_map.get(repo, frozenset()))),
            ]
            for repo, _ in entries
        ]

    print(tabulate(table_rows, headers=headers))  # type: ignore[arg-type]


def print_cache_entries_json(
    entries: List[CacheEntry], *, include_revisions: bool, repo_refs_map: RepoRefsMap
) -> None:
    if include_revisions:
        payload = [
            {
                "id": format_cache_repo_id(repo),
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "revision": revision.commit_hash,
                "size_on_disk": revision.size_on_disk,
                "size_on_disk_str": revision.size_on_disk_str,
                "last_accessed": repo.last_accessed,
                "last_accessed_str": repo.last_accessed_str,
                "last_modified": revision.last_modified,
                "last_modified_str": revision.last_modified_str,
                "refs": sorted(revision.refs),
                "snapshot_path": str(revision.snapshot_path),
            }
            for repo, revision in entries
            if revision is not None
        ]
    else:
        payload = [
            {
                "id": format_cache_repo_id(repo),
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size_on_disk": repo.size_on_disk,
                "size_on_disk_str": repo.size_on_disk_str,
                "last_accessed": repo.last_accessed,
                "last_accessed_str": repo.last_accessed_str,
                "last_modified": repo.last_modified,
                "last_modified_str": repo.last_modified_str,
                "refs": sorted(repo_refs_map.get(repo, frozenset())),
            }
            for repo, _ in entries
        ]

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def print_cache_entries_csv(entries: List[CacheEntry], *, include_revisions: bool, repo_refs_map: RepoRefsMap) -> None:
    writer = csv.writer(sys.stdout)
    if include_revisions:
        writer.writerow(
            ["id", "revision", "repo_type", "size_bytes", "size", "last_modified", "last_modified_str", "refs"]
        )
        for repo, revision in entries:
            if revision is None:
                continue
            writer.writerow(
                [
                    format_cache_repo_id(repo),
                    revision.commit_hash,
                    repo.repo_type,
                    revision.size_on_disk,
                    revision.size_on_disk_str,
                    revision.last_modified,
                    revision.last_modified_str,
                    "; ".join(sorted(revision.refs)),
                ]
            )
    else:
        writer.writerow(
            [
                "id",
                "repo_type",
                "size_bytes",
                "size",
                "last_accessed",
                "last_accessed_str",
                "last_modified",
                "last_modified_str",
                "refs",
            ]
        )
        for repo, _ in entries:
            writer.writerow(
                [
                    format_cache_repo_id(repo),
                    repo.repo_type,
                    repo.size_on_disk,
                    repo.size_on_disk_str,
                    repo.last_accessed if repo.last_accessed is not None else "",
                    repo.last_accessed_str or "",
                    repo.last_modified,
                    repo.last_modified_str,
                    "; ".join(sorted(repo_refs_map.get(repo, frozenset()))),
                ]
            )


def _compare_numeric(left: Optional[float], op: str, right: float) -> bool:
    if left is None:
        return False

    return {
        "=": left == right,
        "!=": left != right,
        ">": left > right,
        "<": left < right,
        ">=": left >= right,
        "<=": left <= right,
    }.get(op, False)
