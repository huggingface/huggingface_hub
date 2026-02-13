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
"""Contains commands to interact with buckets via the CLI."""

import fnmatch
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Iterator, Literal, Optional, Union

import typer

from huggingface_hub import HfApi, logging
from huggingface_hub.hf_api import BucketFile
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from ._cli_utils import (
    FormatOpt,
    OutputFormat,
    QuietOpt,
    StatusLine,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    print_list_output,
    typer_factory,
)


logger = logging.get_logger(__name__)


BUCKET_PREFIX = "hf://buckets/"


bucket_cli = typer_factory(help="Commands to interact with buckets.")


def _parse_bucket_path(path: str) -> tuple[str, str]:
    """Parse a bucket path like hf://buckets/namespace/bucket_name/prefix into (bucket_id, prefix).

    Returns:
        tuple: (bucket_id, prefix) where bucket_id is "namespace/bucket_name" and prefix may be empty string.
    """
    if not path.startswith(BUCKET_PREFIX):
        raise ValueError(f"Invalid bucket path: {path}. Must start with {BUCKET_PREFIX}")

    path_without_prefix = path[len(BUCKET_PREFIX) :]
    parts = path_without_prefix.split("/", 2)

    if len(parts) < 2:
        raise ValueError(
            f"Invalid bucket path: {path}. Must be in format {BUCKET_PREFIX}namespace/bucket_name(/prefix)"
        )

    namespace = parts[0]
    bucket_name = parts[1]
    bucket_id = f"{namespace}/{bucket_name}"
    prefix = parts[2] if len(parts) > 2 else ""

    return bucket_id, prefix


def _parse_bucket_argument(argument: str) -> tuple[str, str]:
    """Parse a bucket argument accepting both 'namespace/name(/prefix)' and 'hf://buckets/namespace/name(/prefix)'.

    Returns:
        tuple: (bucket_id, prefix) where bucket_id is "namespace/bucket_name" and prefix may be empty string.
    """
    if argument.startswith(BUCKET_PREFIX):
        return _parse_bucket_path(argument)

    parts = argument.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid bucket argument: {argument}. Must be in format namespace/bucket_name"
            f" or {BUCKET_PREFIX}namespace/bucket_name"
        )

    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = parts[2] if len(parts) > 2 else ""
    return bucket_id, prefix


def _format_size(size: Union[int, float], human_readable: bool = False) -> str:
    """Format a size in bytes."""
    if not human_readable:
        return str(size)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            if unit == "B":
                return f"{size} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _format_mtime(mtime: Optional[datetime]) -> str:
    """Format mtime datetime to a readable date string."""
    if mtime is None:
        return ""
    return mtime.strftime("%Y-%m-%d %H:%M:%S")


def _build_tree(items: list[BucketFile]) -> list[str]:
    """Build a tree representation of files and directories.

    Args:
        items: List of items with 'path', 'type', 'size', 'mtime' keys

    Returns:
        List of formatted tree lines
    """
    # Build a nested structure
    tree: dict = {}

    for item in items:
        parts = item.path.split("/")
        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {"__children__": {}}
            current = current[part]["__children__"]

        # Store the item at the final level
        final_part = parts[-1]
        current[final_part] = {"__item__": item}

    # Render tree
    lines: list[str] = []
    _render_tree(tree, lines, "")
    return lines


def _render_tree(node: dict, lines: list[str], indent: str) -> None:
    """Recursively render a tree structure."""
    items = sorted(node.items())
    for i, (name, value) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "

        children = value.get("__children__", {})
        if children:
            lines.append(f"{indent}{connector}{name}/")
        else:
            lines.append(f"{indent}{connector}{name}")

        if children:
            child_indent = indent + ("    " if is_last else "│   ")
            _render_tree(children, lines, child_indent)


@bucket_cli.command(
    name="tree",
    examples=[
        "hf bucket tree user/my-bucket",
        "hf bucket tree hf://buckets/user/my-bucket",
        "hf bucket tree user/my-bucket/models",
        "hf bucket tree user/my-bucket -h",
        "hf bucket tree user/my-bucket --tree",
    ],
)
def tree_cmd(
    bucket: Annotated[
        str,
        typer.Argument(
            help="Bucket: namespace/bucket_name(/prefix) or hf://buckets/namespace/bucket_name(/prefix)",
        ),
    ],
    human_readable: Annotated[
        bool,
        typer.Option(
            "--human-readable",
            "-h",
            help="Show file size in human readable format.",
        ),
    ] = False,
    as_tree: Annotated[
        bool,
        typer.Option(
            "--tree",
            help="List files in tree format.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """List files in a bucket."""
    api = get_hf_api(token=token)

    try:
        bucket_id, prefix = _parse_bucket_argument(bucket)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    # Fetch items from the bucket
    items = list(
        api.list_bucket_tree(
            bucket_id,
            prefix=prefix or None,
        )
    )

    if not items:
        print("(empty)")
        return

    if as_tree:
        # Tree format
        tree_lines = _build_tree(items)
        for line in tree_lines:
            print(line)
    else:
        # Table format
        for item in items:
            size_str = _format_size(item.size, human_readable)
            mtime_str = _format_mtime(item.mtime)
            print(f"{size_str:>12}  {mtime_str:>19}  {item.path}")


@bucket_cli.command(
    name="create",
    examples=[
        "hf bucket create my-bucket",
        "hf bucket create user/my-bucket",
        "hf bucket create hf://buckets/user/my-bucket",
        "hf bucket create user/my-bucket --private",
        "hf bucket create user/my-bucket --exist-ok",
    ],
)
def create(
    bucket_id: Annotated[
        str,
        typer.Argument(
            help="Bucket ID: bucket_name, namespace/bucket_name, or hf://buckets/namespace/bucket_name",
        ),
    ],
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Create a private bucket.",
        ),
    ] = False,
    exist_ok: Annotated[
        bool,
        typer.Option(
            "--exist-ok",
            help="Do not raise an error if the bucket already exists.",
        ),
    ] = False,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """Create a new bucket."""
    api = get_hf_api(token=token)

    if bucket_id.startswith(BUCKET_PREFIX):
        try:
            parsed_id, prefix = _parse_bucket_argument(bucket_id)
        except ValueError as e:
            raise typer.BadParameter(str(e))
        if prefix:
            raise typer.BadParameter(
                f"Cannot specify a prefix for bucket creation: {bucket_id}."
                f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
            )
        bucket_id = parsed_id

    bucket_url = api.create_bucket(
        bucket_id,
        private=private if private else None,
        exist_ok=exist_ok,
    )
    if quiet:
        print(bucket_url.handle)
    else:
        print(f"Bucket created: {bucket_url.url} (handle: {bucket_url.handle})")


@bucket_cli.command(
    name="list",
    examples=[
        "hf bucket list",
        "hf bucket list huggingface",
    ],
)
def list_cmd(
    token: TokenOpt = None,
    namespace: Annotated[
        Optional[str],
        typer.Argument(help="Namespace to list buckets from (user or organization). Defaults to user's namespace."),
    ] = None,
    human_readable: Annotated[
        bool,
        typer.Option(
            "--human-readable",
            "-h",
            help="Show sizes in human readable format.",
        ),
    ] = False,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
) -> None:
    """List all accessible buckets."""
    if namespace is not None and ("/" in namespace or namespace.startswith(BUCKET_PREFIX)):
        raise typer.BadParameter(
            f"Expected a namespace (user or organization), not a bucket ID: '{namespace}'."
            " To list files in a bucket, use: hf bucket tree " + namespace
        )

    api = get_hf_api(token=token)
    results = [api_object_to_dict(bucket) for bucket in api.list_buckets(namespace=namespace)]
    headers = ["id", "private", "size", "total_files", "created_at"]

    def row_fn(item: dict) -> list[str]:
        from ._cli_utils import _format_cell

        return [
            _format_cell(item.get("id")),
            _format_cell(item.get("private")),
            _format_size(item.get("size", 0), human_readable=human_readable),
            _format_cell(item.get("total_files")),
            _format_cell(item.get("created_at")),
        ]

    alignments = {"size": "right", "total_files": "right"}
    print_list_output(results, format=format, quiet=quiet, headers=headers, row_fn=row_fn, alignments=alignments)


@bucket_cli.command(
    name="info",
    examples=[
        "hf bucket info user/my-bucket",
        "hf bucket info hf://buckets/user/my-bucket",
    ],
)
def info(
    bucket_id: Annotated[
        str,
        typer.Argument(
            help="Bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """Get info about a bucket."""
    api = get_hf_api(token=token)

    try:
        parsed_id, _ = _parse_bucket_argument(bucket_id)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    bucket = api.bucket_info(parsed_id)
    if quiet:
        print(bucket.id)
    else:
        print(json.dumps(api_object_to_dict(bucket), indent=2))


@bucket_cli.command(
    name="delete",
    examples=[
        "hf bucket delete user/my-bucket",
        "hf bucket delete hf://buckets/user/my-bucket",
        "hf bucket delete user/my-bucket --yes",
        "hf bucket delete user/my-bucket --missing-ok",
    ],
)
def delete(
    bucket_id: Annotated[
        str,
        typer.Argument(
            help="Bucket ID: namespace/bucket_name or hf://buckets/namespace/bucket_name",
        ),
    ],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    missing_ok: Annotated[
        bool,
        typer.Option(
            "--missing-ok",
            help="Do not raise an error if the bucket does not exist.",
        ),
    ] = False,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """Delete a bucket."""
    api = get_hf_api(token=token)

    if bucket_id.startswith(BUCKET_PREFIX):
        try:
            parsed_id, prefix = _parse_bucket_argument(bucket_id)
        except ValueError as e:
            raise typer.BadParameter(str(e))
        if prefix:
            raise typer.BadParameter(
                f"Cannot specify a prefix for bucket deletion: {bucket_id}."
                f" Use namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
            )
        bucket_id = parsed_id
    elif "/" not in bucket_id:
        raise typer.BadParameter(
            f"Invalid bucket ID: {bucket_id}."
            f" Must be in format namespace/bucket_name or {BUCKET_PREFIX}namespace/bucket_name."
        )

    # Confirm deletion unless -y flag is provided
    if not yes:
        confirm = typer.confirm(f"Are you sure you want to delete bucket '{bucket_id}'?")
        if not confirm:
            print("Aborted.")
            raise typer.Abort()

    api.delete_bucket(
        bucket_id,
        missing_ok=missing_ok,
    )
    if quiet:
        print(bucket_id)
    else:
        print(f"Bucket deleted: {bucket_id}")


# =============================================================================
# Sync command
# =============================================================================


@dataclass
class SyncOperation:
    """Represents a sync operation to be performed."""

    action: Literal["upload", "download", "delete", "skip"]
    path: str
    size: Optional[int] = None
    reason: str = ""
    local_mtime: Optional[str] = None
    remote_mtime: Optional[str] = None
    bucket_file: Optional[Any] = None  # BucketFile when available (not serialized to plan file)


@dataclass
class SyncPlan:
    """Represents a complete sync plan."""

    source: str
    dest: str
    timestamp: str
    operations: list[SyncOperation] = field(default_factory=list)

    def summary(self) -> dict[str, Union[int, str]]:
        uploads = sum(1 for op in self.operations if op.action == "upload")
        downloads = sum(1 for op in self.operations if op.action == "download")
        deletes = sum(1 for op in self.operations if op.action == "delete")
        skips = sum(1 for op in self.operations if op.action == "skip")
        total_size = sum(op.size or 0 for op in self.operations if op.action in ("upload", "download"))
        return {
            "uploads": uploads,
            "downloads": downloads,
            "deletes": deletes,
            "skips": skips,
            "total_size": total_size,
        }


def _is_bucket_path(path: str) -> bool:
    """Check if a path is a bucket path."""
    return path.startswith(BUCKET_PREFIX)


def _list_local_files(local_path: str) -> Iterator[tuple[str, int, float]]:
    """List all files in a local directory.

    Yields:
        tuple: (relative_path, size, mtime_ms) for each file
    """
    local_path = os.path.abspath(local_path)
    if not os.path.isdir(local_path):
        raise ValueError(f"Local path must be a directory: {local_path}")

    for root, _, files in os.walk(local_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, local_path)
            # Normalize to forward slashes for consistency
            rel_path = rel_path.replace(os.sep, "/")
            size = os.path.getsize(full_path)
            mtime_ms = os.path.getmtime(full_path) * 1000
            yield rel_path, size, mtime_ms


def _list_remote_files(api: HfApi, bucket_id: str, prefix: str) -> Iterator[tuple[str, int, float, Any]]:
    """List all files in a bucket with a given prefix.

    Yields:
        tuple: (relative_path, size, mtime_ms, bucket_file) for each file.
            bucket_file is the BucketFile object from list_bucket_tree.
    """
    for item in api.list_bucket_tree(bucket_id, prefix=prefix or None):
        path = item.path
        # Remove prefix from path to get relative path
        if prefix:
            if path.startswith(prefix + "/"):
                rel_path = path[len(prefix) + 1 :]
            elif path.startswith(prefix):
                rel_path = path[len(prefix) :]
            else:
                rel_path = path
        else:
            rel_path = path
        mtime_ms = item.mtime.timestamp() * 1000 if item.mtime else 0
        yield rel_path, item.size, mtime_ms, item


class FilterMatcher:
    """Matches file paths against include/exclude patterns."""

    def __init__(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        filter_rules: Optional[list[tuple[str, str]]] = None,
    ):
        """Initialize the filter matcher.

        Args:
            include_patterns: Patterns to include (from --include)
            exclude_patterns: Patterns to exclude (from --exclude)
            filter_rules: Rules from filter file as list of ("+"/"-", pattern) tuples
        """
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.filter_rules = filter_rules or []

    def matches(self, path: str) -> bool:
        """Check if a path should be included based on the filter rules.

        Filtering rules:
        - Filters are evaluated in order, first matching rule decides
        - If no rules match, include by default (unless include patterns are specified)
        """
        # First check filter rules from file (in order)
        for sign, pattern in self.filter_rules:
            if fnmatch.fnmatch(path, pattern):
                return sign == "+"

        # Then check CLI patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, pattern):
                return False

        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True

        # If include patterns were specified but none matched, exclude
        if self.include_patterns:
            return False

        # Default: include
        return True


def _parse_filter_file(filter_file: str) -> list[tuple[str, str]]:
    """Parse a filter file and return a list of (sign, pattern) tuples.

    Filter file format:
    - Lines starting with "+" are include patterns
    - Lines starting with "-" are exclude patterns
    - Empty lines and lines starting with "#" are ignored
    """
    rules = []
    with open(filter_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("+"):
                rules.append(("+", line[1:].strip()))
            elif line.startswith("-"):
                rules.append(("-", line[1:].strip()))
            else:
                # Default to include if no prefix
                rules.append(("+", line))
    return rules


def _mtime_to_iso(mtime_ms: float) -> str:
    """Convert mtime in milliseconds to ISO format string."""
    return datetime.fromtimestamp(mtime_ms / 1000, tz=timezone.utc).isoformat()


def _compute_sync_plan(
    source: str,
    dest: str,
    api,
    delete: bool = False,
    ignore_times: bool = False,
    ignore_sizes: bool = False,
    existing: bool = False,
    ignore_existing: bool = False,
    filter_matcher: Optional[FilterMatcher] = None,
    status: Optional[StatusLine] = None,
) -> SyncPlan:
    """Compute the sync plan by comparing source and destination.

    Returns:
        SyncPlan with all operations to be performed
    """
    filter_matcher = filter_matcher or FilterMatcher()
    is_upload = not _is_bucket_path(source) and _is_bucket_path(dest)
    is_download = _is_bucket_path(source) and not _is_bucket_path(dest)

    if not is_upload and not is_download:
        raise ValueError("One of source or dest must be a bucket path (hf://buckets/...) and the other must be local.")

    plan = SyncPlan(
        source=source,
        dest=dest,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if is_upload:
        # Local -> Remote
        local_path = os.path.abspath(source)
        bucket_id, prefix = _parse_bucket_path(dest)

        if not os.path.isdir(local_path):
            raise ValueError(f"Source must be a directory: {local_path}")

        # Get local and remote file lists
        local_files = {}
        for rel_path, size, mtime_ms in _list_local_files(local_path):
            if filter_matcher.matches(rel_path):
                local_files[rel_path] = (size, mtime_ms)
            if status:
                status.update(f"Scanning local directory ({len(local_files)} files)")
        if status:
            status.done(f"Scanning local directory ({len(local_files)} files)")

        remote_files = {}
        remote_total: Optional[int] = None
        if status:
            try:
                remote_total = api.bucket_info(bucket_id).total_files
            except Exception:
                pass
        try:
            for rel_path, size, mtime_ms, _ in _list_remote_files(api, bucket_id, prefix):
                if filter_matcher.matches(rel_path):
                    remote_files[rel_path] = (size, mtime_ms)
                if status:
                    total_str = f"/{remote_total}" if remote_total is not None else ""
                    status.update(f"Scanning remote bucket ({len(remote_files)}{total_str} files)")
        except Exception as e:
            # Bucket might not exist yet or be empty
            logger.debug(f"Could not list remote files: {e}")
        if status:
            status.done(f"Scanning remote bucket ({len(remote_files)} files)")

        # Compare files
        all_paths = set(local_files.keys()) | set(remote_files.keys())
        if status:
            status.done(f"Comparing files ({len(all_paths)} paths)")
        for path in sorted(all_paths):
            local_info = local_files.get(path)
            remote_info = remote_files.get(path)

            if local_info and not remote_info:
                # New file
                if existing:
                    # --existing: skip new files
                    plan.operations.append(
                        SyncOperation(
                            action="skip",
                            path=path,
                            size=local_info[0],
                            reason="new file (--existing)",
                            local_mtime=_mtime_to_iso(local_info[1]),
                        )
                    )
                else:
                    plan.operations.append(
                        SyncOperation(
                            action="upload",
                            path=path,
                            size=local_info[0],
                            reason="new file",
                            local_mtime=_mtime_to_iso(local_info[1]),
                        )
                    )
            elif local_info and remote_info:
                # File exists in both
                local_size, local_mtime = local_info
                remote_size, remote_mtime = remote_info

                if ignore_existing:
                    # --ignore-existing: skip files that exist on receiver
                    plan.operations.append(
                        SyncOperation(
                            action="skip",
                            path=path,
                            size=local_size,
                            reason="exists on receiver (--ignore-existing)",
                            local_mtime=_mtime_to_iso(local_mtime),
                            remote_mtime=_mtime_to_iso(remote_mtime),
                        )
                    )
                    continue

                size_differs = local_size != remote_size
                local_newer = local_mtime > remote_mtime

                if ignore_sizes:
                    # Only check mtime
                    if local_newer:
                        plan.operations.append(
                            SyncOperation(
                                action="upload",
                                path=path,
                                size=local_size,
                                reason="local newer",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=local_size,
                                reason="same mtime",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                elif ignore_times:
                    # Only check size
                    if size_differs:
                        plan.operations.append(
                            SyncOperation(
                                action="upload",
                                path=path,
                                size=local_size,
                                reason="size differs",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=local_size,
                                reason="same size",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                else:
                    # Check both size and mtime
                    if size_differs or local_newer:
                        plan.operations.append(
                            SyncOperation(
                                action="upload",
                                path=path,
                                size=local_size,
                                reason="size differs" if size_differs else "local newer",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=local_size,
                                reason="identical",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
            elif not local_info and remote_info and delete:
                # File only in remote and --delete mode
                plan.operations.append(
                    SyncOperation(
                        action="delete",
                        path=path,
                        size=remote_info[0],
                        reason="not in source (--delete)",
                        remote_mtime=_mtime_to_iso(remote_info[1]),
                    )
                )

    else:
        # Remote -> Local (download)
        bucket_id, prefix = _parse_bucket_path(source)
        local_path = os.path.abspath(dest)

        # Get remote and local file lists
        remote_files = {}
        bucket_file_map: dict[str, Any] = {}
        remote_total: Optional[int] = None
        if status:
            try:
                remote_total = api.bucket_info(bucket_id).total_files
            except Exception:
                pass
        for rel_path, size, mtime_ms, bucket_file in _list_remote_files(api, bucket_id, prefix):
            if filter_matcher.matches(rel_path):
                remote_files[rel_path] = (size, mtime_ms)
                bucket_file_map[rel_path] = bucket_file
            if status:
                total_str = f"/{remote_total}" if remote_total is not None else ""
                status.update(f"Scanning remote bucket ({len(remote_files)}{total_str} files)")
        if status:
            status.done(f"Scanning remote bucket ({len(remote_files)} files)")

        local_files = {}
        if os.path.isdir(local_path):
            for rel_path, size, mtime_ms in _list_local_files(local_path):
                if filter_matcher.matches(rel_path):
                    local_files[rel_path] = (size, mtime_ms)
                if status:
                    status.update(f"Scanning local directory ({len(local_files)} files)")
        if status:
            status.done(f"Scanning local directory ({len(local_files)} files)")

        # Compare files
        all_paths = set(remote_files.keys()) | set(local_files.keys())
        if status:
            status.done(f"Comparing files ({len(all_paths)} paths)")
        for path in sorted(all_paths):
            remote_info = remote_files.get(path)
            local_info = local_files.get(path)

            if remote_info and not local_info:
                # New file
                if existing:
                    # --existing: skip new files
                    plan.operations.append(
                        SyncOperation(
                            action="skip",
                            path=path,
                            size=remote_info[0],
                            reason="new file (--existing)",
                            remote_mtime=_mtime_to_iso(remote_info[1]),
                        )
                    )
                else:
                    plan.operations.append(
                        SyncOperation(
                            action="download",
                            path=path,
                            size=remote_info[0],
                            reason="new file",
                            remote_mtime=_mtime_to_iso(remote_info[1]),
                            bucket_file=bucket_file_map.get(path),
                        )
                    )
            elif remote_info and local_info:
                # File exists in both
                remote_size, remote_mtime = remote_info
                local_size, local_mtime = local_info

                if ignore_existing:
                    # --ignore-existing: skip files that exist on receiver
                    plan.operations.append(
                        SyncOperation(
                            action="skip",
                            path=path,
                            size=remote_size,
                            reason="exists on receiver (--ignore-existing)",
                            local_mtime=_mtime_to_iso(local_mtime),
                            remote_mtime=_mtime_to_iso(remote_mtime),
                        )
                    )
                    continue

                size_differs = remote_size != local_size
                remote_newer = remote_mtime > local_mtime

                if ignore_sizes:
                    # Only check mtime
                    if remote_newer:
                        plan.operations.append(
                            SyncOperation(
                                action="download",
                                path=path,
                                size=remote_size,
                                reason="remote newer",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                                bucket_file=bucket_file_map.get(path),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=remote_size,
                                reason="same mtime",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                elif ignore_times:
                    # Only check size
                    if size_differs:
                        plan.operations.append(
                            SyncOperation(
                                action="download",
                                path=path,
                                size=remote_size,
                                reason="size differs",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                                bucket_file=bucket_file_map.get(path),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=remote_size,
                                reason="same size",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                else:
                    # Check both size and mtime
                    if size_differs or remote_newer:
                        plan.operations.append(
                            SyncOperation(
                                action="download",
                                path=path,
                                size=remote_size,
                                reason="size differs" if size_differs else "remote newer",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                                bucket_file=bucket_file_map.get(path),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=remote_size,
                                reason="identical",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
            elif not remote_info and local_info and delete:
                # File only in local and --delete mode
                plan.operations.append(
                    SyncOperation(
                        action="delete",
                        path=path,
                        size=local_info[0],
                        reason="not in source (--delete)",
                        local_mtime=_mtime_to_iso(local_info[1]),
                    )
                )

    return plan


def _save_plan(plan: SyncPlan, plan_file: str) -> None:
    """Save a sync plan to a JSONL file."""
    with open(plan_file, "w") as f:
        # Write header
        header = {
            "type": "header",
            "source": plan.source,
            "dest": plan.dest,
            "timestamp": plan.timestamp,
            "summary": plan.summary(),
        }
        f.write(json.dumps(header) + "\n")

        # Write operations
        for op in plan.operations:
            op_dict: dict[str, Any] = {
                "type": "operation",
                "action": op.action,
                "path": op.path,
                "reason": op.reason,
            }
            if op.size is not None:
                op_dict["size"] = op.size
            if op.local_mtime is not None:
                op_dict["local_mtime"] = op.local_mtime
            if op.remote_mtime is not None:
                op_dict["remote_mtime"] = op.remote_mtime
            f.write(json.dumps(op_dict) + "\n")


def _load_plan(plan_file: str) -> SyncPlan:
    """Load a sync plan from a JSONL file."""
    with open(plan_file) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty plan file: {plan_file}")

    # Parse header
    header = json.loads(lines[0])
    if header.get("type") != "header":
        raise ValueError("Invalid plan file: expected header as first line")

    plan = SyncPlan(
        source=header["source"],
        dest=header["dest"],
        timestamp=header["timestamp"],
    )

    # Parse operations
    for line in lines[1:]:
        op_dict = json.loads(line)
        if op_dict.get("type") != "operation":
            continue
        plan.operations.append(
            SyncOperation(
                action=op_dict["action"],
                path=op_dict["path"],
                size=op_dict.get("size"),
                reason=op_dict.get("reason", ""),
                local_mtime=op_dict.get("local_mtime"),
                remote_mtime=op_dict.get("remote_mtime"),
            )
        )

    return plan


def _execute_plan(plan: SyncPlan, api, verbose: bool = False, status: Optional[StatusLine] = None) -> None:
    """Execute a sync plan."""
    is_upload = not _is_bucket_path(plan.source) and _is_bucket_path(plan.dest)
    is_download = _is_bucket_path(plan.source) and not _is_bucket_path(plan.dest)

    if is_upload:
        local_path = os.path.abspath(plan.source)
        bucket_id, prefix = _parse_bucket_path(plan.dest)

        # Collect operations
        add_files = []
        delete_paths = []

        for op in plan.operations:
            if op.action == "upload":
                local_file = os.path.join(local_path, op.path)
                remote_path = f"{prefix}/{op.path}" if prefix else op.path
                if verbose:
                    print(f"  Uploading: {op.path} ({op.reason})")
                add_files.append((local_file, remote_path))
            elif op.action == "delete":
                remote_path = f"{prefix}/{op.path}" if prefix else op.path
                if verbose:
                    print(f"  Deleting: {op.path} ({op.reason})")
                delete_paths.append(remote_path)
            elif op.action == "skip" and verbose:
                print(f"  Skipping: {op.path} ({op.reason})")

        # Execute batch operations
        if add_files or delete_paths:
            if status:
                parts = []
                if add_files:
                    parts.append(f"uploading {len(add_files)} files")
                if delete_paths:
                    parts.append(f"deleting {len(delete_paths)} files")
                status.done(", ".join(parts).capitalize())
            api.batch_bucket_files(
                bucket_id,
                add=add_files or None,
                delete=delete_paths or None,
            )

    elif is_download:
        bucket_id, prefix = _parse_bucket_path(plan.source)
        local_path = os.path.abspath(plan.dest)

        # Ensure local directory exists
        os.makedirs(local_path, exist_ok=True)

        # Collect download operations
        download_files = []
        delete_files = []

        for op in plan.operations:
            if op.action == "download":
                local_file = os.path.join(local_path, op.path)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                if verbose:
                    print(f"  Downloading: {op.path} ({op.reason})")
                # Use BucketFile when available (avoids extra metadata fetch per file)
                if op.bucket_file is not None:
                    download_files.append((op.bucket_file, local_file))
                else:
                    remote_path = f"{prefix}/{op.path}" if prefix else op.path
                    download_files.append((remote_path, local_file))
            elif op.action == "delete":
                local_file = os.path.join(local_path, op.path)
                if verbose:
                    print(f"  Deleting: {op.path} ({op.reason})")
                delete_files.append(local_file)
            elif op.action == "skip" and verbose:
                print(f"  Skipping: {op.path} ({op.reason})")

        # Execute downloads
        if download_files:
            if status:
                status.done(f"Downloading {len(download_files)} files")
            api.download_bucket_files(bucket_id, download_files)

        # Execute deletes
        if status and delete_files:
            status.done(f"Deleting {len(delete_files)} local files")
        for file_path in delete_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                # Remove empty parent directories
                parent = os.path.dirname(file_path)
                while parent != local_path:
                    try:
                        os.rmdir(parent)
                        parent = os.path.dirname(parent)
                    except OSError:
                        break


def _print_plan_summary(plan: SyncPlan) -> None:
    """Print a summary of the sync plan."""
    summary = plan.summary()
    print(f"Sync plan: {plan.source} -> {plan.dest}")
    print(f"  Uploads: {summary['uploads']}")
    print(f"  Downloads: {summary['downloads']}")
    print(f"  Deletes: {summary['deletes']}")
    print(f"  Skips: {summary['skips']}")


@bucket_cli.command(
    name="sync",
    examples=[
        "hf bucket sync ./data hf://buckets/user/my-bucket",
        "hf bucket sync hf://buckets/user/my-bucket ./data",
        "hf bucket sync ./data hf://buckets/user/my-bucket --delete",
        'hf bucket sync hf://buckets/user/my-bucket ./data --include "*.safetensors" --exclude "*.tmp"',
        "hf bucket sync ./data hf://buckets/user/my-bucket --plan sync-plan.jsonl",
        "hf bucket sync --apply sync-plan.jsonl",
    ],
)
def sync(
    source: Annotated[
        Optional[str],
        typer.Argument(
            help="Source path: local directory or hf://buckets/namespace/bucket_name(/prefix)",
        ),
    ] = None,
    dest: Annotated[
        Optional[str],
        typer.Argument(
            help="Destination path: local directory or hf://buckets/namespace/bucket_name(/prefix)",
        ),
    ] = None,
    delete: Annotated[
        bool,
        typer.Option(
            help="Delete destination files not present in source.",
        ),
    ] = False,
    ignore_times: Annotated[
        bool,
        typer.Option(
            "--ignore-times",
            help="Skip files only based on size, ignoring modification times.",
        ),
    ] = False,
    ignore_sizes: Annotated[
        bool,
        typer.Option(
            "--ignore-sizes",
            help="Skip files only based on modification times, ignoring sizes.",
        ),
    ] = False,
    plan: Annotated[
        Optional[str],
        typer.Option(
            help="Save sync plan to JSONL file for review instead of executing.",
        ),
    ] = None,
    apply: Annotated[
        Optional[str],
        typer.Option(
            help="Apply a previously saved plan file.",
        ),
    ] = None,
    include: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Include files matching pattern (can specify multiple).",
        ),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Exclude files matching pattern (can specify multiple).",
        ),
    ] = None,
    filter_from: Annotated[
        Optional[str],
        typer.Option(
            help="Read include/exclude patterns from file.",
        ),
    ] = None,
    existing: Annotated[
        bool,
        typer.Option(
            "--existing",
            help="Skip creating new files on receiver (only update existing files).",
        ),
    ] = False,
    ignore_existing: Annotated[
        bool,
        typer.Option(
            "--ignore-existing",
            help="Skip updating files that exist on receiver (only create new files).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed logging with reasoning.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Minimal output.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Sync files between local directory and a bucket."""
    api = get_hf_api(token=token)

    # Validate arguments
    if apply:
        # Apply mode: load and execute plan
        if source is not None or dest is not None:
            raise typer.BadParameter("Cannot specify source/dest when using --apply")
        if plan is not None:
            raise typer.BadParameter("Cannot specify both --plan and --apply")
        # Planning-related options are not allowed when applying
        if delete:
            raise typer.BadParameter("Cannot specify --delete when using --apply")
        if ignore_times:
            raise typer.BadParameter("Cannot specify --ignore-times when using --apply")
        if ignore_sizes:
            raise typer.BadParameter("Cannot specify --ignore-sizes when using --apply")
        if include:
            raise typer.BadParameter("Cannot specify --include when using --apply")
        if exclude:
            raise typer.BadParameter("Cannot specify --exclude when using --apply")
        if filter_from:
            raise typer.BadParameter("Cannot specify --filter-from when using --apply")
        if existing:
            raise typer.BadParameter("Cannot specify --existing when using --apply")
        if ignore_existing:
            raise typer.BadParameter("Cannot specify --ignore-existing when using --apply")

        sync_plan = _load_plan(apply)
        apply_status = StatusLine(enabled=not quiet)
        if not quiet:
            _print_plan_summary(sync_plan)
            print("Executing plan...")

        if quiet:
            disable_progress_bars()
        try:
            _execute_plan(sync_plan, api, verbose=verbose, status=apply_status)
        finally:
            if quiet:
                enable_progress_bars()

        if not quiet:
            print("Sync completed.")
        return

    # Normal mode: compute and optionally execute plan
    if source is None or dest is None:
        raise typer.BadParameter("Both source and dest are required (unless using --apply)")

    # Validate source/dest
    source_is_bucket = _is_bucket_path(source)
    dest_is_bucket = _is_bucket_path(dest)

    if source_is_bucket and dest_is_bucket:
        raise typer.BadParameter("Remote to remote sync is not supported. One path must be local.")

    if not source_is_bucket and not dest_is_bucket:
        raise typer.BadParameter("One of source or dest must be a bucket path (hf://buckets/...).")

    if ignore_times and ignore_sizes:
        raise typer.BadParameter("Cannot specify both --ignore-times and --ignore-sizes")

    if existing and ignore_existing:
        raise typer.BadParameter("Cannot specify both --existing and --ignore-existing")

    # Validate local path
    if source_is_bucket:
        # Download: dest is local
        if os.path.exists(dest) and not os.path.isdir(dest):
            raise typer.BadParameter(f"Destination must be a directory: {dest}")
    else:
        # Upload: source is local
        if not os.path.isdir(source):
            raise typer.BadParameter(f"Source must be an existing directory: {source}")

    # Build filter matcher
    filter_rules = None
    if filter_from:
        filter_rules = _parse_filter_file(filter_from)

    filter_matcher = FilterMatcher(
        include_patterns=include,
        exclude_patterns=exclude,
        filter_rules=filter_rules,
    )

    # Status line for TTY feedback (disabled in quiet mode)
    status = StatusLine(enabled=not quiet)

    # Compute sync plan
    if not quiet:
        print(f"Computing sync plan: {source} -> {dest}")

    sync_plan = _compute_sync_plan(
        source=source,
        dest=dest,
        api=api,
        delete=delete,
        ignore_times=ignore_times,
        ignore_sizes=ignore_sizes,
        existing=existing,
        ignore_existing=ignore_existing,
        filter_matcher=filter_matcher,
        status=status,
    )

    if plan:
        # Save plan to file
        _save_plan(sync_plan, plan)
        if not quiet:
            _print_plan_summary(sync_plan)
            print(f"Plan saved to: {plan}")
        return

    # Execute plan
    if not quiet:
        _print_plan_summary(sync_plan)

    # Check if there's anything to do
    summary = sync_plan.summary()
    if summary["uploads"] == 0 and summary["downloads"] == 0 and summary["deletes"] == 0:
        if not quiet:
            print("Nothing to sync.")
        return

    if not quiet:
        print("Syncing...")

    if quiet:
        disable_progress_bars()
    try:
        _execute_plan(sync_plan, api, verbose=verbose, status=status)
    finally:
        if quiet:
            enable_progress_bars()

    if not quiet:
        print("Sync completed.")


# =============================================================================
# Cp command
# =============================================================================


@bucket_cli.command(
    name="cp",
    examples=[
        "hf bucket cp hf://buckets/user/my-bucket/config.json",
        "hf bucket cp hf://buckets/user/my-bucket/config.json ./data/",
        "hf bucket cp hf://buckets/user/my-bucket/config.json my-config.json",
        "hf bucket cp hf://buckets/user/my-bucket/config.json -",
        "hf bucket cp my-config.json hf://buckets/user/my-bucket",
        "hf bucket cp my-config.json hf://buckets/user/my-bucket/logs/",
        "hf bucket cp my-config.json hf://buckets/user/my-bucket/remote-config.json",
        "hf bucket cp - hf://buckets/user/my-bucket/config.json",
    ],
)
def cp(
    src: Annotated[str, typer.Argument(help="Source: local file, hf://buckets/... path, or - for stdin")],
    dst: Annotated[
        Optional[str], typer.Argument(help="Destination: local path, hf://buckets/... path, or - for stdout")
    ] = None,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """Copy a single file to or from a bucket."""
    api = get_hf_api(token=token)

    src_is_bucket = _is_bucket_path(src)
    dst_is_bucket = dst is not None and _is_bucket_path(dst)
    src_is_stdin = src == "-"
    dst_is_stdout = dst == "-"

    # --- Validation ---
    if src_is_bucket and dst_is_bucket:
        raise typer.BadParameter("Remote-to-remote copy not supported.")

    if not src_is_bucket and not dst_is_bucket and not src_is_stdin:
        if dst is None:
            raise typer.BadParameter("Missing destination. Provide a bucket path as DST.")
        raise typer.BadParameter("One of SRC or DST must be a bucket path (hf://buckets/...).")

    if src_is_stdin and not dst_is_bucket:
        raise typer.BadParameter("Stdin upload requires a bucket destination.")

    if src_is_stdin and dst_is_bucket:
        _, prefix = _parse_bucket_path(dst)
        if prefix == "" or prefix.endswith("/"):
            raise typer.BadParameter("Stdin upload requires a full destination path including filename.")

    if dst_is_stdout and not src_is_bucket:
        raise typer.BadParameter("Cannot pipe to stdout for uploads.")

    if not src_is_bucket and not src_is_stdin and os.path.isdir(src):
        raise typer.BadParameter("Source must be a file, not a directory. Use `hf bucket sync` for directories.")

    # --- Determine direction and execute ---
    if src_is_bucket:
        # Download: remote -> local or stdout
        bucket_id, prefix = _parse_bucket_path(src)
        filename = prefix.rsplit("/", 1)[-1]

        if dst_is_stdout:
            # Download to stdout: always suppress progress bars to avoid polluting output
            disable_progress_bars()
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = os.path.join(tmp_dir, filename)
                    api.download_bucket_files(bucket_id, [(prefix, tmp_path)])
                    with open(tmp_path, "rb") as f:
                        sys.stdout.buffer.write(f.read())
            finally:
                enable_progress_bars()
        else:
            # Download to file
            if dst is None:
                local_path = filename
            elif os.path.isdir(dst) or dst.endswith(os.sep) or dst.endswith("/"):
                local_path = os.path.join(dst, filename)
            else:
                local_path = dst

            # Ensure parent directory exists
            parent_dir = os.path.dirname(local_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            if quiet:
                disable_progress_bars()
            try:
                api.download_bucket_files(bucket_id, [(prefix, local_path)])
            finally:
                if quiet:
                    enable_progress_bars()

            if not quiet:
                print(f"Downloaded: {src} -> {local_path}")

    elif src_is_stdin:
        # Upload from stdin
        bucket_id, remote_path = _parse_bucket_path(dst)  # type: ignore[arg-type]
        data = sys.stdin.buffer.read()

        if quiet:
            disable_progress_bars()
        try:
            api.batch_bucket_files(bucket_id, add=[(data, remote_path)])
        finally:
            if quiet:
                enable_progress_bars()

        if not quiet:
            print(f"Uploaded: stdin -> {dst}")

    else:
        # Upload from file
        if not os.path.isfile(src):
            raise typer.BadParameter(f"Source file not found: {src}")

        bucket_id, prefix = _parse_bucket_path(dst)  # type: ignore[arg-type]

        if prefix == "":
            remote_path = os.path.basename(src)
        elif prefix.endswith("/"):
            remote_path = prefix + os.path.basename(src)
        else:
            remote_path = prefix

        if quiet:
            disable_progress_bars()
        try:
            api.batch_bucket_files(bucket_id, add=[(src, remote_path)])
        finally:
            if quiet:
                enable_progress_bars()

        if not quiet:
            print(f"Uploaded: {src} -> {BUCKET_PREFIX}{bucket_id}/{remote_path}")
