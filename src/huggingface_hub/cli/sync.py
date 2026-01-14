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
"""Contains command to sync files between local and bucket with the CLI.

Usage:
    hf sync --help

    # Basic sync (no deletions)
    hf sync ./data hf://buckets/user/my-bucket
    hf sync hf://buckets/user/my-bucket ./data

    # Basic sync in subfolder
    hf sync ./data hf://buckets/user/my-bucket/data/
    hf sync hf://buckets/user/my-bucket/data ./data

    # Mirror (exact replica with deletions)
    hf sync ./data hf://buckets/user/my-bucket --mirror
    hf sync hf://buckets/user/my-bucket ./data --mirror

    # With filters
    hf sync hf://buckets/user/my-bucket ./data --include "*.safetensors" --exclude "*.tmp"

    # Force upload/download (skip mtime and size checks)
    hf sync ./data hf://buckets/user/my-bucket --force-upload
    hf sync hf://buckets/user/my-bucket ./data --force-download

    # Safe review workflow
    hf sync ./data hf://buckets/user/my-bucket --plan sync-plan.jsonl
    hf sync --execute sync-plan.jsonl
"""

import fnmatch
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Iterator, Literal, Optional, Union

import typer

from huggingface_hub import HfApi, logging
from huggingface_hub.hf_api import BucketAddFile, BucketDeleteFile
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from ._cli_utils import TokenOpt, get_hf_api


logger = logging.get_logger(__name__)

BUCKET_PREFIX = "hf://buckets/"


@dataclass
class SyncOperation:
    """Represents a sync operation to be performed."""

    action: Literal["upload", "download", "delete", "skip"]
    path: str
    size: Optional[int] = None
    reason: str = ""
    local_mtime: Optional[str] = None
    remote_mtime: Optional[str] = None


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


def _list_remote_files(
    api: HfApi, bucket_id: str, prefix: str, token: Optional[str]
) -> Iterator[tuple[str, int, float]]:
    """List all files in a bucket with a given prefix.

    Yields:
        tuple: (relative_path, size, mtime_ms) for each file
    """
    for item in api.list_bucket_tree(bucket_id, prefix=prefix or None, token=token):
        if item.get("type") == "file":
            path = item["path"]
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
            size = item.get("size", 0)
            mtime_ms = item.get("mtime", 0)
            yield rel_path, size, mtime_ms


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
    token: Optional[str],
    mirror: bool = False,
    force_upload: bool = False,
    force_download: bool = False,
    filter_matcher: Optional[FilterMatcher] = None,
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

        remote_files = {}
        try:
            for rel_path, size, mtime_ms in _list_remote_files(api, bucket_id, prefix, token):
                if filter_matcher.matches(rel_path):
                    remote_files[rel_path] = (size, mtime_ms)
        except Exception as e:
            # Bucket might not exist yet or be empty
            logger.debug(f"Could not list remote files: {e}")

        # Compare files
        all_paths = set(local_files.keys()) | set(remote_files.keys())
        for path in sorted(all_paths):
            local_info = local_files.get(path)
            remote_info = remote_files.get(path)

            if local_info and not remote_info:
                # New file
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

                if force_upload:
                    plan.operations.append(
                        SyncOperation(
                            action="upload",
                            path=path,
                            size=local_size,
                            reason="forced",
                            local_mtime=_mtime_to_iso(local_mtime),
                            remote_mtime=_mtime_to_iso(remote_mtime),
                        )
                    )
                elif local_size != remote_size:
                    # Size differs - use mtime to decide
                    if local_mtime > remote_mtime:
                        plan.operations.append(
                            SyncOperation(
                                action="upload",
                                path=path,
                                size=local_size,
                                reason="local newer (size differs)",
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
                                reason="remote newer (size differs)",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                elif local_mtime > remote_mtime:
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
                            reason="identical",
                            local_mtime=_mtime_to_iso(local_mtime),
                            remote_mtime=_mtime_to_iso(remote_mtime),
                        )
                    )
            elif not local_info and remote_info and mirror:
                # File only in remote and mirror mode
                plan.operations.append(
                    SyncOperation(
                        action="delete",
                        path=path,
                        size=remote_info[0],
                        reason="not in source (--mirror mode)",
                        remote_mtime=_mtime_to_iso(remote_info[1]),
                    )
                )

    else:
        # Remote -> Local (download)
        bucket_id, prefix = _parse_bucket_path(source)
        local_path = os.path.abspath(dest)

        # Get remote and local file lists
        remote_files = {}
        for rel_path, size, mtime_ms in _list_remote_files(api, bucket_id, prefix, token):
            if filter_matcher.matches(rel_path):
                remote_files[rel_path] = (size, mtime_ms)

        local_files = {}
        if os.path.isdir(local_path):
            for rel_path, size, mtime_ms in _list_local_files(local_path):
                if filter_matcher.matches(rel_path):
                    local_files[rel_path] = (size, mtime_ms)

        # Compare files
        all_paths = set(remote_files.keys()) | set(local_files.keys())
        for path in sorted(all_paths):
            remote_info = remote_files.get(path)
            local_info = local_files.get(path)

            if remote_info and not local_info:
                # New file
                plan.operations.append(
                    SyncOperation(
                        action="download",
                        path=path,
                        size=remote_info[0],
                        reason="new file",
                        remote_mtime=_mtime_to_iso(remote_info[1]),
                    )
                )
            elif remote_info and local_info:
                # File exists in both
                remote_size, remote_mtime = remote_info
                local_size, local_mtime = local_info

                if force_download:
                    plan.operations.append(
                        SyncOperation(
                            action="download",
                            path=path,
                            size=remote_size,
                            reason="forced",
                            local_mtime=_mtime_to_iso(local_mtime),
                            remote_mtime=_mtime_to_iso(remote_mtime),
                        )
                    )
                elif remote_size != local_size:
                    # Size differs - use mtime to decide
                    if remote_mtime > local_mtime:
                        plan.operations.append(
                            SyncOperation(
                                action="download",
                                path=path,
                                size=remote_size,
                                reason="remote newer (size differs)",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                    else:
                        plan.operations.append(
                            SyncOperation(
                                action="skip",
                                path=path,
                                size=remote_size,
                                reason="local newer (size differs)",
                                local_mtime=_mtime_to_iso(local_mtime),
                                remote_mtime=_mtime_to_iso(remote_mtime),
                            )
                        )
                elif remote_mtime > local_mtime:
                    plan.operations.append(
                        SyncOperation(
                            action="download",
                            path=path,
                            size=remote_size,
                            reason="remote newer",
                            local_mtime=_mtime_to_iso(local_mtime),
                            remote_mtime=_mtime_to_iso(remote_mtime),
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
            elif not remote_info and local_info and mirror:
                # File only in local and mirror mode
                plan.operations.append(
                    SyncOperation(
                        action="delete",
                        path=path,
                        size=local_info[0],
                        reason="not in source (--mirror mode)",
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
            op_dict = {
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


def _execute_plan(plan: SyncPlan, api, token: Optional[str], verbose: bool = False) -> None:
    """Execute a sync plan."""
    is_upload = not _is_bucket_path(plan.source) and _is_bucket_path(plan.dest)
    is_download = _is_bucket_path(plan.source) and not _is_bucket_path(plan.dest)

    if is_upload:
        local_path = os.path.abspath(plan.source)
        bucket_id, prefix = _parse_bucket_path(plan.dest)

        # Collect operations
        add_operations = []
        delete_operations = []

        for op in plan.operations:
            if op.action == "upload":
                local_file = os.path.join(local_path, op.path)
                remote_path = f"{prefix}/{op.path}" if prefix else op.path
                if verbose:
                    print(f"  Uploading: {op.path} ({op.reason})")
                add_operations.append(BucketAddFile(path_in_repo=remote_path, path_or_fileobj=local_file))
            elif op.action == "delete":
                remote_path = f"{prefix}/{op.path}" if prefix else op.path
                if verbose:
                    print(f"  Deleting: {op.path} ({op.reason})")
                delete_operations.append(BucketDeleteFile(path_in_repo=remote_path))
            elif op.action == "skip" and verbose:
                print(f"  Skipping: {op.path} ({op.reason})")

        # Execute batch operations
        all_operations: list[Union[BucketAddFile, BucketDeleteFile]] = add_operations + delete_operations
        if all_operations:
            api.batch_bucket_files(bucket_id, all_operations, token=token)

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
                remote_path = f"{prefix}/{op.path}" if prefix else op.path
                local_file = os.path.join(local_path, op.path)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                if verbose:
                    print(f"  Downloading: {op.path} ({op.reason})")
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
            api.download_bucket_files(bucket_id, download_files, token=token)

        # Execute deletes
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
    if summary["total_size"]:
        print(f"  Total transfer size: {_format_size(summary['total_size'])}")


def _format_size(size: int) -> str:
    """Format a size in bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


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
    mirror: Annotated[
        bool,
        typer.Option(
            help="Make dest identical to source (DELETES destination files not in source).",
        ),
    ] = False,
    force_upload: Annotated[
        bool,
        typer.Option(
            help="Always upload/overwrite remote (ignore remote mtime and size).",
        ),
    ] = False,
    force_download: Annotated[
        bool,
        typer.Option(
            help="Always download/overwrite local (ignore local mtime and size).",
        ),
    ] = False,
    plan: Annotated[
        Optional[str],
        typer.Option(
            help="Save sync plan to JSONL file for review instead of executing.",
        ),
    ] = None,
    execute: Annotated[
        Optional[str],
        typer.Option(
            help="Execute a previously saved plan file.",
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
    if execute:
        # Execute mode: load and execute plan
        if source is not None or dest is not None:
            raise typer.BadParameter("Cannot specify source/dest when using --execute")
        if plan is not None:
            raise typer.BadParameter("Cannot specify both --plan and --execute")

        sync_plan = _load_plan(execute)
        if not quiet:
            _print_plan_summary(sync_plan)
            print("Executing plan...")

        if quiet:
            disable_progress_bars()
        try:
            _execute_plan(sync_plan, api, token, verbose=verbose)
        finally:
            if quiet:
                enable_progress_bars()

        if not quiet:
            print("Sync completed.")
        return

    # Normal mode: compute and optionally execute plan
    if source is None or dest is None:
        raise typer.BadParameter("Both source and dest are required (unless using --execute)")

    # Validate source/dest
    source_is_bucket = _is_bucket_path(source)
    dest_is_bucket = _is_bucket_path(dest)

    if source_is_bucket and dest_is_bucket:
        raise typer.BadParameter("Remote to remote sync is not supported. One path must be local.")

    if not source_is_bucket and not dest_is_bucket:
        raise typer.BadParameter("One of source or dest must be a bucket path (hf://buckets/...).")

    # Validate force options
    if force_upload and force_download:
        raise typer.BadParameter("Cannot specify both --force-upload and --force-download")

    if force_upload and source_is_bucket:
        raise typer.BadParameter("--force-upload is only valid when uploading (local -> remote)")

    if force_download and dest_is_bucket:
        raise typer.BadParameter("--force-download is only valid when downloading (remote -> local)")

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

    # Compute sync plan
    if not quiet:
        print(f"Computing sync plan: {source} -> {dest}")

    sync_plan = _compute_sync_plan(
        source=source,
        dest=dest,
        api=api,
        token=token,
        mirror=mirror,
        force_upload=force_upload,
        force_download=force_download,
        filter_matcher=filter_matcher,
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
        _execute_plan(sync_plan, api, token, verbose=verbose)
    finally:
        if quiet:
            enable_progress_bars()

    if not quiet:
        print("Sync completed.")
