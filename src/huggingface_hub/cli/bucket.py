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
"""Contains commands to interact with buckets via the CLI.

Usage:
    hf bucket --help

    # Create a bucket
    hf bucket create user/my-bucket
    hf bucket create user/my-bucket --private

    # Delete a bucket
    hf bucket delete user/my-bucket

    # List files in a bucket
    hf bucket ls hf://buckets/user/my-bucket
    hf bucket ls hf://buckets/user/my-bucket/models

    # List files with human-readable sizes
    hf bucket ls hf://buckets/user/my-bucket -h

    # List files in tree format
    hf bucket ls hf://buckets/user/my-bucket --tree
"""

from datetime import datetime, timezone
from typing import Annotated, Union

import typer

from ._cli_utils import TokenOpt, get_hf_api, typer_factory


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


def _format_mtime(mtime_ms: float) -> str:
    """Format mtime in milliseconds to a readable date string."""
    return datetime.fromtimestamp(mtime_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _build_tree(items: list[dict], prefix: str = "") -> list[str]:
    """Build a tree representation of files and directories.

    Args:
        items: List of items with 'path', 'type', 'size', 'mtime' keys
        prefix: Prefix to remove from paths for display

    Returns:
        List of formatted tree lines
    """
    # Build a nested structure
    tree: dict = {}

    for item in items:
        path = item["path"]
        # Remove prefix from path
        if prefix:
            if path.startswith(prefix + "/"):
                path = path[len(prefix) + 1 :]
            elif path.startswith(prefix):
                path = path[len(prefix) :]

        parts = path.split("/")
        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {"__children__": {}}
            current = current[part]["__children__"]

        # Store the item at the final level
        final_part = parts[-1]
        if item["type"] == "directory":
            if final_part not in current:
                current[final_part] = {"__children__": {}, "__item__": item}
            else:
                current[final_part]["__item__"] = item
        else:
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

        item = value.get("__item__", {})
        if item.get("type") == "directory":
            lines.append(f"{indent}{connector}{name}/")
        else:
            lines.append(f"{indent}{connector}{name}")

        children = value.get("__children__", {})
        if children:
            child_indent = indent + ("    " if is_last else "│   ")
            _render_tree(children, lines, child_indent)


@bucket_cli.command(name="ls")
def ls(
    bucket: Annotated[
        str,
        typer.Argument(
            help="Bucket path: hf://buckets/namespace/bucket_name(/prefix)",
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
    tree: Annotated[
        bool,
        typer.Option(
            "--tree",
            help="List files in tree format.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """List files in a bucket."""
    if not bucket.startswith(BUCKET_PREFIX):
        raise typer.BadParameter(f"Bucket path must start with {BUCKET_PREFIX}")

    api = get_hf_api(token=token)

    try:
        bucket_id, prefix = _parse_bucket_path(bucket)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    # Fetch items from the bucket
    items = list(
        api.list_bucket_tree(
            bucket_id,
            prefix=prefix or None,
            token=token,
        )
    )

    if not items:
        print("(empty)")
        return

    if tree:
        # Tree format
        tree_lines = _build_tree(items, prefix)
        for line in tree_lines:
            print(line)
    else:
        # Table format
        for item in items:
            path = item.get("path", "")
            # Remove prefix from path for display
            if prefix:
                if path.startswith(prefix + "/"):
                    display_path = path[len(prefix) + 1 :]
                elif path.startswith(prefix):
                    display_path = path[len(prefix) :]
                else:
                    display_path = path
            else:
                display_path = path

            item_type = item.get("type", "file")
            size = item.get("size", 0)
            mtime = item.get("mtime", 0)

            if item_type == "directory":
                # Show directory with trailing slash
                print(f"{'':>12}  {'':>19}  {display_path}/")
            else:
                # Show file with size and mtime
                size_str = _format_size(size, human_readable)
                mtime_str = _format_mtime(mtime)
                print(f"{size_str:>12}  {mtime_str}  {display_path}")


@bucket_cli.command(name="create")
def create(
    bucket_id: Annotated[
        str,
        typer.Argument(
            help="Bucket ID: namespace/bucket_name (e.g., user/my-bucket)",
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
    token: TokenOpt = None,
) -> None:
    """Create a new bucket."""
    api = get_hf_api(token=token)

    # Validate bucket_id format
    if "/" not in bucket_id:
        raise typer.BadParameter(f"Invalid bucket ID: {bucket_id}. Must be in format namespace/bucket_name")

    api.create_bucket(
        bucket_id,
        private=private if private else None,
        exist_ok=exist_ok,
        token=token,
    )
    print(f"Bucket created: {BUCKET_PREFIX}{bucket_id}")


@bucket_cli.command(name="delete")
def delete(
    bucket_id: Annotated[
        str,
        typer.Argument(
            help="Bucket ID: namespace/bucket_name (e.g., user/my-bucket)",
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
    token: TokenOpt = None,
) -> None:
    """Delete a bucket."""
    api = get_hf_api(token=token)

    # Validate bucket_id format
    if "/" not in bucket_id:
        raise typer.BadParameter(f"Invalid bucket ID: {bucket_id}. Must be in format namespace/bucket_name")

    # Confirm deletion unless -y flag is provided
    if not yes:
        confirm = typer.confirm(f"Are you sure you want to delete bucket '{bucket_id}'?")
        if not confirm:
            print("Aborted.")
            raise typer.Abort()

    api.delete_bucket(
        bucket_id,
        missing_ok=missing_ok,
        token=token,
    )
    print(f"Bucket deleted: {bucket_id}")
