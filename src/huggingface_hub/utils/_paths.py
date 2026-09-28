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
"""Contains utilities to handle paths in Huggingface Hub."""

import os
from collections.abc import Callable, Generator, Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TypeVar


T = TypeVar("T")

# Always ignore `.git` and `.cache/huggingface` folders in commits
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "*/.git",
    "**/.git/**",
    ".cache/huggingface",
    ".cache/huggingface/*",
    "*/.cache/huggingface",
    "**/.cache/huggingface/**",
]
# Forbidden to commit these folders
FORBIDDEN_FOLDERS = [".git", ".cache"]


def as_extended_path(path: str | Path, max_length: int = 255) -> str:
    r"""Return `path` in its Windows extended-length form if it is too long, unchanged otherwise.

    Some Windows versions do not allow for paths longer than 255 characters (247 for directories, i.e. MAX_PATH minus
    room for an 8.3 file name). In this case, we must specify them as extended paths by using the `\\?\` prefix, which
    only works on absolute paths. Network shares take the `\\?\UNC\server\share\...` form: prefixing them verbatim
    would produce an invalid `\\?\\\server\...` path.

    Args:
        path (`str` or `Path`):
            The path to convert. Returned unchanged on non-Windows platforms, if it is short enough, or if it is
            already an extended path.
        max_length (`int`, *optional*):
            Length above which the path must be converted. Defaults to 255, the limit for files. Pass 247 for paths
            whose parent directories are created by the caller.
    """
    path = str(path)
    if os.name != "nt":
        return path
    absolute_path = os.path.abspath(path)
    if len(absolute_path) <= max_length or absolute_path.startswith("\\\\?\\"):
        return path
    if absolute_path.startswith("\\\\"):  # UNC share: `\\server\share\...` => `\\?\UNC\server\share\...`
        return "\\\\?\\UNC\\" + absolute_path[2:]
    return "\\\\?\\" + absolute_path


def filter_repo_objects(
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    key: Callable[[T], str] | None = None,
) -> Generator[T, None, None]:
    """Filter repo objects based on an allowlist and a denylist.

    Input must be a list of paths (`str` or `Path`) or a list of arbitrary objects.
    In the later case, `key` must be provided and specifies a function of one argument
    that is used to extract a path from each element in iterable.

    Patterns are Standard Wildcards (globbing patterns), NOT regular expressions. The pattern matching is based on
    Python's `fnmatch.fnmatchcase`, so it is case-sensitive on every platform. Backslashes are treated as path separators
    and normalized to forward slashes in both patterns and paths before matching, so patterns built with `os.path.join`
    on Windows work as expected.

    Note that it matches `*` across path boundaries, unlike traditional Unix shell globbing. For example, `"data/*.json"`
    will match both `data/file.json` and `data/subdir/file.json`.

    See https://docs.python.org/3/library/fnmatch.html for more details.

    Args:
        items (`Iterable`):
            List of items to filter.
        allow_patterns (`str` or `list[str]`, *optional*):
            Patterns constituting the allowlist. If provided, item paths must match at
            least one pattern from the allowlist.
        ignore_patterns (`str` or `list[str]`, *optional*):
            Patterns constituting the denylist. If provided, item paths must not match
            any patterns from the denylist.
        key (`Callable[[T], str]`, *optional*):
            Single-argument function to extract a path from each item. If not provided,
            the `items` must already be `str` or `Path`.

    Returns:
        Filtered list of objects, as a generator.

    Raises:
        :class:`ValueError`:
            If `key` is not provided and items are not `str` or `Path`.

    Example usage with paths:
    ```python
    >>> # Filter only PDFs that are not hidden.
    >>> list(filter_repo_objects(
    ...     ["aaa.pdf", "bbb.jpg", ".ccc.pdf", ".ddd.png"],
    ...     allow_patterns=["*.pdf"],
    ...     ignore_patterns=[".*"],
    ... ))
    ["aaa.pdf"]
    ```

    Example usage with objects:
    ```python
    >>> list(filter_repo_objects(
    ... [
    ...     CommitOperationAdd(path_or_fileobj="/tmp/aaa.pdf", path_in_repo="aaa.pdf")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/bbb.jpg", path_in_repo="bbb.jpg")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/.ccc.pdf", path_in_repo=".ccc.pdf")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/.ddd.png", path_in_repo=".ddd.png")
    ... ],
    ... allow_patterns=["*.pdf"],
    ... ignore_patterns=[".*"],
    ... key=lambda x: x.path_in_repo
    ... ))
    [CommitOperationAdd(path_or_fileobj="/tmp/aaa.pdf", path_in_repo="aaa.pdf")]
    ```
    """
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]

    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    if allow_patterns is not None:
        allow_patterns = [_add_wildcard_to_directories(_normalize_separators(p)) for p in allow_patterns]
    if ignore_patterns is not None:
        ignore_patterns = [_add_wildcard_to_directories(_normalize_separators(p)) for p in ignore_patterns]

    if key is None:

        def _identity(item: T) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, Path):
                return str(item)
            raise ValueError(f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string.")

        key = _identity  # Items must be `str` or `Path`, otherwise raise ValueError

    for item in items:
        path = _normalize_separators(key(item))

        # Skip if there's an allowlist and path doesn't match any
        if allow_patterns is not None and not any(fnmatchcase(path, r) for r in allow_patterns):
            continue

        # Skip if there's a denylist and path matches any
        if ignore_patterns is not None and any(fnmatchcase(path, r) for r in ignore_patterns):
            continue

        yield item


def _normalize_separators(value: str | Path) -> str:
    # Repo paths always use `/` and `\` is not an fnmatch escape character, so treating backslashes as path separators
    # (e.g. patterns built with `os.path.join` on Windows) is safe.
    # `value` can be a `Path` if a custom `key` returns one; coerce to `str` first.
    return str(value).replace("\\", "/")


def _add_wildcard_to_directories(pattern: str) -> str:
    if pattern.endswith("/"):
        return pattern + "*"
    return pattern
