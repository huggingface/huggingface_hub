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
"""Unified parser for hf:// URLs and Hub-style paths.

This module provides :func:`parse_hf_url` to parse ``hf://`` handles (and plain
Hub-style paths) into a structured :class:`HfUrl` dataclass. All code that
needs to interpret ``hf://`` URLs should use this instead of ad-hoc parsing.
"""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote

from huggingface_hub import constants


HF_URL_PREFIX = "hf://"

SPECIAL_REFS_REVISION_REGEX = re.compile(
    r"""
    (^refs\/convert\/\w+)     # `refs/convert/parquet` revisions
    |
    (^refs\/pr\/\d+)          # PR revisions
    """,
    re.VERBOSE,
)

_RESOURCE_TYPE_MAPPING: dict[str, str] = {
    "models": constants.REPO_TYPE_MODEL,
    "datasets": constants.REPO_TYPE_DATASET,
    "spaces": constants.REPO_TYPE_SPACE,
    "buckets": "bucket",
}


@dataclass(frozen=True)
class HfUrl:
    """Parsed representation of an ``hf://`` URL or plain Hub-style path.

    Attributes:
        resource_type: The resource type (``"model"``, ``"dataset"``, ``"space"``,
            ``"bucket"``), or ``None`` if not specified in the URL.
        repo_id: The full identifier (e.g. ``"user/repo"`` or ``"user/bucket"``),
            or a single-segment identifier (e.g. ``"namespace"``), or ``None``
            when only a type prefix was provided (e.g. ``hf://models``).
        revision: The revision extracted from ``@revision`` syntax, or ``None``.
        path: The remaining path after the identifier (empty string if at root).
    """

    resource_type: Optional[str] = None
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    path: str = ""


def _parse_revision_and_path(rev_and_path: str) -> tuple[str, str]:
    """Split ``revision/path`` into ``(revision, path)``, handling special refs.

    Handles ``refs/convert/parquet``, ``refs/pr/N`` refs that contain ``/``.
    """
    if "/" in rev_and_path:
        match = SPECIAL_REFS_REVISION_REGEX.search(rev_and_path)
        if match is not None:
            path = SPECIAL_REFS_REVISION_REGEX.sub("", rev_and_path).lstrip("/")
            return match.group(), path
        revision, path = rev_and_path.split("/", 1)
        return revision, path
    return rev_and_path, ""


def parse_hf_url(url: str) -> HfUrl:
    """Parse an ``hf://`` URL or plain Hub-style path into its components.

    This is the single source of truth for interpreting ``hf://`` handles and
    plain Hub-style paths across the codebase.

    Args:
        url: An ``hf://`` URL or a plain Hub-style path.

    Returns:
        A :class:`HfUrl` with the parsed components.

    Examples::

        # Type-only (listing)
        >>> parse_hf_url("hf://models")
        HfUrl(resource_type='model', repo_id=None, revision=None, path='')

        # Type + namespace (listing by namespace)
        >>> parse_hf_url("hf://models/huggingface")
        HfUrl(resource_type='model', repo_id='huggingface', revision=None, path='')

        # Type + full repo ID
        >>> parse_hf_url("hf://datasets/user/repo")
        HfUrl(resource_type='dataset', repo_id='user/repo', revision=None, path='')

        # With revision
        >>> parse_hf_url("hf://datasets/user/repo@main")
        HfUrl(resource_type='dataset', repo_id='user/repo', revision='main', path='')

        # With revision and path
        >>> parse_hf_url("hf://datasets/user/repo@v1/data/train")
        HfUrl(resource_type='dataset', repo_id='user/repo', revision='v1', path='data/train')

        # Special refs
        >>> parse_hf_url("user/repo@refs/pr/123/some/path")
        HfUrl(resource_type=None, repo_id='user/repo', revision='refs/pr/123', path='some/path')

        # Bucket paths
        >>> parse_hf_url("hf://buckets/user/bucket/prefix/file.txt")
        HfUrl(resource_type='bucket', repo_id='user/bucket', revision=None, path='prefix/file.txt')

        # Plain paths (no hf:// prefix)
        >>> parse_hf_url("user/repo")
        HfUrl(resource_type=None, repo_id='user/repo', revision=None, path='')

        >>> parse_hf_url("namespace")
        HfUrl(resource_type=None, repo_id='namespace', revision=None, path='')

        >>> parse_hf_url("")
        HfUrl(resource_type=None, repo_id=None, revision=None, path='')
    """
    path = url
    if path.startswith(HF_URL_PREFIX):
        path = path[len(HF_URL_PREFIX) :]

    # Detect resource type from first segment
    resource_type: Optional[str] = None
    first_segment = path.split("/")[0] if path else ""
    if first_segment in _RESOURCE_TYPE_MAPPING:
        resource_type = _RESOURCE_TYPE_MAPPING[first_segment]
        path = "/".join(path.split("/")[1:])

    if not path:
        return HfUrl(resource_type=resource_type)

    # Buckets don't support @revision syntax
    if resource_type == "bucket":
        parts = path.split("/", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return HfUrl(resource_type=resource_type, repo_id=path)
        bucket_id = f"{parts[0]}/{parts[1]}"
        prefix = parts[2] if len(parts) > 2 else ""
        return HfUrl(resource_type=resource_type, repo_id=bucket_id, path=prefix)

    # Single segment (no "/")
    if "/" not in path:
        if "@" in path:
            repo_id, rev = path.split("@", 1)
            return HfUrl(resource_type=resource_type, repo_id=repo_id, revision=unquote(rev))
        return HfUrl(resource_type=resource_type, repo_id=path)

    # Multiple segments: namespace/name, possibly with @revision and/or trailing path
    parts = path.split("/")
    repo_id_candidate = "/".join(parts[:2])
    remaining = "/".join(parts[2:])

    if "@" in repo_id_candidate:
        repo_id, rev_and_path = repo_id_candidate.split("@", 1)
        if remaining:
            rev_and_path = f"{rev_and_path}/{remaining}" if rev_and_path else remaining
        revision, path_remainder = _parse_revision_and_path(rev_and_path)
        return HfUrl(
            resource_type=resource_type,
            repo_id=repo_id,
            revision=unquote(revision),
            path=path_remainder,
        )

    return HfUrl(
        resource_type=resource_type,
        repo_id=repo_id_candidate,
        path=remaining,
    )
